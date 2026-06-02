"""End-to-end training orchestration for the HGT recommender.

Training strategy: Full-Graph Forward with Two-Stage Mini-Batch Listwise Loss.

The HGT graph is forwarded in full once per epoch so every node embedding
reflects its complete neighbourhood context. The listwise loss is computed in
user-batch chunks, but instead of retaining the full HGT computation graph
across every chunk's backward (O(n_batches) HGT backward passes), we use a
two-stage backward:

  Stage 1 — cheap: detach HGT outputs into float32 leaves; accumulate
  d_loss/d_user_embs and d_loss/d_track_embs across all user batches with no
  retain_graph (just the small loss sub-graph each time).

  Stage 2 — one single HGT backward: feed the accumulated output-gradients
  from stage 1 into torch.autograd.backward([user_embs, track_embs], ...).

This is mathematically identical to the old retain_graph approach but requires
exactly ONE pass through the HGT backward per epoch regardless of batch size.

Loss: Intensity-Weighted Listwise cross-entropy with Logit Adjustment
(see bpr.debiased_listwise_loss). Raw listen counts serve as graded relevance
and a log-popularity penalty discourages the model from concentrating
recommendations on already-popular tracks.

Evaluation: full-ranking Recall@K and NDCG@K on held-out edges produced by
RandomLinkSplit, computed once every eval_every epochs and at the final epoch.

Extras: cosine LR decay (single cycle, no restarts), gradient clipping,
torch.compile. Per-epoch metrics are returned in ``TrainResult.history`` (the
caller persists them to CSV/PNG); there is no experiment-tracker dependency.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from .hgt import RecommenderHGT
from .loss import compute_log_pop_prior, debiased_listwise_loss, evaluate_top_k


def _gt_to_eval_edges(gt, device):
    """Flatten a ``{user_idx: iterable[item_idx]}`` ground-truth dict into the
    parallel ``(eval_user_idx, eval_item_idx)`` LongTensors that
    :func:`evaluate_top_k` expects. Indices must already be in the graph's
    node-index space.
    """
    us: list[int] = []
    its: list[int] = []
    for u, items in gt.items():
        for i in items:
            us.append(int(u))
            its.append(int(i))
    return (torch.tensor(us, dtype=torch.long, device=device),
            torch.tensor(its, dtype=torch.long, device=device))


@dataclass
class TrainResult:
    model: RecommenderHGT
    history: list[dict] = field(default_factory=list)
    best_val: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    timing_seconds: float = 0.0

def train_hgt(
    data: HeteroData,
    user_interaction_matrix: torch.Tensor,
    track_listen_counts: torch.Tensor,
    *,
    edge_type: tuple[str, str, str] = ("user", "listened_to", "track"),
    rev_edge_type: tuple[str, str, str] = ("track", "rev_listened_to", "user"),
    hidden_channels: int = 128,
    out_channels: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.1,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    user_batch_size: int = 1024,
    eval_every: int = 10,
    early_stopping_patience: Optional[int] = None,
    k_list: Iterable[int] = (10, 20),
    monitor: str = "overall_score",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    disjoint_train_ratio: float = 0.3,
    val_gt: Optional[dict] = None,
    test_gt: Optional[dict] = None,
    train_edge_index: Optional[torch.Tensor] = None,
    lambda_reg: float = 0.2,
    temperature: float = 0.1,
    lr_t0: int = 0,
    lr_eta_min: float = 1e-5,
    clip_grad_norm: float = 1.0,
    device: Optional[str] = None,
    use_amp: bool = True,
    use_checkpoint: bool = False,
    compile_model: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> TrainResult:
    """Train an HGT recommender with full-graph forward and mini-batch listwise loss.

    The full heterogeneous graph is forwarded through the HGT stack once per epoch
    to obtain globally-consistent node embeddings. The listwise loss is then
    accumulated in user_batch_size chunks so the full (U x I) logit matrix is never
    held in memory at once. A single backward pass closes the epoch, preserving the
    global gradient signal of the listwise objective.

    Parameters are:
      data                    -- populated HeteroData graph from build_rich_hetero_graph.
      user_interaction_matrix -- (U, I) tensor of raw listen counts per user per track.
                                 May live on CPU; rows are indexed by HGT user node ids.
      track_listen_counts     -- (I,) tensor of total listen counts per track, used to
                                 compute the log-popularity prior for logit adjustment.
      edge_type               -- relation triple for user-to-track supervision edges.
      rev_edge_type           -- reverse relation triple; added automatically if missing.
      hidden_channels         -- width of HGT message-passing layers.
      out_channels            -- dimension of final L2-normalised embeddings.
      num_heads               -- attention heads per HGTConv layer.
      num_layers              -- number of message-passing hops.
      dropout                 -- dropout probability in GNN layers and projection head.
      epochs                  -- total training epochs.
      lr                      -- initial AdamW learning rate.
      weight_decay            -- L2 regularisation coefficient.
      user_batch_size         -- users per mini-batch during loss accumulation.
      eval_every              -- evaluate validation metrics every this many epochs.
      k_list                  -- Recall@K and NDCG@K cut-offs.
      monitor                 -- scalar used for best-checkpoint selection AND
                                 early stopping. "overall_score" (default) uses
                                 models.evaluation.metrics.overall_score at the
                                 smallest cut-off (0.6*NDCG + 0.2*Coverage +
                                 0.2*(1-PopularityBias)); coverage/pop-bias are
                                 computed cheaply from the top-K already ranked
                                 for Recall/NDCG. Any other value is read as a
                                 single metric key, e.g. "ndcg@10" or "recall@20".
      val_ratio               -- fraction of edges held out for validation
                                 (RandomLinkSplit regime only; ignored when
                                 val_gt is provided).
      test_ratio              -- fraction of edges held out for testing
                                 (RandomLinkSplit regime only).
      disjoint_train_ratio    -- fraction of training edges used as supervision
                                 labels (RandomLinkSplit regime only).
      val_gt                  -- optional {user_idx: set(item_idx)} held-out
                                 validation ground truth, indexed in the graph's
                                 node-index space. When provided, RandomLinkSplit
                                 is SKIPPED: the graph is assumed to already hold
                                 only the training interactions (no leakage), the
                                 best checkpoint is selected on this val set, and
                                 the same stratified split used by the baselines
                                 drives the HGT.
      test_gt                 -- optional held-out test ground truth (same shape
                                 as val_gt); defaults to val_gt when omitted.
      train_edge_index        -- optional (2, E) positive train edges for seen-
                                 item masking; defaults to the graph's
                                 user→track edge_index (train-only by
                                 construction in the explicit-split regime).
      lambda_reg              -- popularity penalty scale in the listwise loss.
      temperature             -- softmax temperature (lower = sharper distribution).
      lr_t0                   -- cosine half-period in epochs (T_max for
                                 CosineAnnealingLR). 0 (default) or any value
                                 >= epochs means a single smooth decay from lr
                                 to lr_eta_min over the whole run. A smaller
                                 positive value makes the LR cycle: plain
                                 CosineAnnealingLR ramps back UP to lr after each
                                 trough (period 2*lr_t0), so e.g. lr_t0=20 over
                                 200 epochs gives 5 down-up cycles. Use
                                 CosineAnnealingWarmRestarts for proper SGDR.
      lr_eta_min              -- minimum learning rate at the cosine trough.
      clip_grad_norm          -- maximum L2 norm for gradient clipping.
      device                  -- target device; defaults to cuda when available.
      use_amp                 -- enable automatic mixed precision (default True on CUDA).
                                 Halves activation VRAM; uses GradScaler for stable backward.
      use_checkpoint          -- enable gradient checkpointing in HGTConv layers.
                                 Trades ~20% extra compute for ~40-60% less activation VRAM.
                                 Recommended when use_amp alone is still not enough.
      compile_model           -- call torch.compile() on the model before training
                                 (PyTorch >= 2.0 only; adds ~30s startup, then 20-40%
                                 faster per epoch on T4/A100 via Triton JIT).
      seed                    -- random seed for reproducibility.
      verbose                 -- print one-line summary after each evaluation epoch.

    Returns a TrainResult with the best-val-checkpoint model, full epoch history,
    best validation metrics, and final test metrics.
    """
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # The (U, I) interaction matrix can live on CPU or GPU. debiased_listwise_loss
    # slices [user_batch] rows per call; if already on GPU those slices are free
    # (no PCIe transfer). On a small GPU keep it on CPU to save VRAM; on a large
    # GPU (e.g. A100/H100) pass it pre-moved to device for slightly faster batches.

    _amp_enabled = use_amp and dev.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled)
    # The two-stage backward below injects gradients directly via
    # torch.autograd.backward(...) and never calls scaler.scale(loss).backward(),
    # so the scaler's internal _scale tensor is never created lazily. Touch
    # scale() once on a throwaway tensor to force that initialisation now;
    # otherwise scaler.unscale_(opt) asserts "Attempted unscale_ but _scale is
    # None." on the very first epoch.
    if _amp_enabled:
        scaler.scale(torch.zeros((), device=dev))

    if dev.startswith("cuda"):
        # cuDNN selects the fastest convolution algorithm for fixed input shapes.
        torch.backends.cudnn.benchmark = True
        # TF32 uses Tensor Cores on Ampere/Turing (T4, A100) for matmuls and
        # convolutions. Precision loss is negligible for GNN embeddings
        # (~1e-3 relative error) while throughput is 2-3× higher than FP32.
        torch.backends.cuda.matmul.allow_tf32  = True
        torch.backends.cudnn.allow_tf32        = True

    # Ensure the reverse user→track relation exists for message passing
    # (needed in both evaluation regimes below).
    if rev_edge_type not in data.edge_types:
        data = T.ToUndirected(merge=False)(data)
        if rev_edge_type not in data.edge_types:
            raise ValueError(
                f"ToUndirected() did not create the expected reverse edge type "
                f"{rev_edge_type!r}. Available edge types: {data.edge_types}"
            )

    # Two evaluation regimes:
    #   • EXPLICIT stratified split (val_gt provided): the graph already holds
    #     ONLY the training interactions (built upstream from the stratified
    #     train split), so message passing AND the listwise target are
    #     leak-free. Validation/test use the caller's held-out ground-truth
    #     dicts, keyed in the SAME node-index space as the graph. This makes the
    #     HGT share the exact split used by the baselines.
    #   • RandomLinkSplit (default): legacy behaviour for callers that pass the
    #     full interaction graph and want an internal random split (e.g. the
    #     ablation cells).
    _explicit_split = val_gt is not None
    val_data = test_data = None
    val_eval = test_eval = None
    if _explicit_split:
        train_data = data.to(dev)
        _tp = (train_edge_index if train_edge_index is not None
               else data[edge_type].edge_index)
        train_pos = _tp.to(dev)
        val_eval  = _gt_to_eval_edges(val_gt, dev)
        test_eval = _gt_to_eval_edges(test_gt, dev) if test_gt else val_eval
    else:
        transform = T.RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            disjoint_train_ratio=disjoint_train_ratio,
            neg_sampling_ratio=0.0,
            add_negative_train_samples=False,
            edge_types=[edge_type],
            rev_edge_types=[rev_edge_type],
        )
        train_data, val_data, test_data = transform(data)
        # train_data stays on GPU permanently (accessed every epoch).
        # val_data / test_data stay on CPU and are moved to GPU only inside the
        # evaluation windows, then the GPU copy is deleted immediately.
        train_data = train_data.to(dev)
        train_pos = train_data[edge_type].edge_label_index

    src_t, _, dst_t = edge_type
    n_users = data[src_t].num_nodes

    # Pre-compute log-popularity prior once; reused every epoch
    log_track_pop = compute_log_pop_prior(track_listen_counts, device=dev)

    # Per-track popularity normalised to [0, 1] (same max-normalisation the
    # notebook uses for the final eval), aligned with the track-node order of
    # item_emb. Fed to evaluate_top_k so it can also report coverage / pop-bias,
    # which the overall_score monitor needs.
    _pop = track_listen_counts.detach().float()
    pop_norm = (_pop / (_pop.max() + 1e-9)).to(dev)

    # Held-out VALIDATION listwise loss (same objective + lambda_reg/temperature as
    # training, but with the val positives as targets) — a diagnostic that exposes
    # the train↔val gap (overfitting). Model SELECTION still uses the val ranking
    # metrics, not this loss. Built once as a sparse (U, I) target matrix; only the
    # explicit-split regime (the notebook's stratified val_gt) supports it.
    _val_counts = _val_users = None
    if _explicit_split:
        _vu_e, _vi_e = val_eval
        if _vu_e.numel():
            _val_counts = torch.sparse_coo_tensor(
                torch.stack([_vu_e, _vi_e]),
                torch.ones(_vu_e.numel(), device=dev),
                size=(n_users, data[dst_t].num_nodes),
            ).coalesce()
            _val_users = torch.unique(_vu_e)

    # DataLoader over user indices for mini-batch loss accumulation
    user_loader = DataLoader(
        range(n_users), batch_size=user_batch_size, shuffle=True
    )

    # Gradient checkpointing is incompatible with the per-batch retain_graph
    # loop below: retaining the graph across user batches would trigger a full
    # forward recompute on *every* batch's backward (num_batches× slower). The
    # per-batch backward already bounds activation memory, so AMP alone covers
    # VRAM and checkpointing is force-disabled here.
    if use_checkpoint:
        if verbose:
            print("[hgt] use_checkpoint=True is incompatible with the per-batch "
                  "backward loop; disabling it (AMP still active for VRAM).")
        use_checkpoint = False

    # Model
    model = RecommenderHGT(
        metadata=data.metadata(),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
    ).to(dev)

    # Materialise lazy Linear(-1, h) weights with a dry forward pass
    with torch.no_grad():
        model(train_data.x_dict, train_data.edge_index_dict)

    # torch.compile fuses kernel launches via Triton JIT — ~30 s startup cost,
    # then 20-40 % faster per epoch.  Silently skipped on PyTorch < 2.0.
    #
    # Robustness: Inductor lowers to Triton and shells out to `ptxas` to build
    # the GPU kernels. On GPU architectures the bundled Triton/ptxas can't
    # target yet (e.g. Blackwell sm_120a) this fails *lazily* on the first
    # compiled forward with an InductorError / NoTritonConfigsError / PTXASError,
    # which would otherwise abort the whole run mid-epoch. We therefore:
    #   • capture_scalar_outputs=True — lets dynamo trace through HGTConv's
    #     grouped-Linear `.item()` loop instead of breaking the graph there;
    #   • suppress_errors=True — any remaining Inductor/ptxas failure degrades
    #     to eager for that graph (with a warning) rather than crashing.
    # Note these only matter when the eager fallback is actually exercised; on a
    # supported GPU compilation proceeds normally.
    if compile_model and hasattr(torch, "compile"):
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.capture_scalar_outputs = True
            _dynamo.config.suppress_errors = True
        except Exception:
            pass
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    _t_max = lr_t0 if (0 < lr_t0 < epochs) else epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=_t_max, eta_min=lr_eta_min
    )

    history: list[dict] = []
    best_monitor = -float("inf")
    best_state: dict | None = None
    best_val: dict = {}
    _evals_no_improve = 0   # consecutive evals without a monitor-score gain

    k_list = sorted(set(int(k) for k in k_list))
    primary_k = k_list[0]

    # ── Selection / early-stopping criterion ─────────────────────────────────
    # `monitor` is either "overall_score" (the metrics.overall_score composite
    # evaluated at primary_k) or a single metric key like "ndcg@10". overall_score
    # is imported lazily so single-metric monitors never pull in the evaluation/kg
    # import stack (recommenders -> kg_to_hetero -> data.kg, plus scipy).
    _overall_fn = None
    if monitor == "overall_score":
        try:
            from .evaluation.metrics import overall_score as _overall_fn
        except Exception as _e:
            raise ImportError(
                "monitor='overall_score' requires "
                "models.evaluation.metrics.overall_score, which failed to import "
                f"({_e.__class__.__name__}: {_e}). Pass monitor='ndcg@10' (or "
                "another recall@k / ndcg@k key) to select on a single metric."
            ) from _e
    else:
        _valid = {f"{m}@{k}" for m in ("recall", "ndcg", "coverage", "pop_bias")
                  for k in k_list}
        if monitor not in _valid:
            raise ValueError(
                f"monitor={monitor!r} must be 'overall_score' or one of "
                f"{sorted(_valid)}")

    def _monitor_score(vm: dict[str, float]) -> float:
        """Scalar that best-checkpoint selection and early stopping maximise."""
        if _overall_fn is not None:
            return _overall_fn(
                {f"NDCG@{primary_k}":           vm[f"ndcg@{primary_k}"],
                 f"PopularityBias@{primary_k}": vm.get(f"pop_bias@{primary_k}", 0.0),
                 "Coverage":                    vm.get(f"coverage@{primary_k}", 0.0)},
                k=primary_k,
            )
        return float(vm[monitor])

    t_start = time.time()
    _epoch_bar = tqdm(range(1, epochs + 1), desc="[hgt] training",
                      disable=not verbose, leave=True)
    for ep in _epoch_bar:
        model.train()
        opt.zero_grad(set_to_none=True)

        # ── Stage 1: full-graph forward; accumulate output grads cheaply ───────
        # AMP halves VRAM for the HGT forward pass and speeds up matmuls on
        # Tensor-core GPUs (T4, A100).
        with torch.cuda.amp.autocast(enabled=_amp_enabled):
            out = model(train_data.x_dict, train_data.edge_index_dict)
            all_track_embs = out[dst_t]  # (I, D) — still in HGT comp graph
            all_user_embs  = out[src_t]  # (U, D) — still in HGT comp graph

        # Detach into float32 leaves so stage-1 backward never touches the HGT.
        # float32 avoids AMP underflow in the loss function (softmax, log).
        u_det = all_user_embs.detach().float().requires_grad_(True)   # (U, D)
        t_det = all_track_embs.detach().float().requires_grad_(True)  # (I, D)

        epoch_loss_scalar = 0.0
        for user_batch_indices in user_loader:
            user_batch_indices = user_batch_indices.to(dev)
            # Pure float32 loss — no AMP context, no retain_graph, no scaler.
            # Backward only flows through the cheap float32 leaf sub-graph.
            batch_loss = debiased_listwise_loss(
                u_det[user_batch_indices],
                t_det,
                user_batch_indices,
                user_interaction_matrix,
                log_track_pop,
                lambda_reg=lambda_reg,
                temperature=temperature,
            )
            scaled = batch_loss * (user_batch_indices.numel() / n_users)
            scaled.backward()   # accumulates into u_det.grad and t_det.grad
            epoch_loss_scalar += scaled.item()

        # ── Stage 2: single backward through the HGT ─────────────────────────
        # u_det.grad / t_det.grad now hold d_total_loss / d_{user,track}_embs.
        # Multiply by the AMP loss-scale so the HGT parameter gradients stay in
        # the same numerical regime as scaler.scale(loss).backward(); then
        # scaler.unscale_(opt) restores normal magnitude before clipping.
        u_g = u_det.grad  # (U, D) float32
        t_g = t_det.grad  # (I, D) float32
        if _amp_enabled and u_g is not None:
            _s = scaler.get_scale()
            u_g = (u_g * _s).to(all_user_embs.dtype)
            t_g = (t_g * _s).to(all_track_embs.dtype) if t_g is not None else None
        torch.autograd.backward(
            [all_user_embs, all_track_embs],
            grad_tensors=[u_g, t_g],
        )
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(opt)
        scaler.update()

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        log: dict = {
            "epoch":               ep,
            "train/listwise_loss": epoch_loss_scalar,
            "train/lr":            current_lr,
        }

        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            _val_loss = None
            with torch.inference_mode():
                out_eval = model(train_data.x_dict, train_data.edge_index_dict)
                # Held-out listwise loss on the val positives (diagnostic only).
                if _val_counts is not None and _val_users.numel():
                    _ue, _te = out_eval[src_t], out_eval[dst_t]
                    _val_loss = 0.0
                    for _vb in torch.split(_val_users, user_batch_size):
                        _bl = debiased_listwise_loss(
                            _ue[_vb], _te, _vb, _val_counts, log_track_pop,
                            lambda_reg=lambda_reg, temperature=temperature)
                        _val_loss += float(_bl) * (_vb.numel() / _val_users.numel())

            if _explicit_split:
                _vu, _vi = val_eval
            else:
                # Move val graph to GPU only for this evaluation window.
                _val_dev = val_data.to(dev)
                _veli = _val_dev[edge_type].edge_label_index
                _vu, _vi = _veli[0], _veli[1]
            val_metrics = evaluate_top_k(
                user_emb=out_eval[src_t],
                item_emb=out_eval[dst_t],
                eval_user_idx=_vu,
                eval_item_idx=_vi,
                train_edge_index=train_pos,
                k_list=k_list,
                pop_norm=pop_norm,   # enables coverage@k / pop_bias@k for overall_score
            )
            if not _explicit_split:
                del _val_dev
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()

            monitor_score = _monitor_score(val_metrics)
            log.update({f"val/{k}": v for k, v in val_metrics.items()})
            log["val/monitor_score"] = monitor_score
            if _val_loss is not None:
                log["val/listwise_loss"] = _val_loss

            if monitor_score > best_monitor:
                best_monitor = monitor_score
                best_state = copy.deepcopy(model.state_dict())
                best_val = dict(val_metrics)
                _evals_no_improve = 0
            else:
                _evals_no_improve += 1

        history.append(log)

        if verbose:
            _post = {k.split("/")[-1]: (f"{v:.4f}" if isinstance(v, float) else v)
                     for k, v in log.items() if k != "epoch"}
            _epoch_bar.set_postfix(_post)

        # -- early stopping (counted in evaluation steps) ---------------------
        if (early_stopping_patience is not None
                and (ep % eval_every == 0 or ep == epochs)
                and _evals_no_improve >= early_stopping_patience):
            if verbose:
                _epoch_bar.write(
                    f"[hgt] early stopping at epoch {ep} "
                    f"(no val {monitor} gain for "
                    f"{early_stopping_patience} evals; best={best_monitor:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    if _explicit_split:
        with torch.inference_mode():
            out_test = model(train_data.x_dict, train_data.edge_index_dict)
        _tu, _ti = test_eval
    else:
        _test_dev = test_data.to(dev)
        with torch.inference_mode():
            out_test = model(_test_dev.x_dict, _test_dev.edge_index_dict)
        _teli = _test_dev[edge_type].edge_label_index
        _tu, _ti = _teli[0], _teli[1]
    test_metrics = evaluate_top_k(
        user_emb=out_test[src_t],
        item_emb=out_test[dst_t],
        eval_user_idx=_tu,
        eval_item_idx=_ti,
        train_edge_index=train_pos,
        k_list=k_list,
        pop_norm=pop_norm,   # test metrics also carry coverage@k / pop_bias@k
    )
    if not _explicit_split:
        del _test_dev
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    if verbose:
        print("[hgt test]", "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    t_elapsed = time.time() - t_start
    return TrainResult(
        model=model,
        history=history,
        best_val=best_val,
        test_metrics=test_metrics,
        timing_seconds=t_elapsed,
    )


__all__ = ("train_hgt", "TrainResult")