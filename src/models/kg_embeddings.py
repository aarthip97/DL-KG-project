"""Knowledge Graph Embedding trainer backed by PyKEEN (RotatE and ComplEx).

Trains a KGE model on the consolidated triple TSV produced by
extract_dl_artifacts() in kg_to_hetero.py, then exports a flat dict
{uri_string -> float32 numpy array} that build_rich_hetero_graph() uses
to seed node features.

Supported models
----------------
RotatE  (default)
    Each entity is a unit-norm complex vector; each relation is a rotation
    in complex space.  entity_dim=128 produces a 256-D real output
    [real || imag] after flattening via torch.view_as_real.
    Reference: Sun et al. 2019, https://arxiv.org/abs/1902.10197

ComplEx
    Entities and relations are complex vectors; the score is
    Re(<h, r, conj(t)>).  Same [real || imag] layout as RotatE.
    Reference: Trouillon et al. 2016, https://arxiv.org/abs/1606.06357

Both models are trained through PyKEEN pipeline() which handles
negative sampling, loss calculation, and optimisation internally.

Typical usage
-------------
    from src.models.kg_to_hetero import extract_dl_artifacts
    from src.models.kg_embeddings import train_kge, load_kge_checkpoint
    import rdflib

    g = rdflib.Graph()
    g.parse("data/interim/populated_graph.ttl")
    edge_dict = extract_dl_artifacts(
        g,
        tsv_out_path="data/interim/kg_triples.tsv",
        dict_out_path="data/interim/edge_dict.json",
    )

    result = train_kge(
        "data/interim/kg_triples.tsv",
        model="RotatE",
        entity_dim=128,
        epochs=500,
        device="cuda",
        checkpoint_path="data/interim/kge_checkpoint.pt",
    )

    data = build_rich_hetero_graph(
        edge_dict=edge_dict,
        rotate_embeddings=result.embeddings,
        track_audio_features=audio_feats,
    )

Output shape
------------
RotatE / ComplEx with entity_dim=128:
    each value in the returned dict has shape (256,) float32  [real || imag]
This matches the kge_dim=256 constant in build_rich_hetero_graph().
"""
from __future__ import annotations

import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

logger = logging.getLogger(__name__)

# W&B is optional -- import once and guard every call with _WANDB
try:
    import wandb as _wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KGEResult:
    """Container returned by train_kge().

    embeddings
        dict mapping each URI string to a float32 numpy array of shape
        (2*entity_dim,).  Real and imaginary parts are concatenated:
        [re_0, ..., re_{d-1}, im_0, ..., im_{d-1}].
    pipeline_result
        The raw pykeen.pipeline.PipelineResult object.  Useful for accessing
        loss curves, metric summaries, and the trained PyKEEN model directly.
    triples_factory
        The TriplesFactory built from the TSV.  Contains entity_to_id and
        relation_to_id dicts for downstream look-ups.
    """

    embeddings:      dict[str, np.ndarray]
    pipeline_result: object           # pykeen.pipeline.PipelineResult
    triples_factory: TriplesFactory


# ---------------------------------------------------------------------------
# Embedding extraction helper
# ---------------------------------------------------------------------------

def _extract_embeddings(
    pykeen_model,
    triples_factory: TriplesFactory,
) -> dict[str, np.ndarray]:
    """Pull the entity embedding matrix out of any PyKEEN model.

    PyKEEN stores entity representations in model.entity_representations[0].
    Calling that module with no arguments returns the full embedding table.
    For complex-valued models (RotatE, ComplEx) the tensor dtype is
    torch.complex64; torch.view_as_real converts it to a real tensor of
    shape (n_entities, 2*entity_dim) with real part first then imaginary
    part, matching the kge_dim=256 layout in build_rich_hetero_graph().
    For real-valued models the tensor is returned as-is.
    """
    with torch.no_grad():
        raw: torch.Tensor = pykeen_model.entity_representations[0]()

    if raw.is_complex():
        # (n_entities, entity_dim) complex -> (n_entities, 2*entity_dim) real
        real_tensor = torch.view_as_real(raw).flatten(start_dim=1)
    else:
        real_tensor = raw

    vecs = real_tensor.detach().cpu().float().numpy()   # (n_entities, out_dim)

    # entity_to_id maps URI string -> integer index
    return {
        uri: vecs[idx].astype(np.float32)
        for uri, idx in triples_factory.entity_to_id.items()
    }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def train_kge(
    triples_path: str | pathlib.Path,
    *,
    model: Literal["RotatE", "ComplEx"] = "RotatE",
    entity_dim: int = 128,
    epochs: int = 500,
    batch_size: int = 512,
    num_negs_per_pos: int = 1,
    lr: float = 1e-3,
    device: str | None = None,
    seed: int = 42,
    checkpoint_path: str | pathlib.Path | None = None,
    history_csv_path: str | pathlib.Path | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> KGEResult:
    """Train a KGE model via PyKEEN and return a URI->embedding dict.

    Parameters
    ----------
    triples_path
        Path to the 3-column TSV produced by extract_dl_artifacts().
        Format: head_uri TAB relation_uri TAB tail_uri, no header row.
        Typically data/interim/kg_triples.tsv.
    model
        PyKEEN model name.  "RotatE" (default) or "ComplEx".
        Both produce complex embeddings; output vectors are 2*entity_dim wide.
    entity_dim
        Complex embedding dimension.  entity_dim=128 yields 256-D output
        vectors, matching the kge_dim used in build_rich_hetero_graph().
    epochs
        Number of training epochs.
    batch_size
        Positive triples per mini-batch.  On a strong GPU (A100/H100) use
        32_768–65_536; the default 512 starves the GPU between Python steps.
    num_negs_per_pos
        Negatives sampled per positive triple.  Higher values give more GPU
        work per batch (closer to full negative-sampling coverage) and are
        the main lever for GPU utilisation alongside batch_size.  The RotatE
        paper uses 256; 128–512 is reasonable on modern hardware.
    lr
        Adam learning rate.
    device
        "cuda", "cpu", or None (auto-detects CUDA if available).
        PyKEEN passes this directly to PyTorch, so all tensor operations
        and the embedding tables run on the specified device.
    seed
        Random seed for reproducibility.
    checkpoint_path
        If given, the embedding dict is persisted here as a .pt file and
        can be reloaded with load_kge_checkpoint() to skip retraining.
    history_csv_path
        If given, the per-epoch training loss is written to this CSV
        (columns: epoch, train_loss) plus a "<stem>_meta.json" sidecar with
        the run hyperparameters and final/best loss. This is the on-disk
        duplicate of the W&B loss curve and needs no unpickling to read.
    wandb_project
        Weights and Biases project name.  If None (default) or if wandb is
        not installed, no W&B logging is performed.
    wandb_run_name
        Optional display name for the W&B run.  Defaults to
        "{model}_dim{entity_dim}_ep{epochs}" when wandb_project is set.

    Returns
    -------
    KGEResult with:
        embeddings       -- dict {uri_string -> float32 array (2*entity_dim,)}
        pipeline_result  -- pykeen PipelineResult (loss curves, evaluation)
        triples_factory  -- TriplesFactory (entity_to_id, relation_to_id)
    """
    triples_path = pathlib.Path(triples_path)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading triples from %s", triples_path)
    tf = TriplesFactory.from_path(triples_path)
    logger.info(
        "TriplesFactory: %d triples | %d entities | %d relations",
        tf.num_triples, tf.num_entities, tf.num_relations,
    )

    # -- Run config (shared by W&B and the on-disk metadata sidecar) ---------
    _cfg = {
        "model":            model,
        "entity_dim":       entity_dim,
        "epochs":           epochs,
        "batch_size":       batch_size,
        "num_negs_per_pos": num_negs_per_pos,
        "lr":               lr,
        "device":           resolved_device,
        "seed":             seed,
        "n_triples":        tf.num_triples,
        "n_entities":       tf.num_entities,
        "n_relations":      tf.num_relations,
    }

    # -- W&B initialisation (optional; never fatal) --------------------------
    _run = None
    if _WANDB and wandb_project is not None:
        run_name = wandb_run_name or f"{model}_dim{entity_dim}_ep{epochs}"
        try:
            _run = _wandb.init(project=wandb_project, name=run_name, config=_cfg)
            logger.info("W&B run initialised: %s / %s", wandb_project, run_name)
        except Exception as exc:  # noqa: BLE001
            _run = None
            logger.warning("W&B init failed (%s); continuing without W&B.", exc)

    logger.info(
        "Starting PyKEEN pipeline: model=%s  entity_dim=%d  epochs=%d  device=%s",
        model, entity_dim, epochs, resolved_device,
    )

    _t0 = time.perf_counter()
    # testing=tf reuses all triples so every entity is trained on; evaluation
    # metrics are inflated but are not used downstream (loss curve is the only
    # signal of record for KGE here).
    pykeen_result = pipeline(
        training=tf,
        testing=tf,
        model=model,
        model_kwargs={"embedding_dim": entity_dim},
        training_kwargs={
            "num_epochs": epochs,
            "batch_size": batch_size,
            "use_tqdm_batch": True,
        },
        negative_sampler_kwargs={"num_negs_per_pos": num_negs_per_pos},
        optimizer="Adam",
        optimizer_kwargs={"lr": lr},
        random_seed=seed,
        device=resolved_device,
    )
    _train_seconds = time.perf_counter() - _t0
    logger.info("KGE training finished in %.1fs", _train_seconds)

    # -- W&B: log per-epoch losses -------------------------------------------
    # pykeen_result.losses is a list of per-epoch mean loss floats
    if _run is not None:
        for ep, loss_val in enumerate(pykeen_result.losses, start=1):
            _wandb.log({"train/loss": loss_val, "epoch": ep}, step=ep)

        # Log final link-prediction metrics if an evaluator was used
        if pykeen_result.metric_results is not None:
            metrics = pykeen_result.metric_results.to_flat_dict()
            flat_metrics = {f"eval/{k}": v for k, v in metrics.items()
                            if isinstance(v, (int, float))}
            _wandb.summary.update(flat_metrics)

        _wandb.summary["train/seconds"] = _train_seconds
        _wandb.finish()

    # -- Extract and optionally save embeddings ------------------------------
    embeddings = _extract_embeddings(pykeen_result.model, tf)
    out_dim = next(iter(embeddings.values())).shape[0]
    logger.info("Extracted %d entity embeddings of size %d.", len(embeddings), out_dim)

    if checkpoint_path is not None:
        _save_checkpoint(embeddings, tf, checkpoint_path)

    if history_csv_path is not None:
        _save_history(pykeen_result, history_csv_path,
                      config={**_cfg, "out_dim": out_dim,
                              "train_seconds": round(_train_seconds, 2),
                              "train_time_hms": time.strftime("%H:%M:%S",
                                                              time.gmtime(_train_seconds))})

    return KGEResult(
        embeddings=embeddings,
        pipeline_result=pykeen_result,
        triples_factory=tf,
    )


def load_kge_checkpoint(path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Load a checkpoint saved by train_kge() and return the embedding dict.

    Returns a dict {uri_string -> float32 numpy array} compatible with
    build_rich_hetero_graph() without needing to retrain.
    """
    cp = torch.load(path, map_location="cpu", weights_only=False)
    uris: list[str] = cp["uris"]
    vecs: np.ndarray = cp["vecs"]   # (n_entities, out_dim) float32
    return {uri: vecs[i] for i, uri in enumerate(uris)}


# ---------------------------------------------------------------------------
# Internal save helper
# ---------------------------------------------------------------------------

def _save_checkpoint(
    embeddings: dict[str, np.ndarray],
    tf: TriplesFactory,
    path: str | pathlib.Path,
) -> None:
    """Persist the embedding dict to a .pt file loadable with weights_only=True.

    Stores two entries:
        uris -- list[str] of entity URI strings ordered by PyKEEN entity index
        vecs -- float32 numpy array of shape (n_entities, out_dim)

    The index ordering matches tf.entity_to_id so that uris[i] == vecs[i].
    """
    cp_path = pathlib.Path(path)
    cp_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(tf.entity_to_id.items(), key=lambda kv: kv[1])
    uris = [uri for uri, _ in sorted_items]
    vecs = np.stack([embeddings[uri] for uri in uris]).astype(np.float32)

    torch.save({"uris": uris, "vecs": vecs}, cp_path)
    logger.info("Checkpoint saved to %s", cp_path)


def _save_history(
    pipeline_result,
    path: str | pathlib.Path,
    *,
    config: dict | None = None,
) -> None:
    """Persist the per-epoch training loss to CSV + a metadata JSON sidecar.

    Writes two files:
        <path>            -- CSV with columns epoch, train_loss
        <path stem>_meta.json -- run hyperparameters + final/best train loss

    PyKEEN exposes the per-epoch mean loss as pipeline_result.losses (a list of
    floats); KGE here is trained without a held-out split, so there is no test
    metric to store -- the loss curve is the training signal of record.
    """
    import csv
    import json

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    losses = list(getattr(pipeline_result, "losses", []) or [])
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss"])
        for ep, loss_val in enumerate(losses, start=1):
            writer.writerow([ep, float(loss_val)])

    meta = dict(config or {})
    if losses:
        meta["final_train_loss"] = float(losses[-1])
        meta["best_train_loss"]  = float(min(losses))
        meta["n_epochs_recorded"] = len(losses)
    meta_path = p.with_name(p.stem + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("KGE history saved to %s (+ %s)", p, meta_path.name)


__all__ = (
    "KGEResult",
    "train_kge",
    "load_kge_checkpoint",
)
