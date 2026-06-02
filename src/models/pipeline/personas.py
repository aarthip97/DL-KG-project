"""Persona cold-start glue for the pipeline notebook (§14).

Two loaders that keep the §14 cells thin:

* :func:`assemble_persona_pack` — build the persona pack from the latent analysis
  (in-memory → ``data/final/latent/`` → a previously saved pack), self-healing
  the index bridges + ``song_meta`` straight from disk.
* :func:`build_cold_start` — wire the faithful-attention :class:`ColdStartHGT`
  engine, rehydrating the live model + graph from disk when needed (no retraining).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional


def assemble_persona_pack(
    *,
    latent_analysis=None,
    latent_dir,
    in_memory: Optional[Mapping[str, Any]] = None,
    edge_dict_path,
    kg_input_path,
    splits_dir,
    save_path,
    verbose: bool = True,
):
    """Build (or reload) the persona pack via the no-recompute ladder.

    Uses an in-memory :class:`LatentAnalysis` (or one on disk under
    ``latent_dir``) to build the pack; if no latent artefacts exist but a saved
    pack does, it loads that instead. Index bridges + ``song_meta`` are reused
    from ``in_memory`` when warm, else rebuilt from ``node_dict.json`` + splits +
    ``kg_input.parquet`` (no model needed). The built pack is saved to ``save_path``.
    """
    from ..latent_personas import build_persona_pack, save_persona_pack, load_persona_pack
    from ..latent_space import LatentAnalysis
    from ..evaluation import load_index_bridges_from_disk, load_song_meta

    g = in_memory or {}
    log = print if verbose else (lambda *_: None)
    latent_dir, save_path = Path(latent_dir), Path(save_path)

    la = latent_analysis if latent_analysis is not None else g.get("latent_analysis")
    has_latent = la is not None or (latent_dir / "gmm.pkl").exists()
    if not has_latent:
        if save_path.exists():
            pack = load_persona_pack(save_path)
            log(f"[persona] loaded ← {save_path.name} (latent not in memory)")
            return pack
        raise RuntimeError(
            "No latent artefacts and no persona pack. Run §13 once (it persists "
            "to data/final/latent/), then re-run this cell.")
    if la is None:
        la = LatentAnalysis.load(latent_dir)

    if all(n in g for n in ("edge_dict", "_track_kg_to_song", "idx2song")):
        edge_dict, tk2s, i2s = g["edge_dict"], g["_track_kg_to_song"], g["idx2song"]
    else:
        b = load_index_bridges_from_disk(
            edge_dict_path=edge_dict_path, kg_input_path=kg_input_path,
            splits_dir=splits_dir)
        edge_dict, tk2s, i2s = b["edge_dict"], b["track_kg_to_song"], b["idx2song"]
        log("[rehydrate] index bridges ← node_dict.json + splits (no model)")

    song_meta = g.get("song_meta")
    if song_meta is None:
        song_meta = load_song_meta(kg_input_path)
        log("[rehydrate] song_meta ← kg_input.parquet")

    pack = build_persona_pack(
        la.emb_df, edge_dict, la.gmm.means, la.gmm.best_params,
        track_kg_to_song=tk2s, idx2song=i2s, song_meta=song_meta,
        composition=la.composition,
        user_centroids=(None if la.gmm_users is None else la.gmm_users.means),
        user_composition=la.user_composition)
    save_persona_pack(pack, save_path)
    n_user = 0 if pack.user_centroids is None else pack.user_centroids.shape[0]
    log(f"[persona] {pack.centroids.shape[0]} all-type personas · {n_user} user "
        f"archetypes · {len(pack.track_meta):,} tracks → {save_path.name}")
    return pack


def build_cold_start(
    persona_pack,
    in_memory: Optional[Mapping[str, Any]],
    *,
    kg_input_path,
    splits_dir,
    hgt_model_path,
    kge_rotate_path,
    edge_dict_path,
    ae_embeddings_path,
    device: str = "cpu",
    verbose: bool = True,
):
    """Wire the :class:`ColdStartHGT` faithful-attention engine for the §14 GUI.

    Reuses the live model + graph when in memory; otherwise rebuilds them from
    disk (no retraining). Returns ``(cold_start, exposed)`` where ``cold_start``
    is ``None`` when the HGT weights are unavailable (the GUI then runs in
    cosine-only mode) and ``exposed`` are any globals rebuilt for reuse.
    """
    from ..latent_personas import ColdStartHGT

    g = in_memory or {}
    log = print if verbose else (lambda *_: None)
    exposed: dict = {}

    if all(n in g for n in ("result", "data", "edge_dict")):
        model, data, edge_dict = g["result"].model, g["data"], g["edge_dict"]
    elif Path(hgt_model_path).exists():
        from .context import ensure_hgt_context
        log("[app] rebuilding HGT from disk for attention mode (no retraining) …")
        hctx = ensure_hgt_context(
            g, kg_input_path=kg_input_path, splits_dir=splits_dir,
            hgt_model_path=hgt_model_path, kge_rotate_path=kge_rotate_path,
            edge_dict_path=edge_dict_path, ae_embeddings_path=ae_embeddings_path,
            device=device, verbose=verbose)
        if hctx is None:
            return None, exposed
        exposed = hctx.exposed()
        model, data, edge_dict = hctx.result.model, hctx.data, hctx.edge_dict
    else:
        return None, exposed

    try:
        cold_start = ColdStartHGT(model, data, edge_dict["node_mappings"], persona_pack)
    except Exception as e:  # noqa: BLE001
        log(f"[app] HGT-attention disabled ({type(e).__name__}: {e}); cosine mode only")
        return None, exposed
    return cold_start, exposed


__all__ = ["assemble_persona_pack", "build_cold_start"]
