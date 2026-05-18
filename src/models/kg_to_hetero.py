"""Convert the populated TTL + listening sidecar into a PyG HeteroData graph.

Node features
-------------
track nodes
    Concatenation of audio autoencoder features (audio_dim=128) and KGE
    embeddings (kge_dim=256 for RotatE/ComplEx with entity_dim=128):
        x shape = (n_tracks, audio_dim + kge_dim) = (n_tracks, 384)

all other node types
    KGE embedding only:
        x shape = (n_nodes, kge_dim) = (n_nodes, 256)

Edge weights
------------
user->track (listened_to)
    log1p-scaled listen counts, clamped at 1000 to dampen super-fans.
track->key / track->mode
    Confidence float supplied in edge_dict["track_key_weights"] /
    edge_dict["track_mode_weights"].  If absent, no edge_weight is set.
artist->genre
    Artist term weight from the MSD/AcousticBrainz annotation, stored in
    edge_dict["artist_genre_weights"].  If absent, no edge_weight is set.

Obtaining embeddings
--------------------
Call train_kge() from kg_embeddings.py to produce the rotate_embeddings dict.
The returned KGEResult.embeddings maps URI strings to float32 arrays.
Pass that dict here as rotate_embeddings.

Example
-------
    from src.models.kg_embeddings import train_kge
    from src.models.kg_to_hetero import build_rich_hetero_graph

    result = train_kge("data/interim/kg_triples.tsv", entity_dim=128, epochs=500)
    data = build_rich_hetero_graph(
        edge_dict=edge_dict,
        rotate_embeddings=result.embeddings,
        track_audio_features=audio_feats,
        listen_counts=counts,
    )
"""
import gc
import torch
import numpy as np
import rdflib
from rdflib import BNode, Literal
import json
from collections import defaultdict
from typing import Literal as TypingLiteral

from torch_geometric.data import HeteroData

from data.kg.canonicalize import build_canonical_map


# ─────────────────────────────────────────────────────────────────────────────
# Memory-safe sidecar streaming
# ─────────────────────────────────────────────────────────────────────────────
# rdflib Graph carries ~250–400 B of object overhead per triple, so a 200 MiB
# .nt sidecar with ~8 M listening triples balloons to >2 GiB in RAM and
# easily OOMs an 8 GiB laptop.  We avoid this by streaming the sidecar with
# a tiny line-based parser specialised for the *exact* triples our writer
# produces (see ``stream_users_to_ntriples`` in ``listening.py``):
#
#   <user>   <hasListeningInteraction>  _:bN .
#   _:bN     <onTrack>                  <track> .
#   _:bN     <listenCount>              "N"^^<xsd:integer> .
#   <user>   <listenedTo>               <track> .          (simple variant)
#
# RAM cost is ~80 B/event for the bnode→track/count dicts (≈300 MiB for
# 4 M events) instead of multi-GiB.

_MRC = "http://purl.org/ontology/mrc/"
_PRED_HAS_LISTENING_FULL = f"<{_MRC}hasListeningInteraction>"
_PRED_ON_TRACK_FULL      = f"<{_MRC}onTrack>"
_PRED_LISTEN_COUNT_FULL  = f"<{_MRC}listenCount>"
_PRED_LISTENED_TO_FULL   = f"<{_MRC}listenedTo>"


def _iter_listening_sidecar(sidecar_path, *, verbose: bool = True):
    """Yield ``(user_uri, track_uri, listen_count)`` tuples by streaming a
    listening sidecar ``.nt`` file **without** loading it into rdflib.

    Two passes over the file:

    1. **First pass** — collect ``bnode → track`` and ``bnode → count``
       dicts (and yield ``mrc:listenedTo`` simple-variant rows on the fly).
    2. **Second pass** — for each ``<user> hasListeningInteraction _:bN``
       triple, look up the cached bnode and yield a resolved row.

    Parameters
    ----------
    sidecar_path : path to the ``.nt`` file.
    verbose      : print one progress line after each pass.
    """
    import pathlib as _pl
    p = _pl.Path(sidecar_path)
    if not p.exists():
        if verbose:
            print(f"[WARN] sidecar not found, skipping stream: {p}")
        return

    bnode_track: dict[str, str] = {}
    bnode_count: dict[str, int] = {}

    # ── Pass 1 — bnode → track / count, plus simple variant pass-through ────
    n_lines = 0
    n_simple = 0
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            n_lines += 1
            # Cheap predicate filtering before splitting (most lines are skipped).
            if _PRED_ON_TRACK_FULL in line:
                # _:bN <onTrack> <track> .
                parts = line.rstrip(" .\n\r").split(" ", 2)
                if len(parts) == 3 and parts[2].startswith("<"):
                    bnode_track[parts[0]] = parts[2][1:-1]
            elif _PRED_LISTEN_COUNT_FULL in line:
                # _:bN <listenCount> "N"^^<xsd:integer> .
                first_q = line.find('"')
                second_q = line.find('"', first_q + 1)
                if first_q > 0 and second_q > first_q:
                    bn = line.split(" ", 1)[0]
                    try:
                        bnode_count[bn] = int(line[first_q + 1:second_q])
                    except ValueError:
                        pass
            elif _PRED_LISTENED_TO_FULL in line:
                # <user> <listenedTo> <track> .   (simple variant — emit now)
                parts = line.rstrip(" .\n\r").split(" ", 2)
                if (len(parts) == 3 and parts[0].startswith("<")
                        and parts[2].startswith("<")):
                    yield parts[0][1:-1], parts[2][1:-1], 1
                    n_simple += 1

    if verbose:
        print(f"  Sidecar pass 1: {n_lines:,} lines  |  "
              f"{len(bnode_track):,} listening events  |  "
              f"{n_simple:,} simple-variant edges yielded")

    if not bnode_track:
        # Simple variant only — nothing more to do.
        return

    # ── Pass 2 — resolve <user> hasListeningInteraction _:bN ────────────────
    n_resolved = 0
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            if _PRED_HAS_LISTENING_FULL not in line:
                continue
            parts = line.rstrip(" .\n\r").split(" ", 2)
            if len(parts) < 3 or not parts[0].startswith("<"):
                continue
            user = parts[0][1:-1]
            bn = parts[2]
            track = bnode_track.get(bn)
            if track is None:
                continue
            yield user, track, bnode_count.get(bn, 1)
            n_resolved += 1

    if verbose:
        print(f"  Sidecar pass 2: {n_resolved:,} user→track edges resolved")

    bnode_track.clear()
    bnode_count.clear()


def extract_dl_artifacts(
    g: rdflib.Graph,
    tsv_out_path: str,
    dict_out_path: str,
    sidecar_nt: str | None = None,
    *,
    kge_scope: TypingLiteral["semantic", "semantic+listen", "all"] = "semantic",
):
    """Extract PyKEEN TSV triples and a PyG edge_dict from a populated KG.

    Parameters
    ----------
    g            : the in-memory rdflib Graph (populated .ttl, no listeners).
    tsv_out_path : path for the RotatE / PyKEEN TSV file.
    dict_out_path: path for the JSON edge_dict consumed by build_rich_hetero_graph.
    sidecar_nt   : optional path to the listening sidecar N-Triples file.
                   When provided the sidecar is **streamed** (line-based parser,
                   no rdflib load) and user-track edges are emitted directly.
                   ``g`` is *not* mutated, so the caller's graph stays small
                   and the function is safe to run on machines that cannot
                   afford to materialise the merged graph in RAM.
    kge_scope    : which triples enter the PyKEEN TSV (the PyG edge_dict is
                   always fully populated regardless).

                   * ``"semantic"`` (default) -- semantic backbone only.
                     Listening, OWL machinery, and reflexive/alias triples
                     are excluded.  Recommended for a recsys pipeline where
                     HGT consumes user-track interactions and KGE only seeds
                     metadata node features.  Avoids leaking held-out
                     listening events into KGE.
                   * ``"semantic+listen"`` -- semantic backbone PLUS
                     user-track listening triples.  Use only when KGE is
                     expected to score interactions directly.
                   * ``"all"`` -- writes every non-BNode triple seen
                     (legacy behaviour, kept for backwards compatibility).

    Side effects
    ------------
    Writes ``tsv_out_path`` (TSV for pykeen.triples.TriplesFactory) and
    ``dict_out_path`` (JSON; see edge_dict structure below).  The JSON
    additionally contains ``canonical_map`` mapping each non-canonical URI
    to its canonical form, so ``build_rich_hetero_graph`` can resolve
    embedding look-ups across alias classes.
    """
    # Stream TSV rows straight to disk to avoid keeping millions of
    # f-string objects in memory.  We open the file early so all writers
    # (graph traversal + sidecar streamer) share it.
    tsv_fh = open(tsv_out_path, "w", encoding="utf-8")
    tsv_count = 0

    def _tsv_write(line: str) -> None:
        nonlocal tsv_count
        tsv_fh.write(line)
        tsv_count += 1

    print("1. Building canonical-URI map (owl:sameAs + skos:exactMatch)...")
    # Collect equivalence pairs in BOTH directions, then collapse via
    # path-compressed union-find with deterministic canonical preference
    # (mrc:resource > mrc: > wd: > mo: > foaf: > lex-min).  Shared with the
    # GraphDB exporter so the two paths agree on canonical URIs.
    _EXACT_MATCH = rdflib.URIRef("http://www.w3.org/2004/02/skos/core#exactMatch")
    _SAME_AS     = rdflib.URIRef("http://www.w3.org/2002/07/owl#sameAs")

    def _iter_equiv_pairs():
        for s, _, o in g.triples((None, _EXACT_MATCH, None)):
            yield str(s), str(o)
        for s, _, o in g.triples((None, _SAME_AS, None)):
            yield str(s), str(o)

    canon_map = build_canonical_map(_iter_equiv_pairs())
    n_canonical = len(set(canon_map.values()))
    print(
        f"   Equivalence classes: {n_canonical:,} canonical URIs cover "
        f"{len(canon_map):,} aliased URIs"
    )

    # Helper to resolve a URI (or rdflib node) to its canonical form
    def resolve(uri_ref) -> str:
        uri_str = str(uri_ref)
        return canon_map.get(uri_str, uri_str)

    print("2. Parsing graph for PyKEEN and PyG artifacts...")

    # Dictionaries to build PyG integer indices
    node_to_idx  = defaultdict(dict)
    node_mappings = defaultdict(list)

    # edge_dict structure expected by build_rich_hetero_graph
    edge_dict = {
        "user_track":        [[], []],
        "track_artist":      [[], []],
        "track_tempo":       [[], []],
        "track_key":         [[], []],
        "track_mode":        [[], []],
        "track_instrument":  [[], []],
        "track_decade":      [[], []],
        "artist_genre":      [[], []],
        "genre_parent":      [[], []],
        "instrument_parent": [[], []],
        "node_mappings":     {},
        # Optional weight arrays (populated below)
        "track_key_weights":    [],
        "track_mode_weights":   [],
        "artist_genre_weights": [],
        "user_track_counts":    [],
    }

    def get_or_create_idx(node_type, uri_str):
        if uri_str not in node_to_idx[node_type]:
            idx = len(node_mappings[node_type])
            node_to_idx[node_type][uri_str] = idx
            node_mappings[node_type].append(uri_str)
        return node_to_idx[node_type][uri_str]

    # ── Full predicate URIs as used by KGBuilder / listening.py ──────────────
    # (never rely on substring matching — predicates must be exact)
    MRC_BASE = "http://purl.org/ontology/mrc/"
    MO_BASE  = "http://purl.org/ontology/mo/"

    PRED_LISTENED_TO            = MRC_BASE + "listenedTo"           # user → track (simple)
    PRED_HAS_LISTENING          = MRC_BASE + "hasListeningInteraction" # user → BNode
    PRED_ON_TRACK               = MRC_BASE + "onTrack"              # BNode → track
    PRED_LISTEN_COUNT           = MRC_BASE + "listenCount"          # BNode → int
    PRED_HAS_TRACK              = MRC_BASE + "hasTrack"             # performance → track
    PRED_PERFORMER              = MO_BASE  + "performer"            # performance → artist
    PRED_HAS_GENRE              = MRC_BASE + "hasGenre"             # artist → genre
    PRED_HAS_GENRE_ASSOC        = MRC_BASE + "hasGenreAssoc"        # artist → BNode
    PRED_GENRE_IN_ASSOC         = MRC_BASE + "genre"               # BNode → genre
    PRED_GENRE_WEIGHT           = MRC_BASE + "weight"              # BNode → float
    PRED_HAS_KEY                = MRC_BASE + "hasKey"               # track/perf → key
    PRED_HAS_MODE               = MRC_BASE + "hasMode"              # track/perf → mode
    PRED_HAS_TEMPO_CLASS        = MRC_BASE + "hasTempoClass"        # perf → tempo class
    PRED_INSTRUMENT             = MO_BASE  + "instrument"           # perf → instrument
    PRED_IN_DECADE              = MRC_BASE + "inDecade"             # track → decade
    PRED_SKOS_BROADER           = "http://www.w3.org/2004/02/skos/core#broader"

    # ── OWL / RDF machinery predicates excluded from KGE TSV ─────────────────
    # These describe class restrictions and equivalence axioms, not real
    # binary facts about instances.  Including them would inflate the TSV
    # with self-loops and constructs PyKEEN cannot learn meaningfully.
    _OWL_MACHINERY_PREDS = frozenset({
        "http://www.w3.org/2002/07/owl#onProperty",
        "http://www.w3.org/2002/07/owl#someValuesFrom",
        "http://www.w3.org/2002/07/owl#allValuesFrom",
        "http://www.w3.org/2002/07/owl#hasValue",
        "http://www.w3.org/2002/07/owl#equivalentClass",
        "http://www.w3.org/2002/07/owl#sameAs",
        "http://www.w3.org/2002/07/owl#disjointWith",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#first",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#rest",
        "http://www.w3.org/2004/02/skos/core#exactMatch",
    })

    # Listening-related predicates -- gated by kge_scope
    _LISTENING_PREDS = frozenset({
        PRED_LISTENED_TO,
        PRED_HAS_LISTENING,
        PRED_ON_TRACK,
        PRED_LISTEN_COUNT,
    })

    def _allow_in_tsv(pred_uri: str, head_uri: str, tail_uri: str) -> bool:
        """Return True iff this triple should be written to the PyKEEN TSV."""
        if kge_scope == "all":
            # Legacy behaviour: only filter self-loops and BNode artefacts
            return head_uri != tail_uri
        if pred_uri in _OWL_MACHINERY_PREDS:
            return False
        if kge_scope == "semantic" and pred_uri in _LISTENING_PREDS:
            return False
        # Drop self-loops -- KGE cannot learn anything from them.
        return head_uri != tail_uri

    # ── Per-entity-type resource namespace prefixes ──────────────────────────
    # Every individual minted by KGBuilder lives in a namespace dedicated to
    # its type, so we can identify a node's type from its URI alone:
    #   track/  artist/  user/  genre/  instrument/  decade/
    #   tempo/  key/  mode/  performance/
    # We only need the ones used for ambiguity resolution below.
    RES_BASE       = "http://purl.org/ontology/mrc/resource/"
    NS_GENRE       = RES_BASE + "genre/"
    NS_INSTRUMENT  = RES_BASE + "instrument/"
    NS_PERFORMANCE = RES_BASE + "performance/"

    def _is_performance(uri_str: str) -> bool:
        return uri_str.startswith(NS_PERFORMANCE)

    def _is_genre(uri_str: str) -> bool:
        return uri_str.startswith(NS_GENRE)

    # ── First pass: collect BNode roles ──────────────────────────────────────
    # A BNode can be a ListeningEvent (user→track) or a GenreAssociation
    # (artist→genre+weight).  We resolve them in a first scan so the second
    # pass can emit clean direct edges.
    bnode_track: dict[str, str]        = {}  # bnode_str → track_uri_str
    bnode_count: dict[str, int]        = {}  # bnode_str → listen count
    bnode_genre: dict[str, str]        = {}  # bnode_str → genre_uri_str
    bnode_weight: dict[str, float]     = {}  # bnode_str → genre weight
    # performance → track / artist (collected so we can emit track↔artist edges)
    perf_track:  dict[str, str]        = {}  # perf_uri → track_uri
    perf_artist: dict[str, str]        = {}  # perf_uri → artist_uri
    # performance → key / mode / tempo (indexed by perf, will be forwarded to track)
    perf_key:    dict[str, str]        = {}  # perf_uri → key_uri
    perf_mode:   dict[str, str]        = {}  # perf_uri → mode_uri
    perf_tempo:  dict[str, str]        = {}  # perf_uri → tempo_class_uri
    perf_instrument: dict[str, list]   = defaultdict(list)  # perf_uri → [inst_uri]
    # direct track-level key/mode (simple graph variant)
    track_key_direct:  dict[str, str]  = {}  # track_uri → key_uri
    track_mode_direct: dict[str, str]  = {}  # track_uri → mode_uri
    # user → {bnode_str} (to resolve listening events)
    user_listening_bnodes: dict[str, str] = {}  # bnode_str → user_uri

    for s, p, o in g:
        p_str = str(p)
        s_str = str(s)
        o_str = str(o) if not isinstance(o, Literal) else None

        if p_str == PRED_HAS_LISTENING and o_str:
            user_listening_bnodes[o_str] = s_str          # BNode → user
        elif p_str == PRED_ON_TRACK and o_str:
            bnode_track[s_str] = o_str                    # BNode → track
        elif p_str == PRED_LISTEN_COUNT and isinstance(o, Literal):
            try:
                bnode_count[s_str] = int(o)
            except (ValueError, TypeError):
                pass
        elif p_str == PRED_HAS_TRACK and o_str:
            perf_track[s_str] = o_str
        elif p_str == PRED_PERFORMER and o_str:
            perf_artist[s_str] = o_str
        elif p_str == PRED_HAS_KEY and o_str:
            # Could be on a performance (rich) or track (simple)
            if _is_performance(s_str):
                perf_key[s_str] = o_str
            else:
                track_key_direct[s_str] = o_str
        elif p_str == PRED_HAS_MODE and o_str:
            if _is_performance(s_str):
                perf_mode[s_str] = o_str
            else:
                track_mode_direct[s_str] = o_str
        elif p_str == PRED_HAS_TEMPO_CLASS and o_str:
            perf_tempo[s_str] = o_str
        elif p_str == PRED_INSTRUMENT and o_str:
            perf_instrument[s_str].append(o_str)
        elif p_str == PRED_HAS_GENRE_ASSOC and o_str:
            pass  # just a container, resolved via PRED_GENRE_IN_ASSOC below
        elif p_str == PRED_GENRE_IN_ASSOC and o_str:
            bnode_genre[s_str] = o_str
        elif p_str == PRED_GENRE_WEIGHT and isinstance(o, Literal):
            try:
                bnode_weight[s_str] = float(o)
            except (ValueError, TypeError):
                pass

    # ── Second pass: emit TSV + PyG edges ────────────────────────────────────
    # Track which (artist, genre-assoc-bnode) links exist so we can map weights
    artist_genre_assoc: dict[str, list] = defaultdict(list)  # artist_uri → [bnode]

    for s, p, o in g:
        p_str = str(p)
        s_str = str(s)
        o_str = str(o) if not isinstance(o, Literal) else None

        if "exactMatch" in p_str or "sameAs" in p_str:
            continue

        s_res = resolve(s)
        o_res = resolve(o) if o_str else str(o)

        # Add every non-BNode triple to PyKEEN TSV (filtered by kge_scope)
        if (
            not isinstance(s, BNode)
            and not isinstance(o, BNode)
            and o_str is not None
            and _allow_in_tsv(p_str, s_res, o_res)
        ):
            _tsv_write(f"{s_res}\t{p_str}\t{o_res}\n")

        # ── PyG structural edges ──────────────────────────────────────────────

        # 1. User → Track  (simple: mrc:listenedTo)
        if p_str == PRED_LISTENED_TO and o_str:
            u_idx = get_or_create_idx("user",  s_res)
            t_idx = get_or_create_idx("track", o_res)
            edge_dict["user_track"][0].append(u_idx)
            edge_dict["user_track"][1].append(t_idx)
            edge_dict["user_track_counts"].append(1)

        # 2. Artist → Genre  (direct: mrc:hasGenre)
        elif p_str == PRED_HAS_GENRE and o_str:
            a_idx = get_or_create_idx("artist", s_res)
            g_idx = get_or_create_idx("genre",  o_res)
            edge_dict["artist_genre"][0].append(a_idx)
            edge_dict["artist_genre"][1].append(g_idx)
            edge_dict["artist_genre_weights"].append(1.0)  # weight resolved below

        # 3. Artist → Genre via association BNode (mrc:hasGenreAssoc)
        elif p_str == PRED_HAS_GENRE_ASSOC and o_str:
            artist_genre_assoc[s_res].append(o_str)

        # 4. SKOS broader (genre/instrument hierarchies)
        elif p_str == PRED_SKOS_BROADER and o_str:
            # Identify the child node by its resource namespace prefix —
            # cleaner and safer than substring matching on labels.
            if _is_genre(s_res):
                child_idx  = get_or_create_idx("genre", s_res)
                parent_idx = get_or_create_idx("genre", o_res)
                edge_dict["genre_parent"][0].append(child_idx)
                edge_dict["genre_parent"][1].append(parent_idx)
            elif s_res.startswith(NS_INSTRUMENT):
                child_idx  = get_or_create_idx("instrument", s_res)
                parent_idx = get_or_create_idx("instrument", o_res)
                edge_dict["instrument_parent"][0].append(child_idx)
                edge_dict["instrument_parent"][1].append(parent_idx)
            # else: skos:broader on schemes / WD upper concepts — skip

        # 5. Track key/mode (simple graph variant — attached directly to track)
        elif p_str == PRED_HAS_KEY and not _is_performance(s_str) and o_str:
            t_idx = get_or_create_idx("track", s_res)
            k_idx = get_or_create_idx("key",   o_res)
            edge_dict["track_key"][0].append(t_idx)
            edge_dict["track_key"][1].append(k_idx)

        elif p_str == PRED_HAS_MODE and not _is_performance(s_str) and o_str:
            t_idx = get_or_create_idx("track", s_res)
            m_idx = get_or_create_idx("mode",  o_res)
            edge_dict["track_mode"][0].append(t_idx)
            edge_dict["track_mode"][1].append(m_idx)

        # 6. Track → Decade
        elif p_str == PRED_IN_DECADE and o_str:
            t_idx = get_or_create_idx("track",  s_res)
            d_idx = get_or_create_idx("decade", o_res)
            edge_dict["track_decade"][0].append(t_idx)
            edge_dict["track_decade"][1].append(d_idx)

    # ── Resolve BNode listening events that landed in g (legacy / merged) ────
    # In the new streaming-sidecar architecture g normally carries no user
    # data, so this loop is a no-op.  We keep it for backwards compatibility
    # with callers that still merge the sidecar into g manually.
    for bnode_str, user_str in user_listening_bnodes.items():
        track_str = bnode_track.get(bnode_str)
        if track_str is None:
            continue
        count = bnode_count.get(bnode_str, 1)
        u_idx = get_or_create_idx("user",  user_str)
        t_idx = get_or_create_idx("track", track_str)
        edge_dict["user_track"][0].append(u_idx)
        edge_dict["user_track"][1].append(t_idx)
        edge_dict["user_track_counts"].append(count)
        if _allow_in_tsv(PRED_LISTENED_TO, user_str, track_str):
            _tsv_write(f"{user_str}\t{PRED_LISTENED_TO}\t{track_str}\n")

    # Free the bnode dicts now that g-walk users have been resolved.
    user_listening_bnodes.clear()
    bnode_track.clear()
    bnode_count.clear()
    gc.collect()

    # ── Stream the listening sidecar (memory-safe) ───────────────────────────
    # _iter_listening_sidecar yields (user, track, count) triples with a
    # tiny line-based parser — never loads the file into rdflib.
    if sidecar_nt is not None:
        print(f"3. Streaming listening sidecar → user→track edges  ({sidecar_nt})")
        for user_str, track_str, count in _iter_listening_sidecar(
            sidecar_nt, verbose=True
        ):
            u_idx = get_or_create_idx("user",  user_str)
            t_idx = get_or_create_idx("track", track_str)
            edge_dict["user_track"][0].append(u_idx)
            edge_dict["user_track"][1].append(t_idx)
            edge_dict["user_track_counts"].append(count)
            if _allow_in_tsv(PRED_LISTENED_TO, user_str, track_str):
                _tsv_write(f"{user_str}\t{PRED_LISTENED_TO}\t{track_str}\n")
        gc.collect()

    # ── Resolve Performance nodes → track_artist / key / mode / tempo / instr ─
    for perf_str, track_str in perf_track.items():
        artist_str = perf_artist.get(perf_str)
        if artist_str and track_str:
            t_idx = get_or_create_idx("track",  track_str)
            a_idx = get_or_create_idx("artist", artist_str)
            edge_dict["track_artist"][0].append(t_idx)
            edge_dict["track_artist"][1].append(a_idx)
            # TSV: emit track-performed_by-artist (collapsed from performance)
            _perf_pred = MRC_BASE + "performed_by"
            if _allow_in_tsv(_perf_pred, track_str, artist_str):
                _tsv_write(
                    f"{track_str}\t{_perf_pred}\t{artist_str}\n"
                )

        key_str = perf_key.get(perf_str)
        if key_str and track_str:
            t_idx = get_or_create_idx("track", track_str)
            k_idx = get_or_create_idx("key",   key_str)
            edge_dict["track_key"][0].append(t_idx)
            edge_dict["track_key"][1].append(k_idx)

        mode_str = perf_mode.get(perf_str)
        if mode_str and track_str:
            t_idx = get_or_create_idx("track", track_str)
            m_idx = get_or_create_idx("mode",  mode_str)
            edge_dict["track_mode"][0].append(t_idx)
            edge_dict["track_mode"][1].append(m_idx)

        tempo_str = perf_tempo.get(perf_str)
        if tempo_str and track_str:
            t_idx = get_or_create_idx("track",      track_str)
            tc_idx = get_or_create_idx("tempo_class", tempo_str)
            edge_dict["track_tempo"][0].append(t_idx)
            edge_dict["track_tempo"][1].append(tc_idx)

        for inst_str in perf_instrument.get(perf_str, []):
            if track_str:
                t_idx  = get_or_create_idx("track",      track_str)
                i_idx  = get_or_create_idx("instrument", inst_str)
                edge_dict["track_instrument"][0].append(t_idx)
                edge_dict["track_instrument"][1].append(i_idx)

    # ── Resolve genre-association BNodes → weighted artist→genre edges ────────
    for artist_str, bnodes in artist_genre_assoc.items():
        for bn in bnodes:
            genre_str  = bnode_genre.get(bn)
            weight_val = bnode_weight.get(bn, 1.0)
            if genre_str is None:
                continue
            a_idx = get_or_create_idx("artist", artist_str)
            g_idx = get_or_create_idx("genre",  genre_str)
            edge_dict["artist_genre"][0].append(a_idx)
            edge_dict["artist_genre"][1].append(g_idx)
            edge_dict["artist_genre_weights"].append(weight_val)
            if _allow_in_tsv(PRED_HAS_GENRE, artist_str, genre_str):
                _tsv_write(
                    f"{artist_str}\t{PRED_HAS_GENRE}\t{genre_str}\n"
                )

    print("3. Exporting artifacts...")
    edge_dict["node_mappings"] = node_mappings
    edge_dict["canonical_map"] = canon_map

    # TSV is already written incrementally — just close the handle.
    tsv_fh.close()

    # Single JSON write — defaultdict values are converted to plain lists first,
    # and node_mappings (also defaultdict) is serialised inline.
    serialisable = {
        k: (list(v) if isinstance(v, defaultdict) else v)
        for k, v in edge_dict.items()
        if k not in ("node_mappings", "canonical_map")
    }
    serialisable["node_mappings"] = {
        ntype: list(uris)
        for ntype, uris in node_mappings.items()
    }
    serialisable["canonical_map"] = canon_map
    with open(dict_out_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f)

    n_tracks = len(node_mappings["track"])
    n_users  = len(node_mappings["user"])
    n_edges  = len(edge_dict["user_track"][0])
    print(
        f"Done! Extracted {tsv_count:,} RotatE triples  |  "
        f"tracks={n_tracks:,}  users={n_users:,}  user→track edges={n_edges:,}"
    )
    return edge_dict


def build_rich_hetero_graph(
    edge_dict: dict,
    rotate_embeddings: dict,
    track_audio_features: dict,
    listen_counts: np.ndarray | None = None,
) -> HeteroData:
    
    data = HeteroData()

    # -- Node features -------------------------------------------------------
    # RotatE entity_dim=128 -> 256D real vector [real || imag]
    kge_dim   = 256
    audio_dim = 128

    # Canonical map: alias URI -> canonical URI.  Built by extract_dl_artifacts
    # so embedding look-ups can transparently resolve owl:sameAs /
    # skos:exactMatch aliases (PyKEEN only trained on canonical URIs).
    canon_map: dict = edge_dict.get("canonical_map", {})

    def _kge_lookup(uri: str) -> np.ndarray:
        return rotate_embeddings.get(
            canon_map.get(uri, uri),
            np.zeros(kge_dim),
        )

    missing_kge = 0
    total_nodes = 0

    for node_type, uris in edge_dict["node_mappings"].items():
        n_nodes = len(uris)
        total_nodes += n_nodes

        if node_type == "track":
            # Tracks: audio (128D) || KGE (256D) = 384D
            x = torch.zeros(n_nodes, audio_dim + kge_dim, dtype=torch.float32)
            for i, uri in enumerate(uris):
                audio = torch.as_tensor(
                    track_audio_features.get(uri, np.zeros(audio_dim)),
                    dtype=torch.float32,
                )
                kge_vec = _kge_lookup(uri)
                if not np.any(kge_vec):
                    missing_kge += 1
                kge = torch.as_tensor(kge_vec, dtype=torch.float32)
                x[i] = torch.cat([audio, kge])
        else:
            # Metadata nodes: KGE only (256D)
            x = torch.zeros(n_nodes, kge_dim, dtype=torch.float32)
            for i, uri in enumerate(uris):
                kge_vec = _kge_lookup(uri)
                if not np.any(kge_vec):
                    missing_kge += 1
                x[i] = torch.as_tensor(kge_vec, dtype=torch.float32)

        data[node_type].x = x

    if total_nodes:
        pct = 100.0 * missing_kge / total_nodes
        print(
            f"build_rich_hetero_graph: {missing_kge:,}/{total_nodes:,} "
            f"nodes ({pct:.1f}%) have a zero KGE slice"
        )
        if pct > 25.0:
            print(
                "  WARNING: more than 25% of nodes have no KGE embedding. "
                "Check that the canonical_map and rotate_embeddings are "
                "consistent (same canonicalisation, same training corpus)."
            )

    # -- Edge topology -------------------------------------------------------
    # User interactions
    data["user", "listened_to", "track"].edge_index = torch.tensor(
        edge_dict["user_track"], dtype=torch.long
    )

    # Track properties
    data["track", "performed_by", "artist"].edge_index  = torch.tensor(edge_dict["track_artist"],     dtype=torch.long)
    data["track", "has_tempo",    "tempo_class"].edge_index = torch.tensor(edge_dict["track_tempo"],  dtype=torch.long)
    data["track", "has_key",      "key"].edge_index         = torch.tensor(edge_dict["track_key"],    dtype=torch.long)
    data["track", "has_mode",     "mode"].edge_index        = torch.tensor(edge_dict["track_mode"],   dtype=torch.long)
    data["track", "has_instrument","instrument"].edge_index = torch.tensor(edge_dict["track_instrument"], dtype=torch.long)
    data["track", "in_decade",    "decade"].edge_index      = torch.tensor(edge_dict["track_decade"], dtype=torch.long)

    # Artist connections
    data["artist", "has_genre", "genre"].edge_index = torch.tensor(
        edge_dict["artist_genre"], dtype=torch.long
    )

    # Semantic hierarchies
    data["genre",      "has_parent_genre",      "genre"].edge_index      = torch.tensor(edge_dict["genre_parent"],      dtype=torch.long)
    data["instrument", "has_parent_instrument", "instrument"].edge_index = torch.tensor(edge_dict["instrument_parent"], dtype=torch.long)

    # -- Edge weights --------------------------------------------------------
    # user -> track: log1p-scaled listen counts, clamped to prevent super-fan dominance
    if listen_counts is not None:
        clamped = torch.clamp(
            torch.as_tensor(listen_counts, dtype=torch.float32), max=1_000.0
        )
        data["user", "listened_to", "track"].edge_weight = torch.log1p(clamped)

    # track -> key: pitch detection confidence (0..1 float per edge)
    _set_optional_weight(data, ("track", "has_key", "key"),   edge_dict, "track_key_weights")

    # track -> mode: key/mode confidence (0..1 float per edge)
    _set_optional_weight(data, ("track", "has_mode", "mode"), edge_dict, "track_mode_weights")

    # artist -> genre: MSD artist term weight (float per edge)
    _set_optional_weight(data, ("artist", "has_genre", "genre"), edge_dict, "artist_genre_weights")

    return data


def _set_optional_weight(
    data: HeteroData,
    edge_type: tuple[str, str, str],
    edge_dict: dict,
    key: str,
) -> None:
    """Attach edge_weight to edge_type from edge_dict[key] if the key exists.

    This is a no-op when the key is missing, so callers do not have to guard
    against absent confidence values.
    """
    weights = edge_dict.get(key)
    if weights is not None:
        data[edge_type].edge_weight = torch.as_tensor(weights, dtype=torch.float32)

