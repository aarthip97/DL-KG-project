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
import torch
import numpy as np
import rdflib
from rdflib import BNode, Literal
import json
from collections import defaultdict
from torch_geometric.data import HeteroData

def extract_dl_artifacts(
    g: rdflib.Graph,
    tsv_out_path: str,
    dict_out_path: str,
    sidecar_nt: str | None = None,
):
    """Extract PyKEEN TSV triples and a PyG edge_dict from a populated KG.

    Parameters
    ----------
    g            : the in-memory rdflib Graph (populated .ttl, no listeners).
    tsv_out_path : path for the RotatE / PyKEEN TSV file.
    dict_out_path: path for the JSON edge_dict consumed by build_rich_hetero_graph.
    sidecar_nt   : optional path to the listening sidecar N-Triples file.
                   When provided the sidecar is parsed into ``g`` **in-place**
                   before extraction so user/track edges are captured.
                   This avoids the caller having to do a separate
                   ``g.parse(sidecar_nt)`` step.  The merge is skipped
                   silently if the file does not exist.
    """
    if sidecar_nt is not None:
        import pathlib as _pl
        _p = _pl.Path(sidecar_nt)
        if _p.exists():
            _n_before = len(g)
            _size_mb  = _p.stat().st_size / 1024 / 1024
            print(f"Merging sidecar {_p.name}  ({_size_mb:,.1f} MiB) …", end=" ", flush=True)
            g.parse(str(_p), format="nt")
            print(f"+{len(g) - _n_before:,} triples  (total: {len(g):,})")
        else:
            print(f"[WARN] sidecar_nt not found, skipping merge: {sidecar_nt}")

    print("1. Identifying skos:exactMatch / owl:sameAs aliases...")
    alias_map = {}
    # Catch both standard alias predicates
    for s, p, o in g.triples((None, rdflib.URIRef("http://www.w3.org/2004/02/skos/core#exactMatch"), None)):
        alias_map[str(s)] = str(o)
    for s, p, o in g.triples((None, rdflib.URIRef("http://www.w3.org/2002/07/owl#sameAs"), None)):
        alias_map[str(s)] = str(o)

    # Helper to resolve aliases instantly
    def resolve(uri_ref):
        uri_str = str(uri_ref)
        return alias_map.get(uri_str, uri_str)

    print("2. Parsing graph for PyKEEN and PyG artifacts...")
    tsv_triples = []

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

        # Add every non-BNode triple to PyKEEN TSV
        if not isinstance(s, BNode) and not isinstance(o, BNode):
            tsv_triples.append(f"{s_res}\t{p_str}\t{o_res}\n")

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

    # ── Resolve BNode listening events → user→track edges ────────────────────
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
        # Also add to TSV (user → track — compact form for KGE)
        tsv_triples.append(f"{user_str}\t{PRED_LISTENED_TO}\t{track_str}\n")

    # ── Resolve Performance nodes → track_artist / key / mode / tempo / instr ─
    for perf_str, track_str in perf_track.items():
        artist_str = perf_artist.get(perf_str)
        if artist_str and track_str:
            t_idx = get_or_create_idx("track",  track_str)
            a_idx = get_or_create_idx("artist", artist_str)
            edge_dict["track_artist"][0].append(t_idx)
            edge_dict["track_artist"][1].append(a_idx)
            # TSV: emit track—performed_by→artist (collapsed from performance)
            tsv_triples.append(
                f"{track_str}\t{MRC_BASE}performed_by\t{artist_str}\n"
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
            tsv_triples.append(
                f"{artist_str}\t{PRED_HAS_GENRE}\t{genre_str}\n"
            )

    print("3. Exporting artifacts...")
    edge_dict["node_mappings"] = node_mappings

    with open(tsv_out_path, "w", encoding="utf-8") as f:
        f.writelines(tsv_triples)

    # Single JSON write — defaultdict values are converted to plain lists first,
    # and node_mappings (also defaultdict) is serialised inline.
    serialisable = {
        k: (list(v) if isinstance(v, defaultdict) else v)
        for k, v in edge_dict.items()
        if k != "node_mappings"
    }
    serialisable["node_mappings"] = {
        ntype: list(uris)
        for ntype, uris in node_mappings.items()
    }
    with open(dict_out_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f)

    n_tracks = len(node_mappings["track"])
    n_users  = len(node_mappings["user"])
    n_edges  = len(edge_dict["user_track"][0])
    print(
        f"Done! Extracted {len(tsv_triples):,} RotatE triples  |  "
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

    for node_type, uris in edge_dict["node_mappings"].items():
        n_nodes = len(uris)

        if node_type == "track":
            # Tracks: audio (128D) || KGE (256D) = 384D
            x = torch.zeros(n_nodes, audio_dim + kge_dim, dtype=torch.float32)
            for i, uri in enumerate(uris):
                audio = torch.as_tensor(
                    track_audio_features.get(uri, np.zeros(audio_dim)),
                    dtype=torch.float32,
                )
                kge = torch.as_tensor(
                    rotate_embeddings.get(uri, np.zeros(kge_dim)),
                    dtype=torch.float32,
                )
                x[i] = torch.cat([audio, kge])
        else:
            # Metadata nodes: KGE only (256D)
            x = torch.zeros(n_nodes, kge_dim, dtype=torch.float32)
            for i, uri in enumerate(uris):
                x[i] = torch.as_tensor(
                    rotate_embeddings.get(uri, np.zeros(kge_dim)),
                    dtype=torch.float32,
                )

        data[node_type].x = x

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

