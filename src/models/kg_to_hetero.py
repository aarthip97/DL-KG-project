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
import json
from collections import defaultdict
from torch_geometric.data import HeteroData

def extract_dl_artifacts(g: rdflib.Graph, tsv_out_path: str, dict_out_path: str):
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
    # e.g., node_to_idx['track']['http://...track_1'] = 0
    node_to_idx = defaultdict(dict)
    node_mappings = defaultdict(list)
    
    # edge_dict structure expected by your PyG script
    edge_dict = {
        "user_track": [[], []],
        "track_artist": [[], []],
        "track_tempo": [[], []],
        "track_key": [[], []],
        "track_mode": [[], []],
        "track_instrument": [[], []],
        "track_decade": [[], []],
        "artist_genre": [[], []],
        "genre_parent": [[], []],
        "instrument_parent": [[], []],
        "node_mappings": {},
        # Optional weights arrays
        "track_key_weights": [],
        "track_mode_weights": [],
        "artist_genre_weights": [],
        "user_track_counts": [] 
    }

    def get_or_create_idx(node_type, uri_str):
        if uri_str not in node_to_idx[node_type]:
            idx = len(node_mappings[node_type])
            node_to_idx[node_type][uri_str] = idx
            node_mappings[node_type].append(uri_str)
        return node_to_idx[node_type][uri_str]

    # Predicate string matchers for your specific topology
    PRED_LISTEN = "listened_to"
    PRED_PERFORMED = "performed_by"
    PRED_GENRE = "has_genre"
    PRED_BROADER = "broader" # Resolves to has_parent_* in PyG

    for s, p, o in g:
        p_str = str(p)
        
        # Skip the alias edges themselves so PyKEEN doesn't waste time on them
        if "exactMatch" in p_str or "sameAs" in p_str:
            continue
            
        s_res = resolve(s)
        o_res = resolve(o)
        
        # 1. Add to PyKEEN TSV list
        tsv_triples.append(f"{s_res}\t{p_str}\t{o_res}\n")
        
        # 2. Route specific structural edges into PyG integer indices
        if PRED_LISTEN in p_str:
            u_idx = get_or_create_idx("user", s_res)
            t_idx = get_or_create_idx("track", o_res)
            edge_dict["user_track"][0].append(u_idx)
            edge_dict["user_track"][1].append(t_idx)
            # If you appended weights/counts as a separate edge or sidecar, route them here
            
        elif PRED_PERFORMED in p_str:
            t_idx = get_or_create_idx("track", s_res)
            a_idx = get_or_create_idx("artist", o_res)
            edge_dict["track_artist"][0].append(t_idx)
            edge_dict["track_artist"][1].append(a_idx)
            
        elif PRED_GENRE in p_str:
            a_idx = get_or_create_idx("artist", s_res)
            g_idx = get_or_create_idx("genre", o_res)
            edge_dict["artist_genre"][0].append(a_idx)
            edge_dict["artist_genre"][1].append(g_idx)
            
        elif PRED_BROADER in p_str:
            # Check URIs to determine if it's an instrument hierarchy or genre hierarchy
            if "Genre" in s_res or "genre" in s_res:
                child_idx = get_or_create_idx("genre", s_res)
                parent_idx = get_or_create_idx("genre", o_res)
                edge_dict["genre_parent"][0].append(child_idx)
                edge_dict["genre_parent"][1].append(parent_idx)
            else:
                child_idx = get_or_create_idx("instrument", s_res)
                parent_idx = get_or_create_idx("instrument", o_res)
                edge_dict["instrument_parent"][0].append(child_idx)
                edge_dict["instrument_parent"][1].append(parent_idx)
                
        # (Add your other track properties like key, mode, tempo here following the same pattern)

    print("3. Exporting artifacts...")
    # Finalize node mappings for PyG
    edge_dict["node_mappings"] = node_mappings

    # Write the PyKEEN TSV
    with open(tsv_out_path, "w", encoding="utf-8") as f:
        f.writelines(tsv_triples)

    # Write the PyG dictionary
    with open(dict_out_path, "w", encoding="utf-8") as f:
        json.dump(edge_dict, f)

    print(f"Done! Extracted {len(tsv_triples)} RotatE triples and mapped {len(node_mappings['track'])} tracks.")
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