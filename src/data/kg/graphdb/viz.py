"""Plot helpers that read the CSVs produced by :mod:`graphdb.exports`.

The functions here are deliberately *file-driven* — they take a CSV path
and produce a PNG path — so the same code works whether you ran the
exports locally or pulled them down from Google Drive.

All figures are saved with ``bbox_inches='tight'`` so they drop straight
into a notebook ``IPython.display.Image`` call without further fiddling.

Subgraph visualisation
----------------------
:func:`plot_subgraph` pulls a CONSTRUCT result from a live GraphDB endpoint
and produces two outputs:

* **Interactive HTML** (pyvis / vis.js) — open in any browser; drag nodes,
  pin/unpin them with a double-click, toggle physics, etc.
* **High-res static PNG** (matplotlib, 300 dpi) — suitable for papers /
  reports.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:  # only imported for type-checkers; not at runtime
    import networkx as nx
    import rdflib

log = logging.getLogger(__name__)


# ── small private helpers ──────────────────────────────────────────────
def _save(fig: Figure, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure → %s", out_path)
    return out_path


def _bar(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    out_path: Path,
    *,
    top_n: int = 20,
    horizontal: bool = True,
) -> Path:
    """Generic bar chart from a 2-column dataframe."""
    if label_col not in df.columns or value_col not in df.columns:
        raise KeyError(
            f"Expected columns '{label_col}' and '{value_col}' in "
            f"DataFrame; got {list(df.columns)}"
        )
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    sub = df.nlargest(top_n, value_col).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(sub))))
    if horizontal:
        ax.barh(sub[label_col].astype(str), sub[value_col])
        ax.set_xlabel(value_col)
    else:
        ax.bar(sub[label_col].astype(str), sub[value_col])
        ax.set_ylabel(value_col)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out_path)


# ── public plot functions ──────────────────────────────────────────────
def plot_genre_distribution(csv_path: Path, out_path: Path,
                            top_n: int = 20) -> Path:
    df = pd.read_csv(csv_path)
    value = "total_weight" if "total_weight" in df.columns else "n_artists"
    return _bar(df, "genreLabel", value,
                f"Top {top_n} genres by {value}", out_path, top_n=top_n)


def plot_key_distribution(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    value = "n_high_confidence" if "n_high_confidence" in df.columns else "n"
    return _bar(df, "keyLabel", value,
                "Detected musical keys", out_path,
                top_n=24, horizontal=False)


def plot_node_type_histogram(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    type_col = "type" if "type" in df.columns else df.columns[0]
    n_col    = "n"    if "n"    in df.columns else df.columns[-1]
    # Strip the long URI prefix for readability — keep only the last path
    # segment / fragment.
    df = df.copy()
    df[type_col] = (
        df[type_col].astype(str)
                    .str.rsplit("#", n=1).str[-1]
                    .str.rsplit("/", n=1).str[-1]
    )
    return _bar(df, type_col, n_col,
                "KG node type distribution", out_path, top_n=30)


def plot_relation_histogram(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    rel_col = "r" if "r" in df.columns else df.columns[0]
    n_col   = "n" if "n" in df.columns else df.columns[-1]
    df = df.copy()
    df[rel_col] = (
        df[rel_col].astype(str)
                   .str.rsplit("#", n=1).str[-1]
                   .str.rsplit("/", n=1).str[-1]
    )
    return _bar(df, rel_col, n_col,
                "Predicate (relation) usage", out_path, top_n=30)


# ── batch driver ───────────────────────────────────────────────────────
# Maps each known stats CSV to the function that knows how to plot it.
# Anything not in this map is simply skipped (no error) — keeping the
# stats catalog and the plot catalog decoupled.
_PLOTTERS = {
    "genres_simple":         plot_genre_distribution,
    "genres_rich":           plot_genre_distribution,
    "confident_keys_simple": plot_key_distribution,
    "confident_keys_rich":   plot_key_distribution,
    "node_type_histogram":   plot_node_type_histogram,
    "relation_histogram":    plot_relation_histogram,
}


def plot_all(stats_dir: Path, plots_dir: Path,
             only: Optional[list[str]] = None) -> dict[str, Path]:
    """Render every stats CSV in ``stats_dir`` for which we have a plotter.

    Returns a dict of stats-name → output PNG path.
    """
    stats_dir = Path(stats_dir)
    plots_dir = Path(plots_dir)
    out: dict[str, Path] = {}
    for name, plotter in _PLOTTERS.items():
        if only is not None and name not in only:
            continue
        csv = stats_dir / f"{name}.csv"
        if not csv.is_file():
            log.debug("No CSV for %s; skipping plot", name)
            continue
        try:
            out[name] = plotter(csv, plots_dir / f"{name}.png")
        except (KeyError, ValueError) as e:
            log.warning("Could not plot %s: %s", name, e)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph visualisation — interactive pyvis HTML + high-res matplotlib PNG
# Adapted from the SPARQL CONSTRUCT queries in docs/graphdb_visual_graph_config.md
# ─────────────────────────────────────────────────────────────────────────────

# Default CONSTRUCT — Option B: mixed T-Box + A-Box sampler (schema skeleton
# plus a handful of real tracks so you immediately see actual data).
# The {limit} placeholder is filled at call time.
_DEFAULT_CONSTRUCT = textwrap.dedent("""\
    PREFIX mrc:  <http://purl.org/ontology/mrc/>
    PREFIX mo:   <http://purl.org/ontology/mo/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX dct:  <http://purl.org/dc/terms/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl:  <http://www.w3.org/2002/07/owl#>
    PREFIX wdt:  <http://www.wikidata.org/prop/direct/>

    CONSTRUCT {{
        ?cls  a owl:Class ; rdfs:label ?clsLabel .
        ?cls  rdfs:subClassOf ?parent .
        ?prop a owl:ObjectProperty ;
              rdfs:domain ?domain ; rdfs:range ?range ; rdfs:label ?propLabel .
        ?track  a mrc:MSDTrack ; dct:title ?title .
        ?track  mrc:hasGenre  ?genre .
        ?genre  skos:prefLabel ?genreLabel .
        ?artist foaf:name     ?artistName .
        ?perf   a mo:Performance ;
                mrc:hasTrack  ?track ;
                mo:performer  ?artist ;
                mrc:hasKey    ?key ;
                mrc:hasMode   ?mode .
        ?key    skos:prefLabel ?keyLabel .
        ?mode   skos:prefLabel ?modeLabel .
        ?decade a mrc:Decade ; skos:prefLabel ?decadeLabel .
        ?track  mrc:releasedInDecade ?decade .
    }}
    WHERE {{
        {{ ?cls a owl:Class . OPTIONAL {{ ?cls rdfs:label ?clsLabel }} }}
        UNION {{ ?cls rdfs:subClassOf ?parent . FILTER(isIRI(?parent)) }}
        UNION {{
            ?prop a owl:ObjectProperty .
            OPTIONAL {{ ?prop rdfs:domain ?domain }}
            OPTIONAL {{ ?prop rdfs:range  ?range  }}
            OPTIONAL {{ ?prop rdfs:label  ?propLabel }}
        }}
        UNION {{
            {{ SELECT ?track WHERE {{ ?track a mrc:MSDTrack }} LIMIT {n_tracks} }}
            ?track dct:title ?title .
            OPTIONAL {{ ?track mrc:hasGenre ?genre . ?genre skos:prefLabel ?genreLabel }}
            OPTIONAL {{
                ?perf mrc:hasTrack ?track ;
                      mo:performer ?artist .
                ?artist foaf:name ?artistName .
                OPTIONAL {{ ?perf mrc:hasKey  ?key  . ?key  skos:prefLabel ?keyLabel  }}
                OPTIONAL {{ ?perf mrc:hasMode ?mode . ?mode skos:prefLabel ?modeLabel }}
            }}
            OPTIONAL {{
                ?track mrc:releasedInDecade ?decade .
                ?decade skos:prefLabel ?decadeLabel
            }}
        }}
    }}
""")

# ── Colour palette by rdf:type local name ─────────────────────────────────────
_TYPE_COLOURS: dict[str, str] = {
    "MSDTrack":         "#4FC3F7",   # sky blue
    "Performance":      "#FFB74D",   # amber
    "MusicArtist":      "#81C784",   # green
    "Listener":         "#F06292",   # pink
    "Concept":          "#CE93D8",   # purple  (genres, keys, modes)
    "Decade":           "#4DB6AC",   # teal
    "Class":            "#B0BEC5",   # light grey  (owl:Class T-Box nodes)
    "ObjectProperty":   "#78909C",   # darker grey
    "_default":         "#EEEEEE",
}

_TYPE_SIZES: dict[str, int] = {
    "MSDTrack":         28,
    "Performance":      20,
    "MusicArtist":      26,
    "Listener":         18,
    "Concept":          22,
    "Decade":           20,
    "Class":            16,
    "ObjectProperty":   14,
    "_default":         18,
}


def _local_name(uri: str) -> str:
    """Return the fragment / last path segment of a URI."""
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


def _best_label(uri: str, label_map: dict[str, str]) -> str:
    return label_map.get(uri) or _local_name(uri)


def _detect_type(uri: str, type_map: dict[str, list[str]]) -> str:
    for t in type_map.get(uri, []):
        name = _local_name(t)
        if name in _TYPE_COLOURS:
            return name
    # Heuristic fallbacks from URI prefix
    if "ObjectProperty" in uri:
        return "ObjectProperty"
    if "Class" in uri or "owl#" in uri:
        return "Class"
    return "_default"


def _sparql_construct(endpoint: str, query: str, timeout: int = 90) -> "rdflib.Graph":
    """POST a CONSTRUCT query to a SPARQL endpoint; return an rdflib.Graph."""
    try:
        import rdflib
        import requests
    except ImportError as exc:
        raise ImportError(
            "rdflib and requests are required for plot_subgraph. "
            "Install with:  pip install rdflib requests"
        ) from exc

    resp = requests.post(
        endpoint,
        data={"query": query},
        headers={"Accept": "text/turtle"},
        timeout=timeout,
    )
    resp.raise_for_status()
    g = rdflib.Graph()
    g.parse(data=resp.text, format="turtle")
    return g


def _rdflib_to_nx(
    g: "rdflib.Graph",
) -> tuple["nx.DiGraph", dict[str, str], dict[str, list[str]]]:
    """
    Convert an rdflib.Graph to a networkx.DiGraph.

    Returns
    -------
    nxg        : the directed graph (only IRI nodes; literals become node attrs)
    label_map  : uri → best human-readable label
    type_map   : uri → list of rdf:type IRIs
    """
    try:
        import networkx as nx
        import rdflib
    except ImportError as exc:
        raise ImportError("networkx and rdflib are required.") from exc

    RDF_TYPE  = str(rdflib.RDF.type)
    LABEL_PREDS = {
        str(rdflib.URIRef("http://purl.org/dc/terms/title")),
        str(rdflib.URIRef("http://xmlns.com/foaf/0.1/name")),
        str(rdflib.URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")),
        str(rdflib.RDFS.label),
    }

    label_map: dict[str, str] = {}
    type_map:  dict[str, list[str]] = {}
    nxg = nx.DiGraph()

    for s, p, o in g:
        s_str = str(s)
        p_str = str(p)

        if isinstance(o, rdflib.Literal):
            # Capture as label if applicable
            if p_str in LABEL_PREDS and s_str not in label_map:
                label_map[s_str] = str(o)
            continue  # don't add literal as a node

        o_str = str(o)
        if p_str == RDF_TYPE:
            type_map.setdefault(s_str, []).append(o_str)
        else:
            nxg.add_edge(s_str, o_str, predicate=_local_name(p_str))

    # Ensure every node that appears only as subject/object is in the graph
    for s, _, o in g:
        if not isinstance(s, rdflib.Literal):
            nxg.add_node(str(s))
        if not isinstance(o, rdflib.Literal):
            nxg.add_node(str(o))

    return nxg, label_map, type_map


# ── pyvis options JSON ────────────────────────────────────────────────────────
_PYVIS_OPTIONS = json.dumps({
    "nodes": {
        "font": {"size": 13, "color": "#ffffff", "face": "Inter, Arial, sans-serif"},
        "borderWidth": 2,
        "shadow": True,
    },
    "edges": {
        "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
        "color": {"color": "#888888", "highlight": "#ffffff"},
        "font": {"size": 10, "color": "#cccccc", "align": "middle"},
        "smooth": {"type": "dynamic"},
        "shadow": False,
    },
    "physics": {
        "enabled": True,
        "forceAtlas2Based": {
            "gravitationalConstant": -60,
            "centralGravity": 0.005,
            "springLength": 120,
            "springConstant": 0.08,
            "damping": 0.6,
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 200},
    },
    "interaction": {
        "hover": True,
        "tooltipDelay": 150,
        "navigationButtons": True,
        "keyboard": True,
        "multiselect": True,
    },
    "configure": {"enabled": False},
})


def plot_subgraph(
    endpoint: str,
    out_html: Path,
    out_png: Path,
    *,
    query: Optional[str] = None,
    n_tracks: int = 6,
    timeout: int = 90,
    png_dpi: int = 300,
    height: str = "780px",
) -> tuple[Path, Path]:
    """Pull a CONSTRUCT subgraph from GraphDB and produce two outputs.

    Parameters
    ----------
    endpoint:
        SPARQL endpoint URL, e.g.
        ``"http://localhost:7200/repositories/music_recsys"``.
    out_html:
        Path for the interactive pyvis HTML file.  Open in any browser —
        drag nodes freely, **double-click** a node to pin/unpin it, use the
        physics panel (top-right ⚙ button) to freeze the whole layout, and
        use browser **File → Save** (or ``Ctrl+S``) for a local copy.
    out_png:
        Path for the high-resolution static PNG (default 300 dpi).
    query:
        Custom SPARQL CONSTRUCT string.  If ``None`` the built-in mixed
        T-Box + A-Box sampler (Option B from the visual graph config docs)
        is used.
    n_tracks:
        Number of example tracks included by the default query.
    timeout:
        HTTP timeout in seconds.
    png_dpi:
        Resolution of the static PNG.
    height:
        Height of the pyvis canvas (CSS string).

    Returns
    -------
    (html_path, png_path)
    """
    try:
        from pyvis.network import Network
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "pyvis and networkx are required for plot_subgraph.\n"
            "Install with:  pip install pyvis networkx"
        ) from exc

    out_html = Path(out_html)
    out_png  = Path(out_png)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch subgraph from GraphDB ────────────────────────────────────────
    sparql = (query or _DEFAULT_CONSTRUCT).format(n_tracks=n_tracks)
    log.info("Fetching CONSTRUCT subgraph from %s …", endpoint)
    rdf_g = _sparql_construct(endpoint, sparql, timeout=timeout)
    log.info("  → %d triples", len(rdf_g))

    # ── 2. Build NetworkX graph ───────────────────────────────────────────────
    nxg, label_map, type_map = _rdflib_to_nx(rdf_g)
    log.info("  → %d nodes, %d edges", nxg.number_of_nodes(), nxg.number_of_edges())

    # ── 3. Interactive HTML with pyvis ────────────────────────────────────────
    net = Network(
        height=height,
        width="100%",
        bgcolor="#1e1e2e",   # dark background
        font_color=True,     # type: ignore[arg-type]  # pyvis stub is wrong; str works fine
        directed=True,
        notebook=False,
    )

    # Build a simple legend via disconnected nodes at fixed positions
    legend_x, legend_y = -1800, -700
    shown_types: set[str] = set()

    for node_uri in nxg.nodes():
        ntype  = _detect_type(node_uri, type_map)
        colour = _TYPE_COLOURS.get(ntype, _TYPE_COLOURS["_default"])
        size   = _TYPE_SIZES.get(ntype, _TYPE_SIZES["_default"])
        label  = _best_label(node_uri, label_map)
        short  = textwrap.shorten(label, width=28, placeholder="…")
        tooltip = (
            f"<b>{label}</b><br/>"
            f"<i>type:</i> {ntype}<br/>"
            f"<i>uri:</i> <small>{node_uri}</small>"
        )
        net.add_node(
            node_uri,
            label=short,
            title=tooltip,
            color={"background": colour, "border": "#ffffff",  # type: ignore[arg-type]
                   "highlight": {"background": "#ffffff", "border": colour}},
            size=size,
            font={"size": 12, "color": "#ffffff"},
            physics=True,
        )
        shown_types.add(ntype)

    for src, dst, data in nxg.edges(data=True):
        pred = data.get("predicate", "")
        net.add_edge(
            src, dst,
            title=pred,
            label=pred if len(pred) <= 20 else "",
            arrows="to",
        )

    # Legend nodes (pinned, no physics)
    for i, ntype in enumerate(sorted(shown_types)):
        leg_id = f"__legend_{ntype}"
        net.add_node(
            leg_id,
            label=ntype,
            color={"background": _TYPE_COLOURS.get(ntype, "#888"),  # type: ignore[arg-type]
                   "border": "#ffffff"},
            size=14,
            x=legend_x,
            y=legend_y + i * 50,
            physics=False,
            fixed={"x": True, "y": True},
            font={"size": 11, "color": "#ffffff"},
            shape="dot",
        )

    net.set_options(_PYVIS_OPTIONS)

    # Inject a tiny toolbar above the canvas for convenience
    _extra_html = textwrap.dedent("""\
        <div style="font-family:sans-serif;background:#2a2a3e;color:#ccc;
                    padding:8px 14px;font-size:12px;border-bottom:1px solid #444">
          <b>Controls:</b>
          drag nodes &nbsp;|&nbsp;
          <b>double-click</b> to pin/unpin &nbsp;|&nbsp;
          scroll to zoom &nbsp;|&nbsp;
          click ⚙ (bottom-left) to toggle physics &nbsp;|&nbsp;
          <b>Ctrl+S</b> to save page
        </div>
    """)

    net.save_graph(str(out_html))

    # Patch the saved HTML to inject the toolbar and a double-click pin handler
    _html = out_html.read_text(encoding="utf-8")
    _pin_js = textwrap.dedent("""\
        <script>
        // Double-click a node to toggle pin/unpin
        network.on("doubleClick", function(params) {
            if (params.nodes.length > 0) {
                var nid = params.nodes[0];
                var pos = network.getPositions([nid])[nid];
                var cur = nodes.get(nid);
                var pinned = cur.fixed && cur.fixed.x;
                nodes.update({
                    id: nid,
                    fixed: {x: !pinned, y: !pinned},
                    color: pinned
                        ? undefined
                        : {background: cur.color ? cur.color.background : "#FFD700",
                           border: "#FFD700"}
                });
            }
        });
        </script>
    """)
    # Insert toolbar after <body> and pin JS just before </body>
    _html = _html.replace("<body>", "<body>\n" + _extra_html, 1)
    _html = _html.replace("</body>", _pin_js + "\n</body>", 1)
    out_html.write_text(_html, encoding="utf-8")
    log.info("Saved interactive graph → %s", out_html)

    # ── 4. High-resolution static PNG with matplotlib ─────────────────────────
    pos = nx.spring_layout(nxg, seed=42, k=2.5 / max(nxg.number_of_nodes() ** 0.5, 1))

    node_list   = list(nxg.nodes())
    node_colors = [_TYPE_COLOURS.get(_detect_type(n, type_map), _TYPE_COLOURS["_default"])
                   for n in node_list]
    node_sizes  = [_TYPE_SIZES.get(_detect_type(n, type_map), 18) * 25
                   for n in node_list]
    node_labels = {n: textwrap.shorten(_best_label(n, label_map), 22, placeholder="…")
                   for n in node_list}

    edge_labels = {(u, v): d.get("predicate", "")
                   for u, v, d in nxg.edges(data=True)
                   if len(d.get("predicate", "")) <= 18}

    fig, ax = plt.subplots(figsize=(18, 13), facecolor="#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    ax.axis("off")

    nx.draw_networkx_nodes(
        nxg, pos, ax=ax,
        nodelist=node_list,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
    )
    nx.draw_networkx_labels(
        nxg, pos, ax=ax,
        labels=node_labels,
        font_size=7,
        font_color="#ffffff",
    )
    nx.draw_networkx_edges(
        nxg, pos, ax=ax,
        edge_color="#666688",
        arrows=True,
        arrowsize=14,
        width=1.2,
        alpha=0.75,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_edge_labels(
        nxg, pos, ax=ax,
        edge_labels=edge_labels,
        font_size=6,
        font_color="#aaaacc",
        bbox={"boxstyle": "round,pad=0.15", "fc": "#2a2a3e", "ec": "none", "alpha": 0.7},
    )

    # Legend patch
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_TYPE_COLOURS.get(t, "#888"), label=t, edgecolor="#ffffff")
        for t in sorted(shown_types)
        if not t.startswith("_")
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=8,
        facecolor="#2a2a3e",
        edgecolor="#555566",
        labelcolor="#ffffff",
        framealpha=0.9,
    )
    ax.set_title(
        "Music Recommendation KG — subgraph overview",
        color="#ccccdd",
        fontsize=13,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=png_dpi, bbox_inches="tight", facecolor="#1e1e2e")
    plt.close(fig)
    log.info("Saved static PNG → %s", out_png)

    return out_html, out_png
