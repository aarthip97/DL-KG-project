"""
Add Echo Nest Taste Profile users + listening interactions to the KG.

Modelling
---------
We **reuse** the ontology terms that already exist:

* ``mrc:Listener``   — already declared as ``rdfs:subClassOf foaf:Agent``;
  every user becomes a Listener.
* ``mrc:listenCount`` — already declared with ``xsd:integer`` range; we
  attach it to the blank-node listening interaction.
* ``mo:Performance`` is intentionally **not** reused for user–track
  interactions: a performance in the Music Ontology is the *artist's*
  performance event (range of ``mo:performer``), not a *listener's*
  consumption event. Mixing the two would conflate "X performed track Y"
  with "user X listened to track Y N times".

We mint a tiny amount of extra schema, all in the ``mrc:`` namespace and
all consistent with the existing event-based design:

* ``mrc:ListeningEvent``  — ``owl:Class``, ``rdfs:subClassOf event:Event``.
* ``mrc:hasListeningInteraction`` — ``owl:ObjectProperty`` from a
  ``mrc:Listener`` to a ``mrc:ListeningEvent`` (blank node).
* ``mrc:onTrack`` — ``owl:ObjectProperty`` from ``mrc:ListeningEvent`` to
  ``mrc:Track`` (sub-property of ``event:factor`` so the event-ontology
  semantics are preserved).

Each row of the KG-restricted taste profile is folded as:

    ex:user/<uid>  a mrc:Listener, foaf:Agent ;
                   foaf:name      "<uid>" ;
                   dcterms:identifier "<uid>" ;
                   mrc:hasListeningInteraction [
                       a              mrc:ListeningEvent ;
                       mrc:onTrack    ex:track/<track_id> ;
                       mrc:listenCount "42"^^xsd:integer
                   ] .

The blank node keeps the ``(user, track, count)`` triple grouped without
having to mint a stable URI for every interaction (we have ~hundreds of
thousands).
"""
from __future__ import annotations

from typing import Mapping, Optional
from pathlib import Path
import pandas as pd
from rdflib import BNode, Literal, URIRef, Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD
from tqdm.auto import tqdm

from .kg_builder import DCT, EVENT, FOAF, KGBuilder, MO, MRC, USER_NS, _slug


# ── URI helpers ────────────────────────────────────────────────────────────
def user_uri(user_id: str) -> URIRef:
    """Stable URI for an Echo Nest taste-profile user.

    Uses the per-entity-type ``user:`` resource namespace so the URI is
    self-documenting and trivially distinguishable from tracks / artists
    by the namespace alone (no string parsing required downstream).
    Result: ``user:<slug>`` →
    ``http://purl.org/ontology/mrc/resource/user/<slug>``.
    """
    return USER_NS[_slug(str(user_id))]


# ── Schema additions ──────────────────────────────────────────────────────
def add_listening_schema(builder: KGBuilder) -> None:
    """Declare ``mrc:ListeningEvent`` + ``mrc:hasListeningInteraction``
    + ``mrc:onTrack`` on the graph if they aren't there already.

    Idempotent — safe to call repeatedly.
    """
    g = builder.g

    # mrc:ListeningEvent  ⊑ event:Event
    cls = MRC["ListeningEvent"]
    g.add((cls, RDF.type, OWL.Class))
    g.add((cls, RDFS.subClassOf, EVENT["Event"]))
    g.add((cls, RDFS.label, Literal("Listening Event", lang="en")))
    g.add((cls, RDFS.comment, Literal(
        "A consumption event: a Listener listened to a Track a given "
        "number of times. Materialised as a blank node hanging off the "
        "Listener via mrc:hasListeningInteraction.", lang="en")))

    # mrc:hasListeningInteraction  Listener -> ListeningEvent
    p = MRC["hasListeningInteraction"]
    g.add((p, RDF.type, OWL.ObjectProperty))
    g.add((p, RDFS.domain, MRC["Listener"]))
    g.add((p, RDFS.range,  cls))
    g.add((p, RDFS.label,  Literal("has listening interaction", lang="en")))

    # mrc:onTrack  ListeningEvent -> mrc:Track  (sub-property of event:factor
    # so the event-ontology participation semantics carry over).
    p2 = MRC["onTrack"]
    g.add((p2, RDF.type, OWL.ObjectProperty))
    g.add((p2, RDFS.subPropertyOf, EVENT["factor"]))
    g.add((p2, RDFS.domain, cls))
    g.add((p2, RDFS.range,  MRC["Track"]))
    g.add((p2, RDFS.label,  Literal("on track", lang="en")))


# ── Population ────────────────────────────────────────────────────────────
def add_users_to_graph(
    builder: KGBuilder,
    taste: pd.DataFrame,
    *,
    song_to_track: Optional[Mapping[str, str]] = None,
    merged: Optional[pd.DataFrame] = None,
    user_col: str = "user_id",
    song_col: str = "song_id",
    count_col: str = "play_count",
    simple: Optional[bool] = None,
    batch_size: int = 50_000,
    checkpoint_every: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, int]:
    """Fold the KG-restricted taste profile into the populated graph.

    The DataFrame is consumed in **chunks of ``batch_size`` rows**, each
    iterated via the lightweight ``itertuples`` machinery (no per-row
    ``__dict__`` allocation). A tqdm bar tracks progress at the row level.

    Parameters
    ----------
    builder : the live ``KGBuilder``.
    taste : DataFrame with columns ``user_id``, ``song_id``, ``play_count``.
    song_to_track : optional ``{song_id: track_id}`` mapping. If omitted
        and ``merged`` is given, it's derived from ``merged[['song_id',
        'track_id']]``.
    merged : the KG-input DataFrame (only used if ``song_to_track`` is
        not provided).
    simple : if ``True`` emit *direct* ``user mrc:listenedTo track`` edges
        with no blank node and no listenCount; if ``False`` emit the
        rich blank-node ``mrc:ListeningEvent`` carrying ``mrc:listenCount``.
        Defaults to ``builder.simple`` so it follows the variant chosen at
        builder construction time.
    batch_size : how many taste-profile rows to fold per inner loop. Only
        affects the tqdm cadence and lets us optionally checkpoint to disk
        between chunks; the in-memory rdflib graph is the same either way.
    checkpoint_every : if set (and ``builder.out_ttl`` is configured), call
        ``builder.save()`` every ``checkpoint_every`` *rows*. Useful on
        very large taste profiles so a crash mid-fold doesn't lose the
        previous N hours of work. ``None`` disables.
    verbose : show a tqdm progress bar + final summary line.

    Returns
    -------
    dict with keys ``listeners``, ``interactions``, ``orphan_songs``
    (rows whose ``song_id`` had no track in the KG and were skipped).
    """
    required = {user_col, song_col, count_col}
    missing = required - set(taste.columns)
    if missing:
        raise KeyError(f"taste profile missing columns: {missing}")

    use_simple = bool(getattr(builder, "simple", False)) if simple is None else bool(simple)

    # Schema bits are only needed for the rich blank-node variant.
    if not use_simple:
        add_listening_schema(builder)
    else:
        # Declare the lightweight predicate so consumers can find it via
        # introspection / SPARQL (idempotent; harmless if already present).
        g = builder.g
        p = MRC["listenedTo"]
        g.add((p, RDF.type, OWL.ObjectProperty))
        g.add((p, RDFS.domain, MRC["Listener"]))
        g.add((p, RDFS.range,  MRC["Track"]))
        g.add((p, RDFS.label,  Literal("listened to", lang="en")))

    # Build the song_id -> track_id lookup.
    if song_to_track is None:
        if merged is None:
            raise ValueError(
                "add_users_to_graph: provide either song_to_track or merged."
            )
        s2t = (
            merged[["song_id", "track_id"]]
            .dropna()
            .drop_duplicates(subset="song_id")
        )
        song_to_track = dict(zip(s2t["song_id"].astype(str),
                                 s2t["track_id"].astype(str)))

    g = builder.g
    counts = {"listeners": 0, "interactions": 0, "orphan_songs": 0}
    seen_users: set[URIRef] = set()
    n_rows = len(taste)

    # Project to just the columns we need — avoids dragging unused columns
    # through itertuples and keeps the per-chunk pandas overhead low.
    taste_view = taste[[user_col, song_col, count_col]]

    progress = tqdm(
        total=n_rows,
        desc="listening",
        unit="row",
        unit_scale=True,
        disable=not verbose,
        leave=True,
    )

    last_checkpoint = 0
    out_ttl = getattr(builder, "out_ttl", None)

    try:
        for chunk_start in range(0, n_rows, max(batch_size, 1)):
            chunk = taste_view.iloc[chunk_start : chunk_start + batch_size]
            for uid, sid, plays in chunk.itertuples(index=False, name=None):
                if pd.isna(uid) or pd.isna(sid) or pd.isna(plays):
                    progress.update(1)
                    continue
                track_id = song_to_track.get(str(sid))
                if track_id is None:
                    counts["orphan_songs"] += 1
                    progress.update(1)
                    continue

                u = user_uri(uid)
                if u not in seen_users:
                    g.add((u, RDF.type, MRC["Listener"]))
                    g.add((u, RDF.type, FOAF.Agent))
                    g.add((u, FOAF.name, Literal(str(uid))))
                    g.add((u, DCT.identifier, Literal(str(uid))))
                    seen_users.add(u)
                    counts["listeners"] += 1

                track_uri = builder.track_uri(track_id)
                if use_simple:
                    g.add((u, MRC["listenedTo"], track_uri))
                else:
                    ev = BNode()
                    g.add((u,  MRC["hasListeningInteraction"], ev))
                    g.add((ev, RDF.type,        MRC["ListeningEvent"]))
                    g.add((ev, MRC["onTrack"],  track_uri))
                    g.add((ev, MRC["listenCount"],
                           Literal(int(plays), datatype=XSD.integer)))
                counts["interactions"] += 1
                progress.update(1)

            # Refresh the postfix with cumulative stats every chunk.
            progress.set_postfix(
                listeners=counts["listeners"],
                interactions=counts["interactions"],
                orphan=counts["orphan_songs"],
                refresh=False,
            )

            # Optional crash-safety checkpoint.
            if (
                checkpoint_every
                and out_ttl is not None
                and counts["interactions"] - last_checkpoint >= checkpoint_every
            ):
                builder.save()
                last_checkpoint = counts["interactions"]
                tqdm.write(
                    f"[listening] checkpoint -> {out_ttl} "
                    f"({counts['interactions']:,} interactions so far)"
                )
    finally:
        progress.close()

    if verbose:
        print(f"[listening] folded {counts['interactions']:,} interactions "
              f"for {counts['listeners']:,} listeners "
              f"(orphaned rows: {counts['orphan_songs']:,})")
    return counts


# ── Streaming sidecar variant (RAM-friendly) ──────────────────────────────
# Listening events are write-only: nothing else in the build pipeline queries
# them. So instead of materialising ~11 M extra rdflib triples in RAM, we
# append them straight to a sidecar N-Triples file. Downstream notebooks
# load it back into a Graph with `g.parse(sidecar, format="nt")` (or use
# the union of the base TTL + the sidecar in their own store).
#
# RAM cost drops from O(n_rows) to O(n_unique_users) — only the user-seen
# set stays in memory, and one Python str per ~n_buffer triples in the
# write buffer.

# Constants used by the N-Triples writer.
_RDF_TYPE_NT      = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
_FOAF_AGENT_NT    = "<http://xmlns.com/foaf/0.1/Agent>"
_FOAF_NAME_NT     = "<http://xmlns.com/foaf/0.1/name>"
_DCT_ID_NT        = "<http://purl.org/dc/terms/identifier>"
_XSD_INTEGER_NT   = "<http://www.w3.org/2001/XMLSchema#integer>"

def _esc_nt_literal(s: str) -> str:
    """Escape a Python string for use inside an N-Triples literal."""
    return (
        s.replace("\\", "\\\\")
         .replace("\"", "\\\"")
         .replace("\n", "\\n")
         .replace("\r", "\\r")
         .replace("\t", "\\t")
    )


def stream_users_to_ntriples(
    builder: KGBuilder,
    taste: pd.DataFrame,
    sidecar_path,
    *,
    song_to_track: Optional[Mapping[str, str]] = None,
    merged: Optional[pd.DataFrame] = None,
    user_col: str = "user_id",
    song_col: str = "song_id",
    count_col: str = "play_count",
    simple: Optional[bool] = None,
    batch_size: int = 100_000,
    flush_every: int = 1_000_000,
    verbose: bool = True,
) -> dict[str, int]:
    """Append listening interactions to a **sidecar N-Triples file** instead
    of growing ``builder.g``.

    This is the RAM-friendly version of :func:`add_users_to_graph` — use it
    when the rich (4 triples/row) variant would otherwise OOM. Listening
    events are pure data leaves, so leaving them out of the in-memory graph
    is safe; downstream code that needs them can load the sidecar with
    ``rdflib.Graph().parse(sidecar_path, format="nt")``.

    The schema declarations (``mrc:ListeningEvent`` etc.) are still written
    onto ``builder.g`` so the populated TTL remains self-describing.

    Parameters mirror :func:`add_users_to_graph`. Two extras:

    sidecar_path : path to the ``.nt`` file. Will be **overwritten** at the
        start of the call (one fold = one sidecar).
    flush_every : flush the write buffer to disk every N triples (defaults
        to 1 M; tune down on slow disks).
    """
    sidecar_path = Path(sidecar_path)

    required = {user_col, song_col, count_col}
    missing = required - set(taste.columns)
    if missing:
        raise KeyError(f"taste profile missing columns: {missing}")

    use_simple = bool(getattr(builder, "simple", False)) if simple is None else bool(simple)

    # Always declare the schema on the in-memory graph so consumers can
    # discover the predicates by introspecting the populated TTL alone.
    if not use_simple:
        add_listening_schema(builder)
    else:
        g = builder.g
        p = MRC["listenedTo"]
        g.add((p, RDF.type, OWL.ObjectProperty))
        g.add((p, RDFS.domain, MRC["Listener"]))
        g.add((p, RDFS.range,  MRC["Track"]))
        g.add((p, RDFS.label,  Literal("listened to", lang="en")))

    # song_id -> track_id lookup
    if song_to_track is None:
        if merged is None:
            raise ValueError(
                "stream_users_to_ntriples: provide either song_to_track or merged."
            )
        s2t = (
            merged[["song_id", "track_id"]]
            .dropna()
            .drop_duplicates(subset="song_id")
        )
        song_to_track = dict(zip(s2t["song_id"].astype(str),
                                 s2t["track_id"].astype(str)))

    # Pre-resolve N-Triples constants for the predicates we'll emit.
    LISTENER_NT     = f"<{MRC['Listener']}>"
    LISTENED_TO_NT  = f"<{MRC['listenedTo']}>"
    HAS_INTER_NT    = f"<{MRC['hasListeningInteraction']}>"
    LISTEN_EVENT_NT = f"<{MRC['ListeningEvent']}>"
    ON_TRACK_NT     = f"<{MRC['onTrack']}>"
    LISTEN_COUNT_NT = f"<{MRC['listenCount']}>"

    counts = {"listeners": 0, "interactions": 0, "orphan_songs": 0}
    seen_users: set[str] = set()
    n_rows = len(taste)

    taste_view = taste[[user_col, song_col, count_col]]

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    buf: list[str] = []
    bnode_counter = 0

    progress = tqdm(
        total=n_rows,
        desc="listening→nt",
        unit="row",
        unit_scale=True,
        disable=not verbose,
        leave=True,
    )

    try:
        with open(sidecar_path, "w", encoding="utf-8") as fout:
            for chunk_start in range(0, n_rows, max(batch_size, 1)):
                chunk = taste_view.iloc[chunk_start : chunk_start + batch_size]
                for uid, sid, plays in chunk.itertuples(index=False, name=None):
                    if pd.isna(uid) or pd.isna(sid) or pd.isna(plays):
                        progress.update(1)
                        continue
                    track_id = song_to_track.get(str(sid))
                    if track_id is None:
                        counts["orphan_songs"] += 1
                        progress.update(1)
                        continue

                    user_uri_str = str(user_uri(uid))
                    user_nt      = f"<{user_uri_str}>"
                    track_nt     = f"<{builder.track_uri(track_id)}>"

                    if user_uri_str not in seen_users:
                        uid_lit = _esc_nt_literal(str(uid))
                        buf.append(f"{user_nt} {_RDF_TYPE_NT} {LISTENER_NT} .\n")
                        buf.append(f"{user_nt} {_RDF_TYPE_NT} {_FOAF_AGENT_NT} .\n")
                        buf.append(f'{user_nt} {_FOAF_NAME_NT} "{uid_lit}" .\n')
                        buf.append(f'{user_nt} {_DCT_ID_NT} "{uid_lit}" .\n')
                        seen_users.add(user_uri_str)
                        counts["listeners"] += 1

                    if use_simple:
                        buf.append(f"{user_nt} {LISTENED_TO_NT} {track_nt} .\n")
                    else:
                        bnode_counter += 1
                        ev_nt = f"_:ev{bnode_counter}"
                        buf.append(f"{user_nt} {HAS_INTER_NT} {ev_nt} .\n")
                        buf.append(f"{ev_nt} {_RDF_TYPE_NT} {LISTEN_EVENT_NT} .\n")
                        buf.append(f"{ev_nt} {ON_TRACK_NT} {track_nt} .\n")
                        buf.append(
                            f'{ev_nt} {LISTEN_COUNT_NT} '
                            f'"{int(plays)}"^^{_XSD_INTEGER_NT} .\n'
                        )
                    counts["interactions"] += 1
                    progress.update(1)

                    if len(buf) >= flush_every:
                        fout.write("".join(buf))
                        buf.clear()

                progress.set_postfix(
                    listeners=counts["listeners"],
                    interactions=counts["interactions"],
                    orphan=counts["orphan_songs"],
                    refresh=False,
                )

            # Final flush.
            if buf:
                fout.write("".join(buf))
                buf.clear()
    finally:
        progress.close()

    if verbose:
        size_mb = sidecar_path.stat().st_size / 1024 / 1024
        print(f"[listening→nt] wrote {counts['interactions']:,} interactions "
              f"for {counts['listeners']:,} listeners "
              f"(orphaned rows: {counts['orphan_songs']:,}) "
              f"-> {sidecar_path} ({size_mb:,.1f} MiB)")
    return counts


def ensure_listening_sidecar(
    builder: "KGBuilder",
    taste: pd.DataFrame,
    sidecar_path,
    *,
    merged: Optional[pd.DataFrame] = None,
    song_to_track: Optional[Mapping[str, str]] = None,
    simple: Optional[bool] = None,
    force_rebuild: bool = False,
    batch_size: int = 100_000,
    flush_every: int = 1_000_000,
    verbose: bool = True,
) -> Path:
    """Create the sidecar N-Triples file if it does not yet exist (or
    ``force_rebuild=True``), then return its path.

    This is a convenience wrapper around :func:`stream_users_to_ntriples`
    that implements the standard *skip-if-exists* gate used throughout the
    notebook:

    * **Exists & not forcing** → print a one-liner and return immediately.
    * **Missing or forcing**   → call :func:`stream_users_to_ntriples`,
      then call ``builder.save()`` once to persist the schema additions
      (``mrc:ListeningEvent`` etc.) that the function writes to
      ``builder.g``.

    Parameters are identical to :func:`stream_users_to_ntriples`; the
    only additions are:

    force_rebuild : if True, always rebuild even if the file exists.

    Returns
    -------
    Path — path to the (possibly newly written) sidecar file.
    """
    sidecar_path = Path(sidecar_path)
    if sidecar_path.exists() and not force_rebuild:
        size_mb = sidecar_path.stat().st_size / 1024 / 1024
        if verbose:
            print(f"[SKIP] listening sidecar exists: {sidecar_path.name} "
                  f"({size_mb:,.1f} MiB)")
        return sidecar_path

    stream_users_to_ntriples(
        builder,
        taste,
        sidecar_path=sidecar_path,
        song_to_track=song_to_track,
        merged=merged,
        simple=simple,
        batch_size=batch_size,
        flush_every=flush_every,
        verbose=verbose,
    )
    # Persist the schema triples that stream_users_to_ntriples added to
    # builder.g (mrc:ListeningEvent, mrc:hasListeningInteraction, …).
    builder.save()
    if verbose:
        print(f"[SAVED] schema additions → {builder.out_ttl.name}")
    return sidecar_path


def merge_sidecar_into_graph(
    g: Graph,
    sidecar_path,
    *,
    verbose: bool = True,
) -> int:
    """Parse a sidecar N-Triples file into an existing rdflib graph **only
    if** the sidecar is present.

    Memory-safety note
    ------------------
    This *will* load all sidecar triples into RAM.  Only call it when you
    actually need in-memory SPARQL access to listening data (e.g. for the
    user-centric SPARQL queries or :func:`extract_dl_artifacts`).  After
    the operation is done you can call ``gc.collect()`` to release pressure,
    but the triples remain in ``g`` until it goes out of scope.

    For the RotatE / PyG extraction pipeline prefer the two-file approach:
    pass the ``.ttl`` to ``KGBuilder`` and the sidecar path to
    ``extract_dl_artifacts`` separately (see its ``sidecar_nt`` parameter).

    Parameters
    ----------
    g            : the live rdflib Graph (e.g. ``builder.g``).
    sidecar_path : path to the ``.nt`` file produced by
                   :func:`stream_users_to_ntriples`.
    verbose      : print a summary line.

    Returns
    -------
    int — number of triples added (``len(g)`` delta).
    """
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        if verbose:
            print(f"[WARN] sidecar not found, skipping merge: {sidecar_path}")
        return 0

    n_before = len(g)
    size_mb  = sidecar_path.stat().st_size / 1024 / 1024
    if verbose:
        print(f"Merging sidecar {sidecar_path.name}  ({size_mb:,.1f} MiB) …", end=" ", flush=True)
    g.parse(str(sidecar_path), format="nt")
    added = len(g) - n_before
    if verbose:
        print(f"+{added:,} triples  (total: {len(g):,})")
    return added


__all__ = (
    "user_uri",
    "add_listening_schema",
    "add_users_to_graph",
    "stream_users_to_ntriples",
    "ensure_listening_sidecar",
    "merge_sidecar_into_graph",
)
