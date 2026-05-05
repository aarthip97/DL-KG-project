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

import pandas as pd
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from tqdm.auto import tqdm

from .kg_builder import DCT, EVENT, EX, FOAF, KGBuilder, MO, MRC, _slug


# ── URI helpers ────────────────────────────────────────────────────────────
def user_uri(user_id: str) -> URIRef:
    """Stable URI for an Echo Nest taste-profile user."""
    return EX[f"user/{_slug(str(user_id))}"]


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


__all__ = (
    "user_uri",
    "add_listening_schema",
    "add_users_to_graph",
)
