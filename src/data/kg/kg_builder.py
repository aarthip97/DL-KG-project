"""
Populate the MusicRecSys knowledge graph with instances derived from
the LMD × MSD dataset (+ jSymbolic interim features).

Design goals
------------
* **Never modify the source ontology TTL.**  We always load a *copy*
  (e.g. ``MusicRecSyst_populated.ttl``) and serialize back to it.
* Stay close to the existing ontology: we instantiate ``mo:MusicArtist``,
  ``mo:Performance``, ``mrc:MSDTrack``, ``mrc:Genre``,
  ``mo:Instrument``, plus a controlled vocabulary of
  ``mrc:TempoClass`` individuals (Larghissimo … Prestissimo).
* Every instance gets a **stable** URI derived from a deterministic
  identifier so re-runs are idempotent.

Usage
-----
    from data.kg import KGBuilder
    builder = KGBuilder(
        base_ttl="notebooks/MusicRecSyst.ttl",          # read-only template
        out_ttl="notebooks/MusicRecSyst_populated.ttl", # working copy
    )
    builder.add_tempo_class_individuals()
    builder.populate_from_dataframe(merged_df)
    builder.save()
"""
from __future__ import annotations

import pathlib
import re
import shutil
import urllib.parse
from typing import Iterable, Optional, Sequence

import pandas as pd

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD, FOAF as _FOAF

from .tempo_classes import TEMPO_CLASSES, classify_tempo


# ─────────────────────────────────────────────────────────────────────────────
# Namespaces  (mirror MusicRecSyst.ttl prefixes)
# ─────────────────────────────────────────────────────────────────────────────
MRC   = Namespace("http://purl.org/ontology/mrc/")
MO    = Namespace("http://purl.org/ontology/mo/")
FOAF  = _FOAF
EVENT = Namespace("http://purl.org/NET/c4dm/event.owl#")
DCT   = Namespace("http://purl.org/dc/terms/")

# Local namespace for *individuals* we mint (kept distinct from the
# ontology's own URIs so resources can be told apart at a glance).
EX = Namespace("http://purl.org/ontology/mrc/resource/")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_SLUG_RE = re.compile(r"[^A-Za-z0-9_\-]+")


def _slug(s: str) -> str:
    """URL-safe slug, conservative on punctuation."""
    s = (s or "").strip()
    s = _SLUG_RE.sub("_", s)
    return urllib.parse.quote(s, safe="_-")[:80] or "_"


def _is_missing(v) -> bool:
    """Robust NaN / None / empty-string check (avoids importing numpy here)."""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _to_float(v) -> Optional[float]:
    """Coerce to float, returning None on missing / unparseable values."""
    if _is_missing(v):
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _to_int(v) -> Optional[int]:
    """Coerce to int, returning None on missing / unparseable values."""
    if _is_missing(v):
        return None
    try:
        return int(v)    # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _iter_strings(v) -> list[str]:
    """
    Normalise a cell that *might* contain a sequence of labels into a
    plain ``list[str]``.

    Parquet round-trips list-typed columns as ``numpy.ndarray``; CSV
    parsing may leave them as strings; some upstream code uses tuples.
    This helper accepts list / tuple / ndarray / pd.Series / single
    string and returns a deduplicated, stripped list of non-empty
    strings.  Anything else returns ``[]``.
    """
    if v is None:
        return []
    # Reject scalar NaN (but ndarray/Series would error in pd.isna scalar
    # check, so guard).
    if not hasattr(v, "__iter__") or isinstance(v, str):
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        try:
            if pd.isna(v):
                return []
        except (TypeError, ValueError):
            pass
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in v:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s and s not in seen:
            out.append(s); seen.add(s)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────
class KGBuilder:
    """
    Incrementally populate a copy of the MusicRecSys ontology with
    instances harvested from a DataFrame of (MSD × LMD × jSymbolic)
    rows.

    Parameters
    ----------
    base_ttl : path to the *read-only* ontology TTL (e.g.
        ``notebooks/MusicRecSyst.ttl``).
    out_ttl  : path where the populated graph will be written.
    overwrite_copy : if True (default), the file at ``out_ttl`` is
        regenerated from ``base_ttl`` *before* parsing, ensuring a fresh
        run from the pristine schema.  Set to False to keep extending an
        existing populated graph.
    """

    # Map of jSymbolic feature → rdflib predicate URI used by
    # _add_jsymbolic_features.  Kept here so the user can extend it.
    # All ranges are xsd:double.
    JSYMBOLIC_PROPS: dict[str, URIRef] = {
        "Mean_Tempo":                MRC["meanTempo"],
        "Initial_Tempo":             MRC["initialTempo"],
        "Tempo_Variability":         MRC["tempoVariability"],
        "Note_Density":              MRC["noteDensity"],
        "Average_Note_Duration":     MRC["averageNoteDuration"],
        "Amount_of_Staccato":        MRC["amountOfStaccato"],
        "Amount_of_Arpeggiation":    MRC["amountOfArpeggiation"],
        "Chromatic_Motion":          MRC["chromaticMotion"],
        "Chord_Duration":            MRC["chordDuration"],
        "Average_Number_of_Independent_Voices":
            MRC["averageNumberOfIndependentVoices"],
        "Average_Note_to_Note_Change_in_Dynamics":
            MRC["averageNoteToNoteChangeInDynamics"],
    }
    # All jSymbolic feature predicates above are xsd:double in the ontology.
    _JS_RANGE = XSD.double

    def __init__(
        self,
        base_ttl: str | pathlib.Path,
        out_ttl: str | pathlib.Path,
        overwrite_copy: bool = True,
    ):
        self.base_ttl = pathlib.Path(base_ttl)
        self.out_ttl  = pathlib.Path(out_ttl)
        if not self.base_ttl.exists():
            raise FileNotFoundError(f"Base ontology not found: {self.base_ttl}")

        if overwrite_copy or not self.out_ttl.exists():
            shutil.copyfile(self.base_ttl, self.out_ttl)

        self.g = Graph()
        self.g.parse(self.out_ttl, format="turtle")

        # Make sure all our prefixes are visible in the serialized output.
        self.g.bind("mrc", MRC)
        self.g.bind("mo",  MO)
        self.g.bind("foaf", FOAF)
        self.g.bind("event", EVENT)
        self.g.bind("dcterms", DCT)
        self.g.bind("ex",  EX)

        # Cache to avoid re-asserting rdf:type triples.
        self._known_uris: set[URIRef] = set()

    # ── URI minting ─────────────────────────────────────────────────────────
    @staticmethod
    def artist_uri(artist_id: str) -> URIRef:
        return EX[f"artist/{_slug(artist_id)}"]

    @staticmethod
    def track_uri(track_id: str) -> URIRef:
        return EX[f"track/{_slug(track_id)}"]

    @staticmethod
    def performance_uri(track_id: str) -> URIRef:
        return EX[f"performance/{_slug(track_id)}"]

    @staticmethod
    def genre_uri(label: str) -> URIRef:
        return EX[f"genre/{_slug(label)}"]

    @staticmethod
    def instrument_uri(label: str) -> URIRef:
        return EX[f"instrument/{_slug(label)}"]

    @staticmethod
    def tempo_class_uri(name: str) -> URIRef:
        return MRC[f"TempoClass/{name}"]

    # ── Schema additions: tempo-class controlled vocabulary ─────────────────
    def add_tempo_class_individuals(self) -> None:
        """
        Declare one ``mrc:TempoClass`` (an owl:Class) plus one
        ``skos:Concept``-flavoured individual per Music Theory Academy
        marking, with rdfs:label, rdfs:comment, and BPM bounds.
        """
        TC = MRC["TempoClass"]
        self.g.add((TC, RDF.type, OWL.Class))
        self.g.add((TC, RDFS.label, Literal("Tempo Class", lang="en")))
        self.g.add((TC, RDFS.comment, Literal(
            "Categorical tempo marking (Larghissimo … Prestissimo) "
            "derived from BPM ranges as published by Music Theory Academy.",
            lang="en")))

        for tc in TEMPO_CLASSES:
            uri = self.tempo_class_uri(tc.name)
            self.g.add((uri, RDF.type, TC))
            self.g.add((uri, RDF.type, OWL.NamedIndividual))
            self.g.add((uri, RDFS.label, Literal(tc.name, lang="en")))
            self.g.add((uri, RDFS.comment, Literal(tc.description, lang="en")))
            self.g.add((uri, MRC["minBPM"], Literal(tc.lo, datatype=XSD.double)))
            if tc.hi != float("inf"):
                self.g.add((uri, MRC["maxBPM"],
                            Literal(tc.hi, datatype=XSD.double)))

    # ── Populating from a DataFrame ─────────────────────────────────────────
    def populate_from_dataframe(
        self,
        df: pd.DataFrame,
        max_rows: Optional[int] = None,
        verbose: bool = True,
    ) -> dict[str, int]:
        """
        Walk ``df`` row-by-row and mint instances + assertions.

        Required columns
        ----------------
            track_id, artist_id, artist_name, title

        Optional columns (used when present + non-null)
        ----------------
            song_id, artist_mbid, album_name, year,
            tempo, Mean_Tempo, key, mode,
            time_signature, duration, loudness, danceability, energy,
            primary_genre, top3_genres, artist_terms,
            midi_n_instruments, midi_instrument_names,
            <any of self.JSYMBOLIC_PROPS keys>

        Returns
        -------
        dict — counts of created instances per class, useful for sanity.
        """
        required = {"track_id", "artist_id", "artist_name", "title"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(
                f"populate_from_dataframe: missing required columns {missing}."
            )

        counts = {"artists": 0, "tracks": 0, "performances": 0,
                  "genres": 0, "instruments": 0, "rows_skipped": 0}

        for i, (_, row) in enumerate(df.iterrows()):
            if max_rows is not None and i >= max_rows:
                break
            try:
                self._add_row(row, counts)
            except Exception as e:                            # noqa: BLE001
                counts["rows_skipped"] += 1
                if verbose:
                    print(f"[WARN] row {i} ({row.get('track_id')}): {e}")

        if verbose:
            print("KG population summary:", counts)
        return counts

    # ── per-row machinery ───────────────────────────────────────────────────
    def _add_row(self, row: pd.Series, counts: dict[str, int]) -> None:
        track_id    = str(row["track_id"])
        artist_id   = str(row["artist_id"])
        artist_name = str(row["artist_name"])
        title       = str(row["title"])

        artist = self.artist_uri(artist_id)
        track  = self.track_uri(track_id)
        perf   = self.performance_uri(track_id)

        # ── Artist (mo:MusicArtist + foaf:name) ─────────────────────────────
        if artist not in self._known_uris:
            self.g.add((artist, RDF.type, MO["MusicArtist"]))
            self.g.add((artist, RDF.type, FOAF.Agent))
            self.g.add((artist, FOAF.name, Literal(artist_name)))
            mbid = row.get("artist_mbid")
            if not _is_missing(mbid):
                self.g.add((artist, MO["musicbrainz_guid"],
                            Literal(str(mbid))))
            self._known_uris.add(artist)
            counts["artists"] += 1

        # Genres (one tag per artist; primary_genre is canonical, top3 add color)
        for genre_label in self._collect_genres(row):
            g_uri = self.genre_uri(genre_label)
            if g_uri not in self._known_uris:
                self.g.add((g_uri, RDF.type, MRC["Genre"]))
                self.g.add((g_uri, RDFS.label, Literal(genre_label, lang="en")))
                self._known_uris.add(g_uri)
                counts["genres"] += 1
            self.g.add((artist, MRC["hasGenre"], g_uri))

        # ── Track (mrc:MSDTrack) ────────────────────────────────────────────
        if track not in self._known_uris:
            self.g.add((track, RDF.type, MRC["MSDTrack"]))
            self.g.add((track, DCT.title, Literal(title)))
            self.g.add((track, MO["uuid"], Literal(track_id)))
            sid = row.get("song_id")
            if not _is_missing(sid):
                self.g.add((track, DCT.identifier, Literal(str(sid))))
            release = row.get("album_name")
            if not _is_missing(release):
                self.g.add((track, DCT["isPartOf"], Literal(str(release))))
            year_int = _to_int(row.get("year"))
            if year_int is not None:
                self.g.add((track, DCT.date,
                            Literal(year_int, datatype=XSD.gYear)))
            self._known_uris.add(track)
            counts["tracks"] += 1

        # ── Performance (mo:Performance — links Artist ↔ Track) ─────────────
        if perf not in self._known_uris:
            self.g.add((perf, RDF.type, MO["Performance"]))
            self.g.add((perf, MO["performer"], artist))
            self.g.add((perf, MRC["hasTrack"], track))
            self._known_uris.add(perf)
            counts["performances"] += 1

        # Tempo (MSD scalar) → mo:tempo on the Performance, plus categorical class
        # We prefer Mean_Tempo (jSymbolic) when available, falling back to
        # MSD's `tempo`.
        tempo_val = row.get("Mean_Tempo")
        if _is_missing(tempo_val):
            tempo_val = row.get("tempo")
        tempo_f = _to_float(tempo_val)
        if tempo_f is not None:
            self.g.add((perf, MO["tempo"],
                        Literal(tempo_f, datatype=XSD.double)))
            tc = classify_tempo(tempo_f)
            if tc is not None:
                self.g.add((perf, MRC["hasTempoClass"], self.tempo_class_uri(tc)))

        # Key / mode → mo:key on the Performance (literal label form for now)
        key_name = row.get("key")
        mode_name = row.get("mode")
        if not _is_missing(key_name):
            self.g.add((perf, MO["key"],
                        Literal(f"{key_name} {mode_name or ''}".strip())))

        # Duration / loudness / danceability / energy → datatype properties on Track
        for src, pred, dt in (
            ("duration",     MO["duration"],     XSD.double),
            ("loudness",     MRC["loudness"],    XSD.double),
        ):
            v = _to_float(row.get(src))
            if v is not None:
                self.g.add((track, pred, Literal(v, datatype=dt)))

        # MIDI instrumentation → one mo:Instrument per unique GM name
        for inst_label in _iter_strings(row.get("midi_instrument_names")):
            inst = self.instrument_uri(inst_label)
            if inst not in self._known_uris:
                self.g.add((inst, RDF.type, MO["Instrument"]))
                self.g.add((inst, RDFS.label, Literal(inst_label, lang="en")))
                self._known_uris.add(inst)
                counts["instruments"] += 1
            self.g.add((perf, MO["instrument"], inst))

        # jSymbolic numeric features → datatype properties on the Track
        self._add_jsymbolic_features(track, row)

    # ── ancillary helpers ───────────────────────────────────────────────────
    def _collect_genres(self, row: pd.Series) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        # primary_genre is a single canonical string
        v = row.get("primary_genre")
        if isinstance(v, str) and v.strip() and v not in seen:
            labels.append(v.strip()); seen.add(v.strip())
        # top3_genres / artist_terms are list-like (may be ndarray after parquet)
        for src in ("top3_genres", "artist_terms"):
            for label in _iter_strings(row.get(src)):
                if label not in seen:
                    labels.append(label); seen.add(label)
        return labels

    def _add_jsymbolic_features(self, track: URIRef, row: pd.Series) -> None:
        for col, predicate in self.JSYMBOLIC_PROPS.items():
            if col not in row.index:
                continue
            v = _to_float(row[col])
            if v is None:
                continue
            self.g.add((track, predicate,
                        Literal(v, datatype=self._JS_RANGE)))

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: Optional[str | pathlib.Path] = None) -> pathlib.Path:
        """Serialize the populated graph back to ``out_ttl`` (or override)."""
        target = pathlib.Path(path) if path else self.out_ttl
        target.parent.mkdir(parents=True, exist_ok=True)
        self.g.serialize(destination=str(target), format="turtle")
        return target

    def stats(self) -> dict[str, int]:
        """Return a small dict with #triples + counts by anchor class."""

        def count_instances(cls: URIRef) -> int:
            return sum(1 for _ in self.g.subjects(RDF.type, cls))

        return {
            "triples":       len(self.g),
            "artists":       count_instances(MO["MusicArtist"]),
            "tracks":        count_instances(MRC["MSDTrack"]),
            "performances":  count_instances(MO["Performance"]),
            "genres":        count_instances(MRC["Genre"]),
            "instruments":   count_instances(MO["Instrument"]),
            "tempo_classes": count_instances(MRC["TempoClass"]),
        }


__all__ = ("MRC", "MO", "FOAF", "EVENT", "DCT", "EX", "KGBuilder")
