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
    builder.add_music_concept_hierarchy()   # upper WD hierarchy + domain class links
    builder.add_tempo_class_individuals()   # mrc:Allegro a mrc:TempoClass etc.
    builder.add_key_mode_individuals()      # mrc:KeyC a mrc:Key, mrc:MajorMode a mrc:Mode
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
from tqdm.auto import tqdm

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD, FOAF as _FOAF

from .tempo_classes import TEMPO_CLASSES, classify_tempo


# ConceptScheme URIs — live in their own ``scheme:`` namespace so the
# ontology vocabulary (``mrc:``) cleanly separates from the SKOS schemes.
# Pattern: ``scheme:KeyScheme`` → ``http://purl.org/ontology/mrc/scheme/KeyScheme``.
# IMPORTANT: must stay in sync with the constants in wikidata_mapping.py.
INSTRUMENT_SCHEME_URI    = "http://purl.org/ontology/mrc/scheme/InstrumentScheme"
GENRE_SCHEME_URI         = "http://purl.org/ontology/mrc/scheme/GenreScheme"
DECADE_SCHEME_URI        = "http://purl.org/ontology/mrc/scheme/DecadeScheme"
KEY_SCHEME_URI           = "http://purl.org/ontology/mrc/scheme/KeyScheme"
TEMPO_SCHEME_URI         = "http://purl.org/ontology/mrc/scheme/TempoScheme"
MODE_SCHEME_URI          = "http://purl.org/ontology/mrc/scheme/ModeScheme"
ELEMENTS_SCHEME_URI      = "http://purl.org/ontology/mrc/scheme/ElementsOfMusicScheme"

# ─────────────────────────────────────────────────────────────────────────────
# Legacy sub-path URIs present in the *base* ontology TTL that must be
# removed from the graph at load time so only our flat replacements remain.
# ─────────────────────────────────────────────────────────────────────────────
_LEGACY_KEY_FRAGS = (
    "Key/C", "Key/C_sharp", "Key/D", "Key/D_sharp", "Key/E",
    "Key/F", "Key/F_sharp", "Key/G", "Key/G_sharp",
    "Key/A", "Key/A_sharp", "Key/B",
)
_LEGACY_TEMPO_FRAGS = (
    "TempoClass/Larghissimo", "TempoClass/Grave", "TempoClass/Largo",
    "TempoClass/Adagio", "TempoClass/Andante", "TempoClass/Moderato",
    "TempoClass/Allegro", "TempoClass/Presto", "TempoClass/Prestissimo",
)
# Older (now-superseded) named individual URIs that lived inside ``mrc:``
# itself (e.g. ``mrc:KeyC_sharp``, ``mrc:MajorMode``, ``mrc:Allegro``) before
# the per-entity-type resource namespaces were introduced. They must be
# purged at load time so the KG only contains the new ``key:C_sharp``,
# ``mode:Major``, ``tempo:Allegro`` … forms.
_LEGACY_FLAT_KEY_FRAGS = (
    "KeyC", "KeyC_sharp", "KeyD", "KeyD_sharp", "KeyE",
    "KeyF", "KeyF_sharp", "KeyG", "KeyG_sharp",
    "KeyA", "KeyA_sharp", "KeyB",
)
_LEGACY_FLAT_MODE_FRAGS  = ("MajorMode", "MinorMode")
_LEGACY_FLAT_TEMPO_FRAGS = (
    "Larghissimo", "Grave", "Largo", "Lento", "Adagio", "Adagietto",
    "Andante", "Andantino", "Moderato", "Allegretto", "Allegro",
    "Vivace", "Presto", "Prestissimo",
)
# Every old ConceptScheme URI that has ever existed (sub-path *or* flat
# inside ``mrc:``). Any of these that survive in the graph must be purged
# and, where they appear as ``skos:inScheme`` objects, replaced with the
# canonical ``scheme:<X>`` URI.
_LEGACY_SCHEME_URIS: tuple[tuple[str, str], ...] = (
    # (old URI,                                                                  new flat URI)
    # Original sub-path scheme names from the base ontology TTL
    ("http://purl.org/ontology/mrc/scheme/Keys",         KEY_SCHEME_URI),
    ("http://purl.org/ontology/mrc/scheme/Tempos",       TEMPO_SCHEME_URI),
    ("http://purl.org/ontology/mrc/scheme/Modes",        MODE_SCHEME_URI),
    ("http://purl.org/ontology/mrc/scheme/Genres",       GENRE_SCHEME_URI),
    ("http://purl.org/ontology/mrc/scheme/ElementsOfMusic", ELEMENTS_SCHEME_URI),
    ("http://purl.org/ontology/mrc/scheme/Instruments",  INSTRUMENT_SCHEME_URI),
    # Previous "flat-inside-mrc:" generation
    ("http://purl.org/ontology/mrc/InstrumentScheme",       INSTRUMENT_SCHEME_URI),
    ("http://purl.org/ontology/mrc/GenreScheme",            GENRE_SCHEME_URI),
    ("http://purl.org/ontology/mrc/DecadeScheme",           DECADE_SCHEME_URI),
    ("http://purl.org/ontology/mrc/KeyScheme",              KEY_SCHEME_URI),
    ("http://purl.org/ontology/mrc/TempoScheme",            TEMPO_SCHEME_URI),
    ("http://purl.org/ontology/mrc/ModeScheme",             MODE_SCHEME_URI),
    ("http://purl.org/ontology/mrc/ElementsOfMusicScheme",  ELEMENTS_SCHEME_URI),
)


# ─────────────────────────────────────────────────────────────────────────────
# Namespaces  (mirror MusicRecSyst.ttl prefixes)
# ─────────────────────────────────────────────────────────────────────────────
MRC   = Namespace("http://purl.org/ontology/mrc/")
MO    = Namespace("http://purl.org/ontology/mo/")
FOAF  = _FOAF
EVENT = Namespace("http://purl.org/NET/c4dm/event.owl#")
DCT   = Namespace("http://purl.org/dc/terms/")

# SKOS ConceptScheme namespace — keeps schemes (``scheme:KeyScheme``) out
# of the ``mrc:`` ontology vocabulary so the two stay cleanly separated.
SCHEME = Namespace("http://purl.org/ontology/mrc/scheme/")

# ─────────────────────────────────────────────────────────────────────────────
# Per-entity-type *resource* namespaces
# ─────────────────────────────────────────────────────────────────────────────
# Every individual we mint lives in a namespace dedicated to its type, so
# the URI itself self-documents what the node is — invaluable for
# downstream extraction (just check the prefix, no substring parsing):
#
#   track:TRXXXXX            http://purl.org/ontology/mrc/resource/track/TRXXXXX
#   artist:ARXXXXX           http://purl.org/ontology/mrc/resource/artist/ARXXXXX
#   user:<slug>              http://purl.org/ontology/mrc/resource/user/<slug>
#   genre:rock               http://purl.org/ontology/mrc/resource/genre/rock
#   inst:Violin              http://purl.org/ontology/mrc/resource/instrument/Violin
#   decade:1970s             http://purl.org/ontology/mrc/resource/decade/1970s
#   tempo:Allegro            http://purl.org/ontology/mrc/resource/tempo/Allegro
#   key:C_sharp              http://purl.org/ontology/mrc/resource/key/C_sharp
#   mode:Major               http://purl.org/ontology/mrc/resource/mode/Major
#   perf:TRXXXXX             http://purl.org/ontology/mrc/resource/performance/TRXXXXX
#
# The bare resource namespace is kept (as ``EX``) because external tools
# and downstream code historically import it; new code should prefer the
# typed namespaces.
EX             = Namespace("http://purl.org/ontology/mrc/resource/")
TRACK_NS       = Namespace("http://purl.org/ontology/mrc/resource/track/")
ARTIST_NS      = Namespace("http://purl.org/ontology/mrc/resource/artist/")
USER_NS        = Namespace("http://purl.org/ontology/mrc/resource/user/")
GENRE_NS       = Namespace("http://purl.org/ontology/mrc/resource/genre/")
INSTRUMENT_NS  = Namespace("http://purl.org/ontology/mrc/resource/instrument/")
DECADE_NS      = Namespace("http://purl.org/ontology/mrc/resource/decade/")
TEMPO_NS       = Namespace("http://purl.org/ontology/mrc/resource/tempo/")
KEY_NS         = Namespace("http://purl.org/ontology/mrc/resource/key/")
MODE_NS        = Namespace("http://purl.org/ontology/mrc/resource/mode/")
PERFORMANCE_NS = Namespace("http://purl.org/ontology/mrc/resource/performance/")


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
# Wikidata upper-concept hierarchy
#
# These nodes are minted explicitly by add_music_concept_hierarchy() so that
# all music-domain classes and individuals are connected upward:
#
#   wd:Q115211517  musical concept
#   └── wd:Q11696608  elements of music
#         ├── wd:Q534932   key (music theory)    ← mrc:Key   skos:exactMatch
#         ├── wd:Q34379    musical instrument     ← mo:Instrument skos:exactMatch
#         └── wd:Q188451   music genre            ← mrc:Genre skos:exactMatch
#   wd:Q115211517  musical concept
#         ├── mrc:TempoClass  skos:broader
#         └── mrc:Mode        skos:broader
# ─────────────────────────────────────────────────────────────────────────────
_WD_NS                  = Namespace("http://www.wikidata.org/entity/")
WD_MUSICAL_CONCEPT      = _WD_NS["Q115211517"]   # musical concept
WD_ELEMENTS_OF_MUSIC    = _WD_NS["Q11696608"]    # elements of music
WD_MUSIC_GENRE          = _WD_NS["Q188451"]      # music genre
WD_MUSICAL_INSTRUMENT   = _WD_NS["Q34379"]       # musical instrument
WD_KEY_MUSIC            = _WD_NS["Q534932"]      # key (music theory)


# ─────────────────────────────────────────────────────────────────────────────
# Key / mode → URI maps
#
# Per-entity-type named-individual URIs in the ``key:`` and ``mode:``
# resource namespaces:
#   mrc:Key        — owl:Class declared at runtime by add_key_mode_individuals()
#   key:C, key:C_sharp …  — owl:NamedIndividuals a mrc:Key
#   mrc:Mode       — owl:Class declared at runtime by add_key_mode_individuals()
#   mode:Major     — owl:NamedIndividual a mrc:Mode
#   mode:Minor     — owl:NamedIndividual a mrc:Mode
# ─────────────────────────────────────────────────────────────────────────────
KEY_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Chromatic note labels (enharmonic equivalents for SPARQL / display)
_KEY_LABELS: dict[str, tuple[str, ...]] = {
    "C":  ("C",),
    "C#": ("C♯", "D♭"),
    "D":  ("D",),
    "D#": ("D♯", "E♭"),
    "E":  ("E",),
    "F":  ("F",),
    "F#": ("F♯", "G♭"),
    "G":  ("G",),
    "G#": ("G♯", "A♭"),
    "A":  ("A",),
    "A#": ("A♯", "B♭"),
    "B":  ("B",),
}


def _key_iri_frag(name: str) -> str:
    """Sharps are escaped as ``_sharp`` so the URI is valid without %-encoding.

    e.g. ``"C#"`` → ``"C_sharp"`` → ``key:C_sharp``
    """
    return name.replace("#", "_sharp")


KEY_URI_MAP: dict[str, URIRef] = {
    name: KEY_NS[_key_iri_frag(name)]   # key:C, key:C_sharp …
    for name in KEY_NAMES
}

# Accept the canonical strings produced by ``dataset_extraction``
# (`major`, `minor`) *and* the integer codes (1, 0) of raw MSD.
MODE_URI_MAP: dict[object, URIRef] = {
    "major": MODE_NS["Major"], "Major": MODE_NS["Major"], "MAJOR": MODE_NS["Major"],
    "minor": MODE_NS["Minor"], "Minor": MODE_NS["Minor"], "MINOR": MODE_NS["Minor"],
    1: MODE_NS["Major"],
    0: MODE_NS["Minor"],
}


def _resolve_key_uri(value) -> Optional[URIRef]:
    """Map an MSD key value (``'C'`` … ``'B'`` or 0–11) to a ``key:X`` named-individual URI."""
    if _is_missing(value):
        return None
    # numeric (incl. floats that round to ints, e.g. parquet)
    if isinstance(value, (int,)) or (
        isinstance(value, float) and float(value).is_integer()
    ):
        return KEY_URI_MAP[KEY_NAMES[int(value) % 12]]
    return KEY_URI_MAP.get(str(value).strip())


def _resolve_mode_uri(value) -> Optional[URIRef]:
    """Map an MSD mode value (``'major'`` / ``'minor'`` or 1/0) to a ``mode:Major``/``mode:Minor`` URI."""
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return MODE_URI_MAP.get(value.strip())
    try:
        return MODE_URI_MAP.get(int(value))
    except (TypeError, ValueError):
        return None


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
    # _add_jsymbolic_features. (extendable)
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
        simple: bool = False,
    ):
        self.base_ttl = pathlib.Path(base_ttl)
        self.out_ttl  = pathlib.Path(out_ttl)
        # ``simple=True`` produces a flatter KG variant intended for the
        # baseline DL pipeline:
        #   * no genre-association blank nodes — direct ``artist mrc:hasGenre``
        #     edges only (no per-edge weight);
        #   * no key/mode confidence — key & mode are attached to the *Track*
        #     directly via ``mrc:hasKey`` / ``mrc:hasMode``;
        #   * the listening module skips the ListeningEvent blank node and
        #     emits ``user mrc:listenedTo track`` instead (no listenCount).
        # ``simple=False`` (default) keeps the rich, weighted, per-Performance
        # encoding plus the blank-node listening events with listenCount.
        self.simple = bool(simple)
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
        self.g.bind("skos", SKOS)
        self.g.bind("wd",  Namespace("http://www.wikidata.org/entity/"))
        self.g.bind("wdt", Namespace("http://www.wikidata.org/prop/direct/"))
        # SKOS ConceptScheme namespace — keeps the schemes out of mrc:.
        self.g.bind("scheme", SCHEME)
        # Per-entity-type resource namespaces.  Each individual minted by
        # KGBuilder lives in the namespace dedicated to its type, so the
        # serialised TTL self-documents what every node is.
        self.g.bind("track",  TRACK_NS)
        self.g.bind("artist", ARTIST_NS)
        self.g.bind("user",   USER_NS)
        self.g.bind("genre",  GENRE_NS)
        self.g.bind("inst",   INSTRUMENT_NS)
        self.g.bind("decade", DECADE_NS)
        self.g.bind("tempo",  TEMPO_NS)
        self.g.bind("key",    KEY_NS)
        self.g.bind("mode",   MODE_NS)
        self.g.bind("perf",   PERFORMANCE_NS)

         # Cache to avoid re-asserting rdf:type triples.
        self._known_uris: set[URIRef] = set()
        # Cache to avoid duplicate genre-association blank nodes per (artist, genre).
        self._known_assocs: set[tuple[URIRef, URIRef]] = set()

        # Remove old sub-path Key/TempoClass/scheme nodes inherited from the
        # base TTL so they do not coexist with the flat replacements we add.
        self._purge_legacy_individuals()


    # ── Legacy-individual purge ──────────────────────────────────────────────
    def _purge_legacy_individuals(self) -> None:
        """Remove every triple whose subject is one of the old URIs that
        previous generations of this builder used (sub-path Key/TempoClass
        nodes from the base ontology TTL, and the older ``mrc:KeyC`` /
        ``mrc:MajorMode`` / ``mrc:Allegro`` flat individuals that lived
        directly inside ``mrc:`` before the per-entity resource namespaces
        were introduced).

        Any ``skos:inScheme`` pointer that still references a legacy scheme
        URI is *rewritten* to the canonical ``scheme:<X>`` replacement so
        concepts keep their scheme membership.
        """
        # Old Key and TempoClass named individuals (sub-path) ----------------
        old_nodes: list[URIRef] = [
            MRC[frag] for frag in _LEGACY_KEY_FRAGS + _LEGACY_TEMPO_FRAGS
        ]
        # Older "flat-inside-mrc:" generation of named individuals ----------
        # These lived as ``mrc:KeyC_sharp``, ``mrc:MajorMode``, ``mrc:Allegro``
        # before the move to the typed ``key:`` / ``mode:`` / ``tempo:``
        # resource namespaces.
        old_nodes.extend(MRC[frag] for frag in _LEGACY_FLAT_KEY_FRAGS)
        old_nodes.extend(MRC[frag] for frag in _LEGACY_FLAT_MODE_FRAGS)
        old_nodes.extend(MRC[frag] for frag in _LEGACY_FLAT_TEMPO_FRAGS)
        # Old ConceptScheme nodes — every triple that mentions them
        for old_uri, _new_uri in _LEGACY_SCHEME_URIS:
            old_nodes.append(URIRef(old_uri))

        for node in old_nodes:
            self.g.remove((node, None, None))
            self.g.remove((None, None, node))

        # Rewrite any surviving skos:inScheme triples that still point at a
        # legacy scheme URI so they reference the canonical ``scheme:<X>``.
        for old_uri, new_uri in _LEGACY_SCHEME_URIS:
            old_ref = URIRef(old_uri)
            new_ref = URIRef(new_uri)
            for subj in list(self.g.subjects(SKOS.inScheme, old_ref)):
                self.g.remove((subj, SKOS.inScheme, old_ref))
                self.g.add((subj, SKOS.inScheme, new_ref))


    # ── URI minting ─────────────────────────────────────────────────────────
    @staticmethod
    def artist_uri(artist_id: str) -> URIRef:
        """Per-type URI, e.g. ``artist:ARXXXXX`` →
        ``http://purl.org/ontology/mrc/resource/artist/ARXXXXX``."""
        return ARTIST_NS[_slug(artist_id)]

    @staticmethod
    def track_uri(track_id: str) -> URIRef:
        """Per-type URI, e.g. ``track:TRXXXXX`` →
        ``http://purl.org/ontology/mrc/resource/track/TRXXXXX``."""
        return TRACK_NS[_slug(track_id)]

    @staticmethod
    def performance_uri(track_id: str) -> URIRef:
        """Per-type URI, e.g. ``perf:TRXXXXX`` →
        ``http://purl.org/ontology/mrc/resource/performance/TRXXXXX``."""
        return PERFORMANCE_NS[_slug(track_id)]

    @staticmethod
    def genre_uri(label: str) -> URIRef:
        """Per-type URI, e.g. ``genre:rock`` →
        ``http://purl.org/ontology/mrc/resource/genre/rock``."""
        return GENRE_NS[_slug(label)]

    @staticmethod
    def instrument_uri(label: str) -> URIRef:
        """Per-type URI, e.g. ``inst:Distortion_Guitar`` →
        ``http://purl.org/ontology/mrc/resource/instrument/Distortion_Guitar``."""
        return INSTRUMENT_NS[_slug(label)]

    @staticmethod
    def tempo_class_uri(name: str) -> URIRef:
        """Per-type URI, e.g. ``tempo:Allegro`` →
        ``http://purl.org/ontology/mrc/resource/tempo/Allegro``."""
        return TEMPO_NS[name]

    @staticmethod
    def decade_uri(start_year: int) -> URIRef:
        """Per-type URI, e.g. ``decade:2010s`` →
        ``http://purl.org/ontology/mrc/resource/decade/2010s``."""
        return DECADE_NS[f"{int(start_year)}s"]

    # ── Schema additions: tempo-class controlled vocabulary ─────────────────
    def add_tempo_class_individuals(self) -> None:
        """
        Declare ``mrc:TempoClass`` as an ``owl:Class`` and populate one flat
        ``owl:NamedIndividual`` per Music Theory Academy marking.

        URI pattern: ``mrc:Allegro``, ``mrc:Largo``, … (no sub-path).

        Each individual is typed both as ``mrc:TempoClass`` and
        ``owl:NamedIndividual``, carries ``rdfs:label`` (Protégé / standard)
        and ``skos:prefLabel`` (SKOS interop), and links back to its class
        via ``skos:broader mrc:TempoClass`` so SPARQL can walk the concept
        hierarchy.

        Call ``add_music_concept_hierarchy()`` first to ensure
        ``mrc:TempoClass`` itself is connected to the Wikidata upper layer.
        """
        TC = MRC["TempoClass"]
        self.g.add((TC, RDF.type, OWL.Class))
        self.g.add((TC, RDFS.label,   Literal("Tempo Class", lang="en")))
        self.g.add((TC, SKOS.prefLabel, Literal("Tempo Class", lang="en")))
        self.g.add((TC, RDFS.comment, Literal(
            "Categorical tempo marking (Larghissimo … Prestissimo) "
            "derived from BPM ranges as published by Music Theory Academy.",
            lang="en")))

        # Declare the TempoScheme inside ``scheme:`` (replaces both the
        # legacy ``mrc:scheme/Tempos`` *and* the older ``mrc:TempoScheme``).
        TEMPO_SCH = URIRef(TEMPO_SCHEME_URI)
        self.g.add((TEMPO_SCH, RDF.type, SKOS.ConceptScheme))
        self.g.add((TEMPO_SCH, RDFS.label,    Literal("Tempo Class Scheme", lang="en")))
        self.g.add((TEMPO_SCH, SKOS.prefLabel, Literal("Tempo Class Scheme", lang="en")))

        for tc in TEMPO_CLASSES:
            uri = self.tempo_class_uri(tc.name)
            self.g.add((uri, RDF.type, TC))
            self.g.add((uri, RDF.type, OWL.NamedIndividual))
            self.g.add((uri, RDFS.label,     Literal(tc.name,        lang="en")))
            self.g.add((uri, SKOS.prefLabel, Literal(tc.name,        lang="en")))
            self.g.add((uri, RDFS.comment,   Literal(tc.description, lang="en")))
            self.g.add((uri, MRC["minBPM"],  Literal(tc.lo, datatype=XSD.double)))
            if tc.hi != float("inf"):
                self.g.add((uri, MRC["maxBPM"], Literal(tc.hi, datatype=XSD.double)))
            # SKOS hierarchy
            self.g.add((uri, SKOS.broader,  TC))
            self.g.add((uri, SKOS.inScheme, TEMPO_SCH))

    # ── Schema additions: key / mode controlled vocabularies ────────────────
    def add_key_mode_individuals(self) -> None:
        """
        Declare ``mrc:Key`` and ``mrc:Mode`` as ``owl:Class`` nodes and
        populate one flat ``owl:NamedIndividual`` per chromatic pitch class /
        mode.

        URI pattern:
            key:C, key:C_sharp … key:B
            mode:Major, mode:Minor

        Each individual carries both ``rdfs:label`` (Protégé / community
        standard) and ``skos:prefLabel`` (SKOS interop), plus ``skos:altLabel``
        for enharmonic equivalents (C♯ / D♭).  A ``skos:broader`` edge links
        each individual back to its class-concept so the hierarchy is
        traversable in SPARQL.

        Call ``add_music_concept_hierarchy()`` first to ensure ``mrc:Key``
        and ``mrc:Mode`` are connected to the Wikidata upper layer.
        """
        # ── mrc:Key class + 12 chromatic individuals ────────────────────────
        KEY_CLASS = MRC["Key"]
        self.g.add((KEY_CLASS, RDF.type, OWL.Class))
        self.g.add((KEY_CLASS, RDFS.label,    Literal("Musical Key", lang="en")))
        self.g.add((KEY_CLASS, SKOS.prefLabel, Literal("Musical Key", lang="en")))
        self.g.add((KEY_CLASS, RDFS.comment,  Literal(
            "One of the twelve chromatic pitch classes used as the tonal "
            "centre of a musical piece (C, C♯/D♭, D, … B).", lang="en")))

        # Declare the KeyScheme inside ``scheme:``
        KEY_SCH = URIRef(KEY_SCHEME_URI)
        self.g.add((KEY_SCH, RDF.type, SKOS.ConceptScheme))
        self.g.add((KEY_SCH, RDFS.label,    Literal("Musical Key Scheme", lang="en")))
        self.g.add((KEY_SCH, SKOS.prefLabel, Literal("Musical Key Scheme", lang="en")))

        for name, uri in KEY_URI_MAP.items():
            self.g.add((uri, RDF.type, KEY_CLASS))
            self.g.add((uri, RDF.type, OWL.NamedIndividual))
            self.g.add((uri, RDFS.label,     Literal(name, lang="en")))
            self.g.add((uri, SKOS.prefLabel, Literal(name, lang="en")))
            # Enharmonic / Unicode alt-labels (e.g. "C♯", "D♭")
            for alt in _KEY_LABELS.get(name, ()):
                if alt != name:
                    self.g.add((uri, SKOS.altLabel, Literal(alt, lang="en")))
            # SKOS hierarchy
            self.g.add((uri, SKOS.broader,  KEY_CLASS))
            self.g.add((uri, SKOS.inScheme, KEY_SCH))

        # ── mrc:Mode class + Major / Minor individuals ───────────────────────
        MODE_CLASS = MRC["Mode"]
        self.g.add((MODE_CLASS, RDF.type, OWL.Class))
        self.g.add((MODE_CLASS, RDFS.label,    Literal("Musical Mode", lang="en")))
        self.g.add((MODE_CLASS, SKOS.prefLabel, Literal("Musical Mode", lang="en")))
        self.g.add((MODE_CLASS, RDFS.comment,  Literal(
            "The modality of a musical key: Major or Minor.", lang="en")))

        # Declare the ModeScheme inside ``scheme:``
        MODE_SCH = URIRef(MODE_SCHEME_URI)
        self.g.add((MODE_SCH, RDF.type, SKOS.ConceptScheme))
        self.g.add((MODE_SCH, RDFS.label,    Literal("Musical Mode Scheme", lang="en")))
        self.g.add((MODE_SCH, SKOS.prefLabel, Literal("Musical Mode Scheme", lang="en")))

        for label, uri in (("Major", MODE_NS["Major"]), ("Minor", MODE_NS["Minor"])):
            self.g.add((uri, RDF.type, MODE_CLASS))
            self.g.add((uri, RDF.type, OWL.NamedIndividual))
            self.g.add((uri, RDFS.label,     Literal(label, lang="en")))
            self.g.add((uri, SKOS.prefLabel, Literal(label, lang="en")))
            self.g.add((uri, SKOS.broader,   MODE_CLASS))
            self.g.add((uri, SKOS.inScheme,  MODE_SCH))

    # ── Schema additions: Wikidata upper-concept hierarchy ───────────────────
    def add_music_concept_hierarchy(self) -> None:
        """
        Mint a minimal set of Wikidata upper-concept nodes and link all
        music-domain classes to them.  Call this **once**, before the
        other ``add_*_individuals`` methods.

        Hierarchy added to the graph::

            wd:Q115211517  (musical concept)
            └─ wd:Q11696608  (elements of music)
                  ├─ wd:Q534932   (key)           ← mrc:Key   skos:exactMatch
                  ├─ wd:Q34379    (musical instr.) ← mo:Instrument skos:exactMatch
                  └─ wd:Q188451   (music genre)   ← mrc:Genre skos:exactMatch
            wd:Q115211517  (musical concept)
                  ├─ mrc:TempoClass  skos:broader
                  └─ mrc:Mode        skos:broader

        Object-property choices:
        * ``skos:broader`` — SKOS concept-to-concept hierarchy (traversable
          in SPARQL via property paths);
        * ``skos:exactMatch`` — semantic equivalence between our local class
          and the canonical Wikidata entity;
        * ``owl:subClassOf`` — OWL subsumption, allows Protégé's class
          hierarchy panel to render the domain structure.
        """
        # ── Wikidata upper concept nodes ─────────────────────────────────────
        _wd_nodes = [
            (WD_MUSICAL_CONCEPT,    "musical concept",    None),
            (WD_ELEMENTS_OF_MUSIC,  "elements of music",  WD_MUSICAL_CONCEPT),
            (WD_MUSIC_GENRE,        "music genre",         WD_ELEMENTS_OF_MUSIC),
            (WD_MUSICAL_INSTRUMENT, "musical instrument",  WD_ELEMENTS_OF_MUSIC),
            (WD_KEY_MUSIC,          "key",                 WD_ELEMENTS_OF_MUSIC),
        ]
        for node, label, broader in _wd_nodes:
            self.g.add((node, RDF.type,      SKOS.Concept))
            self.g.add((node, RDFS.label,    Literal(label, lang="en")))
            self.g.add((node, SKOS.prefLabel, Literal(label, lang="en")))
            if broader is not None:
                self.g.add((node, SKOS.broader,    broader))
                self.g.add((node, RDFS.subClassOf, broader))   # Protégé view

        # ── Link our OWL classes to Wikidata via skos:exactMatch / skos:broader ─
        # mrc:Genre — exact match to wd:Q188451 (music genre)
        # The base ontology TTL labels this class "MRC Genre"; overwrite with
        # the canonical Wikidata-aligned lowercase label so hierarchy queries
        # return a consistent single value ("music genre") at every hop.
        self.g.remove((MRC["Genre"], RDFS.label, None))
        self.g.remove((MRC["Genre"], SKOS.prefLabel, None))
        self.g.add((MRC["Genre"], RDFS.label,    Literal("music genre", lang="en")))
        self.g.add((MRC["Genre"], SKOS.prefLabel, Literal("music genre", lang="en")))
        self.g.add((MRC["Genre"],      SKOS.exactMatch, WD_MUSIC_GENRE))
        self.g.add((MRC["Genre"],      SKOS.broader,    WD_ELEMENTS_OF_MUSIC))
        self.g.add((MRC["Genre"],      RDFS.subClassOf, WD_ELEMENTS_OF_MUSIC))

        # mo:Instrument — exact match to wd:Q34379 (musical instrument)
        self.g.add((MO["Instrument"],  SKOS.exactMatch, WD_MUSICAL_INSTRUMENT))
        self.g.add((MO["Instrument"],  SKOS.broader,    WD_ELEMENTS_OF_MUSIC))
        self.g.add((MO["Instrument"],  RDFS.subClassOf, WD_ELEMENTS_OF_MUSIC))

        # mrc:Key — exact match to wd:Q534932 (key in music theory)
        self.g.add((MRC["Key"],        SKOS.exactMatch, WD_KEY_MUSIC))
        self.g.add((MRC["Key"],        SKOS.broader,    WD_ELEMENTS_OF_MUSIC))
        self.g.add((MRC["Key"],        RDFS.subClassOf, WD_ELEMENTS_OF_MUSIC))

        # mrc:TempoClass — broader to musical concept (no single WD exact match
        # for the class; individual markings like Allegro are Q2081524 etc.)
        self.g.add((MRC["TempoClass"], SKOS.broader,    WD_MUSICAL_CONCEPT))
        self.g.add((MRC["TempoClass"], RDFS.subClassOf, WD_MUSICAL_CONCEPT))

        # mrc:Mode — broader to musical concept
        self.g.add((MRC["Mode"],       SKOS.broader,    WD_MUSICAL_CONCEPT))
        self.g.add((MRC["Mode"],       RDFS.subClassOf, WD_MUSICAL_CONCEPT))

        # Declare the flat ElementsOfMusicScheme (replaces mrc:scheme/ElementsOfMusic)
        ELEM_SCH = URIRef(ELEMENTS_SCHEME_URI)
        self.g.add((ELEM_SCH, RDF.type, SKOS.ConceptScheme))
        self.g.add((ELEM_SCH, RDFS.label,    Literal("Elements of Music Scheme", lang="en")))
        self.g.add((ELEM_SCH, SKOS.prefLabel, Literal("Elements of Music Scheme", lang="en")))
        # Link the top concept
        self.g.add((ELEM_SCH, SKOS.hasTopConcept, MRC["ElementsOfMusic"]))

        # Normalise mrc:ElementsOfMusic (ontology class, singular) so it carries
        #TODO SOMETHING IS WRONG, STILL NOT WORKING
        # the same canonical lowercase label as its Wikidata counterpart
        # wd:Q11696608.  This prevents SPARQL results from showing both
        # "Element of Music" and "elements of music" as distinct values.
        ELEM_OF_MUSIC = MRC["ElementsOfMusic"]
        # Remove any existing labels written by the base ontology TTL
        self.g.remove((ELEM_OF_MUSIC, RDFS.label, None))
        self.g.remove((ELEM_OF_MUSIC, SKOS.prefLabel, None))
        # Set the canonical Wikidata-aligned labels
        self.g.add((ELEM_OF_MUSIC, RDFS.label,    Literal("elements of music", lang="en")))
        self.g.add((ELEM_OF_MUSIC, SKOS.prefLabel, Literal("elements of music", lang="en")))
        # Declare equivalence with the WD node so they are semantically unified
        self.g.add((ELEM_OF_MUSIC, SKOS.exactMatch, WD_ELEMENTS_OF_MUSIC))

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

        sub_df = df if max_rows is None else df.iloc[:max_rows]
        progress = tqdm(
            sub_df.iterrows(),
            total=len(sub_df),
            desc="populate",
            unit="row",
            unit_scale=True,
            disable=not verbose,
            leave=True,
        )
        for i, (_, row) in enumerate(progress):
            try:
                self._add_row(row, counts)
            except Exception as e:            
                counts["rows_skipped"] += 1
                if verbose:
                    tqdm.write(f"[WARN] row {i} ({row.get('track_id')}): {e}")
            if verbose and i > 0 and (i & 0x3FF) == 0:
                progress.set_postfix(
                    artists=counts["artists"],
                    tracks=counts["tracks"],
                    perf=counts["performances"],
                    refresh=False,
                )
        progress.close()

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

        # ── Artist (mo:Performer + foaf:name) ─────────────────────────────
        if artist not in self._known_uris:
            self.g.add((artist, RDF.type, MO["Performer"]))
            self.g.add((artist, FOAF.name, Literal(artist_name)))
            mbid = row.get("artist_mbid")
            if not _is_missing(mbid):
                self.g.add((artist, MO["musicbrainz_guid"],
                            Literal(str(mbid))))
            self._known_uris.add(artist)
            counts["artists"] += 1

        # ── Artist genres ────────────────────────────────────────────────────
        # Guard with a per-artist-genre set so we don't mint a new BNode
        # for every track that shares the same artist.
        for genre_label, weight in self._collect_genres(row):
            g_uri = self.genre_uri(genre_label)
            if g_uri not in self._known_uris:
                self.g.add((g_uri, RDF.type, MRC["Genre"]))
                self.g.add((g_uri, RDF.type, OWL.NamedIndividual))
                self.g.add((g_uri, RDF.type, SKOS.Concept))
                self.g.add((g_uri, SKOS.inScheme, URIRef(GENRE_SCHEME_URI)))
                self.g.add((g_uri, SKOS.prefLabel, Literal(genre_label, lang="en")))
                self.g.add((g_uri, RDFS.label, Literal(genre_label, lang="en")))
                self._known_uris.add(g_uri)
                counts["genres"] += 1

            # ──  deduplicate per (artist, genre) pair ──────────
            assoc_key = (artist, g_uri)
            if assoc_key not in self._known_assocs:
                self._known_assocs.add(assoc_key)
                if self.simple or weight is None:
                    self.g.add((artist, MRC["hasGenre"], g_uri))
                else:
                    assoc = BNode()
                    self.g.add((artist, MRC["hasGenreAssoc"], assoc))
                    self.g.add((assoc, RDF.type, MRC["GenreAssociation"]))
                    self.g.add((assoc, MRC["genre"], g_uri))
                    self.g.add((assoc, MRC["weight"],
                                Literal(float(weight), datatype=XSD.double)))
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

        # ── Key / mode → URI references to mrc:Key/<x> and mrc:Mode/Major /
        #    mrc:Mode/Minor (owl:NamedIndividuals declared by
        #    add_key_mode_individuals()).
        #    * rich variant: attach to the *Performance* and carry the MSD
        #      key/mode confidence values as datatype properties on the
        #      Performance (so two Performances of the same Track may
        #      disagree on the interpretation);
        #    * simple variant: attach directly to the *Track* and skip
        #      confidences entirely.
        key_uri  = _resolve_key_uri(row.get("key"))
        mode_uri = _resolve_mode_uri(row.get("mode"))
        anchor   = track if self.simple else perf
        if key_uri is not None:
            self.g.add((anchor, MRC["hasKey"], key_uri))
        if mode_uri is not None:
            self.g.add((anchor, MRC["hasMode"], mode_uri))
        if not self.simple:
            kc = _to_float(row.get("key_confidence"))
            if kc is not None:
                self.g.add((perf, MRC["keyConfidence"],
                            Literal(kc, datatype=XSD.double)))
            mc = _to_float(row.get("mode_confidence"))
            if mc is not None:
                self.g.add((perf, MRC["modeConfidence"],
                            Literal(mc, datatype=XSD.double)))

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
                self.g.add((inst, RDF.type, OWL.NamedIndividual))
                self.g.add((inst, RDF.type, SKOS.Concept))
                self.g.add((inst, SKOS.inScheme, URIRef(INSTRUMENT_SCHEME_URI)))
                self.g.add((inst, SKOS.prefLabel, Literal(inst_label, lang="en")))
                self.g.add((inst, RDFS.label, Literal(inst_label, lang="en")))
                self._known_uris.add(inst)
                counts["instruments"] += 1
            self.g.add((perf, MO["instrument"], inst))

        # jSymbolic numeric features → datatype properties on the Track
        self._add_jsymbolic_features(track, row)

    # ── ancillary helpers ───────────────────────────────────────────────────
    def _collect_genres(
        self, row: pd.Series
    ) -> list[tuple[str, Optional[float]]]:
        """Return ``[(label, weight_or_None), ...]`` deduped by label.

        Weight sources, in priority order:
          1. ``artist_terms_weight`` (MSD-published per-term weight, paired
             positionally with ``artist_terms``);
          2. rank-based fallback for ``primary_genre`` (=1.0) and
             ``top3_genres`` positions 0/1/2 (0.9 / 0.6 / 0.3) when no
             upstream weight is available;
          3. ``None`` otherwise (caller falls back to an unweighted edge).
        """
        out: list[tuple[str, Optional[float]]] = []
        seen: set[str] = set()

        def _push(label: str, weight: Optional[float]) -> None:
            label = label.strip()
            if not label or label in seen:
                return
            seen.add(label)
            out.append((label, weight))

        # 1. primary genre (canonical) — rank 0 fallback weight = 1.0
        v = row.get("primary_genre")
        if isinstance(v, str) and v.strip():
            _push(v, 1.0)

        # 2. top3_genres — rank-based fallback weights
        TOP3_FALLBACK = (0.9, 0.6, 0.3)
        for i, label in enumerate(_iter_strings(row.get("top3_genres"))):
            _push(label, TOP3_FALLBACK[i] if i < len(TOP3_FALLBACK) else 0.2)

        # 3. artist_terms (+ optional artist_terms_weight) — authoritative
        #    weights when present
        terms = _iter_strings(row.get("artist_terms"))
        weights_cell = row.get("artist_terms_weight") if "artist_terms_weight" in row.index else None
        weights: list[Optional[float]] = []
        if weights_cell is not None:
            try:
                weights = [
                    _to_float(w) if w is not None else None
                    for w in weights_cell
                ]
            except TypeError:
                weights = []
        for i, label in enumerate(terms):
            w = weights[i] if i < len(weights) else None
            _push(label, w)
        return out

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
            "artists":       count_instances(MO["Performer"]),      # _add_row uses mo:Performer
            "tracks":        count_instances(MRC["MSDTrack"]),
            "performances":  count_instances(MO["Performance"]),
            "genres":        count_instances(MRC["Genre"]),
            "instruments":   count_instances(MO["Instrument"]),
            "tempo_classes": count_instances(MRC["TempoClass"]),
            "decades":       count_instances(MRC["Decade"]),
            "users":         count_instances(MRC["User"]),          # populated by listening.py
            "skos_concepts": count_instances(SKOS.Concept),
            "skos_schemes":  count_instances(SKOS.ConceptScheme),
        }


__all__ = (
    "MRC", "MO", "FOAF", "EVENT", "DCT",
    "EX", "SCHEME",
    "TRACK_NS", "ARTIST_NS", "USER_NS", "GENRE_NS", "INSTRUMENT_NS",
    "DECADE_NS", "TEMPO_NS", "KEY_NS", "MODE_NS", "PERFORMANCE_NS",
    "INSTRUMENT_SCHEME_URI", "GENRE_SCHEME_URI", "DECADE_SCHEME_URI",
    "KEY_SCHEME_URI", "TEMPO_SCHEME_URI", "MODE_SCHEME_URI", "ELEMENTS_SCHEME_URI",
    "KEY_NAMES", "KEY_URI_MAP", "_KEY_LABELS", "MODE_URI_MAP",
    "WD_MUSICAL_CONCEPT", "WD_ELEMENTS_OF_MUSIC",
    "WD_MUSIC_GENRE", "WD_MUSICAL_INSTRUMENT", "WD_KEY_MUSIC",
    "KGBuilder",
)
