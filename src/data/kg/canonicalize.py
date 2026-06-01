"""Canonical-URI helpers shared by the in-memory and GraphDB exporters.

The KG contains explicit equivalence assertions linking our local terms to
their Wikidata/MO counterparts (``owl:sameAs`` and ``skos:exactMatch``).
Before handing triples to PyKEEN we collapse each equivalence class into
a single canonical URI so the embedding model learns ONE vector per
real-world entity instead of one per duplicate URI.

Public surface
--------------
``pick_canonical(a, b)``
    Deterministic tie-breaker for two URIs already known to be equivalent.
    Preference order (highest first):
        mrc:resource/ > mrc: > wd: > mo: > foaf: > lexicographic min.

``build_canonical_map(equiv_pairs)``
    Path-compressed union-find over an iterable of ``(uri_a, uri_b)``
    pairs.  Returns a fully-materialised ``{uri -> canonical_uri}`` dict
    covering every URI that appeared in at least one equivalence pair.
    URIs that have no equivalences are NOT in the dict; callers should
    use ``canon_map.get(uri, uri)``.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple


# Highest-priority prefixes appear first.  The ordering encodes a deliberate
# bias: our own minted resource URIs beat ontology terms, which beat external
# Wikidata/MO/FOAF references — so canonical winners stay in our namespace
# wherever possible (good for debugging SPARQL output later).
_CANON_PREFERENCE = (
    "purl.org/ontology/mrc/resource/",
    "purl.org/ontology/mrc/",
    "wikidata.org/entity/",
    "purl.org/ontology/mo/",
    "xmlns.com/foaf/",
)


def pick_canonical(a: str, b: str) -> str:
    """Return whichever URI is the preferred canonical representative.

    Falls back to lexicographic ``min`` so the result is deterministic
    even when neither URI matches a known prefix.
    """
    for prefix in _CANON_PREFERENCE:
        if prefix in a and prefix not in b:
            return a
        if prefix in b and prefix not in a:
            return b
    return min(a, b)


def build_canonical_map(equiv_pairs: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    """Build a ``{uri -> canonical_uri}`` map via path-compressed union-find.

    Parameters
    ----------
    equiv_pairs
        Iterable of ``(uri_a, uri_b)`` tuples — each pair asserts that
        the two URIs refer to the same real-world entity.

    Returns
    -------
    Dict[str, str]
        Fully-compressed map.  ``canon_map[uri]`` is the canonical URI of
        ``uri``'s equivalence class.  Trivial (singleton) classes are NOT
        present; use ``canon_map.get(uri, uri)`` at lookup sites.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])     # path compression
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        winner = pick_canonical(ra, rb)
        loser  = rb if winner == ra else ra
        parent[loser] = winner

    for a, b in equiv_pairs:
        if a == b:
            continue
        union(str(a), str(b))

    # Final pass: every node now points (after compression) to its root.
    return {k: find(k) for k in parent}


__all__ = ("pick_canonical", "build_canonical_map")
