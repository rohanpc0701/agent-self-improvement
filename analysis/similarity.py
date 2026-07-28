"""Canonical lexical similarity for Q7 (and any later similarity-gated retrieval).

TF-IDF cosine over lowercased word tokens (>=3 chars), IDF computed over the compared
document pool. This is the same formula used for the cross-stack motivating numbers
(PRBench max-similarity 0.14; terciles +2.7/-3.3) — no similarity method existed in any
repo before this module, so this file IS the definition the prereg pins.

Deliberately dependency-free (stdlib only) and deterministic.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z]{3,}")


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def tfidf_vectors(docs: dict[str, str]) -> dict[str, dict[str, float]]:
    """id -> sparse tf-idf vector, IDF over exactly this pool."""
    toks = {k: tokens(v) for k, v in docs.items()}
    df = Counter(w for d in toks.values() for w in set(d))
    n = len(docs)
    out: dict[str, dict[str, float]] = {}
    for k, d in toks.items():
        tf = Counter(d)
        L = max(1, len(d))
        out[k] = {w: (tf[w] / L) * math.log(n / df[w]) for w in tf if df[w] < n}
    return out


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    num = sum(a[w] * b[w] for w in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def pairwise_max(targets: dict[str, str], sources: dict[str, str]) -> dict[str, float]:
    """For each target id: max cosine to any source. Pool = targets ∪ sources."""
    pool = {**targets, **sources}
    vecs = tfidf_vectors(pool)
    return {
        t: max((cosine(vecs[t], vecs[s]) for s in sources), default=0.0)
        for t in targets
    }
