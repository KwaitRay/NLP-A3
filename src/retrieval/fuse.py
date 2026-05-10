"""Fuse BM25 and dense candidates into a single ranked list.

Two strategies:
- Weighted score: ``score = w_bm25 * bm25_norm + w_dense * dense_norm``
  Per Plan §1.3, default weights are 0.3 / 0.7 (climate task needs more lexical
  weight than swxy's 0.05/0.95). Each side is min-max normalised before fusion.
- Reciprocal Rank Fusion (RRF): k-RRF with k=60 (standard).

Either takes ``[(ev_id, score), ...]`` from each retriever; output is the same
shape but combined.
"""
from __future__ import annotations


def _minmax_norm(hits: list[tuple[str, float]]) -> dict[str, float]:
    if not hits:
        return {}
    scores = [s for _, s in hits]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return {ev: 1.0 for ev, _ in hits}
    return {ev: (s - lo) / (hi - lo) for ev, s in hits}


def weighted_fuse(
    bm25_hits: list[tuple[str, float]],
    dense_hits: list[tuple[str, float]],
    *,
    w_bm25: float = 0.3,
    w_dense: float = 0.7,
    top_k: int = 150,
) -> list[tuple[str, float]]:
    bm = _minmax_norm(bm25_hits)
    de = _minmax_norm(dense_hits)
    keys = set(bm) | set(de)
    fused = [
        (k, w_bm25 * bm.get(k, 0.0) + w_dense * de.get(k, 0.0))
        for k in keys
    ]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def rrf_fuse(
    *hit_lists: list[tuple[str, float]],
    k_rrf: int = 60,
    top_k: int = 150,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion. Score-agnostic, uses ranks only."""
    fused: dict[str, float] = {}
    for hits in hit_lists:
        for rank, (ev, _) in enumerate(hits):
            fused[ev] = fused.get(ev, 0.0) + 1.0 / (k_rrf + rank + 1)
    out = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return out[:top_k]
