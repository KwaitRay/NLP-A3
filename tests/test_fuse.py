"""Unit tests for retrieval/fuse.py — works without bm25s/torch."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.fuse import rrf_fuse, weighted_fuse  # noqa: E402


def test_weighted_normalised_per_side() -> None:
    bm = [("a", 100.0), ("b", 50.0)]   # min-max → a=1, b=0
    de = [("b", 0.9), ("c", 0.1)]      # min-max → b=1, c=0
    fused = weighted_fuse(bm, de, w_bm25=0.5, w_dense=0.5, top_k=3)
    out = dict(fused)
    # a: 0.5 * 1 + 0.5 * 0 = 0.5
    # b: 0.5 * 0 + 0.5 * 1 = 0.5
    # c: 0.5 * 0 + 0.5 * 0 = 0
    assert abs(out["a"] - 0.5) < 1e-9
    assert abs(out["b"] - 0.5) < 1e-9
    assert abs(out["c"] - 0.0) < 1e-9
    print("  [pass] weighted normalised per side")


def test_weighted_handles_one_sided() -> None:
    fused = weighted_fuse([], [("a", 1.0)], top_k=3)
    assert fused[0][0] == "a"
    print("  [pass] weighted handles empty one side")


def test_rrf_uses_ranks_only() -> None:
    bm = [("a", 999.0), ("b", 0.0001)]    # rank: a=0, b=1
    de = [("b", 0.001), ("a", 0.0001)]    # rank: b=0, a=1
    fused = rrf_fuse(bm, de, k_rrf=60, top_k=3)
    # a's rrf = 1/61 + 1/62 ≈ 0.0325
    # b's rrf = 1/62 + 1/61 ≈ 0.0325 (same)
    # tie → either ordering. Just ensure both present.
    ids = [eid for eid, _ in fused]
    assert set(ids) == {"a", "b"}
    print("  [pass] rrf uses ranks only")


def test_rrf_favours_higher_ranks_consistently() -> None:
    bm = [("a", 999), ("b", 100), ("c", 1)]
    de = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
    fused = rrf_fuse(bm, de, top_k=3)
    assert [eid for eid, _ in fused] == ["a", "b", "c"]
    print("  [pass] rrf consistently ranked")


if __name__ == "__main__":
    print("test_fuse")
    test_weighted_normalised_per_side()
    test_weighted_handles_one_sided()
    test_rrf_uses_ranks_only()
    test_rrf_favours_higher_ranks_consistently()
    print("all green")
