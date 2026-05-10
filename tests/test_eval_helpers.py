"""Verify our eval helpers reproduce eval.py output bit-for-bit on the baseline.

This is the single most important sanity check in the project — if the
helper drifts from the official scorer, our ablation tables become wrong.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval_helpers import recall_at_k, score_per_bucket, score_predictions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_matches_official_eval_on_baseline() -> None:
    preds = json.loads((ROOT / "data" / "dev-claims-baseline.json").read_text(encoding="utf-8"))
    gold = json.loads((ROOT / "data" / "dev-claims.json").read_text(encoding="utf-8"))
    m = score_predictions(preds, gold)
    # Numbers from running `python eval.py --predictions data/dev-claims-baseline.json --groundtruth data/dev-claims.json`
    assert abs(m["f_score"] - 0.3377705627705628) < 1e-15
    assert abs(m["accuracy"] - 0.3506493506493507) < 1e-15
    assert abs(m["harmonic_mean"] - 0.3440894901357093) < 1e-15
    assert m["n"] == 154
    print("  [pass] matches eval.py exactly on baseline")


def test_score_per_bucket_partitions_correctly() -> None:
    preds = json.loads((ROOT / "data" / "dev-claims-baseline.json").read_text(encoding="utf-8"))
    gold = json.loads((ROOT / "data" / "dev-claims.json").read_text(encoding="utf-8"))
    buckets = score_per_bucket(preds, gold, lambda cid: gold[cid]["claim_label"])
    total_n = sum(b["n"] for b in buckets.values())
    assert total_n == 154
    assert set(buckets.keys()) == {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"}
    print(f"  [pass] per-bucket partition n=154 across {len(buckets)} labels")


def test_recall_at_k_basic() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a", "c"], k=1) == 0.5
    assert recall_at_k(["x"], ["a"], k=1) == 0.0
    assert recall_at_k([], ["a"], k=5) == 0.0
    assert recall_at_k(["a"], [], k=5) == 0.0
    print("  [pass] recall@k basic cases")


if __name__ == "__main__":
    print("test_eval_helpers")
    test_matches_official_eval_on_baseline()
    test_score_per_bucket_partitions_correctly()
    test_recall_at_k_basic()
    print("all green")
