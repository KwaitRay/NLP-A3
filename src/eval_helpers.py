"""Evaluation helpers that mirror ``eval.py`` and add per-bucket slicing.

Two layers:
1. ``score_predictions(preds, gold)`` — exact replica of eval.py's three
   metrics (mean F-score over claims, accuracy, harmonic mean). Used for
   ablation tables and for sanity-checking against the official script.
2. ``score_per_bucket(preds, gold, bucket_lookup)`` — same metric but sliced
   by an arbitrary tag function (claim_id → bucket). Powers the per-domain /
   per-scenario / per-difficulty diagnostic tables.
3. ``recall_at_k(retrieved, gold, k)`` — retrieval-only metric for tuning
   the BM25 / dense / fusion stages without involving the LLM.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Callable, Iterable, Sequence


def _evidence_fscore(pred_evs: Sequence[str], gold_evs: Sequence[str]) -> float:
    """Replicates eval.py:39-52 verbatim.

    F = 2*P*R / (P+R), with P = |pred ∩ gold| / |pred|, R = |pred ∩ gold| / |gold|.
    Returns 0 when no overlap."""
    if not pred_evs or not gold_evs:
        return 0.0
    pred_set = set(pred_evs)
    correct = sum(1 for g in gold_evs if g in pred_set)
    if correct == 0:
        return 0.0
    p = correct / len(pred_evs)  # NB: eval.py uses len(predictions[claim_id]['evidences']),
                                  # which is len(pred_set) only if pred has no dups; the
                                  # official script effectively dedups via set membership.
                                  # We follow eval.py's exact arithmetic here.
    p = correct / len(pred_evs)
    r = correct / len(gold_evs)
    return (2 * p * r) / (p + r)


def score_predictions(
    preds: dict[str, dict], gold: dict[str, dict]
) -> dict[str, float]:
    """Compute (mean F-score, accuracy, harmonic mean) over the gold's claim ids.

    Mirrors ``eval.py:24-76``. Claims missing from ``preds`` count as 0/0 (same
    as eval.py — they're skipped from the per-claim aggregate, dragging the
    arithmetic mean only via missing entries which the official script also
    skips silently).
    """
    f_scores: list[float] = []
    accs: list[float] = []
    for cid, gd in sorted(gold.items()):
        if cid not in preds:
            continue
        pr = preds[cid]
        if "claim_label" not in pr or "evidences" not in pr:
            continue
        accs.append(1.0 if pr["claim_label"] == gd["claim_label"] else 0.0)
        if isinstance(pr["evidences"], list) and len(pr["evidences"]) > 0:
            f_scores.append(_evidence_fscore(pr["evidences"], gd["evidences"]))
        else:
            f_scores.append(0.0)
    mf = mean(f_scores) if f_scores else 0.0
    ma = mean(accs) if accs else 0.0
    hm = (2 * mf * ma) / (mf + ma) if (mf + ma) > 0 else 0.0
    return {"f_score": mf, "accuracy": ma, "harmonic_mean": hm, "n": len(accs)}


def score_per_bucket(
    preds: dict[str, dict],
    gold: dict[str, dict],
    bucket_lookup: Callable[[str], str | None],
) -> dict[str, dict[str, float]]:
    """Slice metrics by an arbitrary bucketing function (claim_id → tag).

    ``bucket_lookup`` returning ``None`` excludes the claim from all buckets.
    Returns dict[bucket → metrics dict].
    """
    grouped: dict[str, list[str]] = defaultdict(list)
    for cid in gold:
        b = bucket_lookup(cid)
        if b is not None:
            grouped[b].append(cid)
    out: dict[str, dict[str, float]] = {}
    for bucket, cids in grouped.items():
        sub_gold = {c: gold[c] for c in cids}
        sub_pred = {c: preds[c] for c in cids if c in preds}
        out[bucket] = score_predictions(sub_pred, sub_gold)
    return out


def recall_at_k(
    retrieved: list[str], gold: Sequence[str], k: int | None = None
) -> float:
    """Fraction of gold evidences present in the top-k retrieved list."""
    if not gold:
        return 0.0
    cap = retrieved if k is None else retrieved[:k]
    cap_set = set(cap)
    hits = sum(1 for g in gold if g in cap_set)
    return hits / len(gold)


def mean_recall_at_k(
    retrieved_per_claim: dict[str, list[str]],
    gold: dict[str, dict],
    k: int,
) -> float:
    """Average recall@k across all claims with gold evidences."""
    rs: list[float] = []
    for cid, gd in gold.items():
        if cid in retrieved_per_claim:
            rs.append(recall_at_k(retrieved_per_claim[cid], gd["evidences"], k))
    return mean(rs) if rs else 0.0


def recall_curve(
    retrieved_per_claim: dict[str, list[str]],
    gold: dict[str, dict],
    ks: Iterable[int] = (1, 3, 5, 10, 20, 50, 100, 200),
) -> dict[int, float]:
    return {k: mean_recall_at_k(retrieved_per_claim, gold, k) for k in ks}
