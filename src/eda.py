"""Exploratory data analysis on the claim files (no evidence corpus needed).

Renders a markdown report at outputs/eda/eda_report.md plus claim-level CSVs.
"""
from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path

from .data_io import load_dev, load_test_unlabelled, load_train
from .paths import EDA_DIR, LABELS


def _claim_token_count(text: str) -> int:
    return len(text.split())


def _summarise(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": int(statistics.median(values)),
        "stdev": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
    }


def _label_dist(claims: dict) -> dict:
    c = Counter(v["claim_label"] for v in claims.values())
    total = sum(c.values()) or 1
    return {lbl: {"count": c.get(lbl, 0), "pct": round(100 * c.get(lbl, 0) / total, 1)} for lbl in LABELS}


def _ev_per_claim(claims: dict) -> list[int]:
    return [len(v.get("evidences", [])) for v in claims.values() if "evidences" in v]


def _ev_per_claim_by_label(claims: dict) -> dict:
    by = {lbl: [] for lbl in LABELS}
    for v in claims.values():
        if "claim_label" in v and "evidences" in v:
            by[v["claim_label"]].append(len(v["evidences"]))
    return {lbl: _summarise(vs) for lbl, vs in by.items()}


def build_report() -> Path:
    train = load_train()
    dev = load_dev()
    test = load_test_unlabelled()

    train_lens = [_claim_token_count(v["claim_text"]) for v in train.values()]
    dev_lens = [_claim_token_count(v["claim_text"]) for v in dev.values()]
    test_lens = [_claim_token_count(v["claim_text"]) for v in test.values()]

    out = EDA_DIR / "eda_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    push = lines.append

    push("# Climate Fact-Check — Claim-side EDA")
    push("")
    push("> No evidence.json needed. Pure claim metadata.")
    push("")

    push("## Sizes")
    push("")
    push(f"- train: **{len(train)}** claims")
    push(f"- dev:   **{len(dev)}** claims")
    push(f"- test:  **{len(test)}** claims (unlabelled)")
    push("")

    push("## Claim length (whitespace-tokenised)")
    push("")
    push("| split | n | min | max | mean | median | stdev |")
    push("|---|---|---|---|---|---|---|")
    for name, vals in [("train", train_lens), ("dev", dev_lens), ("test", test_lens)]:
        s = _summarise(vals)
        push(f"| {name} | {s['n']} | {s['min']} | {s['max']} | {s['mean']} | {s['median']} | {s['stdev']} |")
    push("")

    push("## Label distribution")
    push("")
    for split_name, claims in [("train", train), ("dev", dev)]:
        push(f"### {split_name}")
        push("")
        push("| label | count | % |")
        push("|---|---|---|")
        for lbl, v in _label_dist(claims).items():
            push(f"| {lbl} | {v['count']} | {v['pct']}% |")
        push("")

    push("## Gold evidence count per claim")
    push("")
    for split_name, claims in [("train", train), ("dev", dev)]:
        push(f"### {split_name}")
        push("")
        push("| label | n | min | max | mean | median |")
        push("|---|---|---|---|---|---|")
        per = _ev_per_claim_by_label(claims)
        for lbl in LABELS:
            s = per[lbl]
            if s.get("n", 0) > 0:
                push(f"| {lbl} | {s['n']} | {s['min']} | {s['max']} | {s['mean']} | {s['median']} |")
        push("")

    push("## Sample claims (first 3 per label, train split)")
    push("")
    for lbl in LABELS:
        push(f"### {lbl}")
        push("")
        sampled = 0
        for cid, v in train.items():
            if v["claim_label"] == lbl:
                ev = ", ".join(v["evidences"][:3])
                push(f"- `{cid}` | n_ev={len(v['evidences'])} | ev_sample=[{ev}]")
                push(f"  > {v['claim_text']}")
                sampled += 1
                if sampled == 3:
                    break
        push("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build_report()
    print(f"Wrote {p}")
