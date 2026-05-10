"""Stage 0.4 — Hash-based train/dev_holdout/diag_test split + leakage check.

Why hash split: deterministic across re-runs, salt makes assignment stable
even if the train file is reordered, and the bucket layout 0-7/8/9
gives ~80/10/10 without manual shuffling.

Strict invariants enforced after split:
  - train_split ∩ dev_holdout = ∅
  - train_split ∩ diag_test  = ∅
  - dev_holdout ∩ diag_test  = ∅
  - any of the three ∩ official_dev = ∅   (sanity: official dev claim IDs
    should not collide with train IDs anyway, but verify)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .data_io import load_dev, load_train, read_jsonl, write_jsonl
from .paths import LABELS, SFT_DIR, SPLITS_DIR

SALT = "comp90042-2026-fancy-mapping-lemur"


def hash_bucket(claim_id: str, salt: str = SALT, n: int = 10) -> int:
    """Stable bucket in [0, n). md5(salt || claim_id) % n."""
    h = hashlib.md5(f"{salt}|{claim_id}".encode("utf-8")).hexdigest()
    return int(h, 16) % n


def assign_split(claim_id: str) -> str:
    b = hash_bucket(claim_id)
    if b <= 7:
        return "train_split"
    if b == 8:
        return "dev_holdout"
    return "diag_test"


def _load_tagged() -> list[dict]:
    p = SFT_DIR / "claims_tagged.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run `python -m src.stage0_tag` first."
        )
    return list(read_jsonl(p))


def _label_dist(rows: list[dict]) -> dict[str, int]:
    out = {lbl: 0 for lbl in LABELS}
    for r in rows:
        if r.get("claim_label") in out:
            out[r["claim_label"]] += 1
    return out


def run() -> dict[str, Path]:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    tagged = _load_tagged()
    train_tagged = [r for r in tagged if r.get("split") == "train"]
    # dev_tagged is kept separately for the official-dev leakage check.
    dev_tagged = [r for r in tagged if r.get("split") == "dev"]

    # Bucket the train rows.
    buckets: dict[str, list[dict]] = {
        "train_split": [],
        "dev_holdout": [],
        "diag_test": [],
    }
    for r in train_tagged:
        target = assign_split(r["id"])
        rec = dict(r)
        rec.pop("split", None)
        buckets[target].append(rec)

    # Strict leakage assertions.
    train_ids = {r["id"] for r in buckets["train_split"]}
    devh_ids = {r["id"] for r in buckets["dev_holdout"]}
    diag_ids = {r["id"] for r in buckets["diag_test"]}
    official_dev_ids = set(load_dev().keys())

    assert train_ids.isdisjoint(devh_ids), "leak: train_split ∩ dev_holdout"
    assert train_ids.isdisjoint(diag_ids), "leak: train_split ∩ diag_test"
    assert devh_ids.isdisjoint(diag_ids), "leak: dev_holdout ∩ diag_test"
    assert train_ids.isdisjoint(official_dev_ids), "leak: train_split ∩ official_dev"
    assert devh_ids.isdisjoint(official_dev_ids), "leak: dev_holdout ∩ official_dev"
    assert diag_ids.isdisjoint(official_dev_ids), "leak: diag_test ∩ official_dev"

    paths: dict[str, Path] = {}
    for name, rows in buckets.items():
        p = SPLITS_DIR / f"{name}.jsonl"
        write_jsonl(rows, p)
        paths[name] = p

    # Also copy official dev (with tagging) for downstream consistency.
    p = SPLITS_DIR / "official_dev.jsonl"
    write_jsonl([{k: v for k, v in r.items() if k != "split"} for r in dev_tagged], p)
    paths["official_dev"] = p

    summary = SPLITS_DIR / "split_summary.md"
    lines: list[str] = ["# Stage 0.4 — split summary", ""]
    lines.append("Hash split `md5(salt||claim_id) % 10`: 0-7 → train_split, 8 → dev_holdout, 9 → diag_test.")
    lines.append("")
    lines.append("| split | n | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED |")
    lines.append("|---|---|---|---|---|---|")
    for name in ["train_split", "dev_holdout", "diag_test", "official_dev"]:
        rows = buckets.get(name) or dev_tagged
        d = _label_dist(rows)
        lines.append(
            f"| {name} | {len(rows)} | {d['SUPPORTS']} | {d['REFUTES']} | {d['NOT_ENOUGH_INFO']} | {d['DISPUTED']} |"
        )
    lines.append("")
    lines.append("Leakage assertions: all six pairwise intersections verified empty.")
    summary.write_text("\n".join(lines), encoding="utf-8")
    paths["summary"] = summary

    return paths


if __name__ == "__main__":
    out = run()
    for name, p in out.items():
        print(f"  {name}: {p}")
