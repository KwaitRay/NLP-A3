"""Stage 0 one-shot runner.

Idempotent. Re-running re-uses cached artifacts unless ``--force``. Useful
for both local sanity checks and the notebook (``python -m src.build_stage0``).

Pipeline:
  1. EDA report   →  outputs/eda/eda_report.md
  2. Tagging      →  outputs/sft_data/claims_tagged.jsonl + tag_distribution.md
  3. Hash splits  →  outputs/splits/{train_split,dev_holdout,diag_test,official_dev}.jsonl
  4. SFT data     →  outputs/sft_data/sft_{train,dev_holdout,diag_test}_v1.jsonl
                     (4 needs evidence.json — gracefully skipped if missing)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .data_io import load_evidence, read_jsonl, write_jsonl
from .eda import build_report
from .paths import EVIDENCE_JSON, SFT_DIR, SPLITS_DIR
from .sft_dataset import build_dataset
from .splits import run as run_splits
from .stage0_tag import run as run_tagging


def _exists_and_nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def step_eda(force: bool) -> Path:
    p = build_report() if force else None
    if p is None:
        from src.paths import EDA_DIR
        target = EDA_DIR / "eda_report.md"
        if _exists_and_nonempty(target):
            print(f"[skip] {target} exists")
            return target
        p = build_report()
    print(f"[ok ] eda → {p}")
    return p


def step_tagging(force: bool) -> tuple[Path, Path]:
    target = SFT_DIR / "claims_tagged.jsonl"
    dist = SFT_DIR / "tag_distribution.md"
    if not force and _exists_and_nonempty(target) and _exists_and_nonempty(dist):
        print(f"[skip] {target}, {dist} exist")
        return target, dist
    j, m = run_tagging()
    print(f"[ok ] tagging → {j} | {m}")
    return j, m


def step_splits(force: bool) -> dict[str, Path]:
    expected = ["train_split", "dev_holdout", "diag_test", "official_dev", "summary"]
    if not force and all(_exists_and_nonempty(SPLITS_DIR / f"{n}.jsonl" if n != "summary" else SPLITS_DIR / "split_summary.md")
                          for n in expected):
        print(f"[skip] split files exist in {SPLITS_DIR}")
        return {n: (SPLITS_DIR / "split_summary.md" if n == "summary" else SPLITS_DIR / f"{n}.jsonl") for n in expected}
    out = run_splits()
    for k, v in out.items():
        print(f"[ok ] split {k:14s} → {v}")
    return out


def step_sft(force: bool) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not EVIDENCE_JSON.exists():
        print(f"[warn] {EVIDENCE_JSON} missing — skipping SFT dataset assembly.")
        print("       Download evidence.json (see data/evidence.md) and re-run.")
        return out

    expected = {
        "train": (SPLITS_DIR / "train_split.jsonl",
                  SFT_DIR / "sft_train_v1.jsonl",
                  dict(k=5, pad_with_random=True, n_hard_neg=1, apply_curriculum=True)),
        "dev_holdout": (SPLITS_DIR / "dev_holdout.jsonl",
                        SFT_DIR / "sft_dev_holdout_v1.jsonl",
                        dict(k=5, pad_with_random=True, n_hard_neg=0, apply_curriculum=False)),
        "diag_test": (SPLITS_DIR / "diag_test.jsonl",
                      SFT_DIR / "sft_diag_test_v1.jsonl",
                      dict(k=5, pad_with_random=True, n_hard_neg=0, apply_curriculum=False)),
    }
    if not force and all(_exists_and_nonempty(t) for _, t, _ in expected.values()):
        print("[skip] SFT data files exist")
        return {k: t for k, (_, t, _) in expected.items()}

    print("[..] loading evidence.json (~174 MB)...")
    t0 = time.time()
    ev = load_evidence()
    print(f"     {len(ev):,} passages in {time.time() - t0:.1f}s")

    for split_name, (src_p, out_p, kwargs) in expected.items():
        if not force and _exists_and_nonempty(out_p):
            print(f"[skip] {out_p}")
            out[split_name] = out_p
            continue
        rows = list(read_jsonl(src_p))
        t0 = time.time()
        sft = build_dataset(rows, ev, seed=42, **kwargs)
        write_jsonl(sft, out_p)
        print(f"[ok ] sft {split_name:12s} → {out_p}  ({len(sft)} records, {time.time() - t0:.1f}s)")
        out[split_name] = out_p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 0 end-to-end.")
    ap.add_argument("--force", action="store_true", help="Rebuild all artifacts")
    ap.add_argument("--skip-sft", action="store_true", help="Skip SFT dataset assembly")
    args = ap.parse_args()

    print("=== Stage 0: EDA ===")
    step_eda(args.force)

    print("\n=== Stage 0.3: tagging ===")
    step_tagging(args.force)

    print("\n=== Stage 0.4: hash splits ===")
    step_splits(args.force)

    if not args.skip_sft:
        print("\n=== Stage 0.5: SFT dataset assembly ===")
        step_sft(args.force)

    print("\nStage 0 complete.")


if __name__ == "__main__":
    main()
