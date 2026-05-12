"""Stage 0 one-shot runner.

Idempotent. Re-running re-uses cached artifacts unless ``--force``. Useful
for both local sanity checks and the notebook (``python -m src.build_stage0``).

Pipeline:
  1. EDA report   →  outputs/eda/eda_report.md
  2. Tagging      →  outputs/sft_data/claims_tagged.jsonl + tag_distribution.md
  3. Hash splits  →  outputs/splits/{train_split,dev_holdout,diag_test,official_dev}.jsonl
  4. SFT data     →  outputs/sft_data/sft_{train,dev_holdout,diag_test}_v2.jsonl
                     (4 needs evidence.json — gracefully skipped if missing)

v2 (2026-05-12, Phase 4): k=20 to match the locked production RetrievalConfig
from Phase 3.5; train split also applies ``weak_buckets`` oversampling on the
buckets that diagnose_phase1 flagged worst at k=20 (nei_underspec n=40 HM=0.039,
disputed_conflict n=21 HM=0.164, refutes_clear n=17 HM=0.116). See
``optimization_plan.md §4.3``. v1 files (k=5, no oversampling) stay on disk for
ablation.

v2 revision (2026-05-12 PM, debug_log 复用经验 32): the first v2 cut had
nei_underspec ×4 + n_hard_neg=1 → 79.1% of training labels were NEI and the
SFT model collapsed to "always predict NEI" (Track 3 NEI acc 0.97, non-NEI acc
0.06, total HM 0.140 < Track 2 baseline 0.201). Rebalanced to:

  - nei_underspec ×4 → ×2 (real NEI samples halved)
  - disputed_conflict ×2 → ×3 (DISPUTED still weak, push harder)
  - refutes_clear ×2 (unchanged)
  - n_hard_neg=1 → 0 (drops 2083 synthetic NEI samples that were the dominant
    source of class imbalance; the 606 real nei_underspec ×2 samples still
    provide the "off-topic ev → NEI" signal hard constraint 1 requires)

Target NEI share: ~40% (vs gold 33%, vs broken v2 first cut 79%). v1 (k=5)
remains on disk for ablation; v2 first cut overwritten in place — diagnose log
preserves the broken numbers as evidence.

v2 revision-2 (2026-05-13, debug_log 复用经验 36, design D-019 副推论):
v3-rebalanced ALSO collapsed (Track 3 HM 0.140, predicted NEI 94.2%). Class
balance was a diagnostic signal, not the root cause. Real root cause is
train-inference representation mismatch:

    Old training:  1-5 gold ev + 15-19 random ev (~96% noise per sample)
    Inference:     ~6.6 gold ev + ~13.4 RAG noise (~67% noise)
    Model shortcut: "noise-heavy input → output NEI"

Phase 3.5b retrieval ceiling audit closed retrieval-side options:
  - mode=retriever: fused-no-rerank recall@20=0.357 (best)
  - mode=llm_rewrite (HyDE+sub-claims): recall@20=0.339 (no help at k=20)
  - HyDE only useful at recall@50/100 (+0.04-0.06), not within SFT context

Fix: pad_with_random=True → False for train. Training samples now contain
only the real retrieved ev (no random padding to k), matching inference
distribution exactly. With pad_with_random=False:
  - Non-NEI samples have 1-5 ev each (the gold ev only)
  - Model learns "given 1-5 mostly-relevant ev, output label"
  - Inference's noisier 20-ev top-k matches none of training perfectly,
    but at least removes the "noise → NEI" shortcut

Track 3 v4 target: HM > 0.213 (current Track 2 v1 baseline). If still
collapses, retrieval ceiling is final and we accept Track 2 as the submitted
system.
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

    # Phase 4 weak-bucket oversampling targets — derived from the k=20
    # diag_test diagnostic (outputs/eval_phase1/diagnose_diag_test_k20.md):
    # NEI underspec is the dominant error mode (HM=0.039), DISPUTED/REFUTES
    # secondary. v2 revision (2026-05-12 PM, debug_log 复用经验 32) cut the
    # NEI oversample factor and disabled hard-neg synthesis after the first
    # cut produced 79% NEI labels → SFT collapsed to NEI-default.
    _TRAIN_WEAK_BUCKETS = {
        ("scenario", "nei_underspec"): 2,       # was 4 — halve real NEI oversample
        ("scenario", "disputed_conflict"): 3,   # was 2 — push DISPUTED harder
        ("scenario", "refutes_clear"): 2,       # unchanged
    }
    expected = {
        # n_hard_neg=0 (was 1): hard-neg synth was the dominant NEI source in
        # the broken first cut; real nei_underspec ×2 still provides ~600
        # "off-topic ev → NEI" examples for hard constraint 1.
        # pad_with_random=False (was True): align training input distribution
        # with inference. v3-rebalanced with pad_with_random=True still
        # collapsed because non-NEI training samples had ~96% noise per
        # sample, very close to NEI training samples (100% off-topic). With
        # pad_with_random=False, non-NEI samples contain only the 1-5 real
        # gold ev — model learns "given relevant ev, output label" instead
        # of "given noise, output NEI". See debug_log 复用经验 36 + D-019.
        "train": (SPLITS_DIR / "train_split.jsonl",
                  SFT_DIR / "sft_train_v2.jsonl",
                  dict(k=20, pad_with_random=False, n_hard_neg=0,
                       apply_curriculum=True, weak_buckets=_TRAIN_WEAK_BUCKETS)),
        "dev_holdout": (SPLITS_DIR / "dev_holdout.jsonl",
                        SFT_DIR / "sft_dev_holdout_v2.jsonl",
                        dict(k=20, pad_with_random=False, n_hard_neg=0, apply_curriculum=False)),
        "diag_test": (SPLITS_DIR / "diag_test.jsonl",
                      SFT_DIR / "sft_diag_test_v2.jsonl",
                      dict(k=20, pad_with_random=False, n_hard_neg=0, apply_curriculum=False)),
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
        _print_label_dist(sft, rows, split_name)
        out[split_name] = out_p
    return out


# -- Sanity check ----------------------------------------------------------

def _print_label_dist(sft_records: list[dict], gold_rows: list[dict], split_name: str) -> None:
    """Print SFT vs gold label distribution; warn on > 2× deviation.

    Class-imbalance is the #1 killer of SFT (debug_log 复用经验 32:
    79% NEI in v2-first-cut collapsed Track 3 to "always predict NEI",
    NEI acc 0.97 / non-NEI acc 0.06 / HM 0.140 < Track 2 baseline 0.201).
    This sanity check runs at build time so we catch the imbalance before
    a 4h SFT run wastes itself.
    """
    from collections import Counter
    sft_labels: Counter[str] = Counter()
    for rec in sft_records:
        msgs = rec.get("messages") or []
        if not msgs or msgs[-1].get("role") != "assistant":
            continue
        content = msgs[-1].get("content", "")
        first_tok = content.split()[0] if content.split() else "?"
        sft_labels[first_tok] += 1
    if not sft_labels:
        return
    gold_labels: Counter[str] = Counter(r.get("claim_label", "?") for r in gold_rows)
    n_sft = sum(sft_labels.values())
    n_gold = sum(gold_labels.values()) or 1

    LABELS_ORDER = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED")
    print(f"       label distribution ({split_name}, vs gold split):")
    print(f"       {'label':<18s} {'sft':>10s} {'gold':>10s}  {'sft/gold ratio':>15s}")
    warnings: list[str] = []
    for lab in LABELS_ORDER:
        n_s = sft_labels.get(lab, 0)
        n_g = gold_labels.get(lab, 0)
        p_s = n_s / n_sft
        p_g = (n_g / n_gold) if n_gold else 0.0
        ratio = (p_s / p_g) if p_g > 0 else float("inf") if p_s > 0 else 1.0
        flag = ""
        if p_g > 0 and (ratio > 2.0 or ratio < 0.5):
            flag = " ⚠"
            warnings.append(f"{lab}: {ratio:.2f}× gold")
        print(f"       {lab:<18s} {n_s:>4d} ({p_s:>5.1%}) "
              f"{n_g:>4d} ({p_g:>5.1%})  {ratio:>10.2f}×{flag}")
    if warnings and split_name == "train":
        print(f"       ⚠ class-imbalance warning ({', '.join(warnings)}): "
              f"see debug_log 复用经验 32.")
        print(f"         Consider reducing weak_buckets factors or n_hard_neg "
              f"if SFT collapses to majority class.")


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
