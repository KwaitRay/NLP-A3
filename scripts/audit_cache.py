"""Repository cache audit — single source of truth for "what's already built".

Mirrors notebook cell [9] but as a CLI so it can run on AutoDL / Colab /
local before kicking off any expensive operation. Used by humans (paste-and-run
sanity check after kernel restart or git pull) and by other scripts (exit
code 0 = all critical artifacts present).

Tiers (criticality for the project, NOT migration cost — for migration see
AUTODL_TO_COLAB.md):
  - critical : without this, no inference is possible
  - important: missing degrades quality (e.g. dense → BM25-only RAG)
  - helpful  : reproducible from earlier steps or auto-builds on first use
  - report   : run history (reports, predictions, ledger)

Run::

    python -m scripts.audit_cache                     # human-readable table
    python -m scripts.audit_cache --tier critical     # only critical rows
    python -m scripts.audit_cache --json              # machine-readable
    python -m scripts.audit_cache --quiet             # exit-code-only mode

Exit codes:
  0 — every critical artifact present (downstream inference can run)
  1 — at least one important/helpful artifact missing (degraded)
  2 — at least one critical artifact missing (inference blocked)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import (  # noqa: E402
    DATA_DIR, EVIDENCE_JSON, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT,
    SFT_DIR, SPLITS_DIR, TRAIN_CLAIMS, TEST_CLAIMS_UNLABELLED,
)


Tier = Literal["critical", "important", "helpful", "report"]


def _cache_root() -> Path:
    """Where AutoDL puts SFT/DPO checkpoints — env-driven, falls back to outputs/.

    AutoDL notebook cell [4] sets CACHE_ROOT=/root/autodl-tmp/nlp_a3_cache.
    Colab + local set CACHE_ROOT=$PROJECT_ROOT/outputs. Honor whatever the
    notebook env established; fall back to outputs/ when unset.
    """
    env = os.environ.get("CACHE_ROOT")
    return Path(env) if env else OUTPUTS_DIR


CACHE_ROOT = _cache_root()


@dataclass
class Entry:
    name: str
    tier: Tier
    path: Path
    rebuild_hint: str
    enables: str  # what downstream is unblocked when this is present

    def exists(self) -> bool:
        return self.path.exists()

    def to_row(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "ok": self.exists(),
            "path": str(self.path),
            "rebuild_hint": self.rebuild_hint,
            "enables": self.enables,
        }


CHECKS: list[Entry] = [
    # -- critical: blocks all inference if missing --------------------------
    Entry("train/dev/test claims", "critical", TRAIN_CLAIMS,
          "shipped with repo", "everything"),
    Entry("test set",              "critical", TEST_CLAIMS_UNLABELLED,
          "shipped with repo", "test inference + leaderboard submission"),
    Entry("evidence.json",         "critical", EVIDENCE_JSON,
          "data/evidence.md (Drive/Canvas link)", "BM25 + dense index build, retrieval pipeline"),
    Entry("Qwen3.5-4B base",       "critical", MODELS_DIR / "Qwen3.5-4B" / "config.json",
          "python -m scripts.download_models", "all LLM inference (Track 1/2/3/4)"),
    Entry("BM25 index",            "critical", OUTPUTS_DIR / "bm25_index" / "ev_ids.txt",
          "notebook cell 2.1 (BM25Retriever.build)", "any RAG (Track 2/3/4)"),

    # -- important: degraded quality if missing -----------------------------
    Entry("bge-m3 encoder",        "important", MODELS_DIR / "bge-m3" / "config.json",
          "python -m scripts.download_models", "dense retrieval + reranker"),
    Entry("FAISS dense index",     "important", OUTPUTS_DIR / "dense_index" / "faiss.index",
          "notebook cell 2.2 (DenseRetriever.build), needs GPU + bge-m3", "BM25+dense fusion (vs BM25-only)"),
    Entry("SFT-merged base",       "important", CACHE_ROOT / "sft-merged" / "config.json",
          "swift export --adapters <ckpt> --merge_lora true --output_dir sft-merged", "Track 3 inference (preferred over LoRA attach, see #31)"),
    Entry("SFT checkpoint",        "important", CACHE_ROOT / "sft-out" / "checkpoint-final",
          "notebook cell 2.5 (ms-swift sft) or scripts.run_sft", "fallback if sft-merged missing; needed to re-export merged"),
    Entry("DPO checkpoint",        "important", CACHE_ROOT / "dpo-out" / "checkpoint-final",
          "notebook cell 2.6 (ms-swift dpo)", "Track 4 (DPO-aligned) — optional"),

    # -- helpful: derivable / auto-builds -----------------------------------
    Entry("Stage-0 splits",        "helpful", SPLITS_DIR / "diag_test.jsonl",
          "notebook cell 1.3 (run_splits) — deterministic from train", "diag_test eval, SFT data"),
    Entry("SFT training data",     "helpful", SFT_DIR / "sft_train_v1.jsonl",
          "notebook cell 1.4 (build_sft_dataset)", "SFT/DPO training (cell 2.5/2.6)"),
    Entry("evidence-id cache",     "helpful", OUTPUTS_DIR / "submissions" / ".evidence_ids.txt",
          "auto-builds on first build_submission run", "fast evidence-id corpus check (~1s vs ~30s)"),
    Entry("bge-reranker-base",     "helpful", MODELS_DIR / "bge-reranker-base" / "config.json",
          "python -m scripts.download_models", "ablation only — DEFAULT OFF (#35: hurts climate recall@5 ×1.68)"),

    # -- report: run history --------------------------------------------
    Entry("Phase 1 eval reports",  "report", OUTPUTS_DIR / "eval_phase1",
          "python -m scripts.phase1_eval --tracks 1,2 --prompts v1", "ablation tables for the report"),
    Entry("submission ledger",     "report", OUTPUTS_DIR / "submissions" / "ledger.jsonl",
          "auto-appends on every build_submission run", "Phase 1/2 quota tracking — must migrate cross-machine"),
]


# -- formatting --------------------------------------------------------------

TIER_GLYPH = {"critical": "!!", "important": " *", "helpful": "  ", "report": "  "}
STATUS_GLYPH = {True: "OK  ", False: "MISS"}


def render_table(rows: list[Entry]) -> str:
    """Human-readable table — fits in an 80-col terminal for the most part."""
    name_w = max(len(r.name) for r in rows)
    enables_w = min(45, max(len(r.enables) for r in rows))
    lines = []
    header = f"{'':2s} {'status':6s} {'tier':10s} {'artifact':{name_w}s}  {'enables':{enables_w}s}  path"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        ok = r.exists()
        enables = r.enables[:enables_w]
        try:
            rel = r.path.relative_to(PROJECT_ROOT)
            path_str = str(rel)
        except ValueError:
            path_str = str(r.path)
        lines.append(
            f"{TIER_GLYPH[r.tier]} {STATUS_GLYPH[ok]:6s} "
            f"{r.tier:10s} {r.name:{name_w}s}  {enables:{enables_w}s}  {path_str}"
        )
    return "\n".join(lines)


def render_summary(rows: list[Entry]) -> str:
    by_tier: dict[Tier, list[Entry]] = {}
    for r in rows:
        by_tier.setdefault(r.tier, []).append(r)

    lines = ["", "Summary:"]
    total_ok = sum(1 for r in rows if r.exists())
    lines.append(f"  cached: {total_ok}/{len(rows)} ({100 * total_ok // len(rows)}%)")
    for tier in ("critical", "important", "helpful", "report"):
        items = by_tier.get(tier, [])
        if not items:
            continue
        miss = [r for r in items if not r.exists()]
        if miss:
            names = ", ".join(r.name for r in miss)
            lines.append(f"  {tier:10s}: {len(miss)}/{len(items)} missing — {names}")
        else:
            lines.append(f"  {tier:10s}: all {len(items)} present")

    # Actionable next steps for the missing rows.
    missing = [r for r in rows if not r.exists()]
    if missing:
        lines.append("")
        lines.append("Next steps (rebuild only what you need):")
        for r in missing:
            lines.append(f"  [{r.tier:9s}] {r.name:24s} → {r.rebuild_hint}")
    return "\n".join(lines)


def compute_exit_code(rows: list[Entry]) -> int:
    """0 = all critical present; 2 = critical missing; 1 = only non-critical missing."""
    critical_missing = any(not r.exists() for r in rows if r.tier == "critical")
    if critical_missing:
        return 2
    any_missing = any(not r.exists() for r in rows)
    return 1 if any_missing else 0


# -- main --------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tier", choices=["critical", "important", "helpful", "report"],
                   default=None, help="filter rows by tier (default: show all)")
    p.add_argument("--json", action="store_true",
                   help="emit JSON (one object per row + a summary block) for scripting")
    p.add_argument("--quiet", action="store_true",
                   help="suppress all output; use exit code only")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero on ANY missing row (default: only critical missing → 2)")
    args = p.parse_args()

    rows = [r for r in CHECKS if args.tier is None or r.tier == args.tier]

    if args.json:
        out = {
            "project_root": str(PROJECT_ROOT),
            "cache_root": str(CACHE_ROOT),
            "rows": [r.to_row() for r in rows],
            "summary": {
                "cached": sum(1 for r in rows if r.exists()),
                "total": len(rows),
                "exit_code": compute_exit_code(CHECKS),  # always full-set for exit code
            },
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif not args.quiet:
        print("=" * 70)
        print(f"cache audit  ::  PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"             ::  CACHE_ROOT   = {CACHE_ROOT}")
        print("=" * 70)
        print(render_table(rows))
        print(render_summary(rows))

    rc = compute_exit_code(CHECKS)
    if args.strict and rc == 0 and any(not r.exists() for r in rows):
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
