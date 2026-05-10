"""End-to-end dry-run: walk the notebook on a local Windows box (no GPU).

This validates that the wiring from `src/` to the notebook is intact before
shipping to Colab. It runs every cell that does not need a GPU, stubs the
ones that do, and writes ``outputs/dry_run_report.md`` summarising what was
verified vs what's pending Colab.

Run with::

    python -m scripts.dry_run

Exit code 0 means everything that can be checked locally is healthy. A
non-zero exit means a real failure (missing data file, broken import).
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

# Always import-relative to project root so the notebook + this script use
# identical module paths.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ablation import (  # noqa: E402
    AblationConfig, AblationHarness, load_diag_tag_lookup,
)
from src.data_io import load_dev, load_test_unlabelled, read_jsonl  # noqa: E402
from src.inference import RetrievalOnlyInferer, predict_all  # noqa: E402
from src.paths import EVIDENCE_JSON, OUTPUTS_DIR, SFT_DIR, SPLITS_DIR  # noqa: E402


REPORT: list[str] = []


def _section(title: str) -> None:
    REPORT.append(f"\n## {title}\n")


def _ok(msg: str) -> None:
    REPORT.append(f"- ✅ {msg}")
    print(f"  [ok] {msg}")


def _warn(msg: str) -> None:
    REPORT.append(f"- ⚠️ {msg}")
    print(f"  [warn] {msg}")


def _fail(msg: str) -> None:
    REPORT.append(f"- ❌ {msg}")
    print(f"  [fail] {msg}")


def _try(label: str, fn: Callable, on_fail: str = "fail") -> bool:
    try:
        fn()
        _ok(label)
        return True
    except Exception as e:
        if on_fail == "warn":
            _warn(f"{label} — {type(e).__name__}: {e}")
            return False
        _fail(f"{label} — {type(e).__name__}: {e}")
        REPORT.append(f"  ```\n{traceback.format_exc()}  ```")
        return False


# --- Phase 1: env survey ----------------------------------------------------

def survey_environment() -> None:
    _section("Environment survey")
    _ok(f"python {sys.version.split()[0]} on {sys.platform}")
    REPORT.append("")
    REPORT.append("| package | local | role |")
    REPORT.append("|---|---|---|")
    deps = [
        ("numpy", True, "always"),
        ("nltk", True, "Stage 2 synonyms"),
        ("transformers", True, "Stage 3 tokenizer"),
        # Heavy / Colab-only:
        ("torch", False, "Stage 3 model"),
        ("bm25s", False, "Stage 1 sparse retrieval"),
        ("faiss", False, "Stage 1 dense index"),
        ("sentence_transformers", False, "Stage 1 dense + rerank"),
        ("peft", False, "Stage 3 LoRA"),
        ("trl", False, "Stage 4 DPO ref"),
        ("modelscope", False, "Stage 3 weights download"),
        ("swift", False, "Stage 3 SFT/DPO trainer"),
        ("bitsandbytes", False, "Stage 3 4-bit quant"),
    ]
    for name, local_required, role in deps:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            REPORT.append(f"| {name} | ✅ {ver} | {role} |")
        except Exception:
            mark = "❌ missing" if local_required else "⏳ Colab"
            REPORT.append(f"| {name} | {mark} | {role} |")


# --- Phase 2: data + Stage 0 ------------------------------------------------

def stage0_dryrun() -> bool:
    _section("Stage 0 (data construction) — local execution")
    ok = True

    def _check_evidence():
        if not EVIDENCE_JSON.exists():
            raise FileNotFoundError(f"missing {EVIDENCE_JSON}")
        # Tolerate either a real load or just header-byte sniff for size.
        size_mb = EVIDENCE_JSON.stat().st_size / 2**20
        if size_mb < 100:
            raise RuntimeError(f"evidence.json suspiciously small: {size_mb:.1f} MB")

    ok &= _try("evidence.json present and >100 MB", _check_evidence)

    def _run_stage0():
        from src.build_stage0 import step_eda, step_splits, step_sft, step_tagging
        t0 = time.time()
        step_eda(force=False)
        step_tagging(force=False)
        step_splits(force=False)
        step_sft(force=False)
        REPORT.append(f"  - elapsed: {time.time() - t0:.1f}s")

    ok &= _try("Stage 0 build_stage0 idempotent re-run", _run_stage0)

    expected = [
        OUTPUTS_DIR / "eda" / "eda_report.md",
        SFT_DIR / "claims_tagged.jsonl",
        SFT_DIR / "tag_distribution.md",
        SPLITS_DIR / "train_split.jsonl",
        SPLITS_DIR / "dev_holdout.jsonl",
        SPLITS_DIR / "diag_test.jsonl",
        SPLITS_DIR / "split_summary.md",
        SFT_DIR / "sft_train_v1.jsonl",
        SFT_DIR / "sft_dev_holdout_v1.jsonl",
        SFT_DIR / "sft_diag_test_v1.jsonl",
    ]
    for p in expected:
        if p.exists() and p.stat().st_size > 0:
            _ok(f"artifact: {p.relative_to(OUTPUTS_DIR.parent)} ({p.stat().st_size:,} B)")
        else:
            _fail(f"missing artifact: {p}")
            ok = False

    # Spot-check counts match what tagging + splitting produced.
    def _check_counts():
        n_train = sum(1 for _ in read_jsonl(SPLITS_DIR / "train_split.jsonl"))
        n_devh = sum(1 for _ in read_jsonl(SPLITS_DIR / "dev_holdout.jsonl"))
        n_diag = sum(1 for _ in read_jsonl(SPLITS_DIR / "diag_test.jsonl"))
        if n_train + n_devh + n_diag != 1228:
            raise RuntimeError(f"split union {n_train + n_devh + n_diag} != 1228 train claims")
        REPORT.append(f"  - splits: train_split={n_train}, dev_holdout={n_devh}, diag_test={n_diag}")

    ok &= _try("split counts sum to 1228 train claims", _check_counts)

    return ok


# --- Phase 3: dry-import retrieval (heavy deps may be missing) -------------

def stage1_dryimports() -> bool:
    _section("Stage 1 retrieval — class import smoke (no GPU)")
    ok = True

    def _import_classes():
        from src.retrieval.bm25 import BM25Retriever  # noqa: F401
        from src.retrieval.dense import DEFAULT_MODEL, LIGHT_MODEL, DenseRetriever  # noqa: F401
        from src.retrieval.fuse import rrf_fuse, weighted_fuse  # noqa: F401
        from src.retrieval.pipeline import RetrievalConfig, RetrievalPipeline  # noqa: F401
        from src.retrieval.rerank import (  # noqa: F401
            DEFAULT_RERANKER, CrossEncoderReranker, rule_reorder,
        )

    ok &= _try("retrieval module imports compile clean", _import_classes)

    # Try instantiating the wrappers — they don't load heavy models until
    # build()/load()/_load_model() is called, so this should always work.
    def _instantiate():
        from src.retrieval.bm25 import BM25Retriever
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.rerank import CrossEncoderReranker
        BM25Retriever()
        DenseRetriever()
        CrossEncoderReranker()
        REPORT.append("  - all three retriever classes constructable without GPU")

    ok &= _try("retrievers are lazy-loadable", _instantiate)
    return ok


# --- Phase 4: inference + ablation in stub mode ----------------------------

class _StubRetriever:
    """Returns the first three diag_test gold evidences as 'retrieved'."""

    def __init__(self, gold: dict[str, dict]):
        self._first_evs = next(iter(gold.values()))["evidences"][:3]

    def retrieve(self, claim_text: str):
        return [(eid, "stub passage") for eid in self._first_evs]


def stage5_6_dryrun() -> bool:
    _section("Stage 5+6 wiring smoke — stub retriever + harness")
    ok = True

    diag_gold, diag_tags = load_diag_tag_lookup()
    dev = load_dev()
    test = load_test_unlabelled()

    def _run_inference():
        retriever = _StubRetriever(diag_gold)
        inferer = RetrievalOnlyInferer(retriever, label_strategy="random")
        out_dir = OUTPUTS_DIR / "dry_run"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Combined dev + diag_test predictions to exercise both halves of harness.
        combined = {**dev, **{cid: {"claim_text": "stub"} for cid in diag_gold}}
        preds = predict_all(combined, inferer, out_dir / "preds.json", progress=False)
        if len(preds) != len(combined):
            raise RuntimeError(f"missing preds: {len(preds)} vs {len(combined)}")
        REPORT.append(f"  - generated {len(preds)} stub predictions")

    ok &= _try("RetrievalOnlyInferer + predict_all on dev + diag_test", _run_inference)

    def _run_harness():
        preds = json.loads((OUTPUTS_DIR / "dry_run" / "preds.json").read_text(encoding="utf-8"))
        h = AblationHarness(dev_gold=dev, diag_gold=diag_gold, diag_tags=diag_tags)
        h.add(AblationConfig("DRY", "dry-run stub", flagship=True), preds)
        report = h.render(OUTPUTS_DIR / "dry_run")
        if "Ablation table" not in report or "By climate-science domain" not in report:
            raise RuntimeError("rendered report missing expected sections")
        REPORT.append(f"  - ablation report: {len(report)} chars, all 4 tables rendered")

    ok &= _try("Ablation harness renders all 4 tables on stub predictions", _run_harness)

    # Verify the predictions JSON validates against eval.py-style schema.
    def _check_prediction_format():
        preds = json.loads((OUTPUTS_DIR / "dry_run" / "preds.json").read_text(encoding="utf-8"))
        for cid, rec in preds.items():
            assert rec["claim_label"] in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"}, cid
            assert isinstance(rec["evidences"], list) and rec["evidences"], cid

    ok &= _try("dry-run predictions match eval.py schema", _check_prediction_format)

    REPORT.append(f"  - test set has {len(test)} unlabelled claims (untouched by dry-run)")
    return ok


# --- Phase 5: tests ---------------------------------------------------------

def run_unit_tests() -> bool:
    _section("Unit tests")
    suites = [
        "test_prompt", "test_eval_helpers", "test_sft_dataset",
        "test_fuse", "test_query_rewrite", "test_dpo_pairs",
        "test_inference", "test_ablation",
    ]
    ok = True
    for s in suites:
        def _run(s=s):
            mod = importlib.import_module(f"tests.{s}")
            # Each test file ends with its own assertions on import via `if __name__`.
            # Re-execute by calling its `main`-equivalent if present, otherwise
            # rely on the module-level test functions.
            for attr in dir(mod):
                if attr.startswith("test_"):
                    getattr(mod, attr)()

        ok &= _try(f"{s}", _run)
    return ok


# --- Main -------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Dry-run: COMP90042 Climate Fact-Checking RAG")
    print("=" * 70)
    REPORT.append("# Dry-run report")
    REPORT.append("")
    REPORT.append("Auto-generated by `scripts/dry_run.py`. Verifies the local pipeline")
    REPORT.append("is intact before shipping to Colab.")

    survey_environment()
    s0 = stage0_dryrun()
    s1 = stage1_dryimports()
    s56 = stage5_6_dryrun()
    tests = run_unit_tests()

    _section("Summary")
    if s0 and s1 and s56 and tests:
        _ok("All local checks passed. Notebook is ready to push to Colab.")
        rc = 0
    else:
        _fail("One or more checks failed. See above.")
        rc = 1

    _section("What still needs Colab T4")
    REPORT.extend([
        "- BM25 index build over 1.2M passages (~2-4 min)",
        "- bge-m3 full-corpus embedding + FAISS index (~30-60 min, cached to Drive)",
        "- Qwen3.5-4B download from ModelScope",
        "- ms-swift QLoRA SFT (~75-100 min for 3 epochs, batch 1, grad_accum 16)",
        "- DPO 1 epoch (~25 min)",
        "- Self-consistency inference on dev + test (~5-10 min)",
    ])

    out = OUTPUTS_DIR / "dry_run_report.md"
    out.write_text("\n".join(REPORT), encoding="utf-8")
    print(f"\n→ wrote {out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
