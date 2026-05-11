# Implementation Progress Log

> Live status tracker. Update when crossing milestones. Plan in `~/.claude/plans/fancy-mapping-lemur.md`.

## 2026-05-11 — Session 6 (local prep complete, ready for AutoDL Phase 1)

Local prerequisites for Phase 1 evaluation now fully satisfied. Next session is AutoDL.

- **BM25 index built locally** (`outputs/bm25_index/bm25/`):
  `data.csc.index.npy`, `indices.csc.index.npy`, `indptr.csc.index.npy`,
  `params.index.json`, `vocab.index.json` (~200 MB total) +
  `outputs/bm25_index/ev_ids.txt`. Validates retrieval path can dry-run on
  Windows without needing AutoDL.
- **Doc sync**: `TODO.md` rewritten — Step 1 (local BM25) moved to "已完成";
  remaining AutoDL steps renumbered 1→4. Bottom guidance updated to reflect
  the new step numbers.
- **What remains pending (AutoDL only)**: dense index build (bge-m3 on 4080
  SUPER, ~15 min), Phase 1 baseline eval on `diag_test` (v1 prompt, ~10
  min), Phase 2 prompt sweep (v2/v3/v4, ~15 min). See `TODO.md` Steps 1-3.

## 2026-05-10/11 — Session 5 (AutoDL boot + Phase 1 scaffolding + bilingual plan)

Catches up the period between Session 4 and today; pushed across commits
`9465f9b` → `003122a` to `origin/main`.

- **AutoDL instance up**: PyTorch 2.5.1+cu124, RTX 4080 SUPER 31.5 GB VRAM,
  bf16 + flash-attn 2.x both supported. Smoke test
  (`scripts/test_qwen35_inference.py`) passes end-to-end with Qwen3.5-4B.
- **Phase 1 scaffolding** (all green-tested):
  - `src/prompt.py` — added `PROMPT_VARIANTS` dict with v1 (current baseline)
    through v4 (each layering one more constraint), all consumed by the new
    eval harness via `--prompts vN[,vM,...]`.
  - `scripts/build_indexes.py` — standalone BM25 + dense index builder,
    `--skip-dense` runs BM25-only (used locally today).
  - `scripts/phase1_eval.py` — Track 1 (no-RAG) / Track 2 (RAG) × prompt
    variant sweep harness. Writes `outputs/eval_phase1/track{1,2}_v{1..4}_
    {dataset}.{json,md}` plus `summary_{dataset}.md`. Per-bucket tables in
    Track 2 are sorted by HM ascending so weakest buckets surface for
    Phase 4 targeting.
  - `scripts/download_models.py` — one-shot fetch of Qwen3.5-4B + bge-m3 +
    bge-reranker-base + bge-small-en-v1.5 into `models/` (~11 GB).
- **Persistence refactor**: notebook `cell-1-sft-code` switched to
  cache-first; all model paths now flow through `MODELS_DIR` +
  `resolve_model_path()` so `models/` is authoritative on both local and
  AutoDL.
- **SFT / DPO data migration**: train + dev_holdout + diag_test rewritten
  into ms-swift `messages` standard format; all 8 unit-test suites green.
- **Documentation**:
  - `design.md` bumped to v1.1 (records D-011 through D-015, where D-015
    formalises the eval-driven SFT-data-design loop).
  - `optimization_plan.md` — new 6-phase bilingual (中文 + English) plan,
    executable counterpart to D-015.
  - `debug_log.md` Session 2 — Qwen3.5 / AutoDL pitfalls captured (mixed-
    thinking VL handling, `enable_thinking=False` + thinking-trio, T4 vs
    4080 dtype gating, transformers 5.x `apply_chat_template` returning
    `BatchEncoding` not tensor).
  - `TODO.md` — bilingual single-page recovery doc for tomorrow-self.
- **Models on disk**: `models/{Qwen3.5-4B,bge-m3,bge-reranker-base,bge-
  small-en-v1.5}/` — 4 directories, ~11 GB combined.

## 2026-04-30 — Session 4 (notebook annotated + design.md)

- **Notebook section status badges**: every sub-section header in
  `notebooks/notebook.ipynb` carries one of three markers (✅ verified
  locally / 🧪 stub-validated / ⏳ requires Colab). 15 headers tagged.
  Marker added to README cell explaining the legend; `outputs/dry_run_report.md`
  is the audit trail.
- **`design.md`** at project root — 809 lines, 18 sections covering data
  model, all 7 stages, code organisation, notebook layout, reproducibility,
  37-test matrix, risks, decision records, glossary. Chinese-primary,
  technical identifiers preserved verbatim. Cross-references plan,
  PROGRESS.md, dry_run_report.md.

## 2026-04-30 — Session 3 (dry-run wired)

`scripts/dry_run.py` validates the entire local pipeline in one command (~3 s):
env survey → Stage 0 idempotent re-run → artifact existence checks → Stage 1
class smoke imports → Stage 5+6 stub run with 275 synthetic predictions → all 8
unit-test suites. Writes `outputs/dry_run_report.md` summarising what's
verified vs what still needs Colab. Run `python -m scripts.dry_run` before
each Colab push.

## 2026-04-30 — Session 2 (Stage 5/6 added)

### Added since session 1

- **Stage 5 inference** (`src/inference.py`)
  - `ModelInferer` — self-consistency sampling on top of any retriever (5 samples @ T=0.7, top_p=0.9, majority vote on label, max-confidence sample's evidence list).
  - `ZeroShotInferer` — same shape but greedy decoding for ablation rows A1-A4.
  - `RetrievalOnlyInferer` — no LLM; predicts SUPPORTS (or arbitrary label) and emits retrieved evidences. Lets us measure retrieval F-score in isolation.
  - `predict_all` — batch driver, tqdm-aware progress, writes JSON validated against `eval.py` schema, gracefully degrades to NEI on per-claim failure.
- **Stage 6 ablation harness** (`src/ablation.py`)
  - `AblationConfig` dataclass (declarative pipeline toggles + `flagship` flag).
  - `AblationHarness` — model-agnostic; takes (config, predictions_dict_or_path) pairs; renders main table on official dev + diagnostic slice tables on `diag_test` (domain × 8, scenario × 7, difficulty × 3) + per-label slice on dev.
  - `DEFAULT_CONFIGS` — the nine A1-C2 configurations from Plan §6.1.
  - End-to-end demo confirmed renders all 4 tables from a single `predict()` dict spanning dev + diag_test.

### Tests

| Suite | Cases | Status |
|---|---|---|
| test_prompt | 8 | green |
| test_eval_helpers | 3 | green |
| test_sft_dataset | 3 | green |
| test_fuse | 4 | green |
| test_query_rewrite | 7 | green |
| test_dpo_pairs | 5 | green |
| test_inference | 4 | green |
| test_ablation | 3 | green |
| **total** | **37** | **all green** |

### Code surface

`src/` 14 modules, ~2400 lines. `tests/` 8 suites. Covered modules:
`data_io paths eda tagging splits prompt sft_dataset query_rewrite dpo_pairs eval_helpers retrieval/{bm25,dense,fuse,rerank,pipeline} inference ablation build_stage0`.

### Demo artifact

`outputs/ablation/ablation_report.md` — synthesised from baseline + 70%-correct flagship simulation. Confirms diagnostic tables surface the expected DISPUTED-hardest / supports_clear-easiest pattern.

---

## Session 1 (2026-04-30) — Stage 0 + Stage 1 scaffolding

### What's done

- **Project skeleton**: `src/` (11 modules), `tests/` (6 suites, 30 cases all green), `notebooks/`, `outputs/{eda,splits,sft_data}/`. `.gitignore` excludes evidence.json, checkpoints, embeddings, predictions.
- **Notebook ported to official template** (`notebooks/notebook.ipynb`, 45 cells). The 3 mandatory section headers (`1.DataSet Processing`, `2.Model Implementation`, `3.Testing and Evaluation`) untouched per assignment rule. Sub-sections fill them. OOP section at bottom re-imports key classes for grading visibility.
- **Stage 0 fully runnable locally** (`python -m src.build_stage0`, ~2 s force rebuild):
  - EDA report (key prior: NEI claims always have exactly 5 gold evidences)
  - Three-axis tagging: scenario × climate-domain × difficulty
  - Hash split: train_split 986 / dev_holdout 121 / diag_test 121 / official_dev 154
  - Six pairwise leakage assertions all pass
  - SFT data: train 1972 (with hard-neg ×1) / dev_holdout 121 / diag_test 121 in ms-swift format
- **Stage 1 retrieval scaffolding** (Colab-targeted but interface-tested locally):
  - `bm25.py` — `bm25s` wrapper with on-disk caching
  - `dense.py` — sentence-transformers (`bge-m3` default, `bge-small-en-v1.5` fallback) + FAISS, chunked encoding
  - `fuse.py` — weighted-sum (0.3 BM25 + 0.7 dense) + RRF
  - `rerank.py` — cross-encoder (`bge-reranker-base`) + rule-based reorder (NER boost, near-dup suppress, diversity cap)
  - `pipeline.py` — composable end-to-end with label-conditioned-k toggle
- **Stage 2 query rewriting** (`query_rewrite.py`): WordNet synonym expansion + sub-claim decomposition prompt + HyDE prompt + claim/hypothesis text/embedding blending
- **Stage 4 DPO pair builder** (`dpo_pairs.py`): mines errors from `dev_holdout` (never dev), supports DISPUTED-vs-SUPPORTS contrast augmentation
- **Eval helpers** (`eval_helpers.py`): bit-for-bit match with `eval.py` (verified to 1e-15 on baseline), plus per-bucket slicer + recall@k

### Performance fix

`build_dataset` had O(N×n_claims) blowup: rebuilt 1.2M-id pool per claim during random padding. Replaced with index-cached rejection sampling. **376 s → 0.1 s** for 1972-record build.

### Outputs on disk

```
outputs/
  eda/eda_report.md
  splits/{train_split,dev_holdout,diag_test,official_dev}.jsonl + split_summary.md
  sft_data/claims_tagged.jsonl + tag_distribution.md
  sft_data/sft_{train,dev_holdout,diag_test}_v1.jsonl
```

### Tests

| Suite | Cases | Status |
|---|---|---|
| test_prompt | 8 | green |
| test_eval_helpers | 3 | green (matches eval.py to 1e-15) |
| test_sft_dataset | 3 | green |
| test_fuse | 4 | green |
| test_query_rewrite | 7 | green |
| test_dpo_pairs | 5 | green |
| **total** | **30** | **all green** |

### What's blocked / pending

- `data/evidence.json` ✓ downloaded (174 MB, 1,208,827 passages)
- `notebooks/GroupID__COMP90042_Project_2026.ipynb` ✓ official template at hand
- **Needs Colab T4** (not local):
  - BM25 index build (~2-4 min)
  - bge-m3 full-corpus embedding (~30-60 min, cached to Drive)
  - Qwen3.5-4B download from ModelScope
  - ms-swift SFT 3 epochs (~75-105 min)
  - DPO 1 epoch (~25 min)
  - Inference on dev + test

### Decisions deferred until first Colab run

- Confirm `Qwen/Qwen3.5-4B-Instruct` exists on ModelScope. Fallback: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Confirm ms-swift's `--model_type` slug for Qwen3.5-VL. Fallback: Unsloth.
- Pick final retrieval weights (0.3/0.7 default) by k-sweep on dev.
