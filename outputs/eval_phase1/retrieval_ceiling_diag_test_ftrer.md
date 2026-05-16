# Retrieval ceiling audit — diag_test_ftrer

Modes run: retriever.  Rerank: on.  k_grid = [5, 10, 20, 50, 100].

Reference: Track 2 v1 macro recall@5 was **0.1090** in `outputs/eval_phase1/diagnose_diag_test.md`. Numbers below should reproduce that for the matching config (full pipeline, w_bm25=0.3, final_k=5, rerank on).

## 🏆 Best overall at recall@5

- mode: `retriever`
- config: `fused (no rerank), final_k=5`
- macro recall@5: **0.2003**  (micro 0.1765)
- per-label macro: S 0.327 / R 0.138 / NEI 0.090 / D 0.246

## Mode: `retriever`

Elapsed: 877s

### recall@5 (production k)

| config | n | macro recall@5 | micro recall@5 | S | R | NEI | D |
|---|---|---|---|---|---|---|---|
| fused (no rerank), final_k=5 | 121 | 0.2003 | 0.1765 | 0.327 | 0.138 | 0.090 | 0.246 |
| dense only, final_k=5 | 121 | 0.1704 | 0.1593 | 0.284 | 0.102 | 0.085 | 0.199 |
| full (fused + rerank), final_k=5 | 121 | 0.1456 | 0.1275 | 0.266 | 0.032 | 0.055 | 0.220 |
| BM25 only, final_k=5 | 121 | 0.1358 | 0.1250 | 0.242 | 0.061 | 0.060 | 0.167 |

### recall@k curve (k ∈ [5, 10, 20, 50, 100])

| base config | r@5 | r@10 | r@20 | r@50 | r@100 |
|---|---|---|---|---|---|
| BM25 only | 0.136 | 0.185 | 0.263 | 0.340 | 0.393 |
| dense only | 0.170 | 0.235 | 0.319 | 0.444 | 0.541 |
| **fused (no rerank)** | 0.200 | 0.273 | 0.360 | 0.485 | 0.579 |
| full (fused + rerank) | 0.146 | 0.195 | 0.306 | 0.485 | 0.579 |

---

Next actions:
1. Lock the best `RetrievalConfig` into `src/retrieval/pipeline.py` (default) and re-run `phase1_eval --tracks 2 --prompts v1` to confirm Track 2 HM lifts.
2. Rebuild SFT data with the new retrieval config: `python -m src.build_stage0 --force`.
3. If recall@100 is still < 0.30, escalate to LLM-driven rewrite (HyDE / sub-claim, see `optimization_plan.md` §3.5.4).