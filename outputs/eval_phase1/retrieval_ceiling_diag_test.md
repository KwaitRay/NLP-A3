# Retrieval ceiling audit — diag_test

Modes run: final_k.  Rerank: on.  k_grid = [5, 10, 20, 50, 100].

Reference: Track 2 v1 macro recall@5 was **0.1090** in `outputs/eval_phase1/diagnose_diag_test.md`. Numbers below should reproduce that for the matching config (full pipeline, w_bm25=0.3, final_k=5, rerank on).

## 🏆 Best overall at recall@5

- mode: `final_k`
- config: `full pipeline, final_k=5`
- macro recall@5: **0.1191**  (micro 0.1078)
- per-label macro: S 0.198 / R 0.073 / NEI 0.055 / D 0.147

## Mode: `final_k`

Elapsed: 339s

### recall@5 (production k)

| config | n | macro recall@5 | micro recall@5 | S | R | NEI | D |
|---|---|---|---|---|---|---|---|
| full pipeline, final_k=5 | 121 | 0.1191 | 0.1078 | 0.198 | 0.073 | 0.055 | 0.147 |

### recall@k curve (k ∈ [5, 10, 20, 50, 100])

| base config | r@5 | r@10 | r@20 | r@50 | r@100 |
|---|---|---|---|---|---|
| **full pipeline** | 0.119 | 0.210 | 0.333 | 0.485 | 0.579 |

---

Next actions:
1. Lock the best `RetrievalConfig` into `src/retrieval/pipeline.py` (default) and re-run `phase1_eval --tracks 2 --prompts v1` to confirm Track 2 HM lifts.
2. Rebuild SFT data with the new retrieval config: `python -m src.build_stage0 --force`.
3. If recall@100 is still < 0.30, escalate to LLM-driven rewrite (HyDE / sub-claim, see `optimization_plan.md` §3.5.4).