# Retrieval ceiling audit — diag_test

Modes run: retriever, fusion_w, synonym_expand.  Rerank: on.  k_grid = [5, 10, 20, 50, 100].

Reference: Track 2 v1 macro recall@5 was **0.1090** in `outputs/eval_phase1/diagnose_diag_test.md`. Numbers below should reproduce that for the matching config (full pipeline, w_bm25=0.3, final_k=5, rerank on).

## 🏆 Best overall at recall@5

- mode: `retriever`
- config: `fused (no rerank), final_k=5`
- macro recall@5: **0.2003**  (micro 0.1765)
- per-label macro: S 0.327 / R 0.138 / NEI 0.090 / D 0.246

## Mode: `retriever`

Elapsed: 871s

### recall@5 (production k)

| config | n | macro recall@5 | micro recall@5 | S | R | NEI | D |
|---|---|---|---|---|---|---|---|
| fused (no rerank), final_k=5 | 121 | 0.2003 | 0.1765 | 0.327 | 0.138 | 0.090 | 0.246 |
| dense only, final_k=5 | 121 | 0.1704 | 0.1593 | 0.284 | 0.102 | 0.085 | 0.199 |
| BM25 only, final_k=5 | 121 | 0.1358 | 0.1250 | 0.242 | 0.061 | 0.060 | 0.167 |
| full (fused + rerank), final_k=5 | 121 | 0.1191 | 0.1078 | 0.198 | 0.073 | 0.055 | 0.147 |

### recall@k curve (k ∈ [5, 10, 20, 50, 100])

| base config | r@5 | r@10 | r@20 | r@50 | r@100 |
|---|---|---|---|---|---|
| BM25 only | 0.136 | 0.185 | 0.263 | 0.340 | 0.393 |
| dense only | 0.170 | 0.235 | 0.319 | 0.444 | 0.541 |
| **fused (no rerank)** | 0.200 | 0.273 | 0.360 | 0.485 | 0.579 |
| full (fused + rerank) | 0.119 | 0.210 | 0.333 | 0.485 | 0.579 |

## Mode: `fusion_w`

Elapsed: 1451s

### recall@5 (production k)

| config | n | macro recall@5 | micro recall@5 | S | R | NEI | D |
|---|---|---|---|---|---|---|---|
| w_bm25=0.9, w_dense=0.1, final_k=5 | 121 | 0.1544 | 0.1324 | 0.276 | 0.109 | 0.075 | 0.133 |
| w_bm25=0.7, w_dense=0.3, final_k=5 | 121 | 0.1543 | 0.1397 | 0.232 | 0.109 | 0.090 | 0.184 |
| w_bm25=0.5, w_dense=0.5, final_k=5 | 121 | 0.1236 | 0.1127 | 0.198 | 0.073 | 0.060 | 0.163 |
| w_bm25=0.1, w_dense=0.9, final_k=5 | 121 | 0.1191 | 0.1078 | 0.204 | 0.073 | 0.055 | 0.137 |
| w_bm25=0.3, w_dense=0.7, final_k=5 | 121 | 0.1191 | 0.1078 | 0.198 | 0.073 | 0.055 | 0.147 |

### recall@k curve (k ∈ [5, 10, 20, 50, 100])

| base config | r@5 | r@10 | r@20 | r@50 | r@100 |
|---|---|---|---|---|---|
| w_bm25=0.1, w_dense=0.9 | 0.119 | 0.189 | 0.319 | 0.458 | 0.552 |
| w_bm25=0.3, w_dense=0.7 | 0.119 | 0.210 | 0.333 | 0.485 | 0.579 |
| w_bm25=0.5, w_dense=0.5 | 0.124 | 0.221 | 0.343 | 0.474 | 0.559 |
| w_bm25=0.7, w_dense=0.3 | 0.154 | 0.250 | 0.330 | 0.419 | 0.509 |
| w_bm25=0.9, w_dense=0.1 | 0.154 | 0.215 | 0.283 | 0.360 | 0.422 |

## Mode: `synonym_expand`

Elapsed: 1575s

### recall@5 (production k)

| config | n | macro recall@5 | micro recall@5 | S | R | NEI | D |
|---|---|---|---|---|---|---|---|
| claim only (baseline), final_k=5 | 121 | 0.1237 | 0.1152 | 0.199 | 0.073 | 0.060 | 0.163 |
| claim + WordNet synonyms, final_k=5 | 121 | 0.1197 | 0.1103 | 0.200 | 0.073 | 0.055 | 0.147 |

### recall@k curve (k ∈ [5, 10, 20, 50, 100])

| base config | r@5 | r@10 | r@20 | r@50 | r@100 |
|---|---|---|---|---|---|
| claim only (baseline) | 0.124 | 0.209 | 0.335 | 0.467 | 0.558 |
| claim + WordNet synonyms | 0.120 | 0.215 | 0.331 | 0.458 | 0.552 |

---

Next actions:
1. Lock the best `RetrievalConfig` into `src/retrieval/pipeline.py` (default) and re-run `phase1_eval --tracks 2 --prompts v1` to confirm Track 2 HM lifts.
2. Rebuild SFT data with the new retrieval config: `python -m src.build_stage0 --force`.
3. If recall@100 is still < 0.30, escalate to LLM-driven rewrite (HyDE / sub-claim, see `optimization_plan.md` §3.5.4).