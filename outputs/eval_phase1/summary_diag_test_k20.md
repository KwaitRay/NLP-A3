# Phase 1 summary on diag_test_k20

Prompt variant sweep (D-015 Phase 2). Higher HM is better.
Track 1 = no-RAG (base model parametric only). Track 1 F is 0 by design.
Track 2 = full RAG (BM25 + dense + rerank) → base model.

| Track | Prompt | Variant | n | F | Acc | HM |
|---|---|---|---|---|---|---|
| 2 | v1 | baseline | 121 | 0.1360 | 0.3967 | 0.2025 |

## Phase 2 next step

1. Pick the prompt with the highest Track-2 HM as the locked production prompt.
2. Open the matching `track2_<prompt>_<dataset>.md` and inspect the per-bucket tables.
3. Buckets with HM < 0.30 are the SFT-data-augmentation targets for Phase 4.