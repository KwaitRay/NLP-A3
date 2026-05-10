# Stage 0.4 — split summary

Hash split `md5(salt||claim_id) % 10`: 0-7 → train_split, 8 → dev_holdout, 9 → diag_test.

| split | n | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED |
|---|---|---|---|---|---|
| train_split | 986 | 433 | 160 | 303 | 90 |
| dev_holdout | 121 | 48 | 17 | 43 | 13 |
| diag_test | 121 | 38 | 22 | 40 | 21 |
| official_dev | 154 | 68 | 27 | 41 | 18 |

Leakage assertions: all six pairwise intersections verified empty.