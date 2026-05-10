# Ablation report

## Ablation table (official dev)

| config | description | F | A | HM | n |
|---|---|---|---|---|---|
| B3 | flagship demo (combined dev+diag preds) | 0.8381 | 0.7078 | 0.7675 | 154 |
| A1 | BM25 + zero-shot baseline | 0.3378 | 0.3506 | 0.3441 | 154 |

## Diagnostic slices on `diag_test`

Format: F-score / Accuracy. Buckets follow Stage 0.3 tagging.

### By climate-science domain

| bucket | B3 | n |
|---|---|---|
| temperature | F=0.791/A=0.692 | 26 |
| co2_atmospheric | F=0.767/A=0.611 | 18 |
| sea_level | F=0.773/A=0.800 | 10 |
| extreme_weather | F=0.889/A=1.000 | 3 |
| paleoclimate | F=0.833/A=0.500 | 2 |
| models_attribution | F=0.810/A=0.571 | 7 |
| policy_economics | — | 0 |
| general_other | F=0.841/A=0.709 | 55 |

### By scenario

| bucket | B3 | n |
|---|---|---|
| supports_clear | F=0.965/A=0.737 | 19 |
| supports_aggregated | F=0.730/A=0.684 | 19 |
| refutes_clear | F=0.922/A=0.765 | 17 |
| refutes_aggregated | F=0.867/A=0.800 | 5 |
| nei_topic_off | — | 0 |
| nei_underspec | F=0.767/A=0.750 | 40 |
| disputed_conflict | F=0.737/A=0.476 | 21 |

### By difficulty

| bucket | B3 | n |
|---|---|---|
| easy | F=0.891/A=0.704 | 27 |
| medium | F=0.820/A=0.741 | 54 |
| hard | F=0.749/A=0.625 | 40 |

### By gold label (official dev)

| bucket | B3 | n |
|---|---|---|
| SUPPORTS | F=0.871/A=0.721 | 68 |
| REFUTES | F=0.894/A=0.704 | 27 |
| NOT_ENOUGH_INFO | F=0.756/A=0.707 | 41 |
| DISPUTED | F=0.819/A=0.667 | 18 |
