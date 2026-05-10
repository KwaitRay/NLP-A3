# Ablation report

## Ablation table (official dev)

| config | description | F | A | HM | n |
|---|---|---|---|---|---|
| DRY | dry-run stub | 0.0000 | 0.2532 | 0.0000 | 154 |

## Diagnostic slices on `diag_test`

Format: F-score / Accuracy. Buckets follow Stage 0.3 tagging.

### By climate-science domain

| bucket | DRY | n |
|---|---|---|
| temperature | F=0.000/A=0.346 | 26 |
| co2_atmospheric | F=0.000/A=0.389 | 18 |
| sea_level | F=0.000/A=0.500 | 10 |
| extreme_weather | F=0.000/A=0.000 | 3 |
| paleoclimate | F=0.500/A=0.000 | 2 |
| models_attribution | F=0.000/A=0.429 | 7 |
| policy_economics | — | 0 |
| general_other | F=0.000/A=0.291 | 55 |

### By scenario

| bucket | DRY | n |
|---|---|---|
| supports_clear | F=0.053/A=0.000 | 19 |
| supports_aggregated | F=0.000/A=0.000 | 19 |
| refutes_clear | F=0.000/A=0.000 | 17 |
| refutes_aggregated | F=0.000/A=0.000 | 5 |
| nei_topic_off | — | 0 |
| nei_underspec | F=0.000/A=1.000 | 40 |
| disputed_conflict | F=0.000/A=0.000 | 21 |

### By difficulty

| bucket | DRY | n |
|---|---|---|
| easy | F=0.037/A=0.000 | 27 |
| medium | F=0.000/A=0.296 | 54 |
| hard | F=0.000/A=0.600 | 40 |

### By gold label (official dev)

| bucket | DRY | n |
|---|---|---|
| SUPPORTS | F=0.000/A=0.294 | 68 |
| REFUTES | F=0.000/A=0.259 | 27 |
| NOT_ENOUGH_INFO | F=0.000/A=0.220 | 41 |
| DISPUTED | F=0.000/A=0.167 | 18 |
