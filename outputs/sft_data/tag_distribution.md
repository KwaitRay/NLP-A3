# Stage 0.3 — tag distribution (train split)

### Domain distribution by label

| domain | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED | total |
|---|---|---|---|---|---|
| temperature | 127 | 41 | 68 | 29 | 265 |
| co2_atmospheric | 94 | 44 | 54 | 23 | 215 |
| sea_level | 44 | 18 | 18 | 12 | 92 |
| extreme_weather | 12 | 11 | 11 | 3 | 37 |
| paleoclimate | 11 | 0 | 7 | 2 | 20 |
| models_attribution | 25 | 6 | 15 | 7 | 53 |
| policy_economics | 6 | 7 | 10 | 4 | 27 |
| general_other | 200 | 72 | 203 | 44 | 519 |
| **total** | 519 | 199 | 386 | 124 | 1228 |

### Scenario distribution

| scenario | count |
|---|---|
| supports_clear | 264 |
| supports_aggregated | 255 |
| refutes_clear | 124 |
| refutes_aggregated | 75 |
| nei_topic_off | 0 |
| nei_underspec | 386 |
| disputed_conflict | 124 |

### Difficulty by label

| label | easy | medium | hard | total |
|---|---|---|---|---|
| SUPPORTS | 312 | 207 | 0 | 519 |
| REFUTES | 34 | 162 | 3 | 199 |
| NOT_ENOUGH_INFO | 0 | 111 | 275 | 386 |
| DISPUTED | 0 | 41 | 83 | 124 |
