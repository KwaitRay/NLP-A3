# Track 2 — prompt v1 on diag_test_k20

- variant: **baseline** (Original production prompt; minimal output rules, no examples.)
- claims: 121
- elapsed: 232.8s

## Overall
| F | Acc | HM |
|---|---|---|
| 0.1360 | 0.3967 | 0.2025 |

#### Per-domain

| bucket | n | F | Acc | HM |
|---|---|---|---|---|
| extreme_weather | 3 | 0.206 | 0.000 | 0.000 |
| co2_atmospheric | 18 | 0.097 | 0.444 | 0.160 |
| general_other | 55 | 0.131 | 0.400 | 0.197 |
| temperature | 26 | 0.130 | 0.423 | 0.199 |
| sea_level | 10 | 0.174 | 0.300 | 0.220 |
| models_attribution | 7 | 0.156 | 0.429 | 0.229 |
| paleoclimate | 2 | 0.333 | 0.500 | 0.400 |


#### Per-scenario

| bucket | n | F | Acc | HM |
|---|---|---|---|---|
| nei_underspec | 40 | 0.084 | 0.025 | 0.039 |
| refutes_clear | 17 | 0.064 | 0.588 | 0.116 |
| disputed_conflict | 21 | 0.144 | 0.190 | 0.164 |
| refutes_aggregated | 5 | 0.092 | 1.000 | 0.169 |
| supports_aggregated | 19 | 0.215 | 0.789 | 0.338 |
| supports_clear | 19 | 0.232 | 0.684 | 0.346 |


#### Per-difficulty

| bucket | n | F | Acc | HM |
|---|---|---|---|---|
| hard | 40 | 0.117 | 0.100 | 0.108 |
| medium | 54 | 0.096 | 0.444 | 0.158 |
| easy | 27 | 0.244 | 0.741 | 0.367 |
