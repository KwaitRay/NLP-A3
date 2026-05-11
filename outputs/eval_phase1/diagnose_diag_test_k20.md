# Phase 1 diagnostic — diag_test_k20

Source: `outputs/eval_phase1/track*_*_diag_test_k20.json`  (1 run(s))

## Cross-run summary

| Track | Prompt | n | Acc | non-NEI acc | predicted NEI share | NEI-default? |
|---|---|---|---|---|---|---|
| 2 | v1 | 121 | 0.3967 | 0.5802 | 2.5% | no |

Legend:
- *non-NEI acc*: accuracy on the non-NEI gold claims only. If this is near 0 while overall acc ≈ gold NEI share, the model is probably defaulting to NEI for everything (parse-fallback pattern).
- *predicted NEI share*: fraction of claims the parser labelled NEI.
- *NEI-default?*: see per-run heuristic in each section below.

## Track 2 — prompt `v1`  (n=121, Acc=0.3967)

### Predicted vs gold label distribution

| label | gold | predicted | Δ |
|---|---|---|---|
| SUPPORTS | 38 (31.4%) | 50 (41.3%) | +12 |
| REFUTES | 22 (18.2%) | 40 (33.1%) | +18 |
| NOT_ENOUGH_INFO | 40 (33.1%) | 3 (2.5%) | -37 |
| DISPUTED | 21 (17.4%) | 28 (23.1%) | +7 |

### Confusion matrix (rows = gold, columns = pred)

| gold\pred | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED | **total** |
|---|---|---|---|---|---|
| SUPPORTS | **28** | 4 | 1 | 5 | 38 |
| REFUTES | 4 | **15** | 0 | 3 | 22 |
| NOT_ENOUGH_INFO | 10 | 13 | **1** | 16 | 40 |
| DISPUTED | 8 | 8 | 1 | **4** | 21 |
| **total** | 50 | 40 | 3 | 28 | 121 |

### Per-gold-label correctness

| gold label | correct / n | accuracy |
|---|---|---|
| SUPPORTS | 28 / 38 | 0.737 |
| REFUTES | 15 / 22 | 0.682 |
| NOT_ENOUGH_INFO | 1 / 40 | 0.025 |
| DISPUTED | 4 / 21 | 0.190 |

### Evidence recall (predicted ∩ gold) / gold

- macro: 0.2036  (mean over 121 claims with gold ev)
- micro: 0.1642

### Diagnostic flag

- No defaulting-to-NEI pattern detected.

### Sample mispredictions

- non-NEI gold predicted as NEI:
  - `claim-1931` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "New Jersey is "losing 50 football fields of open space to development every day and the more we develop upstream the more flooding we hav..."
  - `claim-1877` gold=DISPUTED pred=NOT_ENOUGH_INFO — "During a state House debate on a jobs and energy bill this week, Democrats offered an amendment that would put the Legislature on record ..."
- non-NEI gold predicted as a *different* non-NEI label:
  - `claim-2494` gold=DISPUTED pred=SUPPORTS — "There is ample evidence that Earth's average temperature has increased in the past 100 years and the decline of mid- and high-latitude gl..."
  - `claim-1115` gold=SUPPORTS pred=DISPUTED — "This “blocking” effect means extreme events can unfold.”"
  - `claim-663` gold=DISPUTED pred=REFUTES — "“For example, Canadian polar bear biologist Ian Stirling learned in the 1970s that spring sea ice in the southern Beaufort Sea periodical..."
- correct non-NEI predictions (sanity):
  - `claim-2834` gold=SUPPORTS pred=SUPPORTS — "When stomata-derived CO2 (red) is compared to ice core-derived CO2 (blue), the stomata generally show much more variability in the atmosp..."
  - `claim-2789` gold=SUPPORTS pred=SUPPORTS — "At that time, Hansen also produced a model of the future behavior of the globe’s temperature, which he had turned into a video movie that..."
  - `claim-2257` gold=REFUTES pred=REFUTES — "'Our harmless emissions of trifling quantities of carbon dioxide cannot possibly acidify the oceans."
