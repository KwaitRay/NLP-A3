# Phase 1 diagnostic — diag_test_k10

Source: `outputs/eval_phase1/track*_*_diag_test_k10.json`  (1 run(s))

## Cross-run summary

| Track | Prompt | n | Acc | non-NEI acc | predicted NEI share | NEI-default? |
|---|---|---|---|---|---|---|
| 2 | v1 | 121 | 0.3884 | 0.4938 | 11.6% | no |

Legend:
- *non-NEI acc*: accuracy on the non-NEI gold claims only. If this is near 0 while overall acc ≈ gold NEI share, the model is probably defaulting to NEI for everything (parse-fallback pattern).
- *predicted NEI share*: fraction of claims the parser labelled NEI.
- *NEI-default?*: see per-run heuristic in each section below.

## Track 2 — prompt `v1`  (n=121, Acc=0.3884)

### Predicted vs gold label distribution

| label | gold | predicted | Δ |
|---|---|---|---|
| SUPPORTS | 38 (31.4%) | 48 (39.7%) | +10 |
| REFUTES | 22 (18.2%) | 31 (25.6%) | +9 |
| NOT_ENOUGH_INFO | 40 (33.1%) | 14 (11.6%) | -26 |
| DISPUTED | 21 (17.4%) | 28 (23.1%) | +7 |

### Confusion matrix (rows = gold, columns = pred)

| gold\pred | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED | **total** |
|---|---|---|---|---|---|
| SUPPORTS | **25** | 3 | 5 | 5 | 38 |
| REFUTES | 5 | **10** | 0 | 7 | 22 |
| NOT_ENOUGH_INFO | 11 | 11 | **7** | 11 | 40 |
| DISPUTED | 7 | 7 | 2 | **5** | 21 |
| **total** | 48 | 31 | 14 | 28 | 121 |

### Per-gold-label correctness

| gold label | correct / n | accuracy |
|---|---|---|
| SUPPORTS | 25 / 38 | 0.658 |
| REFUTES | 10 / 22 | 0.455 |
| NOT_ENOUGH_INFO | 7 / 40 | 0.175 |
| DISPUTED | 5 / 21 | 0.238 |

### Evidence recall (predicted ∩ gold) / gold

- macro: 0.1471  (mean over 121 claims with gold ev)
- micro: 0.1275

### Diagnostic flag

- No defaulting-to-NEI pattern detected.

### Sample mispredictions

- non-NEI gold predicted as NEI:
  - `claim-3084` gold=DISPUTED pred=NOT_ENOUGH_INFO — ""the temperature increase in the second half of the 20th century could have taken place in steps driven by major ENSO events" (Jens Rauns..."
  - `claim-1931` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "New Jersey is "losing 50 football fields of open space to development every day and the more we develop upstream the more flooding we hav..."
  - `claim-133` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "The amount of land we use for meat — humankind’s biggest use of land — has declined by an area nearly as large as Alaska"
- non-NEI gold predicted as a *different* non-NEI label:
  - `claim-2834` gold=SUPPORTS pred=DISPUTED — "When stomata-derived CO2 (red) is compared to ice core-derived CO2 (blue), the stomata generally show much more variability in the atmosp..."
  - `claim-2789` gold=SUPPORTS pred=DISPUTED — "At that time, Hansen also produced a model of the future behavior of the globe’s temperature, which he had turned into a video movie that..."
  - `claim-2494` gold=DISPUTED pred=SUPPORTS — "There is ample evidence that Earth's average temperature has increased in the past 100 years and the decline of mid- and high-latitude gl..."
- correct non-NEI predictions (sanity):
  - `claim-2257` gold=REFUTES pred=REFUTES — "'Our harmless emissions of trifling quantities of carbon dioxide cannot possibly acidify the oceans."
  - `claim-69` gold=REFUTES pred=REFUTES — "Sea level rise is not going to happen."
  - `claim-2627` gold=SUPPORTS pred=SUPPORTS — "Neptune's orbit is 164 years so observations (1950 to present day) span less than a third of a Neptunian year."
