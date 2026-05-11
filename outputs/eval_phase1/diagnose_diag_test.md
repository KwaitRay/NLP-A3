# Phase 1 diagnostic — diag_test

Source: `outputs/eval_phase1/track*_*_diag_test.json`  (7 run(s))

## Cross-run summary

| Track | Prompt | n | Acc | non-NEI acc | predicted NEI share | NEI-default? |
|---|---|---|---|---|---|---|
| 1 | v1 | 121 | 0.3223 | 0.4691 | 4.1% | no |
| 1 | v2 | 121 | 0.3223 | 0.4691 | 4.1% | no |
| 1 | v3 | 121 | 0.3223 | 0.4691 | 4.1% | no |
| 2 | v1 | 121 | 0.4215 | 0.4568 | 24.8% | no |
| 2 | v2 | 121 | 0.4132 | 0.3457 | 42.1% | no |
| 2 | v3 | 121 | 0.2562 | 0.2963 | 14.9% | no |
| 2 | v4 | 121 | 0.2893 | 0.4074 | 4.1% | no |

Legend:
- *non-NEI acc*: accuracy on the non-NEI gold claims only. If this is near 0 while overall acc ≈ gold NEI share, the model is probably defaulting to NEI for everything (parse-fallback pattern).
- *predicted NEI share*: fraction of claims the parser labelled NEI.
- *NEI-default?*: see per-run heuristic in each section below.

## Track 1 — prompt `v1`  (n=121, Acc=0.3223)

### Predicted vs gold label distribution

| label | gold | predicted | Δ |
|---|---|---|---|
| SUPPORTS | 38 (31.4%) | 56 (46.3%) | +18 |
| REFUTES | 22 (18.2%) | 59 (48.8%) | +37 |
| NOT_ENOUGH_INFO | 40 (33.1%) | 5 (4.1%) | -35 |
| DISPUTED | 21 (17.4%) | 1 (0.8%) | -20 |

### Confusion matrix (rows = gold, columns = pred)

| gold\pred | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED | **total** |
|---|---|---|---|---|---|
| SUPPORTS | **25** | 11 | 2 | 0 | 38 |
| REFUTES | 7 | **13** | 2 | 0 | 22 |
| NOT_ENOUGH_INFO | 15 | 23 | **1** | 1 | 40 |
| DISPUTED | 9 | 12 | 0 | 0 | 21 |
| **total** | 56 | 59 | 5 | 1 | 121 |

### Per-gold-label correctness

| gold label | correct / n | accuracy |
|---|---|---|
| SUPPORTS | 25 / 38 | 0.658 |
| REFUTES | 13 / 22 | 0.591 |
| NOT_ENOUGH_INFO | 1 / 40 | 0.025 |
| DISPUTED | 0 / 21 | 0.000 |

### Diagnostic flag

- No defaulting-to-NEI pattern detected.

### Sample mispredictions

- non-NEI gold predicted as NEI:
  - `claim-1115` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "This “blocking” effect means extreme events can unfold.”"
  - `claim-630` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "I conclude that it must be ice accumulation, through evaporation of ocean water, and subsequent precipitation turning into ice."
  - `claim-2959` gold=REFUTES pred=NOT_ENOUGH_INFO — "Even if the warming were as big as the IPCC imagines, it would not be as dangerous as Mr. Brown suggests."
- non-NEI gold predicted as a *different* non-NEI label:
  - `claim-3084` gold=DISPUTED pred=REFUTES — ""the temperature increase in the second half of the 20th century could have taken place in steps driven by major ENSO events" (Jens Rauns..."
  - `claim-2494` gold=DISPUTED pred=SUPPORTS — "There is ample evidence that Earth's average temperature has increased in the past 100 years and the decline of mid- and high-latitude gl..."
  - `claim-1931` gold=SUPPORTS pred=REFUTES — "New Jersey is "losing 50 football fields of open space to development every day and the more we develop upstream the more flooding we hav..."
- correct non-NEI predictions (sanity):
  - `claim-2834` gold=SUPPORTS pred=SUPPORTS — "When stomata-derived CO2 (red) is compared to ice core-derived CO2 (blue), the stomata generally show much more variability in the atmosp..."
  - `claim-2789` gold=SUPPORTS pred=SUPPORTS — "At that time, Hansen also produced a model of the future behavior of the globe’s temperature, which he had turned into a video movie that..."
  - `claim-2257` gold=REFUTES pred=REFUTES — "'Our harmless emissions of trifling quantities of carbon dioxide cannot possibly acidify the oceans."

## Track 1 — prompt `v2`  (n=121, Acc=0.3223)

### Predicted vs gold label distribution

| label | gold | predicted | Δ |
|---|---|---|---|
| SUPPORTS | 38 (31.4%) | 56 (46.3%) | +18 |
| REFUTES | 22 (18.2%) | 59 (48.8%) | +37 |
| NOT_ENOUGH_INFO | 40 (33.1%) | 5 (4.1%) | -35 |
| DISPUTED | 21 (17.4%) | 1 (0.8%) | -20 |

### Confusion matrix (rows = gold, columns = pred)

| gold\pred | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED | **total** |
|---|---|---|---|---|---|
| SUPPORTS | **25** | 11 | 2 | 0 | 38 |
| REFUTES | 7 | **13** | 2 | 0 | 22 |
| NOT_ENOUGH_INFO | 15 | 23 | **1** | 1 | 40 |
| DISPUTED | 9 | 12 | 0 | 0 | 21 |
| **total** | 56 | 59 | 5 | 1 | 121 |

### Per-gold-label correctness

| gold label | correct / n | accuracy |
|---|---|---|
| SUPPORTS | 25 / 38 | 0.658 |
| REFUTES | 13 / 22 | 0.591 |
| NOT_ENOUGH_INFO | 1 / 40 | 0.025 |
| DISPUTED | 0 / 21 | 0.000 |

### Diagnostic flag

- No defaulting-to-NEI pattern detected.

### Sample mispredictions

- non-NEI gold predicted as NEI:
  - `claim-1115` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "This “blocking” effect means extreme events can unfold.”"
  - `claim-630` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "I conclude that it must be ice accumulation, through evaporation of ocean water, and subsequent precipitation turning into ice."
  - `claim-2959` gold=REFUTES pred=NOT_ENOUGH_INFO — "Even if the warming were as big as the IPCC imagines, it would not be as dangerous as Mr. Brown suggests."
- non-NEI gold predicted as a *different* non-NEI label:
  - `claim-3084` gold=DISPUTED pred=REFUTES — ""the temperature increase in the second half of the 20th century could have taken place in steps driven by major ENSO events" (Jens Rauns..."
  - `claim-2494` gold=DISPUTED pred=SUPPORTS — "There is ample evidence that Earth's average temperature has increased in the past 100 years and the decline of mid- and high-latitude gl..."
  - `claim-1931` gold=SUPPORTS pred=REFUTES — "New Jersey is "losing 50 football fields of open space to development every day and the more we develop upstream the more flooding we hav..."
- correct non-NEI predictions (sanity):
  - `claim-2834` gold=SUPPORTS pred=SUPPORTS — "When stomata-derived CO2 (red) is compared to ice core-derived CO2 (blue), the stomata generally show much more variability in the atmosp..."
  - `claim-2789` gold=SUPPORTS pred=SUPPORTS — "At that time, Hansen also produced a model of the future behavior of the globe’s temperature, which he had turned into a video movie that..."
  - `claim-2257` gold=REFUTES pred=REFUTES — "'Our harmless emissions of trifling quantities of carbon dioxide cannot possibly acidify the oceans."

## Track 1 — prompt `v3`  (n=121, Acc=0.3223)

### Predicted vs gold label distribution

| label | gold | predicted | Δ |
|---|---|---|---|
| SUPPORTS | 38 (31.4%) | 56 (46.3%) | +18 |
| REFUTES | 22 (18.2%) | 59 (48.8%) | +37 |
| NOT_ENOUGH_INFO | 40 (33.1%) | 5 (4.1%) | -35 |
| DISPUTED | 21 (17.4%) | 1 (0.8%) | -20 |

### Confusion matrix (rows = gold, columns = pred)

| gold\pred | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED | **total** |
|---|---|---|---|---|---|
| SUPPORTS | **25** | 11 | 2 | 0 | 38 |
| REFUTES | 7 | **13** | 2 | 0 | 22 |
| NOT_ENOUGH_INFO | 15 | 23 | **1** | 1 | 40 |
| DISPUTED | 9 | 12 | 0 | 0 | 21 |
| **total** | 56 | 59 | 5 | 1 | 121 |

### Per-gold-label correctness

| gold label | correct / n | accuracy |
|---|---|---|
| SUPPORTS | 25 / 38 | 0.658 |
| REFUTES | 13 / 22 | 0.591 |
| NOT_ENOUGH_INFO | 1 / 40 | 0.025 |
| DISPUTED | 0 / 21 | 0.000 |

### Diagnostic flag

- No defaulting-to-NEI pattern detected.

### Sample mispredictions

- non-NEI gold predicted as NEI:
  - `claim-1115` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "This “blocking” effect means extreme events can unfold.”"
  - `claim-630` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "I conclude that it must be ice accumulation, through evaporation of ocean water, and subsequent precipitation turning into ice."
  - `claim-2959` gold=REFUTES pred=NOT_ENOUGH_INFO — "Even if the warming were as big as the IPCC imagines, it would not be as dangerous as Mr. Brown suggests."
- non-NEI gold predicted as a *different* non-NEI label:
  - `claim-3084` gold=DISPUTED pred=REFUTES — ""the temperature increase in the second half of the 20th century could have taken place in steps driven by major ENSO events" (Jens Rauns..."
  - `claim-2494` gold=DISPUTED pred=SUPPORTS — "There is ample evidence that Earth's average temperature has increased in the past 100 years and the decline of mid- and high-latitude gl..."
  - `claim-1931` gold=SUPPORTS pred=REFUTES — "New Jersey is "losing 50 football fields of open space to development every day and the more we develop upstream the more flooding we hav..."
- correct non-NEI predictions (sanity):
  - `claim-2834` gold=SUPPORTS pred=SUPPORTS — "When stomata-derived CO2 (red) is compared to ice core-derived CO2 (blue), the stomata generally show much more variability in the atmosp..."
  - `claim-2789` gold=SUPPORTS pred=SUPPORTS — "At that time, Hansen also produced a model of the future behavior of the globe’s temperature, which he had turned into a video movie that..."
  - `claim-2257` gold=REFUTES pred=REFUTES — "'Our harmless emissions of trifling quantities of carbon dioxide cannot possibly acidify the oceans."

## Track 2 — prompt `v1`  (n=121, Acc=0.4215)

### Predicted vs gold label distribution

| label | gold | predicted | Δ |
|---|---|---|---|
| SUPPORTS | 38 (31.4%) | 40 (33.1%) | +2 |
| REFUTES | 22 (18.2%) | 27 (22.3%) | +5 |
| NOT_ENOUGH_INFO | 40 (33.1%) | 30 (24.8%) | -10 |
| DISPUTED | 21 (17.4%) | 24 (19.8%) | +3 |

### Confusion matrix (rows = gold, columns = pred)

| gold\pred | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED | **total** |
|---|---|---|---|---|---|
| SUPPORTS | **20** | 4 | 8 | 6 | 38 |
| REFUTES | 4 | **11** | 3 | 4 | 22 |
| NOT_ENOUGH_INFO | 9 | 9 | **14** | 8 | 40 |
| DISPUTED | 7 | 3 | 5 | **6** | 21 |
| **total** | 40 | 27 | 30 | 24 | 121 |

### Per-gold-label correctness

| gold label | correct / n | accuracy |
|---|---|---|
| SUPPORTS | 20 / 38 | 0.526 |
| REFUTES | 11 / 22 | 0.500 |
| NOT_ENOUGH_INFO | 14 / 40 | 0.350 |
| DISPUTED | 6 / 21 | 0.286 |

### Evidence recall (predicted ∩ gold) / gold

- macro: 0.1090  (mean over 121 claims with gold ev)
- micro: 0.0980

### Diagnostic flag

- No defaulting-to-NEI pattern detected.

### Sample mispredictions

- non-NEI gold predicted as NEI:
  - `claim-3084` gold=DISPUTED pred=NOT_ENOUGH_INFO — ""the temperature increase in the second half of the 20th century could have taken place in steps driven by major ENSO events" (Jens Rauns..."
  - `claim-1115` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "This “blocking” effect means extreme events can unfold.”"
  - `claim-1931` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "New Jersey is "losing 50 football fields of open space to development every day and the more we develop upstream the more flooding we hav..."
- non-NEI gold predicted as a *different* non-NEI label:
  - `claim-2834` gold=SUPPORTS pred=DISPUTED — "When stomata-derived CO2 (red) is compared to ice core-derived CO2 (blue), the stomata generally show much more variability in the atmosp..."
  - `claim-2789` gold=SUPPORTS pred=DISPUTED — "At that time, Hansen also produced a model of the future behavior of the globe’s temperature, which he had turned into a video movie that..."
  - `claim-2494` gold=DISPUTED pred=SUPPORTS — "There is ample evidence that Earth's average temperature has increased in the past 100 years and the decline of mid- and high-latitude gl..."
- correct non-NEI predictions (sanity):
  - `claim-2257` gold=REFUTES pred=REFUTES — "'Our harmless emissions of trifling quantities of carbon dioxide cannot possibly acidify the oceans."
  - `claim-69` gold=REFUTES pred=REFUTES — "Sea level rise is not going to happen."
  - `claim-2627` gold=SUPPORTS pred=SUPPORTS — "Neptune's orbit is 164 years so observations (1950 to present day) span less than a third of a Neptunian year."

## Track 2 — prompt `v2`  (n=121, Acc=0.4132)

### Predicted vs gold label distribution

| label | gold | predicted | Δ |
|---|---|---|---|
| SUPPORTS | 38 (31.4%) | 43 (35.5%) | +5 |
| REFUTES | 22 (18.2%) | 12 (9.9%) | -10 |
| NOT_ENOUGH_INFO | 40 (33.1%) | 51 (42.1%) | +11 |
| DISPUTED | 21 (17.4%) | 15 (12.4%) | -6 |

### Confusion matrix (rows = gold, columns = pred)

| gold\pred | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED | **total** |
|---|---|---|---|---|---|
| SUPPORTS | **20** | 4 | 11 | 3 | 38 |
| REFUTES | 4 | **5** | 9 | 4 | 22 |
| NOT_ENOUGH_INFO | 11 | 2 | **22** | 5 | 40 |
| DISPUTED | 8 | 1 | 9 | **3** | 21 |
| **total** | 43 | 12 | 51 | 15 | 121 |

### Per-gold-label correctness

| gold label | correct / n | accuracy |
|---|---|---|
| SUPPORTS | 20 / 38 | 0.526 |
| REFUTES | 5 / 22 | 0.227 |
| NOT_ENOUGH_INFO | 22 / 40 | 0.550 |
| DISPUTED | 3 / 21 | 0.143 |

### Evidence recall (predicted ∩ gold) / gold

- macro: 0.1123  (mean over 121 claims with gold ev)
- micro: 0.1029

### Diagnostic flag

- No defaulting-to-NEI pattern detected.

### Sample mispredictions

- non-NEI gold predicted as NEI:
  - `claim-2789` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "At that time, Hansen also produced a model of the future behavior of the globe’s temperature, which he had turned into a video movie that..."
  - `claim-3084` gold=DISPUTED pred=NOT_ENOUGH_INFO — ""the temperature increase in the second half of the 20th century could have taken place in steps driven by major ENSO events" (Jens Rauns..."
  - `claim-1115` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "This “blocking” effect means extreme events can unfold.”"
- non-NEI gold predicted as a *different* non-NEI label:
  - `claim-2834` gold=SUPPORTS pred=DISPUTED — "When stomata-derived CO2 (red) is compared to ice core-derived CO2 (blue), the stomata generally show much more variability in the atmosp..."
  - `claim-69` gold=REFUTES pred=DISPUTED — "Sea level rise is not going to happen."
  - `claim-2494` gold=DISPUTED pred=SUPPORTS — "There is ample evidence that Earth's average temperature has increased in the past 100 years and the decline of mid- and high-latitude gl..."
- correct non-NEI predictions (sanity):
  - `claim-2257` gold=REFUTES pred=REFUTES — "'Our harmless emissions of trifling quantities of carbon dioxide cannot possibly acidify the oceans."
  - `claim-2627` gold=SUPPORTS pred=SUPPORTS — "Neptune's orbit is 164 years so observations (1950 to present day) span less than a third of a Neptunian year."
  - `claim-2506` gold=SUPPORTS pred=SUPPORTS — "The IPCC confirms that computer modeling predicts the existence of a tropical, mid-troposphere “hot spot” about 10km above the Earth’s su..."

## Track 2 — prompt `v3`  (n=121, Acc=0.2562)

### Predicted vs gold label distribution

| label | gold | predicted | Δ |
|---|---|---|---|
| SUPPORTS | 38 (31.4%) | 28 (23.1%) | -10 |
| REFUTES | 22 (18.2%) | 2 (1.7%) | -20 |
| NOT_ENOUGH_INFO | 40 (33.1%) | 18 (14.9%) | -22 |
| DISPUTED | 21 (17.4%) | 73 (60.3%) | +52 |

### Confusion matrix (rows = gold, columns = pred)

| gold\pred | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED | **total** |
|---|---|---|---|---|---|
| SUPPORTS | **14** | 1 | 5 | 18 | 38 |
| REFUTES | 2 | 0 | 2 | 18 | 22 |
| NOT_ENOUGH_INFO | 6 | 0 | **7** | 27 | 40 |
| DISPUTED | 6 | 1 | 4 | **10** | 21 |
| **total** | 28 | 2 | 18 | 73 | 121 |

### Per-gold-label correctness

| gold label | correct / n | accuracy |
|---|---|---|
| SUPPORTS | 14 / 38 | 0.368 |
| REFUTES | 0 / 22 | 0.000 |
| NOT_ENOUGH_INFO | 7 / 40 | 0.175 |
| DISPUTED | 10 / 21 | 0.476 |

### Evidence recall (predicted ∩ gold) / gold

- macro: 0.1059  (mean over 121 claims with gold ev)
- micro: 0.0956

### Diagnostic flag

- No defaulting-to-NEI pattern detected.

### Sample mispredictions

- non-NEI gold predicted as NEI:
  - `claim-3084` gold=DISPUTED pred=NOT_ENOUGH_INFO — ""the temperature increase in the second half of the 20th century could have taken place in steps driven by major ENSO events" (Jens Rauns..."
  - `claim-1931` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "New Jersey is "losing 50 football fields of open space to development every day and the more we develop upstream the more flooding we hav..."
  - `claim-663` gold=DISPUTED pred=NOT_ENOUGH_INFO — "“For example, Canadian polar bear biologist Ian Stirling learned in the 1970s that spring sea ice in the southern Beaufort Sea periodical..."
- non-NEI gold predicted as a *different* non-NEI label:
  - `claim-2834` gold=SUPPORTS pred=DISPUTED — "When stomata-derived CO2 (red) is compared to ice core-derived CO2 (blue), the stomata generally show much more variability in the atmosp..."
  - `claim-2789` gold=SUPPORTS pred=DISPUTED — "At that time, Hansen also produced a model of the future behavior of the globe’s temperature, which he had turned into a video movie that..."
  - `claim-2257` gold=REFUTES pred=DISPUTED — "'Our harmless emissions of trifling quantities of carbon dioxide cannot possibly acidify the oceans."
- correct non-NEI predictions (sanity):
  - `claim-2627` gold=SUPPORTS pred=SUPPORTS — "Neptune's orbit is 164 years so observations (1950 to present day) span less than a third of a Neptunian year."
  - `claim-583` gold=SUPPORTS pred=SUPPORTS — "But despite [the Gulf Stream], the summer of 2018 looks set to be one of the hottest on record."
  - `claim-1010` gold=SUPPORTS pred=SUPPORTS — "“‘The Arctic may be remote, but changes that occur there directly affect us."

## Track 2 — prompt `v4`  (n=121, Acc=0.2893)

### Predicted vs gold label distribution

| label | gold | predicted | Δ |
|---|---|---|---|
| SUPPORTS | 38 (31.4%) | 46 (38.0%) | +8 |
| REFUTES | 22 (18.2%) | 8 (6.6%) | -14 |
| NOT_ENOUGH_INFO | 40 (33.1%) | 5 (4.1%) | -35 |
| DISPUTED | 21 (17.4%) | 62 (51.2%) | +41 |

### Confusion matrix (rows = gold, columns = pred)

| gold\pred | SUPPORTS | REFUTES | NOT_ENOUGH_INFO | DISPUTED | **total** |
|---|---|---|---|---|---|
| SUPPORTS | **23** | 1 | 1 | 13 | 38 |
| REFUTES | 4 | **3** | 0 | 15 | 22 |
| NOT_ENOUGH_INFO | 9 | 2 | **2** | 27 | 40 |
| DISPUTED | 10 | 2 | 2 | **7** | 21 |
| **total** | 46 | 8 | 5 | 62 | 121 |

### Per-gold-label correctness

| gold label | correct / n | accuracy |
|---|---|---|
| SUPPORTS | 23 / 38 | 0.605 |
| REFUTES | 3 / 22 | 0.136 |
| NOT_ENOUGH_INFO | 2 / 40 | 0.050 |
| DISPUTED | 7 / 21 | 0.333 |

### Evidence recall (predicted ∩ gold) / gold

- macro: 0.1045  (mean over 121 claims with gold ev)
- micro: 0.0931

### Diagnostic flag

- No defaulting-to-NEI pattern detected.

### Sample mispredictions

- non-NEI gold predicted as NEI:
  - `claim-168` gold=SUPPORTS pred=NOT_ENOUGH_INFO — "As the temperature has increased, so has the ability of scientists to determine whether specific events are linked to climate change."
  - `claim-1598` gold=DISPUTED pred=NOT_ENOUGH_INFO — "Adapting to global warming is cheaper than preventing it."
  - `claim-914` gold=DISPUTED pred=NOT_ENOUGH_INFO — "The Clean Power Plan, a major component of fulfilling the agreement, would spike energy costs for working and middle-class Texans by 16% ..."
- non-NEI gold predicted as a *different* non-NEI label:
  - `claim-2834` gold=SUPPORTS pred=DISPUTED — "When stomata-derived CO2 (red) is compared to ice core-derived CO2 (blue), the stomata generally show much more variability in the atmosp..."
  - `claim-2789` gold=SUPPORTS pred=DISPUTED — "At that time, Hansen also produced a model of the future behavior of the globe’s temperature, which he had turned into a video movie that..."
  - `claim-2494` gold=DISPUTED pred=SUPPORTS — "There is ample evidence that Earth's average temperature has increased in the past 100 years and the decline of mid- and high-latitude gl..."
- correct non-NEI predictions (sanity):
  - `claim-2257` gold=REFUTES pred=REFUTES — "'Our harmless emissions of trifling quantities of carbon dioxide cannot possibly acidify the oceans."
  - `claim-69` gold=REFUTES pred=REFUTES — "Sea level rise is not going to happen."
  - `claim-3084` gold=DISPUTED pred=DISPUTED — ""the temperature increase in the second half of the 20th century could have taken place in steps driven by major ENSO events" (Jens Rauns..."
