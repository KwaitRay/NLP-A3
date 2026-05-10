# Climate Fact-Check — Claim-side EDA

> No evidence.json needed. Pure claim metadata.

## Sizes

- train: **1228** claims
- dev:   **154** claims
- test:  **153** claims (unlabelled)

## Claim length (whitespace-tokenised)

| split | n | min | max | mean | median | stdev |
|---|---|---|---|---|---|---|
| train | 1228 | 4 | 67 | 20.1 | 19 | 9.23 |
| dev | 154 | 4 | 65 | 21.08 | 18 | 11.32 |
| test | 153 | 4 | 53 | 20.04 | 19 | 10.17 |

## Label distribution

### train

| label | count | % |
|---|---|---|
| SUPPORTS | 519 | 42.3% |
| REFUTES | 199 | 16.2% |
| NOT_ENOUGH_INFO | 386 | 31.4% |
| DISPUTED | 124 | 10.1% |

### dev

| label | count | % |
|---|---|---|
| SUPPORTS | 68 | 44.2% |
| REFUTES | 27 | 17.5% |
| NOT_ENOUGH_INFO | 41 | 26.6% |
| DISPUTED | 18 | 11.7% |

## Gold evidence count per claim

### train

| label | n | min | max | mean | median |
|---|---|---|---|---|---|
| SUPPORTS | 519 | 1 | 5 | 2.59 | 2 |
| REFUTES | 199 | 1 | 5 | 2.3 | 2 |
| NOT_ENOUGH_INFO | 386 | 5 | 5 | 5 | 5 |
| DISPUTED | 124 | 2 | 5 | 3.16 | 3 |

### dev

| label | n | min | max | mean | median |
|---|---|---|---|---|---|
| SUPPORTS | 68 | 1 | 5 | 2.51 | 2 |
| REFUTES | 27 | 1 | 4 | 2.11 | 2 |
| NOT_ENOUGH_INFO | 41 | 5 | 5 | 5 | 5 |
| DISPUTED | 18 | 2 | 5 | 3.22 | 3 |

## Sample claims (first 3 per label, train split)

### SUPPORTS

- `claim-2510` | n_ev=2 | ev_sample=[evidence-530063, evidence-984887]
  > In 1946, PDO switched to a cool phase.
- `claim-2834` | n_ev=1 | ev_sample=[evidence-439640]
  > When stomata-derived CO2 (red) is compared to ice core-derived CO2 (blue), the stomata generally show much more variability in the atmospheric CO2 level and often show levels much higher than the ice cores.
- `claim-1441` | n_ev=1 | ev_sample=[evidence-217743]
  > 195 countries signed the 2015 Paris Agreement, agreeing to limit global warming and adapt to climate change, partly by protecting nature.

### REFUTES

- `claim-126` | n_ev=2 | ev_sample=[evidence-338219, evidence-1127398]
  > El Niño drove record highs in global temperatures suggesting rise may not be down to man-made emissions.
- `claim-2152` | n_ev=5 | ev_sample=[evidence-515817, evidence-1018575, evidence-791159]
  > Venus doesn't have a runaway greenhouse effect
- `claim-3003` | n_ev=3 | ev_sample=[evidence-175982, evidence-515817, evidence-1009205]
  > Venus is not hot because of a runaway greenhouse.

### NOT_ENOUGH_INFO

- `claim-2449` | n_ev=5 | ev_sample=[evidence-1010750, evidence-91661, evidence-722725]
  > "January 2008 capped a 12 month period of global temperature drops on all of the major well respected indicators.
- `claim-851` | n_ev=5 | ev_sample=[evidence-226174, evidence-1049316, evidence-358301]
  > The last time the planet was even four degrees warmer, Peter Brannen points out in The Ends of the World, his new history of the planet’s major extinction events, the oceans were hundreds of feet higher.
- `claim-1019` | n_ev=5 | ev_sample=[evidence-863309, evidence-61462, evidence-639818]
  > An additional kick was supplied by an El Niño weather pattern that peaked in 2016 and temporarily warmed much of the surface of the planet, causing the hottest year in a historical record dating to 1880.

### DISPUTED

- `claim-1937` | n_ev=3 | ev_sample=[evidence-442946, evidence-1194317, evidence-12171]
  > Not only is there no scientific evidence that CO2 is a pollutant, higher CO2 concentrations actually help ecosystems support more plant and animal life.
- `claim-2021` | n_ev=5 | ev_sample=[evidence-1177431, evidence-782448, evidence-540069]
  > Weather Channel co-founder John Coleman provided evidence that convincingly refutes the concept of anthropogenic global warming.
- `claim-2773` | n_ev=2 | ev_sample=[evidence-974673, evidence-602109]
  > Tree-ring proxy reconstructions are reliable before 1960, tracking closely with the instrumental record and other independent proxies.
