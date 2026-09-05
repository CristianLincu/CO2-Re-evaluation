# Reducing CO₂ Levels Using Genetic Algorithms

A carbon-aware dispatch optimiser for Denmark's power grid: a monotone
emissions model, gradient-boosted demand and renewables forecasts, and a
genetic algorithm searching over four-hour dispatch trajectories.

**[Live dashboard](https://cristianlincu.github.io/CO2-Re-evaluation/)** ·
**[Paper (PDF)](paper/CO2_Reevaluation.pdf)**

**Author:** Cristian Lincu

## What it does

Every thirty minutes the pipeline pulls Energinet's latest power-system
records, forecasts demand and renewable production four hours ahead, and
searches for a dispatch trajectory that minimises Denmark's
consumption-based CO₂ intensity. The search is a 16-island genetic
algorithm. A chromosome is the whole trajectory, not a single time
point, so consecutive steps are coupled by ramp limits. Candidates are
repaired onto the feasible set — bounds, ramps, and the national energy
balance — before they are scored.

The emissions model predicts the *total* emission rate and is constrained
monotone non-decreasing in every supply source. Softplus keeps the
prediction above zero. A plausibility floor — the 5th percentile of
metered outcomes under comparable demand and wind — stops the search
proposing an emission rate the grid has never achieved.

The headline reduction is measured against a hold-current
counterfactual: the same forecasts, with the controllable resources left
where they are.

## Backtest

A rolling-origin replay of 225 origins across five expanding training
windows (February–September 2026) is the evidence for those claims.
Every model in a fold is fitted only on data preceding its test window.

| Quantity | Result |
| --- | --- |
| Emissions-model $R^2$ on realised dispatch | 0.941 |
| Ranking slope (predicted vs metered difference) | 0.80 ($r = 0.83$) |
| Pipeline claim vs the dispatch that ran | 35.8% |
| After the historical-floor cap and ranking discount | 27.2% |
| Share from less domestic generation | 13% |
| Share from cross-border reallocation | 85% |
| Negative predicted emissions | 0% |
| Steps still balanced under realised weather | 13% |

The ranking test matches historical hours on demand and wind, so the
balance identity forces both members of a pair to serve the same load.
When the model says one schedule is $X$ t/h cleaner, the meters deliver
about $0.80X$. Most of the remaining reduction is a reallocation of
imports toward Norway and Sweden, not a reduction in Danish combustion.
The schedule balances only against its own forecast; under the weather
that materialised the imbalance grows from 1.9% of demand at 15 minutes
to 10.5% at four hours.

The full write-up is in the paper, Section 5, and on the dashboard.

## Layout

```
pipeline/     data access, models, optimiser, dashboard export
scripts/      training, backtest, figures
models/       fitted artefacts served by the live job
paper/        LaTeX source and figures
docs/         GitHub Pages dashboard
backtest/     replay outputs (summary.json, analysis.txt)
```

## Running

```
python scripts/build_dataset.py
python scripts/derive_limits.py
python scripts/train_co2.py
python scripts/train_forecast.py
python pipeline/run.py
python scripts/backtest.py --periods 45 --workers 12
python scripts/analyze_backtest.py
```

A GitHub Actions workflow refreshes `docs/data/latest.json` every thirty
minutes. The historical backtest is a static artefact; it is not
recomputed on that schedule.

## Limitations

The framework does not model unit commitment, market prices, or
time-varying transfer capacities. Cross-border flows are outcomes of
coupled European market clearing, not dials a single operator turns.
Energinet's CO₂ figure is consumption-based, so a cleaner Danish number
can be a reallocation of Nordic hydro rather than abatement. The
backtest measures what the emissions model says about a hypothetical
re-dispatch; it does not establish that anyone could have executed it.
