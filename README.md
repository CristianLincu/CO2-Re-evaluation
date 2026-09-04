# Reducing CO₂ Levels Using Genetic Algorithms — A Re-evaluation

A carbon-aware dispatch optimiser for Denmark's power grid: a monotone emissions
model, gradient-boosted demand and renewables forecasts, and a genetic algorithm
searching over four-hour dispatch trajectories.

**[Live dashboard](https://cristianlincu.github.io/CO2-Re-evaluation/)** ·
**[Paper (PDF)](paper/CO2_Reevaluation.pdf)**

This is a rebuild of [an earlier
framework](https://github.com/CristianLincu/Reducing-CO2-via-Genetic-Algorithms)
that reported 50–75% emission reductions. Those figures did not survive scrutiny.
The structure of the original — a regression model inferring CO₂ levels, XGBoost
regressors forecasting demand and renewables, and a genetic ensemble searching for
the emission-minimising distribution — is preserved. The errors are not.

## What was wrong, and what changed

| Issue | Correction |
| --- | --- |
| The optimiser exploited its own surrogate. An unconstrained tree fitted to wind-confounded data reported that emissions *fall* as thermal generation rises, so the search ramped thermal plants. | The emissions model predicts **total** emissions rather than intensity and is constrained monotone non-decreasing in every supply source, which is physically true under consumption-based accounting. |
| A fully grown tree with 25,863 leaves is a lookup table, not a model of dispatch. | Depth- and leaf-capped gradient-boosted ensemble. |
| Nothing stopped the optimiser evaluating combinations the grid has never been in. | Leaf-support and Mahalanobis novelty penalties in the fitness function. |
| `train_test_split(shuffle=True)` on a time series puts near-duplicate neighbours on both sides of the split. | Chronological splits and `TimeSeriesSplit` throughout, with persistence baselines. |
| Cosine similarity was used to keep consecutive solutions coherent, but it is scale invariant. Deployed solutions swung 1,170 MW on one interconnector between adjacent steps. | Explicit per-resource ramp limits, derived from observed 15-minute changes. |
| `Exchange_DK1_DK2` was counted in the national balance, but it joins two Danish bidding zones and cannot change national supply. | Restricted to the eight cross-border links, matching Energinet's own `Exchange_Sum`. |
| Forecast CO₂ was overwritten with the current value whenever it looked worse, so the dashboard could not show a bad result. | Removed. The counterfactual is published next to the optimised result. |
| Reduction was measured against *present* emissions, crediting the optimiser with whatever the wind was going to do. | Measured against a hold-current counterfactual over the same horizon. |
| Each forecast step was optimised independently. | A chromosome is the whole trajectory, with steps coupled by ramp constraints. |
| Bounds were hard-coded and stale; the deployed GA also ran with selection pressure 150 where the validated notebook used 8, collapsing diversity. | Bounds and ramp limits re-derived from the training year; hyperparameters set in one place. |

Two silent failures are worth recording. Energinet's `PowerSystemRightNow` schema
changed after the original models were trained — `ImbalanceDK1`, `ImbalanceDK2` and
`mFRRActivated*` no longer exist — and the original notebooks selected features by
position, so the same code now reads different columns. All selection here is by
name. And renewables were extrapolated from their own lags, which cannot work at a
four-hour horizon; the model now post-processes Energinet's physical forecast.

## Results

| Model | Evaluation | Intensity MAE | Monotone |
| --- | --- | --- | --- |
| Fully grown tree on intensity | shuffled split | 5.77 g/kWh | no |
| Fully grown tree on intensity | chronological | 13.01 g/kWh | no |
| Monotone ensemble on total emissions | chronological | **11.05 g/kWh** | yes |

The constrained model is *more* accurate than the original was under honest
evaluation, and it cannot be gamed: sweeping large-plant output from 0 to 1500 MW,
the original model's predicted emissions bottom out at 300 MW, while the
constrained model's minimum is at zero.

Forecast RMSE at four hours ahead: demand 374 MW against 816 MW for persistence;
renewables 469 MW against 1394 MW for persistence and 700 MW for the raw Energinet
forecast.

At matched evaluation budget and wall-clock time the genetic ensemble reaches
60.8 g/kWh against 87.7 g/kWh for random search, and keeps every step inside the
region where the emissions model has support.

## How it works

Decisions are made on the 15-minute grid Denmark uses for imbalance settlement,
over a four-hour horizon — 16 steps of 11 resources, so 176 coupled decision
variables. A chromosome is an entire trajectory. Fitness combines a grid-imbalance
penalty, a ramp penalty, an out-of-distribution penalty, the inferred emissions
level and the model's own dispersion.

Three things keep a solve near 60 seconds:

- the ensemble of 16 optimisers is evaluated as one batched array (an island model
  with periodic migration) rather than 16 sequential runs;
- a repair operator enforces bounds, ramp limits and the energy balance by
  construction, so the search spends its effort on emissions instead of
  rediscovering feasibility;
- emissions, dispersion, leaf support and novelty all come from a single traversal
  per generation.

## Layout

```
pipeline/     config, data access, models, optimiser, export, entry point
scripts/      dataset build, training, limit derivation, figures
models/       trained artefacts and their validation metrics
docs/         the GitHub Pages dashboard
paper/        LaTeX source, figures, compiled PDF
```

## Running it

```bash
pip install -r requirements.txt
python pipeline/run.py          # solve now, write docs/data/latest.json
```

To retrain from scratch:

```bash
python scripts/build_dataset.py   # one year of Energinet data, cached locally
python scripts/train_co2.py
python scripts/train_forecast.py
python scripts/derive_limits.py
python scripts/make_figures.py
```

`.github/workflows/update-data.yml` runs the pipeline every 30 minutes and commits
the refreshed payload.

If you are behind TLS-inspecting antivirus, point `REQUESTS_CA_BUNDLE` at a bundle
containing its root certificate; the data layer honours it.

## Limitations

Unit commitment is not modelled — large-plant output is a single aggregate where
reality is dozens of units with minimum stable loads and start-up times.
Interconnector limits are fixed over the horizon rather than read from published
per-market-time-unit capacities. Prices and contract terms are absent, so a
schedule may be physically feasible and economically impossible.

Most importantly, Energinet's CO₂ figure is a **consumption-based** intensity: it
assigns emission factors to imports. A schedule can lower the Danish number by
importing more low-carbon Nordic power without lowering emissions anywhere, since
hydro export capacity is finite and the displaced consumer may burn gas instead.
What is reported here is a reduction in Denmark's *attributed* intensity.
Establishing how much is genuine abatement needs marginal emissions data for the
surrounding system. This caveat applied equally to the original framework, where it
was not stated.

## Author

Cristian Lincu
