"""Read the replay output and answer the question the dashboard cannot.

The backtest produces three kinds of evidence and they have very different
evidential weight, so they are reported separately.

* **Measured.** The emissions model's error on dispatches that actually
  happened, against emissions that were actually metered. This is the only
  ground truth in the exercise.
* **Differential.** Whether the model can *rank* two dispatches under matched
  demand and renewables. The optimiser never asks the model for a level; it
  asks which of two schedules is cleaner. A model can be accurate in levels
  and useless at that. The slope of observed differences on predicted
  differences is the factor by which any claimed reduction must be discounted.
* **Model-internal.** The reduction the optimiser reports. This is what the
  live dashboard shows and it is worth exactly as much as the two above allow.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import numpy as np
import pandas as pd
from joblib import load
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import NearestNeighbors

from pipeline.co2_model import total_emission_rate
from pipeline.config import CO2_FEATURES, DECISION_COLUMNS, ROOT as PROJECT_ROOT

BACKTEST_DIR = PROJECT_ROOT / "backtest"
FOLD_CUTOFFS = ["2026-02-04", "2026-03-21", "2026-05-06", "2026-06-21", "2026-08-06"]


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def describe(name, values, unit="", fmt="{:.1f}"):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    q = np.percentile(v, [10, 25, 50, 75, 90])
    line = "  ".join(fmt.format(x) for x in q)
    print(f"  {name:38s} mean {fmt.format(v.mean()):>9s}{unit}   p10/25/50/75/90: {line}")


# --- the differential test ----------------------------------------------------


def counterfactual_ranking(
    frame, cutoffs, ends, tol=75.0, min_move=400.0, max_partners=8, seed=3, direction=None
):
    """Can the model rank two dispatches observed under matched conditions?

    Pairs are matched on demand and renewables, so the balance identity forces
    the two dispatch vectors to sum alike: what differs between them is purely
    how the same load was served. That is exactly the comparison the genetic
    algorithm makes thousands of times per generation, and unlike the
    optimiser's own proposals these pairs both have a metered outcome.

    Matching is never perfect -- two moments with identical demand and wind
    can still differ in what the neighbours were burning, which the feature
    set cannot see. The slope recovered here is therefore an upper bound on
    how well *any* model of these features could rank dispatches.
    """
    rng = np.random.default_rng(seed)
    blocks = []

    for k, (cutoff, end) in enumerate(zip(cutoffs, ends), start=1):
        bundle = load(BACKTEST_DIR / f"fold_{k}" / "fold.joblib")
        model, marginal = bundle["co2"], bundle["marginal"]

        test = frame[(frame["timestamp"] >= cutoff) & (frame["timestamp"] < end)].reset_index(drop=True)
        observed = total_emission_rate(
            test["CO2Emission"].to_numpy(float), test["Demand"].to_numpy(float)
        )
        predicted = model.predict_total(test[CO2_FEATURES].to_numpy(float))
        dispatch = test[DECISION_COLUMNS].to_numpy(float)
        times = test["timestamp"].to_numpy()

        # Scale first, then search the unit ball: both conditions must agree
        # to within `tol` for the pair to count as matched.
        scaled = test[["Demand", "Renewables"]].to_numpy(float) / tol
        neighbours = NearestNeighbors(radius=1.0).fit(scaled).radius_neighbors(
            scaled, return_distance=False
        )

        left, right = [], []
        for a, partners in enumerate(neighbours):
            partners = partners[partners > a]
            if len(partners) > max_partners:
                partners = rng.choice(partners, max_partners, replace=False)
            left.append(np.full(len(partners), a))
            right.append(partners)

        a = np.concatenate(left)
        b = np.concatenate(right)
        move = np.abs(dispatch[a] - dispatch[b]).sum(axis=1)
        keep = move >= min_move
        a, b, move = a[keep], b[keep], move[keep]

        difference = dispatch[a] - dispatch[b]
        block = pd.DataFrame(
            {
                "fold": k,
                "observed_delta": observed[a] - observed[b],
                "predicted_delta": predicted[a] - predicted[b],
                "marginal_delta": difference @ marginal[1:],
                "move_mw": move,
                "days_apart": np.abs((times[a] - times[b]) / np.timedelta64(1, "D")),
            }
        )
        if direction is not None:
            norm = np.linalg.norm(difference, axis=1) * np.linalg.norm(direction)
            block["alignment"] = np.divide(
                difference @ direction, norm, out=np.zeros(len(difference)), where=norm > 1e-9
            )
        blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


def plausibility_floor(frame, steps, cutoffs, ends, tol=200.0, min_neighbours=25):
    """Has the grid ever been as clean as the optimiser proposes?

    The Mahalanobis guard asks whether an operating *point* resembles history.
    This asks a blunter question of the *outcome*: under demand and renewables
    like these, what is the lowest emission rate ever metered? A proposal
    below that floor is an extrapolation the data cannot support, however
    ordinary its coordinates look.
    """
    out = []
    for k, (cutoff, end) in enumerate(zip(cutoffs, ends), start=1):
        train = frame[frame["timestamp"] < cutoff]
        observed = total_emission_rate(
            train["CO2Emission"].to_numpy(float), train["Demand"].to_numpy(float)
        )
        reference = train[["Demand", "Renewables"]].to_numpy(float) / tol

        fold_steps = steps[steps["fold"] == k]
        if fold_steps.empty:
            continue
        query = fold_steps[["demand_real", "renewables_real"]].to_numpy(float) / tol
        neighbours = NearestNeighbors(radius=1.0).fit(reference).radius_neighbors(
            query, return_distance=False
        )

        for j, nb in enumerate(neighbours):
            if len(nb) < min_neighbours:
                continue
            values = observed[nb]
            row = fold_steps.iloc[j]
            out.append(
                {
                    "fold": k,
                    "optimised": row["model_total_opt"],
                    "realised_model": row["model_total_real"],
                    "observed": row["observed_total"],
                    "hist_min": values.min(),
                    "hist_p05": np.percentile(values, 5),
                    "hist_median": np.median(values),
                    "n_neighbours": len(nb),
                }
            )
    return pd.DataFrame(out)


def slope_and_skill(observed, predicted, label):
    """Robust slope of observed on predicted, plus sign agreement."""
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    keep = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[keep], predicted[keep]

    huber = HuberRegressor(fit_intercept=True).fit(predicted[:, None], observed)
    slope = float(huber.coef_[0])

    corr = float(np.corrcoef(predicted, observed)[0, 1])
    agree = float(np.mean(np.sign(predicted) == np.sign(observed)))
    # Fraction of the observed spread the prediction accounts for, using the
    # prediction as-is rather than a refitted line.
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))

    print(
        f"  {label:34s} slope={slope:5.2f}  r={corr:5.2f}  "
        f"sign agreement={100 * agree:4.1f}%  R2(as-is)={1 - ss_res / ss_tot:6.2f}  n={len(observed)}"
    )
    return {"slope": slope, "r": corr, "sign_agreement": agree, "n": int(len(observed))}


def main():
    periods = pd.read_parquet(BACKTEST_DIR / "periods.parquet")
    steps = pd.read_parquet(BACKTEST_DIR / "steps.parquet")
    frame = pd.read_parquet(BACKTEST_DIR / "frame.parquet")

    if "error" in periods:
        failed = periods["error"].notna().sum()
        periods = periods[periods["error"].isna()].copy()
        print(f"dropped {failed} failed origins")

    summary = {"n_periods": int(len(periods)), "n_steps": int(len(steps))}
    print(f"{len(periods)} replayed origins, {len(steps)} horizon steps")
    print(f"window: {periods['timestamp'].min()} -> {periods['timestamp'].max()}")

    # --- 1. measured: does the model track reality on real dispatch? ----------
    section("1. Emissions model against metered outcomes (out-of-sample)")
    err = steps["model_total_real"] - steps["observed_total"]
    describe("signed error (t/h)", err, fmt="{:+.1f}")
    describe("absolute error (t/h)", err.abs())
    rel = 100 * err.abs() / steps["observed_total"].clip(lower=1e-6)
    describe("absolute error (% of observed)", rel)
    r2 = 1 - np.sum(err**2) / np.sum((steps["observed_total"] - steps["observed_total"].mean()) ** 2)
    print(f"  R2 on realised dispatch: {r2:.3f}")
    summary["model_bias_tph"] = float(err.mean())
    summary["model_mae_tph"] = float(err.abs().mean())
    summary["model_r2"] = float(r2)

    # --- 2. differential: can it rank counterfactual dispatches? --------------
    section("2. Can the model rank two dispatches under matched conditions?")
    cutoffs = [pd.Timestamp(c, tz="UTC") for c in FOLD_CUTOFFS]
    ends = cutoffs[1:] + [frame["timestamp"].iloc[-1]]

    direction = periods[[f"delta_{c}" for c in DECISION_COLUMNS]].mean().to_numpy(float)
    pairs = counterfactual_ranking(frame, cutoffs, ends, direction=direction)

    # A pair is unordered, so orient each one the way the optimiser moves.
    # Slope is invariant to flipping both sides, so this only changes which
    # pairs count as "the same kind of move".
    flip = np.where(pairs["alignment"] < 0, -1.0, 1.0)
    for column in ["observed_delta", "predicted_delta", "marginal_delta", "alignment"]:
        pairs[column] *= flip

    print(f"  {len(pairs)} matched pairs (demand and renewables within 75 MW, dispatch differing >400 MW)")
    print("  A slope of 1 means a predicted saving of X t/h delivers X t/h.\n")
    summary["ranking_all"] = slope_and_skill(pairs["observed_delta"], pairs["predicted_delta"], "monotone ensemble")
    summary["ranking_marginal"] = slope_and_skill(
        pairs["observed_delta"], pairs["marginal_delta"], "empirical marginal factors"
    )

    near = pairs[pairs["days_apart"] <= 7]
    print(f"\n  restricted to pairs within 7 days ({len(near)} pairs), which controls for seasonality:")
    summary["ranking_near"] = slope_and_skill(near["observed_delta"], near["predicted_delta"], "monotone ensemble")
    summary["ranking_near_marginal"] = slope_and_skill(
        near["observed_delta"], near["marginal_delta"], "empirical marginal factors"
    )

    big = pairs[pairs["move_mw"] > pairs["move_mw"].quantile(0.75)]
    print(f"\n  restricted to the largest dispatch differences ({len(big)} pairs):")
    summary["ranking_large"] = slope_and_skill(big["observed_delta"], big["predicted_delta"], "monotone ensemble")

    # The aggregate slope averages over every kind of re-dispatch the grid
    # happened to perform. What matters is skill along the direction the
    # optimiser actually pushes: less domestic thermal, more low-carbon import.
    aligned = pairs[pairs["alignment"] > 0.5]
    print(
        f"\n  restricted to pairs resembling the optimiser's own move "
        f"({len(aligned)} pairs, cosine > 0.5):"
    )
    summary["ranking_aligned"] = slope_and_skill(
        aligned["observed_delta"], aligned["predicted_delta"], "monotone ensemble"
    )
    summary["ranking_aligned_marginal"] = slope_and_skill(
        aligned["observed_delta"], aligned["marginal_delta"], "empirical marginal factors"
    )

    optimiser_move = periods[[f"delta_{c}" for c in DECISION_COLUMNS]].abs().sum(axis=1)
    inside = 100 * (pairs["move_mw"] >= optimiser_move.median()).mean()
    print(
        f"\n  the optimiser's median move is {optimiser_move.median():.0f} MW (L1); "
        f"{inside:.0f}% of observed pairs differ by at least that much,"
    )
    print("  so the slope is measured over a range the grid genuinely explored.")
    summary["optimiser_move_mw"] = float(optimiser_move.median())
    summary["pairs_at_least_as_large_pct"] = float(inside)

    # --- 3. what the live system would have claimed ---------------------------
    section("3. The reduction the live pipeline would have reported")
    # Per-period ratios are unusable here: the optimiser drives some baselines
    # close to zero, and a handful even negative, so the mean of the ratios is
    # meaningless. Pooling energy across every horizon step is the robust form
    # and is also the quantity an operator would care about.
    pooled = 100 * (1 - steps["model_total_opt"].sum() / steps["model_total_real"].sum())
    pooled_hold = 100 * (1 - steps["model_total_opt"].sum() / steps["model_total_hold"].sum())
    print(f"  pooled over all {len(steps)} horizon steps:")
    print(f"    against the hold-current counterfactual: {pooled_hold:5.1f}%")
    print(f"    against the dispatch that actually ran:  {pooled:5.1f}%")
    describe("saving vs what actually happened (t/h)", periods["reduction_vs_real_tph"])
    describe("reported reduction, per period (%)", periods["reported_reduction_pct"].clip(-200, 200))

    negative = 100 * (steps["model_total_opt"] < 0).mean()
    print(f"\n  optimised steps for which the model predicts NEGATIVE emissions: {negative:.1f}%")
    if "within_floor_pct" in periods:
        describe("steps inside the historical floor (%)", periods["within_floor_pct"])

    summary["pooled_reduction_vs_real_pct"] = float(pooled)
    summary["pooled_reduction_vs_hold_pct"] = float(pooled_hold)
    summary["reduction_vs_real_tph"] = float(periods["reduction_vs_real_tph"].mean())
    summary["negative_prediction_pct"] = float(negative)

    mae = (steps["model_total_real"] - steps["observed_total"]).abs().mean()
    ratio = periods["reduction_vs_real_tph"].mean() / mae
    print(f"\n  claimed saving is {ratio:.1f}x the model's own error on realised dispatch")
    summary["claim_to_error_ratio"] = float(ratio)

    print("\n  the ranking slope says a predicted saving is only partly delivered, so:")
    for name, key in [
        ("all pairs", "ranking_all"),
        ("pairs within 7 days", "ranking_near"),
        ("pairs resembling the optimiser's move", "ranking_aligned"),
    ]:
        s = summary[key]["slope"]
        print(
            f"    discounted by the slope from {name:38s} ({s:4.2f}): "
            f"{periods['reduction_vs_real_tph'].mean() * s:6.1f} t/h "
            f"({pooled * s:4.1f}% pooled)"
        )

    # --- 4. where does the reduction come from? -------------------------------
    section("4. Attribution: domestic generation or cross-border re-labelling?")
    total = periods["delta_total_tph"]
    describe("total change (t/h)", total, fmt="{:+.1f}")
    describe("from domestic generation (t/h)", periods["delta_gen_tph"], fmt="{:+.1f}")
    describe("from cross-border schedule (t/h)", periods["delta_xb_tph"], fmt="{:+.1f}")
    share = 100 * periods["delta_xb_tph"] / (periods["delta_gen_tph"] + periods["delta_xb_tph"])
    describe("cross-border share of the saving (%)", share.clip(-200, 300))
    print()
    describe("domestic generation moved (MWh/4h)", periods["gen_energy_delta_mwh"], fmt="{:+.0f}")
    describe("net imports moved (MWh/4h)", periods["xb_energy_delta_mwh"], fmt="{:+.0f}")
    summary["delta_gen_tph"] = float(periods["delta_gen_tph"].mean())
    summary["delta_xb_tph"] = float(periods["delta_xb_tph"].mean())
    summary["gen_energy_delta_mwh"] = float(periods["gen_energy_delta_mwh"].mean())
    summary["xb_energy_delta_mwh"] = float(periods["xb_energy_delta_mwh"].mean())

    # --- 5. does the schedule survive the forecast being wrong? ---------------
    section("5. Feasibility once the forecast is replaced by what happened")
    by_step = steps.groupby("step").agg(
        balance_mw=("balance_real", lambda s: s.abs().mean()),
        demand=("demand_real", "mean"),
    )
    by_step["balance_pct"] = 100 * by_step["balance_mw"] / by_step["demand"]
    print("  mean |imbalance| of the optimised schedule under realised conditions:")
    for step in [1, 2, 4, 8, 12, 16]:
        row = by_step.loc[step]
        flag = "" if row["balance_pct"] <= 1.0 else "   <- outside the 1% tolerance"
        print(f"    step {step:2d} ({step * 15:3d} min): {row['balance_mw']:6.1f} MW  ({row['balance_pct']:4.2f}%){flag}")
    within = 100 * (steps["balance_real"].abs() <= 0.01 * steps["demand_real"]).mean()
    print(f"\n  individual steps meeting the 1% balance tolerance: {within:.0f}%")
    print("  The residual is the forecast error, which a real operator covers with")
    print("  regulating reserve -- typically the fastest and dirtiest plant on the")
    print("  system, and an emissions cost this model does not account for.")
    describe("demand forecast MAE (MW)", periods["demand_mae"])
    describe("renewables forecast MAE (MW)", periods["renewables_mae"])
    summary["steps_within_tolerance_pct"] = float(within)
    summary["balance_step1_pct"] = float(by_step.loc[1, "balance_pct"])
    summary["balance_step16_pct"] = float(by_step.loc[16, "balance_pct"])

    # --- 6. is the in-distribution guard doing anything? ----------------------
    section("6. Is the in-distribution guard discriminating?")
    describe("optimised points judged in-distribution (%)", periods["opt_in_dist_pct"])
    describe("realised points judged in-distribution (%)", periods["real_in_dist_pct"])
    describe("optimised novelty (Mahalanobis^2)", periods["opt_novelty_median"], fmt="{:.1f}")
    describe("realised novelty (Mahalanobis^2)", periods["real_novelty_median"], fmt="{:.1f}")

    q = pd.qcut(steps["real_novelty"], 5, duplicates="drop")
    bins = steps.assign(bin=q, abs_err=(steps["model_total_real"] - steps["observed_total"]).abs())
    print("\n  model error against novelty, on realised points:")
    for name, group in bins.groupby("bin", observed=True):
        print(f"    novelty {name.left:7.1f}-{name.right:7.1f}: MAE {group['abs_err'].mean():6.1f} t/h  (n={len(group)})")
    above_median = 100 * np.mean(
        steps["opt_novelty"].to_numpy() > np.median(steps["real_novelty"].to_numpy())
    )
    print(f"\n  {above_median:.0f}% of optimised points are more novel than the median realised point")

    section("6b. Has the grid ever been as clean as the optimiser proposes?")
    floor = plausibility_floor(frame, steps, cutoffs, ends)
    print(f"  {len(floor)} optimised steps had 25+ historical matches on demand and renewables")
    below_min = 100 * (floor["optimised"] < floor["hist_min"]).mean()
    below_p05 = 100 * (floor["optimised"] < floor["hist_p05"]).mean()
    print(f"  below the lowest emission rate ever metered under such conditions: {below_min:.1f}%")
    print(f"  below the 5th percentile of those conditions:                      {below_p05:.1f}%")
    describe("optimised (t/h)", floor["optimised"])
    describe("historical minimum for those conditions (t/h)", floor["hist_min"])
    describe("historical median for those conditions (t/h)", floor["hist_median"])
    describe("actually metered (t/h)", floor["observed"])
    summary["below_historical_min_pct"] = float(below_min)
    summary["below_historical_p05_pct"] = float(below_p05)

    # Credit the optimiser only down to the cleanest hour the grid has ever
    # actually achieved under comparable demand and wind. Everything below
    # that is the model extrapolating past its evidence, and the ranking test
    # of section 2 says nothing about that region.
    credited = np.maximum(floor["optimised"], floor["hist_p05"])
    raw_pct = 100 * (1 - floor["optimised"].sum() / floor["realised_model"].sum())
    capped_pct = 100 * (1 - credited.sum() / floor["realised_model"].sum())
    slope = summary["ranking_aligned"]["slope"]
    print(f"\n  reduction claimed on these steps:              {raw_pct:5.1f}%")
    print(f"  after capping at the historical 5th percentile: {capped_pct:5.1f}%")
    print(f"  and after the {slope:.2f} ranking discount:              {capped_pct * slope:5.1f}%")
    summary["capped_reduction_pct"] = float(capped_pct)
    summary["capped_discounted_pct"] = float(capped_pct * slope)

    # --- 7. by fold -----------------------------------------------------------
    section("7. Stability across the year")
    fold_view = periods.groupby("fold").agg(
        n=("origin", "size"),
        start=("timestamp", "min"),
        saving_tph=("reduction_vs_real_tph", "mean"),
        model_mae=("model_abs_error_tph", "mean"),
        xb_tph=("delta_xb_tph", "mean"),
        gen_tph=("delta_gen_tph", "mean"),
    )
    fold_view["start"] = fold_view["start"].dt.strftime("%Y-%m-%d")
    pooled_by_fold = steps.groupby("fold").apply(
        lambda g: 100 * (1 - g["model_total_opt"].sum() / g["model_total_real"].sum()),
        include_groups=False,
    )
    fold_view["pooled_pct"] = pooled_by_fold
    fold_view["negative_pct"] = 100 * steps.groupby("fold")["model_total_opt"].apply(lambda s: (s < 0).mean())
    print(fold_view.round(1).to_string())

    (BACKTEST_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {BACKTEST_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
