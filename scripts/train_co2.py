"""Train the CO2 model and benchmark it against the original formulation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor

from pipeline.co2_model import (
    build_safe_model,
    intensity_from_total,
    make_monotone_ensemble,
    make_partition_tree,
    total_emission_rate,
)
from pipeline.config import CACHE_DIR, CO2_FEATURES, MODELS_DIR

PARAM_SPACE = {
    "n_estimators": [300, 400, 600],
    "learning_rate": [0.03, 0.05, 0.08],
    "max_depth": [4, 6, 8],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [5, 20, 50],
    "reg_lambda": [1.0, 5.0, 20.0],
}


def thermal_ramp_probe(predict_total, base_row, base_demand):
    """Does the model reward ramping thermal generation?

    Holds every other resource fixed and sweeps large-plant output. A model
    fit to confounded data reports emissions *falling* as thermal output
    rises, which is the relationship the original optimiser exploited.
    """
    idx = CO2_FEATURES.index("ProductionGe100MW")
    sweep = np.linspace(0.0, 1500.0, 31)

    rows = np.repeat(np.asarray(base_row, dtype=float)[None, :], len(sweep), axis=0)
    rows[:, idx] = sweep

    totals = np.asarray(predict_total(rows), dtype=float)
    demand = base_demand + (sweep - float(base_row[idx]))

    return {
        "thermal_mw": sweep.tolist(),
        "total_tph": totals.tolist(),
        "intensity_g_per_kwh": intensity_from_total(totals, demand).tolist(),
        "monotone_non_decreasing": bool(np.all(np.diff(totals) >= -1e-6)),
        "total_change_tph": float(totals[-1] - totals[0]),
        "min_at_mw": float(sweep[int(np.argmin(totals))]),
    }


def main():
    frame = pd.read_parquet(CACHE_DIR / "training_frame.parquet")
    frame = frame.dropna(subset=CO2_FEATURES + ["CO2Emission", "Demand"]).reset_index(drop=True)
    frame = frame[frame["Demand"] > 500].reset_index(drop=True)

    X = frame[CO2_FEATURES].to_numpy(dtype=float)
    demand = frame["Demand"].to_numpy(dtype=float)
    intensity = frame["CO2Emission"].to_numpy(dtype=float)
    y = total_emission_rate(intensity, demand)

    print(f"samples={len(X)}  features={len(CO2_FEATURES)}")
    print(f"period: {frame['timestamp'].min()} -> {frame['timestamp'].max()}")
    print(f"intensity g/kWh: min={intensity.min():.1f} mean={intensity.mean():.1f} max={intensity.max():.1f}")

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    d_test = demand[split:]
    i_train, i_test = intensity[:split], intensity[split:]
    print(f"chronological split: train={len(X_train)} test={len(X_test)}")

    # --- deployed estimator ---------------------------------------------------
    print("\ntuning monotone ensemble...")
    search = RandomizedSearchCV(
        estimator=make_monotone_ensemble(),
        param_distributions=PARAM_SPACE,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=4),
        random_state=9,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    print(f"best params: {search.best_params_}")

    ensemble = make_monotone_ensemble(**search.best_params_)
    ensemble.fit(X_train, y_train)

    pred_total = ensemble.predict(X_test)
    pred_intensity = intensity_from_total(pred_total, d_test)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_period": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[split - 1])],
        "test_period": [str(frame["timestamp"].iloc[split]), str(frame["timestamp"].iloc[-1])],
        "best_params": search.best_params_,
        "total_rmse_tph": float(root_mean_squared_error(y_test, pred_total)),
        "total_mae_tph": float(mean_absolute_error(y_test, pred_total)),
        "total_r2": float(r2_score(y_test, pred_total)),
        "intensity_rmse": float(root_mean_squared_error(i_test, pred_intensity)),
        "intensity_mae": float(mean_absolute_error(i_test, pred_intensity)),
    }
    print("\nmonotone ensemble (deployed):")
    print(f"  total     R2={metrics['total_r2']:.4f}  RMSE={metrics['total_rmse_tph']:.2f} t/h")
    print(f"  intensity MAE={metrics['intensity_mae']:.2f} g/kWh  RMSE={metrics['intensity_rmse']:.2f}")

    # --- partition tree for local support ------------------------------------
    partition_tree = make_partition_tree()
    partition_tree.fit(X_train, y_train)
    print(f"  partition tree leaves={partition_tree.get_n_leaves()}")

    # --- original formulation, for the record --------------------------------
    Xo_tr, Xo_te, yo_tr, yo_te = train_test_split(X, intensity, test_size=0.25, random_state=9)
    original_shuffled = DecisionTreeRegressor(random_state=9).fit(Xo_tr, yo_tr)
    shuffled_mae = mean_absolute_error(yo_te, original_shuffled.predict(Xo_te))

    original = DecisionTreeRegressor(random_state=9).fit(X_train, i_train)
    chrono_mae = mean_absolute_error(i_test, original.predict(X_test))

    metrics["original_shuffled_split_mae"] = float(shuffled_mae)
    metrics["original_chronological_mae"] = float(chrono_mae)
    metrics["original_n_leaves"] = int(original.get_n_leaves())

    print(f"\noriginal fully grown tree on intensity ({metrics['original_n_leaves']} leaves):")
    print(f"  MAE, shuffled split      = {shuffled_mae:.2f} g/kWh   <- leakage-inflated")
    print(f"  MAE, chronological split = {chrono_mae:.2f} g/kWh   <- honest")
    print(f"  deployed model           = {metrics['intensity_mae']:.2f} g/kWh")

    # --- refit on everything for deployment ----------------------------------
    # The metrics above describe what this configuration achieves on unseen
    # future data. What gets deployed is refitted on the whole year, including
    # the most recent weeks; serving a model fitted only to the first 80% of
    # the data leaves it months stale on day one, which showed up as a
    # systematic bias against live observations.
    print("\nrefitting on the full period for deployment...")
    deployed = make_monotone_ensemble(**search.best_params_)
    deployed.fit(X, y)
    deployed_tree = make_partition_tree()
    deployed_tree.fit(X, y)
    original_full = DecisionTreeRegressor(random_state=9).fit(X, intensity)

    # --- exploitation probe --------------------------------------------------
    base_idx = len(X_test) // 2
    base_row = X_test[base_idx].copy()
    base_demand = float(d_test[base_idx])

    # Probed on the split-fitted models against a held-out base row. Probing a
    # fully grown tree that has already seen that row measures memorisation
    # rather than the shape of the surface the optimiser would search.
    new_probe = thermal_ramp_probe(ensemble.predict, base_row, base_demand)
    old_probe = thermal_ramp_probe(
        lambda rows: total_emission_rate(original.predict(rows), base_demand),
        base_row,
        base_demand,
    )
    # Monotonicity is structural, so it must survive the refit.
    deployed_probe = thermal_ramp_probe(deployed.predict, base_row, base_demand)
    assert deployed_probe["monotone_non_decreasing"], "deployed model is not monotone"
    metrics["probe_deployed"] = new_probe
    metrics["probe_original"] = old_probe

    print("\nthermal ramp probe (0 -> 1500 MW, all else fixed):")
    print(
        f"  deployed: {new_probe['total_change_tph']:+.1f} t/h, "
        f"monotone={new_probe['monotone_non_decreasing']}, argmin at {new_probe['min_at_mw']:.0f} MW"
    )
    print(
        f"  original: {old_probe['total_change_tph']:+.1f} t/h, "
        f"monotone={old_probe['monotone_non_decreasing']}, argmin at {old_probe['min_at_mw']:.0f} MW"
    )

    # --- wrap and persist ----------------------------------------------------
    safe = build_safe_model(deployed, deployed_tree, X, y, CO2_FEATURES)
    _, _, support, novelty = safe.evaluate(X_test)
    in_dist = safe.is_supported(support, novelty)

    metrics["novelty_threshold"] = float(safe.novelty_threshold)
    metrics["test_in_distribution_pct"] = float(in_dist.mean() * 100)
    print(f"\nnovelty threshold (Mahalanobis^2 @ q99) = {safe.novelty_threshold:.2f}")
    print(f"test rows judged in-distribution: {metrics['test_in_distribution_pct']:.1f}%")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(safe, MODELS_DIR / "co2_model.joblib")
    (MODELS_DIR / "co2_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nsaved {MODELS_DIR / 'co2_model.joblib'}")


if __name__ == "__main__":
    main()
