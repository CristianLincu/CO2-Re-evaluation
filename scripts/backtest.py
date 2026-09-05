"""Rolling-origin backtest of the full optimisation stack.

The headline number the dashboard reports is a *model* claim: the emissions
model says the optimised trajectory is cleaner than holding dispatch fixed.
Nothing in the live pipeline tests whether that claim survives contact with
what the grid actually did.

This script replays the whole system through history. It walks five expanding
training windows; within each, every model -- the CO2 ensemble, both
forecasters, the operating envelope -- is fitted only on data preceding the
test window, and the genetic algorithm runs at its deployed budget. For each
replayed origin it records

* what the model predicts for the dispatch that *actually happened*, against
  the emissions that were actually measured -- the only place in the whole
  exercise where a ground truth exists;
* what the model predicts for the optimised trajectory, under both the
  forecast conditions the optimiser saw and the conditions that materialised;
* where the reduction comes from, by substituting the optimiser's domestic
  generation and its cross-border schedule into the realised dispatch one
  group at a time;
* the novelty and leaf support of both realised and optimised operating
  points, so that model error measured on realised points can be projected
  onto the region the optimiser actually visits.

Hyperparameters are tuned once on the earliest training window and reused, so
no fold sees anything from its own future.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from pipeline.co2_model import build_safe_model, make_monotone_ensemble, make_partition_tree, total_emission_rate
from pipeline.config import (
    BALANCE_MASK,
    CACHE_DIR,
    CO2_FEATURES,
    CROSS_BORDER,
    DECISION_COLUMNS,
    GENERATION,
    HORIZON_STEPS,
    INTERNAL_LINK,
    ROOT as PROJECT_ROOT,
    TECHNICAL_LIMITS,
)
from pipeline.forecast import MultiStepForecaster, build_supervised
from pipeline.models import align_renewable_exog
from pipeline.optimize import ProblemSpec, derive_limits, hold_current_baseline, optimize

N_LAGS = 16
BACKTEST_DIR = PROJECT_ROOT / "backtest"
BALANCE_IDX = np.flatnonzero(np.array(BALANCE_MASK))

# Search space for the CO2 ensemble, duplicated from scripts/train_co2 so the
# backtest does not depend on that module being importable as a package.
CO2_SPACE = {
    "n_estimators": [300, 400, 600],
    "learning_rate": [0.03, 0.05, 0.08],
    "max_depth": [4, 6, 8],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [5, 20, 50],
    "reg_lambda": [1.0, 5.0, 20.0],
}

GEN_IDX = np.array([DECISION_COLUMNS.index(c) for c in GENERATION])
XB_IDX = np.array([DECISION_COLUMNS.index(c) for c in CROSS_BORDER])
LINK_IDX = DECISION_COLUMNS.index(INTERNAL_LINK)

# Cutoffs separating each fold's training window from its test window. The
# first five months are never tested so that fold 1 has a usable fit.
FOLD_CUTOFFS = [
    "2026-02-04",
    "2026-03-21",
    "2026-05-06",
    "2026-06-21",
    "2026-08-06",
]


# --- fold fitting -------------------------------------------------------------


def fit_forecaster(frame, target, lag_columns, exog_column, params):
    """One direct XGBoost model per horizon step, fitted on the whole window."""
    X, y, _, _ = build_supervised(frame, target, lag_columns, N_LAGS, HORIZON_STEPS, exog_column)
    models = []
    for step in range(HORIZON_STEPS):
        model = XGBRegressor(
            objective="reg:squarederror", tree_method="hist", n_jobs=8, random_state=9, **params
        )
        model.fit(X[:, step, :], y[:, step])
        # Serving is one row at a time inside a process pool; extra threads
        # there only cost synchronisation.
        model.set_params(n_jobs=1)
        models.append(model)
    return MultiStepForecaster(
        models=models,
        target=target,
        lag_columns=list(lag_columns),
        n_lags=N_LAGS,
        exog_column=exog_column,
        horizon=HORIZON_STEPS,
    )


def tune_once(frame, target, lag_columns, exog_column, space, n_iter=15):
    """Hyperparameter search on the earliest window only."""
    X, y, _, _ = build_supervised(frame, target, lag_columns, N_LAGS, HORIZON_STEPS, exog_column)
    step = HORIZON_STEPS // 2
    search = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=-1, random_state=9),
        param_distributions=space,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=3),
        random_state=9,
        n_jobs=1,
    )
    search.fit(X[:, step, :], y[:, step])
    return search.best_params_


def marginal_factors(frame):
    """Empirical marginal emission factors from 15-minute differences.

    Average attributed factors answer "who is charged for this megawatt".
    Differencing asks a closer question: when a source moves, how do total
    attributed emissions move with it. Non-negativity is imposed because an
    emission factor cannot be negative.
    """
    from scipy.optimize import nnls

    X = frame[CO2_FEATURES].to_numpy(float)
    y = total_emission_rate(frame["CO2Emission"].to_numpy(float), frame["Demand"].to_numpy(float))

    dX = np.diff(X, axis=0)
    dy = np.diff(y)
    keep = np.isfinite(dy) & np.isfinite(dX).all(axis=1)
    coef, _ = nnls(dX[keep], dy[keep])
    return coef


def build_fold(frame, cutoff, co2_params, demand_params, renewables_params, outdir):
    train = frame[frame["timestamp"] < cutoff].reset_index(drop=True)
    outdir.mkdir(parents=True, exist_ok=True)

    X = train[CO2_FEATURES].to_numpy(float)
    y = total_emission_rate(train["CO2Emission"].to_numpy(float), train["Demand"].to_numpy(float))

    ensemble = make_monotone_ensemble(n_jobs=8, **co2_params)
    ensemble.fit(X, y)
    tree = make_partition_tree()
    tree.fit(X, y)
    safe = build_safe_model(
        ensemble, tree, X, y, CO2_FEATURES, demand=train["Demand"].to_numpy(float)
    )

    demand_model = fit_forecaster(train, "Demand", ["Demand"], None, demand_params)
    ren_train = train.dropna(subset=["Forecast5Hour"]).reset_index(drop=True)
    renewables_model = fit_forecaster(
        ren_train, "Renewables", ["Renewables"], "Forecast5Hour", renewables_params
    )

    lower, upper, ramp = derive_limits(train, technical=TECHNICAL_LIMITS)

    dump(
        {
            "co2": safe,
            "demand_model": demand_model,
            "renewables_model": renewables_model,
            "lower": lower,
            "upper": upper,
            "ramp": ramp,
            "marginal": marginal_factors(train),
            "n_train": len(train),
            "cutoff": str(cutoff),
        },
        outdir / "fold.joblib",
    )
    return len(train)


# --- replay -------------------------------------------------------------------

_G = {}


def _init_worker(frame_path, fold_path, budget):
    frame = pd.read_parquet(frame_path)
    bundle = load(fold_path)
    _G.update(bundle)
    _G["frame"] = frame
    _G["decision"] = frame[DECISION_COLUMNS].to_numpy(float)
    _G["demand"] = frame["Demand"].to_numpy(float)
    _G["renewables"] = frame["Renewables"].to_numpy(float)
    _G["intensity"] = frame["CO2Emission"].to_numpy(float)
    _G["fcst5"] = frame[["timestamp", "Forecast5Hour"]]
    _G["budget"] = budget


def _score(dispatch, renewables):
    """Model outputs for an arbitrary (T, G) dispatch under given renewables."""
    features = np.concatenate([np.asarray(renewables)[:, None], dispatch], axis=1)
    total, dispersion, support, novelty = _G["co2"].evaluate(features)
    return total, dispersion, support, novelty


def run_period(i):
    frame = _G["frame"]
    horizon = HORIZON_STEPS
    future = slice(i + 1, i + 1 + horizon)

    history = frame.iloc[: i + 1]
    # Keep the tz-aware dtype: align_renewable_exog matches on timestamp
    # identity, and a silent drop to naive UTC would miss every lookup.
    index = pd.DatetimeIndex(frame["timestamp"].iloc[future])

    exog = align_renewable_exog(_G["fcst5"], index, fallback=history["Renewables"].iloc[-1])
    demand_hat = np.clip(_G["demand_model"].predict(history, index), 500.0, None)
    renewables_hat = np.clip(
        _G["renewables_model"].predict(history, index, exog_future=exog), 0.0, None
    )

    current = _G["decision"][i].copy()
    demand_real = _G["demand"][future]
    renewables_real = _G["renewables"][future]
    dispatch_real = _G["decision"][future]
    observed_total = total_emission_rate(_G["intensity"][future], demand_real)

    islands, size, epochs = _G["budget"]
    spec_f = ProblemSpec(
        demand=demand_hat,
        renewables=renewables_hat,
        current=current,
        lower=_G["lower"].copy(),
        upper=_G["upper"].copy(),
        ramp=_G["ramp"],
        # Derived from the forecast conditions, exactly as it would be live.
        floor=_G["co2"].emission_floor(demand_hat, renewables_hat),
    )
    result = optimize(spec_f, _G["co2"], islands=islands, size=size, epochs=epochs)
    hold_traj, hold_diag = hold_current_baseline(spec_f, _G["co2"])

    # Everything below is re-scored under the conditions that actually
    # materialised, so the optimised and realised paths are compared on equal
    # footing rather than across two different weather outcomes.
    opt_total, opt_disp, opt_support, opt_novelty = _score(result.trajectory, renewables_real)
    hold_total, _, _, _ = _score(hold_traj, renewables_real)
    real_total, _, real_support, real_novelty = _score(dispatch_real, renewables_real)

    # Group attribution: substitute one block of the optimiser's decision into
    # the realised dispatch and re-predict.
    hybrid_gen = dispatch_real.copy()
    hybrid_gen[:, GEN_IDX] = result.trajectory[:, GEN_IDX]
    hybrid_xb = dispatch_real.copy()
    hybrid_xb[:, XB_IDX] = result.trajectory[:, XB_IDX]
    gen_total, _, _, _ = _score(hybrid_gen, renewables_real)
    xb_total, _, _, _ = _score(hybrid_xb, renewables_real)

    # Marginal-factor view of the same move.
    delta = result.trajectory - dispatch_real
    factors = _G["marginal"][1:]  # drop the renewables column; it cancels
    marginal_delta = (delta * factors).sum(axis=1)

    # Does the optimised schedule still balance under realised conditions?
    supplied = result.trajectory[:, BALANCE_IDX].sum(axis=1)
    balance_real = supplied + renewables_real - demand_real

    hours = 0.25
    step_rows = pd.DataFrame(
        {
            "origin": i,
            "timestamp": index,
            "step": np.arange(1, horizon + 1),
            "demand_real": demand_real,
            "demand_hat": demand_hat,
            "renewables_real": renewables_real,
            "renewables_hat": renewables_hat,
            "observed_total": observed_total,
            "model_total_real": real_total,
            "model_total_opt": opt_total,
            "model_total_hold": hold_total,
            "opt_novelty": opt_novelty,
            "real_novelty": real_novelty,
            "opt_support": opt_support,
            "real_support": real_support,
            "balance_real": balance_real,
            "marginal_delta": marginal_delta,
        }
    )

    record = {
        "origin": i,
        "timestamp": index[0],
        # --- ground truth ----------------------------------------------------
        "observed_tph": observed_total.mean(),
        "model_real_tph": real_total.mean(),
        "model_error_tph": real_total.mean() - observed_total.mean(),
        "model_abs_error_tph": np.abs(real_total - observed_total).mean(),
        # --- what the live system would have reported -------------------------
        "reported_hold_tph": hold_diag["total"][0].mean(),
        "reported_opt_tph": result.total_tph.mean(),
        "reported_reduction_pct": 100
        * (1 - result.total_tph.mean() / max(hold_diag["total"][0].mean(), 1e-9)),
        # --- the same comparison under realised conditions --------------------
        "opt_tph": opt_total.mean(),
        "hold_tph": hold_total.mean(),
        "reduction_vs_hold_pct": 100 * (1 - opt_total.mean() / max(hold_total.mean(), 1e-9)),
        "reduction_vs_real_pct": 100 * (1 - opt_total.mean() / max(real_total.mean(), 1e-9)),
        "reduction_vs_real_tph": real_total.mean() - opt_total.mean(),
        # --- decomposition ----------------------------------------------------
        "delta_gen_tph": gen_total.mean() - real_total.mean(),
        "delta_xb_tph": xb_total.mean() - real_total.mean(),
        "delta_total_tph": opt_total.mean() - real_total.mean(),
        "marginal_delta_tph": marginal_delta.mean(),
        # --- energy moved -----------------------------------------------------
        "gen_energy_delta_mwh": (delta[:, GEN_IDX].sum(axis=1) * hours).sum(),
        "xb_energy_delta_mwh": (delta[:, XB_IDX].sum(axis=1) * hours).sum(),
        "link_energy_delta_mwh": (delta[:, LINK_IDX] * hours).sum(),
        # --- feasibility and domain ------------------------------------------
        "feasible_forecast": bool(result.feasible),
        "balance_real_abs_mw": np.abs(balance_real).mean(),
        "balance_real_pct": 100 * np.abs(balance_real).mean() / demand_real.mean(),
        "opt_in_dist_pct": 100
        * _G["co2"].is_supported(opt_support, opt_novelty).mean(),
        "real_in_dist_pct": 100
        * _G["co2"].is_supported(real_support, real_novelty).mean(),
        "opt_novelty_median": float(np.median(opt_novelty)),
        "real_novelty_median": float(np.median(real_novelty)),
        "within_floor_pct": 100 * float(np.mean(result.within_floor)),
        # --- forecast quality -------------------------------------------------
        "demand_mae": np.abs(demand_hat - demand_real).mean(),
        "renewables_mae": np.abs(renewables_hat - renewables_real).mean(),
        "uncertainty_tph": opt_disp.mean(),
    }
    # The mean move per resource. Averaged over origins this gives the
    # optimiser's characteristic direction, which lets the ranking test be
    # restricted to the region of dispatch space it actually exploits.
    record.update({f"delta_{c}": delta[:, j].mean() for j, c in enumerate(DECISION_COLUMNS)})
    return record, step_rows


def _run_period_safe(i):
    try:
        return run_period(i)
    except Exception as exc:  # a single bad origin must not sink the run
        return {"origin": i, "error": str(exc)}, None


# --- driver -------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=40, help="replayed origins per fold")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--islands", type=int, default=16)
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--refit", action="store_true", help="rebuild fold models")
    args = parser.parse_args()

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    frame_path = CACHE_DIR / "training_frame.parquet"
    frame = pd.read_parquet(frame_path)
    frame = frame[frame["Demand"] > 500].reset_index(drop=True)
    frame = frame.dropna(subset=CO2_FEATURES + ["CO2Emission", "Demand"]).reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    replay_path = BACKTEST_DIR / "frame.parquet"
    frame.to_parquet(replay_path, index=False)

    cutoffs = [pd.Timestamp(c, tz="UTC") for c in FOLD_CUTOFFS]
    ends = cutoffs[1:] + [frame["timestamp"].iloc[-1]]

    # The series has a handful of Energinet outages, including one of three
    # days. An origin whose lag window or horizon straddles a gap would be
    # fed features from the wrong times, so only fully regular spans qualify.
    regular = frame["timestamp"].diff() == pd.Timedelta(minutes=15)
    span = N_LAGS + HORIZON_STEPS
    contiguous = (
        regular.rolling(span, min_periods=span).sum().shift(-HORIZON_STEPS) == span
    ).fillna(False).to_numpy()
    print(f"{contiguous.sum()} of {len(frame)} origins sit on an unbroken span")

    # --- hyperparameters, chosen once on the earliest window ------------------
    params_path = BACKTEST_DIR / "params.json"
    if params_path.exists() and not args.refit:
        params = json.loads(params_path.read_text(encoding="utf-8"))
    else:
        first = frame[frame["timestamp"] < cutoffs[0]].reset_index(drop=True)
        print(f"tuning on the earliest window ({len(first)} samples, no fold sees its own future)")

        from pipeline.forecast import PARAM_SPACE

        t0 = time.perf_counter()
        Xc = first[CO2_FEATURES].to_numpy(float)
        yc = total_emission_rate(first["CO2Emission"].to_numpy(float), first["Demand"].to_numpy(float))
        search = RandomizedSearchCV(
            make_monotone_ensemble(),
            param_distributions=CO2_SPACE,
            n_iter=12,
            scoring="neg_root_mean_squared_error",
            cv=TimeSeriesSplit(n_splits=3),
            random_state=9,
            n_jobs=1,
        )
        search.fit(Xc, yc)
        params = {"co2": search.best_params_}
        print(f"  co2: {params['co2']}  ({time.perf_counter() - t0:.0f}s)")

        params["demand"] = tune_once(first, "Demand", ["Demand"], None, PARAM_SPACE)
        print(f"  demand: {params['demand']}")
        ren_first = first.dropna(subset=["Forecast5Hour"]).reset_index(drop=True)
        params["renewables"] = tune_once(
            ren_first, "Renewables", ["Renewables"], "Forecast5Hour", PARAM_SPACE
        )
        print(f"  renewables: {params['renewables']}")
        params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    # --- folds ----------------------------------------------------------------
    rng = np.random.default_rng(11)
    all_records, all_steps = [], []

    for k, (cutoff, end) in enumerate(zip(cutoffs, ends), start=1):
        outdir = BACKTEST_DIR / f"fold_{k}"
        fold_path = outdir / "fold.joblib"
        if not fold_path.exists() or args.refit:
            t0 = time.perf_counter()
            n_train = build_fold(
                frame, cutoff, params["co2"], params["demand"], params["renewables"], outdir
            )
            print(f"fold {k}: trained on {n_train} rows up to {cutoff:%Y-%m-%d} ({time.perf_counter() - t0:.0f}s)")

        window = frame.index[(frame["timestamp"] >= cutoff) & (frame["timestamp"] < end)]
        usable = window[(window >= N_LAGS) & (window < len(frame) - HORIZON_STEPS - 1)]
        usable = usable[contiguous[usable]]
        chosen = np.sort(rng.choice(usable, size=min(args.periods, len(usable)), replace=False))

        print(
            f"fold {k}: replaying {len(chosen)} origins in "
            f"{cutoff:%Y-%m-%d} -> {end:%Y-%m-%d}",
            flush=True,
        )

        t0 = time.perf_counter()
        os.environ["OMP_NUM_THREADS"] = "1"
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(replay_path, fold_path, (args.islands, args.size, args.epochs)),
        ) as pool:
            for n, (record, steps) in enumerate(pool.map(_run_period_safe, chosen), start=1):
                record["fold"] = k
                all_records.append(record)
                if steps is not None:
                    steps["fold"] = k
                    all_steps.append(steps)
                if n % 10 == 0:
                    print(f"  {n}/{len(chosen)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        os.environ.pop("OMP_NUM_THREADS", None)
        print(f"fold {k}: done in {time.perf_counter() - t0:.0f}s", flush=True)

    records = pd.DataFrame(all_records)
    steps = pd.concat(all_steps, ignore_index=True)
    records.to_parquet(BACKTEST_DIR / "periods.parquet", index=False)
    steps.to_parquet(BACKTEST_DIR / "steps.parquet", index=False)

    failures = records["error"].notna().sum() if "error" in records else 0
    print(f"\nwrote {len(records)} periods ({failures} failed) and {len(steps)} steps")


if __name__ == "__main__":
    main()
