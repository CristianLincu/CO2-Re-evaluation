"""Multi-step demand and renewables forecasting.

Two changes matter relative to the original framework.

First, evaluation is honest. The original split a five-minute series with a
shuffled ``train_test_split`` and tuned with plain ``KFold``, which places
near-duplicate neighbours on both sides of the split; reported errors were
therefore far better than the model would achieve live. Here every split is
chronological and every model is reported against a persistence baseline.

Second, renewables are no longer extrapolated from their own lags. Wind four
hours out is not recoverable from recent output, so the model consumes
Energinet's own physical forecast and learns a bias correction on top of it.
The ``Forecast5Hour`` vintage is issued roughly five hours before its target,
so it is available at decision time across the whole horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

PARAM_SPACE = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 4, 6, 8],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 5, 20],
    "reg_lambda": [1.0, 5.0, 20.0],
}


def calendar_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Cyclical time-of-day and day-of-week encodings.

    Demand is strongly periodic on both scales; the original model saw only
    raw lags and so had nothing to lean on beyond persistence.
    """
    ts = pd.DatetimeIndex(timestamps)
    minute_of_day = ts.hour * 60 + ts.minute
    day_angle = 2 * np.pi * minute_of_day / (24 * 60)
    week_angle = 2 * np.pi * ts.dayofweek / 7.0

    return np.column_stack(
        [
            np.sin(day_angle),
            np.cos(day_angle),
            np.sin(week_angle),
            np.cos(week_angle),
            (ts.dayofweek >= 5).astype(float),
        ]
    )


@dataclass
class MultiStepForecaster:
    """One direct model per horizon step."""

    models: list
    target: str
    lag_columns: list[str]
    n_lags: int
    exog_column: str | None
    horizon: int

    def build_row(self, history: pd.DataFrame, future_index: pd.DatetimeIndex, exog_future=None):
        """Assemble the feature matrix for a single live prediction.

        Returns one row per horizon step, since each step has its own model
        and its own calendar and exogenous values.
        """
        lags = []
        for column in self.lag_columns:
            series = history[column].to_numpy(dtype=float)[-self.n_lags :]
            if len(series) < self.n_lags:
                series = np.pad(series, (self.n_lags - len(series), 0), mode="edge")
            lags.append(series[::-1])  # most recent first
        lag_vector = np.concatenate(lags)

        cal = calendar_features(future_index)

        rows = []
        for step in range(self.horizon):
            row = [lag_vector, cal[step]]
            if self.exog_column is not None:
                value = 0.0 if exog_future is None else float(exog_future[step])
                row.append(np.array([value]))
            rows.append(np.concatenate(row))

        return np.vstack(rows)

    def predict(self, history: pd.DataFrame, future_index: pd.DatetimeIndex, exog_future=None):
        X = self.build_row(history, future_index, exog_future)
        return np.array([self.models[step].predict(X[step : step + 1])[0] for step in range(self.horizon)])


def build_supervised(
    frame: pd.DataFrame,
    target: str,
    lag_columns: list[str],
    n_lags: int,
    horizon: int,
    exog_column: str | None = None,
):
    """Chronological supervised frames, one target column per horizon step."""
    values = {c: frame[c].to_numpy(dtype=float) for c in lag_columns}
    target_values = frame[target].to_numpy(dtype=float)
    index = pd.DatetimeIndex(frame["timestamp"])
    cal = calendar_features(index)
    exog = frame[exog_column].to_numpy(dtype=float) if exog_column else None

    n = len(frame)
    rows, targets, target_times, persistence = [], [], [], []

    for i in range(n_lags - 1, n - horizon):
        lag_vector = np.concatenate([values[c][i - n_lags + 1 : i + 1][::-1] for c in lag_columns])
        step_rows, step_targets = [], []
        ok = True
        for step in range(1, horizon + 1):
            j = i + step
            pieces = [lag_vector, cal[j]]
            if exog is not None:
                if not np.isfinite(exog[j]):
                    ok = False
                    break
                pieces.append(np.array([exog[j]]))
            step_rows.append(np.concatenate(pieces))
            step_targets.append(target_values[j])
        if not ok:
            continue

        rows.append(np.vstack(step_rows))
        targets.append(step_targets)
        target_times.append(index[i])
        persistence.append(target_values[i])

    if not rows:
        raise RuntimeError(f"no usable samples for {target}")

    X = np.stack(rows)  # (samples, horizon, features)
    y = np.array(targets)  # (samples, horizon)
    return X, y, np.array(persistence), pd.DatetimeIndex(target_times)


def _search_params(X, y, n_iter=25, n_splits=4, seed=9):
    """Tune once on a mid-horizon step and reuse across steps."""
    model = XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=-1, random_state=seed)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_SPACE,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=n_splits),
        random_state=seed,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_params_


def train_forecaster(
    frame: pd.DataFrame,
    target: str,
    lag_columns: list[str],
    n_lags: int,
    horizon: int,
    exog_column: str | None = None,
    test_fraction: float = 0.2,
    label: str = "",
):
    """Fit one model per step, with a chronological hold-out and baselines."""
    X, y, persistence, _ = build_supervised(frame, target, lag_columns, n_lags, horizon, exog_column)

    split = int(len(X) * (1 - test_fraction))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    persist_test = persistence[split:]

    tune_step = horizon // 2
    print(f"[{label}] tuning on step {tune_step + 1} ({len(X_train)} train samples)")
    params = _search_params(X_train[:, tune_step, :], y_train[:, tune_step])
    print(f"[{label}] params: {params}")

    models, metrics = [], []
    for step in range(horizon):
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=9,
            **params,
        )
        model.fit(X_train[:, step, :], y_train[:, step])
        models.append(model)

        pred = model.predict(X_test[:, step, :])
        row = {
            "step": step + 1,
            "minutes_ahead": (step + 1) * 15,
            "model_rmse": root_mean_squared_error(y_test[:, step], pred),
            "model_mae": mean_absolute_error(y_test[:, step], pred),
            "persistence_rmse": root_mean_squared_error(y_test[:, step], persist_test),
        }
        if exog_column is not None:
            # The raw Energinet forecast sits in the last feature column.
            raw = X_test[:, step, -1]
            row["energinet_rmse"] = root_mean_squared_error(y_test[:, step], raw)
        metrics.append(row)

    forecaster = MultiStepForecaster(
        models=models,
        target=target,
        lag_columns=list(lag_columns),
        n_lags=n_lags,
        exog_column=exog_column,
        horizon=horizon,
    )
    return forecaster, pd.DataFrame(metrics), params
