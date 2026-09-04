"""Model loading and the serving-time forecast path."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import load

from pipeline.config import DECISION_COLUMNS, HORIZON_STEPS, MODELS_DIR, RESOLUTION_MIN

# Preference order for Energinet's forecast vintages. Forecast5Hour is issued
# about five hours ahead of its target and so is available for the whole
# horizon; the others are fallbacks for the rare gap.
FORECAST_VINTAGES = ["Forecast5Hour", "ForecastCurrent", "ForecastDayAhead"]


def load_models():
    co2 = load(MODELS_DIR / "co2_model.joblib")
    demand = load(MODELS_DIR / "demand_forecaster.joblib")
    renewables = load(MODELS_DIR / "renewables_forecaster.joblib")
    return co2, demand, renewables


def load_limits():
    payload = json.loads((MODELS_DIR / "operating_limits.json").read_text(encoding="utf-8"))
    return (
        np.array([payload["lower"][c] for c in DECISION_COLUMNS]),
        np.array([payload["upper"][c] for c in DECISION_COLUMNS]),
        np.array([payload["ramp"][c] for c in DECISION_COLUMNS]),
    )


def future_index(last_timestamp, steps=HORIZON_STEPS):
    start = pd.Timestamp(last_timestamp)
    return pd.DatetimeIndex(
        [start + pd.Timedelta(minutes=RESOLUTION_MIN * (i + 1)) for i in range(steps)]
    )


def align_renewable_exog(forecast_frame, index, fallback):
    """Line Energinet's forecast up with the horizon, with fallbacks."""
    if forecast_frame is None or forecast_frame.empty:
        return np.full(len(index), float(fallback))

    frame = forecast_frame.set_index("timestamp").sort_index()
    values = []
    for timestamp in index:
        value = np.nan
        if timestamp in frame.index:
            row = frame.loc[timestamp]
            for column in FORECAST_VINTAGES:
                if column in frame.columns and np.isfinite(row.get(column, np.nan)):
                    value = float(row[column])
                    break
        values.append(value)

    series = pd.Series(values, index=index).ffill().bfill()
    return series.fillna(float(fallback)).to_numpy(dtype=float)


def build_forecasts(history, renewable_forecast_frame, demand_model, renewables_model):
    """Demand and renewables trajectories for the coming horizon."""
    index = future_index(history["timestamp"].iloc[-1])

    demand = demand_model.predict(history, index)

    exog = align_renewable_exog(
        renewable_forecast_frame, index, fallback=history["Renewables"].iloc[-1]
    )
    renewables = renewables_model.predict(history, index, exog_future=exog)

    # Renewable output cannot be negative, and demand below a plausible floor
    # signals a data problem rather than a real forecast.
    renewables = np.clip(renewables, 0.0, None)
    demand = np.clip(demand, 500.0, None)

    current = history[DECISION_COLUMNS].iloc[-1].to_numpy(dtype=float)
    return demand, renewables, current, index, exog
