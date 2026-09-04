"""Train the demand and renewables forecasters."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import pandas as pd
from joblib import dump

from pipeline.config import CACHE_DIR, HORIZON_STEPS, MODELS_DIR
from pipeline.forecast import train_forecaster

N_LAGS = 16  # four hours of history on the 15-minute grid


def main():
    frame = pd.read_parquet(CACHE_DIR / "training_frame.parquet")
    frame = frame[frame["Demand"] > 500].reset_index(drop=True)
    print(f"samples={len(frame)}  {frame['timestamp'].min()} -> {frame['timestamp'].max()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    # --- demand ---------------------------------------------------------------
    demand_model, demand_metrics, demand_params = train_forecaster(
        frame,
        target="Demand",
        lag_columns=["Demand"],
        n_lags=N_LAGS,
        horizon=HORIZON_STEPS,
        exog_column=None,
        label="demand",
    )
    print("\ndemand forecast (MW RMSE, chronological hold-out):")
    print(demand_metrics.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    dump(demand_model, MODELS_DIR / "demand_forecaster.joblib")
    summary["demand"] = {
        "params": demand_params,
        "metrics": demand_metrics.to_dict(orient="records"),
    }

    # --- renewables -----------------------------------------------------------
    # Energinet's own physical forecast enters as an exogenous feature. The
    # Forecast5Hour vintage is issued about five hours before its target, so it
    # is genuinely available across the whole four-hour horizon.
    renewables_frame = frame.dropna(subset=["Forecast5Hour"]).reset_index(drop=True)
    print(f"\nrenewables samples with Energinet forecast: {len(renewables_frame)}")

    renewables_model, renewables_metrics, renewables_params = train_forecaster(
        renewables_frame,
        target="Renewables",
        lag_columns=["Renewables"],
        n_lags=N_LAGS,
        horizon=HORIZON_STEPS,
        exog_column="Forecast5Hour",
        label="renewables",
    )
    print("\nrenewables forecast (MW RMSE, chronological hold-out):")
    print(renewables_metrics.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    dump(renewables_model, MODELS_DIR / "renewables_forecaster.joblib")
    summary["renewables"] = {
        "params": renewables_params,
        "metrics": renewables_metrics.to_dict(orient="records"),
    }

    (MODELS_DIR / "forecast_metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nsaved forecasters to {MODELS_DIR}")


if __name__ == "__main__":
    main()
