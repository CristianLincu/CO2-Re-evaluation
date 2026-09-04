"""Build the dashboard payload."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pipeline.config import DECISION_COLUMNS, DOCS_DATA, MODELS_DIR, RESOLUTION_MIN

COPENHAGEN = ZoneInfo("Europe/Copenhagen")

TABLE_COLUMNS = [
    "Time",
    "Power Plants Ge100MW",
    "Power Plants Lt100MW",
    "DK1-DE",
    "DK1-NL",
    "DK1-GB",
    "DK1-NO",
    "DK1-SE",
    "DK1-DK2",
    "DK2-DE",
    "DK2-SE",
    "Bornholm-SE",
    "Demand Forecast",
    "Renewables Forecast",
    "CO2 Baseline",
    "CO2 Optimised",
]

PAST_POINTS = 48  # twelve hours of observed history on the 15-minute grid


def _local(ts):
    return pd.Timestamp(ts).tz_convert(COPENHAGEN)


def _iso(ts):
    return _local(ts).isoformat()


def _load_metrics():
    """Validation numbers, surfaced on the dashboard rather than buried."""
    out = {}
    for name, path in (
        ("co2", MODELS_DIR / "co2_metrics.json"),
        ("forecast", MODELS_DIR / "forecast_metrics.json"),
    ):
        try:
            out[name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            out[name] = None
    return out


def build_payload(
    history,
    index,
    demand,
    renewables,
    result,
    baseline_diag,
    status="ok",
    error_message=None,
    runtime_seconds=None,
):
    """Assemble everything the dashboard renders.

    The central reporting decision: the headline reduction is measured against
    the hold-current counterfactual, not against the present measured value.
    Both are published so the difference is visible.
    """
    metrics = _load_metrics()

    past = history.tail(PAST_POINTS)
    current_co2 = float(history["CO2Emission"].iloc[-1])

    optimised = np.asarray(result.intensity, dtype=float)
    baseline = np.asarray(baseline_diag["intensity"][0], dtype=float)
    uncertainty = np.asarray(result.uncertainty, dtype=float)

    baseline_mean = float(baseline.mean())
    optimised_mean = float(optimised.mean())
    reduction = (
        round(100.0 * (1.0 - optimised_mean / baseline_mean), 1) if baseline_mean > 0 else None
    )

    # Allocation table, one row per horizon step.
    rows = []
    for step, timestamp in enumerate(index):
        row = [_local(timestamp).strftime("%H:%M")]
        row += [f"{result.trajectory[step, g]:.0f}" for g in range(len(DECISION_COLUMNS))]
        row += [
            f"{demand[step]:.0f}",
            f"{renewables[step]:.0f}",
            f"{baseline[step]:.1f}",
            f"{optimised[step]:.1f}",
        ]
        rows.append(row)

    # Resource trajectories, for the stacked dispatch chart.
    trajectories = {
        name: [float(v) for v in result.trajectory[:, i]]
        for i, name in enumerate(DECISION_COLUMNS)
    }

    scatter = [
        {
            "x": float(r.ProductionGe100MW),
            "y": float(r.Renewables),
            "z": float(r.NetImports),
            "co2": float(r.CO2Emission),
        }
        for r in history.tail(96).itertuples()
    ]

    return {
        "lastUpdated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "errorMessage": error_message,
        "timezone": "Europe/Copenhagen",
        "resolutionMinutes": RESOLUTION_MIN,
        "horizonSteps": len(index),
        "co2Chart": {
            "past": {
                "timestamps": [_iso(t) for t in past["timestamp"]],
                "values": [float(v) for v in past["CO2Emission"]],
            },
            "forecast": {
                "timestamps": [_iso(t) for t in index],
                "optimised": optimised.tolist(),
                "baseline": baseline.tolist(),
                "uncertainty": uncertainty.tolist(),
            },
        },
        "forecasts": {
            "timestamps": [_iso(t) for t in index],
            "demand": [float(v) for v in demand],
            "renewables": [float(v) for v in renewables],
        },
        "trajectories": trajectories,
        "allocationTable": {"columns": TABLE_COLUMNS, "rows": rows},
        "scatter3d": {"points": scatter},
        "diagnostics": {
            "feasible": bool(result.feasible),
            "inDistributionPct": float(np.mean(result.supported) * 100.0),
            "maxBalanceErrorMW": float(np.abs(result.balance_error).max()),
            "meanUncertainty": float(uncertainty.mean()),
            "runtimeSeconds": runtime_seconds,
        },
        "meta": {
            "author": "Cristian Lincu",
            "currentCo2": current_co2,
            "baselineCo2": baseline_mean,
            "optimisedCo2": optimised_mean,
            "reductionPct": reduction,
            "reductionBasis": "hold-current counterfactual over the same horizon",
            "timezoneLabel": "Denmark local time (CET/CEST)",
        },
        "validation": metrics,
    }


def write_payload(payload, output_path=None):
    output_path = output_path or (DOCS_DATA / "latest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return output_path


def load_existing(output_path=None):
    output_path = output_path or (DOCS_DATA / "latest.json")
    if not output_path.exists():
        return None
    try:
        with output_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None
