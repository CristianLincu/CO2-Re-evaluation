"""Pipeline entry point for local runs and scheduled jobs."""

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import DOCS_DATA, HORIZON_STEPS
from pipeline.data import get_live_frame, get_live_renewable_forecast
from pipeline.export import build_payload, load_existing, write_payload
from pipeline.models import build_forecasts, load_limits, load_models
from pipeline.optimize import ProblemSpec, hold_current_baseline, optimize


def run_pipeline():
    output_path = DOCS_DATA / "latest.json"
    existing = load_existing(output_path)
    started = time.perf_counter()

    try:
        print("Loading models...")
        co2_model, demand_model, renewables_model = load_models()
        lower, upper, ramp = load_limits()

        print("Fetching Energinet data...")
        history = get_live_frame(hours=30)
        renewable_forecast = get_live_renewable_forecast(HORIZON_STEPS)

        print("Forecasting demand and renewables...")
        demand, renewables, current, index, _ = build_forecasts(
            history, renewable_forecast, demand_model, renewables_model
        )

        spec = ProblemSpec(
            demand=demand,
            renewables=renewables,
            current=current,
            lower=lower,
            upper=upper,
            ramp=ramp,
            floor=co2_model.emission_floor(demand, renewables),
        )

        print("Evaluating hold-current counterfactual...")
        _, baseline_diag = hold_current_baseline(spec, co2_model)

        print("Running genetic optimisation...")
        result = optimize(spec, co2_model)
        runtime = time.perf_counter() - started

        baseline_mean = float(baseline_diag["intensity"][0].mean())
        optimised_mean = float(result.intensity.mean())
        print(
            f"  baseline={baseline_mean:.1f} g/kWh  optimised={optimised_mean:.1f} g/kWh  "
            f"reduction={100 * (1 - optimised_mean / baseline_mean):.1f}%  "
            f"feasible={result.feasible}  in-dist={result.supported.mean() * 100:.0f}%"
        )

        payload = build_payload(
            history=history,
            index=index,
            demand=demand,
            renewables=renewables,
            result=result,
            baseline_diag=baseline_diag,
            status="ok",
            runtime_seconds=round(runtime, 1),
        )

    except Exception as exc:
        print(f"Pipeline error: {exc}")
        traceback.print_exc()

        if existing:
            # Keep serving the last good result, clearly marked as stale.
            payload = dict(existing)
            payload["status"] = "error"
            payload["errorMessage"] = str(exc)
        else:
            payload = {
                "lastUpdated": None,
                "status": "error",
                "errorMessage": str(exc),
                "co2Chart": {
                    "past": {"timestamps": [], "values": []},
                    "forecast": {
                        "timestamps": [],
                        "optimised": [],
                        "baseline": [],
                        "uncertainty": [],
                    },
                },
                "allocationTable": {"columns": [], "rows": []},
                "scatter3d": {"points": []},
                "meta": {"author": "Cristian Lincu", "reductionPct": None},
            }

    path = write_payload(payload, output_path)
    print(f"Wrote {path}  ({time.perf_counter() - started:.1f}s total)")
    return payload


if __name__ == "__main__":
    result = run_pipeline()
    sys.exit(0 if result.get("status") == "ok" else 1)
