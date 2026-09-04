"""Generate the figures used in the paper."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import CACHE_DIR, DECISION_COLUMNS, HORIZON_STEPS, MODELS_DIR
from pipeline.data import get_live_frame, get_live_renewable_forecast
from pipeline.models import build_forecasts, load_limits, load_models
from pipeline.optimize import ProblemSpec, evaluate, hold_current_baseline, optimize, repair

FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

INK = "#1f3b57"
ACCENT = "#c1462f"
MUTED = "#7b8a99"


def fig_probe():
    """The exploitation probe: what each model says about ramping thermal."""
    metrics = json.loads((MODELS_DIR / "co2_metrics.json").read_text(encoding="utf-8"))
    new, old = metrics["probe_deployed"], metrics["probe_original"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    for ax, probe, title, colour in (
        (axes[0], old, "Original: unconstrained tree", ACCENT),
        (axes[1], new, "This work: monotone ensemble", INK),
    ):
        x = np.array(probe["thermal_mw"])
        y = np.array(probe["total_tph"])
        ax.plot(x, y, color=colour, lw=1.8)
        argmin = int(np.argmin(y))
        ax.plot(x[argmin], y[argmin], "o", color=colour, ms=5)
        ax.annotate(
            f"minimum at {x[argmin]:.0f} MW",
            xy=(x[argmin], y[argmin]),
            xytext=(0.42, 0.12),
            textcoords="axes fraction",
            fontsize=8,
            color=colour,
            arrowprops=dict(arrowstyle="->", color=colour, lw=0.8),
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Large-plant output (MW)")

    axes[0].set_ylabel("Predicted total emissions (t/h)")
    fig.suptitle(
        "Response to ramping thermal generation, all other resources held fixed",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "probe.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote probe.pdf")


def fig_forecast_skill():
    """Forecast error against horizon, with the baselines it must beat."""
    metrics = json.loads((MODELS_DIR / "forecast_metrics.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    demand = pd.DataFrame(metrics["demand"]["metrics"])
    axes[0].plot(demand.minutes_ahead, demand.persistence_rmse, "--", color=MUTED, label="persistence")
    axes[0].plot(demand.minutes_ahead, demand.model_rmse, color=INK, lw=1.8, label="XGBoost")
    axes[0].set_title("Demand", fontsize=9)

    renew = pd.DataFrame(metrics["renewables"]["metrics"])
    axes[1].plot(renew.minutes_ahead, renew.persistence_rmse, "--", color=MUTED, label="persistence")
    axes[1].plot(renew.minutes_ahead, renew.energinet_rmse, ":", color=ACCENT, lw=1.6, label="Energinet forecast")
    axes[1].plot(renew.minutes_ahead, renew.model_rmse, color=INK, lw=1.8, label="XGBoost post-processed")
    axes[1].set_title("Renewables", fontsize=9)

    for ax in axes:
        ax.set_xlabel("Minutes ahead")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("RMSE (MW)")

    fig.suptitle("Forecast skill on a chronological hold-out", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "forecast_skill.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote forecast_skill.pdf")


def fig_convergence_and_dispatch():
    """Convergence against random search, plus an example trajectory."""
    co2_model, demand_model, renewables_model = load_models()
    lower, upper, ramp = load_limits()

    history = get_live_frame(hours=30)
    forecast_frame = get_live_renewable_forecast(HORIZON_STEPS)
    demand, renewables, current, index, _ = build_forecasts(
        history, forecast_frame, demand_model, renewables_model
    )
    spec = ProblemSpec(demand, renewables, current, lower, upper, ramp)

    _, baseline_diag = hold_current_baseline(spec, co2_model)
    baseline = baseline_diag["intensity"][0]

    result = optimize(spec, co2_model)

    # Random search with a matched evaluation budget.
    rng = np.random.default_rng(0)
    budget = len(result.history) * 16 * 150
    best = np.inf
    trace = []
    drawn = 0
    per_generation = 16 * 150
    while drawn < budget:
        shape = (per_generation, spec.horizon, len(current))
        candidates = repair(
            rng.uniform(low=np.broadcast_to(lower, shape), high=np.broadcast_to(upper, shape)), spec
        )
        fitness, _ = evaluate(candidates, spec, co2_model)
        best = min(best, float(fitness.min()))
        trace.append(best)
        drawn += per_generation

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    axes[0].plot(result.history, color=INK, lw=1.8, label="genetic ensemble")
    axes[0].plot(trace, "--", color=ACCENT, lw=1.5, label="random search")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Best fitness")
    axes[0].set_title("Convergence at equal evaluation budget", fontsize=9)
    axes[0].legend(fontsize=7.5)

    hours = np.arange(1, spec.horizon + 1) * 0.25
    axes[1].plot(hours, baseline, "--", color=MUTED, lw=1.6, label="hold current")
    axes[1].plot(hours, result.intensity, color=INK, lw=1.8, label="optimised")
    axes[1].fill_between(
        hours,
        result.intensity - result.uncertainty,
        result.intensity + result.uncertainty,
        color=INK,
        alpha=0.15,
        label="model dispersion",
    )
    axes[1].set_xlabel("Hours ahead")
    axes[1].set_ylabel("CO$_2$ intensity (g/kWh)")
    axes[1].set_title("Optimised trajectory vs counterfactual", fontsize=9)
    axes[1].legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote convergence.pdf")

    # --- dispatch detail ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    interesting = [
        "ProductionGe100MW",
        "Exchange_DK1_DE",
        "Exchange_DK1_NO",
        "Exchange_DK2_SE",
    ]
    labels = {
        "ProductionGe100MW": "Large plants",
        "Exchange_DK1_DE": "DK1-DE",
        "Exchange_DK1_NO": "DK1-NO",
        "Exchange_DK2_SE": "DK2-SE",
    }
    colours = [INK, ACCENT, "#2e7d5b", "#8a6bbf"]
    for name, colour in zip(interesting, colours):
        i = DECISION_COLUMNS.index(name)
        ax.plot(hours, result.trajectory[:, i], color=colour, lw=1.6, label=labels[name])

    ax2 = ax.twinx()
    ax2.plot(hours, renewables, ":", color=MUTED, lw=1.6, label="renewables forecast")
    ax2.set_ylabel("Renewables (MW)", color=MUTED)
    ax2.grid(False)

    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("Scheduled power (MW)")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_title("Optimised dispatch trajectory", fontsize=9)
    handles, lbls = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, lbls + l2, fontsize=7.5, ncol=3)

    fig.tight_layout()
    fig.savefig(FIGURES / "dispatch.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote dispatch.pdf")


if __name__ == "__main__":
    fig_probe()
    fig_forecast_skill()
    fig_convergence_and_dispatch()
    print(f"\nfigures in {FIGURES}")
