"""Figures summarising the rolling-origin backtest."""

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

from pipeline.config import DECISION_COLUMNS
from scripts.analyze_backtest import BACKTEST_DIR, FOLD_CUTOFFS, counterfactual_ranking

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
GOOD = "#2e7d5b"


def save(fig, name):
    """PDF for the paper, PNG for the dashboard and for quick viewing."""
    for suffix in (".pdf", ".png"):
        fig.savefig(FIGURES / f"{name}{suffix}", bbox_inches="tight")
    plt.close(fig)


def fig_ranking(pairs, summary):
    """Does a predicted saving materialise? The load-bearing evidence."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    for ax, subset, title, key in (
        (axes[0], pairs, "All matched pairs", "ranking_all"),
        (
            axes[1],
            pairs[pairs["alignment"] > 0.5],
            "Pairs resembling the optimiser's move",
            "ranking_aligned",
        ),
    ):
        x = subset["predicted_delta"].to_numpy()
        y = subset["observed_delta"].to_numpy()
        limit = np.percentile(np.abs(np.concatenate([x, y])), 99)

        ax.hexbin(x, y, gridsize=45, extent=(-limit, limit, -limit, limit), cmap="Blues", mincnt=1, linewidths=0)
        grid = np.array([-limit, limit])
        ax.plot(grid, grid, color=MUTED, lw=1.0, ls="--", label="perfect delivery")
        slope = summary[key]["slope"]
        ax.plot(grid, slope * grid, color=ACCENT, lw=1.8, label=f"fitted slope {slope:.2f}")

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xlabel("predicted difference (t CO$_2$/h)")
        ax.set_title(f"{title}\nr = {summary[key]['r']:.2f}, n = {summary[key]['n']:,}", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel("metered difference (t CO$_2$/h)")
    fig.suptitle(
        "Under matched demand and wind, does a predicted saving actually arrive?",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "backtest_ranking")


def fig_discount(summary, steps, periods):
    """From the headline claim to what the evidence supports."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # --- panel A: the discount ladder -------------------------------------
    ax = axes[0]
    stages = [
        ("dashboard\nheadline", summary["pooled_reduction_vs_hold_pct"], MUTED),
        ("vs actual\ndispatch", summary["pooled_reduction_vs_real_pct"], MUTED),
        ("capped at\nobserved floor", summary["capped_reduction_pct"], INK),
        ("after ranking\ndiscount", summary["capped_discounted_pct"], GOOD),
    ]
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colours = [s[2] for s in stages]
    bars = ax.bar(range(len(values)), values, color=colours, width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.6, f"{value:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("reduction in attributed emissions (%)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_title("What survives each correction", fontsize=9)

    # --- panel B: where the saving comes from ------------------------------
    ax = axes[1]
    gen = -periods["delta_gen_tph"].mean()
    xb = -periods["delta_xb_tph"].mean()
    total = gen + xb
    ax.barh([1, 0], [gen, xb], color=[GOOD, ACCENT], height=0.55)
    for y, value in ((1, gen), (0, xb)):
        ax.text(value + total * 0.02, y, f"{value:.0f} t/h  ({100 * value / total:.0f}%)", va="center", fontsize=8.5)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["less domestic\ngeneration", "cross-border\nre-labelling"], fontsize=8)
    ax.set_xlim(0, total * 1.05)
    ax.set_xlabel("mean saving attributed to each lever (t CO$_2$/h)")
    ax.set_title("Reduced combustion, or reassigned accounting?", fontsize=9)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    save(fig, "backtest_discount")


def fig_diagnostics(steps, summary):
    """The two failure modes the backtest exposed."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # --- optimiser proposals against the observed floor --------------------
    ax = axes[0]
    ax.hist(
        steps["model_total_opt"],
        bins=60,
        color=INK,
        alpha=0.85,
        label="optimised proposals",
    )
    ax.axvline(0.0, color=ACCENT, lw=1.6, label="physical zero")
    ax.set_xlabel("predicted emissions (t CO$_2$/h)")
    ax.set_ylabel("horizon steps")
    ax.set_title(
        f"{summary['negative_prediction_pct']:.0f}% of proposals are below zero",
        fontsize=9,
    )
    ax.legend(fontsize=8)

    # --- balance error against horizon -------------------------------------
    ax = axes[1]
    by_step = steps.groupby("step").apply(
        lambda g: 100 * g["balance_real"].abs().mean() / g["demand_real"].mean(),
        include_groups=False,
    )
    ax.plot(by_step.index * 15, by_step.to_numpy(), color=INK, lw=1.8, marker="o", ms=3)
    ax.axhline(1.0, color=ACCENT, ls="--", lw=1.2, label="1% tolerance the optimiser enforces")
    ax.set_xlabel("minutes ahead")
    ax.set_ylabel("mean |imbalance| (% of demand)")
    ax.set_title("The schedule balances only against its own forecast", fontsize=9)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "backtest_diagnostics")


def main():
    periods = pd.read_parquet(BACKTEST_DIR / "periods.parquet")
    steps = pd.read_parquet(BACKTEST_DIR / "steps.parquet")
    frame = pd.read_parquet(BACKTEST_DIR / "frame.parquet")
    summary = json.loads((BACKTEST_DIR / "summary.json").read_text(encoding="utf-8"))

    cutoffs = [pd.Timestamp(c, tz="UTC") for c in FOLD_CUTOFFS]
    ends = cutoffs[1:] + [frame["timestamp"].iloc[-1]]
    direction = periods[[f"delta_{c}" for c in DECISION_COLUMNS]].mean().to_numpy(float)

    pairs = counterfactual_ranking(frame, cutoffs, ends, direction=direction)
    flip = np.where(pairs["alignment"] < 0, -1.0, 1.0)
    for column in ["observed_delta", "predicted_delta", "marginal_delta", "alignment"]:
        pairs[column] *= flip

    fig_ranking(pairs, summary)
    fig_discount(summary, steps, periods)
    fig_diagnostics(steps, summary)
    print(f"wrote three figures to {FIGURES}")


if __name__ == "__main__":
    main()
