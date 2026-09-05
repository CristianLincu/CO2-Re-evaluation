"""Shared configuration: horizon, resources, physical limits, feature order.

Everything the optimizer and the models need to agree on lives here so that
training and serving cannot drift apart.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / "data_cache"
DOCS_DATA = ROOT / "docs" / "data"

# --- Horizon -----------------------------------------------------------------
# Denmark settles imbalance in 15-minute periods, so the optimizer works on the
# same grid. Sixteen steps gives a four-hour look-ahead.
RESOLUTION_MIN = 15
HORIZON_STEPS = 16

# --- Resources ---------------------------------------------------------------
# Decision variables, in chromosome gene order.
GENERATION = ["ProductionGe100MW", "ProductionLt100MW"]

# Cross-border links. These are the only exchanges that change the national
# energy balance.
CROSS_BORDER = [
    "Exchange_DK1_DE",
    "Exchange_DK1_NL",
    "Exchange_DK1_GB",
    "Exchange_DK1_NO",
    "Exchange_DK1_SE",
    "Exchange_DK2_DE",
    "Exchange_DK2_SE",
    "Exchange_Bornholm_SE",
]

# The Great Belt link joins the two Danish bidding zones. It is internal, so it
# nets out of the national balance -- Energinet excludes it from Exchange_Sum
# for exactly this reason. It stays a decision variable (it carries zonal
# information the CO2 model uses) but it must not appear in the balance.
INTERNAL_LINK = "Exchange_DK1_DK2"

DECISION_COLUMNS = GENERATION + [
    "Exchange_DK1_DE",
    "Exchange_DK1_NL",
    "Exchange_DK1_GB",
    "Exchange_DK1_NO",
    "Exchange_DK1_SE",
    INTERNAL_LINK,
    "Exchange_DK2_DE",
    "Exchange_DK2_SE",
    "Exchange_Bornholm_SE",
]

N_GENES = len(DECISION_COLUMNS)  # 11

# Mask marking which genes contribute to the national energy balance.
BALANCE_MASK = [c != INTERNAL_LINK for c in DECISION_COLUMNS]

# CO2 model features: renewables (not controllable) followed by the decision
# variables, in this exact order.
RENEWABLES = "Renewables"
CO2_FEATURES = [RENEWABLES] + DECISION_COLUMNS

# Monotonic constraints for the CO2 model, aligned with CO2_FEATURES.
#
# The model predicts the *total* emission rate (t CO2/h), not the intensity.
# Under consumption-based accounting every supply source carries a
# non-negative emission factor, so total emissions cannot fall when any source
# is increased. That single constraint removes the spurious "more thermal
# generation implies lower CO2" relationship that an unconstrained tree learns
# from wind-confounded data.
#
# Renewables are left unconstrained: they are not a decision variable, and the
# sign of their effect on *total* emissions is not physically determined.
MONOTONIC_CST = [0] + [1] * N_GENES

# --- Physical limits ---------------------------------------------------------
# Sanity caps (MW), applied on top of the observed operating range. They exist
# to stop a data spike from widening the search space -- metered small-plant
# output has a one-off excursion to 5.9 GW against a 99.9th percentile of
# 946 MW -- not to define capacity, which the observed range does better.
#
# Values reflect what the links actually carry: DK2-DE covers Kontek plus the
# Kriegers Flak Combined Grid Solution, and the German border runs well beyond
# the old 2.5 GW figure.
TECHNICAL_LIMITS = {
    "ProductionGe100MW": (0.0, 2600.0),
    "ProductionLt100MW": (0.0, 1300.0),
    "Exchange_DK1_DE": (-3600.0, 3600.0),
    "Exchange_DK1_NL": (-750.0, 750.0),
    "Exchange_DK1_GB": (-1500.0, 1500.0),
    "Exchange_DK1_NO": (-1700.0, 1700.0),
    "Exchange_DK1_SE": (-800.0, 800.0),
    "Exchange_DK1_DK2": (-650.0, 650.0),
    "Exchange_DK2_DE": (-1100.0, 1100.0),
    "Exchange_DK2_SE": (-1700.0, 1700.0),
    "Exchange_Bornholm_SE": (-60.0, 60.0),
}

# --- Fitness weights ---------------------------------------------------------
# Kept in the spirit of the original formulation: a large penalty for
# infeasibility, a smaller weight on the emissions term.
XI_BALANCE = 1.0e4  # grid imbalance penalty
XI_RAMP = 1.0e4  # ramp violation penalty
XI_SUPPORT = 1.0e4  # out-of-distribution penalty
XI_FLOOR = 5.0e3  # proposals cleaner than anything ever metered
ZETA_CO2 = 1.0e2  # emissions weight
GAMMA_UNCERTAINTY = 5.0e1  # model-uncertainty penalty

# --- Outcome-space plausibility floor ----------------------------------------
# The Mahalanobis guard polices where a *point* sits in feature space. Backtest
# showed it is nearly inert against this optimiser: proposed operating points
# had slightly lower novelty than the ones the grid actually visited, while
# 60% of them implied an emission rate below anything ever metered under
# comparable demand and wind, and 17% were below zero outright.
#
# Monotonicity constrains the shape of the response, not its range, so the
# search can walk off the bottom of the surface while every coordinate still
# looks ordinary. This floor closes that route directly: for the conditions of
# each step, find comparable hours in the record and refuse to believe any
# outcome cleaner than the best of them.
FLOOR_QUANTILE = 0.05
FLOOR_TOLERANCES = (200.0, 400.0, 800.0)  # MW, widened until enough matches
FLOOR_MIN_NEIGHBOURS = 25

BALANCE_TOLERANCE = 0.01  # fraction of forecast demand

# Minimum number of training samples backing a leaf for it to be trusted.
MIN_LEAF_SUPPORT = 20

# Mahalanobis novelty threshold, as a quantile of the training distribution.
NOVELTY_QUANTILE = 0.99

# --- GA hyperparameters ------------------------------------------------------
# Chosen by scaling test: 16x150x200 costs about 57 s and lands within 1.5% of
# what four times the budget achieves, comfortably inside the scheduled job.
GA_ISLANDS = 16
GA_POPULATION = 150
GA_EPOCHS = 200
GA_BETA = 4.0  # selection pressure
GA_MUTATION_RATE = 0.15
GA_MIGRATION_EVERY = 20
GA_ELITE = 2
