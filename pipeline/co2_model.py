"""A CO2 model a search heuristic cannot game.

The original framework fitted a fully grown decision tree to CO2 *intensity*
and let a genetic algorithm minimise it. Three properties made that unsafe:

* a fully grown tree is a lookup table with one leaf per training row, so the
  optimiser retrieves historical rows rather than reasoning about dispatch;
* intensity is confounded -- large thermal output coincides with high wind in
  Denmark, so an unconstrained tree learns "more thermal implies less CO2",
  which is precisely the relationship the optimiser then exploited;
* trees are piecewise constant, so nothing stopped the optimiser evaluating
  combinations the grid has never been in.

This module addresses all three.

The target becomes the *total* emission rate rather than the intensity. Under
consumption-based accounting every supply source carries a non-negative
emission factor, so total emissions are monotonically non-decreasing in every
source. That is imposed as a hard constraint, which removes the confounded
relationship by construction.

The estimator is a gradient-boosted ensemble of regression trees rather than
one tree. A single constrained tree lacks the capacity to fit the data without
leaning on the confound; boosting recovers it, and the constrained ensemble is
in fact more accurate than the original unconstrained tree was under honest
chronological evaluation.

Every prediction carries two safety signals: a companion partitioning tree
supplies the local training support and dispersion, and a Mahalanobis score
measures distance from the training distribution. The optimiser is penalised
for leaving either envelope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor

from pipeline.config import CO2_FEATURES, MIN_LEAF_SUPPORT, MONOTONIC_CST, NOVELTY_QUANTILE


def total_emission_rate(co2_intensity, demand):
    """Convert intensity (g/kWh) and demand (MW) to total emissions (t/h)."""
    return np.asarray(co2_intensity) * np.asarray(demand) / 1000.0


def intensity_from_total(total_tph, demand):
    """Inverse of :func:`total_emission_rate`, guarding against zero demand."""
    demand = np.asarray(demand, dtype=float)
    safe = np.where(np.abs(demand) < 1e-6, np.nan, demand)
    return 1000.0 * np.asarray(total_tph) / safe


@dataclass
class SafeCO2Model:
    """Monotone ensemble plus the statistics needed to police its outputs."""

    booster: xgb.Booster
    partition_tree: DecisionTreeRegressor
    leaf_std: np.ndarray  # indexed by partition-tree node id
    leaf_count: np.ndarray
    mean: np.ndarray
    inv_cov: np.ndarray
    novelty_threshold: float
    feature_names: list[str]
    min_leaf_support: int = MIN_LEAF_SUPPORT

    def evaluate(self, X):
        """Total emissions, dispersion, support and novelty.

        ``X`` is ``(n, 12)`` in :data:`CO2_FEATURES` order. Returns
        ``(total_tph, dispersion_tph, support, novelty)``.
        """
        X = np.ascontiguousarray(X, dtype=np.float32)

        total = self.booster.inplace_predict(X)

        nodes = self.partition_tree.apply(X)
        dispersion = self.leaf_std[nodes]
        support = self.leaf_count[nodes]

        centred = X.astype(np.float64) - self.mean
        novelty = np.einsum("ij,jk,ik->i", centred, self.inv_cov, centred)

        return total, dispersion, support, novelty

    def predict_total(self, X):
        return self.booster.inplace_predict(np.ascontiguousarray(X, dtype=np.float32))

    def predict_intensity(self, X, demand):
        return intensity_from_total(self.predict_total(X), demand)

    def is_supported(self, support, novelty):
        return (support >= self.min_leaf_support) & (novelty <= self.novelty_threshold)


def make_monotone_ensemble(**params):
    """Gradient-boosted trees carrying the physical monotonicity."""
    defaults = {
        "objective": "reg:squarederror",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 20,
        "reg_lambda": 5.0,
        "tree_method": "hist",
        "monotone_constraints": tuple(MONOTONIC_CST),
        "n_jobs": -1,
        "random_state": 9,
    }
    defaults.update(params)
    return xgb.XGBRegressor(**defaults)


def make_partition_tree(**params):
    """Depth-capped monotone tree used only to localise training support."""
    defaults = {
        "criterion": "squared_error",
        "max_leaf_nodes": 512,
        "min_samples_leaf": MIN_LEAF_SUPPORT,
        "monotonic_cst": np.array(MONOTONIC_CST, dtype=np.int8),
        "random_state": 9,
    }
    defaults.update(params)
    return DecisionTreeRegressor(**defaults)


def _leaf_statistics(tree, X, y):
    """Per-leaf dispersion and support from the training rows."""
    nodes = tree.apply(X)
    n_nodes = tree.tree_.node_count

    count = np.bincount(nodes, minlength=n_nodes).astype(np.float64)
    sum_y = np.bincount(nodes, weights=y, minlength=n_nodes)
    sum_y2 = np.bincount(nodes, weights=y * y, minlength=n_nodes)

    safe = np.maximum(count, 1.0)
    mean = np.where(count > 0, sum_y / safe, 0.0)
    var = np.where(count > 0, sum_y2 / safe - mean**2, 0.0)

    return np.sqrt(np.clip(var, 0.0, None)), count


def _novelty_model(X, quantile=NOVELTY_QUANTILE):
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = cov + np.eye(cov.shape[0]) * 1e-6 * np.trace(cov) / cov.shape[0]
    inv_cov = np.linalg.pinv(cov)

    centred = X - mean
    d2 = np.einsum("ij,jk,ik->i", centred, inv_cov, centred)
    return mean, inv_cov, float(np.quantile(d2, quantile))


def build_safe_model(ensemble, partition_tree, X, y, feature_names=None):
    """Bundle a fitted ensemble with its support and novelty machinery."""
    std, count = _leaf_statistics(partition_tree, X, y)
    mean, inv_cov, threshold = _novelty_model(np.asarray(X, dtype=np.float64))

    booster = ensemble.get_booster()
    booster.set_param({"nthread": 1})  # avoid thread thrash on tiny batches

    return SafeCO2Model(
        booster=booster,
        partition_tree=partition_tree,
        leaf_std=std,
        leaf_count=count,
        mean=mean,
        inv_cov=inv_cov,
        novelty_threshold=threshold,
        feature_names=list(feature_names or CO2_FEATURES),
    )
