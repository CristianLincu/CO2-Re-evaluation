"""Genetic optimisation of a dispatch trajectory.

The original formulation evolved each of five time points independently, with
a cosine-similarity term meant to keep consecutive solutions coherent. Cosine
similarity is scale invariant, so it could not do that job: the deployed
system produced 1,170 MW swings on a single interconnector between adjacent
five-minute steps.

Here a chromosome is the whole trajectory -- a ``T x G`` matrix flattened into
one genome -- and coherence is imposed directly through ramp-rate limits
between consecutive steps. That makes the search space much larger and
genuinely coupled, which is precisely the regime where an evolutionary
heuristic earns its place.

Three things keep it fast enough to run inside a scheduled job:

* the ensemble of independent optimisers becomes an island model evaluated as
  a single batched array, so 12 populations cost one vectorised pass rather
  than 12 sequential runs;
* every chromosome is passed through a repair operator that enforces bounds,
  ramp limits and the energy balance by construction, so the search spends its
  effort on emissions rather than on rediscovering feasibility;
* CO2, dispersion, leaf support and novelty all come out of a single tree
  traversal per generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.config import (
    BALANCE_MASK,
    BALANCE_TOLERANCE,
    DECISION_COLUMNS,
    GA_BETA,
    GA_ELITE,
    GA_EPOCHS,
    GA_ISLANDS,
    GA_MIGRATION_EVERY,
    GA_MUTATION_RATE,
    GA_POPULATION,
    GAMMA_UNCERTAINTY,
    XI_BALANCE,
    XI_FLOOR,
    XI_RAMP,
    XI_SUPPORT,
    ZETA_CO2,
)

BALANCE_IDX = np.flatnonzero(np.array(BALANCE_MASK))


@dataclass
class ProblemSpec:
    """Everything that defines one optimisation instance."""

    demand: np.ndarray  # (T,) forecast demand, MW
    renewables: np.ndarray  # (T,) forecast renewables, MW
    current: np.ndarray  # (G,) latest observed distribution
    lower: np.ndarray  # (G,)
    upper: np.ndarray  # (G,)
    ramp: np.ndarray  # (G,) max change per step, MW
    # Cleanest emission rate on record for each step's conditions. Fixed for a
    # solve, so it is computed once by the caller from the CO2 model.
    floor: np.ndarray | None = None  # (T,) t/h
    horizon: int = field(init=False)

    def __post_init__(self):
        self.horizon = len(self.demand)
        # The envelope must contain the present state. A link already carrying
        # more than its assumed limit cannot be ordered below that limit within
        # one step, and a box that excludes the starting point makes the ramp
        # constraint unsatisfiable at t = 0.
        self.lower = np.minimum(self.lower, self.current)
        self.upper = np.maximum(self.upper, self.current)


@dataclass
class OptimizationResult:
    trajectory: np.ndarray  # (T, G)
    intensity: np.ndarray  # (T,) g/kWh
    total_tph: np.ndarray  # (T,)
    uncertainty: np.ndarray  # (T,) g/kWh
    balance_error: np.ndarray  # (T,) MW
    feasible: bool
    supported: np.ndarray  # (T,) bool
    within_floor: np.ndarray  # (T,) bool: outcome has historical precedent
    fitness: float
    history: np.ndarray


# --- feasibility -------------------------------------------------------------


def repair(pop: np.ndarray, spec: ProblemSpec) -> np.ndarray:
    """Project a population onto the feasible set.

    Walks the horizon forward. At each step the admissible box is the
    intersection of the technical limits with the ramp window around the
    previous step; the residual energy imbalance is then distributed across
    the balance-carrying resources in proportion to their remaining headroom.

    Operates on ``(N, T, G)`` and returns the same shape.
    """
    out = np.clip(pop, spec.lower, spec.upper)
    target = spec.demand - spec.renewables  # distributable energy per step
    tol = BALANCE_TOLERANCE * spec.demand

    previous = np.broadcast_to(spec.current, (out.shape[0], out.shape[2])).copy()

    for t in range(spec.horizon):
        low = np.maximum(spec.lower, previous - spec.ramp)
        high = np.minimum(spec.upper, previous + spec.ramp)
        step = np.clip(out[:, t, :], low, high)

        # Two passes are enough: the first move may saturate some resources,
        # the second redistributes what is left.
        for _ in range(2):
            residual = target[t] - step[:, BALANCE_IDX].sum(axis=1)
            need = np.abs(residual) > (tol[t] * 0.25)
            if not need.any():
                break

            headroom = np.where(
                residual[:, None] > 0,
                high[:, BALANCE_IDX] - step[:, BALANCE_IDX],
                step[:, BALANCE_IDX] - low[:, BALANCE_IDX],
            )
            total_headroom = headroom.sum(axis=1, keepdims=True)
            share = np.divide(
                headroom,
                total_headroom,
                out=np.zeros_like(headroom),
                where=total_headroom > 1e-9,
            )
            step[:, BALANCE_IDX] += residual[:, None] * share
            step = np.clip(step, low, high)

        out[:, t, :] = step
        previous = step

    return out


# --- fitness -----------------------------------------------------------------


def evaluate(pop: np.ndarray, spec: ProblemSpec, model):
    """Score a population of trajectories.

    ``pop`` is ``(N, T, G)``. Returns the fitness vector plus the diagnostic
    pieces, all computed in one tree traversal.
    """
    n, horizon, n_genes = pop.shape

    renewables = np.broadcast_to(spec.renewables[None, :, None], (n, horizon, 1))
    features = np.concatenate([renewables, pop], axis=2).reshape(n * horizon, n_genes + 1)

    total, dispersion, support, novelty = model.evaluate(features)
    total = total.reshape(n, horizon)
    dispersion = dispersion.reshape(n, horizon)
    support = support.reshape(n, horizon)
    novelty = novelty.reshape(n, horizon)

    demand = spec.demand[None, :]
    intensity = 1000.0 * total / demand
    dispersion_intensity = 1000.0 * dispersion / demand

    # Grid balance: indicator, scaled by how badly it is missed so the
    # heuristic has something to descend rather than a flat plateau.
    supplied = pop[:, :, BALANCE_IDX].sum(axis=2) + spec.renewables[None, :]
    balance_error = supplied - demand
    tolerance = BALANCE_TOLERANCE * demand
    balance_excess = np.maximum(np.abs(balance_error) - tolerance, 0.0)
    balance_penalty = np.where(balance_excess > 0, 1.0 + balance_excess / tolerance, 0.0)

    # Ramp coherence between consecutive steps, anchored on the current state.
    # The repair operator clips exactly onto the ramp boundary, so a small
    # tolerance keeps floating-point dust from registering as a violation.
    previous = np.concatenate(
        [np.broadcast_to(spec.current, (n, 1, n_genes)), pop[:, :-1, :]], axis=1
    )
    allowance = spec.ramp * (1.0 + 1e-9) + 1e-6
    ramp_excess = np.maximum(np.abs(pop - previous) - allowance, 0.0)
    ramp_penalty = (ramp_excess / np.maximum(spec.ramp, 1e-6)).sum(axis=2)

    # Stay where the model has evidence.
    unsupported = (support < model.min_leaf_support).astype(float)
    novel = np.maximum(novelty / model.novelty_threshold - 1.0, 0.0)
    support_penalty = unsupported + novel

    # Stay where the *outcome* has precedent. The novelty term above asks
    # whether these coordinates look familiar; this asks whether anyone has
    # ever run this clean on a day like today. Backtest showed the first
    # question is far too easy to pass while failing the second.
    if spec.floor is None:
        floor_penalty = np.zeros_like(intensity)
    else:
        floor = spec.floor[None, :]
        deficit = np.maximum(floor - total, 0.0)
        floor_penalty = np.where(
            np.isfinite(floor) & (deficit > 0),
            1.0 + deficit / np.maximum(np.abs(floor), 1.0),
            0.0,
        )

    fitness = (
        XI_BALANCE * balance_penalty.mean(axis=1)
        + XI_RAMP * ramp_penalty.mean(axis=1)
        + XI_SUPPORT * support_penalty.mean(axis=1)
        + XI_FLOOR * floor_penalty.mean(axis=1)
        + ZETA_CO2 * intensity.mean(axis=1)
        + GAMMA_UNCERTAINTY * dispersion_intensity.mean(axis=1)
    )

    diagnostics = {
        "intensity": intensity,
        "total": total,
        "dispersion": dispersion_intensity,
        "balance_error": balance_error,
        "balance_penalty": balance_penalty,
        "ramp_penalty": ramp_penalty,
        "floor_penalty": floor_penalty,
        "support": support,
        "novelty": novelty,
        "supported": (support >= model.min_leaf_support)
        & (novelty <= model.novelty_threshold),
    }
    return fitness, diagnostics


# --- evolutionary operators ---------------------------------------------------


def _seed_population(spec: ProblemSpec, islands: int, size: int, rng):
    """Half the population perturbs the current state, half explores freely."""
    shape = (islands * size, spec.horizon, len(spec.current))

    explore = rng.uniform(
        low=np.broadcast_to(spec.lower, shape),
        high=np.broadcast_to(spec.upper, shape),
    )

    # A correlated random walk away from the present state keeps the
    # perturbed half both diverse and physically plausible.
    drift = rng.normal(scale=spec.ramp * 0.5, size=shape).cumsum(axis=1)
    perturb = np.clip(spec.current + drift, spec.lower, spec.upper)

    take_perturb = rng.random(shape[0]) < 0.5
    pop = np.where(take_perturb[:, None, None], perturb, explore)
    return repair(pop, spec)


def _rank_probabilities(size: int, beta: float):
    ranks = np.arange(size)
    scores = -beta * (ranks - ranks.mean()) / max(ranks.std(), 1e-9)
    scores -= scores.max()
    weights = np.exp(scores)
    return weights / weights.sum()


def _next_generation(pop, fitness, spec, rng, probabilities, mutation_rate, elite):
    """Selection, uniform crossover, and mutation, batched across islands."""
    islands, size = pop.shape[0], pop.shape[1]
    order = np.argsort(fitness, axis=1)
    ranked = np.take_along_axis(pop, order[:, :, None, None], axis=1)

    n_children = size - elite
    parents = rng.choice(size, size=(islands, 2, n_children), p=probabilities)

    rows = np.arange(islands)[:, None]
    father = ranked[rows, parents[:, 0, :]]
    mother = ranked[rows, parents[:, 1, :]]

    # Uniform crossover at the level of (step, gene) so trajectories can mix
    # both across resources and across time.
    mask = rng.random(father.shape) < 0.5
    children = np.where(mask, father, mother)

    # Creep mutation refines locally; the original only ever replaced a gene
    # with a fresh uniform draw, which cannot converge on a good value.
    creep = rng.random(children.shape) < mutation_rate
    scale = np.broadcast_to(spec.ramp * 0.3, children.shape)
    children = children + creep * rng.normal(scale=scale)

    # A rare full re-draw preserves global exploration.
    jump = rng.random(children.shape) < mutation_rate * 0.1
    fresh = rng.uniform(
        low=np.broadcast_to(spec.lower, children.shape),
        high=np.broadcast_to(spec.upper, children.shape),
    )
    children = np.where(jump, fresh, children)

    children = repair(children.reshape(-1, *children.shape[2:]), spec)
    children = children.reshape(islands, n_children, spec.horizon, -1)

    return np.concatenate([ranked[:, :elite], children], axis=1)


def _migrate(pop, fitness, rng):
    """Send each island's best chromosome to its neighbour."""
    islands = pop.shape[0]
    best = np.argmin(fitness, axis=1)
    worst = np.argmax(fitness, axis=1)
    travellers = pop[np.arange(islands), best].copy()
    destination = np.roll(np.arange(islands), 1)
    pop[destination, worst[destination]] = travellers
    return pop


# --- driver -------------------------------------------------------------------


def optimize(
    spec: ProblemSpec,
    model,
    islands: int = GA_ISLANDS,
    size: int = GA_POPULATION,
    epochs: int = GA_EPOCHS,
    beta: float = GA_BETA,
    mutation_rate: float = GA_MUTATION_RATE,
    elite: int = GA_ELITE,
    migrate_every: int = GA_MIGRATION_EVERY,
    seed: int = 9,
):
    rng = np.random.default_rng(seed)

    flat = _seed_population(spec, islands, size, rng)
    pop = flat.reshape(islands, size, spec.horizon, len(spec.current))
    probabilities = _rank_probabilities(size, beta)

    history = np.empty(epochs)
    best_fitness = np.inf
    best_chromosome = None

    for epoch in range(epochs):
        fitness, _ = evaluate(pop.reshape(-1, *pop.shape[2:]), spec, model)
        fitness = fitness.reshape(islands, size)

        island_best = fitness.min(axis=1)
        generation_best = island_best.min()
        if generation_best < best_fitness:
            best_fitness = float(generation_best)
            i = int(np.argmin(island_best))
            best_chromosome = pop[i, int(np.argmin(fitness[i]))].copy()
        history[epoch] = best_fitness

        if migrate_every and epoch and epoch % migrate_every == 0:
            pop = _migrate(pop, fitness, rng)
            fitness, _ = evaluate(pop.reshape(-1, *pop.shape[2:]), spec, model)
            fitness = fitness.reshape(islands, size)

        pop = _next_generation(pop, fitness, spec, rng, probabilities, mutation_rate, elite)

    final_fitness, diagnostics = evaluate(best_chromosome[None, ...], spec, model)

    return OptimizationResult(
        trajectory=best_chromosome,
        intensity=diagnostics["intensity"][0],
        total_tph=diagnostics["total"][0],
        uncertainty=diagnostics["dispersion"][0],
        balance_error=diagnostics["balance_error"][0],
        feasible=bool(
            (diagnostics["balance_penalty"][0] == 0).all()
            and (diagnostics["ramp_penalty"][0] == 0).all()
        ),
        supported=diagnostics["supported"][0],
        within_floor=diagnostics["floor_penalty"][0] == 0,
        fitness=float(final_fitness[0]),
        history=history,
    )


def hold_current_baseline(spec: ProblemSpec, model):
    """Counterfactual in which the controllable resources simply stay put.

    This is the comparison the optimiser must be judged against. The original
    dashboard reported optimised future emissions against the *present*
    measured value, which credits the optimiser with whatever the wind was
    going to do anyway. Holding dispatch fixed while letting demand and
    renewables follow their forecasts isolates the part that is actually
    attributable to re-dispatch.
    """
    trajectory = repair(
        np.tile(spec.current, (1, spec.horizon, 1)).astype(float), spec
    )
    _, diagnostics = evaluate(trajectory, spec, model)
    return trajectory[0], diagnostics


def derive_limits(frame, quantile=0.999, technical=None):
    """Operating envelope and ramp limits taken from observed behaviour.

    Bounds come from the observed range, capped by technical capacity; ramp
    limits come from the distribution of realised 15-minute changes, which is
    a far better description of what the fleet can actually do than a fixed
    guess.
    """
    lower, upper, ramp = [], [], []
    for column in DECISION_COLUMNS:
        series = frame[column].to_numpy(dtype=float)
        lo = np.nanquantile(series, 1 - quantile)
        hi = np.nanquantile(series, quantile)
        step = np.nanquantile(np.abs(np.diff(series)), 0.99)

        if technical and column in technical:
            tlo, thi = technical[column]
            lo, hi = max(lo, tlo), min(hi, thi)

        lower.append(lo)
        upper.append(hi)
        ramp.append(max(step, 1.0))

    return np.array(lower), np.array(upper), np.array(ramp)
