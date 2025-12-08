"""Sampling strategies for initial sets."""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import qmc

if TYPE_CHECKING:
    from .sets import Set


class SamplingStrategy(Enum):
    """Available sampling strategies."""
    UNIFORM = auto()       # Uniform random sampling
    GRID = auto()          # Regular grid sampling
    LATIN_HYPERCUBE = auto()  # Latin hypercube sampling
    SOBOL = auto()         # Sobol low-discrepancy sequence
    HALTON = auto()        # Halton low-discrepancy sequence


def sample_set(
    s: Set,
    n: int,
    strategy: SamplingStrategy = SamplingStrategy.LATIN_HYPERCUBE,
    seed: int | None = None
) -> np.ndarray:
    """Sample points from a set using the specified strategy.

    Args:
        s: Set to sample from.
        n: Number of points to sample.
        strategy: Sampling strategy to use.
        seed: Random seed for reproducibility.

    Returns:
        Array of sampled points, shape (n, n_dims).
    """
    if strategy == SamplingStrategy.UNIFORM:
        return s.sample(n, seed=seed)

    elif strategy == SamplingStrategy.GRID:
        return _sample_grid(s, n)

    elif strategy == SamplingStrategy.LATIN_HYPERCUBE:
        return _sample_qmc(s, n, 'latin', seed)

    elif strategy == SamplingStrategy.SOBOL:
        return _sample_qmc(s, n, 'sobol', seed)

    elif strategy == SamplingStrategy.HALTON:
        return _sample_qmc(s, n, 'halton', seed)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def _sample_grid(s: Set, n: int) -> np.ndarray:
    """Sample on a regular grid, accepting only points in the set."""
    from .sets import HyperRectangle

    # Need bounding box for grid
    if isinstance(s, HyperRectangle):
        bounds = s
    elif hasattr(s, 'bounds') and s.bounds is not None:
        bounds = s.bounds
    else:
        raise ValueError(
            "Grid sampling requires a HyperRectangle or a set with bounds"
        )

    # Determine grid resolution per dimension
    n_per_dim = int(np.ceil(n ** (1 / s.n_dims)))

    # Create grid
    grids = [
        np.linspace(bounds.lower[i], bounds.upper[i], n_per_dim)
        for i in range(s.n_dims)
    ]
    mesh = np.meshgrid(*grids, indexing='ij')
    candidates = np.column_stack([m.ravel() for m in mesh])

    # Filter to points in set
    if isinstance(s, HyperRectangle):
        # All grid points are in the rectangle
        valid = candidates
    else:
        mask = np.array([s.contains(x) for x in candidates])
        valid = candidates[mask]

    if len(valid) < n:
        # Not enough grid points, return what we have
        return valid
    elif len(valid) > n:
        # Subsample uniformly
        indices = np.linspace(0, len(valid) - 1, n, dtype=int)
        return valid[indices]
    else:
        return valid


def _sample_qmc(
    s: Set,
    n: int,
    method: str,
    seed: int | None
) -> np.ndarray:
    """Sample using quasi-Monte Carlo methods."""
    from .sets import HyperRectangle

    # Need bounding box for QMC
    if isinstance(s, HyperRectangle):
        bounds = s
    elif hasattr(s, 'bounds') and s.bounds is not None:
        bounds = s.bounds
    else:
        raise ValueError(
            f"{method.upper()} sampling requires a HyperRectangle "
            "or a set with bounds"
        )

    # Create QMC sampler
    if method == 'latin':
        sampler = qmc.LatinHypercube(d=s.n_dims, seed=seed)
    elif method == 'sobol':
        sampler = qmc.Sobol(d=s.n_dims, seed=seed)
    elif method == 'halton':
        sampler = qmc.Halton(d=s.n_dims, seed=seed)
    else:
        raise ValueError(f"Unknown QMC method: {method}")

    # For non-rectangular sets, oversample and filter
    if isinstance(s, HyperRectangle):
        sample_unit = sampler.random(n=n)
        return qmc.scale(sample_unit, bounds.lower, bounds.upper)
    else:
        # Oversample and reject
        samples = []
        batch_multiplier = 2

        while len(samples) < n:
            sample_unit = sampler.random(n=n * batch_multiplier)
            candidates = qmc.scale(sample_unit, bounds.lower, bounds.upper)

            for x in candidates:
                if s.contains(x):
                    samples.append(x)
                    if len(samples) >= n:
                        break

            batch_multiplier *= 2
            if batch_multiplier > 64:
                raise RuntimeError(
                    f"Could not generate {n} samples from set after "
                    "extensive rejection sampling. Set may be too small "
                    "relative to bounding box."
                )

        return np.array(samples[:n])


def adaptive_sample(
    s: Set,
    objective_func,
    n_initial: int = 100,
    n_refine: int = 50,
    seed: int | None = None
) -> np.ndarray:
    """Adaptively sample, adding more points where objective varies rapidly.

    Args:
        s: Set to sample from.
        objective_func: Function to evaluate (used to guide refinement).
        n_initial: Number of initial samples.
        n_refine: Number of refinement samples to add.
        seed: Random seed.

    Returns:
        Array of sampled points, shape (n_initial + n_refine, n_dims).
    """
    rng = np.random.default_rng(seed)

    # Initial sampling
    samples = sample_set(s, n_initial, SamplingStrategy.LATIN_HYPERCUBE, seed)
    values = np.array([objective_func(x) for x in samples])

    # Refinement: add points near where gradient is high
    for _ in range(n_refine):
        # Estimate local variation using nearest neighbor differences
        from scipy.spatial import KDTree
        tree = KDTree(samples)

        # Find points with high local variation
        variations = []
        for i, (x, v) in enumerate(zip(samples, values)):
            _, neighbors = tree.query(x, k=min(5, len(samples)))
            neighbor_vals = values[neighbors]
            variation = np.std(neighbor_vals)
            variations.append(variation)

        variations = np.array(variations)

        # Sample new point near high-variation region
        probs = variations / (variations.sum() + 1e-10)
        idx = rng.choice(len(samples), p=probs)
        center = samples[idx]

        # Add small perturbation
        from .sets import HyperRectangle
        if isinstance(s, HyperRectangle):
            scale = (s.upper - s.lower) * 0.1
        else:
            scale = np.ones(s.n_dims) * 0.1

        new_point = center + rng.normal(0, 1, s.n_dims) * scale

        # Project back to set if needed (simple clipping for rectangles)
        if isinstance(s, HyperRectangle):
            new_point = np.clip(new_point, s.lower, s.upper)

        if s.contains(new_point):
            samples = np.vstack([samples, new_point])
            values = np.append(values, objective_func(new_point))

    return samples
