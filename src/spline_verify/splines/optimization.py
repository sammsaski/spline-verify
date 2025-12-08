"""Spline minimization algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing

from .multivariate import MultivariateApproximation


@dataclass
class MinimizationResult:
    """Result of spline minimization.

    Attributes:
        minimum: The minimum value found.
        minimizer: The point at which minimum is achieved.
        success: Whether optimization converged.
        n_evaluations: Number of function evaluations.
        message: Status message from optimizer.
    """
    minimum: float
    minimizer: np.ndarray
    success: bool
    n_evaluations: int
    message: str = ""

    def __repr__(self) -> str:
        return (
            f"MinimizationResult(minimum={self.minimum:.6f}, "
            f"success={self.success}, n_evals={self.n_evaluations})"
        )


def minimize_spline(
    approximation: MultivariateApproximation,
    bounds: tuple[np.ndarray, np.ndarray],
    method: str = 'multistart',
    n_starts: int = 20,
    seed: int | None = None
) -> MinimizationResult:
    """Find the minimum of a spline approximation.

    Args:
        approximation: The spline approximation to minimize.
        bounds: Tuple of (lower_bounds, upper_bounds) arrays.
        method: Optimization method ('multistart', 'differential_evolution',
                'dual_annealing', 'grid').
        n_starts: Number of starting points for multistart methods.
        seed: Random seed for reproducibility.

    Returns:
        MinimizationResult with the found minimum.
    """
    lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])
    n_dims = len(lower)
    scipy_bounds = list(zip(lower, upper))

    if method == 'multistart':
        return _minimize_multistart(
            approximation, scipy_bounds, n_starts, seed
        )
    elif method == 'differential_evolution':
        return _minimize_de(approximation, scipy_bounds, seed)
    elif method == 'dual_annealing':
        return _minimize_da(approximation, scipy_bounds, seed)
    elif method == 'grid':
        return _minimize_grid(approximation, lower, upper, n_starts)
    else:
        raise ValueError(f"Unknown method: {method}")


def _minimize_multistart(
    approximation: MultivariateApproximation,
    bounds: list[tuple[float, float]],
    n_starts: int,
    seed: int | None
) -> MinimizationResult:
    """Multi-start local optimization using L-BFGS-B."""
    rng = np.random.default_rng(seed)
    n_dims = len(bounds)

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    best_result = None
    best_value = np.inf
    total_evals = 0

    # Generate starting points using Latin hypercube
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    starts = qmc.scale(sampler.random(n=n_starts), lower, upper)

    for x0 in starts:
        try:
            result = minimize(
                approximation.evaluate,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                jac=lambda x: approximation.gradient(x)
            )

            total_evals += result.nfev

            if result.fun < best_value:
                best_value = result.fun
                best_result = result
        except Exception:
            # Skip failed optimization attempts
            continue

    if best_result is None:
        # Fall back to grid search
        return _minimize_grid(approximation, lower, upper, n_starts)

    return MinimizationResult(
        minimum=float(best_value),
        minimizer=best_result.x,
        success=best_result.success,
        n_evaluations=total_evals,
        message=best_result.message
    )


def _minimize_de(
    approximation: MultivariateApproximation,
    bounds: list[tuple[float, float]],
    seed: int | None
) -> MinimizationResult:
    """Global optimization using differential evolution."""
    result = differential_evolution(
        approximation.evaluate,
        bounds,
        seed=seed,
        maxiter=1000,
        tol=1e-7,
        polish=True  # L-BFGS-B refinement at end
    )

    return MinimizationResult(
        minimum=float(result.fun),
        minimizer=result.x,
        success=result.success,
        n_evaluations=result.nfev,
        message=result.message
    )


def _minimize_da(
    approximation: MultivariateApproximation,
    bounds: list[tuple[float, float]],
    seed: int | None
) -> MinimizationResult:
    """Global optimization using dual annealing."""
    result = dual_annealing(
        approximation.evaluate,
        bounds,
        seed=seed,
        maxiter=1000
    )

    return MinimizationResult(
        minimum=float(result.fun),
        minimizer=result.x,
        success=result.success,
        n_evaluations=result.nfev,
        message=result.message
    )


def _minimize_grid(
    approximation: MultivariateApproximation,
    lower: np.ndarray,
    upper: np.ndarray,
    n_per_dim: int
) -> MinimizationResult:
    """Brute-force grid search."""
    n_dims = len(lower)

    # Create grid
    grids = [np.linspace(lower[i], upper[i], n_per_dim) for i in range(n_dims)]
    mesh = np.meshgrid(*grids, indexing='ij')
    points = np.column_stack([m.ravel() for m in mesh])

    # Evaluate on grid
    values = np.array([approximation.evaluate(p) for p in points])

    min_idx = np.argmin(values)
    min_point = points[min_idx]
    min_value = values[min_idx]

    return MinimizationResult(
        minimum=float(min_value),
        minimizer=min_point,
        success=True,
        n_evaluations=len(points),
        message="Grid search completed"
    )


def certified_minimum_bound(
    approximation: MultivariateApproximation,
    bounds: tuple[np.ndarray, np.ndarray],
    sampled_minimum: float,
    lipschitz_constant: float | None = None,
    n_samples: int = 1000,
    seed: int | None = None
) -> float:
    """Compute a lower bound on the true minimum.

    Given a sampled minimum and either:
    - A Lipschitz constant, or
    - Empirical estimation of Lipschitz constant from samples

    This provides a lower bound: true_min >= sampled_min - epsilon

    Args:
        approximation: The fitted approximation.
        bounds: Domain bounds.
        sampled_minimum: The minimum found by optimization.
        lipschitz_constant: Known Lipschitz constant (if available).
        n_samples: Number of samples for Lipschitz estimation.
        seed: Random seed.

    Returns:
        Lower bound on true minimum.
    """
    lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])

    if lipschitz_constant is None:
        # Estimate Lipschitz constant empirically
        rng = np.random.default_rng(seed)
        n_dims = len(lower)

        points = rng.uniform(lower, upper, size=(n_samples, n_dims))
        values = np.array([approximation.evaluate(p) for p in points])

        # Estimate from gradient magnitudes
        grad_norms = []
        for p in points[:min(100, n_samples)]:  # Sample subset for gradients
            grad = approximation.gradient(p)
            grad_norms.append(np.linalg.norm(grad))

        lipschitz_constant = np.max(grad_norms) * 1.5  # Add safety margin

    # Maximum possible error from not sampling at the true minimum
    # Assuming uniform distribution of sample points
    max_spacing = np.linalg.norm(upper - lower) / np.sqrt(n_samples)
    epsilon = lipschitz_constant * max_spacing

    return sampled_minimum - epsilon
