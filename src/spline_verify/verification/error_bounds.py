"""Error budget analysis for safety verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class ErrorBudget:
    """Tracks all sources of error in the verification pipeline.

    The total error epsilon bounds the difference between the computed
    minimum of the spline approximation and the true minimum of F_T:

        |min F̃_T - min F_T| <= epsilon

    For safety verification:
    - If min F̃_T > epsilon: SAFE (true minimum is positive)
    - If min F̃_T < 0: UNSAFE (found a counterexample)
    - Otherwise: UNKNOWN (cannot certify either way)
    """
    integration_error: float = 0.0
    sampling_error: float = 0.0
    approximation_error: float = 0.0
    minimization_error: float = 0.0

    # Parameters used to estimate errors
    _params: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Total error bound."""
        return (
            self.integration_error +
            self.sampling_error +
            self.approximation_error +
            self.minimization_error
        )

    def set_integration_error(
        self,
        method: str,
        step_size: float,
        lipschitz: float,
        time_horizon: float
    ) -> ErrorBudget:
        """Estimate integration error.

        For explicit methods, error ~ O(h^p * L * T) where:
        - h = step size
        - p = order of method
        - L = Lipschitz constant of dynamics
        - T = time horizon

        Args:
            method: Integration method name.
            step_size: Time step used.
            lipschitz: Lipschitz constant of dynamics.
            time_horizon: Total integration time.

        Returns:
            self for method chaining.
        """
        order = {'euler': 1, 'rk4': 4, 'rk45': 5, 'adams': 4}.get(method.lower(), 4)

        # Conservative error estimate
        self.integration_error = (step_size ** order) * lipschitz * time_horizon * 10

        self._params['integration'] = {
            'method': method,
            'step_size': step_size,
            'lipschitz': lipschitz,
            'time_horizon': time_horizon,
            'order': order
        }

        return self

    def set_sampling_error(
        self,
        n_samples: int,
        n_dims: int,
        lipschitz: float,
        domain_diameter: float
    ) -> ErrorBudget:
        """Estimate sampling error.

        Error from not sampling at the true minimum. Assumes quasi-uniform
        sampling.

        Args:
            n_samples: Number of samples.
            n_dims: Dimension of domain.
            lipschitz: Lipschitz constant of F_T.
            domain_diameter: Diameter of initial set.

        Returns:
            self for method chaining.
        """
        # Expected maximum spacing between samples for quasi-uniform distribution
        # Scales as diameter / n^(1/d)
        max_spacing = domain_diameter / (n_samples ** (1 / n_dims))

        # Error bounded by L * spacing
        self.sampling_error = lipschitz * max_spacing

        self._params['sampling'] = {
            'n_samples': n_samples,
            'n_dims': n_dims,
            'lipschitz': lipschitz,
            'max_spacing': max_spacing
        }

        return self

    def set_approximation_error(
        self,
        residual_norm: float,
        smoothness_factor: float = 1.0
    ) -> ErrorBudget:
        """Set approximation error from spline fitting.

        Args:
            residual_norm: L-infinity norm of residuals at sample points.
            smoothness_factor: Multiplier based on expected smoothness.
                              Set higher for less smooth functions.

        Returns:
            self for method chaining.
        """
        self.approximation_error = residual_norm * smoothness_factor

        self._params['approximation'] = {
            'residual_norm': residual_norm,
            'smoothness_factor': smoothness_factor
        }

        return self

    def set_minimization_error(
        self,
        tolerance: float,
        n_starts: int = 1
    ) -> ErrorBudget:
        """Set error from optimization.

        Args:
            tolerance: Convergence tolerance of optimizer.
            n_starts: Number of random starts (reduces error).

        Returns:
            self for method chaining.
        """
        # With multiple starts, probability of missing global min decreases
        self.minimization_error = tolerance * (1 + 1 / np.sqrt(n_starts))

        self._params['minimization'] = {
            'tolerance': tolerance,
            'n_starts': n_starts
        }

        return self

    def estimate_from_samples(
        self,
        points: np.ndarray,
        values: np.ndarray,
        approximation_values: np.ndarray,
        domain_bounds: tuple[np.ndarray, np.ndarray]
    ) -> ErrorBudget:
        """Automatically estimate errors from sampled data.

        Args:
            points: Sample points, shape (n, d).
            values: True F_T values at samples.
            approximation_values: Spline values at samples.
            domain_bounds: (lower, upper) bounds of domain.

        Returns:
            self for method chaining.
        """
        n_samples, n_dims = points.shape
        lower, upper = domain_bounds

        # Approximation error from residuals
        residuals = np.abs(values - approximation_values)
        self.approximation_error = float(np.max(residuals))

        # Estimate Lipschitz constant from samples
        if n_samples > 1:
            from scipy.spatial import KDTree
            tree = KDTree(points)

            lip_estimates = []
            for i in range(min(100, n_samples)):
                dists, indices = tree.query(points[i], k=min(5, n_samples))
                for j, idx in enumerate(indices[1:]):  # Skip self
                    if dists[j+1] > 1e-10:
                        lip = np.abs(values[i] - values[idx]) / dists[j+1]
                        lip_estimates.append(lip)

            lipschitz = np.percentile(lip_estimates, 95) if lip_estimates else 1.0
        else:
            lipschitz = 1.0

        # Domain diameter
        diameter = float(np.linalg.norm(upper - lower))

        # Sampling error
        self.set_sampling_error(n_samples, n_dims, lipschitz, diameter)

        self._params['estimated_lipschitz'] = lipschitz

        return self

    def summary(self) -> str:
        """Return a human-readable summary of the error budget."""
        lines = [
            "Error Budget Summary",
            "=" * 40,
            f"Integration error:    {self.integration_error:.2e}",
            f"Sampling error:       {self.sampling_error:.2e}",
            f"Approximation error:  {self.approximation_error:.2e}",
            f"Minimization error:   {self.minimization_error:.2e}",
            "-" * 40,
            f"TOTAL ERROR (ε):      {self.total:.2e}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ErrorBudget(total={self.total:.2e})"
