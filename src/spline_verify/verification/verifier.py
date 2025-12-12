"""Main safety verification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from ..dynamics.base import DynamicsModel
from ..geometry.sets import Set, HyperRectangle
from ..geometry.sampling import sample_set, SamplingStrategy
from ..splines.multivariate import ScatteredDataSpline, fit_objective_spline
from ..splines.bspline import GriddedBSpline, create_spline_approximation
from ..splines.optimization import minimize_spline, MinimizationResult
from .objective import ObjectiveSampler, SampledObjective, compute_objective_bundle
from .error_bounds import ErrorBudget


class VerificationStatus(Enum):
    """Result status of safety verification."""
    SAFE = auto()      # Proven safe: min F_T > epsilon
    UNSAFE = auto()    # Found unsafe trajectory: min F_T <= 0
    UNKNOWN = auto()   # Cannot certify: 0 < min F_T <= epsilon


@dataclass
class VerificationResult:
    """Result of safety verification.

    Attributes:
        status: SAFE, UNSAFE, or UNKNOWN.
        min_objective: Minimum of the spline approximation F̃_T.
        minimizer: Initial condition achieving minimum.
        error_bound: Total error epsilon.
        safety_margin: min_objective - error_bound (positive means safe).
        counterexample: If UNSAFE, the initial condition that reaches unsafe set.
        details: Additional diagnostic information.
    """
    status: VerificationStatus
    min_objective: float
    minimizer: np.ndarray
    error_bound: float
    safety_margin: float
    counterexample: np.ndarray | None = None
    details: dict = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.status == VerificationStatus.SAFE

    @property
    def is_unsafe(self) -> bool:
        return self.status == VerificationStatus.UNSAFE

    def summary(self) -> str:
        """Return human-readable summary."""
        status_str = {
            VerificationStatus.SAFE: "SAFE",
            VerificationStatus.UNSAFE: "UNSAFE",
            VerificationStatus.UNKNOWN: "UNKNOWN"
        }[self.status]

        lines = [
            f"Verification Result: {status_str}",
            "=" * 40,
            f"Minimum F̃_T:      {self.min_objective:.6f}",
            f"Error bound ε:    {self.error_bound:.6f}",
            f"Safety margin:    {self.safety_margin:.6f}",
        ]

        if self.counterexample is not None:
            lines.append(f"Counterexample x0: {self.counterexample}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"VerificationResult(status={self.status.name}, "
            f"min={self.min_objective:.4f}, margin={self.safety_margin:.4f})"
        )


@dataclass
class SafetyVerifier:
    """Safety verifier using spline approximation of objective function.

    The verification pipeline:
    1. Sample initial conditions from the initial set
    2. For each sample, simulate trajectory and compute distance to unsafe set
    3. Fit spline approximation to sampled objective function
    4. Minimize the spline approximation
    5. Compare minimum against error bound to determine safety
    """
    # Sampling parameters
    n_samples: int = 500
    sampling_strategy: SamplingStrategy = SamplingStrategy.LATIN_HYPERCUBE
    seed: int | None = None

    # Spline parameters
    spline_method: str = 'rbf'  # 'rbf', 'bspline', or 'auto'
    spline_kernel: str = 'thin_plate_spline'
    spline_smoothing: float = 0.0
    # B-spline specific parameters
    n_grid_points: int = 50
    bspline_degree: int = 3
    grid_method: str = 'linear'  # 'linear', 'nearest', or 'rbf'

    # Optimization parameters
    optimization_method: str = 'multistart'
    n_optimization_starts: int = 20

    # Error bound parameters
    lipschitz_estimate: float | None = None  # If None, estimate from data

    def verify(
        self,
        dynamics: DynamicsModel,
        initial_set: Set,
        unsafe_set: Set,
        time_horizon: float,
        **kwargs
    ) -> VerificationResult:
        """Verify safety of a dynamical system.

        Args:
            dynamics: The dynamics model to verify.
            initial_set: Set of initial conditions I.
            unsafe_set: Unsafe set U to avoid.
            time_horizon: Time horizon T.
            **kwargs: Override default parameters.

        Returns:
            VerificationResult indicating safety status.
        """
        # Override defaults with kwargs
        n_samples = kwargs.get('n_samples', self.n_samples)
        strategy = kwargs.get('sampling_strategy', self.sampling_strategy)
        seed = kwargs.get('seed', self.seed)

        # Step 1: Sample objective function
        sampler = ObjectiveSampler(
            dynamics=dynamics,
            initial_set=initial_set,
            unsafe_set=unsafe_set,
            time_horizon=time_horizon
        )

        points, values = sampler.sample(n_samples, strategy, seed)
        sampled = SampledObjective(points, values)

        # Early exit: if any sample has zero distance, system is UNSAFE
        if sampled.min_value <= 0:
            return VerificationResult(
                status=VerificationStatus.UNSAFE,
                min_objective=sampled.min_value,
                minimizer=sampled.argmin,
                error_bound=0.0,
                safety_margin=sampled.min_value,
                counterexample=sampled.argmin,
                details={
                    'early_exit': True,
                    'reason': 'sample_hit_unsafe',
                    'n_samples': n_samples,
                    'spline': None,  # No spline fitted in early exit
                    'sample_points': points,
                    'sample_values': values,
                }
            )

        # Step 2: Fit spline approximation
        spline_method = kwargs.get('spline_method', self.spline_method)
        n_dims = points.shape[1]

        if spline_method == 'rbf' or (spline_method in ('bspline', 'auto') and n_dims > 2):
            # Use RBF for high dimensions or when explicitly requested
            if spline_method == 'bspline' and n_dims > 2:
                import warnings
                warnings.warn(
                    f"B-spline only supports 1D/2D (got {n_dims}D). "
                    "Falling back to RBF interpolation."
                )
            spline = ScatteredDataSpline(
                kernel=kwargs.get('spline_kernel', self.spline_kernel),
                smoothing=kwargs.get('spline_smoothing', self.spline_smoothing)
            )
        else:
            # Use B-spline for 1D/2D
            spline = GriddedBSpline(
                n_grid_points=kwargs.get('n_grid_points', self.n_grid_points),
                degree=kwargs.get('bspline_degree', self.bspline_degree),
                smoothing=kwargs.get('spline_smoothing', self.spline_smoothing),
                grid_method=kwargs.get('grid_method', self.grid_method)
            )

        spline.fit(points, values)

        # Step 3: Minimize spline
        if isinstance(initial_set, HyperRectangle):
            bounds = (initial_set.lower, initial_set.upper)
        elif hasattr(initial_set, 'bounds') and initial_set.bounds is not None:
            bounds = (initial_set.bounds.lower, initial_set.bounds.upper)
        else:
            # Estimate bounds from samples
            bounds = (points.min(axis=0), points.max(axis=0))

        min_result = minimize_spline(
            spline,
            bounds,
            method=kwargs.get('optimization_method', self.optimization_method),
            n_starts=kwargs.get('n_optimization_starts', self.n_optimization_starts),
            seed=seed
        )

        # Step 4: Compute error bounds
        error_budget = ErrorBudget()

        # Approximation error from residuals
        approx_values = np.array([spline.evaluate(p) for p in points])
        error_budget.estimate_from_samples(points, values, approx_values, bounds)

        # Minimization error
        error_budget.set_minimization_error(
            tolerance=1e-6,
            n_starts=kwargs.get('n_optimization_starts', self.n_optimization_starts)
        )

        # Step 5: Determine safety status
        epsilon = error_budget.total
        safety_margin = min_result.minimum - epsilon

        if min_result.minimum <= 0:
            # Spline minimum is non-positive: likely unsafe
            status = VerificationStatus.UNSAFE
            counterexample = min_result.minimizer
        elif safety_margin > 0:
            # min F̃_T > epsilon: proven safe
            status = VerificationStatus.SAFE
            counterexample = None
        else:
            # 0 < min F̃_T <= epsilon: cannot certify
            status = VerificationStatus.UNKNOWN
            counterexample = None

        return VerificationResult(
            status=status,
            min_objective=min_result.minimum,
            minimizer=min_result.minimizer,
            error_bound=epsilon,
            safety_margin=safety_margin,
            counterexample=counterexample,
            details={
                'n_samples': n_samples,
                'spline_method': 'bspline' if isinstance(spline, GriddedBSpline) else 'rbf',
                'sampled_min': sampled.min_value,
                'sampled_max': sampled.max_value,
                'optimization_evals': min_result.n_evaluations,
                'optimization_success': min_result.success,
                'error_budget': {
                    'integration': error_budget.integration_error,
                    'sampling': error_budget.sampling_error,
                    'approximation': error_budget.approximation_error,
                    'minimization': error_budget.minimization_error,
                },
                # Spline and sample data for visualization
                'spline': spline,
                'sample_points': points,
                'sample_values': values,
            }
        )

    def verify_with_refinement(
        self,
        dynamics: DynamicsModel,
        initial_set: Set,
        unsafe_set: Set,
        time_horizon: float,
        max_iterations: int = 5,
        target_margin: float | None = None,
        **kwargs
    ) -> VerificationResult:
        """Verify with adaptive refinement until conclusive result.

        If initial verification returns UNKNOWN, add more samples near
        the minimum and re-verify.

        Args:
            dynamics: The dynamics model.
            initial_set: Initial set.
            unsafe_set: Unsafe set.
            time_horizon: Time horizon.
            max_iterations: Maximum refinement iterations.
            target_margin: Stop when safety margin exceeds this.
            **kwargs: Additional parameters.

        Returns:
            Final VerificationResult.
        """
        n_samples = kwargs.pop('n_samples', self.n_samples)

        for iteration in range(max_iterations):
            result = self.verify(
                dynamics, initial_set, unsafe_set, time_horizon,
                n_samples=n_samples, **kwargs
            )

            # Stop if conclusive
            if result.status != VerificationStatus.UNKNOWN:
                result.details['iterations'] = iteration + 1
                return result

            # Stop if target margin achieved
            if target_margin is not None and result.safety_margin >= target_margin:
                result.details['iterations'] = iteration + 1
                return result

            # Increase samples for next iteration
            n_samples = int(n_samples * 1.5)

        result.details['iterations'] = max_iterations
        result.details['max_iterations_reached'] = True
        return result
