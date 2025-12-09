"""Switching system verification for hybrid dynamical systems.

This module provides verification tools for systems with discontinuous dynamics,
including event-triggered switching (e.g., bouncing ball) and state-dependent
switching (e.g., thermostat, relay feedback).

Key components:
- SwitchingVerifier: Full pipeline for switching system safety verification
- SwitchingRegionClassifier: SVM-based classification of switching regions
- extract_crossing_labels: Automatic extraction of crossing behavior from trajectories
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

from .verifier import SafetyVerifier, VerificationResult, VerificationStatus
from .objective import ObjectiveSampler, SampledObjective, compute_objective_bundle
from .error_bounds import ErrorBudget
from ..geometry.sets import Set, HyperRectangle
from ..geometry.sampling import sample_set, SamplingStrategy
from ..splines.piecewise import PiecewiseSplineApproximation
from ..splines.optimization import minimize_spline, MinimizationResult

if TYPE_CHECKING:
    from ..dynamics.switching import SwitchingDynamics, SwitchingSurface, FilippovSolver
    from ..dynamics.base import TrajectoryBundle


def extract_crossing_labels(
    initial_points: np.ndarray,
    bundles: list["TrajectoryBundle"],
    surfaces: list["SwitchingSurface"],
    threshold: float = 1e-6,
) -> np.ndarray:
    """Extract crossing labels from simulated trajectories.

    Analyzes each trajectory to determine which switching surfaces it crosses,
    enabling automatic region classification without manual labeling.

    Args:
        initial_points: Initial conditions, shape (n_samples, n_dims).
        bundles: Trajectory bundles from simulation, one per initial point.
        surfaces: List of switching surfaces to check for crossings.
        threshold: Distance threshold for considering a surface crossed.

    Returns:
        Integer labels of shape (n_samples,):
        - 0: trajectory never crosses any surface
        - i: trajectory crosses surface i-1 (1-indexed for compatibility)
        - For multiple crossings, returns the first surface crossed

    Example:
        >>> points = sample_set(initial_set, 100)
        >>> bundles = [dynamics.simulate(p, (0, T)) for p in points]
        >>> labels = extract_crossing_labels(points, bundles, dynamics.surfaces)
        >>> classifier.fit(points, labels)
    """
    n_samples = len(initial_points)
    n_surfaces = len(surfaces)
    labels = np.zeros(n_samples, dtype=int)

    for i, bundle in enumerate(bundles):
        # Check all trajectories in the bundle (for non-unique solutions)
        for traj in bundle.trajectories:
            states = traj.states

            # Check each surface for crossings
            for surf_idx, surface in enumerate(surfaces):
                # Evaluate surface function along trajectory
                surface_values = np.array([
                    surface.evaluate(state) for state in states
                ])

                # Detect sign changes (zero crossings)
                signs = np.sign(surface_values)
                sign_changes = np.where(np.diff(signs) != 0)[0]

                if len(sign_changes) > 0:
                    # Trajectory crosses this surface
                    # Use 1-indexed labels (0 reserved for "no crossing")
                    labels[i] = surf_idx + 1
                    break  # Use first surface crossed

                # Also check for sliding (values near zero)
                if np.any(np.abs(surface_values) < threshold):
                    labels[i] = surf_idx + 1
                    break

    return labels


def extract_crossing_sequence(
    bundle: "TrajectoryBundle",
    surfaces: list["SwitchingSurface"],
    threshold: float = 1e-6,
) -> list[tuple[int, float, np.ndarray]]:
    """Extract the sequence of surface crossings from a trajectory.

    Args:
        bundle: Trajectory bundle from simulation.
        surfaces: List of switching surfaces.
        threshold: Distance threshold for crossings.

    Returns:
        List of (surface_index, time, state) tuples for each crossing,
        in chronological order.
    """
    crossings = []

    for traj in bundle.trajectories:
        states = traj.states
        times = traj.times

        for surf_idx, surface in enumerate(surfaces):
            surface_values = np.array([
                surface.evaluate(state) for state in states
            ])

            signs = np.sign(surface_values)
            sign_change_indices = np.where(np.diff(signs) != 0)[0]

            for idx in sign_change_indices:
                # Approximate crossing time and state
                t_cross = (times[idx] + times[idx + 1]) / 2
                x_cross = (states[idx] + states[idx + 1]) / 2
                crossings.append((surf_idx, t_cross, x_cross))

    # Sort by time
    crossings.sort(key=lambda x: x[1])
    return crossings


class SwitchingRegionClassifier:
    """Classifies initial conditions by their switching behavior.

    Uses SVM (Support Vector Machine) to learn decision boundaries between
    different switching regions in the initial set. This enables piecewise
    spline fitting for the objective function.

    Regions are labeled as:
    - 0: trajectories that never cross any switching surface (I_0)
    - i: trajectories that first cross switching surface i-1 (S_i)

    Attributes:
        kernel: SVM kernel type ('rbf', 'linear', 'poly').
        C: SVM regularization parameter.
        gamma: RBF kernel coefficient ('scale', 'auto', or float).

    Example:
        >>> classifier = SwitchingRegionClassifier(kernel='rbf')
        >>> classifier.fit(initial_points, crossing_labels)
        >>> region = classifier.predict(new_point)
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str | float = 'scale',
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self._classifier = None
        self._fitted = False
        self._unique_labels: np.ndarray | None = None
        self._n_regions: int = 0

    @property
    def is_fitted(self) -> bool:
        """Whether the classifier has been fitted."""
        return self._fitted

    @property
    def n_regions(self) -> int:
        """Number of distinct regions."""
        return self._n_regions

    @property
    def region_labels(self) -> np.ndarray | None:
        """Unique region labels from training data."""
        return self._unique_labels

    def fit(
        self,
        initial_points: np.ndarray,
        crossing_labels: np.ndarray,
    ) -> "SwitchingRegionClassifier":
        """Fit classifier to labeled data.

        Args:
            initial_points: Initial conditions, shape (n, d).
            crossing_labels: Integer labels indicating which surface(s)
                           the trajectory crosses (0 = none).

        Returns:
            self for method chaining.
        """
        from sklearn.svm import SVC

        initial_points = np.asarray(initial_points)
        crossing_labels = np.asarray(crossing_labels)

        self._unique_labels = np.unique(crossing_labels)
        self._n_regions = len(self._unique_labels)

        # If only one region, no need for classifier
        if self._n_regions == 1:
            self._fitted = True
            self._classifier = None
            return self

        self._classifier = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,  # Enable probability estimates
        )
        self._classifier.fit(initial_points, crossing_labels)
        self._fitted = True

        return self

    def predict(self, x: np.ndarray) -> int:
        """Predict which region a point belongs to.

        Args:
            x: Point, shape (d,).

        Returns:
            Region label (0 for I_0, i for S_i).
        """
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        # Single region case
        if self._classifier is None:
            return int(self._unique_labels[0])

        x = np.asarray(x).reshape(1, -1)
        return int(self._classifier.predict(x)[0])

    def predict_batch(self, points: np.ndarray) -> np.ndarray:
        """Predict regions for multiple points.

        Args:
            points: Points, shape (n, d).

        Returns:
            Region labels, shape (n,).
        """
        if not self._fitted:
            raise RuntimeError("Not fitted.")

        if self._classifier is None:
            return np.full(len(points), self._unique_labels[0], dtype=int)

        return self._classifier.predict(points)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get probability estimates for each region.

        Args:
            x: Point, shape (d,).

        Returns:
            Probabilities for each class, shape (n_classes,).
        """
        if not self._fitted:
            raise RuntimeError("Not fitted.")

        if self._classifier is None:
            return np.array([1.0])

        x = np.asarray(x).reshape(1, -1)
        return self._classifier.predict_proba(x)[0]

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Compute distance to decision boundaries.

        Useful for understanding classifier confidence.

        Args:
            x: Point, shape (d,).

        Returns:
            Distance to each hyperplane.
        """
        if not self._fitted or self._classifier is None:
            raise RuntimeError("Not fitted or single-region case.")

        x = np.asarray(x).reshape(1, -1)
        return self._classifier.decision_function(x)[0]


@dataclass
class SwitchingVerifier(SafetyVerifier):
    """Safety verifier for switching/hybrid systems.

    This extends the base SafetyVerifier to handle:
    - State-based switching with Filippov solutions
    - Non-unique trajectories from a single initial condition
    - Piecewise continuous objective function F_T

    The verification pipeline:
    1. Sample initial conditions from the initial set
    2. Simulate using FilippovSolver (handles sliding modes)
    3. Extract crossing labels automatically from trajectories
    4. Train SVM classifier on switching regions
    5. Fit piecewise spline on each continuity region
    6. Minimize each piece independently
    7. Return global minimum across all regions

    Attributes:
        classifier_kernel: SVM kernel for region classification.
        min_samples_per_region: Minimum samples to fit a regional spline.
        use_piecewise: Whether to use piecewise splines (True) or single spline.

    Example:
        >>> dynamics = SwitchingDynamics.bouncing_ball(gravity=9.81)
        >>> verifier = SwitchingVerifier(n_samples=200)
        >>> result = verifier.verify(dynamics, initial_set, unsafe_set, T=3.0)
    """
    # Classification parameters
    classifier_kernel: str = 'rbf'
    classifier_C: float = 1.0

    # Piecewise fitting parameters
    min_samples_per_region: int = 10
    use_piecewise: bool = True

    def verify(
        self,
        dynamics: "SwitchingDynamics",
        initial_set: Set,
        unsafe_set: Set,
        time_horizon: float,
        **kwargs
    ) -> VerificationResult:
        """Verify safety of a switching system.

        Args:
            dynamics: SwitchingDynamics with defined surfaces.
            initial_set: Set of initial conditions I.
            unsafe_set: Unsafe set U to avoid.
            time_horizon: Time horizon T.
            **kwargs: Override default parameters.

        Returns:
            VerificationResult with switching-specific details.
        """
        from ..dynamics.switching import SwitchingDynamics, FilippovSolver

        # Override defaults with kwargs
        n_samples = kwargs.get('n_samples', self.n_samples)
        strategy = kwargs.get('sampling_strategy', self.sampling_strategy)
        seed = kwargs.get('seed', self.seed)
        use_piecewise = kwargs.get('use_piecewise', self.use_piecewise)

        # If no surfaces defined, fall back to base verifier
        if not hasattr(dynamics, 'surfaces') or len(dynamics.surfaces) == 0:
            return super().verify(
                dynamics, initial_set, unsafe_set, time_horizon, **kwargs
            )

        # Step 1: Sample initial conditions
        points = sample_set(initial_set, n_samples, strategy, seed)

        # Step 2: Simulate using FilippovSolver
        solver = FilippovSolver(dynamics)
        bundles = []
        values = []

        for x0 in points:
            bundle = solver.solve(x0, (0, time_horizon))
            bundles.append(bundle)

            # Compute worst-case objective (min distance across all trajectories)
            min_dist = bundle.min_distance_to_set(unsafe_set.distance)
            values.append(min_dist)

        values = np.array(values)

        # Early exit: if any sample reaches unsafe set
        if np.min(values) <= 0:
            min_idx = np.argmin(values)
            return VerificationResult(
                status=VerificationStatus.UNSAFE,
                min_objective=float(values[min_idx]),
                minimizer=points[min_idx],
                error_bound=0.0,
                safety_margin=float(values[min_idx]),
                counterexample=points[min_idx],
                details={
                    'early_exit': True,
                    'reason': 'sample_hit_unsafe',
                    'n_samples': n_samples,
                }
            )

        # Step 3: Extract crossing labels
        labels = extract_crossing_labels(points, bundles, dynamics.surfaces)
        unique_labels = np.unique(labels)
        n_regions = len(unique_labels)

        # Step 4: Train classifier (if multiple regions)
        classifier = SwitchingRegionClassifier(
            kernel=self.classifier_kernel,
            C=self.classifier_C,
        )
        classifier.fit(points, labels)

        # Step 5: Fit spline(s)
        if use_piecewise and n_regions > 1:
            # Piecewise spline fitting
            spline = PiecewiseSplineApproximation()
            spline.fit(points, values, labels)
            spline.set_classifier(classifier.predict)
        else:
            # Single spline (regions merged)
            from ..splines.multivariate import ScatteredDataSpline
            spline = ScatteredDataSpline(
                kernel=kwargs.get('spline_kernel', self.spline_kernel),
                smoothing=kwargs.get('spline_smoothing', self.spline_smoothing)
            )
            spline.fit(points, values)

        # Step 6: Minimize spline(s)
        if isinstance(initial_set, HyperRectangle):
            bounds = (initial_set.lower, initial_set.upper)
        elif hasattr(initial_set, 'bounds') and initial_set.bounds is not None:
            bounds = (initial_set.bounds.lower, initial_set.bounds.upper)
        else:
            bounds = (points.min(axis=0), points.max(axis=0))

        # For piecewise splines, minimize each region
        if use_piecewise and n_regions > 1 and isinstance(spline, PiecewiseSplineApproximation):
            region_minima = []
            for region in unique_labels:
                region_result = spline.region_minimum(int(region))
                if region_result is not None:
                    region_minima.append(region_result)

            # Also do global optimization
            global_result = minimize_spline(
                spline,
                bounds,
                method=kwargs.get('optimization_method', self.optimization_method),
                n_starts=kwargs.get('n_optimization_starts', self.n_optimization_starts),
                seed=seed
            )

            # Take overall minimum
            if region_minima:
                best_region_min = min(region_minima, key=lambda x: x[0])
                if best_region_min[0] < global_result.minimum:
                    min_result = MinimizationResult(
                        minimum=best_region_min[0],
                        minimizer=best_region_min[1],
                        success=True,
                        n_evaluations=global_result.n_evaluations,
                    )
                else:
                    min_result = global_result
            else:
                min_result = global_result
        else:
            min_result = minimize_spline(
                spline,
                bounds,
                method=kwargs.get('optimization_method', self.optimization_method),
                n_starts=kwargs.get('n_optimization_starts', self.n_optimization_starts),
                seed=seed
            )

        # Step 7: Compute error bounds
        error_budget = ErrorBudget()

        # Approximation error from residuals
        approx_values = np.array([spline.evaluate(p) for p in points])
        error_budget.estimate_from_samples(points, values, approx_values, bounds)

        # Add switching-related error (classification uncertainty)
        # Conservative estimate: add RMS of residuals in boundary regions
        if n_regions > 1:
            boundary_error = np.std(values - approx_values) * 0.5
            error_budget.approximation_error += boundary_error

        error_budget.set_minimization_error(
            tolerance=1e-6,
            n_starts=kwargs.get('n_optimization_starts', self.n_optimization_starts)
        )

        # Step 8: Determine safety status
        epsilon = error_budget.total
        safety_margin = min_result.minimum - epsilon

        if min_result.minimum <= 0:
            status = VerificationStatus.UNSAFE
            counterexample = min_result.minimizer
        elif safety_margin > 0:
            status = VerificationStatus.SAFE
            counterexample = None
        else:
            status = VerificationStatus.UNKNOWN
            counterexample = None

        # Compute region statistics
        region_counts = {int(label): int(np.sum(labels == label)) for label in unique_labels}
        sampled_min = float(np.min(values))
        sampled_argmin = points[np.argmin(values)]

        return VerificationResult(
            status=status,
            min_objective=min_result.minimum,
            minimizer=min_result.minimizer,
            error_bound=epsilon,
            safety_margin=safety_margin,
            counterexample=counterexample,
            details={
                'n_samples': n_samples,
                'sampled_min': sampled_min,
                'sampled_max': float(np.max(values)),
                'sampled_argmin': sampled_argmin,
                'optimization_evals': min_result.n_evaluations,
                'optimization_success': min_result.success,
                'error_budget': {
                    'integration': error_budget.integration_error,
                    'sampling': error_budget.sampling_error,
                    'approximation': error_budget.approximation_error,
                    'minimization': error_budget.minimization_error,
                },
                # Switching-specific info
                'switching_info': {
                    'n_regions': n_regions,
                    'region_counts': region_counts,
                    'classifier_kernel': self.classifier_kernel,
                    'use_piecewise': use_piecewise,
                }
            }
        )
