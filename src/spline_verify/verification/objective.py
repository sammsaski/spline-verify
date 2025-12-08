"""Objective function F_T computation for safety verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable, runtime_checkable

import numpy as np

from ..dynamics.base import DynamicsModel, TrajectoryBundle
from ..dynamics.trajectory import Trajectory
from ..geometry.sets import Set
from ..geometry.sampling import sample_set, SamplingStrategy


@runtime_checkable
class ObjectiveFunction(Protocol):
    """Protocol for objective function approximations.

    The objective function F_T(x0) represents the minimum distance from
    the trajectory starting at x0 to the unsafe set over time [0, T].
    """

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the objective function at point x.

        Args:
            x: Initial condition, shape (n_dims,).

        Returns:
            Value of F_T(x).
        """
        ...

    def minimum(self) -> tuple[float, np.ndarray]:
        """Find the minimum of the objective function.

        Returns:
            Tuple of (minimum_value, minimizing_point).
        """
        ...


def compute_objective(
    trajectory: Trajectory,
    unsafe_set: Set
) -> float:
    """Compute the objective function value for a single trajectory.

    F_T(x0) = min_{t in [0,T]} dist(trajectory(t), unsafe_set)

    Args:
        trajectory: Trajectory from initial condition x0.
        unsafe_set: The unsafe set U.

    Returns:
        Minimum distance from trajectory to unsafe set.
    """
    distance_func = unsafe_set.distance_function()
    min_dist, _, _ = trajectory.min_distance_to_set(distance_func)
    return min_dist


def compute_objective_bundle(
    bundle: TrajectoryBundle,
    unsafe_set: Set
) -> float:
    """Compute objective for a trajectory bundle (Filippov case).

    For safety verification with non-unique solutions, we need the
    worst-case (minimum) distance across all possible trajectories.

    Args:
        bundle: Bundle of trajectories from same initial condition.
        unsafe_set: The unsafe set U.

    Returns:
        Minimum distance across all trajectories in bundle.
    """
    distance_func = unsafe_set.distance_function()
    return bundle.min_distance_to_set(distance_func)


@dataclass
class ObjectiveSampler:
    """Samples the objective function F_T over an initial set.

    This class handles the simulation and distance computation for
    multiple initial conditions.
    """
    dynamics: DynamicsModel
    initial_set: Set
    unsafe_set: Set
    time_horizon: float

    def sample(
        self,
        n_samples: int,
        strategy: SamplingStrategy = SamplingStrategy.LATIN_HYPERCUBE,
        seed: int | None = None,
        parallel: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the objective function at multiple initial conditions.

        Args:
            n_samples: Number of initial conditions to sample.
            strategy: Sampling strategy for initial set.
            seed: Random seed for reproducibility.
            parallel: Whether to parallelize simulations (not implemented).

        Returns:
            Tuple of (initial_points, objective_values).
            - initial_points: shape (n_samples, n_dims)
            - objective_values: shape (n_samples,)
        """
        # Sample initial conditions
        initial_points = sample_set(
            self.initial_set, n_samples, strategy, seed
        )

        # Compute objective for each initial condition
        objective_values = np.zeros(n_samples)
        t_span = (0.0, self.time_horizon)

        for i, x0 in enumerate(initial_points):
            bundle = self.dynamics.simulate(x0, t_span)
            objective_values[i] = compute_objective_bundle(bundle, self.unsafe_set)

        return initial_points, objective_values

    def evaluate(self, x0: np.ndarray) -> float:
        """Evaluate objective at a single initial condition.

        Args:
            x0: Initial condition.

        Returns:
            F_T(x0).
        """
        t_span = (0.0, self.time_horizon)
        bundle = self.dynamics.simulate(x0, t_span)
        return compute_objective_bundle(bundle, self.unsafe_set)


@dataclass
class SampledObjective:
    """Stores sampled objective function values.

    This is the intermediate representation between sampling and
    spline approximation.
    """
    points: np.ndarray  # Shape (n_samples, n_dims)
    values: np.ndarray  # Shape (n_samples,)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points)
        self.values = np.asarray(self.values)

        if len(self.points) != len(self.values):
            raise ValueError(
                f"Length mismatch: {len(self.points)} points, "
                f"{len(self.values)} values"
            )

    @property
    def n_samples(self) -> int:
        return len(self.points)

    @property
    def n_dims(self) -> int:
        return self.points.shape[1]

    @property
    def min_value(self) -> float:
        """Minimum sampled objective value."""
        return float(np.min(self.values))

    @property
    def max_value(self) -> float:
        """Maximum sampled objective value."""
        return float(np.max(self.values))

    @property
    def argmin(self) -> np.ndarray:
        """Point with minimum objective value."""
        return self.points[np.argmin(self.values)].copy()

    def unsafe_samples(self, threshold: float = 0.0) -> np.ndarray:
        """Return points where objective is below threshold.

        Args:
            threshold: Distance threshold (points with F_T < threshold
                      are potentially unsafe).

        Returns:
            Array of points with F_T < threshold.
        """
        mask = self.values < threshold
        return self.points[mask]

    def __repr__(self) -> str:
        return (
            f"SampledObjective(n_samples={self.n_samples}, "
            f"min={self.min_value:.4f}, max={self.max_value:.4f})"
        )
