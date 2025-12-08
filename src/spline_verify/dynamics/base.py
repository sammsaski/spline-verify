"""Base protocols and types for dynamics models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Callable, runtime_checkable

import numpy as np

from .trajectory import Trajectory


# Type alias for dynamics function: f(t, x) -> dx/dt
DynamicsFunction = Callable[[float, np.ndarray], np.ndarray]


@runtime_checkable
class DynamicsModel(Protocol):
    """Protocol for dynamics models (ODE or switching systems).

    This protocol defines the interface that both ODE and switching systems
    must implement, enabling the verifier to work with either type.
    """

    @property
    def n_dims(self) -> int:
        """Dimension of the state space."""
        ...

    def simulate(
        self,
        x0: np.ndarray,
        t_span: tuple[float, float],
        **kwargs
    ) -> TrajectoryBundle:
        """Simulate the system from initial condition x0.

        Args:
            x0: Initial state, shape (n_dims,).
            t_span: Time interval (t_start, t_end).
            **kwargs: Additional integration options.

        Returns:
            TrajectoryBundle containing the simulation result(s).
        """
        ...


@dataclass
class TrajectoryBundle:
    """A collection of trajectories from a single initial condition.

    For standard ODEs, this contains exactly one trajectory.
    For Filippov systems with non-unique solutions, this may contain
    multiple trajectories representing different possible evolutions.

    Attributes:
        trajectories: List of trajectories from the same initial condition.
        initial_state: The common initial state.
    """
    trajectories: list[Trajectory]
    initial_state: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        if len(self.trajectories) == 0:
            raise ValueError("TrajectoryBundle must contain at least one trajectory")

        if self.initial_state.size == 0:
            self.initial_state = self.trajectories[0].initial_state

    @property
    def is_unique(self) -> bool:
        """Whether the solution is unique (single trajectory)."""
        return len(self.trajectories) == 1

    @property
    def primary(self) -> Trajectory:
        """The primary (or only) trajectory in the bundle."""
        return self.trajectories[0]

    @classmethod
    def from_single(cls, trajectory: Trajectory) -> TrajectoryBundle:
        """Create a bundle from a single trajectory."""
        return cls(
            trajectories=[trajectory],
            initial_state=trajectory.initial_state
        )

    def min_distance_to_set(
        self,
        distance_func: Callable[[np.ndarray], float]
    ) -> float:
        """Compute minimum distance from any trajectory in bundle to a set.

        For safety verification, we need the worst-case (minimum) distance
        across all possible trajectories.

        Args:
            distance_func: Function computing distance from a point to the set.

        Returns:
            Minimum distance across all trajectories.
        """
        min_dist = float('inf')
        for traj in self.trajectories:
            dist, _, _ = traj.min_distance_to_set(distance_func)
            min_dist = min(min_dist, dist)
        return min_dist

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self):
        return iter(self.trajectories)

    def __repr__(self) -> str:
        return f"TrajectoryBundle(n_trajectories={len(self.trajectories)})"
