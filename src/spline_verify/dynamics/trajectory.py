"""Trajectory data structure for storing and manipulating ODE solutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class Trajectory:
    """A trajectory storing time points and state values.

    Attributes:
        times: 1D array of time points, shape (n_points,)
        states: 2D array of states, shape (n_points, n_dims)
    """
    times: np.ndarray
    states: np.ndarray

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times)
        self.states = np.asarray(self.states)

        if self.states.ndim == 1:
            self.states = self.states.reshape(-1, 1)

        if len(self.times) != len(self.states):
            raise ValueError(
                f"Length mismatch: times has {len(self.times)} points, "
                f"states has {len(self.states)} points"
            )

    @property
    def n_points(self) -> int:
        """Number of time points in the trajectory."""
        return len(self.times)

    @property
    def n_dims(self) -> int:
        """Dimension of the state space."""
        return self.states.shape[1]

    @property
    def t_start(self) -> float:
        """Start time of the trajectory."""
        return float(self.times[0])

    @property
    def t_end(self) -> float:
        """End time of the trajectory."""
        return float(self.times[-1])

    @property
    def duration(self) -> float:
        """Duration of the trajectory."""
        return self.t_end - self.t_start

    @property
    def initial_state(self) -> np.ndarray:
        """Initial state of the trajectory."""
        return self.states[0].copy()

    @property
    def final_state(self) -> np.ndarray:
        """Final state of the trajectory."""
        return self.states[-1].copy()

    def interpolate(self, t: float | np.ndarray) -> np.ndarray:
        """Interpolate the trajectory at given time(s).

        Args:
            t: Time point(s) at which to interpolate.

        Returns:
            State(s) at the given time(s). Shape (n_dims,) for scalar t,
            or (n_times, n_dims) for array t.
        """
        t = np.asarray(t)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)

        # Create interpolation function for each dimension
        interpolator = interp1d(
            self.times, self.states, axis=0,
            kind='linear', bounds_error=True
        )

        result = interpolator(t)

        if scalar_input:
            return result[0]
        return result

    def slice_time(self, t_start: float, t_end: float) -> Trajectory:
        """Extract a sub-trajectory between two time points.

        Args:
            t_start: Start time of the slice.
            t_end: End time of the slice.

        Returns:
            New Trajectory containing only points in [t_start, t_end].
        """
        mask = (self.times >= t_start) & (self.times <= t_end)
        return Trajectory(
            times=self.times[mask].copy(),
            states=self.states[mask].copy()
        )

    def min_distance_to_point(self, point: np.ndarray) -> float:
        """Compute minimum distance from any trajectory point to a given point.

        Args:
            point: Target point, shape (n_dims,).

        Returns:
            Minimum Euclidean distance from trajectory to point.
        """
        point = np.asarray(point)
        distances = np.linalg.norm(self.states - point, axis=1)
        return float(np.min(distances))

    def min_distance_to_set(
        self,
        distance_func: Callable[[np.ndarray], float]
    ) -> tuple[float, float, np.ndarray]:
        """Compute minimum distance from trajectory to a set.

        Args:
            distance_func: Function that computes distance from a point to the set.
                          Should accept array of shape (n_dims,) and return float.

        Returns:
            Tuple of (min_distance, time_of_min, state_at_min).
        """
        distances = np.array([distance_func(state) for state in self.states])
        min_idx = np.argmin(distances)
        return (
            float(distances[min_idx]),
            float(self.times[min_idx]),
            self.states[min_idx].copy()
        )

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        return (
            f"Trajectory(n_points={self.n_points}, n_dims={self.n_dims}, "
            f"t=[{self.t_start:.4f}, {self.t_end:.4f}])"
        )
