"""ODE dynamics model implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .base import DynamicsModel, DynamicsFunction, TrajectoryBundle
from .trajectory import Trajectory
from .integrators import integrate, IntegrationMethod


@dataclass
class ODEDynamics:
    """ODE dynamics model: dx/dt = f(t, x).

    Implements the DynamicsModel protocol for standard ODE systems.

    Attributes:
        f: The dynamics function f(t, x) -> dx/dt.
        n_dims: Dimension of the state space.
        method: Integration method to use.
        step_size: Step size for integration.
        rtol: Relative tolerance for adaptive methods.
        atol: Absolute tolerance for adaptive methods.
    """
    f: DynamicsFunction
    _n_dims: int
    method: IntegrationMethod = IntegrationMethod.RK45
    step_size: float | None = None
    rtol: float = 1e-6
    atol: float = 1e-9

    @property
    def n_dims(self) -> int:
        """Dimension of the state space."""
        return self._n_dims

    def simulate(
        self,
        x0: np.ndarray,
        t_span: tuple[float, float],
        **kwargs
    ) -> TrajectoryBundle:
        """Simulate the ODE from initial condition x0.

        Args:
            x0: Initial state, shape (n_dims,).
            t_span: Time interval (t_start, t_end).
            **kwargs: Override default integration options.

        Returns:
            TrajectoryBundle containing a single trajectory.
        """
        x0 = np.asarray(x0, dtype=float)

        if x0.shape != (self.n_dims,):
            raise ValueError(
                f"Initial state has wrong shape: expected ({self.n_dims},), "
                f"got {x0.shape}"
            )

        trajectory = integrate(
            f=self.f,
            x0=x0,
            t_span=t_span,
            method=kwargs.get('method', self.method),
            step_size=kwargs.get('step_size', self.step_size),
            rtol=kwargs.get('rtol', self.rtol),
            atol=kwargs.get('atol', self.atol),
        )

        return TrajectoryBundle.from_single(trajectory)

    @classmethod
    def from_matrix(cls, A: np.ndarray, **kwargs) -> ODEDynamics:
        """Create linear ODE dynamics dx/dt = Ax.

        Args:
            A: System matrix, shape (n, n).
            **kwargs: Additional options passed to ODEDynamics.

        Returns:
            ODEDynamics for the linear system.
        """
        A = np.asarray(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be a square matrix, got shape {A.shape}")

        n_dims = A.shape[0]

        def f(t: float, x: np.ndarray) -> np.ndarray:
            return A @ x

        return cls(f=f, _n_dims=n_dims, **kwargs)

    @classmethod
    def harmonic_oscillator(cls, omega: float = 1.0, **kwargs) -> ODEDynamics:
        """Create harmonic oscillator: dx/dt = y, dy/dt = -omega^2 * x.

        Args:
            omega: Angular frequency (default 1.0 for unit frequency).
            **kwargs: Additional options passed to ODEDynamics.

        Returns:
            ODEDynamics for the harmonic oscillator.
        """
        A = np.array([
            [0, 1],
            [-omega**2, 0]
        ])
        return cls.from_matrix(A, **kwargs)

    def __repr__(self) -> str:
        return f"ODEDynamics(n_dims={self.n_dims}, method={self.method.name})"
