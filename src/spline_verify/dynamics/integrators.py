"""Numerical integrators for ODE systems."""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .trajectory import Trajectory
from .base import DynamicsFunction


class IntegrationMethod(Enum):
    """Available numerical integration methods."""
    EULER = auto()       # Forward Euler (first order, for testing)
    RK4 = auto()         # Classic Runge-Kutta 4th order
    RK45 = auto()        # Adaptive Runge-Kutta 4(5), scipy default
    ADAMS = auto()       # Adams-Bashforth multi-step method (via scipy LSODA)


def integrate(
    f: DynamicsFunction,
    x0: np.ndarray,
    t_span: tuple[float, float],
    method: IntegrationMethod = IntegrationMethod.RK45,
    step_size: float | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float | None = None,
) -> Trajectory:
    """Integrate an ODE system from initial condition.

    Args:
        f: Dynamics function f(t, x) -> dx/dt.
        x0: Initial state, shape (n_dims,).
        t_span: Time interval (t_start, t_end).
        method: Integration method to use.
        step_size: Step size for fixed-step methods (Euler, RK4).
                  For adaptive methods, this sets the output resolution.
        rtol: Relative tolerance for adaptive methods.
        atol: Absolute tolerance for adaptive methods.
        max_step: Maximum step size for adaptive methods.

    Returns:
        Trajectory containing the solution.
    """
    x0 = np.asarray(x0, dtype=float)
    t_start, t_end = t_span

    if method == IntegrationMethod.EULER:
        return _integrate_euler(f, x0, t_span, step_size or 0.01)
    elif method == IntegrationMethod.RK4:
        return _integrate_rk4(f, x0, t_span, step_size or 0.01)
    elif method == IntegrationMethod.RK45:
        return _integrate_scipy(f, x0, t_span, 'RK45', rtol, atol, step_size, max_step)
    elif method == IntegrationMethod.ADAMS:
        return _integrate_scipy(f, x0, t_span, 'LSODA', rtol, atol, step_size, max_step)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def _integrate_euler(
    f: DynamicsFunction,
    x0: np.ndarray,
    t_span: tuple[float, float],
    step_size: float
) -> Trajectory:
    """Forward Euler integration (first order)."""
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / step_size))
    times = np.linspace(t_start, t_end, n_steps + 1)
    actual_dt = (t_end - t_start) / n_steps

    states = np.zeros((n_steps + 1, len(x0)))
    states[0] = x0

    for i in range(n_steps):
        t = times[i]
        x = states[i]
        states[i + 1] = x + actual_dt * f(t, x)

    return Trajectory(times=times, states=states)


def _integrate_rk4(
    f: DynamicsFunction,
    x0: np.ndarray,
    t_span: tuple[float, float],
    step_size: float
) -> Trajectory:
    """Classic Runge-Kutta 4th order integration."""
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / step_size))
    times = np.linspace(t_start, t_end, n_steps + 1)
    dt = (t_end - t_start) / n_steps

    states = np.zeros((n_steps + 1, len(x0)))
    states[0] = x0

    for i in range(n_steps):
        t = times[i]
        x = states[i]

        k1 = f(t, x)
        k2 = f(t + dt/2, x + dt/2 * k1)
        k3 = f(t + dt/2, x + dt/2 * k2)
        k4 = f(t + dt, x + dt * k3)

        states[i + 1] = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return Trajectory(times=times, states=states)


def _integrate_scipy(
    f: DynamicsFunction,
    x0: np.ndarray,
    t_span: tuple[float, float],
    method: str,
    rtol: float,
    atol: float,
    output_step: float | None,
    max_step: float | None
) -> Trajectory:
    """Integration using scipy.integrate.solve_ivp."""
    t_start, t_end = t_span

    # If output_step specified, create evaluation points
    if output_step is not None:
        n_points = int(np.ceil((t_end - t_start) / output_step)) + 1
        t_eval = np.linspace(t_start, t_end, n_points)
    else:
        t_eval = None

    kwargs = {'rtol': rtol, 'atol': atol}
    if max_step is not None:
        kwargs['max_step'] = max_step

    result = solve_ivp(
        f, t_span, x0,
        method=method,
        t_eval=t_eval,
        dense_output=False,
        **kwargs
    )

    if not result.success:
        raise RuntimeError(f"Integration failed: {result.message}")

    return Trajectory(times=result.t, states=result.y.T)
