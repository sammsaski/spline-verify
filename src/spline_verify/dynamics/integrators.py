"""Numerical integrators for ODE systems.

Supports event detection for switching systems (Phase 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


# Type alias for event functions: g(t, x) = 0 triggers event
EventFunction = Callable[[float, np.ndarray], float]


@dataclass
class Event:
    """Event specification for integration with event detection.

    An event occurs when the event function crosses zero.

    Attributes:
        function: Event function g(t, x). Event triggers when g crosses zero.
        terminal: If True, stop integration when event occurs.
        direction: +1 for increasing crossings, -1 for decreasing, 0 for both.
    """
    function: EventFunction
    terminal: bool = True
    direction: int = 0  # 0 = both, +1 = increasing, -1 = decreasing


@dataclass
class IntegrationResult:
    """Result of integration with event detection.

    Attributes:
        trajectory: The computed trajectory.
        events: List of (event_index, event_time, event_state) tuples.
        terminated_by_event: True if integration stopped due to terminal event.
        termination_event_index: Index of the event that terminated, or None.
    """
    trajectory: Trajectory
    events: list[tuple[int, float, np.ndarray]] = field(default_factory=list)
    terminated_by_event: bool = False
    termination_event_index: int | None = None


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


def integrate_with_events(
    f: DynamicsFunction,
    x0: np.ndarray,
    t_span: tuple[float, float],
    events: list[Event],
    method: IntegrationMethod = IntegrationMethod.RK45,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float | None = None,
) -> IntegrationResult:
    """Integrate ODE with event detection.

    Uses scipy's solve_ivp event handling for accurate event detection.
    Events are detected when the event function crosses zero.

    Args:
        f: Dynamics function f(t, x) -> dx/dt.
        x0: Initial state, shape (n_dims,).
        t_span: Time interval (t_start, t_end).
        events: List of Event objects specifying event functions.
        method: Integration method (only RK45 and ADAMS support events).
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        max_step: Maximum step size.

    Returns:
        IntegrationResult with trajectory and event information.
    """
    if method not in (IntegrationMethod.RK45, IntegrationMethod.ADAMS):
        raise ValueError(
            f"Event detection only supported for RK45 and ADAMS methods, got {method}"
        )

    x0 = np.asarray(x0, dtype=float)
    scipy_method = 'RK45' if method == IntegrationMethod.RK45 else 'LSODA'

    # Convert Event objects to scipy event functions
    scipy_events = []
    for event in events:
        def make_event_func(e: Event):
            def event_func(t, x):
                return e.function(t, x)
            event_func.terminal = e.terminal
            event_func.direction = e.direction
            return event_func
        scipy_events.append(make_event_func(event))

    kwargs = {'rtol': rtol, 'atol': atol}
    if max_step is not None:
        kwargs['max_step'] = max_step

    result = solve_ivp(
        f, t_span, x0,
        method=scipy_method,
        events=scipy_events,
        dense_output=False,
        **kwargs
    )

    if not result.success:
        raise RuntimeError(f"Integration failed: {result.message}")

    trajectory = Trajectory(times=result.t, states=result.y.T)

    # Extract event information
    detected_events = []
    termination_event_index = None
    terminated_by_event = False

    for i, (t_events, y_events) in enumerate(zip(result.t_events, result.y_events)):
        for t_e, y_e in zip(t_events, y_events):
            detected_events.append((i, t_e, y_e))
            # Check if this event terminated the integration
            if events[i].terminal and len(t_events) > 0:
                if np.isclose(t_e, trajectory.times[-1], rtol=1e-10):
                    terminated_by_event = True
                    termination_event_index = i

    # Sort events by time
    detected_events.sort(key=lambda x: x[1])

    return IntegrationResult(
        trajectory=trajectory,
        events=detected_events,
        terminated_by_event=terminated_by_event,
        termination_event_index=termination_event_index
    )


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
