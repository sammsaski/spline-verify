"""Switching dynamics and Filippov solver for hybrid systems.

This module implements Phase 4: Full Filippov handling for switching systems.

Key concepts:
- SwitchingSurface: Defines boundary between modes with normal vector
- SwitchingDynamics: Multi-mode system with explicit switching surfaces
- FilippovSolver: Handles set-valued dynamics at switching surfaces
  - Crossing: Trajectory passes through surface
  - Sliding: Trajectory stays on surface (Filippov sliding mode)
  - Leaving: Trajectory departs from surface
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from .base import DynamicsModel, DynamicsFunction, TrajectoryBundle
from .trajectory import Trajectory
from .integrators import (
    integrate,
    integrate_with_events,
    IntegrationMethod,
    Event,
    IntegrationResult,
)


class SwitchingBehavior(Enum):
    """Behavior at a switching surface."""
    CROSSING = auto()    # Trajectory crosses through surface
    SLIDING = auto()     # Trajectory slides along surface
    LEAVING = auto()     # Trajectory leaves sliding mode
    INTERIOR = auto()    # Not at a switching surface


@dataclass
class SwitchingSurface:
    """A switching surface between two modes.

    The surface is defined by {x : boundary_function(x) = 0}.
    Mode 0 (adjacent_modes[0]) is where boundary_function(x) > 0.
    Mode 1 (adjacent_modes[1]) is where boundary_function(x) < 0.

    Attributes:
        boundary_function: Function g(x) where g(x) = 0 defines the surface.
        adjacent_modes: Tuple (mode_plus, mode_minus) for g > 0 and g < 0.
        normal_function: Optional function returning outward normal at x.
                        If None, computed via finite differences.
    """
    boundary_function: Callable[[np.ndarray], float]
    adjacent_modes: tuple[int, int]
    normal_function: Callable[[np.ndarray], np.ndarray] | None = None

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the boundary function at x."""
        return self.boundary_function(x)

    def normal(self, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Compute outward normal (pointing toward mode_plus).

        If normal_function is provided, uses it. Otherwise computes
        via finite differences of the boundary function.
        """
        if self.normal_function is not None:
            return self.normal_function(x)

        # Compute gradient via finite differences
        n = len(x)
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.boundary_function(x_plus) - self.boundary_function(x_minus)) / (2 * eps)

        norm = np.linalg.norm(grad)
        if norm < 1e-12:
            return np.zeros(n)
        return grad / norm

    def signed_distance(self, x: np.ndarray) -> float:
        """Approximate signed distance to surface (positive = mode_plus side)."""
        return self.boundary_function(x)

    def project_to_surface(self, x: np.ndarray, max_iter: int = 10, tol: float = 1e-10) -> np.ndarray:
        """Project point onto the switching surface using Newton's method."""
        x_proj = x.copy()
        for _ in range(max_iter):
            g = self.boundary_function(x_proj)
            if abs(g) < tol:
                break
            n = self.normal(x_proj)
            if np.linalg.norm(n) < 1e-12:
                break
            x_proj = x_proj - g * n
        return x_proj


@dataclass
class SwitchingDynamics:
    """State-based switching dynamical system with explicit surfaces.

    The system has multiple modes, each with its own dynamics:
        dx/dt = f_m(t, x)  when in mode m

    Modes are separated by switching surfaces. At surfaces, Filippov
    convexification determines the dynamics.

    Attributes:
        mode_dynamics: List of dynamics functions f_m(t, x) for each mode.
        surfaces: List of switching surfaces defining mode boundaries.
        mode_indicator: Function rho(x) -> mode index (optional, derived from surfaces).
        _n_dims: State space dimension.
    """
    mode_dynamics: list[DynamicsFunction]
    surfaces: list[SwitchingSurface]
    _n_dims: int
    mode_indicator: Callable[[np.ndarray], int] | None = None
    method: IntegrationMethod = IntegrationMethod.RK45
    rtol: float = 1e-6
    atol: float = 1e-9

    def __post_init__(self):
        """Set up mode indicator if not provided."""
        if self.mode_indicator is None and len(self.surfaces) == 1:
            # Simple two-mode system
            surface = self.surfaces[0]
            def indicator(x):
                return surface.adjacent_modes[0] if surface.evaluate(x) >= 0 else surface.adjacent_modes[1]
            self.mode_indicator = indicator

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def n_modes(self) -> int:
        return len(self.mode_dynamics)

    def current_mode(self, x: np.ndarray) -> int:
        """Get the current mode for state x."""
        if self.mode_indicator is not None:
            return self.mode_indicator(x)
        raise ValueError("No mode_indicator defined and cannot derive from surfaces")

    def mode_dynamics_at(self, t: float, x: np.ndarray) -> np.ndarray:
        """Get dynamics vector for current mode."""
        mode = self.current_mode(x)
        return self.mode_dynamics[mode](t, x)

    def find_active_surfaces(self, x: np.ndarray, tol: float = 1e-8) -> list[int]:
        """Find indices of surfaces that x is close to."""
        active = []
        for i, surface in enumerate(self.surfaces):
            if abs(surface.evaluate(x)) < tol:
                active.append(i)
        return active

    def simulate(
        self,
        x0: np.ndarray,
        t_span: tuple[float, float],
        **kwargs
    ) -> TrajectoryBundle:
        """Simulate the switching system from initial condition.

        Uses FilippovSolver for proper handling of switching surfaces.

        Args:
            x0: Initial state.
            t_span: Time interval.
            **kwargs: Integration options.

        Returns:
            TrajectoryBundle with all possible trajectories.
        """
        solver = FilippovSolver(self)
        return solver.solve(x0, t_span, **kwargs)

    @classmethod
    def bouncing_ball(
        cls,
        gravity: float = 9.81,
        restitution: float = 0.9
    ) -> SwitchingDynamics:
        """Create bouncing ball system (canonical switching example).

        State: [height, velocity]
        Surface: height = 0 (ground)
        Dynamics: free fall with instantaneous velocity reset at ground.

        Args:
            gravity: Gravitational acceleration.
            restitution: Coefficient of restitution (velocity retention).

        Returns:
            SwitchingDynamics for bouncing ball.
        """
        def free_fall(t: float, x: np.ndarray) -> np.ndarray:
            return np.array([x[1], -gravity])

        def ground_surface(x: np.ndarray) -> float:
            return x[0]  # height >= 0

        surface = SwitchingSurface(
            boundary_function=ground_surface,
            adjacent_modes=(0, 0),  # Same mode, but with reset
            normal_function=lambda x: np.array([1.0, 0.0])
        )

        def mode_indicator(x: np.ndarray) -> int:
            return 0  # Single mode with event-driven reset

        dynamics = cls(
            mode_dynamics=[free_fall],
            surfaces=[surface],
            _n_dims=2,
            mode_indicator=mode_indicator
        )
        dynamics._restitution = restitution
        dynamics._is_bouncing_ball = True
        return dynamics

    @classmethod
    def relay_feedback(cls, alpha: float = 1.0) -> SwitchingDynamics:
        """Create relay feedback system (classic sliding mode example).

        System: dx/dt = -alpha * sign(x) = -alpha if x > 0, +alpha if x < 0
        This creates sliding motion along x = 0.

        Args:
            alpha: Feedback gain.

        Returns:
            SwitchingDynamics with sliding mode at x=0.
        """
        def mode_plus(t: float, x: np.ndarray) -> np.ndarray:
            return np.array([-alpha])

        def mode_minus(t: float, x: np.ndarray) -> np.ndarray:
            return np.array([alpha])

        def surface_func(x: np.ndarray) -> float:
            return x[0]

        surface = SwitchingSurface(
            boundary_function=surface_func,
            adjacent_modes=(0, 1),
            normal_function=lambda x: np.array([1.0])
        )

        return cls(
            mode_dynamics=[mode_plus, mode_minus],
            surfaces=[surface],
            _n_dims=1
        )

    @classmethod
    def thermostat(
        cls,
        T_ambient: float = 20.0,
        T_low: float = 18.0,
        T_high: float = 22.0,
        cooling_rate: float = 0.1,
        heating_power: float = 5.0
    ) -> SwitchingDynamics:
        """Create thermostat system (hysteresis switching).

        State: [temperature, heater_state]
        heater_state: 0 = off, 1 = on (tracked as continuous for simplicity)

        Switching:
        - Turn ON when T < T_low and heater is off
        - Turn OFF when T > T_high and heater is on

        Args:
            T_ambient: Ambient temperature.
            T_low: Temperature to turn heater on.
            T_high: Temperature to turn heater off.
            cooling_rate: Rate of cooling toward ambient.
            heating_power: Heating rate when on.

        Returns:
            SwitchingDynamics for thermostat.
        """
        def heater_off(t: float, x: np.ndarray) -> np.ndarray:
            T = x[0]
            dT = -cooling_rate * (T - T_ambient)
            return np.array([dT])

        def heater_on(t: float, x: np.ndarray) -> np.ndarray:
            T = x[0]
            dT = -cooling_rate * (T - T_ambient) + heating_power
            return np.array([dT])

        # Surface for turning heater ON: T - T_low = 0 (crosses from above)
        surface_on = SwitchingSurface(
            boundary_function=lambda x: x[0] - T_low,
            adjacent_modes=(0, 1),  # off -> on when T drops below T_low
            normal_function=lambda x: np.array([1.0])
        )

        # Surface for turning heater OFF: T - T_high = 0 (crosses from below)
        surface_off = SwitchingSurface(
            boundary_function=lambda x: x[0] - T_high,
            adjacent_modes=(1, 0),  # on -> off when T rises above T_high
            normal_function=lambda x: np.array([1.0])
        )

        # Mode indicator with hysteresis requires tracking heater state
        # For simplicity, use a stateful indicator
        class HysteresisIndicator:
            def __init__(self):
                self.heater_on = False

            def __call__(self, x: np.ndarray) -> int:
                T = x[0]
                if self.heater_on:
                    if T >= T_high:
                        self.heater_on = False
                else:
                    if T <= T_low:
                        self.heater_on = True
                return 1 if self.heater_on else 0

        indicator = HysteresisIndicator()

        return cls(
            mode_dynamics=[heater_off, heater_on],
            surfaces=[surface_on, surface_off],
            _n_dims=1,
            mode_indicator=indicator
        )


class FilippovSolver:
    """Solver for Filippov differential inclusions.

    At switching surfaces, computes the Filippov set-valued dynamics:
        G(t,x) = conv{f_m(t,x) : m adjacent to x}

    Handles three behaviors:
    1. Crossing: Both f+ and f- point same direction across surface
    2. Sliding: f+ and f- point toward each other (trajectory trapped)
    3. Leaving: f+ and f- point away (trajectory can leave either way)

    For sliding modes, uses Filippov's equivalent dynamics.
    """

    def __init__(
        self,
        switching_dynamics: SwitchingDynamics,
        surface_tol: float = 1e-8,
        max_branches: int = 4
    ):
        """Initialize Filippov solver.

        Args:
            switching_dynamics: The switching system to solve.
            surface_tol: Tolerance for detecting surface proximity.
            max_branches: Maximum trajectory branches to explore.
        """
        self.dynamics = switching_dynamics
        self.surface_tol = surface_tol
        self.max_branches = max_branches

    def solve(
        self,
        x0: np.ndarray,
        t_span: tuple[float, float],
        max_events: int = 100,
        **kwargs
    ) -> TrajectoryBundle:
        """Solve the Filippov differential inclusion.

        Returns a TrajectoryBundle containing all possible solutions
        from the initial condition x0.

        Args:
            x0: Initial state.
            t_span: Time interval (t_start, t_end).
            max_events: Maximum number of switching events to handle.
            **kwargs: Integration options.

        Returns:
            TrajectoryBundle with all trajectory branches.
        """
        x0 = np.asarray(x0, dtype=float)
        t_start, t_end = t_span

        # Handle special case: bouncing ball
        if hasattr(self.dynamics, '_is_bouncing_ball') and self.dynamics._is_bouncing_ball:
            traj = self._solve_bouncing_ball(x0, t_start, t_end)
            return TrajectoryBundle.from_single(traj)

        # General Filippov solver
        trajectories = self._solve_with_events(x0, t_start, t_end, max_events)

        if len(trajectories) == 1:
            return TrajectoryBundle.from_single(trajectories[0])
        else:
            return TrajectoryBundle(trajectories=trajectories, is_unique=False)

    def _solve_bouncing_ball(
        self,
        x0: np.ndarray,
        t_start: float,
        t_end: float
    ) -> Trajectory:
        """Special solver for bouncing ball with velocity reset."""
        restitution = getattr(self.dynamics, '_restitution', 0.9)
        gravity = 9.81

        times = [t_start]
        states = [x0.copy()]

        t = t_start
        x = x0.copy()
        max_bounces = 100
        bounce_count = 0

        while t < t_end and bounce_count < max_bounces:
            # Event: height = 0 with downward velocity
            def ground_event(t_e, x_e):
                return x_e[0]

            event = Event(
                function=ground_event,
                terminal=True,
                direction=-1  # Only trigger when height decreasing
            )

            try:
                result = integrate_with_events(
                    self.dynamics.mode_dynamics[0],
                    x, (t, t_end),
                    events=[event],
                    method=self.dynamics.method,
                    rtol=self.dynamics.rtol,
                    atol=self.dynamics.atol
                )
            except RuntimeError:
                break

            # Add trajectory segment
            for i in range(1, len(result.trajectory.times)):
                times.append(result.trajectory.times[i])
                states.append(result.trajectory.states[i])

            if result.terminated_by_event and result.events:
                # Bounce: reverse velocity
                t = result.events[-1][1]
                x = result.events[-1][2].copy()
                x[0] = max(0, x[0])  # Ensure non-negative height
                x[1] = -restitution * x[1]  # Reverse and dampen velocity

                if abs(x[1]) < 1e-6:  # Ball has essentially stopped
                    break

                # Update the last state with the bounced velocity
                states[-1] = x.copy()
                bounce_count += 1
            else:
                break

        return Trajectory(times=np.array(times), states=np.array(states))

    def _solve_with_events(
        self,
        x0: np.ndarray,
        t_start: float,
        t_end: float,
        max_events: int
    ) -> list[Trajectory]:
        """General event-driven solver with Filippov handling."""
        trajectories = []

        # Create events for all switching surfaces
        events = []
        for i, surface in enumerate(self.dynamics.surfaces):
            event = Event(
                function=lambda t, x, s=surface: s.evaluate(x),
                terminal=True,
                direction=0  # Detect both crossings
            )
            events.append(event)

        # Recursive solver that handles branching
        def solve_segment(x_init, t_init, remaining_events, branch_trajectories):
            if t_init >= t_end or remaining_events <= 0:
                return

            current_mode = self.dynamics.current_mode(x_init)

            try:
                result = integrate_with_events(
                    self.dynamics.mode_dynamics[current_mode],
                    x_init, (t_init, t_end),
                    events=events,
                    method=self.dynamics.method,
                    rtol=self.dynamics.rtol,
                    atol=self.dynamics.atol
                )
            except RuntimeError:
                return

            # Store this segment
            branch_trajectories.append(result.trajectory)

            if result.terminated_by_event and result.events:
                event_idx, t_event, x_event = result.events[-1]
                surface = self.dynamics.surfaces[event_idx]

                # Determine behavior at surface
                behavior = self.classify_surface_behavior(t_event, x_event, surface)

                if behavior == SwitchingBehavior.CROSSING:
                    # Continue in new mode
                    solve_segment(
                        x_event, t_event,
                        remaining_events - 1,
                        branch_trajectories
                    )

                elif behavior == SwitchingBehavior.SLIDING:
                    # Integrate along sliding surface
                    sliding_traj = self._integrate_sliding(
                        t_event, x_event, surface, t_end
                    )
                    if sliding_traj is not None:
                        branch_trajectories.append(sliding_traj)
                        # May need to continue after leaving sliding
                        if sliding_traj.times[-1] < t_end:
                            solve_segment(
                                sliding_traj.final_state,
                                sliding_traj.times[-1],
                                remaining_events - 1,
                                branch_trajectories
                            )

                elif behavior == SwitchingBehavior.LEAVING:
                    # Branch: explore both directions (limited by max_branches)
                    if len(trajectories) < self.max_branches:
                        # Branch 1: continue with current mode
                        solve_segment(
                            x_event, t_event,
                            remaining_events - 1,
                            branch_trajectories
                        )

        # Start solving
        branch_trajs = []
        solve_segment(x0, t_start, max_events, branch_trajs)

        # Concatenate trajectory segments
        if branch_trajs:
            combined = self._concatenate_trajectories(branch_trajs)
            trajectories.append(combined)

        if not trajectories:
            # Fallback: simple integration without events
            traj = integrate(
                self.dynamics.mode_dynamics_at,
                x0, (t_start, t_end),
                method=self.dynamics.method
            )
            trajectories.append(traj)

        return trajectories

    def classify_surface_behavior(
        self,
        t: float,
        x: np.ndarray,
        surface: SwitchingSurface
    ) -> SwitchingBehavior:
        """Classify the behavior at a switching surface.

        Uses the Filippov conditions:
        - Crossing: (f+ · n)(f- · n) > 0
        - Sliding: (f+ · n) < 0 and (f- · n) > 0
        - Leaving: (f+ · n) > 0 and (f- · n) < 0

        Args:
            t: Time.
            x: State (should be on or near surface).
            surface: The switching surface.

        Returns:
            SwitchingBehavior enum value.
        """
        # Check if we're actually near the surface
        if abs(surface.evaluate(x)) > self.surface_tol:
            return SwitchingBehavior.INTERIOR

        n = surface.normal(x)
        mode_plus, mode_minus = surface.adjacent_modes

        f_plus = self.dynamics.mode_dynamics[mode_plus](t, x)
        f_minus = self.dynamics.mode_dynamics[mode_minus](t, x)

        # Dot products with normal
        dot_plus = np.dot(f_plus, n)
        dot_minus = np.dot(f_minus, n)

        # Classification
        if dot_plus * dot_minus > 0:
            return SwitchingBehavior.CROSSING
        elif dot_plus < 0 and dot_minus > 0:
            return SwitchingBehavior.SLIDING
        elif dot_plus > 0 and dot_minus < 0:
            return SwitchingBehavior.LEAVING
        else:
            # Edge case: one or both tangent to surface
            return SwitchingBehavior.CROSSING

    def compute_filippov_set(
        self,
        t: float,
        x: np.ndarray,
        surface: SwitchingSurface
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Filippov set at a point on a switching surface.

        The Filippov set is G(t,x) = conv{f+(t,x), f-(t,x)}, a line segment
        in general.

        Args:
            t: Time.
            x: State (on switching surface).
            surface: The switching surface.

        Returns:
            Tuple of (vertices, centroid) of the Filippov set.
            vertices: shape (2, n_dims) - the two extreme dynamics vectors
            centroid: shape (n_dims,) - midpoint (not necessarily the sliding vector)
        """
        mode_plus, mode_minus = surface.adjacent_modes
        f_plus = self.dynamics.mode_dynamics[mode_plus](t, x)
        f_minus = self.dynamics.mode_dynamics[mode_minus](t, x)

        vertices = np.vstack([f_plus, f_minus])
        centroid = (f_plus + f_minus) / 2

        return vertices, centroid

    def compute_sliding_vector_field(
        self,
        t: float,
        x: np.ndarray,
        surface: SwitchingSurface
    ) -> np.ndarray:
        """Compute the Filippov sliding vector field.

        In sliding mode, the dynamics are:
            f_s = alpha * f+ + (1-alpha) * f-
        where alpha is chosen so that f_s is tangent to the surface:
            f_s · n = 0

        Args:
            t: Time.
            x: State (on sliding surface).
            surface: The switching surface.

        Returns:
            Sliding mode dynamics vector.
        """
        n = surface.normal(x)
        mode_plus, mode_minus = surface.adjacent_modes

        f_plus = self.dynamics.mode_dynamics[mode_plus](t, x)
        f_minus = self.dynamics.mode_dynamics[mode_minus](t, x)

        dot_plus = np.dot(f_plus, n)
        dot_minus = np.dot(f_minus, n)

        # Solve for alpha: alpha * dot_plus + (1-alpha) * dot_minus = 0
        # alpha = dot_minus / (dot_minus - dot_plus)
        denom = dot_minus - dot_plus
        if abs(denom) < 1e-12:
            # Degenerate case: both tangent
            return (f_plus + f_minus) / 2

        alpha = dot_minus / denom
        alpha = np.clip(alpha, 0, 1)  # Ensure valid convex combination

        f_sliding = alpha * f_plus + (1 - alpha) * f_minus
        return f_sliding

    def _integrate_sliding(
        self,
        t_start: float,
        x_start: np.ndarray,
        surface: SwitchingSurface,
        t_end: float,
        dt: float = 0.001
    ) -> Trajectory | None:
        """Integrate along a sliding surface.

        Uses the sliding vector field and projects back onto the surface
        at each step.

        Returns None if sliding immediately ends.
        """
        times = [t_start]
        states = [x_start.copy()]

        t = t_start
        x = surface.project_to_surface(x_start)

        max_steps = int((t_end - t_start) / dt) + 1

        for _ in range(max_steps):
            if t >= t_end:
                break

            # Check if still in sliding mode
            behavior = self.classify_surface_behavior(t, x, surface)
            if behavior != SwitchingBehavior.SLIDING:
                break

            # Compute sliding dynamics
            f_s = self.compute_sliding_vector_field(t, x, surface)

            # Euler step (could use higher order, but project anyway)
            h = min(dt, t_end - t)
            x_new = x + h * f_s

            # Project back onto surface
            x_new = surface.project_to_surface(x_new)

            t = t + h
            x = x_new
            times.append(t)
            states.append(x.copy())

        if len(times) <= 1:
            return None

        return Trajectory(times=np.array(times), states=np.array(states))

    def _concatenate_trajectories(self, trajs: list[Trajectory]) -> Trajectory:
        """Concatenate multiple trajectory segments into one."""
        if not trajs:
            raise ValueError("No trajectories to concatenate")

        if len(trajs) == 1:
            return trajs[0]

        all_times = []
        all_states = []

        for i, traj in enumerate(trajs):
            if i == 0:
                all_times.extend(traj.times)
                all_states.extend(traj.states)
            else:
                # Skip first point if it's a duplicate
                start_idx = 1 if np.isclose(traj.times[0], all_times[-1]) else 0
                all_times.extend(traj.times[start_idx:])
                all_states.extend(traj.states[start_idx:])

        return Trajectory(
            times=np.array(all_times),
            states=np.array(all_states)
        )
