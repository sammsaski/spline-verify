"""Dynamics module: ODE definitions, integrators, and trajectory handling."""

from .trajectory import Trajectory
from .base import DynamicsModel, TrajectoryBundle
from .integrators import (
    integrate,
    integrate_with_events,
    IntegrationMethod,
    Event,
    IntegrationResult,
)
from .ode import ODEDynamics
from .switching import (
    SwitchingDynamics,
    SwitchingSurface,
    SwitchingBehavior,
    FilippovSolver,
)

__all__ = [
    "Trajectory",
    "TrajectoryBundle",
    "DynamicsModel",
    "integrate",
    "integrate_with_events",
    "IntegrationMethod",
    "Event",
    "IntegrationResult",
    "ODEDynamics",
    "SwitchingDynamics",
    "SwitchingSurface",
    "SwitchingBehavior",
    "FilippovSolver",
]
