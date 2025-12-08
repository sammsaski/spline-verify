"""Splines module: approximation, fitting, and optimization."""

from .approximation import SplineApproximation
from .multivariate import MultivariateSpline, ScatteredDataSpline
from .optimization import minimize_spline, MinimizationResult

__all__ = [
    "SplineApproximation",
    "MultivariateSpline",
    "ScatteredDataSpline",
    "minimize_spline",
    "MinimizationResult",
]
