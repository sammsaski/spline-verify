"""Splines module: approximation, fitting, and optimization."""

from .approximation import SplineApproximation
from .multivariate import MultivariateSpline, ScatteredDataSpline
from .optimization import minimize_spline, MinimizationResult
from .piecewise import PiecewiseSplineApproximation, RegionInfo

__all__ = [
    # 1D splines
    "SplineApproximation",
    # Multivariate splines
    "MultivariateSpline",
    "ScatteredDataSpline",
    # Piecewise splines for switching systems
    "PiecewiseSplineApproximation",
    "RegionInfo",
    # Optimization
    "minimize_spline",
    "MinimizationResult",
]
