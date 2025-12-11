"""Miller-Sznaier Distance Estimation via Occupation Measures.

This module implements the convex optimization approach for bounding
the distance to unsafe sets, based on:

    Miller, J. & Sznaier, M. (2023). "Bounding the Distance to Unsafe Sets
    with Convex Optimization", IEEE Transactions on Automatic Control.
    https://arxiv.org/abs/2110.14047

The method uses:
- Occupation measures to represent trajectory distributions
- Monge-Kantorovich optimal transport relaxation
- Moment-SOS hierarchy for semidefinite programming

This provides an alternative to spline approximation with:
- Certified lower bounds (vs upper bounds from sampling)
- No discretization error from sampling
- Higher computational cost (SDP scaling)
"""

from .distance_estimator import (
    DistanceEstimator,
    DistanceResult,
    CVXPY_AVAILABLE,
)
from .problem import (
    UnsafeSupport,
    create_flow_system,
    create_twist_system,
)

__all__ = [
    'DistanceEstimator',
    'DistanceResult',
    'CVXPY_AVAILABLE',
    'UnsafeSupport',
    'create_flow_system',
    'create_twist_system',
]
