"""Geometry module: set representations, distance computation, and sampling."""

from .sets import Set, HyperRectangle, Ball, HalfSpace, ConvexPolytope, LevelSet, UnionSet
from .sampling import sample_set, SamplingStrategy

__all__ = [
    "Set",
    "HyperRectangle",
    "Ball",
    "HalfSpace",
    "ConvexPolytope",
    "LevelSet",
    "UnionSet",
    "sample_set",
    "SamplingStrategy",
]
