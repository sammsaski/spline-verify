"""Piecewise spline fitting for switching systems (Phase 4 placeholder)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .multivariate import MultivariateApproximation, ScatteredDataSpline


@dataclass
class PiecewiseSplineApproximation(MultivariateApproximation):
    """Piecewise spline approximation for discontinuous objective functions.

    For switching systems, F_T may be discontinuous at boundaries between
    different switching regions. This class fits separate splines on each
    continuous region.

    This is a placeholder implementation for Phase 4.
    """
    _region_splines: dict[int, ScatteredDataSpline] = field(default_factory=dict)
    _classifier: Callable[[np.ndarray], int] | None = None
    _n_dims: int = 0

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def is_fitted(self) -> bool:
        return len(self._region_splines) > 0

    @property
    def n_regions(self) -> int:
        """Number of continuity regions."""
        return len(self._region_splines)

    def fit(
        self,
        points: np.ndarray,
        values: np.ndarray,
        region_labels: np.ndarray | None = None
    ) -> PiecewiseSplineApproximation:
        """Fit piecewise spline approximation.

        Args:
            points: Sample points, shape (n_samples, n_dims).
            values: Function values, shape (n_samples,).
            region_labels: Integer labels for each point's region.
                          If None, treats all points as one region.

        Returns:
            self for method chaining.
        """
        points = np.asarray(points)
        values = np.asarray(values)

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        self._n_dims = points.shape[1]

        if region_labels is None:
            region_labels = np.zeros(len(points), dtype=int)

        region_labels = np.asarray(region_labels)
        unique_regions = np.unique(region_labels)

        for region in unique_regions:
            mask = region_labels == region
            region_points = points[mask]
            region_values = values[mask]

            if len(region_points) >= 3:  # Need at least 3 points for RBF
                spline = ScatteredDataSpline()
                spline.fit(region_points, region_values)
                self._region_splines[int(region)] = spline

        return self

    def set_classifier(
        self,
        classifier: Callable[[np.ndarray], int]
    ) -> PiecewiseSplineApproximation:
        """Set the region classifier function.

        Args:
            classifier: Function that takes a point and returns region label.

        Returns:
            self for method chaining.
        """
        self._classifier = classifier
        return self

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the piecewise approximation at a point.

        Args:
            x: Point, shape (n_dims,).

        Returns:
            Approximated function value.
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        x = np.asarray(x)

        # Determine which region x belongs to
        if self._classifier is not None:
            region = self._classifier(x)
        else:
            # Default: use region 0
            region = 0

        if region in self._region_splines:
            return self._region_splines[region].evaluate(x)
        else:
            # Fall back to nearest region
            return list(self._region_splines.values())[0].evaluate(x)

    def gradient(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute gradient via finite differences."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")

        x = np.asarray(x)
        grad = np.zeros(self._n_dims)

        for i in range(self._n_dims):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps

            grad[i] = (self.evaluate(x_plus) - self.evaluate(x_minus)) / (2 * eps)

        return grad

    def region_minimum(self, region: int) -> tuple[float, np.ndarray] | None:
        """Find minimum within a specific region.

        Note: This requires the region's bounding box to be known.
        Placeholder implementation.

        Args:
            region: Region label.

        Returns:
            Tuple of (minimum_value, minimizing_point) or None if region not found.
        """
        if region not in self._region_splines:
            return None

        # Placeholder: return minimum of sampled points
        # Full implementation would need region bounds
        spline = self._region_splines[region]
        if spline._values is not None:
            min_idx = np.argmin(spline._values)
            return (float(spline._values[min_idx]), spline._points[min_idx].copy())
        return None

    def __repr__(self) -> str:
        if self.is_fitted:
            return (
                f"PiecewiseSplineApproximation(n_regions={self.n_regions}, "
                f"n_dims={self._n_dims})"
            )
        return "PiecewiseSplineApproximation(not fitted)"
