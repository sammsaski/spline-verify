"""Piecewise spline fitting for switching systems.

This module provides piecewise spline approximation for discontinuous objective
functions that arise in switching systems. Each continuity region gets its own
spline fit, with region boundaries learned via SVM classification.

Key features:
- Automatic region-based fitting from labeled data
- Per-region minimization with multi-start optimization
- Global minimum search across all regions
- Smooth evaluation within regions, discontinuous at boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize, differential_evolution

from .multivariate import MultivariateApproximation, ScatteredDataSpline


@dataclass
class RegionInfo:
    """Information about a single continuity region.

    Attributes:
        label: Integer label for this region.
        spline: Fitted spline for this region.
        points: Sample points in this region.
        values: Objective values at sample points.
        bounds: Bounding box of points in this region.
        n_samples: Number of samples in this region.
    """
    label: int
    spline: ScatteredDataSpline
    points: np.ndarray
    values: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray]
    n_samples: int


@dataclass
class PiecewiseSplineApproximation(MultivariateApproximation):
    """Piecewise spline approximation for discontinuous objective functions.

    For switching systems, F_T may be discontinuous at boundaries between
    different switching regions. This class fits separate splines on each
    continuous region and provides methods to minimize each region independently.

    Attributes:
        min_samples_per_region: Minimum samples needed to fit a regional spline.
        kernel: RBF kernel for spline fitting.
        smoothing: Smoothing parameter for spline fitting.

    Example:
        >>> spline = PiecewiseSplineApproximation()
        >>> spline.fit(points, values, region_labels)
        >>> spline.set_classifier(classifier.predict)
        >>> min_val, min_point = spline.global_minimum()
    """
    min_samples_per_region: int = 3
    kernel: str = 'thin_plate_spline'
    smoothing: float = 0.0

    _region_splines: dict[int, ScatteredDataSpline] = field(default_factory=dict)
    _region_info: dict[int, RegionInfo] = field(default_factory=dict)
    _classifier: Callable[[np.ndarray], int] | None = None
    _n_dims: int = 0
    _global_bounds: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def n_dims(self) -> int:
        """Dimension of the input space."""
        return self._n_dims

    @property
    def is_fitted(self) -> bool:
        """Whether any region splines have been fitted."""
        return len(self._region_splines) > 0

    @property
    def n_regions(self) -> int:
        """Number of continuity regions."""
        return len(self._region_splines)

    @property
    def region_labels(self) -> list[int]:
        """Labels of all fitted regions."""
        return list(self._region_splines.keys())

    def fit(
        self,
        points: np.ndarray,
        values: np.ndarray,
        region_labels: np.ndarray | None = None,
    ) -> "PiecewiseSplineApproximation":
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
        self._global_bounds = (points.min(axis=0), points.max(axis=0))

        if region_labels is None:
            region_labels = np.zeros(len(points), dtype=int)

        region_labels = np.asarray(region_labels)
        unique_regions = np.unique(region_labels)

        # Clear previous fits
        self._region_splines.clear()
        self._region_info.clear()

        for region in unique_regions:
            mask = region_labels == region
            region_points = points[mask]
            region_values = values[mask]

            if len(region_points) >= self.min_samples_per_region:
                spline = ScatteredDataSpline(
                    kernel=self.kernel,
                    smoothing=self.smoothing,
                )
                spline.fit(region_points, region_values)
                self._region_splines[int(region)] = spline

                # Store region info for minimization
                region_bounds = (
                    region_points.min(axis=0),
                    region_points.max(axis=0),
                )
                self._region_info[int(region)] = RegionInfo(
                    label=int(region),
                    spline=spline,
                    points=region_points.copy(),
                    values=region_values.copy(),
                    bounds=region_bounds,
                    n_samples=len(region_points),
                )

        return self

    def fit_with_classifier(
        self,
        points: np.ndarray,
        values: np.ndarray,
        classifier: Callable[[np.ndarray], int],
    ) -> "PiecewiseSplineApproximation":
        """Fit using a classifier to determine region labels.

        Args:
            points: Sample points, shape (n_samples, n_dims).
            values: Function values, shape (n_samples,).
            classifier: Function that returns region label for each point.

        Returns:
            self for method chaining.
        """
        points = np.asarray(points)
        region_labels = np.array([classifier(p) for p in points])
        self._classifier = classifier
        return self.fit(points, values, region_labels)

    def set_classifier(
        self,
        classifier: Callable[[np.ndarray], int],
    ) -> "PiecewiseSplineApproximation":
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
            # Default: use first available region
            region = list(self._region_splines.keys())[0]

        if region in self._region_splines:
            return self._region_splines[region].evaluate(x)
        else:
            # Fall back to nearest region based on minimum distance
            return self._fallback_evaluate(x)

    def _fallback_evaluate(self, x: np.ndarray) -> float:
        """Evaluate using nearest region when classifier returns unknown region."""
        min_dist = float('inf')
        nearest_region = None

        for region, info in self._region_info.items():
            # Distance to region centroid
            centroid = info.points.mean(axis=0)
            dist = np.linalg.norm(x - centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_region = region

        if nearest_region is not None:
            return self._region_splines[nearest_region].evaluate(x)
        else:
            return list(self._region_splines.values())[0].evaluate(x)

    def evaluate_region(self, x: np.ndarray, region: int) -> float:
        """Evaluate the spline for a specific region.

        Args:
            x: Point, shape (n_dims,).
            region: Region label.

        Returns:
            Approximated function value for that region.

        Raises:
            KeyError: If region not fitted.
        """
        if region not in self._region_splines:
            raise KeyError(f"Region {region} not fitted.")
        return self._region_splines[region].evaluate(x)

    def gradient(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute gradient via finite differences.

        Args:
            x: Point, shape (n_dims,).
            eps: Finite difference step size.

        Returns:
            Gradient vector, shape (n_dims,).
        """
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

    def region_minimum(
        self,
        region: int,
        n_starts: int = 10,
        method: str = 'multistart',
    ) -> tuple[float, np.ndarray] | None:
        """Find minimum within a specific region.

        Uses multi-start local optimization or differential evolution
        within the region's bounding box.

        Args:
            region: Region label.
            n_starts: Number of random starting points for multistart.
            method: 'multistart' for L-BFGS-B or 'de' for differential evolution.

        Returns:
            Tuple of (minimum_value, minimizing_point) or None if region not found.
        """
        if region not in self._region_info:
            return None

        info = self._region_info[region]
        spline = info.spline
        bounds_lower, bounds_upper = info.bounds

        # Slightly expand bounds to avoid boundary issues
        margin = 0.01 * (bounds_upper - bounds_lower)
        bounds_lower = bounds_lower - margin
        bounds_upper = bounds_upper + margin

        # Clip to global bounds if available
        if self._global_bounds is not None:
            bounds_lower = np.maximum(bounds_lower, self._global_bounds[0])
            bounds_upper = np.minimum(bounds_upper, self._global_bounds[1])

        scipy_bounds = list(zip(bounds_lower, bounds_upper))

        if method == 'de':
            # Differential evolution (global optimizer)
            result = differential_evolution(
                spline.evaluate,
                scipy_bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
            )
            return (float(result.fun), result.x.copy())

        # Multistart L-BFGS-B
        best_min = float('inf')
        best_x = None

        # Start from sample points in this region
        start_points = info.points[:min(n_starts, len(info.points))]

        # Add random points if needed
        n_random = max(0, n_starts - len(start_points))
        if n_random > 0:
            rng = np.random.default_rng(42)
            random_points = rng.uniform(
                bounds_lower, bounds_upper, size=(n_random, self._n_dims)
            )
            start_points = np.vstack([start_points, random_points])

        for x0 in start_points:
            try:
                result = minimize(
                    spline.evaluate,
                    x0,
                    method='L-BFGS-B',
                    bounds=scipy_bounds,
                    options={'maxiter': 100},
                )
                if result.fun < best_min:
                    best_min = result.fun
                    best_x = result.x.copy()
            except Exception:
                continue

        if best_x is not None:
            return (float(best_min), best_x)

        # Fallback: return minimum of sampled points
        min_idx = np.argmin(info.values)
        return (float(info.values[min_idx]), info.points[min_idx].copy())

    def global_minimum(
        self,
        n_starts_per_region: int = 10,
    ) -> tuple[float, np.ndarray, int]:
        """Find global minimum across all regions.

        Args:
            n_starts_per_region: Number of optimization starts per region.

        Returns:
            Tuple of (minimum_value, minimizing_point, region_label).
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")

        best_min = float('inf')
        best_x = None
        best_region = -1

        for region in self._region_splines:
            result = self.region_minimum(region, n_starts=n_starts_per_region)
            if result is not None:
                val, x = result
                if val < best_min:
                    best_min = val
                    best_x = x
                    best_region = region

        if best_x is None:
            raise RuntimeError("No minimum found in any region.")

        return (best_min, best_x, best_region)

    def get_region_info(self, region: int) -> RegionInfo | None:
        """Get information about a specific region.

        Args:
            region: Region label.

        Returns:
            RegionInfo object or None if region not found.
        """
        return self._region_info.get(region)

    def all_region_minima(
        self,
        n_starts_per_region: int = 10,
    ) -> dict[int, tuple[float, np.ndarray]]:
        """Find minima for all regions.

        Args:
            n_starts_per_region: Number of optimization starts per region.

        Returns:
            Dict mapping region label to (minimum_value, minimizing_point).
        """
        results = {}
        for region in self._region_splines:
            result = self.region_minimum(region, n_starts=n_starts_per_region)
            if result is not None:
                results[region] = result
        return results

    def __repr__(self) -> str:
        if self.is_fitted:
            region_sizes = [
                f"{r}:{self._region_info[r].n_samples}"
                for r in sorted(self._region_info.keys())
            ]
            return (
                f"PiecewiseSplineApproximation(n_regions={self.n_regions}, "
                f"n_dims={self._n_dims}, samples_per_region={{{', '.join(region_sizes)}}})"
            )
        return "PiecewiseSplineApproximation(not fitted)"
