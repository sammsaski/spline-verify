"""Multivariate spline approximation for F_T over initial set."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.interpolate import RBFInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator


class MultivariateApproximation(ABC):
    """Abstract base class for multivariate function approximation."""

    @property
    @abstractmethod
    def n_dims(self) -> int:
        """Dimension of the domain."""
        ...

    @abstractmethod
    def fit(self, points: np.ndarray, values: np.ndarray) -> MultivariateApproximation:
        """Fit the approximation to data.

        Args:
            points: Sample points, shape (n_samples, n_dims).
            values: Function values, shape (n_samples,).

        Returns:
            self for method chaining.
        """
        ...

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the approximation at a point.

        Args:
            x: Point, shape (n_dims,).

        Returns:
            Approximated function value.
        """
        ...

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of the approximation.

        Args:
            x: Point, shape (n_dims,).

        Returns:
            Gradient vector, shape (n_dims,).
        """
        ...

    def __call__(self, x: np.ndarray) -> float:
        return self.evaluate(x)


@dataclass
class ScatteredDataSpline(MultivariateApproximation):
    """Scattered data interpolation using radial basis functions.

    This uses scipy's RBFInterpolator which supports various RBF kernels
    including thin-plate splines.
    """
    kernel: str = 'thin_plate_spline'  # or 'cubic', 'gaussian', 'multiquadric'
    smoothing: float = 0.0
    _interpolator: RBFInterpolator | None = field(default=None, repr=False)
    _n_dims: int = field(default=0, repr=False)
    _points: np.ndarray | None = field(default=None, repr=False)
    _values: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def is_fitted(self) -> bool:
        return self._interpolator is not None

    def fit(self, points: np.ndarray, values: np.ndarray) -> ScatteredDataSpline:
        """Fit RBF interpolation to scattered data.

        Args:
            points: Sample points, shape (n_samples, n_dims).
            values: Function values, shape (n_samples,).

        Returns:
            self for method chaining.
        """
        points = np.asarray(points)
        values = np.asarray(values)

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        self._n_dims = points.shape[1]
        self._points = points.copy()
        self._values = values.copy()

        self._interpolator = RBFInterpolator(
            points, values,
            kernel=self.kernel,
            smoothing=self.smoothing
        )

        return self

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the RBF interpolation at a point."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        x = np.asarray(x)
        if x.ndim == 0 or (x.ndim == 1 and self._n_dims == 1):
            x = x.reshape(1, -1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        result = self._interpolator(x)
        return float(result[0])

    def evaluate_batch(self, x: np.ndarray) -> np.ndarray:
        """Evaluate at multiple points.

        Args:
            x: Points, shape (n_points, n_dims).

        Returns:
            Values, shape (n_points,).
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return self._interpolator(x)

    def gradient(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute gradient via finite differences.

        Args:
            x: Point, shape (n_dims,).
            eps: Finite difference step size.

        Returns:
            Gradient vector, shape (n_dims,).
        """
        x = np.asarray(x)
        grad = np.zeros(self._n_dims)

        for i in range(self._n_dims):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps

            grad[i] = (self.evaluate(x_plus) - self.evaluate(x_minus)) / (2 * eps)

        return grad

    def residuals(self) -> np.ndarray:
        """Compute residuals at training points (for smoothing splines)."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")

        predicted = self.evaluate_batch(self._points)
        return self._values - predicted

    def __repr__(self) -> str:
        if self.is_fitted:
            return (
                f"ScatteredDataSpline(kernel='{self.kernel}', "
                f"n_points={len(self._points)}, n_dims={self._n_dims})"
            )
        return f"ScatteredDataSpline(kernel='{self.kernel}', not fitted)"


@dataclass
class MultivariateSpline(MultivariateApproximation):
    """Tensor-product spline for regular grid data.

    For scattered data, falls back to Delaunay triangulation + linear
    interpolation, or RBF if specified.
    """
    method: str = 'linear'  # 'linear', 'nearest', or 'rbf'
    _interpolator: Callable | None = field(default=None, repr=False)
    _n_dims: int = field(default=0, repr=False)
    _bounds: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def is_fitted(self) -> bool:
        return self._interpolator is not None

    def fit(self, points: np.ndarray, values: np.ndarray) -> MultivariateSpline:
        """Fit interpolation to data.

        Args:
            points: Sample points, shape (n_samples, n_dims).
            values: Function values, shape (n_samples,).

        Returns:
            self for method chaining.
        """
        points = np.asarray(points)
        values = np.asarray(values)

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        self._n_dims = points.shape[1]
        self._bounds = (points.min(axis=0), points.max(axis=0))

        if self.method == 'linear':
            self._interpolator = LinearNDInterpolator(points, values)
        elif self.method == 'nearest':
            self._interpolator = NearestNDInterpolator(points, values)
        elif self.method == 'rbf':
            self._interpolator = RBFInterpolator(points, values)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the interpolation at a point."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        result = self._interpolator(x)

        # Handle NaN from extrapolation
        if np.isnan(result[0]):
            # Fall back to nearest neighbor for points outside convex hull
            if self.method != 'nearest':
                nearest = NearestNDInterpolator(
                    self._interpolator.points if hasattr(self._interpolator, 'points')
                    else self._interpolator._tree.data,
                    self._interpolator.values if hasattr(self._interpolator, 'values')
                    else self._interpolator._values
                )
                result = nearest(x)

        return float(result[0])

    def gradient(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute gradient via finite differences."""
        x = np.asarray(x)
        grad = np.zeros(self._n_dims)

        for i in range(self._n_dims):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps

            grad[i] = (self.evaluate(x_plus) - self.evaluate(x_minus)) / (2 * eps)

        return grad

    def __repr__(self) -> str:
        if self.is_fitted:
            return f"MultivariateSpline(method='{self.method}', n_dims={self._n_dims})"
        return f"MultivariateSpline(method='{self.method}', not fitted)"


def fit_objective_spline(
    points: np.ndarray,
    values: np.ndarray,
    method: str = 'rbf',
    **kwargs
) -> MultivariateApproximation:
    """Convenience function to fit a spline approximation to objective samples.

    Args:
        points: Sample points, shape (n_samples, n_dims).
        values: Objective values, shape (n_samples,).
        method: Interpolation method ('rbf', 'linear', 'nearest').
        **kwargs: Additional arguments passed to the approximation class.

    Returns:
        Fitted approximation.
    """
    if method == 'rbf':
        spline = ScatteredDataSpline(**kwargs)
    else:
        spline = MultivariateSpline(method=method)

    return spline.fit(points, values)
