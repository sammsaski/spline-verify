"""B-spline approximation for 1D/2D gridded data.

This module provides tensor-product B-spline fitting for cases where
scattered samples can be interpolated to a regular grid. B-splines have
better theoretical properties for approximation than RBF interpolation
in low dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import (
    UnivariateSpline,
    RectBivariateSpline,
    LinearNDInterpolator,
    NearestNDInterpolator,
    RBFInterpolator,
)
from scipy.ndimage import map_coordinates

from .multivariate import MultivariateApproximation


@dataclass
class GriddedBSpline(MultivariateApproximation):
    """Tensor-product B-spline for 1D/2D gridded data.

    This class implements B-spline approximation by:
    1. Creating a regular grid over the domain
    2. Interpolating scattered samples to the grid
    3. Fitting scipy's UnivariateSpline (1D) or RectBivariateSpline (2D)

    B-splines are more natural for gridded data and have better theoretical
    properties for approximation than RBF interpolation.

    Attributes:
        n_grid_points: Number of grid points per dimension.
        degree: Spline degree (1=linear, 3=cubic, 5=quintic).
        smoothing: Smoothing parameter (0 = interpolation, >0 = smoothing).
        grid_method: Method for interpolating scattered data to grid.
    """
    n_grid_points: int = 50
    degree: int = 3
    smoothing: float = 0.0
    grid_method: Literal['linear', 'nearest', 'rbf'] = 'linear'

    # Private attributes
    _n_dims: int = field(default=0, repr=False)
    _spline_1d: UnivariateSpline | None = field(default=None, repr=False)
    _spline_2d: RectBivariateSpline | None = field(default=None, repr=False)
    _grid_axes: list[np.ndarray] | None = field(default=None, repr=False)
    _grid_values: np.ndarray | None = field(default=None, repr=False)
    _bounds: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    _original_points: np.ndarray | None = field(default=None, repr=False)
    _original_values: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def is_fitted(self) -> bool:
        return self._spline_1d is not None or self._spline_2d is not None

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return domain bounds."""
        return self._bounds

    def fit(self, points: np.ndarray, values: np.ndarray) -> GriddedBSpline:
        """Fit B-spline approximation to scattered data.

        Args:
            points: Sample points, shape (n_samples, n_dims). n_dims must be 1 or 2.
            values: Function values, shape (n_samples,).

        Returns:
            self for method chaining.

        Raises:
            ValueError: If n_dims > 2.
        """
        points = np.asarray(points)
        values = np.asarray(values)

        if points.ndim == 1:
            points = points.reshape(-1, 1)

        n_dims = points.shape[1]
        if n_dims > 2:
            raise ValueError(
                f"GriddedBSpline only supports 1D and 2D (got {n_dims}D). "
                "Use ScatteredDataSpline for higher dimensions."
            )

        self._n_dims = n_dims
        self._original_points = points.copy()
        self._original_values = values.copy()

        # Compute bounds
        self._bounds = (points.min(axis=0), points.max(axis=0))

        # Create regular grid
        self._create_grid()

        # Interpolate scattered data to grid
        self._interpolate_to_grid(points, values)

        # Fit B-spline on gridded data
        if n_dims == 1:
            self._fit_1d()
        else:
            self._fit_2d()

        return self

    def _create_grid(self) -> None:
        """Create regular grid axes over the domain."""
        lower, upper = self._bounds
        self._grid_axes = []

        for d in range(self._n_dims):
            # Add small padding to avoid boundary issues
            margin = 0.01 * (upper[d] - lower[d])
            axis = np.linspace(
                lower[d] - margin,
                upper[d] + margin,
                self.n_grid_points
            )
            self._grid_axes.append(axis)

    def _interpolate_to_grid(
        self,
        points: np.ndarray,
        values: np.ndarray
    ) -> None:
        """Interpolate scattered data to regular grid."""
        if self._n_dims == 1:
            # 1D: create grid and interpolate
            x_grid = self._grid_axes[0]

            if self.grid_method == 'linear':
                # Sort by x and interpolate
                sort_idx = np.argsort(points[:, 0])
                x_sorted = points[sort_idx, 0]
                v_sorted = values[sort_idx]
                self._grid_values = np.interp(x_grid, x_sorted, v_sorted)

            elif self.grid_method == 'nearest':
                interp = NearestNDInterpolator(points, values)
                self._grid_values = interp(x_grid.reshape(-1, 1))

            elif self.grid_method == 'rbf':
                interp = RBFInterpolator(points, values, kernel='thin_plate_spline')
                self._grid_values = interp(x_grid.reshape(-1, 1))

        else:
            # 2D: create meshgrid and interpolate
            x_grid, y_grid = self._grid_axes
            xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])

            if self.grid_method == 'linear':
                interp = LinearNDInterpolator(points, values)
                grid_vals = interp(grid_points)

                # Handle NaN from extrapolation with nearest neighbor
                nan_mask = np.isnan(grid_vals)
                if nan_mask.any():
                    nearest = NearestNDInterpolator(points, values)
                    grid_vals[nan_mask] = nearest(grid_points[nan_mask])

            elif self.grid_method == 'nearest':
                interp = NearestNDInterpolator(points, values)
                grid_vals = interp(grid_points)

            elif self.grid_method == 'rbf':
                interp = RBFInterpolator(points, values, kernel='thin_plate_spline')
                grid_vals = interp(grid_points)

            self._grid_values = grid_vals.reshape(
                len(x_grid), len(y_grid)
            )

    def _fit_1d(self) -> None:
        """Fit 1D B-spline to gridded data."""
        x = self._grid_axes[0]
        y = self._grid_values

        # Determine smoothing factor
        # s=0 means interpolation, s>0 means smoothing
        if self.smoothing == 0:
            # For interpolation, use splrep with s=0
            s = 0
        else:
            # Smoothing factor scales with number of points
            s = self.smoothing * len(x)

        # Fit spline
        self._spline_1d = UnivariateSpline(
            x, y,
            k=min(self.degree, 5),  # scipy max is 5
            s=s,
            ext=0  # Extrapolate
        )

    def _fit_2d(self) -> None:
        """Fit 2D tensor-product B-spline to gridded data."""
        x = self._grid_axes[0]
        y = self._grid_axes[1]
        z = self._grid_values

        # RectBivariateSpline uses smoothing parameter s
        if self.smoothing == 0:
            s = 0
        else:
            s = self.smoothing * z.size

        # Fit spline (kx, ky are degrees in each direction)
        k = min(self.degree, 5)
        self._spline_2d = RectBivariateSpline(
            x, y, z,
            kx=k, ky=k,
            s=s
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the B-spline approximation at a point.

        Args:
            x: Point, shape (n_dims,).

        Returns:
            Approximated function value.
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        x = np.asarray(x).flatten()

        if self._n_dims == 1:
            result = self._spline_1d(x[0])
            return float(result)
        else:
            # RectBivariateSpline expects x, y as 1D arrays
            # With grid=False, it returns a scalar or 1D array
            result = self._spline_2d(x[0], x[1], grid=False)
            # Handle both scalar and array returns
            result = np.asarray(result)
            if result.ndim == 0:
                return float(result)
            return float(result.flat[0])

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

        if self._n_dims == 1:
            return self._spline_1d(x[:, 0])
        else:
            return self._spline_2d(x[:, 0], x[:, 1], grid=False)

    def gradient(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute gradient of the B-spline approximation.

        For 1D, uses the spline derivative directly.
        For 2D, uses the spline partial derivatives directly.

        Args:
            x: Point, shape (n_dims,).
            eps: Not used (analytical derivatives available).

        Returns:
            Gradient vector, shape (n_dims,).
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        x = np.asarray(x).flatten()

        if self._n_dims == 1:
            # UnivariateSpline has derivative() method
            deriv = self._spline_1d.derivative()
            return np.array([float(deriv(x[0]))])
        else:
            # RectBivariateSpline partial derivatives
            dx = self._spline_2d(x[0], x[1], dx=1, grid=False)
            dy = self._spline_2d(x[0], x[1], dy=1, grid=False)
            # Handle scalar results
            dx = np.asarray(dx)
            dy = np.asarray(dy)
            dx_val = float(dx) if dx.ndim == 0 else float(dx.flat[0])
            dy_val = float(dy) if dy.ndim == 0 else float(dy.flat[0])
            return np.array([dx_val, dy_val])

    def residuals(self) -> np.ndarray:
        """Compute residuals at original training points."""
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")

        predicted = self.evaluate_batch(self._original_points)
        return self._original_values - predicted

    def get_knots(self) -> list[np.ndarray]:
        """Return the knot vectors of the fitted spline.

        Returns:
            List of knot arrays, one per dimension.
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")

        if self._n_dims == 1:
            return [self._spline_1d.get_knots()]
        else:
            return [self._spline_2d.get_knots()[0], self._spline_2d.get_knots()[1]]

    def get_coefficients(self) -> np.ndarray:
        """Return the B-spline coefficients.

        Returns:
            Coefficient array.
        """
        if not self.is_fitted:
            raise RuntimeError("Not fitted.")

        if self._n_dims == 1:
            return self._spline_1d.get_coeffs()
        else:
            return self._spline_2d.get_coeffs()

    def __repr__(self) -> str:
        if self.is_fitted:
            return (
                f"GriddedBSpline(n_grid={self.n_grid_points}, degree={self.degree}, "
                f"n_dims={self._n_dims}, grid_method='{self.grid_method}')"
            )
        return (
            f"GriddedBSpline(n_grid={self.n_grid_points}, degree={self.degree}, "
            f"grid_method='{self.grid_method}', not fitted)"
        )


def create_spline_approximation(
    points: np.ndarray,
    values: np.ndarray,
    method: str = 'rbf',
    **kwargs
) -> MultivariateApproximation:
    """Factory function to create appropriate spline approximation.

    Automatically selects B-spline for 1D/2D data if requested,
    or falls back to RBF for higher dimensions.

    Args:
        points: Sample points, shape (n_samples, n_dims).
        values: Function values, shape (n_samples,).
        method: Approximation method ('rbf', 'bspline', or 'auto').
            - 'rbf': Always use RBF interpolation (ScatteredDataSpline)
            - 'bspline': Use GriddedBSpline for 1D/2D, fall back to RBF otherwise
            - 'auto': Use bspline for 1D/2D, rbf otherwise
        **kwargs: Additional arguments passed to the approximation class.

    Returns:
        Fitted MultivariateApproximation.
    """
    from .multivariate import ScatteredDataSpline

    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    n_dims = points.shape[1]

    if method == 'rbf':
        spline = ScatteredDataSpline(
            kernel=kwargs.get('kernel', 'thin_plate_spline'),
            smoothing=kwargs.get('smoothing', 0.0)
        )
    elif method in ('bspline', 'auto'):
        if n_dims <= 2:
            spline = GriddedBSpline(
                n_grid_points=kwargs.get('n_grid_points', 50),
                degree=kwargs.get('degree', 3),
                smoothing=kwargs.get('smoothing', 0.0),
                grid_method=kwargs.get('grid_method', 'linear')
            )
        else:
            if method == 'bspline':
                import warnings
                warnings.warn(
                    f"B-spline only supports 1D/2D (got {n_dims}D). "
                    "Falling back to RBF interpolation."
                )
            spline = ScatteredDataSpline(
                kernel=kwargs.get('kernel', 'thin_plate_spline'),
                smoothing=kwargs.get('smoothing', 0.0)
            )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rbf', 'bspline', or 'auto'.")

    return spline.fit(points, values)
