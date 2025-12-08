"""Spline approximation for 1D functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline, splrep


@dataclass
class SplineApproximation:
    """1D spline approximation wrapping scipy's BSpline.

    This class provides a simple interface for fitting and evaluating
    univariate splines.
    """
    _spline: BSpline | None = None
    degree: int = 3  # Cubic splines by default

    @property
    def is_fitted(self) -> bool:
        """Whether the spline has been fitted."""
        return self._spline is not None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        smoothing: float | None = None
    ) -> SplineApproximation:
        """Fit a spline to the given data.

        Args:
            x: Independent variable values, shape (n,).
            y: Dependent variable values, shape (n,).
            smoothing: Smoothing factor. If None, interpolates exactly.
                      Larger values give smoother approximations.

        Returns:
            self for method chaining.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # Sort by x
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        if smoothing is None:
            # Interpolating spline
            self._spline = make_interp_spline(x, y, k=self.degree)
        else:
            # Smoothing spline
            tck = splrep(x, y, k=self.degree, s=smoothing)
            self._spline = BSpline(*tck)

        return self

    def evaluate(self, x: np.ndarray | float) -> np.ndarray | float:
        """Evaluate the spline at given point(s).

        Args:
            x: Point(s) at which to evaluate.

        Returns:
            Spline value(s) at x.
        """
        if not self.is_fitted:
            raise RuntimeError("Spline not fitted. Call fit() first.")

        x = np.asarray(x)
        scalar_input = x.ndim == 0

        result = self._spline(x)

        if scalar_input:
            return float(result)
        return result

    def derivative(self, x: np.ndarray | float, order: int = 1) -> np.ndarray | float:
        """Evaluate the derivative of the spline.

        Args:
            x: Point(s) at which to evaluate.
            order: Order of the derivative.

        Returns:
            Derivative value(s) at x.
        """
        if not self.is_fitted:
            raise RuntimeError("Spline not fitted. Call fit() first.")

        deriv_spline = self._spline.derivative(order)
        x = np.asarray(x)
        scalar_input = x.ndim == 0

        result = deriv_spline(x)

        if scalar_input:
            return float(result)
        return result

    @property
    def knots(self) -> np.ndarray:
        """Knot vector of the spline."""
        if not self.is_fitted:
            raise RuntimeError("Spline not fitted.")
        return self._spline.t

    @property
    def coefficients(self) -> np.ndarray:
        """Spline coefficients."""
        if not self.is_fitted:
            raise RuntimeError("Spline not fitted.")
        return self._spline.c

    @property
    def domain(self) -> tuple[float, float]:
        """Domain of the spline (min and max knot values)."""
        if not self.is_fitted:
            raise RuntimeError("Spline not fitted.")
        k = self._spline.k
        return (float(self._spline.t[k]), float(self._spline.t[-k-1]))

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        """Allow calling the spline directly."""
        return self.evaluate(x)

    def __repr__(self) -> str:
        if self.is_fitted:
            return f"SplineApproximation(degree={self.degree}, n_knots={len(self.knots)})"
        return f"SplineApproximation(degree={self.degree}, not fitted)"
