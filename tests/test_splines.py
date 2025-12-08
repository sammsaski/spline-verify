"""Tests for spline approximation and optimization."""

import numpy as np
import pytest

from spline_verify.splines.approximation import SplineApproximation
from spline_verify.splines.multivariate import ScatteredDataSpline, MultivariateSpline
from spline_verify.splines.optimization import minimize_spline


class TestSplineApproximation:
    """Tests for 1D spline approximation."""

    def test_interpolation(self):
        """Test exact interpolation."""
        x = np.array([0, 1, 2, 3, 4])
        y = np.array([0, 1, 4, 9, 16])  # y = x^2

        spline = SplineApproximation(degree=3)
        spline.fit(x, y)

        # Check interpolation at data points
        for xi, yi in zip(x, y):
            assert abs(spline.evaluate(xi) - yi) < 1e-10

    def test_smoothing(self):
        """Test smoothing spline."""
        np.random.seed(42)
        x = np.linspace(0, 4, 20)
        y = x**2 + np.random.normal(0, 0.5, len(x))

        spline = SplineApproximation(degree=3)
        spline.fit(x, y, smoothing=1.0)

        # Smoothed values should be close but not exact
        residuals = np.array([spline.evaluate(xi) - yi for xi, yi in zip(x, y)])
        assert np.std(residuals) < 1.0  # Residuals bounded

    def test_derivative(self):
        """Test derivative computation."""
        x = np.linspace(0, 2*np.pi, 20)
        y = np.sin(x)

        spline = SplineApproximation(degree=3)
        spline.fit(x, y)

        # Derivative of sin is cos
        test_x = np.array([0, np.pi/2, np.pi])
        expected = np.cos(test_x)

        for xi, expected_deriv in zip(test_x, expected):
            actual_deriv = spline.derivative(xi)
            assert abs(actual_deriv - expected_deriv) < 0.1


class TestScatteredDataSpline:
    """Tests for scattered data interpolation."""

    def test_2d_interpolation(self):
        """Test 2D RBF interpolation."""
        # Sample a simple function: f(x,y) = x + y
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(50, 2))
        values = points[:, 0] + points[:, 1]

        spline = ScatteredDataSpline(kernel='thin_plate_spline')
        spline.fit(points, values)

        # Test on new points
        test_points = np.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])
        expected = test_points[:, 0] + test_points[:, 1]

        for p, e in zip(test_points, expected):
            assert abs(spline.evaluate(p) - e) < 0.1

    def test_gradient(self):
        """Test gradient computation."""
        # f(x,y) = x^2 + y^2
        np.random.seed(42)
        points = np.random.uniform(-1, 1, size=(100, 2))
        values = points[:, 0]**2 + points[:, 1]**2

        spline = ScatteredDataSpline()
        spline.fit(points, values)

        # Gradient at (0.5, 0.5) should be approximately (1, 1)
        grad = spline.gradient(np.array([0.5, 0.5]))

        assert abs(grad[0] - 1.0) < 0.3
        assert abs(grad[1] - 1.0) < 0.3


class TestMultivariateSpline:
    """Tests for multivariate spline."""

    def test_linear_interpolation(self):
        """Test linear interpolation."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(50, 2))
        values = points[:, 0] + points[:, 1]

        spline = MultivariateSpline(method='linear')
        spline.fit(points, values)

        # Test at a point inside convex hull
        test_point = np.array([0.5, 0.5])
        expected = 1.0

        assert abs(spline.evaluate(test_point) - expected) < 0.2

    def test_nearest_neighbor(self):
        """Test nearest neighbor interpolation."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        values = np.array([0, 1, 2, 3])

        spline = MultivariateSpline(method='nearest')
        spline.fit(points, values)

        # Point closest to (0, 0) should give value 0
        assert spline.evaluate(np.array([0.1, 0.1])) == 0


class TestMinimization:
    """Tests for spline minimization."""

    def test_minimize_quadratic(self):
        """Test minimization of a quadratic function."""
        # f(x,y) = (x-0.5)^2 + (y-0.5)^2
        # Minimum at (0.5, 0.5) with value 0
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(100, 2))
        values = (points[:, 0] - 0.5)**2 + (points[:, 1] - 0.5)**2

        spline = ScatteredDataSpline()
        spline.fit(points, values)

        bounds = (np.array([0, 0]), np.array([1, 1]))
        result = minimize_spline(spline, bounds, method='multistart', n_starts=10, seed=42)

        assert result.success
        assert result.minimum < 0.1
        assert abs(result.minimizer[0] - 0.5) < 0.2
        assert abs(result.minimizer[1] - 0.5) < 0.2

    def test_minimize_with_boundary_minimum(self):
        """Test when minimum is on boundary."""
        # f(x,y) = x + y - minimum at (0, 0)
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(100, 2))
        values = points[:, 0] + points[:, 1]

        spline = ScatteredDataSpline()
        spline.fit(points, values)

        bounds = (np.array([0, 0]), np.array([1, 1]))
        result = minimize_spline(spline, bounds, method='differential_evolution', seed=42)

        # Minimum should be near (0, 0)
        assert result.minimum < 0.1
        assert result.minimizer[0] < 0.1
        assert result.minimizer[1] < 0.1
