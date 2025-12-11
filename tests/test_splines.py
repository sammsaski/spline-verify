"""Tests for spline approximation and optimization."""

import numpy as np
import pytest

from spline_verify.splines.approximation import SplineApproximation
from spline_verify.splines.multivariate import ScatteredDataSpline, MultivariateSpline
from spline_verify.splines.bspline import GriddedBSpline, create_spline_approximation
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


class TestGriddedBSpline:
    """Tests for B-spline approximation on gridded data."""

    def test_1d_interpolation(self):
        """Test 1D B-spline interpolation."""
        # Sample a quadratic: f(x) = x^2
        np.random.seed(42)
        points = np.random.uniform(0, 2, size=(30, 1))
        values = points[:, 0]**2

        spline = GriddedBSpline(n_grid_points=40, degree=3)
        spline.fit(points, values)

        # Test evaluation
        test_x = np.array([0.5])
        expected = 0.25
        assert abs(spline.evaluate(test_x) - expected) < 0.1

        # Test at multiple points (relax tolerance at boundary due to extrapolation)
        test_points = np.array([[0.0], [1.0], [2.0]])
        tolerances = [0.2, 0.2, 0.3]  # Larger tolerance at boundary
        for p, e, tol in zip(test_points, [0.0, 1.0, 4.0], tolerances):
            assert abs(spline.evaluate(p) - e) < tol

    def test_1d_gradient(self):
        """Test 1D B-spline gradient (derivative)."""
        # f(x) = x^2, f'(x) = 2x
        np.random.seed(42)
        points = np.random.uniform(0, 2, size=(40, 1))
        values = points[:, 0]**2

        spline = GriddedBSpline(n_grid_points=50, degree=3)
        spline.fit(points, values)

        # Gradient at x=1 should be approximately 2
        grad = spline.gradient(np.array([1.0]))
        assert abs(grad[0] - 2.0) < 0.3

    def test_2d_interpolation(self):
        """Test 2D B-spline interpolation."""
        # f(x,y) = x^2 + y^2
        np.random.seed(42)
        points = np.random.uniform(-1, 1, size=(100, 2))
        values = points[:, 0]**2 + points[:, 1]**2

        spline = GriddedBSpline(n_grid_points=30, degree=3)
        spline.fit(points, values)

        # Test at center (should be near 0)
        center = np.array([0.0, 0.0])
        assert abs(spline.evaluate(center)) < 0.1

        # Test at (0.5, 0.5) - should be near 0.5
        test_point = np.array([0.5, 0.5])
        expected = 0.5
        assert abs(spline.evaluate(test_point) - expected) < 0.15

    def test_2d_gradient(self):
        """Test 2D B-spline gradient (partial derivatives)."""
        # f(x,y) = x^2 + y^2
        # grad f = (2x, 2y)
        np.random.seed(42)
        points = np.random.uniform(-1, 1, size=(150, 2))
        values = points[:, 0]**2 + points[:, 1]**2

        spline = GriddedBSpline(n_grid_points=40, degree=3)
        spline.fit(points, values)

        # Gradient at (0.5, 0.3) should be approximately (1.0, 0.6)
        grad = spline.gradient(np.array([0.5, 0.3]))
        assert abs(grad[0] - 1.0) < 0.3
        assert abs(grad[1] - 0.6) < 0.3

    def test_different_grid_methods(self):
        """Test different methods for grid interpolation."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(50, 2))
        values = points[:, 0] + points[:, 1]

        test_point = np.array([0.5, 0.5])
        expected = 1.0

        for method in ['linear', 'nearest', 'rbf']:
            spline = GriddedBSpline(n_grid_points=25, grid_method=method)
            spline.fit(points, values)
            result = spline.evaluate(test_point)
            assert abs(result - expected) < 0.2, f"Failed for grid_method={method}"

    def test_smoothing(self):
        """Test B-spline smoothing."""
        np.random.seed(42)
        points = np.random.uniform(0, 2, size=(50, 1))
        # Add noise
        true_values = points[:, 0]**2
        noisy_values = true_values + np.random.normal(0, 0.2, len(points))

        # With smoothing
        spline = GriddedBSpline(n_grid_points=30, smoothing=0.1)
        spline.fit(points, noisy_values)

        # Should still be close to true function
        test_x = np.array([1.0])
        assert abs(spline.evaluate(test_x) - 1.0) < 0.3

    def test_batch_evaluation(self):
        """Test batch evaluation."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(50, 2))
        values = points[:, 0] + points[:, 1]

        spline = GriddedBSpline(n_grid_points=25)
        spline.fit(points, values)

        # Batch evaluate
        test_points = np.array([[0.2, 0.3], [0.5, 0.5], [0.7, 0.8]])
        results = spline.evaluate_batch(test_points)

        expected = test_points[:, 0] + test_points[:, 1]
        for i in range(len(test_points)):
            assert abs(results[i] - expected[i]) < 0.2

    def test_residuals(self):
        """Test residual computation."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(40, 2))
        values = points[:, 0]**2 + points[:, 1]**2

        spline = GriddedBSpline(n_grid_points=30)
        spline.fit(points, values)

        residuals = spline.residuals()
        # Residuals should be small for smooth function
        assert np.max(np.abs(residuals)) < 0.3

    def test_dimension_limit(self):
        """Test that 3D+ raises ValueError."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(50, 3))  # 3D
        values = np.sum(points, axis=1)

        spline = GriddedBSpline()
        with pytest.raises(ValueError, match="only supports 1D and 2D"):
            spline.fit(points, values)

    def test_knots_and_coefficients(self):
        """Test getting knots and coefficients."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(30, 1))
        values = points[:, 0]**2

        spline = GriddedBSpline(n_grid_points=20)
        spline.fit(points, values)

        knots = spline.get_knots()
        coeffs = spline.get_coefficients()

        assert len(knots) == 1  # 1D
        assert len(knots[0]) > 0
        assert len(coeffs) > 0


class TestCreateSplineApproximation:
    """Tests for the factory function."""

    def test_auto_selects_bspline_for_1d(self):
        """Test that 'auto' selects B-spline for 1D."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(30, 1))
        values = points[:, 0]**2

        spline = create_spline_approximation(points, values, method='auto')
        assert isinstance(spline, GriddedBSpline)

    def test_auto_selects_bspline_for_2d(self):
        """Test that 'auto' selects B-spline for 2D."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(50, 2))
        values = points[:, 0] + points[:, 1]

        spline = create_spline_approximation(points, values, method='auto')
        assert isinstance(spline, GriddedBSpline)

    def test_auto_selects_rbf_for_3d(self):
        """Test that 'auto' selects RBF for 3D+."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(50, 3))
        values = np.sum(points, axis=1)

        spline = create_spline_approximation(points, values, method='auto')
        assert isinstance(spline, ScatteredDataSpline)

    def test_bspline_fallback_for_3d(self):
        """Test that 'bspline' falls back to RBF for 3D+ with warning."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(50, 3))
        values = np.sum(points, axis=1)

        with pytest.warns(UserWarning, match="only supports 1D/2D"):
            spline = create_spline_approximation(points, values, method='bspline')
        assert isinstance(spline, ScatteredDataSpline)

    def test_rbf_always_uses_rbf(self):
        """Test that 'rbf' always uses RBF regardless of dimension."""
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(30, 1))
        values = points[:, 0]**2

        spline = create_spline_approximation(points, values, method='rbf')
        assert isinstance(spline, ScatteredDataSpline)


class TestBSplineMinimization:
    """Tests for minimizing B-spline approximations."""

    def test_minimize_1d_bspline(self):
        """Test minimization of 1D B-spline."""
        # f(x) = (x - 0.5)^2, minimum at x=0.5
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(40, 1))
        values = (points[:, 0] - 0.5)**2

        spline = GriddedBSpline(n_grid_points=50)
        spline.fit(points, values)

        bounds = (np.array([0]), np.array([1]))
        result = minimize_spline(spline, bounds, method='multistart', n_starts=10, seed=42)

        assert result.success
        assert result.minimum < 0.05
        assert abs(result.minimizer[0] - 0.5) < 0.1

    def test_minimize_2d_bspline(self):
        """Test minimization of 2D B-spline."""
        # f(x,y) = (x - 0.3)^2 + (y - 0.7)^2, minimum at (0.3, 0.7)
        np.random.seed(42)
        points = np.random.uniform(0, 1, size=(100, 2))
        values = (points[:, 0] - 0.3)**2 + (points[:, 1] - 0.7)**2

        spline = GriddedBSpline(n_grid_points=40)
        spline.fit(points, values)

        bounds = (np.array([0, 0]), np.array([1, 1]))
        result = minimize_spline(spline, bounds, method='multistart', n_starts=15, seed=42)

        assert result.success
        assert result.minimum < 0.1
        assert abs(result.minimizer[0] - 0.3) < 0.2
        assert abs(result.minimizer[1] - 0.7) < 0.2
