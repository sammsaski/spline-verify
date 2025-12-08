"""Tests for numerical integrators."""

import numpy as np
import pytest

from spline_verify.dynamics.integrators import integrate, IntegrationMethod
from spline_verify.dynamics.trajectory import Trajectory


class TestTrajectory:
    """Tests for Trajectory class."""

    def test_basic_creation(self):
        """Test basic trajectory creation."""
        times = np.array([0, 1, 2, 3])
        states = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

        traj = Trajectory(times, states)

        assert traj.n_points == 4
        assert traj.n_dims == 2
        assert traj.t_start == 0
        assert traj.t_end == 3

    def test_1d_trajectory(self):
        """Test 1D trajectory."""
        times = np.array([0, 1, 2])
        states = np.array([0, 1, 2])

        traj = Trajectory(times, states)

        assert traj.n_dims == 1
        assert traj.states.shape == (3, 1)

    def test_interpolation(self):
        """Test trajectory interpolation."""
        times = np.array([0, 1, 2])
        states = np.array([[0, 0], [1, 1], [2, 2]])

        traj = Trajectory(times, states)

        # Interpolate at midpoint
        state = traj.interpolate(0.5)
        np.testing.assert_array_almost_equal(state, [0.5, 0.5])

    def test_min_distance_to_point(self):
        """Test minimum distance computation."""
        times = np.array([0, 1, 2])
        states = np.array([[0, 0], [1, 0], [2, 0]])

        traj = Trajectory(times, states)

        # Point at (1, 1) - closest to (1, 0)
        dist = traj.min_distance_to_point(np.array([1, 1]))
        assert abs(dist - 1.0) < 1e-10


class TestIntegrators:
    """Tests for numerical integrators."""

    def test_euler_linear(self):
        """Test Euler on linear system."""
        # dx/dt = -x has solution x(t) = x0 * exp(-t)
        def f(t, x):
            return -x

        x0 = np.array([1.0])
        traj = integrate(f, x0, (0, 1), IntegrationMethod.EULER, step_size=0.001)

        # Check final value
        expected = np.exp(-1)
        actual = traj.final_state[0]

        # Euler is first-order, so error should be O(h)
        assert abs(actual - expected) < 0.01

    def test_rk4_harmonic(self):
        """Test RK4 on harmonic oscillator."""
        # dx/dt = y, dy/dt = -x
        # Solution: x = cos(t), y = -sin(t) starting from (1, 0)
        def f(t, x):
            return np.array([x[1], -x[0]])

        x0 = np.array([1.0, 0.0])
        T = 2 * np.pi  # One period

        traj = integrate(f, x0, (0, T), IntegrationMethod.RK4, step_size=0.01)

        # Should return close to initial state after one period
        np.testing.assert_array_almost_equal(
            traj.final_state, x0, decimal=3
        )

    def test_rk45_conservation(self):
        """Test RK45 energy conservation."""
        def f(t, x):
            return np.array([x[1], -x[0]])

        x0 = np.array([1.0, 0.0])
        T = 4 * np.pi

        traj = integrate(f, x0, (0, T), IntegrationMethod.RK45)

        # Energy: E = x^2 + y^2 should be conserved
        energies = traj.states[:, 0]**2 + traj.states[:, 1]**2
        assert np.std(energies) < 1e-5

    def test_stiff_system(self):
        """Test on mildly stiff system."""
        # dx/dt = -100*x
        def f(t, x):
            return -100 * x

        x0 = np.array([1.0])
        traj = integrate(f, x0, (0, 0.1), IntegrationMethod.RK45)

        expected = np.exp(-10)
        assert abs(traj.final_state[0] - expected) < 1e-5


class TestIntegrationMethods:
    """Compare different integration methods."""

    @pytest.fixture
    def simple_ode(self):
        """Simple test ODE."""
        def f(t, x):
            return np.array([x[1], -x[0]])
        return f

    def test_all_methods_agree(self, simple_ode):
        """All methods should give similar results on simple problem."""
        x0 = np.array([1.0, 0.0])
        T = 1.0

        results = {}
        for method in [IntegrationMethod.EULER, IntegrationMethod.RK4,
                       IntegrationMethod.RK45]:
            traj = integrate(simple_ode, x0, (0, T), method, step_size=0.001)
            results[method] = traj.final_state

        # All should be close to analytical solution
        expected = np.array([np.cos(1), -np.sin(1)])

        for method, final in results.items():
            np.testing.assert_array_almost_equal(
                final, expected, decimal=2,
                err_msg=f"{method.name} gave wrong result"
            )
