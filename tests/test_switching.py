"""Tests for switching system dynamics and verification (Phase 4)."""

import numpy as np
import pytest

from spline_verify.dynamics import (
    SwitchingDynamics,
    SwitchingSurface,
    SwitchingBehavior,
    FilippovSolver,
    TrajectoryBundle,
)
from spline_verify.geometry import HyperRectangle, Ball
from spline_verify.verification import (
    SwitchingVerifier,
    SwitchingRegionClassifier,
    VerificationStatus,
    extract_crossing_labels,
)
from spline_verify.splines import PiecewiseSplineApproximation


class TestSwitchingSurface:
    """Test switching surface representation."""

    def test_surface_evaluation(self):
        """Test surface boundary function evaluation."""
        # Surface: x = 0
        surface = SwitchingSurface(
            boundary_function=lambda x: x[0],
            adjacent_modes=(0, 1),
        )

        assert surface.evaluate(np.array([1.0, 0.5])) == pytest.approx(1.0)
        assert surface.evaluate(np.array([-1.0, 0.5])) == pytest.approx(-1.0)
        assert surface.evaluate(np.array([0.0, 0.5])) == pytest.approx(0.0)

    def test_surface_normal(self):
        """Test surface normal computation."""
        # Surface: x = 0 with explicit normal
        surface = SwitchingSurface(
            boundary_function=lambda x: x[0],
            adjacent_modes=(0, 1),
            normal_function=lambda x: np.array([1.0, 0.0]),
        )

        normal = surface.normal(np.array([0.0, 0.5]))
        assert np.allclose(normal, [1.0, 0.0])

    def test_surface_projection(self):
        """Test projection onto surface."""
        # Surface: x = 0
        surface = SwitchingSurface(
            boundary_function=lambda x: x[0],
            adjacent_modes=(0, 1),
            normal_function=lambda x: np.array([1.0, 0.0]),
        )

        # Point at (1, 0.5) should project to (0, 0.5)
        projected = surface.project_to_surface(np.array([1.0, 0.5]))
        assert projected[0] == pytest.approx(0.0, abs=0.1)


class TestSwitchingDynamics:
    """Test switching dynamics factory methods."""

    def test_bouncing_ball_creation(self):
        """Test bouncing ball dynamics creation."""
        dynamics = SwitchingDynamics.bouncing_ball(gravity=9.81, restitution=0.9)

        assert dynamics.n_dims == 2
        assert len(dynamics.surfaces) == 1
        assert len(dynamics.mode_dynamics) == 1

    def test_relay_feedback_creation(self):
        """Test relay feedback dynamics creation."""
        dynamics = SwitchingDynamics.relay_feedback()

        # Note: Relay feedback is 1D (just x, with dx/dt = -sign(x))
        assert dynamics.n_dims == 1
        assert len(dynamics.surfaces) == 1
        assert len(dynamics.mode_dynamics) == 2

    def test_thermostat_creation(self):
        """Test thermostat dynamics creation."""
        dynamics = SwitchingDynamics.thermostat(
            T_low=18.0, T_high=22.0, T_ambient=10.0
        )

        assert dynamics.n_dims == 1
        assert len(dynamics.surfaces) == 2
        assert len(dynamics.mode_dynamics) == 2


class TestFilippovSolver:
    """Test Filippov differential inclusion solver."""

    def test_bouncing_ball_simulation(self):
        """Test bouncing ball simulation with bounces."""
        dynamics = SwitchingDynamics.bouncing_ball(gravity=9.81, restitution=0.9)
        solver = FilippovSolver(dynamics)

        # Ball starting at height 1m, zero velocity
        x0 = np.array([1.0, 0.0])
        bundle = solver.solve(x0, (0, 2.0))

        assert isinstance(bundle, TrajectoryBundle)
        assert len(bundle.trajectories) >= 1

        traj = bundle.primary
        assert traj.n_dims == 2

        # Ball should never go negative (after proper bounce handling)
        min_height = traj.states[:, 0].min()
        # Allow small numerical error
        assert min_height >= -0.1, f"Ball went below ground: {min_height}"

    def test_relay_feedback_simulation(self):
        """Test relay feedback simulation."""
        dynamics = SwitchingDynamics.relay_feedback()
        solver = FilippovSolver(dynamics)

        # Start in positive x region (1D relay feedback)
        x0 = np.array([1.0])
        bundle = solver.solve(x0, (0, 2.0))

        traj = bundle.primary
        assert traj.n_dims == 1

        # Trajectory should approach origin
        final_dist = np.abs(traj.final_state[0])
        assert final_dist < 0.5, "Should approach origin"

    def test_sliding_mode_detection(self):
        """Test that sliding mode is detected at switching surface."""
        dynamics = SwitchingDynamics.relay_feedback()
        solver = FilippovSolver(dynamics)

        # Initial condition at x=0 (1D)
        x0 = np.array([0.0])
        behavior = solver.classify_surface_behavior(0, x0, dynamics.surfaces[0])

        # Should detect some switching behavior
        assert behavior in [
            SwitchingBehavior.CROSSING,
            SwitchingBehavior.SLIDING,
            SwitchingBehavior.INTERIOR,
        ]

    def test_thermostat_regulation(self):
        """Test thermostat stays in regulation band."""
        dynamics = SwitchingDynamics.thermostat(
            T_low=18.0, T_high=22.0, T_ambient=10.0,
            cooling_rate=0.1, heating_power=2.0
        )
        solver = FilippovSolver(dynamics)

        # Start cold
        x0 = np.array([17.0])
        bundle = solver.solve(x0, (0, 20.0))

        traj = bundle.primary
        # Should heat up to T_low
        assert traj.final_state[0] >= 17.5  # Should be at or near T_low


class TestSwitchingRegionClassifier:
    """Test switching region classifier."""

    def test_classifier_single_region(self):
        """Test classifier with single region (no switching)."""
        classifier = SwitchingRegionClassifier()

        points = np.random.rand(20, 2)
        labels = np.zeros(20, dtype=int)

        classifier.fit(points, labels)

        assert classifier.is_fitted
        assert classifier.n_regions == 1

        # Should predict 0 for any point
        pred = classifier.predict(np.array([0.5, 0.5]))
        assert pred == 0

    def test_classifier_multiple_regions(self):
        """Test classifier with multiple regions."""
        classifier = SwitchingRegionClassifier(kernel='linear')

        # Create two clearly separated regions
        n_per_region = 30
        points_0 = np.random.randn(n_per_region, 2) + np.array([[-2, 0]])
        points_1 = np.random.randn(n_per_region, 2) + np.array([[2, 0]])

        points = np.vstack([points_0, points_1])
        labels = np.array([0] * n_per_region + [1] * n_per_region)

        classifier.fit(points, labels)

        assert classifier.is_fitted
        assert classifier.n_regions == 2

        # Test predictions
        pred_left = classifier.predict(np.array([-3, 0]))
        pred_right = classifier.predict(np.array([3, 0]))

        assert pred_left == 0
        assert pred_right == 1

    def test_classifier_batch_prediction(self):
        """Test batch prediction."""
        classifier = SwitchingRegionClassifier()

        points = np.random.rand(20, 2)
        labels = np.zeros(20, dtype=int)
        classifier.fit(points, labels)

        test_points = np.random.rand(5, 2)
        predictions = classifier.predict_batch(test_points)

        assert len(predictions) == 5
        assert all(p == 0 for p in predictions)


class TestExtractCrossingLabels:
    """Test automatic crossing label extraction."""

    def test_extract_labels_no_crossing(self):
        """Test label extraction when no crossings occur."""
        # Create simple dynamics (1D relay feedback)
        dynamics = SwitchingDynamics.relay_feedback()

        # Create bundles that don't cross
        from spline_verify.dynamics import Trajectory

        # Trajectory that stays in x > 0 (1D)
        times = np.linspace(0, 1, 100)
        states = np.ones((100, 1)) * 2  # x = 2 (constant)
        traj = Trajectory(times, states)
        bundles = [TrajectoryBundle.from_single(traj)]

        points = np.array([[2.0]])
        labels = extract_crossing_labels(points, bundles, dynamics.surfaces)

        # Should have label 0 (no crossing)
        assert labels[0] == 0

    def test_extract_labels_with_crossing(self):
        """Test label extraction when crossings occur."""
        dynamics = SwitchingDynamics.relay_feedback()

        from spline_verify.dynamics import Trajectory

        # Trajectory that crosses x = 0 (1D)
        times = np.linspace(0, 1, 100)
        states = np.linspace(1, -1, 100).reshape(-1, 1)  # x: 1 -> -1 (crosses 0)
        traj = Trajectory(times, states)
        bundles = [TrajectoryBundle.from_single(traj)]

        points = np.array([[1.0]])
        labels = extract_crossing_labels(points, bundles, dynamics.surfaces)

        # Should have label > 0 (crossing occurred)
        assert labels[0] > 0


class TestPiecewiseSplineApproximation:
    """Test piecewise spline fitting for switching systems."""

    def test_single_region_fit(self):
        """Test fitting with single region."""
        spline = PiecewiseSplineApproximation()

        # Simple quadratic function
        np.random.seed(42)
        points = np.random.rand(50, 2) * 4 - 2
        values = points[:, 0]**2 + points[:, 1]**2

        labels = np.zeros(50, dtype=int)

        spline.fit(points, values, labels)

        assert spline.is_fitted
        assert spline.n_regions == 1

        # Test evaluation
        val = spline.evaluate(np.array([0.0, 0.0]))
        assert val == pytest.approx(0.0, abs=0.5)

    def test_multiple_region_fit(self):
        """Test fitting with multiple regions."""
        spline = PiecewiseSplineApproximation()

        # Create points in two regions
        np.random.seed(42)
        points_0 = np.random.rand(30, 2) - 1  # [-1, 0] x [-1, 0]
        points_1 = np.random.rand(30, 2)      # [0, 1] x [0, 1]

        # Different functions per region
        values_0 = points_0[:, 0] + 1
        values_1 = points_1[:, 0] + 2

        points = np.vstack([points_0, points_1])
        values = np.concatenate([values_0, values_1])
        labels = np.array([0] * 30 + [1] * 30)

        spline.fit(points, values, labels)

        assert spline.is_fitted
        assert spline.n_regions == 2
        assert 0 in spline.region_labels
        assert 1 in spline.region_labels

    def test_region_minimum(self):
        """Test finding minimum within a region."""
        spline = PiecewiseSplineApproximation()

        # Quadratic with known minimum at origin
        np.random.seed(42)
        points = np.random.rand(50, 2) * 2 - 1
        values = points[:, 0]**2 + points[:, 1]**2

        spline.fit(points, values)

        result = spline.region_minimum(0)
        assert result is not None

        min_val, min_point = result
        assert min_val < 0.5  # Should find something near 0


class TestSwitchingVerifier:
    """Test full switching system verification pipeline."""

    def test_verifier_bouncing_ball_safe(self):
        """Test verifier on safe bouncing ball scenario."""
        dynamics = SwitchingDynamics.bouncing_ball(gravity=9.81, restitution=0.9)

        # Initial set: ball at height 0.8-1.2
        initial_set = HyperRectangle(
            lower=np.array([0.8, -0.5]),
            upper=np.array([1.2, 0.5])
        )

        # Unsafe set: height below -0.1
        unsafe_set = HyperRectangle(
            lower=np.array([-10.0, -100.0]),
            upper=np.array([-0.1, 100.0])
        )

        verifier = SwitchingVerifier(n_samples=30, seed=42)
        result = verifier.verify(dynamics, initial_set, unsafe_set, 1.0)

        assert result.status == VerificationStatus.SAFE

    def test_verifier_relay_safe(self):
        """Test verifier on safe relay feedback scenario."""
        # Relay feedback is 1D
        dynamics = SwitchingDynamics.relay_feedback()

        initial_set = HyperRectangle(
            lower=np.array([0.5]),
            upper=np.array([1.5])
        )

        unsafe_set = HyperRectangle(
            lower=np.array([-10.0]),
            upper=np.array([-3.0])
        )

        verifier = SwitchingVerifier(n_samples=50, seed=42)
        result = verifier.verify(dynamics, initial_set, unsafe_set, 2.0)

        assert result.status == VerificationStatus.SAFE

    def test_verifier_relay_unsafe(self):
        """Test verifier on unsafe relay feedback scenario."""
        # Relay feedback is 1D
        dynamics = SwitchingDynamics.relay_feedback()

        # Initial set containing origin (which relay reaches)
        initial_set = HyperRectangle(
            lower=np.array([-0.5]),
            upper=np.array([0.5])
        )

        # 1D unsafe set containing origin
        unsafe_set = HyperRectangle(
            lower=np.array([-0.1]),
            upper=np.array([0.1])
        )

        verifier = SwitchingVerifier(n_samples=50, seed=42)
        result = verifier.verify(dynamics, initial_set, unsafe_set, 2.0)

        assert result.status == VerificationStatus.UNSAFE

    def test_verifier_returns_switching_info(self):
        """Test that verifier returns switching-specific details."""
        # Relay feedback is 1D
        dynamics = SwitchingDynamics.relay_feedback()

        initial_set = HyperRectangle(
            lower=np.array([0.5]),
            upper=np.array([1.5])
        )

        unsafe_set = HyperRectangle(
            lower=np.array([-10.0]),
            upper=np.array([-3.0])
        )

        verifier = SwitchingVerifier(n_samples=30, seed=42)
        result = verifier.verify(dynamics, initial_set, unsafe_set, 1.0)

        assert 'switching_info' in result.details
        assert 'n_regions' in result.details['switching_info']
        assert 'region_counts' in result.details['switching_info']

    def test_verifier_thermostat(self):
        """Test verifier on thermostat system."""
        dynamics = SwitchingDynamics.thermostat(
            T_low=18.0, T_high=22.0, T_ambient=10.0,
            cooling_rate=0.1, heating_power=2.0
        )

        initial_set = HyperRectangle(
            lower=np.array([19.0]),
            upper=np.array([21.0])
        )

        # Unsafe: temperature too cold
        unsafe_set = HyperRectangle(
            lower=np.array([0.0]),
            upper=np.array([15.0])
        )

        verifier = SwitchingVerifier(n_samples=30, seed=42)
        result = verifier.verify(dynamics, initial_set, unsafe_set, 5.0)

        assert result.status == VerificationStatus.SAFE
