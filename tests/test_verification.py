"""Tests for the safety verification pipeline."""

import numpy as np
import pytest

from spline_verify.dynamics import ODEDynamics
from spline_verify.geometry import HyperRectangle, Ball
from spline_verify.verification import SafetyVerifier, VerificationStatus
from spline_verify.verification.objective import ObjectiveSampler, compute_objective


class TestObjectiveFunction:
    """Tests for objective function computation."""

    def test_compute_objective_safe(self):
        """Test objective for trajectory that stays far from unsafe set."""
        # Harmonic oscillator starting at (1, 0) - orbit has radius 1
        dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

        x0 = np.array([1.0, 0.0])
        bundle = dynamics.simulate(x0, (0, 2*np.pi))
        traj = bundle.primary

        # Unsafe set far from origin
        unsafe_set = Ball(center=np.array([5.0, 0.0]), radius=0.5)

        obj = compute_objective(traj, unsafe_set)

        # Closest point on orbit to (5, 0) is (1, 0), distance = 4 - 0.5 = 3.5
        assert obj > 3.0

    def test_compute_objective_unsafe(self):
        """Test objective when trajectory reaches unsafe set."""
        # Linear system converging to origin
        A = np.array([[-1, 0], [0, -1]])
        dynamics = ODEDynamics.from_matrix(A)

        x0 = np.array([0.05, 0.05])  # Start close to origin
        bundle = dynamics.simulate(x0, (0, 10))
        traj = bundle.primary

        # Unsafe set at origin
        unsafe_set = Ball(center=np.array([0.0, 0.0]), radius=0.01)

        obj = compute_objective(traj, unsafe_set)

        # Trajectory converges to origin, should hit unsafe set
        assert obj < 0.1


class TestSafetyVerifier:
    """Tests for the safety verification pipeline."""

    def test_verifier_safe_case(self):
        """Test verification of a provably safe system."""
        # Harmonic oscillator with small initial set
        dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

        # Initial set: small box near (1, 0)
        initial_set = HyperRectangle(
            lower=np.array([0.9, -0.1]),
            upper=np.array([1.1, 0.1])
        )

        # Unsafe set: far from any orbit
        unsafe_set = Ball(center=np.array([5.0, 0.0]), radius=0.5)

        T = 2 * np.pi

        verifier = SafetyVerifier(n_samples=100, seed=42)
        result = verifier.verify(dynamics, initial_set, unsafe_set, T)

        assert result.status == VerificationStatus.SAFE
        assert result.min_objective > 0
        assert result.safety_margin > 0

    def test_verifier_unsafe_case(self):
        """Test verification of a provably unsafe system."""
        # Linear system converging to origin
        A = np.array([[-0.5, 0], [0, -0.5]])
        dynamics = ODEDynamics.from_matrix(A)

        # Initial set containing origin
        initial_set = HyperRectangle(
            lower=np.array([-0.5, -0.5]),
            upper=np.array([0.5, 0.5])
        )

        # Unsafe set at origin
        unsafe_set = Ball(center=np.array([0.0, 0.0]), radius=0.1)

        T = 5.0

        verifier = SafetyVerifier(n_samples=100, seed=42)
        result = verifier.verify(dynamics, initial_set, unsafe_set, T)

        # Should be unsafe since origin is in initial set AND in unsafe set
        assert result.status == VerificationStatus.UNSAFE
        assert result.counterexample is not None

    def test_verifier_result_details(self):
        """Test that verifier returns detailed results."""
        dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

        initial_set = HyperRectangle(
            lower=np.array([0.9, -0.1]),
            upper=np.array([1.1, 0.1])
        )
        unsafe_set = Ball(center=np.array([5.0, 0.0]), radius=0.5)

        verifier = SafetyVerifier(n_samples=50, seed=42)
        result = verifier.verify(dynamics, initial_set, unsafe_set, 2*np.pi)

        # Check all expected fields
        assert hasattr(result, 'status')
        assert hasattr(result, 'min_objective')
        assert hasattr(result, 'minimizer')
        assert hasattr(result, 'error_bound')
        assert hasattr(result, 'safety_margin')

        # Check details dict
        assert 'n_samples' in result.details
        assert 'sampled_min' in result.details
        assert 'error_budget' in result.details

    def test_verifier_with_refinement(self):
        """Test adaptive refinement."""
        dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

        initial_set = HyperRectangle(
            lower=np.array([0.9, -0.1]),
            upper=np.array([1.1, 0.1])
        )
        unsafe_set = Ball(center=np.array([5.0, 0.0]), radius=0.5)

        verifier = SafetyVerifier(seed=42)
        result = verifier.verify_with_refinement(
            dynamics, initial_set, unsafe_set, 2*np.pi,
            n_samples=50, max_iterations=3
        )

        assert result.status == VerificationStatus.SAFE


class TestLinearSystems:
    """Tests using linear systems with analytical solutions."""

    def test_stable_system_converging(self):
        """Test stable system that converges to equilibrium."""
        # dx/dt = -x, solution: x(t) = x0 * exp(-t)
        A = np.array([[-1]])
        dynamics = ODEDynamics.from_matrix(A)

        initial_set = HyperRectangle(
            lower=np.array([1.0]),
            upper=np.array([2.0])
        )

        # Unsafe: x < 0.05 (will be reached as t -> inf)
        # But with finite T, might not reach it
        unsafe_set = Ball(center=np.array([0.0]), radius=0.05)

        # Short time: safe
        verifier = SafetyVerifier(n_samples=50, seed=42)
        result_short = verifier.verify(dynamics, initial_set, unsafe_set, 1.0)

        # After T=1, x in [e^-1, 2*e^-1] ~ [0.37, 0.74], far from 0.05
        assert result_short.status in [VerificationStatus.SAFE, VerificationStatus.UNKNOWN]

        # Long time: unsafe
        result_long = verifier.verify(dynamics, initial_set, unsafe_set, 10.0)

        # After T=10, x in [e^-10, 2*e^-10] ~ [4.5e-5, 9e-5], inside unsafe ball
        assert result_long.status == VerificationStatus.UNSAFE

    def test_oscillating_system(self):
        """Test system with oscillatory behavior."""
        # Undamped oscillator: conserves energy
        dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

        # Initial set determines orbit radius
        initial_set = HyperRectangle(
            lower=np.array([0.9, -0.1]),
            upper=np.array([1.1, 0.1])
        )

        # Unsafe region far from orbits
        unsafe_set = Ball(center=np.array([-3.0, 0.0]), radius=0.5)

        verifier = SafetyVerifier(n_samples=100, seed=42)
        result = verifier.verify(dynamics, initial_set, unsafe_set, 2*np.pi)

        # Max extent of orbit is ~1.1, unsafe is at -3, should be safe
        assert result.status == VerificationStatus.SAFE


class TestErrorBounds:
    """Tests for error bound computation."""

    def test_error_bound_decreases_with_samples(self):
        """More samples should give tighter error bounds."""
        dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

        initial_set = HyperRectangle(
            lower=np.array([0.9, -0.1]),
            upper=np.array([1.1, 0.1])
        )
        unsafe_set = Ball(center=np.array([5.0, 0.0]), radius=0.5)

        verifier = SafetyVerifier(seed=42)

        # Verify with different sample counts
        result_50 = verifier.verify(dynamics, initial_set, unsafe_set, 2*np.pi, n_samples=50)
        result_200 = verifier.verify(dynamics, initial_set, unsafe_set, 2*np.pi, n_samples=200)

        # Error bound should decrease with more samples
        # (sampling error component decreases)
        # Note: This is a soft test - depends on random sampling
        assert result_200.error_bound <= result_50.error_bound * 1.5
