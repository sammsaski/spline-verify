"""Ground truth validation for verification results.

Provides systems with analytically known safety properties for validating
the verification algorithm.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..dynamics import ODEDynamics
from ..geometry import HyperRectangle, Ball, Set
from ..verification import VerificationStatus


class GroundTruthSystem(ABC):
    """Base class for systems with known ground truth.

    Subclasses provide:
    - Dynamics model
    - Initial and unsafe sets
    - Analytical solution for safety
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the ground truth system."""
        ...

    @property
    @abstractmethod
    def dynamics(self) -> ODEDynamics:
        """Dynamics model for the system."""
        ...

    @property
    @abstractmethod
    def initial_set(self) -> Set:
        """Initial condition set."""
        ...

    @property
    @abstractmethod
    def unsafe_set(self) -> Set:
        """Unsafe region."""
        ...

    @property
    @abstractmethod
    def time_horizon(self) -> float:
        """Time horizon T."""
        ...

    @property
    @abstractmethod
    def ground_truth_status(self) -> VerificationStatus:
        """Analytically determined safety status."""
        ...

    @abstractmethod
    def compute_exact_min_distance(self) -> float:
        """Compute the exact minimum distance to unsafe set.

        Returns:
            The true minimum of F_T over the initial set.
        """
        ...

    def to_benchmark_spec(self) -> dict:
        """Convert to benchmark specification dictionary."""
        return {
            'name': self.name,
            'dynamics': self.dynamics,
            'initial_set': self.initial_set,
            'unsafe_set': self.unsafe_set,
            'time_horizon': self.time_horizon,
            'ground_truth': self.ground_truth_status,
        }


@dataclass
class LinearGroundTruth(GroundTruthSystem):
    """Linear system with analytical solution.

    For dx/dt = Ax, the solution is x(t) = exp(At) @ x0.

    For stable systems (eigenvalues with negative real parts), trajectories
    converge to the origin.
    """

    A: np.ndarray
    x0_lower: np.ndarray
    x0_upper: np.ndarray
    unsafe_center: np.ndarray
    unsafe_radius: float
    T: float
    system_name: str = "linear_system"

    def __post_init__(self):
        self._dynamics = ODEDynamics.from_matrix(self.A)
        self._initial_set = HyperRectangle(lower=self.x0_lower, upper=self.x0_upper)
        self._unsafe_set = Ball(center=self.unsafe_center, radius=self.unsafe_radius)

        # Precompute matrix exponential at key times
        self._compute_ground_truth()

    @property
    def name(self) -> str:
        return self.system_name

    @property
    def dynamics(self) -> ODEDynamics:
        return self._dynamics

    @property
    def initial_set(self) -> Set:
        return self._initial_set

    @property
    def unsafe_set(self) -> Set:
        return self._unsafe_set

    @property
    def time_horizon(self) -> float:
        return self.T

    @property
    def ground_truth_status(self) -> VerificationStatus:
        return self._ground_truth_status

    def _matrix_exp(self, t: float) -> np.ndarray:
        """Compute matrix exponential exp(At)."""
        from scipy.linalg import expm
        return expm(self.A * t)

    def _trajectory(self, x0: np.ndarray, t: float) -> np.ndarray:
        """Compute trajectory at time t from x0."""
        return self._matrix_exp(t) @ x0

    def _compute_ground_truth(self):
        """Compute ground truth by sampling trajectories densely."""
        # Sample corners and edges of initial set (worst cases for linear systems)
        n_dims = len(self.x0_lower)

        # Generate corner points
        corners = []
        for i in range(2**n_dims):
            corner = np.array([
                self.x0_upper[j] if (i >> j) & 1 else self.x0_lower[j]
                for j in range(n_dims)
            ])
            corners.append(corner)

        # Also sample along edges and interior
        n_samples = 20
        rng = np.random.default_rng(42)
        interior_samples = []
        for _ in range(n_samples):
            sample = rng.uniform(self.x0_lower, self.x0_upper)
            interior_samples.append(sample)

        all_samples = corners + interior_samples

        # Dense time sampling
        times = np.linspace(0, self.T, 500)

        # Compute minimum distance
        min_dist = float('inf')
        for x0 in all_samples:
            for t in times:
                x_t = self._trajectory(x0, t)
                dist = self._unsafe_set.distance(x_t)
                min_dist = min(min_dist, dist)

        self._exact_min_distance = min_dist

        # Determine status
        if min_dist <= 0:
            self._ground_truth_status = VerificationStatus.UNSAFE
        else:
            self._ground_truth_status = VerificationStatus.SAFE

    def compute_exact_min_distance(self) -> float:
        return self._exact_min_distance

    @classmethod
    def stable_spiral_safe(cls) -> LinearGroundTruth:
        """Stable spiral that doesn't reach origin in short time."""
        A = np.array([[-0.5, 1.0], [-1.0, -0.5]])
        return cls(
            A=A,
            x0_lower=np.array([2.0, 2.0]),
            x0_upper=np.array([3.0, 3.0]),
            unsafe_center=np.array([0.0, 0.0]),
            unsafe_radius=0.1,
            T=1.0,
            system_name="stable_spiral_safe",
        )

    @classmethod
    def stable_spiral_unsafe(cls) -> LinearGroundTruth:
        """Stable spiral that reaches origin."""
        A = np.array([[-0.5, 1.0], [-1.0, -0.5]])
        return cls(
            A=A,
            x0_lower=np.array([-0.5, -0.5]),
            x0_upper=np.array([0.5, 0.5]),
            unsafe_center=np.array([0.0, 0.0]),
            unsafe_radius=0.1,
            T=5.0,
            system_name="stable_spiral_unsafe",
        )

    @classmethod
    def fast_decay_safe(cls) -> LinearGroundTruth:
        """Fast decaying system that stays safe."""
        A = np.array([[-2.0, 0.0], [0.0, -2.0]])
        return cls(
            A=A,
            x0_lower=np.array([1.0, 1.0]),
            x0_upper=np.array([2.0, 2.0]),
            unsafe_center=np.array([0.0, 0.0]),
            unsafe_radius=0.3,
            T=1.0,
            system_name="fast_decay_safe",
        )

    @classmethod
    def slow_decay_unsafe(cls) -> LinearGroundTruth:
        """Slow decaying system that reaches origin."""
        A = np.array([[-0.1, 0.0], [0.0, -0.1]])
        return cls(
            A=A,
            x0_lower=np.array([0.2, 0.2]),
            x0_upper=np.array([0.5, 0.5]),
            unsafe_center=np.array([0.0, 0.0]),
            unsafe_radius=0.15,
            T=10.0,
            system_name="slow_decay_unsafe",
        )


@dataclass
class HarmonicOscillatorGroundTruth(GroundTruthSystem):
    """Harmonic oscillator with energy conservation.

    For dx/dt = y, dy/dt = -omega^2 * x:
    - Energy E = (x^2 + y^2/omega^2) / 2 is conserved
    - Trajectories are circles with radius sqrt(2E)
    """

    omega: float
    x0_lower: np.ndarray
    x0_upper: np.ndarray
    unsafe_center: np.ndarray
    unsafe_radius: float
    T: float
    system_name: str = "harmonic_oscillator"

    def __post_init__(self):
        self._dynamics = ODEDynamics.harmonic_oscillator(omega=self.omega)
        self._initial_set = HyperRectangle(lower=self.x0_lower, upper=self.x0_upper)
        self._unsafe_set = Ball(center=self.unsafe_center, radius=self.unsafe_radius)

        self._compute_ground_truth()

    @property
    def name(self) -> str:
        return self.system_name

    @property
    def dynamics(self) -> ODEDynamics:
        return self._dynamics

    @property
    def initial_set(self) -> Set:
        return self._initial_set

    @property
    def unsafe_set(self) -> Set:
        return self._unsafe_set

    @property
    def time_horizon(self) -> float:
        return self.T

    @property
    def ground_truth_status(self) -> VerificationStatus:
        return self._ground_truth_status

    def _trajectory(self, x0: np.ndarray, t: float) -> np.ndarray:
        """Compute trajectory at time t from x0."""
        x, y = x0
        omega = self.omega
        cos_t = np.cos(omega * t)
        sin_t = np.sin(omega * t)
        return np.array([
            x * cos_t + y / omega * sin_t,
            -x * omega * sin_t + y * cos_t
        ])

    def _orbit_radius(self, x0: np.ndarray) -> float:
        """Compute orbit radius (conserved) for initial condition."""
        x, y = x0
        # Energy = (x^2 + y^2/omega^2) / 2
        # Orbit radius = sqrt(2 * Energy)
        energy = (x**2 + (y / self.omega)**2) / 2
        return np.sqrt(2 * energy)

    def _compute_ground_truth(self):
        """Compute ground truth using energy conservation."""
        # For harmonic oscillator, the trajectory sweeps a circle
        # The minimum distance to unsafe set depends on orbit radius

        # Sample initial conditions
        n_samples = 100
        rng = np.random.default_rng(42)
        samples = rng.uniform(self.x0_lower, self.x0_upper, size=(n_samples, 2))

        # Also include corners
        corners = [
            self.x0_lower,
            self.x0_upper,
            np.array([self.x0_lower[0], self.x0_upper[1]]),
            np.array([self.x0_upper[0], self.x0_lower[1]]),
        ]

        all_samples = list(samples) + corners

        # For each sample, compute min distance over one period
        min_dist = float('inf')
        period = 2 * np.pi / self.omega
        n_time = 200
        # Use min of T and one period
        t_max = min(self.T, period)
        times = np.linspace(0, t_max, n_time)

        for x0 in all_samples:
            for t in times:
                x_t = self._trajectory(x0, t)
                dist = self._unsafe_set.distance(x_t)
                min_dist = min(min_dist, dist)

        self._exact_min_distance = min_dist

        if min_dist <= 0:
            self._ground_truth_status = VerificationStatus.UNSAFE
        else:
            self._ground_truth_status = VerificationStatus.SAFE

    def compute_exact_min_distance(self) -> float:
        return self._exact_min_distance

    @classmethod
    def small_orbit_safe(cls) -> HarmonicOscillatorGroundTruth:
        """Small orbit that doesn't reach distant unsafe set."""
        return cls(
            omega=1.0,
            x0_lower=np.array([0.8, -0.2]),
            x0_upper=np.array([1.2, 0.2]),
            unsafe_center=np.array([-3.0, 0.0]),
            unsafe_radius=0.5,
            T=2 * np.pi,  # One period
            system_name="harmonic_small_orbit_safe",
        )

    @classmethod
    def large_orbit_unsafe(cls) -> HarmonicOscillatorGroundTruth:
        """Large orbit that passes through unsafe set."""
        return cls(
            omega=1.0,
            x0_lower=np.array([1.5, -0.2]),
            x0_upper=np.array([2.0, 0.2]),
            unsafe_center=np.array([-1.8, 0.0]),
            unsafe_radius=0.3,
            T=np.pi,  # Half period
            system_name="harmonic_large_orbit_unsafe",
        )

    @classmethod
    def high_frequency_safe(cls) -> HarmonicOscillatorGroundTruth:
        """High frequency oscillator, safe case."""
        return cls(
            omega=3.0,
            x0_lower=np.array([0.5, -0.1]),
            x0_upper=np.array([0.7, 0.1]),
            unsafe_center=np.array([-2.0, 0.0]),
            unsafe_radius=0.3,
            T=2 * np.pi / 3,  # One period
            system_name="harmonic_high_freq_safe",
        )


def validate_against_ground_truth(
    systems: list[GroundTruthSystem],
    n_samples: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Validate verification algorithm against ground truth systems.

    Args:
        systems: List of ground truth systems to validate.
        n_samples: Number of samples for verification.
        seed: Random seed.
        verbose: Whether to print results.

    Returns:
        Dictionary with validation results.
    """
    from .runner import BenchmarkRunner

    runner = BenchmarkRunner(n_samples=n_samples, seed=seed, verbose=verbose)

    specs = [s.to_benchmark_spec() for s in systems]
    suite = runner.run_suite("ground_truth_validation", specs)

    # Compute additional metrics
    results = {
        'n_systems': len(systems),
        'n_correct': suite.n_correct,
        'n_incorrect': suite.n_incorrect,
        'accuracy': suite.accuracy,
        'total_time': suite.total_time,
        'details': [],
    }

    for system, result in zip(systems, suite.results):
        exact_min = system.compute_exact_min_distance()
        error = abs(result.min_objective - exact_min)

        detail = {
            'name': system.name,
            'ground_truth': system.ground_truth_status.name,
            'computed': result.status.name,
            'correct': result.is_correct,
            'exact_min_distance': exact_min,
            'computed_min_objective': result.min_objective,
            'approximation_error': error,
            'relative_error': error / max(abs(exact_min), 1e-10),
        }
        results['details'].append(detail)

    if verbose:
        print("\nValidation Summary:")
        print("=" * 60)
        for d in results['details']:
            status = "OK" if d['correct'] else "WRONG"
            print(f"{d['name']}: {d['ground_truth']} -> {d['computed']} [{status}]")
            print(f"  Exact: {d['exact_min_distance']:.6f}, "
                  f"Computed: {d['computed_min_objective']:.6f}, "
                  f"Error: {d['approximation_error']:.6f}")
        print("=" * 60)
        print(f"Accuracy: {results['accuracy']:.1%}")

    return results


def get_all_ground_truth_systems() -> list[GroundTruthSystem]:
    """Get all predefined ground truth systems for validation."""
    return [
        LinearGroundTruth.stable_spiral_safe(),
        LinearGroundTruth.stable_spiral_unsafe(),
        LinearGroundTruth.fast_decay_safe(),
        LinearGroundTruth.slow_decay_unsafe(),
        HarmonicOscillatorGroundTruth.small_orbit_safe(),
        HarmonicOscillatorGroundTruth.large_orbit_unsafe(),
        HarmonicOscillatorGroundTruth.high_frequency_safe(),
    ]
