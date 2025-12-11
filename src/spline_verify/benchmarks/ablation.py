"""Ablation study framework for verification pipeline.

Provides systematic ablation studies to isolate each component's contribution:
- Integrator methods (Euler, RK4, RK45, Adams)
- Sampling strategies (uniform, Latin hypercube, Sobol, Halton)
- Spline methods (RBF vs B-spline)
- Sample counts (convergence study)
- Optimization methods (multistart, differential evolution)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from pathlib import Path
import json

import numpy as np

from ..dynamics import ODEDynamics, DynamicsModel
from ..geometry import HyperRectangle, Ball, Set
from ..geometry.sampling import SamplingStrategy
from ..verification import SafetyVerifier, VerificationStatus, VerificationResult
from .runner import BenchmarkResult, BenchmarkSuite, TimingResult


@dataclass
class AblationResult:
    """Result of an ablation study comparing configurations.

    Attributes:
        study_name: Name of the ablation study.
        baseline_config: Configuration of the baseline.
        baseline_result: Result with baseline configuration.
        ablations: Dict mapping config name to result.
        problem_name: Name of the test problem.
        metadata: Additional study data.
    """
    study_name: str
    baseline_config: dict
    baseline_result: BenchmarkResult
    ablations: dict[str, BenchmarkResult] = field(default_factory=dict)
    problem_name: str = ""
    metadata: dict = field(default_factory=dict)

    def impact(self, config_name: str) -> dict:
        """Compute impact of a configuration change.

        Returns dict with:
        - time_ratio: time / baseline_time
        - error_diff: error_bound - baseline_error
        - min_diff: min_objective - baseline_min
        """
        if config_name not in self.ablations:
            raise KeyError(f"Unknown configuration: {config_name}")

        ablation = self.ablations[config_name]
        baseline = self.baseline_result

        return {
            'time_ratio': ablation.timing.total_time / baseline.timing.total_time,
            'error_diff': ablation.error_bound - baseline.error_bound,
            'min_diff': ablation.min_objective - baseline.min_objective,
            'status_changed': ablation.status != baseline.status,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'study_name': self.study_name,
            'baseline_config': self.baseline_config,
            'baseline_result': self.baseline_result.to_dict(),
            'ablations': {k: v.to_dict() for k, v in self.ablations.items()},
            'problem_name': self.problem_name,
            'metadata': self.metadata,
        }

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            f"Ablation Study: {self.study_name}",
            f"Problem: {self.problem_name}",
            "=" * 60,
            f"Baseline: {self.baseline_config}",
            f"  Time: {self.baseline_result.timing.total_time:.3f}s",
            f"  Min: {self.baseline_result.min_objective:.6f}",
            f"  Error: {self.baseline_result.error_bound:.6f}",
            f"  Status: {self.baseline_result.status.name}",
            "",
            "Ablations:",
        ]

        for name, result in self.ablations.items():
            impact = self.impact(name)
            lines.append(f"  {name}:")
            lines.append(f"    Time: {result.timing.total_time:.3f}s ({impact['time_ratio']:.2f}x)")
            lines.append(f"    Min: {result.min_objective:.6f} (diff: {impact['min_diff']:+.6f})")
            lines.append(f"    Status: {result.status.name}")

        return '\n'.join(lines)


@dataclass
class AblationStudy:
    """Systematic ablation study for verification pipeline.

    Supports isolating the contribution of different components
    by varying one parameter at a time.
    """
    n_samples: int = 200
    time_horizon: float = 3.0
    seed: int = 42
    verbose: bool = True

    def _run_verification(
        self,
        dynamics: DynamicsModel,
        initial_set: Set,
        unsafe_set: Set,
        config_name: str,
        **verifier_kwargs
    ) -> BenchmarkResult:
        """Run verification with given configuration."""
        if self.verbose:
            print(f"  Running {config_name}...", end=' ', flush=True)

        # Merge default params with overrides
        params = {
            'n_samples': self.n_samples,
            'seed': self.seed,
            **verifier_kwargs
        }

        verifier = SafetyVerifier(**params)

        start = time.perf_counter()
        result = verifier.verify(
            dynamics, initial_set, unsafe_set, self.time_horizon
        )
        elapsed = time.perf_counter() - start

        if self.verbose:
            print(f"{result.status.name} [{elapsed:.2f}s]")

        timing = TimingResult(
            total_time=elapsed,
            n_samples=params['n_samples'],
            n_trajectories=params['n_samples'],
        )

        return BenchmarkResult(
            name=config_name,
            status=result.status,
            min_objective=result.min_objective,
            error_bound=result.error_bound,
            safety_margin=result.safety_margin,
            timing=timing,
            metadata={'config': verifier_kwargs}
        )

    def run_integrator_ablation(
        self,
        dynamics: DynamicsModel,
        initial_set: Set,
        unsafe_set: Set,
        problem_name: str = "test_problem"
    ) -> AblationResult:
        """Compare integrator methods: Euler, RK4, RK45.

        Note: Integrator selection is typically done in ODEDynamics.
        This ablation requires creating dynamics with different integrators.
        """
        if self.verbose:
            print(f"\nIntegrator Ablation ({problem_name})")
            print("-" * 40)

        # Baseline: RK45 (default)
        baseline = self._run_verification(
            dynamics, initial_set, unsafe_set, "RK45 (baseline)"
        )

        result = AblationResult(
            study_name="integrator",
            baseline_config={'integrator': 'RK45'},
            baseline_result=baseline,
            problem_name=problem_name,
        )

        # Note: Integrator ablation would require modifying dynamics creation
        # For now, we keep RK45 as the only option since it's most robust

        return result

    def run_sampling_ablation(
        self,
        dynamics: DynamicsModel,
        initial_set: Set,
        unsafe_set: Set,
        problem_name: str = "test_problem"
    ) -> AblationResult:
        """Compare sampling strategies: uniform, Latin hypercube, Sobol, Halton."""
        if self.verbose:
            print(f"\nSampling Ablation ({problem_name})")
            print("-" * 40)

        # Baseline: Latin Hypercube (default)
        baseline = self._run_verification(
            dynamics, initial_set, unsafe_set, "Latin Hypercube (baseline)",
            sampling_strategy=SamplingStrategy.LATIN_HYPERCUBE
        )

        result = AblationResult(
            study_name="sampling",
            baseline_config={'sampling': 'latin_hypercube'},
            baseline_result=baseline,
            problem_name=problem_name,
        )

        # Ablations
        strategies = [
            ("Uniform", SamplingStrategy.UNIFORM),
            ("Sobol", SamplingStrategy.SOBOL),
            ("Halton", SamplingStrategy.HALTON),
        ]

        for name, strategy in strategies:
            ablation = self._run_verification(
                dynamics, initial_set, unsafe_set, name,
                sampling_strategy=strategy
            )
            result.ablations[name] = ablation

        return result

    def run_spline_ablation(
        self,
        dynamics: DynamicsModel,
        initial_set: Set,
        unsafe_set: Set,
        problem_name: str = "test_problem"
    ) -> AblationResult:
        """Compare spline methods: RBF vs B-spline (1D/2D only)."""
        if self.verbose:
            print(f"\nSpline Method Ablation ({problem_name})")
            print("-" * 40)

        n_dims = initial_set.n_dims
        if n_dims > 2:
            if self.verbose:
                print(f"  Warning: Problem is {n_dims}D, B-spline only supports 1D/2D")
                print("  Running RBF-only comparison")

        # Baseline: RBF with thin-plate spline
        baseline = self._run_verification(
            dynamics, initial_set, unsafe_set, "RBF-TPS (baseline)",
            spline_method='rbf',
            spline_kernel='thin_plate_spline'
        )

        result = AblationResult(
            study_name="spline_method",
            baseline_config={'spline_method': 'rbf', 'kernel': 'thin_plate_spline'},
            baseline_result=baseline,
            problem_name=problem_name,
        )

        # RBF variants
        ablation = self._run_verification(
            dynamics, initial_set, unsafe_set, "RBF-Cubic",
            spline_method='rbf',
            spline_kernel='cubic'
        )
        result.ablations["RBF-Cubic"] = ablation

        # B-spline (only for 1D/2D)
        if n_dims <= 2:
            ablation = self._run_verification(
                dynamics, initial_set, unsafe_set, "B-spline",
                spline_method='bspline',
                n_grid_points=50,
                bspline_degree=3
            )
            result.ablations["B-spline"] = ablation

        return result

    def run_sample_count_ablation(
        self,
        dynamics: DynamicsModel,
        initial_set: Set,
        unsafe_set: Set,
        sample_counts: list[int] = None,
        problem_name: str = "test_problem"
    ) -> AblationResult:
        """Study convergence with sample count."""
        if sample_counts is None:
            sample_counts = [50, 100, 200, 400]

        if self.verbose:
            print(f"\nSample Count Ablation ({problem_name})")
            print("-" * 40)

        # Find baseline (use 200 as default baseline)
        baseline_n = 200 if 200 in sample_counts else sample_counts[len(sample_counts) // 2]
        saved_n = self.n_samples
        self.n_samples = baseline_n

        baseline = self._run_verification(
            dynamics, initial_set, unsafe_set, f"n={baseline_n} (baseline)"
        )

        result = AblationResult(
            study_name="sample_count",
            baseline_config={'n_samples': baseline_n},
            baseline_result=baseline,
            problem_name=problem_name,
        )

        # Other sample counts
        for n in sample_counts:
            if n == baseline_n:
                continue
            self.n_samples = n
            ablation = self._run_verification(
                dynamics, initial_set, unsafe_set, f"n={n}"
            )
            result.ablations[f"n={n}"] = ablation

        self.n_samples = saved_n
        return result

    def run_optimization_ablation(
        self,
        dynamics: DynamicsModel,
        initial_set: Set,
        unsafe_set: Set,
        problem_name: str = "test_problem"
    ) -> AblationResult:
        """Compare optimization methods: multistart, differential evolution."""
        if self.verbose:
            print(f"\nOptimization Ablation ({problem_name})")
            print("-" * 40)

        # Baseline: multistart
        baseline = self._run_verification(
            dynamics, initial_set, unsafe_set, "Multistart (baseline)",
            optimization_method='multistart',
            n_optimization_starts=20
        )

        result = AblationResult(
            study_name="optimization",
            baseline_config={'method': 'multistart', 'n_starts': 20},
            baseline_result=baseline,
            problem_name=problem_name,
        )

        # Differential evolution
        ablation = self._run_verification(
            dynamics, initial_set, unsafe_set, "Differential Evolution",
            optimization_method='differential_evolution'
        )
        result.ablations["Differential Evolution"] = ablation

        return result


# Benchmark problems for ablation studies

def create_diagonal_decay_4d() -> tuple[DynamicsModel, Set, Set]:
    """4D diagonal decay system.

    ẋ = -0.5 * x (stable, decays to origin)
    """
    A = -0.5 * np.eye(4)
    dynamics = ODEDynamics.from_matrix(A)

    center = 2.0 * np.ones(4)
    initial_set = HyperRectangle(lower=center - 0.3, upper=center + 0.3)
    unsafe_set = Ball(center=np.zeros(4), radius=0.2)

    return dynamics, initial_set, unsafe_set


def create_coupled_oscillator_4d() -> tuple[DynamicsModel, Set, Set]:
    """4D coupled oscillator system.

    Two coupled harmonic oscillators:
    ẋ1 = x2
    ẋ2 = -x1 - 0.1*x2 + 0.05*x3
    ẋ3 = x4
    ẋ4 = -x3 - 0.1*x4 + 0.05*x1
    """
    def coupled_oscillator(t, x):
        return np.array([
            x[1],
            -x[0] - 0.1 * x[1] + 0.05 * x[2],
            x[3],
            -x[2] - 0.1 * x[3] + 0.05 * x[0],
        ])

    dynamics = ODEDynamics(f=coupled_oscillator, _n_dims=4)

    initial_set = HyperRectangle(
        lower=np.array([1.5, -0.2, 1.5, -0.2]),
        upper=np.array([2.0, 0.2, 2.0, 0.2])
    )
    unsafe_set = Ball(center=np.array([0, 0, 0, 0]), radius=0.3)

    return dynamics, initial_set, unsafe_set


def create_van_der_pol_2d(mu: float = 0.5) -> tuple[DynamicsModel, Set, Set]:
    """Van der Pol oscillator (2D).

    ẋ1 = x2
    ẋ2 = μ(1 - x1²)x2 - x1

    For small μ, this is a mild nonlinear oscillator.
    """
    def van_der_pol(t, x):
        return np.array([
            x[1],
            mu * (1 - x[0]**2) * x[1] - x[0]
        ])

    dynamics = ODEDynamics(f=van_der_pol, _n_dims=2)

    initial_set = HyperRectangle(
        lower=np.array([1.5, -0.3]),
        upper=np.array([2.0, 0.3])
    )
    unsafe_set = Ball(center=np.array([-2.5, 0.0]), radius=0.4)

    return dynamics, initial_set, unsafe_set


def create_lorenz_like_3d() -> tuple[DynamicsModel, Set, Set]:
    """Simplified Lorenz-like chaotic system (3D).

    Modified Lorenz with reduced parameters for faster dynamics.
    """
    sigma = 5.0  # Reduced from 10
    rho = 14.0   # Reduced from 28
    beta = 2.0   # Reduced from 8/3

    def lorenz_like(t, x):
        return np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2]
        ])

    dynamics = ODEDynamics(f=lorenz_like, _n_dims=3)

    initial_set = HyperRectangle(
        lower=np.array([0.5, 0.5, 10.0]),
        upper=np.array([1.5, 1.5, 12.0])
    )
    unsafe_set = Ball(center=np.array([0, 0, 30]), radius=2.0)

    return dynamics, initial_set, unsafe_set


def create_high_dim_diagonal_6d() -> tuple[DynamicsModel, Set, Set]:
    """6D high-dimensional diagonal decay.

    Tests scalability with higher dimensions.
    """
    A = -0.4 * np.eye(6)
    # Add some coupling
    for i in range(5):
        A[i, i + 1] = 0.1
        A[i + 1, i] = -0.1

    dynamics = ODEDynamics.from_matrix(A)

    center = 1.5 * np.ones(6)
    initial_set = HyperRectangle(lower=center - 0.2, upper=center + 0.2)
    unsafe_set = Ball(center=np.zeros(6), radius=0.15)

    return dynamics, initial_set, unsafe_set


# =============================================================================
# HARDER BENCHMARK PROBLEMS for stress testing
# =============================================================================

def create_high_dim_coupled_8d() -> tuple[DynamicsModel, Set, Set]:
    """8D coupled oscillator chain.

    Four coupled harmonic oscillators in a chain:
    - Higher dimension tests RBF scalability
    - Coupling makes dynamics more complex
    - Longer integration needed for interesting behavior
    """
    def coupled_chain(t, x):
        # x = [q1, p1, q2, p2, q3, p3, q4, p4]
        # Positions: q1, q2, q3, q4 (indices 0, 2, 4, 6)
        # Momenta: p1, p2, p3, p4 (indices 1, 3, 5, 7)
        k = 1.0  # spring constant
        c = 0.05  # coupling strength
        damping = 0.02  # light damping

        dxdt = np.zeros(8)

        # q_i' = p_i
        dxdt[0] = x[1]
        dxdt[2] = x[3]
        dxdt[4] = x[5]
        dxdt[6] = x[7]

        # p_i' = -k*q_i - damping*p_i + coupling terms
        dxdt[1] = -k * x[0] - damping * x[1] + c * (x[2] - x[0])
        dxdt[3] = -k * x[2] - damping * x[3] + c * (x[0] - 2*x[2] + x[4])
        dxdt[5] = -k * x[4] - damping * x[5] + c * (x[2] - 2*x[4] + x[6])
        dxdt[7] = -k * x[6] - damping * x[7] + c * (x[4] - x[6])

        return dxdt

    dynamics = ODEDynamics(f=coupled_chain, _n_dims=8)

    # Initial conditions: slightly perturbed equilibrium
    center = np.array([1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0])
    initial_set = HyperRectangle(lower=center - 0.15, upper=center + 0.15)

    # Unsafe: high energy state
    unsafe_set = Ball(center=np.array([0, 0, 0, 0, 0, 0, 0, 0]), radius=0.1)

    return dynamics, initial_set, unsafe_set


def create_high_dim_diagonal_10d() -> tuple[DynamicsModel, Set, Set]:
    """10D diagonal system with varying decay rates.

    Tests scalability to higher dimensions with non-uniform dynamics.
    """
    # Varying decay rates
    decay_rates = np.array([-0.1, -0.2, -0.3, -0.4, -0.5,
                           -0.6, -0.7, -0.8, -0.9, -1.0])
    A = np.diag(decay_rates)

    # Add weak coupling between adjacent dimensions
    for i in range(9):
        A[i, i + 1] = 0.05
        A[i + 1, i] = 0.05

    dynamics = ODEDynamics.from_matrix(A)

    center = 2.0 * np.ones(10)
    initial_set = HyperRectangle(lower=center - 0.2, upper=center + 0.2)
    unsafe_set = Ball(center=np.zeros(10), radius=0.1)

    return dynamics, initial_set, unsafe_set


def create_stiff_vanderpol_2d(mu: float = 5.0) -> tuple[DynamicsModel, Set, Set]:
    """Stiff Van der Pol oscillator (2D).

    Higher μ makes the system stiffer, requiring adaptive integration.
    Tests integrator robustness.
    """
    def stiff_vdp(t, x):
        return np.array([
            x[1],
            mu * (1 - x[0]**2) * x[1] - x[0]
        ])

    dynamics = ODEDynamics(f=stiff_vdp, _n_dims=2)

    initial_set = HyperRectangle(
        lower=np.array([1.5, -0.5]),
        upper=np.array([2.5, 0.5])
    )
    unsafe_set = Ball(center=np.array([-2.5, 0.0]), radius=0.3)

    return dynamics, initial_set, unsafe_set


def create_rossler_3d() -> tuple[DynamicsModel, Set, Set]:
    """Rössler attractor (3D chaotic system).

    Classic chaotic system, tests handling of sensitive dependence.
    """
    a, b, c = 0.2, 0.2, 5.7

    def rossler(t, x):
        return np.array([
            -x[1] - x[2],
            x[0] + a * x[1],
            b + x[2] * (x[0] - c)
        ])

    dynamics = ODEDynamics(f=rossler, _n_dims=3)

    # Start near the attractor
    initial_set = HyperRectangle(
        lower=np.array([0.0, -1.0, 0.0]),
        upper=np.array([2.0, 1.0, 2.0])
    )
    # Unsafe region in the outer spiral
    unsafe_set = Ball(center=np.array([0.0, 0.0, 15.0]), radius=2.0)

    return dynamics, initial_set, unsafe_set


def create_double_pendulum_4d() -> tuple[DynamicsModel, Set, Set]:
    """Simplified double pendulum (4D).

    Chaotic dynamics with energy conservation (approximately).
    """
    g = 9.81  # gravity
    l1, l2 = 1.0, 1.0  # lengths
    m1, m2 = 1.0, 1.0  # masses

    def double_pendulum(t, x):
        # x = [theta1, omega1, theta2, omega2]
        theta1, omega1, theta2, omega2 = x
        delta = theta2 - theta1

        # Simplified equations (small angle approximation for stability)
        denom1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta)**2
        denom2 = (l2 / l1) * denom1

        dtheta1 = omega1
        dtheta2 = omega2

        domega1 = (m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                   m2 * g * np.sin(theta2) * np.cos(delta) +
                   m2 * l2 * omega2**2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta1)) / denom1

        domega2 = (-m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                   (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                   (m1 + m2) * l1 * omega1**2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta2)) / denom2

        return np.array([dtheta1, domega1, dtheta2, domega2])

    dynamics = ODEDynamics(f=double_pendulum, _n_dims=4)

    # Small initial angles
    initial_set = HyperRectangle(
        lower=np.array([0.3, -0.1, 0.3, -0.1]),
        upper=np.array([0.5, 0.1, 0.5, 0.1])
    )
    # Unsafe: pendulum flips (large angles)
    unsafe_set = Ball(center=np.array([np.pi, 0, np.pi, 0]), radius=0.5)

    return dynamics, initial_set, unsafe_set


def create_lotka_volterra_4d() -> tuple[DynamicsModel, Set, Set]:
    """4-species Lotka-Volterra competition (4D).

    Population dynamics with multiple equilibria.
    """
    # Interaction matrix
    alpha = np.array([
        [1.0, 0.5, 0.3, 0.1],
        [0.5, 1.0, 0.4, 0.2],
        [0.3, 0.4, 1.0, 0.5],
        [0.1, 0.2, 0.5, 1.0]
    ])
    r = np.array([1.0, 0.9, 0.8, 0.7])  # growth rates
    K = np.array([10.0, 10.0, 10.0, 10.0])  # carrying capacities

    def lotka_volterra(t, x):
        # Ensure non-negative
        x_pos = np.maximum(x, 0)
        dxdt = np.zeros(4)
        for i in range(4):
            competition = np.sum(alpha[i] * x_pos) / K[i]
            dxdt[i] = r[i] * x_pos[i] * (1 - competition)
        return dxdt

    dynamics = ODEDynamics(f=lotka_volterra, _n_dims=4)

    # Initial: all species present
    initial_set = HyperRectangle(
        lower=np.array([2.0, 2.0, 2.0, 2.0]),
        upper=np.array([4.0, 4.0, 4.0, 4.0])
    )
    # Unsafe: extinction of any species
    unsafe_set = Ball(center=np.array([0, 5, 5, 5]), radius=0.5)

    return dynamics, initial_set, unsafe_set


def create_neural_network_6d() -> tuple[DynamicsModel, Set, Set]:
    """6D recurrent neural network dynamics.

    Hopfield-like network with sigmoid activations.
    """
    # Weight matrix (symmetric for Hopfield)
    W = np.array([
        [0.0, 0.5, -0.3, 0.2, -0.1, 0.1],
        [0.5, 0.0, 0.4, -0.2, 0.1, -0.1],
        [-0.3, 0.4, 0.0, 0.3, -0.2, 0.2],
        [0.2, -0.2, 0.3, 0.0, 0.4, -0.3],
        [-0.1, 0.1, -0.2, 0.4, 0.0, 0.5],
        [0.1, -0.1, 0.2, -0.3, 0.5, 0.0]
    ])
    tau = 1.0  # time constant
    bias = np.zeros(6)

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def neural_net(t, x):
        return (-x + W @ sigmoid(x) + bias) / tau

    dynamics = ODEDynamics(f=neural_net, _n_dims=6)

    initial_set = HyperRectangle(
        lower=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        upper=np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    )
    # Unsafe: all neurons saturated high
    unsafe_set = Ball(center=np.ones(6), radius=0.1)

    return dynamics, initial_set, unsafe_set


# =============================================================================
# Get all benchmark problems (including harder ones)
# =============================================================================

def get_hard_benchmark_problems() -> dict[str, tuple[DynamicsModel, Set, Set, float]]:
    """Get harder benchmark problems for stress testing.

    These problems have:
    - Higher dimensions (8-10D)
    - Longer time horizons (10-20s)
    - More complex dynamics (chaotic, stiff)

    Returns:
        Dict mapping problem name to (dynamics, initial_set, unsafe_set, time_horizon)
    """
    return {
        'coupled_8d': (*create_high_dim_coupled_8d(), 15.0),
        'diagonal_10d': (*create_high_dim_diagonal_10d(), 10.0),
        'stiff_vdp_2d': (*create_stiff_vanderpol_2d(mu=5.0), 20.0),
        'rossler_3d': (*create_rossler_3d(), 50.0),
        'double_pendulum_4d': (*create_double_pendulum_4d(), 10.0),
        'lotka_volterra_4d': (*create_lotka_volterra_4d(), 20.0),
        'neural_net_6d': (*create_neural_network_6d(), 10.0),
    }


# Convenience functions for running ablation studies

def get_ablation_problems() -> dict[str, tuple[DynamicsModel, Set, Set, float]]:
    """Get all ablation benchmark problems with time horizons.

    Returns:
        Dict mapping problem name to (dynamics, initial_set, unsafe_set, time_horizon)
    """
    return {
        'diagonal_4d': (*create_diagonal_decay_4d(), 3.0),
        'oscillator_4d': (*create_coupled_oscillator_4d(), 5.0),
        'van_der_pol_2d': (*create_van_der_pol_2d(), 10.0),
        'lorenz_3d': (*create_lorenz_like_3d(), 5.0),
        'diagonal_6d': (*create_high_dim_diagonal_6d(), 2.0),
    }


def run_full_ablation_study(
    problems: list[str] = None,
    verbose: bool = True,
    n_samples: int = 200,
    save_path: Optional[Path | str] = None,
) -> dict[str, list[AblationResult]]:
    """Run full ablation study on specified problems.

    Args:
        problems: List of problem names to test. If None, uses a subset for ~10 min runtime.
        verbose: Whether to print progress.
        n_samples: Number of samples for verification.
        save_path: Optional path to save results.

    Returns:
        Dict mapping ablation type to list of AblationResults.
    """
    all_problems = get_ablation_problems()

    if problems is None:
        # Default subset for ~10 min runtime
        problems = ['diagonal_4d', 'van_der_pol_2d', 'lorenz_3d']

    if verbose:
        print("\n" + "=" * 60)
        print("Full Ablation Study")
        print("=" * 60)

    results = {
        'sampling': [],
        'spline_method': [],
        'sample_count': [],
        'optimization': [],
    }

    for problem_name in problems:
        if problem_name not in all_problems:
            print(f"Warning: Unknown problem '{problem_name}', skipping")
            continue

        dynamics, initial_set, unsafe_set, T = all_problems[problem_name]

        study = AblationStudy(
            n_samples=n_samples,
            time_horizon=T,
            verbose=verbose,
        )

        # Sampling ablation (all problems)
        result = study.run_sampling_ablation(
            dynamics, initial_set, unsafe_set, problem_name
        )
        results['sampling'].append(result)

        # Spline ablation (only 1D/2D problems)
        if initial_set.n_dims <= 2:
            result = study.run_spline_ablation(
                dynamics, initial_set, unsafe_set, problem_name
            )
            results['spline_method'].append(result)

        # Sample count ablation (selected problems)
        if problem_name in ['diagonal_4d', 'van_der_pol_2d']:
            result = study.run_sample_count_ablation(
                dynamics, initial_set, unsafe_set,
                sample_counts=[50, 100, 200, 400],
                problem_name=problem_name
            )
            results['sample_count'].append(result)

        # Optimization ablation (selected problems)
        if problem_name in ['diagonal_4d', 'van_der_pol_2d']:
            result = study.run_optimization_ablation(
                dynamics, initial_set, unsafe_set, problem_name
            )
            results['optimization'].append(result)

    # Save results
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            ablation_type: [r.to_dict() for r in ablation_results]
            for ablation_type, ablation_results in results.items()
        }
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        if verbose:
            print(f"\nResults saved to {save_path}")

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("Ablation Study Summary")
        print("=" * 60)
        for ablation_type, ablation_results in results.items():
            if ablation_results:
                print(f"\n{ablation_type.upper()}:")
                for r in ablation_results:
                    print(r.summary())
                    print()

    return results
