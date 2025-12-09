"""Scalability analysis for spline verification.

Provides tools for analyzing how verification performance scales with:
- State space dimension
- Number of samples
- Time horizon
- Number of switching modes
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from pathlib import Path
import json

import numpy as np

from ..dynamics import ODEDynamics, DynamicsModel
from ..geometry import HyperRectangle, Ball, Set
from ..verification import SafetyVerifier, VerificationStatus
from .runner import BenchmarkRunner, BenchmarkResult, TimingResult


@dataclass
class ScalabilityResult:
    """Result of a scalability experiment.

    Attributes:
        parameter_name: Name of the parameter being varied.
        parameter_values: Values of the parameter tested.
        times: Wall-clock time for each parameter value.
        statuses: Verification status for each run.
        min_objectives: Minimum objective values.
        error_bounds: Error bounds for each run.
        metadata: Additional experiment data.
    """
    parameter_name: str
    parameter_values: list
    times: list[float]
    statuses: list[VerificationStatus]
    min_objectives: list[float]
    error_bounds: list[float]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'parameter_name': self.parameter_name,
            'parameter_values': [float(v) if isinstance(v, (int, float, np.number)) else v
                                 for v in self.parameter_values],
            'times': self.times,
            'statuses': [s.name for s in self.statuses],
            'min_objectives': self.min_objectives,
            'error_bounds': self.error_bounds,
            'metadata': self.metadata,
        }

    def save(self, path: Path | str):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> ScalabilityResult:
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            parameter_name=data['parameter_name'],
            parameter_values=data['parameter_values'],
            times=data['times'],
            statuses=[VerificationStatus[s] for s in data['statuses']],
            min_objectives=data['min_objectives'],
            error_bounds=data['error_bounds'],
            metadata=data.get('metadata', {}),
        )


@dataclass
class ScalabilityAnalysis:
    """Container for multiple scalability experiments.

    Attributes:
        name: Name of the analysis.
        results: Dictionary of parameter name to ScalabilityResult.
        metadata: Analysis-level metadata.
    """
    name: str
    results: dict[str, ScalabilityResult] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add_result(self, result: ScalabilityResult):
        """Add a scalability result."""
        self.results[result.parameter_name] = result

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'metadata': self.metadata,
        }

    def save(self, path: Path | str):
        """Save analysis to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            f"Scalability Analysis: {self.name}",
            "=" * 50,
        ]
        for param_name, result in self.results.items():
            lines.append(f"\n{param_name}:")
            for i, val in enumerate(result.parameter_values):
                lines.append(
                    f"  {val}: {result.times[i]:.3f}s "
                    f"({result.statuses[i].name})"
                )
        return '\n'.join(lines)


def _create_n_dim_linear_system(n: int, decay_rate: float = 0.5) -> ODEDynamics:
    """Create an n-dimensional stable linear system.

    Args:
        n: State dimension.
        decay_rate: Decay rate for eigenvalues.

    Returns:
        Linear ODEDynamics with stable behavior.
    """
    # Create a stable matrix with specified decay
    A = -decay_rate * np.eye(n)
    # Add some off-diagonal coupling
    for i in range(n - 1):
        A[i, i + 1] = 0.3
        A[i + 1, i] = -0.3
    return ODEDynamics.from_matrix(A)


def run_dimension_scaling(
    dimensions: list[int],
    n_samples: int = 100,
    time_horizon: float = 2.0,
    seed: int = 42,
    verbose: bool = True,
) -> ScalabilityResult:
    """Analyze how verification time scales with state dimension.

    Creates stable linear systems of varying dimension and measures
    verification time.

    Args:
        dimensions: List of dimensions to test.
        n_samples: Number of samples for verification.
        time_horizon: Time horizon for verification.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress.

    Returns:
        ScalabilityResult with timing data.
    """
    times = []
    statuses = []
    min_objectives = []
    error_bounds = []

    if verbose:
        print(f"Running dimension scaling: {dimensions}")

    for n in dimensions:
        if verbose:
            print(f"  Dimension {n}...", end=' ', flush=True)

        # Create n-dimensional system
        dynamics = _create_n_dim_linear_system(n)

        # Initial set: unit cube centered at (2, ..., 2)
        center = 2.0 * np.ones(n)
        initial_set = HyperRectangle(
            lower=center - 0.5,
            upper=center + 0.5
        )

        # Unsafe set: ball at origin
        unsafe_set = Ball(center=np.zeros(n), radius=0.3)

        # Run verification
        verifier = SafetyVerifier(n_samples=n_samples, seed=seed)
        start = time.perf_counter()
        result = verifier.verify(dynamics, initial_set, unsafe_set, time_horizon)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        statuses.append(result.status)
        min_objectives.append(result.min_objective)
        error_bounds.append(result.error_bound)

        if verbose:
            print(f"{elapsed:.3f}s ({result.status.name})")

    return ScalabilityResult(
        parameter_name='dimension',
        parameter_values=dimensions,
        times=times,
        statuses=statuses,
        min_objectives=min_objectives,
        error_bounds=error_bounds,
        metadata={
            'n_samples': n_samples,
            'time_horizon': time_horizon,
            'seed': seed,
        }
    )


def run_sample_scaling(
    sample_counts: list[int],
    dimension: int = 2,
    time_horizon: float = 2.0,
    seed: int = 42,
    verbose: bool = True,
) -> ScalabilityResult:
    """Analyze how verification time and accuracy scale with sample count.

    Args:
        sample_counts: List of sample counts to test.
        dimension: State space dimension.
        time_horizon: Time horizon for verification.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress.

    Returns:
        ScalabilityResult with timing data.
    """
    times = []
    statuses = []
    min_objectives = []
    error_bounds = []

    if verbose:
        print(f"Running sample scaling: {sample_counts}")

    # Create fixed system
    dynamics = _create_n_dim_linear_system(dimension)
    center = 2.0 * np.ones(dimension)
    initial_set = HyperRectangle(lower=center - 0.5, upper=center + 0.5)
    unsafe_set = Ball(center=np.zeros(dimension), radius=0.3)

    for n_samples in sample_counts:
        if verbose:
            print(f"  {n_samples} samples...", end=' ', flush=True)

        verifier = SafetyVerifier(n_samples=n_samples, seed=seed)
        start = time.perf_counter()
        result = verifier.verify(dynamics, initial_set, unsafe_set, time_horizon)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        statuses.append(result.status)
        min_objectives.append(result.min_objective)
        error_bounds.append(result.error_bound)

        if verbose:
            print(f"{elapsed:.3f}s ({result.status.name}, min={result.min_objective:.4f})")

    return ScalabilityResult(
        parameter_name='n_samples',
        parameter_values=sample_counts,
        times=times,
        statuses=statuses,
        min_objectives=min_objectives,
        error_bounds=error_bounds,
        metadata={
            'dimension': dimension,
            'time_horizon': time_horizon,
            'seed': seed,
        }
    )


def run_time_horizon_scaling(
    time_horizons: list[float],
    dimension: int = 2,
    n_samples: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> ScalabilityResult:
    """Analyze how verification scales with time horizon.

    Longer time horizons require more integration steps and may affect
    error bounds.

    Args:
        time_horizons: List of time horizons to test.
        dimension: State space dimension.
        n_samples: Number of samples for verification.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress.

    Returns:
        ScalabilityResult with timing data.
    """
    times = []
    statuses = []
    min_objectives = []
    error_bounds = []

    if verbose:
        print(f"Running time horizon scaling: {time_horizons}")

    # Create fixed system
    dynamics = _create_n_dim_linear_system(dimension)
    center = 2.0 * np.ones(dimension)
    initial_set = HyperRectangle(lower=center - 0.5, upper=center + 0.5)
    unsafe_set = Ball(center=np.zeros(dimension), radius=0.3)

    for T in time_horizons:
        if verbose:
            print(f"  T={T}...", end=' ', flush=True)

        verifier = SafetyVerifier(n_samples=n_samples, seed=seed)
        start = time.perf_counter()
        result = verifier.verify(dynamics, initial_set, unsafe_set, T)
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        statuses.append(result.status)
        min_objectives.append(result.min_objective)
        error_bounds.append(result.error_bound)

        if verbose:
            print(f"{elapsed:.3f}s ({result.status.name})")

    return ScalabilityResult(
        parameter_name='time_horizon',
        parameter_values=time_horizons,
        times=times,
        statuses=statuses,
        min_objectives=min_objectives,
        error_bounds=error_bounds,
        metadata={
            'dimension': dimension,
            'n_samples': n_samples,
            'seed': seed,
        }
    )


def run_full_scalability_analysis(
    verbose: bool = True,
    save_path: Optional[Path | str] = None,
) -> ScalabilityAnalysis:
    """Run a comprehensive scalability analysis.

    Tests dimension scaling, sample scaling, and time horizon scaling
    with default parameter ranges.

    Args:
        verbose: Whether to print progress.
        save_path: Optional path to save results.

    Returns:
        ScalabilityAnalysis with all results.
    """
    analysis = ScalabilityAnalysis(name="full_scalability_analysis")

    if verbose:
        print("\n" + "=" * 60)
        print("Full Scalability Analysis")
        print("=" * 60 + "\n")

    # Dimension scaling
    dim_result = run_dimension_scaling(
        dimensions=[2, 3, 4, 5, 6],
        n_samples=100,
        verbose=verbose,
    )
    analysis.add_result(dim_result)

    # Sample scaling
    sample_result = run_sample_scaling(
        sample_counts=[50, 100, 200, 400, 800],
        dimension=2,
        verbose=verbose,
    )
    analysis.add_result(sample_result)

    # Time horizon scaling
    time_result = run_time_horizon_scaling(
        time_horizons=[1.0, 2.0, 4.0, 8.0, 16.0],
        dimension=2,
        verbose=verbose,
    )
    analysis.add_result(time_result)

    if save_path:
        analysis.save(save_path)
        if verbose:
            print(f"\nResults saved to {save_path}")

    if verbose:
        print("\n" + analysis.summary())

    return analysis
