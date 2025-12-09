"""Sensitivity analysis for spline verification.

Provides tools for analyzing how verification results depend on:
- Spline parameters (kernel type, smoothing)
- Integration settings (tolerance, method)
- Sampling strategy (uniform, latin hypercube, etc.)
- Initial set geometry
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from pathlib import Path
import json
from itertools import product

import numpy as np

from ..dynamics import ODEDynamics
from ..geometry import HyperRectangle, Ball, Set
from ..verification import SafetyVerifier, VerificationStatus
from ..geometry import SamplingStrategy


@dataclass
class ParameterSweepResult:
    """Result of a single parameter configuration.

    Attributes:
        parameters: Dictionary of parameter values.
        status: Verification result.
        min_objective: Minimum of spline approximation.
        error_bound: Total error bound.
        time: Wall-clock time.
        metadata: Additional data.
    """
    parameters: dict
    status: VerificationStatus
    min_objective: float
    error_bound: float
    time: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'parameters': self.parameters,
            'status': self.status.name,
            'min_objective': self.min_objective,
            'error_bound': self.error_bound,
            'time': self.time,
            'metadata': self.metadata,
        }


@dataclass
class ParameterSweep:
    """Configuration for a parameter sweep.

    Attributes:
        name: Name of the sweep.
        parameters: Dictionary mapping parameter names to lists of values.
        description: Optional description of the sweep.
    """
    name: str
    parameters: dict[str, list]
    description: str = ""

    def get_combinations(self) -> list[dict]:
        """Generate all parameter combinations."""
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]

    @property
    def n_combinations(self) -> int:
        """Total number of parameter combinations."""
        n = 1
        for vals in self.parameters.values():
            n *= len(vals)
        return n


@dataclass
class SensitivityAnalysis:
    """Container for sensitivity analysis results.

    Attributes:
        name: Name of the analysis.
        sweep: Parameter sweep configuration.
        results: List of results for each parameter combination.
        baseline_params: Baseline parameter values for comparison.
        metadata: Analysis-level metadata.
    """
    name: str
    sweep: ParameterSweep
    results: list[ParameterSweepResult] = field(default_factory=list)
    baseline_params: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add_result(self, result: ParameterSweepResult):
        """Add a sweep result."""
        self.results.append(result)

    @property
    def baseline_result(self) -> Optional[ParameterSweepResult]:
        """Find result matching baseline parameters."""
        for result in self.results:
            if result.parameters == self.baseline_params:
                return result
        return None

    def sensitivity_to_parameter(self, param_name: str) -> dict:
        """Compute sensitivity of results to a specific parameter.

        Returns statistics on how the parameter affects results.
        """
        # Group results by this parameter value
        values = self.sweep.parameters.get(param_name, [])
        if not values:
            return {}

        by_value = {v: [] for v in values}
        for result in self.results:
            val = result.parameters.get(param_name)
            if val in by_value:
                by_value[val].append(result)

        # Compute statistics
        stats = {}
        for val, results in by_value.items():
            if results:
                times = [r.time for r in results]
                min_objs = [r.min_objective for r in results]
                errors = [r.error_bound for r in results]
                stats[val] = {
                    'n_runs': len(results),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'mean_min_objective': np.mean(min_objs),
                    'std_min_objective': np.std(min_objs),
                    'mean_error_bound': np.mean(errors),
                    'status_counts': {
                        s.name: sum(1 for r in results if r.status == s)
                        for s in VerificationStatus
                    }
                }
        return stats

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'sweep': {
                'name': self.sweep.name,
                'parameters': self.sweep.parameters,
                'description': self.sweep.description,
            },
            'results': [r.to_dict() for r in self.results],
            'baseline_params': self.baseline_params,
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
            f"Sensitivity Analysis: {self.name}",
            "=" * 50,
            f"Sweep: {self.sweep.name}",
            f"Parameters: {list(self.sweep.parameters.keys())}",
            f"Total combinations: {self.sweep.n_combinations}",
            f"Completed runs: {len(self.results)}",
            "",
        ]

        # Status breakdown
        status_counts = {'SAFE': 0, 'UNSAFE': 0, 'UNKNOWN': 0}
        for result in self.results:
            status_counts[result.status.name] += 1
        lines.append(f"Status counts: {status_counts}")

        # Time statistics
        if self.results:
            times = [r.time for r in self.results]
            lines.append(f"Time: min={min(times):.3f}s, max={max(times):.3f}s, "
                        f"mean={np.mean(times):.3f}s")

        # Per-parameter sensitivity
        for param in self.sweep.parameters.keys():
            stats = self.sensitivity_to_parameter(param)
            lines.append(f"\nSensitivity to {param}:")
            for val, s in stats.items():
                lines.append(f"  {val}: time={s['mean_time']:.3f}s, "
                           f"min_obj={s['mean_min_objective']:.4f}")

        return '\n'.join(lines)


def run_parameter_sweep(
    dynamics: ODEDynamics,
    initial_set: Set,
    unsafe_set: Set,
    time_horizon: float,
    sweep: ParameterSweep,
    seed: int = 42,
    verbose: bool = True,
) -> SensitivityAnalysis:
    """Run a parameter sweep for sensitivity analysis.

    Args:
        dynamics: System dynamics.
        initial_set: Initial condition set.
        unsafe_set: Unsafe region.
        time_horizon: Time horizon T.
        sweep: Parameter sweep configuration.
        seed: Random seed.
        verbose: Whether to print progress.

    Returns:
        SensitivityAnalysis with results.

    The sweep parameters can include:
        - n_samples: Number of samples
        - sampling_strategy: 'uniform', 'latin_hypercube', 'sobol', 'halton'
        - kernel: RBF kernel type ('thin_plate_spline', 'multiquadric', etc.)
        - smoothing: RBF smoothing parameter
    """
    analysis = SensitivityAnalysis(
        name=f"sensitivity_{sweep.name}",
        sweep=sweep,
        metadata={
            'time_horizon': time_horizon,
            'seed': seed,
        }
    )

    combinations = sweep.get_combinations()
    n_total = len(combinations)

    if verbose:
        print(f"\nRunning parameter sweep: {sweep.name}")
        print(f"Total combinations: {n_total}")
        print("=" * 50)

    for i, params in enumerate(combinations):
        if verbose:
            print(f"  [{i+1}/{n_total}] {params}...", end=' ', flush=True)

        # Extract parameters
        n_samples = params.get('n_samples', 100)
        sampling = params.get('sampling_strategy', 'latin_hypercube')
        kernel = params.get('kernel', 'thin_plate_spline')
        smoothing = params.get('smoothing', 0.0)

        # Map sampling strategy string to enum
        strategy_map = {
            'uniform': SamplingStrategy.UNIFORM,
            'latin_hypercube': SamplingStrategy.LATIN_HYPERCUBE,
            'sobol': SamplingStrategy.SOBOL,
            'halton': SamplingStrategy.HALTON,
        }
        strategy = strategy_map.get(sampling, SamplingStrategy.LATIN_HYPERCUBE)

        # Create verifier with parameters
        verifier = SafetyVerifier(
            n_samples=n_samples,
            sampling_strategy=strategy,
            seed=seed,
        )

        # Run verification
        start = time.perf_counter()
        result = verifier.verify(
            dynamics, initial_set, unsafe_set, time_horizon,
            kernel=kernel,
            smoothing=smoothing,
        )
        elapsed = time.perf_counter() - start

        # Store result
        sweep_result = ParameterSweepResult(
            parameters=params,
            status=result.status,
            min_objective=result.min_objective,
            error_bound=result.error_bound,
            time=elapsed,
        )
        analysis.add_result(sweep_result)

        if verbose:
            print(f"{result.status.name} ({elapsed:.3f}s)")

    if verbose:
        print("=" * 50)
        print(f"Sweep complete: {len(analysis.results)} runs")

    return analysis


def create_sampling_sweep() -> ParameterSweep:
    """Create a parameter sweep for sampling strategies."""
    return ParameterSweep(
        name="sampling_strategies",
        parameters={
            'n_samples': [50, 100, 200],
            'sampling_strategy': ['uniform', 'latin_hypercube', 'sobol', 'halton'],
        },
        description="Compare different sampling strategies and sample counts",
    )


def create_spline_sweep() -> ParameterSweep:
    """Create a parameter sweep for spline parameters."""
    return ParameterSweep(
        name="spline_parameters",
        parameters={
            'kernel': ['thin_plate_spline', 'multiquadric', 'inverse_multiquadric', 'gaussian'],
            'smoothing': [0.0, 0.01, 0.1],
        },
        description="Compare different RBF kernels and smoothing levels",
    )


def create_comprehensive_sweep() -> ParameterSweep:
    """Create a comprehensive parameter sweep."""
    return ParameterSweep(
        name="comprehensive",
        parameters={
            'n_samples': [100, 200],
            'sampling_strategy': ['latin_hypercube', 'sobol'],
            'kernel': ['thin_plate_spline', 'multiquadric'],
            'smoothing': [0.0, 0.01],
        },
        description="Comprehensive sweep of key parameters",
    )


def run_sampling_sensitivity(
    dimension: int = 2,
    time_horizon: float = 2.0,
    verbose: bool = True,
    save_path: Optional[Path | str] = None,
) -> SensitivityAnalysis:
    """Run sensitivity analysis on sampling strategies.

    Args:
        dimension: State space dimension.
        time_horizon: Time horizon for verification.
        verbose: Whether to print progress.
        save_path: Optional path to save results.

    Returns:
        SensitivityAnalysis with sampling comparison.
    """
    # Create test system
    A = np.array([[-0.5, 1.0], [-1.0, -0.5]]) if dimension == 2 else \
        -0.5 * np.eye(dimension)
    dynamics = ODEDynamics.from_matrix(A)

    center = 2.0 * np.ones(dimension)
    initial_set = HyperRectangle(lower=center - 0.5, upper=center + 0.5)
    unsafe_set = Ball(center=np.zeros(dimension), radius=0.3)

    sweep = create_sampling_sweep()
    analysis = run_parameter_sweep(
        dynamics, initial_set, unsafe_set, time_horizon,
        sweep=sweep,
        verbose=verbose,
    )

    if save_path:
        analysis.save(save_path)

    return analysis


def run_spline_sensitivity(
    dimension: int = 2,
    time_horizon: float = 2.0,
    verbose: bool = True,
    save_path: Optional[Path | str] = None,
) -> SensitivityAnalysis:
    """Run sensitivity analysis on spline parameters.

    Args:
        dimension: State space dimension.
        time_horizon: Time horizon for verification.
        verbose: Whether to print progress.
        save_path: Optional path to save results.

    Returns:
        SensitivityAnalysis with spline parameter comparison.
    """
    # Create test system
    A = np.array([[-0.5, 1.0], [-1.0, -0.5]]) if dimension == 2 else \
        -0.5 * np.eye(dimension)
    dynamics = ODEDynamics.from_matrix(A)

    center = 2.0 * np.ones(dimension)
    initial_set = HyperRectangle(lower=center - 0.5, upper=center + 0.5)
    unsafe_set = Ball(center=np.zeros(dimension), radius=0.3)

    sweep = create_spline_sweep()
    analysis = run_parameter_sweep(
        dynamics, initial_set, unsafe_set, time_horizon,
        sweep=sweep,
        verbose=verbose,
    )

    if save_path:
        analysis.save(save_path)

    return analysis
