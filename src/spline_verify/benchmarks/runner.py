"""Benchmark runner infrastructure.

Provides classes for running systematic benchmarks and collecting results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Any, Optional
from pathlib import Path
import json

import numpy as np

from ..dynamics import DynamicsModel
from ..geometry import Set
from ..verification import SafetyVerifier, VerificationStatus, VerificationResult


@dataclass
class TimingResult:
    """Timing information for a benchmark run.

    Attributes:
        total_time: Total wall-clock time in seconds.
        simulation_time: Time spent simulating trajectories.
        fitting_time: Time spent fitting spline approximation.
        optimization_time: Time spent minimizing spline.
        n_trajectories: Number of trajectories simulated.
        n_samples: Number of initial conditions sampled.
    """
    total_time: float
    simulation_time: float = 0.0
    fitting_time: float = 0.0
    optimization_time: float = 0.0
    n_trajectories: int = 0
    n_samples: int = 0

    @property
    def time_per_trajectory(self) -> float:
        """Average time per trajectory simulation."""
        if self.n_trajectories == 0:
            return 0.0
        return self.simulation_time / self.n_trajectories

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'total_time': self.total_time,
            'simulation_time': self.simulation_time,
            'fitting_time': self.fitting_time,
            'optimization_time': self.optimization_time,
            'n_trajectories': self.n_trajectories,
            'n_samples': self.n_samples,
            'time_per_trajectory': self.time_per_trajectory,
        }


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes:
        name: Name/identifier of the benchmark.
        status: Verification result (SAFE/UNSAFE/UNKNOWN).
        min_objective: Minimum of spline approximation.
        error_bound: Total error bound epsilon.
        safety_margin: min_objective - error_bound.
        timing: Timing information.
        ground_truth_status: Expected status if known.
        is_correct: Whether result matches ground truth.
        metadata: Additional benchmark-specific data.
    """
    name: str
    status: VerificationStatus
    min_objective: float
    error_bound: float
    safety_margin: float
    timing: TimingResult
    ground_truth_status: Optional[VerificationStatus] = None
    is_correct: Optional[bool] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.ground_truth_status is not None:
            # UNKNOWN is always "correct" since we can't certify
            if self.status == VerificationStatus.UNKNOWN:
                self.is_correct = True
            else:
                self.is_correct = (self.status == self.ground_truth_status)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'status': self.status.name,
            'min_objective': self.min_objective,
            'error_bound': self.error_bound,
            'safety_margin': self.safety_margin,
            'timing': self.timing.to_dict(),
            'ground_truth_status': self.ground_truth_status.name if self.ground_truth_status else None,
            'is_correct': self.is_correct,
            'metadata': self.metadata,
        }

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"Benchmark: {self.name}",
            f"  Status: {self.status.name}",
            f"  Min F_T: {self.min_objective:.6f}",
            f"  Error bound: {self.error_bound:.6f}",
            f"  Safety margin: {self.safety_margin:.6f}",
            f"  Total time: {self.timing.total_time:.3f}s",
        ]
        if self.ground_truth_status is not None:
            lines.append(f"  Ground truth: {self.ground_truth_status.name}")
            lines.append(f"  Correct: {self.is_correct}")
        return '\n'.join(lines)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results.

    Attributes:
        name: Name of the benchmark suite.
        results: List of individual benchmark results.
        metadata: Suite-level metadata.
    """
    name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to the suite."""
        self.results.append(result)

    @property
    def n_benchmarks(self) -> int:
        """Number of benchmarks in suite."""
        return len(self.results)

    @property
    def n_correct(self) -> int:
        """Number of correct results (matching ground truth)."""
        return sum(1 for r in self.results if r.is_correct is True)

    @property
    def n_incorrect(self) -> int:
        """Number of incorrect results."""
        return sum(1 for r in self.results if r.is_correct is False)

    @property
    def accuracy(self) -> float:
        """Accuracy rate (correct / total with ground truth)."""
        with_gt = [r for r in self.results if r.ground_truth_status is not None]
        if not with_gt:
            return float('nan')
        return self.n_correct / len(with_gt)

    @property
    def total_time(self) -> float:
        """Total time for all benchmarks."""
        return sum(r.timing.total_time for r in self.results)

    @property
    def status_counts(self) -> dict[str, int]:
        """Count of each verification status."""
        counts = {'SAFE': 0, 'UNSAFE': 0, 'UNKNOWN': 0}
        for r in self.results:
            counts[r.status.name] += 1
        return counts

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'n_benchmarks': self.n_benchmarks,
            'n_correct': self.n_correct,
            'n_incorrect': self.n_incorrect,
            'accuracy': self.accuracy if not np.isnan(self.accuracy) else None,
            'total_time': self.total_time,
            'status_counts': self.status_counts,
            'results': [r.to_dict() for r in self.results],
            'metadata': self.metadata,
        }

    def save(self, path: Path | str):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> BenchmarkSuite:
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)

        suite = cls(name=data['name'], metadata=data.get('metadata', {}))
        for r in data['results']:
            # Filter out computed properties from timing dict
            timing_data = {k: v for k, v in r['timing'].items()
                         if k != 'time_per_trajectory'}
            timing = TimingResult(**timing_data)
            result = BenchmarkResult(
                name=r['name'],
                status=VerificationStatus[r['status']],
                min_objective=r['min_objective'],
                error_bound=r['error_bound'],
                safety_margin=r['safety_margin'],
                timing=timing,
                ground_truth_status=VerificationStatus[r['ground_truth_status']] if r['ground_truth_status'] else None,
                is_correct=r['is_correct'],
                metadata=r.get('metadata', {}),
            )
            suite.add_result(result)
        return suite

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"Benchmark Suite: {self.name}",
            f"=" * 50,
            f"Total benchmarks: {self.n_benchmarks}",
            f"Status counts: {self.status_counts}",
            f"Correct: {self.n_correct}, Incorrect: {self.n_incorrect}",
            f"Accuracy: {self.accuracy:.1%}" if not np.isnan(self.accuracy) else "Accuracy: N/A",
            f"Total time: {self.total_time:.2f}s",
            "",
        ]
        for r in self.results:
            lines.append(r.summary())
            lines.append("")
        return '\n'.join(lines)


class BenchmarkRunner:
    """Runner for executing verification benchmarks.

    Provides utilities for:
    - Running single benchmarks with timing
    - Running benchmark suites
    - Comparing against ground truth
    - Collecting detailed metrics
    """

    def __init__(
        self,
        n_samples: int = 200,
        seed: int = 42,
        verbose: bool = True
    ):
        """Initialize benchmark runner.

        Args:
            n_samples: Default number of samples for verification.
            seed: Random seed for reproducibility.
            verbose: Whether to print progress.
        """
        self.n_samples = n_samples
        self.seed = seed
        self.verbose = verbose

    def run_single(
        self,
        name: str,
        dynamics: DynamicsModel,
        initial_set: Set,
        unsafe_set: Set,
        time_horizon: float,
        ground_truth: Optional[VerificationStatus] = None,
        n_samples: Optional[int] = None,
        **kwargs
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            name: Benchmark name/identifier.
            dynamics: System dynamics.
            initial_set: Initial condition set.
            unsafe_set: Unsafe region.
            time_horizon: Time horizon T.
            ground_truth: Expected result if known.
            n_samples: Override default sample count.
            **kwargs: Additional arguments passed to verifier.

        Returns:
            BenchmarkResult with timing and verification data.
        """
        n = n_samples or self.n_samples

        if self.verbose:
            print(f"Running benchmark: {name}...", end=' ', flush=True)

        # Create verifier
        verifier = SafetyVerifier(n_samples=n, seed=self.seed)

        # Time the verification
        start_time = time.perf_counter()
        result = verifier.verify(dynamics, initial_set, unsafe_set, time_horizon)
        total_time = time.perf_counter() - start_time

        # Create timing result (detailed timing would require instrumenting verifier)
        timing = TimingResult(
            total_time=total_time,
            n_samples=n,
            n_trajectories=n,  # One trajectory per sample
        )

        # Create benchmark result
        benchmark_result = BenchmarkResult(
            name=name,
            status=result.status,
            min_objective=result.min_objective,
            error_bound=result.error_bound,
            safety_margin=result.safety_margin,
            timing=timing,
            ground_truth_status=ground_truth,
            metadata={
                'time_horizon': time_horizon,
                'n_dims': initial_set.n_dims,
            }
        )

        if self.verbose:
            status_str = result.status.name
            if ground_truth is not None:
                correct = "OK" if benchmark_result.is_correct else "WRONG"
                print(f"{status_str} ({correct}) [{total_time:.2f}s]")
            else:
                print(f"{status_str} [{total_time:.2f}s]")

        return benchmark_result

    def run_suite(
        self,
        suite_name: str,
        benchmarks: list[dict],
        **kwargs
    ) -> BenchmarkSuite:
        """Run a suite of benchmarks.

        Args:
            suite_name: Name for the benchmark suite.
            benchmarks: List of benchmark specifications, each a dict with keys:
                - name: Benchmark name
                - dynamics: DynamicsModel
                - initial_set: Set
                - unsafe_set: Set
                - time_horizon: float
                - ground_truth: Optional[VerificationStatus]
            **kwargs: Additional arguments passed to run_single.

        Returns:
            BenchmarkSuite with all results.
        """
        suite = BenchmarkSuite(name=suite_name)

        if self.verbose:
            print(f"\nRunning benchmark suite: {suite_name}")
            print("=" * 50)

        for spec in benchmarks:
            result = self.run_single(**spec, **kwargs)
            suite.add_result(result)

        if self.verbose:
            print("=" * 50)
            print(f"Suite complete: {suite.n_benchmarks} benchmarks in {suite.total_time:.2f}s")
            print(f"Status counts: {suite.status_counts}")
            if not np.isnan(suite.accuracy):
                print(f"Accuracy: {suite.accuracy:.1%}")

        return suite

    def run_with_varying_samples(
        self,
        name: str,
        dynamics: DynamicsModel,
        initial_set: Set,
        unsafe_set: Set,
        time_horizon: float,
        sample_counts: list[int],
        ground_truth: Optional[VerificationStatus] = None,
    ) -> list[BenchmarkResult]:
        """Run benchmark with varying sample counts.

        Useful for analyzing convergence and sample efficiency.

        Args:
            name: Base benchmark name.
            dynamics: System dynamics.
            initial_set: Initial condition set.
            unsafe_set: Unsafe region.
            time_horizon: Time horizon T.
            sample_counts: List of sample counts to try.
            ground_truth: Expected result if known.

        Returns:
            List of BenchmarkResult, one per sample count.
        """
        results = []
        for n in sample_counts:
            result = self.run_single(
                name=f"{name}_n{n}",
                dynamics=dynamics,
                initial_set=initial_set,
                unsafe_set=unsafe_set,
                time_horizon=time_horizon,
                ground_truth=ground_truth,
                n_samples=n,
            )
            results.append(result)
        return results
