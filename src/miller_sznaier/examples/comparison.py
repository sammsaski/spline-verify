"""Comparison framework: Spline-Verify vs Miller-Sznaier.

This module provides tools for comparing the two distance estimation methods:
1. Spline-Verify: Sample-based spline approximation (upper bounds)
2. Miller-Sznaier: Occupation measure SDP relaxation (lower bounds)

The comparison covers:
- Accuracy of bounds
- Runtime performance
- Scalability with problem parameters
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..problem import UnsafeSupport, create_flow_system, create_twist_system
from ..distance_estimator import DistanceEstimator, CVXPY_AVAILABLE

# Import spline-verify modules
from spline_verify.dynamics import ODEDynamics
from spline_verify.geometry import Ball, HyperRectangle
from spline_verify.verification import SafetyVerifier, VerificationStatus


@dataclass
class ComparisonResult:
    """Result of comparing spline-verify and Miller-Sznaier methods.

    Attributes:
        problem_name: Name of the test problem.
        # Spline-verify results
        spline_min: Minimum from spline approximation.
        spline_error: Error bound from spline method.
        spline_time: Runtime for spline method.
        spline_status: Verification status.
        # Miller-Sznaier results
        ms_lower: Lower bound from SDP.
        ms_upper: Upper bound from sampling.
        ms_time: Runtime for SDP method.
        ms_status: Solver status.
        # Comparison metrics
        gap: Gap between bounds (ms_upper - ms_lower or spline_min - ms_lower).
        metadata: Additional comparison data.
    """
    problem_name: str
    # Spline-verify
    spline_min: float = 0.0
    spline_error: float = 0.0
    spline_time: float = 0.0
    spline_status: str = ""
    # Miller-Sznaier
    ms_lower: float = 0.0
    ms_upper: Optional[float] = None
    ms_time: float = 0.0
    ms_status: str = ""
    # Metadata
    metadata: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        """Gap between bounds."""
        if self.ms_upper is not None:
            return self.ms_upper - self.ms_lower
        return self.spline_min - self.ms_lower

    def summary(self) -> str:
        """Return formatted comparison summary."""
        lines = [
            f"Comparison: {self.problem_name}",
            "=" * 50,
            "",
            "Spline-Verify (sample-based):",
            f"  Minimum F_T:     {self.spline_min:.6f}",
            f"  Error bound:     {self.spline_error:.6f}",
            f"  Safety margin:   {self.spline_min - self.spline_error:.6f}",
            f"  Runtime:         {self.spline_time:.3f}s",
            f"  Status:          {self.spline_status}",
            "",
            "Miller-Sznaier (SDP-based):",
            f"  Lower bound:     {self.ms_lower:.6f}",
        ]
        if self.ms_upper is not None:
            lines.append(f"  Upper bound:     {self.ms_upper:.6f}")
        lines.extend([
            f"  Runtime:         {self.ms_time:.3f}s",
            f"  Status:          {self.ms_status}",
            "",
            f"Gap between methods: {self.gap:.6f}",
        ])
        return '\n'.join(lines)


def convert_unsafe_support_to_spline_verify(
    problem: UnsafeSupport
) -> tuple[ODEDynamics, HyperRectangle, Ball]:
    """Convert UnsafeSupport to spline-verify types.

    Args:
        problem: UnsafeSupport problem specification.

    Returns:
        (dynamics, initial_set, unsafe_set) for spline-verify.
    """
    # Create ODEDynamics
    dynamics = ODEDynamics(f=problem.dynamics, _n_dims=problem.n_vars)

    # Create initial set (convert ball to bounding box)
    lower = problem.initial_center - problem.initial_radius
    upper = problem.initial_center + problem.initial_radius
    initial_set = HyperRectangle(lower=lower, upper=upper)

    # Create unsafe set (ball)
    unsafe_set = Ball(
        center=problem.unsafe_center,
        radius=problem.unsafe_radius
    )

    return dynamics, initial_set, unsafe_set


def compare_methods(
    problem: UnsafeSupport,
    problem_name: str = "test_problem",
    n_samples: int = 200,
    sdp_order: int = 4,
    verbose: bool = True,
) -> ComparisonResult:
    """Compare spline-verify and Miller-Sznaier on the same problem.

    Args:
        problem: UnsafeSupport problem specification.
        problem_name: Name for the comparison.
        n_samples: Number of samples for spline-verify.
        sdp_order: Moment relaxation order for SDP.
        verbose: Whether to print results.

    Returns:
        ComparisonResult with both methods' results.
    """
    if verbose:
        print(f"\nComparing methods on: {problem_name}")
        print("-" * 50)

    result = ComparisonResult(problem_name=problem_name)

    # Run spline-verify
    if verbose:
        print("Running spline-verify...")

    dynamics, initial_set, unsafe_set = convert_unsafe_support_to_spline_verify(problem)

    verifier = SafetyVerifier(n_samples=n_samples, seed=42)

    start = time.perf_counter()
    sv_result = verifier.verify(
        dynamics, initial_set, unsafe_set, problem.time_horizon
    )
    result.spline_time = time.perf_counter() - start

    result.spline_min = sv_result.min_objective
    result.spline_error = sv_result.error_bound
    result.spline_status = sv_result.status.name

    if verbose:
        print(f"  Min: {result.spline_min:.6f}, Error: {result.spline_error:.6f}")
        print(f"  Status: {result.spline_status}, Time: {result.spline_time:.3f}s")

    # Run Miller-Sznaier (if available)
    if CVXPY_AVAILABLE:
        if verbose:
            print("Running Miller-Sznaier SDP...")

        estimator = DistanceEstimator(order=sdp_order, verbose=False)

        start = time.perf_counter()
        ms_result = estimator.estimate(problem, compute_upper_bound=True)
        result.ms_time = time.perf_counter() - start

        result.ms_lower = ms_result.lower_bound
        result.ms_upper = ms_result.upper_bound
        result.ms_status = ms_result.status

        if verbose:
            print(f"  Lower: {result.ms_lower:.6f}, Upper: {result.ms_upper:.6f}")
            print(f"  Status: {result.ms_status}, Time: {result.ms_time:.3f}s")
    else:
        if verbose:
            print("  Skipped (cvxpy not available)")
        result.ms_status = "cvxpy_not_available"

    result.metadata = {
        'n_samples': n_samples,
        'sdp_order': sdp_order,
        'n_dims': problem.n_vars,
        'time_horizon': problem.time_horizon,
    }

    if verbose:
        print()
        print(result.summary())

    return result


def run_comparison_suite(
    verbose: bool = True,
    save_dir: Optional[Path] = None,
) -> list[ComparisonResult]:
    """Run comparison on multiple standard problems.

    Args:
        verbose: Whether to print results.
        save_dir: Directory to save results.

    Returns:
        List of ComparisonResults.
    """
    results = []

    if verbose:
        print("\n" + "=" * 60)
        print("Method Comparison Suite")
        print("=" * 60)

    # Flow system (2D)
    flow = create_flow_system(time_horizon=5.0)
    result = compare_methods(flow, "flow_2d", verbose=verbose)
    results.append(result)

    # Twist system (3D)
    twist = create_twist_system(time_horizon=5.0)
    result = compare_methods(twist, "twist_3d", verbose=verbose)
    results.append(result)

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"\n{'Problem':<15} {'Spline Min':>12} {'MS Lower':>12} {'Gap':>10} {'Time Ratio':>12}")
        print("-" * 60)
        for r in results:
            time_ratio = r.ms_time / r.spline_time if r.spline_time > 0 else 0
            print(f"{r.problem_name:<15} {r.spline_min:>12.6f} {r.ms_lower:>12.6f} "
                  f"{r.gap:>10.6f} {time_ratio:>12.2f}x")

    return results


def plot_comparison(
    results: list[ComparisonResult],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create comparison visualization.

    Args:
        results: List of comparison results.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract data
    names = [r.problem_name for r in results]
    spline_mins = [r.spline_min for r in results]
    ms_lowers = [r.ms_lower for r in results]
    spline_times = [r.spline_time for r in results]
    ms_times = [r.ms_time for r in results]

    x = np.arange(len(names))
    width = 0.35

    # Bounds comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, spline_mins, width, label='Spline-Verify (min)', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, ms_lowers, width, label='Miller-Sznaier (lower)', color='red', alpha=0.7)

    ax.set_ylabel('Distance')
    ax.set_title('Distance Bounds Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Runtime comparison
    ax = axes[1]
    bars1 = ax.bar(x - width/2, spline_times, width, label='Spline-Verify', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, ms_times, width, label='Miller-Sznaier', color='red', alpha=0.7)

    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def main():
    """Command-line interface for comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare spline-verify and Miller-Sznaier methods")
    parser.add_argument('--save', action='store_true', help='Save figures')
    parser.add_argument('--outdir', type=str, default='./figs/comparison',
                       help='Output directory')
    args = parser.parse_args()

    save_dir = Path(args.outdir) if args.save else None

    results = run_comparison_suite(verbose=True, save_dir=save_dir)

    if args.save:
        fig = plot_comparison(results, save_path=save_dir / "method_comparison.png")


if __name__ == '__main__':
    main()
