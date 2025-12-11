#!/usr/bin/env python3
"""Run all spline-verify benchmarks.

This script runs the complete benchmark suite including:
1. Ground truth validation (analytical solutions)
2. Scalability analysis (dimension, samples, time horizon)
3. Sensitivity analysis (parameter sweeps)
4. Ablation studies (component-wise analysis)

Usage:
    python scripts/run_benchmarks.py [--quick] [--save] [--outdir DIR]

Options:
    --quick     Run a quick subset of benchmarks (~2 min)
    --full      Run full benchmark suite (~10 min)
    --save      Save figures and results
    --outdir    Output directory (default: ./results/benchmarks)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spline_verify.benchmarks import (
    run_sampling_sensitivity,
    run_spline_sensitivity,
    validate_against_ground_truth,
    get_all_ground_truth_systems,
    run_dimension_scaling,
    run_sample_scaling,
    run_full_ablation_study,
    get_hard_benchmark_problems,
    get_ablation_problems,
)
from spline_verify.verification import SafetyVerifier


def run_ground_truth_validation(verbose: bool = True) -> dict[str, Any]:
    """Run ground truth validation on linear systems."""
    if verbose:
        print("\n" + "=" * 60)
        print("Ground Truth Validation")
        print("=" * 60)

    systems = get_all_ground_truth_systems()

    # validate_against_ground_truth expects a list and returns aggregate results
    result = validate_against_ground_truth(systems, verbose=False)

    if verbose:
        print(f"\nValidated {result['n_systems']} ground truth systems:")
        print(f"  Correct: {result['n_correct']}/{result['n_systems']}")
        print(f"  Accuracy: {result['accuracy']*100:.1f}%")
        print(f"  Total time: {result['total_time']:.2f}s")

        if 'per_system' in result:
            for name, sys_result in result['per_system'].items():
                status = "✓" if sys_result.get('correct', False) else "✗"
                print(f"    {status} {name}")

    return result


def run_scalability_analysis(
    max_dim: int = 6,
    max_samples: int = 400,
    verbose: bool = True
) -> dict[str, Any]:
    """Run scalability analysis."""
    if verbose:
        print("\n" + "=" * 60)
        print("Scalability Analysis")
        print("=" * 60)

    results = {}

    # Dimension scaling
    if verbose:
        print("\nDimension scaling (2D to {}D)...".format(max_dim))
    start = time.perf_counter()
    dimensions = list(range(2, max_dim + 1))
    dim_result = run_dimension_scaling(dimensions=dimensions, verbose=False)
    results['dimension'] = {
        'dimensions': dim_result.parameter_values,
        'times': dim_result.times,
        'statuses': [s.name for s in dim_result.statuses],
    }
    if verbose:
        print(f"  Completed in {time.perf_counter() - start:.1f}s")
        for i, dim in enumerate(dim_result.parameter_values):
            print(f"    {dim}D: {dim_result.times[i]:.2f}s, {dim_result.statuses[i].name}")

    # Sample count scaling
    if verbose:
        print(f"\nSample count scaling (50 to {max_samples})...")
    start = time.perf_counter()
    sample_counts = [50, 100, 200, max_samples]
    sample_result = run_sample_scaling(sample_counts=sample_counts, verbose=False)
    results['samples'] = {
        'counts': sample_counts,
        'times': sample_result.times,
        'min_objectives': sample_result.min_objectives,
    }
    if verbose:
        print(f"  Completed in {time.perf_counter() - start:.1f}s")
        for i, count in enumerate(sample_counts):
            print(f"    n={count}: {sample_result.times[i]:.2f}s, min={sample_result.min_objectives[i]:.4f}")

    return results


def run_sensitivity_analysis(verbose: bool = True) -> dict[str, Any]:
    """Run sensitivity analysis on key parameters."""
    if verbose:
        print("\n" + "=" * 60)
        print("Sensitivity Analysis")
        print("=" * 60)

    results = {}

    # Sampling strategy sensitivity
    if verbose:
        print("\nSampling strategy comparison...")
    start = time.perf_counter()
    sampling_analysis = run_sampling_sensitivity(verbose=False)
    # Extract results from SensitivityAnalysis object
    results['sampling'] = {
        'parameters': [r.parameters for r in sampling_analysis.results],
        'min_objectives': [r.min_objective for r in sampling_analysis.results],
        'times': [r.time for r in sampling_analysis.results],
    }
    if verbose:
        print(f"  Completed in {time.perf_counter() - start:.1f}s")
        for r in sampling_analysis.results[:4]:  # Show first 4
            print(f"    {r.parameters}: min={r.min_objective:.4f}, time={r.time:.2f}s")

    # Spline kernel sensitivity
    if verbose:
        print("\nSpline kernel comparison...")
    start = time.perf_counter()
    spline_analysis = run_spline_sensitivity(verbose=False)
    results['spline'] = {
        'parameters': [r.parameters for r in spline_analysis.results],
        'min_objectives': [r.min_objective for r in spline_analysis.results],
        'error_bounds': [r.error_bound for r in spline_analysis.results],
    }
    if verbose:
        print(f"  Completed in {time.perf_counter() - start:.1f}s")
        for r in spline_analysis.results[:4]:  # Show first 4
            print(f"    {r.parameters}: min={r.min_objective:.4f}, error={r.error_bound:.4f}")

    return results


def run_hard_benchmarks(
    n_samples: int = 200,
    verbose: bool = True
) -> dict[str, Any]:
    """Run challenging benchmark problems for stress testing."""
    if verbose:
        print("\n" + "=" * 60)
        print("Hard Benchmark Problems (Stress Testing)")
        print("=" * 60)

    problems = get_hard_benchmark_problems()
    results = {}

    total_start = time.perf_counter()
    for name, (dynamics, initial_set, unsafe_set, T) in problems.items():
        if verbose:
            print(f"\n{name} ({dynamics.n_dims}D, T={T})...")

        try:
            verifier = SafetyVerifier(n_samples=n_samples)
            start = time.perf_counter()
            result = verifier.verify(dynamics, initial_set, unsafe_set, T)
            elapsed = time.perf_counter() - start

            results[name] = {
                'n_dims': dynamics.n_dims,
                'time_horizon': T,
                'n_samples': n_samples,
                'status': result.status.name,
                'min_objective': float(result.min_objective),
                'error_bound': float(result.error_bound),
                'runtime': elapsed,
            }

            if verbose:
                print(f"  Status: {result.status.name}")
                print(f"  Min: {result.min_objective:.6f}")
                print(f"  Error: {result.error_bound:.6f}")
                print(f"  Time: {elapsed:.2f}s")

        except Exception as e:
            results[name] = {
                'n_dims': dynamics.n_dims,
                'time_horizon': T,
                'error': str(e),
            }
            if verbose:
                print(f"  ERROR: {e}")

    total_time = time.perf_counter() - total_start
    results['_summary'] = {
        'total_time': total_time,
        'n_problems': len(problems),
        'n_successful': sum(1 for k, v in results.items() if k != '_summary' and 'error' not in v),
    }

    if verbose:
        print(f"\nHard benchmarks total time: {total_time:.1f}s")

    return results


def run_standard_benchmarks(
    n_samples: int = 200,
    verbose: bool = True
) -> dict[str, Any]:
    """Run standard ablation problems for baseline comparison."""
    if verbose:
        print("\n" + "=" * 60)
        print("Standard Benchmark Problems (Baseline)")
        print("=" * 60)

    problems = get_ablation_problems()
    results = {}

    total_start = time.perf_counter()
    for name, (dynamics, initial_set, unsafe_set, T) in problems.items():
        if verbose:
            print(f"\n{name} ({dynamics.n_dims}D, T={T})...")

        try:
            verifier = SafetyVerifier(n_samples=n_samples)
            start = time.perf_counter()
            result = verifier.verify(dynamics, initial_set, unsafe_set, T)
            elapsed = time.perf_counter() - start

            results[name] = {
                'n_dims': dynamics.n_dims,
                'time_horizon': T,
                'n_samples': n_samples,
                'status': result.status.name,
                'min_objective': float(result.min_objective),
                'error_bound': float(result.error_bound),
                'runtime': elapsed,
            }

            if verbose:
                print(f"  Status: {result.status.name}")
                print(f"  Min: {result.min_objective:.6f}")
                print(f"  Error: {result.error_bound:.6f}")
                print(f"  Time: {elapsed:.2f}s")

        except Exception as e:
            results[name] = {
                'n_dims': dynamics.n_dims,
                'time_horizon': T,
                'error': str(e),
            }
            if verbose:
                print(f"  ERROR: {e}")

    total_time = time.perf_counter() - total_start
    results['_summary'] = {
        'total_time': total_time,
        'n_problems': len(problems),
        'n_successful': sum(1 for k, v in results.items() if k != '_summary' and 'error' not in v),
    }

    if verbose:
        print(f"\nStandard benchmarks total time: {total_time:.1f}s")

    return results


def run_ablation_studies(
    quick: bool = False,
    verbose: bool = True
) -> dict[str, Any]:
    """Run ablation studies on pipeline components."""
    if verbose:
        print("\n" + "=" * 60)
        print("Ablation Studies")
        print("=" * 60)

    # Use the convenience function that sets up test problems
    if quick:
        problems = ['van_der_pol_2d']  # Quick: just one problem
    else:
        problems = ['diagonal_4d', 'van_der_pol_2d', 'lorenz_3d']

    start = time.perf_counter()
    ablation_results = run_full_ablation_study(
        problems=problems,
        verbose=verbose,
        n_samples=100 if quick else 200,
    )

    # Convert to serializable format
    results = {}
    for ablation_type, result_list in ablation_results.items():
        results[ablation_type] = {
            'problems': [r.problem_name for r in result_list],
            'baseline_mins': [r.baseline_result.min_objective for r in result_list],
            'n_variants': [len(r.ablations) for r in result_list],
        }

    if verbose:
        print(f"\n  Total ablation time: {time.perf_counter() - start:.1f}s")

    return results


def plot_scalability_results(
    results: dict[str, Any],
    save_path: Path | None = None
) -> plt.Figure:
    """Plot scalability analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dimension scaling
    ax = axes[0]
    dims = results['dimension']['dimensions']
    times = results['dimension']['times']
    ax.plot(dims, times, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('State Dimension', fontsize=12)
    ax.set_ylabel('Runtime (s)', fontsize=12)
    ax.set_title('Dimension Scaling', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Sample count scaling
    ax = axes[1]
    counts = results['samples']['counts']
    times = results['samples']['times']
    ax.plot(counts, times, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Sample Count', fontsize=12)
    ax.set_ylabel('Runtime (s)', fontsize=12)
    ax.set_title('Sample Count Scaling', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scalability plot to {save_path}")

    return fig


def plot_ablation_results(
    results: dict[str, Any],
    save_path: Path | None = None
) -> plt.Figure:
    """Plot ablation study results."""
    n_plots = len(results)
    if n_plots == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, 'No ablation results', ha='center', va='center')
        return fig

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        # Handle different data structures from ablation results
        # The ablation results have 'problems' and 'baseline_mins'
        problems = data.get('problems', [])
        baseline_mins = data.get('baseline_mins', [])
        n_variants = data.get('n_variants', [])

        if problems and baseline_mins:
            x = np.arange(len(problems))
            bars = ax.bar(x, baseline_mins, color='steelblue', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([str(p) for p in problems], rotation=45, ha='right')
            ax.set_ylabel('Baseline Min Objective')
            # Add variant count as text on bars
            for i, (bar, nv) in enumerate(zip(bars, n_variants)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{nv} variants', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'No data for {name}', ha='center', va='center',
                   transform=ax.transAxes)

        ax.set_title(f'{name.replace("_", " ").title()} Ablation')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ablation plot to {save_path}")

    return fig


def plot_hard_benchmark_results(
    results: dict[str, Any],
    save_path: Path | None = None
) -> plt.Figure:
    """Plot hard benchmark results showing runtime and accuracy."""
    # Filter out summary
    problems = {k: v for k, v in results.items() if k != '_summary' and 'error' not in v}

    if not problems:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No hard benchmark results', ha='center', va='center')
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = list(problems.keys())
    runtimes = [problems[n]['runtime'] for n in names]
    n_dims = [problems[n]['n_dims'] for n in names]
    min_objs = [problems[n]['min_objective'] for n in names]
    errors = [problems[n]['error_bound'] for n in names]
    statuses = [problems[n]['status'] for n in names]

    # Color by status
    colors = ['green' if s == 'SAFE' else 'red' if s == 'UNSAFE' else 'orange' for s in statuses]

    # Runtime vs dimension
    ax = axes[0]
    ax.scatter(n_dims, runtimes, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax.annotate(name.replace('_', '\n'), (n_dims[i], runtimes[i]),
                   fontsize=7, ha='center', va='bottom')
    ax.set_xlabel('State Dimension', fontsize=12)
    ax.set_ylabel('Runtime (s)', fontsize=12)
    ax.set_title('Runtime vs Dimension', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Min objective with error bars
    ax = axes[1]
    x = np.arange(len(names))
    ax.bar(x, min_objs, yerr=errors, color=colors, alpha=0.7, capsize=3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Min Objective ± Error', fontsize=12)
    ax.set_title('Safety Margin by Problem', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Runtime breakdown
    ax = axes[2]
    sorted_idx = np.argsort(runtimes)[::-1]
    sorted_names = [names[i] for i in sorted_idx]
    sorted_runtimes = [runtimes[i] for i in sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]
    ax.barh(range(len(sorted_names)), sorted_runtimes, color=sorted_colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels([n.replace('_', ' ') for n in sorted_names], fontsize=9)
    ax.set_xlabel('Runtime (s)', fontsize=12)
    ax.set_title('Runtime Ranking', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='SAFE'),
        Patch(facecolor='red', alpha=0.7, label='UNSAFE'),
        Patch(facecolor='orange', alpha=0.7, label='UNKNOWN'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved hard benchmark plot to {save_path}")

    return fig


def save_results(results: dict[str, Any], save_path: Path) -> None:
    """Save results to JSON file."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(save_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Saved results to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run spline-verify benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--quick', action='store_true',
                       help='Run quick subset of benchmarks (~2 min)')
    parser.add_argument('--full', action='store_true',
                       help='Run full benchmark suite (~10 min)')
    parser.add_argument('--hard', action='store_true',
                       help='Run only hard benchmark problems')
    parser.add_argument('--save', action='store_true',
                       help='Save figures and results')
    parser.add_argument('--outdir', type=str, default='./results/benchmarks',
                       help='Output directory')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Number of samples (default: 100 quick, 400 full)')
    args = parser.parse_args()

    # Default to quick if neither specified
    if not args.quick and not args.full and not args.hard:
        args.quick = True

    outdir = Path(args.outdir)
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Spline-Verify Benchmark Suite")
    print("=" * 60)
    mode = 'hard' if args.hard else ('quick' if args.quick else 'full')
    print(f"Mode: {mode}")
    if args.save:
        print(f"Output directory: {outdir}")

    all_results = {}
    total_start = time.perf_counter()

    # If --hard mode, only run hard benchmarks
    if args.hard:
        n_samples = args.n_samples or 400
        all_results['hard_benchmarks'] = run_hard_benchmarks(n_samples=n_samples)

        total_time = time.perf_counter() - total_start

        # Summary
        print("\n" + "=" * 60)
        print("Hard Benchmark Summary")
        print("=" * 60)
        summary = all_results['hard_benchmarks'].get('_summary', {})
        print(f"Problems: {summary.get('n_successful', 0)}/{summary.get('n_problems', 0)} successful")
        print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

        if args.save:
            save_results(all_results, outdir / 'hard_benchmark_results.json')
            plot_hard_benchmark_results(
                all_results['hard_benchmarks'],
                save_path=outdir / 'hard_benchmarks.png'
            )

        print("\nHard benchmarks complete!")
        return

    # Standard benchmark suite
    # 1. Ground truth validation
    all_results['ground_truth'] = run_ground_truth_validation()

    # 2. Scalability analysis
    max_dim = 4 if args.quick else 8
    max_samples = 200 if args.quick else 800
    all_results['scalability'] = run_scalability_analysis(
        max_dim=max_dim, max_samples=max_samples
    )

    # 3. Sensitivity analysis
    all_results['sensitivity'] = run_sensitivity_analysis()

    # 4. Ablation studies
    all_results['ablation'] = run_ablation_studies(quick=args.quick)

    # 5. Standard benchmark problems (baseline)
    n_samples = args.n_samples or (100 if args.quick else 200)
    all_results['standard_benchmarks'] = run_standard_benchmarks(n_samples=n_samples)

    # 6. Hard benchmark problems (full mode only)
    if args.full:
        n_samples = args.n_samples or 400
        all_results['hard_benchmarks'] = run_hard_benchmarks(n_samples=n_samples)

    total_time = time.perf_counter() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print(f"Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save results and figures
    if args.save:
        save_results(all_results, outdir / 'benchmark_results.json')
        plot_scalability_results(
            all_results['scalability'],
            save_path=outdir / 'scalability.png'
        )
        plot_ablation_results(
            all_results['ablation'],
            save_path=outdir / 'ablation.png'
        )
        if 'hard_benchmarks' in all_results:
            plot_hard_benchmark_results(
                all_results['hard_benchmarks'],
                save_path=outdir / 'hard_benchmarks.png'
            )

    print("\nBenchmarks complete!")


if __name__ == '__main__':
    main()
