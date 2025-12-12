#!/usr/bin/env python3
"""Compare spline-verify vs Miller-Sznaier methods.

This script runs comprehensive comparison experiments between:
1. Spline-Verify: Sample-based spline approximation (upper bounds)
2. Miller-Sznaier: Occupation measure SDP relaxation (lower bounds)

The comparison covers:
- Multiple benchmark problems (Flow, Twist, custom)
- Accuracy of bounds (gap analysis)
- Runtime performance
- Scalability with problem parameters

Usage:
    python scripts/run_comparison.py [--quick] [--save] [--outdir DIR]

Options:
    --quick     Run quick comparison (~1 min)
    --full      Run comprehensive comparison (~5 min)
    --save      Save figures and results
    --outdir    Output directory (default: ./results/comparison)

Requirements:
    pip install cvxpy scs  # For Miller-Sznaier SDP
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

# Import Miller-Sznaier module (problem definitions always work)
from miller_sznaier.problem import (
    UnsafeSupport,
    create_flow_system,
    create_twist_system,
    create_moon_system,
)

# Check for cvxpy (optional for SDP solving)
try:
    from miller_sznaier import DistanceEstimator, CVXPY_AVAILABLE
    from miller_sznaier.examples.comparison import (
        compare_methods,
        ComparisonResult,
        run_comparison_suite,
        plot_comparison,
    )
except ImportError:
    CVXPY_AVAILABLE = False
    DistanceEstimator = None

from spline_verify.dynamics import ODEDynamics
from spline_verify.geometry import HyperRectangle, Ball, LevelSet
from spline_verify.verification import SafetyVerifier


def translate_problem_to_spline_verify(problem: UnsafeSupport):
    """Translate UnsafeSupport problem to spline-verify format.

    This ensures both methods use exactly the same problem definition.

    Key translations:
    - Initial set: HyperRectangle bounding the ball (for QMC sampling compatibility)
    - Unsafe set: Ball + constraints -> LevelSet for complex shapes
    - Distance: Euclidean distance to set boundary

    Note: We use a bounding HyperRectangle for the initial set because the
    sampling infrastructure requires bounds for Latin Hypercube sampling.
    This is slightly larger than the ball (by factor of sqrt(n)), making our
    verification more conservative (we check more initial conditions).

    Returns:
        (dynamics, initial_set, unsafe_set) for spline-verify
    """
    # Dynamics
    dynamics = ODEDynamics(f=problem.dynamics, _n_dims=problem.n_vars)

    # Initial set - use bounding HyperRectangle for QMC sampling compatibility
    # The bounding box is slightly larger than the ball, making verification
    # more conservative (checking more initial conditions than M-S)
    initial_set = HyperRectangle(
        lower=problem.initial_center - problem.initial_radius,
        upper=problem.initial_center + problem.initial_radius
    )

    # Unsafe set - handle constraints if present
    if problem.unsafe_constraints:
        # Create a LevelSet that combines ball + constraints
        # For a point to be "unsafe", it must be in ball AND satisfy all constraints
        def combined_unsafe_level(x):
            x = np.asarray(x)
            # Distance to ball boundary (negative inside, positive outside)
            dist_to_center = np.linalg.norm(x - problem.unsafe_center)
            ball_level = dist_to_center - problem.unsafe_radius  # <=0 means inside ball

            if ball_level > 0:
                # Outside ball, return distance to ball
                return ball_level

            # Inside ball - check constraints
            # Constraints are g(x) >= 0 for points IN the set
            # So if g(x) < 0, point is NOT in unsafe set
            for g in problem.unsafe_constraints:
                constraint_val = g(x)
                if constraint_val < 0:
                    # Violated constraint - outside unsafe set
                    # Return positive distance estimate
                    return -constraint_val  # Approximate distance

            # Inside ball and all constraints satisfied - inside unsafe set
            return ball_level  # Returns negative value

        unsafe_set = LevelSet(g=combined_unsafe_level, _n_dims=problem.n_vars)
    else:
        # Simple ball unsafe set
        unsafe_set = Ball(center=problem.unsafe_center, radius=problem.unsafe_radius)

    return dynamics, initial_set, unsafe_set


@dataclass
class ExtendedComparisonResult:
    """Extended comparison result with additional metrics."""
    problem_name: str
    n_dims: int
    time_horizon: float
    # Spline-verify results
    spline_min: float
    spline_error: float
    spline_lower: float  # min - error
    spline_time: float
    spline_status: str
    # Miller-Sznaier results
    ms_lower: float
    ms_upper: Optional[float]
    ms_time: float
    ms_status: str
    # Comparison metrics
    gap: float
    relative_gap: float
    speedup: float  # ms_time / spline_time

    def to_dict(self) -> dict:
        return asdict(self)


def create_custom_problems() -> list[tuple[str, UnsafeSupport]]:
    """Create additional custom problems for comparison."""
    problems = []

    # Simple linear decay (easy)
    def linear_decay(t, x):
        return -0.5 * x

    problems.append((
        "linear_decay_2d",
        UnsafeSupport(
            n_vars=2,
            time_horizon=3.0,
            initial_center=np.array([2.0, 2.0]),
            initial_radius=0.3,
            unsafe_center=np.array([0.0, 0.0]),
            unsafe_radius=0.5,
            dynamics=linear_decay,
        )
    ))

    # Rotation (periodic, moderate)
    def rotation(t, x):
        return np.array([-x[1], x[0]])

    problems.append((
        "rotation_2d",
        UnsafeSupport(
            n_vars=2,
            time_horizon=2 * np.pi,
            initial_center=np.array([1.5, 0.0]),
            initial_radius=0.2,
            unsafe_center=np.array([-1.5, 0.0]),
            unsafe_radius=0.3,
            dynamics=rotation,
        )
    ))

    return problems


def run_single_comparison(
    problem: UnsafeSupport,
    problem_name: str,
    n_samples: int = 200,
    sdp_order: int = 4,
    verbose: bool = True,
) -> Optional[ExtendedComparisonResult]:
    """Run comparison on a single problem with extended metrics."""
    if verbose:
        print(f"\n  {problem_name} ({problem.n_vars}D, T={problem.time_horizon})...")
        if problem.unsafe_constraints:
            print(f"    Note: Unsafe set has {len(problem.unsafe_constraints)} constraint(s)")

    # Run spline-verify using proper problem translation
    dynamics, initial_set, unsafe_set = translate_problem_to_spline_verify(problem)

    verifier = SafetyVerifier(n_samples=n_samples, seed=42)

    try:
        start = time.perf_counter()
        sv_result = verifier.verify(dynamics, initial_set, unsafe_set, problem.time_horizon)
        spline_time = time.perf_counter() - start

        spline_min = sv_result.min_objective
        spline_error = sv_result.error_bound
        spline_lower = spline_min - spline_error
    except RuntimeError as e:
        if verbose:
            print(f"    ERROR: Integration failed - {e}")
        return None

    spline_status = sv_result.status.name

    # Run Miller-Sznaier
    ms_lower = 0.0
    ms_upper = None
    ms_time = 0.0
    ms_status = "not_available"

    if CVXPY_AVAILABLE and DistanceEstimator is not None:
        estimator = DistanceEstimator(order=sdp_order, verbose=False)
        start = time.perf_counter()
        ms_result = estimator.estimate(problem, compute_upper_bound=True)
        ms_time = time.perf_counter() - start

        ms_lower = ms_result.lower_bound
        ms_upper = ms_result.upper_bound
        ms_status = ms_result.status

    # Compute comparison metrics
    gap = spline_min - ms_lower
    relative_gap = gap / max(spline_min, 1e-6)
    speedup = ms_time / max(spline_time, 1e-6)

    if verbose:
        print(f"    Spline: min={spline_min:.4f}, error={spline_error:.4f}, time={spline_time:.2f}s")
        if CVXPY_AVAILABLE:
            print(f"    M-S:    lower={ms_lower:.4f}, upper={ms_upper:.4f}, time={ms_time:.2f}s")
            print(f"    Gap: {gap:.4f} ({relative_gap*100:.1f}%), Speedup: {speedup:.1f}x")

    return ExtendedComparisonResult(
        problem_name=problem_name,
        n_dims=problem.n_vars,
        time_horizon=problem.time_horizon,
        spline_min=spline_min,
        spline_error=spline_error,
        spline_lower=spline_lower,
        spline_time=spline_time,
        spline_status=spline_status,
        ms_lower=ms_lower,
        ms_upper=ms_upper,
        ms_time=ms_time,
        ms_status=ms_status,
        gap=gap,
        relative_gap=relative_gap,
        speedup=speedup,
    )


def run_sample_count_experiment(
    problem: UnsafeSupport,
    sample_counts: list[int],
    verbose: bool = True,
) -> dict[int, ExtendedComparisonResult]:
    """Run comparison with varying sample counts."""
    results = {}
    for n in sample_counts:
        result = run_single_comparison(
            problem, f"flow_n{n}", n_samples=n, verbose=False
        )
        results[n] = result
        if verbose:
            print(f"    n={n}: spline_min={result.spline_min:.4f}, "
                  f"gap={result.gap:.4f}, time={result.spline_time:.2f}s")
    return results


def run_order_experiment(
    problem: UnsafeSupport,
    orders: list[int],
    verbose: bool = True,
) -> dict[int, ExtendedComparisonResult]:
    """Run comparison with varying SDP relaxation orders."""
    if not CVXPY_AVAILABLE:
        print("    Skipped (cvxpy not available)")
        return {}

    results = {}
    for order in orders:
        result = run_single_comparison(
            problem, f"flow_order{order}", sdp_order=order, verbose=False
        )
        results[order] = result
        if verbose:
            print(f"    order={order}: ms_lower={result.ms_lower:.4f}, "
                  f"time={result.ms_time:.2f}s")
    return results


def plot_bounds_comparison(
    results: list[ExtendedComparisonResult],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot bounds comparison across problems."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [r.problem_name for r in results]
    x = np.arange(len(names))
    width = 0.25

    # Bounds comparison
    ax = axes[0]
    spline_mins = [r.spline_min for r in results]
    spline_lowers = [r.spline_lower for r in results]
    ms_lowers = [r.ms_lower for r in results]

    ax.bar(x - width, spline_mins, width, label='Spline min', color='blue', alpha=0.7)
    ax.bar(x, spline_lowers, width, label='Spline lower (min-ε)', color='lightblue', alpha=0.7)
    ax.bar(x + width, ms_lowers, width, label='M-S lower', color='red', alpha=0.7)

    ax.set_ylabel('Distance Bound', fontsize=12)
    ax.set_title('Distance Bounds Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Runtime comparison
    ax = axes[1]
    spline_times = [r.spline_time for r in results]
    ms_times = [r.ms_time for r in results]

    ax.bar(x - width/2, spline_times, width, label='Spline-Verify', color='blue', alpha=0.7)
    ax.bar(x + width/2, ms_times, width, label='Miller-Sznaier', color='red', alpha=0.7)

    ax.set_ylabel('Runtime (s)', fontsize=12)
    ax.set_title('Runtime Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved bounds comparison to {save_path}")

    return fig


def plot_convergence(
    sample_results: dict[int, ExtendedComparisonResult],
    order_results: dict[int, ExtendedComparisonResult],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot convergence with sample count and SDP order."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sample count convergence
    ax = axes[0]
    counts = sorted(sample_results.keys())
    mins = [sample_results[n].spline_min for n in counts]
    errors = [sample_results[n].spline_error for n in counts]
    lowers = [sample_results[n].spline_lower for n in counts]

    ax.plot(counts, mins, 'bo-', label='Spline min', linewidth=2, markersize=8)
    ax.fill_between(counts, lowers, mins, alpha=0.3, color='blue', label='Error band')

    if sample_results[counts[0]].ms_lower > 0:
        ms_lower = sample_results[counts[0]].ms_lower
        ax.axhline(y=ms_lower, color='red', linestyle='--', label=f'M-S lower ({ms_lower:.3f})')

    ax.set_xlabel('Sample Count', fontsize=12)
    ax.set_ylabel('Distance Bound', fontsize=12)
    ax.set_title('Spline-Verify Convergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SDP order convergence
    ax = axes[1]
    if order_results:
        orders = sorted(order_results.keys())
        ms_lowers = [order_results[o].ms_lower for o in orders]
        times = [order_results[o].ms_time for o in orders]

        ax.plot(orders, ms_lowers, 'ro-', label='M-S lower', linewidth=2, markersize=8)
        ax.set_xlabel('SDP Relaxation Order', fontsize=12)
        ax.set_ylabel('Lower Bound', fontsize=12)
        ax.set_title('Miller-Sznaier Convergence', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add runtime as secondary axis
        ax2 = ax.twinx()
        ax2.plot(orders, times, 'g--', label='Runtime', linewidth=1.5)
        ax2.set_ylabel('Runtime (s)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
    else:
        ax.text(0.5, 0.5, 'cvxpy not available', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_title('Miller-Sznaier Convergence', fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")

    return fig


def plot_gap_analysis(
    results: list[ExtendedComparisonResult],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot gap analysis between methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = [r.problem_name for r in results]
    gaps = [r.gap for r in results]
    rel_gaps = [r.relative_gap * 100 for r in results]
    speedups = [r.speedup for r in results]

    # Absolute gap
    ax = axes[0]
    colors = ['green' if g < 0.1 else 'orange' if g < 0.3 else 'red' for g in gaps]
    bars = ax.bar(names, gaps, color=colors, alpha=0.7)
    ax.set_ylabel('Gap (Spline min - M-S lower)', fontsize=12)
    ax.set_title('Absolute Gap Between Methods', fontsize=14)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add threshold lines
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good (<0.1)')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Moderate (<0.3)')
    ax.legend(loc='upper right')

    # Speedup
    ax = axes[1]
    colors = ['green' if s > 1 else 'red' for s in speedups]
    ax.bar(names, speedups, color=colors, alpha=0.7)
    ax.axhline(y=1, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Speedup (M-S time / Spline time)', fontsize=12)
    ax.set_title('Runtime Speedup (>1 = Spline faster)', fontsize=14)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved gap analysis to {save_path}")

    return fig


def plot_sampling_comparison(
    problem: UnsafeSupport,
    problem_type: str = 'flow',
    n_ball_trajectories: int = 8,
    save_path: Path | None = None,
) -> plt.Figure:
    """Create visualization comparing M-S (ball) vs Spline-verify (box) sampling.

    Color scheme:
    - Blue: M-S method (ball initial set + trajectories from ball)
    - Yellow: Spline-verify method (box initial set + trajectories from corners)
    - Green overlap: Where both methods sample (ball contained in box)
    - Red: Unsafe set

    Args:
        problem: UnsafeSupport problem definition
        problem_type: 'flow' for half-disk unsafe set, 'moon' for moon shape
        n_ball_trajectories: Number of trajectories to sample from ball interior
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    theta = np.linspace(0, 2*np.pi, 100)

    # Plot bounding box (spline-verify) - YELLOW, draw first so ball overlaps
    box_lower = problem.initial_center - problem.initial_radius
    box_upper = problem.initial_center + problem.initial_radius
    box_x = [box_lower[0], box_upper[0], box_upper[0], box_lower[0], box_lower[0]]
    box_y = [box_lower[1], box_lower[1], box_upper[1], box_upper[1], box_lower[1]]
    ax.fill(box_x, box_y, color='yellow', alpha=0.4, zorder=3, edgecolor='none')
    ax.plot(box_x, box_y, color='goldenrod', linewidth=2.5,
            linestyle='--', label='Spline-verify initial set (box)', zorder=10)

    # Plot initial ball (M-S) - BLUE, on top of box
    ball_x = problem.initial_center[0] + problem.initial_radius * np.cos(theta)
    ball_y = problem.initial_center[1] + problem.initial_radius * np.sin(theta)
    ax.fill(ball_x, ball_y, color='blue', alpha=0.2, zorder=4, edgecolor='none')
    ax.plot(ball_x, ball_y, 'b-', linewidth=2.5, label='M-S initial set (ball)', zorder=10)

    # Plot unsafe set based on problem type
    if problem_type == 'flow':
        # Half-disk for flow system
        # Get angle from problem or default to 5π/4
        half_angle = getattr(problem, 'half_space_angle', None) or (5 * np.pi / 4)

        # Convert half-space angle to wedge angles
        # The half-space constraint cos(θ)*(y1-c1) + sin(θ)*(y2-c2) >= 0
        # selects the half-plane in the direction of angle θ
        # Wedge needs start and end angles (in degrees)
        # The selected region is from (θ - 90°) to (θ + 90°)
        angle_deg = np.degrees(half_angle)
        wedge_start = angle_deg - 90
        wedge_end = angle_deg + 90

        wedge = Wedge(problem.unsafe_center, problem.unsafe_radius,
                     wedge_start, wedge_end,
                     facecolor='red', alpha=0.25, edgecolor='none',
                     label='Unsafe set (half-disk)')
        ax.add_patch(wedge)

        # Draw the edge separately for better visibility
        arc_start_rad = np.radians(wedge_start)
        arc_end_rad = np.radians(wedge_end)
        arc_theta = np.linspace(arc_start_rad, arc_end_rad, 100)
        arc_x = problem.unsafe_center[0] + problem.unsafe_radius * np.cos(arc_theta)
        arc_y = problem.unsafe_center[1] + problem.unsafe_radius * np.sin(arc_theta)
        ax.plot(arc_x, arc_y, 'r-', linewidth=2.5)
        # Straight edge (diameter)
        start_pt = problem.unsafe_center + problem.unsafe_radius * np.array([np.cos(arc_start_rad), np.sin(arc_start_rad)])
        end_pt = problem.unsafe_center + problem.unsafe_radius * np.array([np.cos(arc_end_rad), np.sin(arc_end_rad)])
        ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], 'r-', linewidth=2.5)
    elif problem_type == 'moon':
        # Moon shape from MATLAB flow_dist_moon.m
        h_in = 0.4
        h_out = 1.0
        moon_center = np.array([0.4, -0.4])
        moon_theta = -np.pi / 10
        moon_scale = 0.8

        moon_rot = np.array([
            [np.cos(moon_theta), -np.sin(moon_theta)],
            [np.sin(moon_theta), np.cos(moon_theta)]
        ])

        c_in = np.array([0.0, 0.5 * (1.0/h_in - h_in)])
        r_in = 0.5 * (1.0/h_in + h_in)
        c_out = np.array([0.0, 0.5 * (1.0/h_out - h_out)])
        r_out = 0.5 * (1.0/h_out + h_out)

        inner_center = moon_rot @ c_in * moon_scale + moon_center
        outer_center = moon_rot @ c_out * moon_scale + moon_center
        inner_radius = moon_scale * r_in
        outer_radius = moon_scale * r_out

        # Create crescent by sampling points in the moon region
        # Generate a grid and check which points are in the crescent
        from matplotlib.patches import Polygon

        # Sample points along the crescent boundary
        n_pts = 200
        theta_fine = np.linspace(0, 2*np.pi, n_pts)

        # Get points on outer circle that are outside inner circle
        outer_arc = []
        for t in theta_fine:
            pt = outer_center + outer_radius * np.array([np.cos(t), np.sin(t)])
            if np.linalg.norm(pt - inner_center) >= inner_radius:
                outer_arc.append(pt)

        # Get points on inner circle that are inside outer circle
        inner_arc = []
        for t in theta_fine:
            pt = inner_center + inner_radius * np.array([np.cos(t), np.sin(t)])
            if np.linalg.norm(pt - outer_center) <= outer_radius:
                inner_arc.append(pt)

        # Combine to form crescent polygon (outer arc + reversed inner arc)
        if outer_arc and inner_arc:
            outer_arc = np.array(outer_arc)
            inner_arc = np.array(inner_arc)
            # Reverse inner arc to trace boundary correctly
            crescent_pts = np.vstack([outer_arc, inner_arc[::-1]])
            crescent = Polygon(crescent_pts, facecolor='red', alpha=0.25,
                              edgecolor='none', label='Unsafe set (moon)')
            ax.add_patch(crescent)

            # Draw edges
            ax.plot(outer_arc[:, 0], outer_arc[:, 1], 'r-', linewidth=2.5)
            ax.plot(inner_arc[:, 0], inner_arc[:, 1], 'r-', linewidth=2.5)

    # Sample and plot trajectories from INSIDE the ball - BLUE
    np.random.seed(123)
    ball_traj_plotted = False
    for i in range(n_ball_trajectories):
        angle = 2 * np.pi * i / n_ball_trajectories
        r = problem.initial_radius * 0.7
        x0 = problem.initial_center + r * np.array([np.cos(angle), np.sin(angle)])

        sol = solve_ivp(
            problem.dynamics, [0, problem.time_horizon], x0,
            t_eval=np.linspace(0, problem.time_horizon, 200),
            method='RK45'
        )
        if sol.success:
            label = 'Trajectories from ball (M-S)' if not ball_traj_plotted else None
            ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=1.2, alpha=0.6, label=label)
            ax.plot(x0[0], x0[1], 'o', color='blue', markersize=6, alpha=0.7)
            ball_traj_plotted = True

    # Plot trajectories from box corners (outside ball) - YELLOW/GOLD
    corners = [
        np.array([box_lower[0], box_lower[1]]),  # bottom-left
        np.array([box_lower[0], box_upper[1]]),  # top-left
        np.array([box_upper[0], box_lower[1]]),  # bottom-right
        np.array([box_upper[0], box_upper[1]]),  # top-right
    ]

    corner_traj_plotted = False
    for corner in corners:
        # Check if corner is outside the ball
        dist_from_center = np.linalg.norm(corner - problem.initial_center)
        if dist_from_center > problem.initial_radius:
            sol = solve_ivp(
                problem.dynamics, [0, problem.time_horizon], corner,
                t_eval=np.linspace(0, problem.time_horizon, 200),
                method='RK45'
            )
            if sol.success:
                label = 'Trajectories from box corners (Spline-verify)' if not corner_traj_plotted else None
                ax.plot(sol.y[0], sol.y[1], color='goldenrod', linewidth=2, alpha=0.85, label=label)
                ax.plot(corner[0], corner[1], 's', color='goldenrod', markersize=10,
                       markeredgecolor='black', markeredgewidth=1)
                corner_traj_plotted = True

    # Labels and formatting
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    title = 'Flow System' if problem_type == 'flow' else 'Moon System'
    ax.set_title(f'{title}: M-S (ball) vs Spline-verify (box) Sampling', fontsize=14, fontweight='bold')

    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=10, framealpha=0.95)

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Adjust axis limits to reduce unused space
    # Get current data limits and add small padding
    ax.autoscale(enable=True, axis='both', tight=True)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    padding = 0.05  # 5% padding
    ax.set_xlim(xlim[0] - padding * x_range, xlim[1] + padding * x_range)
    ax.set_ylim(ylim[0] - padding * y_range, ylim[1] + padding * y_range)

    # For moon figure, cap y-axis at 0.5 since there's no data above that
    if problem_type == 'moon':
        current_ylim = ax.get_ylim()
        ax.set_ylim(current_ylim[0], min(current_ylim[1], 0.5))

    # Adjust layout for legend
    plt.tight_layout()
    fig.subplots_adjust(right=0.62)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {problem_type} sampling comparison to {save_path}")

    return fig


def plot_spline_approximation(
    problem: UnsafeSupport,
    problem_name: str,
    problem_type: str = 'flow',
    n_samples: int = 200,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot the spline-verify approximation F̃_T(x₀) over the initial set.

    This shows the ACTUAL approximation produced by spline-verify, not brute-force
    computed ground truth. The plot includes:
    - Contour of the fitted spline approximation
    - Sample points used for fitting (scatter overlay)
    - Minimizer location (star marker)
    - Initial and unsafe set boundaries

    Args:
        problem: UnsafeSupport problem definition
        problem_name: Name for the title
        problem_type: 'flow' for half-disk unsafe set, 'moon' for moon shape
        n_samples: Number of samples for spline fitting
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    from matplotlib.patches import Polygon

    print(f"  Running spline-verify for {problem_name}...")

    # Translate problem and run spline-verify
    dynamics, initial_set, unsafe_set = translate_problem_to_spline_verify(problem)
    verifier = SafetyVerifier(n_samples=n_samples, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, problem.time_horizon)

    # Extract spline and sample data from result
    spline = result.details.get('spline')
    sample_points = result.details['sample_points']
    sample_values = result.details['sample_values']

    # Create evaluation grid over initial set
    center = problem.initial_center
    radius = problem.initial_radius
    n_grid = 60

    x1_range = np.linspace(center[0] - radius, center[0] + radius, n_grid)
    x2_range = np.linspace(center[1] - radius, center[1] + radius, n_grid)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Get color scale from sample values
    vmin, vmax = sample_values.min(), sample_values.max()

    # Evaluate spline on grid if available (only within initial ball)
    if spline is not None:
        F_approx = np.full_like(X1, np.nan)
        for i in range(n_grid):
            for j in range(n_grid):
                x = np.array([X1[i, j], X2[i, j]])
                if np.linalg.norm(x - center) <= radius:
                    F_approx[i, j] = spline.evaluate(x)

        # Plot contour of spline approximation
        valid_mask = ~np.isnan(F_approx)
        if np.any(valid_mask):
            # Update vmin/vmax to include spline values
            vmin = min(vmin, np.nanmin(F_approx))
            vmax = max(vmax, np.nanmax(F_approx))
            levels = np.linspace(vmin, vmax, 20)

            # Include zero level if range spans zero
            if vmin < 0 < vmax:
                levels = np.sort(np.unique(np.concatenate([levels, [0]])))

            cs = ax.contourf(X1, X2, F_approx, levels=levels, cmap='RdYlGn', alpha=0.85)
            cbar = plt.colorbar(cs, ax=ax)
            cbar.set_label(r'Spline Approximation $\tilde{F}_T(x_0)$', fontsize=12)

            # Add zero contour if exists
            if vmin < 0 < vmax:
                ax.contour(X1, X2, F_approx, levels=[0], colors='black', linewidths=2)
    else:
        # No spline fitted (early exit case) - just show sample points
        ax.text(
            0.5, 0.5, 'UNSAFE: Sample hit unsafe set\n(No spline fitted)',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='center', color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )

    # Overlay sample points (colored by their objective values)
    scatter = ax.scatter(
        sample_points[:, 0], sample_points[:, 1],
        c=sample_values, cmap='RdYlGn',
        vmin=vmin, vmax=vmax,
        s=25, alpha=0.7, edgecolors='black', linewidths=0.5,
        zorder=5
    )

    # Mark the minimizer with a star
    ax.plot(
        result.minimizer[0], result.minimizer[1],
        'k*', markersize=18, markeredgecolor='white', markeredgewidth=1.5,
        label=f'Minimizer (F̃ = {result.min_objective:.3f})', zorder=10
    )

    # Plot initial set boundary (ball)
    theta = np.linspace(0, 2*np.pi, 100)
    init_x = center[0] + radius * np.cos(theta)
    init_y = center[1] + radius * np.sin(theta)
    ax.plot(init_x, init_y, 'b-', linewidth=2.5, label='Initial set (ball)')

    # Plot unsafe set based on problem type
    if problem_type == 'flow':
        # Half-disk unsafe set
        half_angle = getattr(problem, 'half_space_angle', None) or (5 * np.pi / 4)
        angle_deg = np.degrees(half_angle)
        wedge_start = angle_deg - 90
        wedge_end = angle_deg + 90

        wedge = Wedge(
            problem.unsafe_center, problem.unsafe_radius,
            wedge_start, wedge_end,
            facecolor='red', alpha=0.4, edgecolor='red', linewidth=2,
            label='Unsafe set (half-disk)', zorder=3
        )
        ax.add_patch(wedge)
    elif problem_type == 'moon':
        # Moon shape unsafe set
        h_in, h_out = 0.4, 1.0
        moon_center = np.array([0.4, -0.4])
        moon_theta = -np.pi / 10
        moon_scale = 0.8

        moon_rot = np.array([
            [np.cos(moon_theta), -np.sin(moon_theta)],
            [np.sin(moon_theta), np.cos(moon_theta)]
        ])

        c_in = np.array([0.0, 0.5 * (1.0/h_in - h_in)])
        r_in = 0.5 * (1.0/h_in + h_in)
        c_out = np.array([0.0, 0.5 * (1.0/h_out - h_out)])
        r_out = 0.5 * (1.0/h_out + h_out)

        inner_center = moon_rot @ c_in * moon_scale + moon_center
        outer_center = moon_rot @ c_out * moon_scale + moon_center
        inner_radius = moon_scale * r_in
        outer_radius = moon_scale * r_out

        # Create crescent polygon
        n_pts = 200
        theta_fine = np.linspace(0, 2*np.pi, n_pts)

        outer_arc = [outer_center + outer_radius * np.array([np.cos(t), np.sin(t)])
                     for t in theta_fine
                     if np.linalg.norm(outer_center + outer_radius * np.array([np.cos(t), np.sin(t)]) - inner_center) >= inner_radius]
        inner_arc = [inner_center + inner_radius * np.array([np.cos(t), np.sin(t)])
                     for t in theta_fine
                     if np.linalg.norm(inner_center + inner_radius * np.array([np.cos(t), np.sin(t)]) - outer_center) <= outer_radius]

        if outer_arc and inner_arc:
            outer_arc = np.array(outer_arc)
            inner_arc = np.array(inner_arc)
            crescent_pts = np.vstack([outer_arc, inner_arc[::-1]])
            crescent = Polygon(
                crescent_pts, facecolor='red', alpha=0.4,
                edgecolor='red', linewidth=2, label='Unsafe set (moon)', zorder=3
            )
            ax.add_patch(crescent)

    # Add text annotation with results
    text_lines = [
        f"Spline min: {result.min_objective:.4f}",
        f"Error bound: {result.error_bound:.4f}",
        f"Status: {result.status.name}",
        f"Samples: {n_samples}",
    ]

    # Run M-S if available to show comparison
    if CVXPY_AVAILABLE and DistanceEstimator is not None:
        try:
            estimator = DistanceEstimator(order=4, verbose=False)
            ms_result = estimator.estimate(problem, compute_upper_bound=False)
            text_lines.append(f"M-S lower: {ms_result.lower_bound:.4f}")
        except Exception:
            pass

    ax.text(
        0.02, 0.98, '\n'.join(text_lines),
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )

    # Labels and formatting
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title(f'{problem_name}: Spline Approximation $\\tilde{{F}}_T(x_0)$',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved spline approximation plot to {save_path}")

    return fig


# ============================================================================
# Visualization Options for Comparison
# ============================================================================

def _run_spline_verify(problem: UnsafeSupport, n_samples: int = 200):
    """Helper to run spline-verify and return result with spline."""
    dynamics, initial_set, unsafe_set = translate_problem_to_spline_verify(problem)
    verifier = SafetyVerifier(n_samples=n_samples, seed=42)
    return verifier.verify(dynamics, initial_set, unsafe_set, problem.time_horizon)


def _get_ms_bound(problem: UnsafeSupport) -> float | None:
    """Helper to get M-S lower bound if available."""
    if CVXPY_AVAILABLE and DistanceEstimator is not None:
        try:
            estimator = DistanceEstimator(order=4, verbose=False)
            ms_result = estimator.estimate(problem, compute_upper_bound=False)
            return ms_result.lower_bound
        except Exception:
            pass
    return None


def _evaluate_spline_on_grid(spline, center, radius, n_grid=60):
    """Helper to evaluate spline on a grid over the initial ball."""
    x1_range = np.linspace(center[0] - radius, center[0] + radius, n_grid)
    x2_range = np.linspace(center[1] - radius, center[1] + radius, n_grid)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    F_approx = np.full_like(X1, np.nan)
    for i in range(n_grid):
        for j in range(n_grid):
            x = np.array([X1[i, j], X2[i, j]])
            if np.linalg.norm(x - center) <= radius:
                F_approx[i, j] = spline.evaluate(x)

    return X1, X2, F_approx


def plot_option1_clean_contour(
    problem: UnsafeSupport,
    n_samples: int = 200,
    save_path: Path | None = None,
) -> plt.Figure:
    """Option 1: Clean contour plot of F̃_T(x₀), no unsafe set.

    Shows only the spline approximation as a function of initial state.
    """
    print("  Generating Option 1: Clean contour...")
    result = _run_spline_verify(problem, n_samples)
    ms_bound = _get_ms_bound(problem)

    spline = result.details.get('spline')
    sample_points = result.details['sample_points']
    sample_values = result.details['sample_values']

    if spline is None:
        print("    Skipped (no spline fitted - early exit)")
        return None

    center = problem.initial_center
    radius = problem.initial_radius
    X1, X2, F_approx = _evaluate_spline_on_grid(spline, center, radius)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    vmin, vmax = np.nanmin(F_approx), np.nanmax(F_approx)
    levels = np.linspace(vmin, vmax, 20)

    cs = ax.contourf(X1, X2, F_approx, levels=levels, cmap='viridis', alpha=0.9)
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Min distance to unsafe set', fontsize=12)

    # Sample points
    ax.scatter(sample_points[:, 0], sample_points[:, 1],
               c=sample_values, cmap='viridis', vmin=vmin, vmax=vmax,
               s=20, alpha=0.6, edgecolors='white', linewidths=0.3, zorder=5)

    # Minimizer
    ax.plot(result.minimizer[0], result.minimizer[1],
            'r*', markersize=18, markeredgecolor='white', markeredgewidth=1.5,
            label=f'Minimizer (F̃ = {result.min_objective:.4f})', zorder=10)

    # Initial set boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            'k--', linewidth=2, label='Initial set boundary')

    # Text annotation
    text = f"Spline min: {result.min_objective:.4f}\nError bound: {result.error_bound:.4f}"
    if ms_bound is not None:
        text += f"\nM-S bound: {ms_bound:.4f}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title(r'Option 1: Spline Approximation $\tilde{F}_T(x_0)$', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved to {save_path}")

    return fig


def plot_option2_contour_ms_line(
    problem: UnsafeSupport,
    n_samples: int = 200,
    save_path: Path | None = None,
) -> plt.Figure:
    """Option 2: Contour with M-S bound as contour line."""
    print("  Generating Option 2: Contour with M-S line...")
    result = _run_spline_verify(problem, n_samples)
    ms_bound = _get_ms_bound(problem)

    spline = result.details.get('spline')
    if spline is None:
        print("    Skipped (no spline fitted)")
        return None

    center = problem.initial_center
    radius = problem.initial_radius
    X1, X2, F_approx = _evaluate_spline_on_grid(spline, center, radius)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    vmin, vmax = np.nanmin(F_approx), np.nanmax(F_approx)
    levels = np.linspace(vmin, vmax, 20)

    cs = ax.contourf(X1, X2, F_approx, levels=levels, cmap='viridis', alpha=0.9)
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Min distance to unsafe set', fontsize=12)

    # Add M-S bound as contour line if within range
    if ms_bound is not None and vmin <= ms_bound <= vmax:
        cs_ms = ax.contour(X1, X2, F_approx, levels=[ms_bound],
                          colors='red', linewidths=3, linestyles='--')
        ax.clabel(cs_ms, fmt=f'M-S={ms_bound:.3f}', fontsize=10)

    # Minimizer
    ax.plot(result.minimizer[0], result.minimizer[1],
            'r*', markersize=18, markeredgecolor='white', markeredgewidth=1.5,
            label=f'Minimizer (F̃ = {result.min_objective:.4f})', zorder=10)

    # Initial set boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            'k--', linewidth=2, label='Initial set boundary')

    # Text annotation
    text = f"Spline min: {result.min_objective:.4f}"
    if ms_bound is not None:
        text += f"\nM-S bound: {ms_bound:.4f}"
        if ms_bound > vmax:
            text += " (above range)"
        elif ms_bound < vmin:
            text += " (below range)"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title(r'Option 2: Contour with M-S Bound Line', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved to {save_path}")

    return fig


def plot_option3_1d_slice(
    problem: UnsafeSupport,
    n_samples: int = 200,
    save_path: Path | None = None,
) -> plt.Figure:
    """Option 3: 1D slice through center showing F̃_T vs x₁."""
    print("  Generating Option 3: 1D slice...")
    result = _run_spline_verify(problem, n_samples)
    ms_bound = _get_ms_bound(problem)

    spline = result.details.get('spline')
    sample_points = result.details['sample_points']
    sample_values = result.details['sample_values']

    if spline is None:
        print("    Skipped (no spline fitted)")
        return None

    center = problem.initial_center
    radius = problem.initial_radius

    # Create 1D slice at x2 = center[1]
    x1_vals = np.linspace(center[0] - radius, center[0] + radius, 100)
    f_vals = []
    for x1 in x1_vals:
        x = np.array([x1, center[1]])
        if np.linalg.norm(x - center) <= radius:
            f_vals.append(spline.evaluate(x))
        else:
            f_vals.append(np.nan)
    f_vals = np.array(f_vals)

    # Find sample points near the slice (within tolerance)
    tol = 0.1 * radius
    near_slice = np.abs(sample_points[:, 1] - center[1]) < tol
    slice_samples_x = sample_points[near_slice, 0]
    slice_samples_f = sample_values[near_slice]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot spline approximation
    ax.plot(x1_vals, f_vals, 'b-', linewidth=2.5, label=r'Spline $\tilde{F}_T(x_1, x_2^*)$')

    # Plot M-S bound as horizontal line
    if ms_bound is not None:
        ax.axhline(y=ms_bound, color='red', linestyle='--', linewidth=2,
                   label=f'M-S bound = {ms_bound:.4f}')

    # Plot sample points near slice
    if len(slice_samples_x) > 0:
        ax.scatter(slice_samples_x, slice_samples_f, c='green', s=50,
                   edgecolors='black', linewidths=0.5, zorder=5,
                   label=f'Samples near slice (n={len(slice_samples_x)})')

    # Mark minimizer if on slice
    if np.abs(result.minimizer[1] - center[1]) < tol:
        ax.axvline(x=result.minimizer[0], color='purple', linestyle=':',
                   linewidth=1.5, alpha=0.7)
        ax.plot(result.minimizer[0], result.min_objective, 'r*', markersize=15,
                label=f'Minimizer = {result.min_objective:.4f}')

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel(r'$\tilde{F}_T(x_0)$ (min distance to unsafe)', fontsize=14)
    ax.set_title(f'Option 3: 1D Slice at $x_2 = {center[1]:.2f}$', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Mark initial set bounds
    ax.axvline(x=center[0] - radius, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=center[0] + radius, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved to {save_path}")

    return fig


def plot_option4_3d_surface(
    problem: UnsafeSupport,
    n_samples: int = 200,
    save_path: Path | None = None,
) -> plt.Figure:
    """Option 4: 3D surface plot of F̃_T(x₀) with binary safe/unsafe colors."""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import ListedColormap

    print("  Generating Option 4: 3D surface...")
    result = _run_spline_verify(problem, n_samples)
    ms_bound = _get_ms_bound(problem)

    spline = result.details.get('spline')
    if spline is None:
        print("    Skipped (no spline fitted)")
        return None

    center = problem.initial_center
    radius = problem.initial_radius
    X1, X2, F_approx = _evaluate_spline_on_grid(spline, center, radius, n_grid=40)

    # Get sample statistics for more accurate safety assessment
    sample_values = result.details['sample_values']
    sample_min = sample_values.min()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create binary color array: red (unsafe, F <= 0) vs green (safe, F > 0)
    colors = np.where(F_approx > 0, 1.0, 0.0)  # 1 = safe (green), 0 = unsafe (red)

    # Binary colormap: red and green only
    cmap_binary = ListedColormap(['#CC0000', '#00AA00'])  # Dark red, Dark green

    # Plot surface with binary colors
    surf = ax.plot_surface(X1, X2, F_approx, facecolors=cmap_binary(colors),
                           alpha=0.9, linewidth=0, antialiased=True,
                           shade=True)

    # Mark minimizer
    ax.scatter([result.minimizer[0]], [result.minimizer[1]], [result.min_objective],
               c='black', s=150, marker='*', edgecolors='white', linewidths=1,
               zorder=10)

    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_zlabel(r'$\tilde{F}_T(x_0)$ (min dist to unsafe)', fontsize=12)
    ax.set_title('Option 4: 3D Surface (Binary Safe/Unsafe Colors)',
                fontsize=14, fontweight='bold')

    # Create legend patches instead of colorbar
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00AA00', label='Safe (F > 0)'),
        Patch(facecolor='#CC0000', label='Unsafe (F ≤ 0)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Text annotation - use sample_min for more accurate safety assessment
    actual_status = 'SAFE' if sample_min > 0 else 'UNSAFE'
    text = f"Status: {actual_status}\nSample min: {sample_min:.4f}\nSpline min: {result.min_objective:.4f}"
    if ms_bound is not None:
        text += f"\nM-S bound: {ms_bound:.4f}"
    ax.text2D(0.02, 0.98, text, transform=ax.transAxes, fontsize=11,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved to {save_path}")

    return fig


def plot_option5_samples_comparison(
    problem: UnsafeSupport,
    n_samples: int = 200,
    save_path: Path | None = None,
) -> plt.Figure:
    """Option 5: Side-by-side showing raw samples and fitted approximation."""
    print("  Generating Option 5: Samples comparison...")
    result = _run_spline_verify(problem, n_samples)
    ms_bound = _get_ms_bound(problem)

    spline = result.details.get('spline')
    sample_points = result.details['sample_points']
    sample_values = result.details['sample_values']

    center = problem.initial_center
    radius = problem.initial_radius

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    vmin, vmax = sample_values.min(), sample_values.max()

    # Left: Raw samples only
    ax = axes[0]
    scatter = ax.scatter(sample_points[:, 0], sample_points[:, 1],
                        c=sample_values, cmap='viridis', vmin=vmin, vmax=vmax,
                        s=40, edgecolors='black', linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label='Sampled F_T values')

    # Initial set boundary
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            'k--', linewidth=2)

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title(f'Raw Samples (n={n_samples})\nmin={sample_values.min():.4f}', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Right: Fitted spline contour
    ax = axes[1]
    if spline is not None:
        X1, X2, F_approx = _evaluate_spline_on_grid(spline, center, radius)
        cs = ax.contourf(X1, X2, F_approx, levels=20, cmap='viridis',
                        vmin=vmin, vmax=vmax, alpha=0.9)
        plt.colorbar(cs, ax=ax, label=r'Spline $\tilde{F}_T$')

        # Minimizer
        ax.plot(result.minimizer[0], result.minimizer[1],
                'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
    else:
        ax.text(0.5, 0.5, 'No spline fitted\n(early exit)', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)

    ax.plot(center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            'k--', linewidth=2)

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    title = f'Spline Approximation\nmin={result.min_objective:.4f}'
    if ms_bound is not None:
        title += f', M-S={ms_bound:.4f}'
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Option 5: Raw Samples vs Fitted Spline', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved to {save_path}")

    return fig


def plot_distance_function_3d(
    problem: UnsafeSupport,
    problem_name: str,
    n_samples: int = 500,
    spline_smoothing: float = 0.01,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot 3D surface of spline-verify distance approximation with binary safe/unsafe colors.

    This is the primary visualization for showing how F̃_T(x₀) varies over initial states.
    - Green: SAFE cases (spline min > 0) - entire surface is green
    - Red/Green: UNSAFE cases - binary coloring based on function values

    Args:
        problem: UnsafeSupport problem definition
        problem_name: Name for the title
        n_samples: Number of samples for spline fitting (default 500 for better accuracy)
        spline_smoothing: RBF smoothing parameter (default 0.01 to reduce overshoot)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import ListedColormap
    from scipy.interpolate import griddata

    print(f"  Generating 3D distance function for {problem_name}...")

    # Run spline-verify with improved hyperparameters
    dynamics, initial_set, unsafe_set = translate_problem_to_spline_verify(problem)
    verifier = SafetyVerifier(n_samples=n_samples, seed=42, spline_smoothing=spline_smoothing)
    result = verifier.verify(dynamics, initial_set, unsafe_set, problem.time_horizon)

    ms_bound = _get_ms_bound(problem)

    spline = result.details.get('spline')
    sample_points = result.details['sample_points']
    sample_values = result.details['sample_values']
    center = problem.initial_center
    radius = problem.initial_radius

    # Handle case where no spline is fitted (sample hit unsafe set)
    # Use griddata to create a surface from samples for consistent visualization
    if spline is None:
        print("    Sample hit unsafe set - interpolating surface from samples...")
        # Create grid for interpolation
        n_grid = 50
        x1_range = np.linspace(center[0] - radius, center[0] + radius, n_grid)
        x2_range = np.linspace(center[1] - radius, center[1] + radius, n_grid)
        X1, X2 = np.meshgrid(x1_range, x2_range)

        # Interpolate using griddata (cubic for smooth surface)
        F_approx = griddata(sample_points, sample_values, (X1, X2), method='cubic')
        # Fill NaN values (outside convex hull) with nearest neighbor
        F_approx_nearest = griddata(sample_points, sample_values, (X1, X2), method='nearest')
        F_approx = np.where(np.isnan(F_approx), F_approx_nearest, F_approx)

        spline_min = sample_values.min()
    else:
        X1, X2, F_approx = _evaluate_spline_on_grid(spline, center, radius, n_grid=50)
        # Use actual grid minimum, not optimizer result (optimizer may miss local minima)
        spline_min = np.nanmin(F_approx)

    # Determine safety status based on actual grid minimum (not optimizer result)
    # This ensures consistency between what we visualize and the status we report
    is_safe = spline_min > 0

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Binary colormap: red and green only
    cmap_binary = ListedColormap(['#CC0000', '#00AA00'])  # Dark red, Dark green

    # Determine coloring based on safety status
    if is_safe:
        # SAFE case: entire surface is green (status determined by spline minimum)
        colors = np.ones_like(F_approx)  # All 1.0 = green
        title_color = 'black'
    else:
        # UNSAFE case: binary coloring based on function values
        colors = np.where(F_approx > 0, 1.0, 0.0)  # 1 = green, 0 = red
        title_color = 'red'

    # Plot surface with binary colors
    surf = ax.plot_surface(X1, X2, F_approx, facecolors=cmap_binary(colors),
                           alpha=0.9, linewidth=0, antialiased=True,
                           shade=True)

    # Add M-S bound as horizontal plane if available
    if ms_bound is not None:
        vmin, vmax = np.nanmin(F_approx), np.nanmax(F_approx)
        # Only show if bound is within a reasonable range
        if vmin - 0.1 * (vmax - vmin) <= ms_bound <= vmax + 0.1 * (vmax - vmin):
            xx, yy = np.meshgrid(
                np.linspace(center[0] - radius, center[0] + radius, 10),
                np.linspace(center[1] - radius, center[1] + radius, 10)
            )
            ax.plot_surface(xx, yy, np.full_like(xx, ms_bound),
                           alpha=0.3, color='blue')

    # Mark minimizer
    if spline is not None:
        ax.scatter([result.minimizer[0]], [result.minimizer[1]], [result.min_objective],
                   c='black', s=200, marker='*', edgecolors='white', linewidths=1.5,
                   zorder=10)
    else:
        # For interpolated surface, mark the sample with min value
        min_idx = np.argmin(sample_values)
        ax.scatter([sample_points[min_idx, 0]], [sample_points[min_idx, 1]], [sample_values[min_idx]],
                   c='black', s=200, marker='*', edgecolors='white', linewidths=1.5, zorder=10)

    ax.set_xlabel('$x_1$ (initial state)', fontsize=12)
    ax.set_ylabel('$x_2$ (initial state)', fontsize=12)
    ax.set_zlabel(r'$\tilde{F}_T(x_0)$', fontsize=12)

    status = 'SAFE' if is_safe else 'UNSAFE'
    ax.set_title(f'{problem_name}\nSpline Approximation of Distance Function',
                fontsize=14, fontweight='bold', color=title_color)

    # Create legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#00AA00', label='Safe (F > 0)'),
        Patch(facecolor='#CC0000', label='Unsafe (F ≤ 0)'),
    ]
    if ms_bound is not None:
        legend_elements.append(Patch(facecolor='blue', alpha=0.3, label='M-S lower bound'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Status annotation
    text_lines = [
        f"Status: {status}",
        f"Spline min: {spline_min:.4f}",
        f"Samples: {n_samples}",
    ]
    if ms_bound is not None:
        text_lines.append(f"M-S lower: {ms_bound:.4f}")

    ax.text2D(0.02, 0.98, '\n'.join(text_lines), transform=ax.transAxes, fontsize=11,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.view_init(elev=30, azim=-50)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved to {save_path}")

    return fig


def plot_distance_function_1d_slice(
    problem: UnsafeSupport,
    problem_name: str,
    n_samples: int = 500,
    spline_smoothing: float = 0.01,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot 1D slice through center showing F̃_T vs x₁.

    This visualization shows:
    - Spline approximation along a 1D slice at x₂ = center[1]
    - M-S lower bound as horizontal dashed line
    - Sample points near the slice
    - Minimizer location

    Args:
        problem: UnsafeSupport problem definition
        problem_name: Name for the title
        n_samples: Number of samples for spline fitting (default 500)
        spline_smoothing: RBF smoothing parameter (default 0.01)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    print(f"  Generating 1D slice for {problem_name}...")

    # Run spline-verify with improved hyperparameters
    dynamics, initial_set, unsafe_set = translate_problem_to_spline_verify(problem)
    verifier = SafetyVerifier(n_samples=n_samples, seed=42, spline_smoothing=spline_smoothing)
    result = verifier.verify(dynamics, initial_set, unsafe_set, problem.time_horizon)

    ms_bound = _get_ms_bound(problem)

    spline = result.details.get('spline')
    sample_points = result.details['sample_points']
    sample_values = result.details['sample_values']
    center = problem.initial_center
    radius = problem.initial_radius

    # Determine safety status
    is_safe = result.min_objective > 0

    # Create 1D slice at x2 = center[1]
    x1_vals = np.linspace(center[0] - radius, center[0] + radius, 200)
    f_vals = []

    if spline is not None:
        for x1 in x1_vals:
            x = np.array([x1, center[1]])
            if np.linalg.norm(x - center) <= radius:
                f_vals.append(spline.evaluate(x))
            else:
                f_vals.append(np.nan)
        f_vals = np.array(f_vals)
        # Use actual slice minimum for accurate status
        slice_min = np.nanmin(f_vals)
    else:
        # No spline - use griddata interpolation from samples
        from scipy.interpolate import griddata
        # Get slice values from nearest samples
        slice_points = np.column_stack([x1_vals, np.full_like(x1_vals, center[1])])
        f_vals = griddata(sample_points, sample_values, slice_points, method='linear')
        f_vals_nearest = griddata(sample_points, sample_values, slice_points, method='nearest')
        f_vals = np.where(np.isnan(f_vals), f_vals_nearest, f_vals)
        slice_min = sample_values.min()

    # For a complete picture, we need to check the full 2D spline minimum, not just the slice
    # Evaluate spline on a grid to find the actual minimum
    if spline is not None:
        X1_grid, X2_grid, F_grid = _evaluate_spline_on_grid(spline, center, radius, n_grid=50)
        spline_min = np.nanmin(F_grid)
    else:
        spline_min = sample_values.min()

    # Safety status based on full 2D minimum
    is_safe = spline_min > 0

    # Find sample points near the slice (within tolerance)
    tol = 0.1 * radius
    near_slice = np.abs(sample_points[:, 1] - center[1]) < tol
    slice_samples_x = sample_points[near_slice, 0]
    slice_samples_f = sample_values[near_slice]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot spline approximation
    ax.plot(x1_vals, f_vals, 'b-', linewidth=2.5, label=r'Spline $\tilde{F}_T(x_1, x_2^*)$')

    # Plot M-S bound as horizontal line
    if ms_bound is not None:
        ax.axhline(y=ms_bound, color='red', linestyle='--', linewidth=2,
                   label=f'M-S bound = {ms_bound:.4f}')

    # Plot sample points near slice
    if len(slice_samples_x) > 0:
        ax.scatter(slice_samples_x, slice_samples_f, c='green', s=50,
                   edgecolors='black', linewidths=0.5, zorder=5,
                   label=f'Samples near slice (n={len(slice_samples_x)})')

    # Mark the slice minimum (actual minimum along this slice)
    slice_min_idx = np.nanargmin(f_vals)
    ax.plot(x1_vals[slice_min_idx], f_vals[slice_min_idx], 'r*', markersize=15,
            label=f'Slice min = {slice_min:.4f}')

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel(r'$\tilde{F}_T(x_0)$ (min distance to unsafe)', fontsize=14)

    status = 'SAFE' if is_safe else 'UNSAFE'
    title_color = 'black' if is_safe else 'red'
    ax.set_title(f'{problem_name}: 1D Slice at $x_2 = {center[1]:.2f}$\nStatus: {status}',
                fontsize=14, fontweight='bold', color=title_color)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Mark initial set bounds
    ax.axvline(x=center[0] - radius, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=center[0] + radius, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved to {save_path}")

    return fig


def plot_distance_vs_time(
    problem: UnsafeSupport,
    problem_name: str,
    n_trajectories: int = 50,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot distance to unsafe set over time for sampled trajectories (M-S style).

    This visualization shows:
    - Distance to unsafe set vs time for multiple trajectories (cyan)
    - M-S lower bound as horizontal dashed red line
    - Closest trajectory highlighted in blue

    This is the visualization style used in the Miller-Sznaier paper.

    Args:
        problem: UnsafeSupport problem definition
        problem_name: Name for the title
        n_trajectories: Number of trajectories to sample
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    print(f"  Generating distance vs time plot for {problem_name}...")

    # Get M-S bound
    ms_bound = _get_ms_bound(problem)

    # Sample trajectories and compute distances
    center = problem.initial_center
    radius = problem.initial_radius

    # Get the unsafe set distance function
    dynamics, _, unsafe_set = translate_problem_to_spline_verify(problem)

    # Sample initial conditions from ball
    np.random.seed(42)
    trajectories = []
    min_dist_overall = np.inf
    closest_traj_idx = 0

    n_time_points = 100
    t_eval = np.linspace(0, problem.time_horizon, n_time_points)

    for i in range(n_trajectories):
        # Sample from ball
        angle = 2 * np.pi * np.random.random()
        r = radius * np.sqrt(np.random.random())  # sqrt for uniform in disk
        x0 = center + r * np.array([np.cos(angle), np.sin(angle)])

        # Integrate trajectory
        sol = solve_ivp(
            problem.dynamics, [0, problem.time_horizon], x0,
            t_eval=t_eval, method='RK45'
        )

        if not sol.success:
            continue

        # Compute distance at each time step
        distances = []
        for j in range(len(sol.t)):
            x = sol.y[:, j]
            d = unsafe_set.distance(x)
            distances.append(d)
        distances = np.array(distances)

        min_dist = np.min(distances)
        trajectories.append({
            't': sol.t,
            'x': sol.y,
            'dist': distances,
            'min_dist': min_dist,
            'x0': x0,
        })

        if min_dist < min_dist_overall:
            min_dist_overall = min_dist
            closest_traj_idx = len(trajectories) - 1

    if not trajectories:
        print("    No successful trajectories")
        return None

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot all trajectories (cyan, semi-transparent)
    for i, traj in enumerate(trajectories):
        if i == closest_traj_idx:
            continue  # Plot this one last
        label = 'Trajectories' if i == 0 else None
        ax.plot(traj['t'], traj['dist'], 'c-', alpha=0.4, linewidth=0.8, label=label)

    # Plot closest trajectory (blue, bold)
    closest = trajectories[closest_traj_idx]
    ax.plot(closest['t'], closest['dist'], 'b-', linewidth=2.5,
            label=f'Closest trajectory (min={min_dist_overall:.4f})')

    # Mark minimum point
    min_idx = np.argmin(closest['dist'])
    ax.scatter([closest['t'][min_idx]], [closest['dist'][min_idx]],
               c='blue', s=150, marker='*', zorder=10, edgecolors='white', linewidths=1)

    # M-S lower bound as horizontal line
    if ms_bound is not None:
        ax.axhline(y=ms_bound, color='red', linestyle='--', linewidth=2.5,
                   label=f'M-S lower bound = {ms_bound:.4f}')

    # Zero line (unsafe boundary)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.text(problem.time_horizon * 0.98, 0.02, 'Unsafe (F=0)', fontsize=10,
            ha='right', va='bottom', alpha=0.7)

    ax.set_xlabel('Time $t$', fontsize=14)
    ax.set_ylabel('Distance to unsafe set', fontsize=14)
    ax.set_title(f'{problem_name}\nDistance to Unsafe Set Along Trajectories',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set y-axis to start slightly below min
    ymin = min(0, min_dist_overall * 1.1 if min_dist_overall < 0 else -0.05 * closest['dist'].max())
    ax.set_ylim(bottom=ymin)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"    Saved to {save_path}")

    return fig


def print_summary_table(results: list[ExtendedComparisonResult]) -> None:
    """Print a summary table of all results."""
    print("\n" + "=" * 90)
    print("Comparison Summary")
    print("=" * 90)
    print(f"{'Problem':<18} {'Dim':>4} {'Spline Min':>11} {'M-S Lower':>10} "
          f"{'Gap':>8} {'Spline t':>9} {'M-S t':>8} {'Speedup':>8}")
    print("-" * 90)

    for r in results:
        print(f"{r.problem_name:<18} {r.n_dims:>4} {r.spline_min:>11.4f} "
              f"{r.ms_lower:>10.4f} {r.gap:>8.4f} {r.spline_time:>8.2f}s "
              f"{r.ms_time:>7.2f}s {r.speedup:>7.1f}x")

    print("-" * 90)

    # Averages
    avg_gap = np.mean([r.gap for r in results])
    avg_speedup = np.mean([r.speedup for r in results])
    print(f"{'Average':<18} {'':<4} {'':<11} {'':<10} {avg_gap:>8.4f} "
          f"{'':<9} {'':<8} {avg_speedup:>7.1f}x")


def save_results(results: list[ExtendedComparisonResult], save_path: Path) -> None:
    """Save results to JSON file."""
    data = [r.to_dict() for r in results]
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare spline-verify vs Miller-Sznaier methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--quick', action='store_true',
                       help='Run quick comparison (~1 min)')
    parser.add_argument('--full', action='store_true',
                       help='Run comprehensive comparison (~5 min)')
    parser.add_argument('--save', action='store_true',
                       help='Save figures and results')
    parser.add_argument('--outdir', type=str, default='./results/comparison',
                       help='Output directory')
    parser.add_argument('--viz-options', action='store_true',
                       help='Generate visualization options for comparison')
    args = parser.parse_args()

    # Default to quick if neither specified
    if not args.quick and not args.full:
        args.quick = True

    outdir = Path(args.outdir)
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Spline-Verify vs Miller-Sznaier Comparison")
    print("=" * 60)
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"cvxpy available: {CVXPY_AVAILABLE}")
    if args.save:
        print(f"Output directory: {outdir}")

    if not CVXPY_AVAILABLE:
        print("\nWARNING: cvxpy not available. Miller-Sznaier results will be zeros.")
        print("Install with: pip install cvxpy scs")

    all_results = []
    total_start = time.perf_counter()

    # 1. Standard problems from paper
    print("\n" + "-" * 60)
    print("Standard Problems (from Miller-Sznaier paper)")
    print("-" * 60)

    # Use shorter time horizon for potentially stiff Van der Pol dynamics
    flow = create_flow_system(time_horizon=2.0)
    result = run_single_comparison(flow, "flow_2d")
    if result is not None:
        all_results.append(result)

    if not args.quick:
        twist = create_twist_system(time_horizon=3.0)
        result = run_single_comparison(twist, "twist_3d")
        if result is not None:
            all_results.append(result)

        moon = create_moon_system(time_horizon=2.0)
        result = run_single_comparison(moon, "moon_2d")
        if result is not None:
            all_results.append(result)

    # 2. Custom problems
    print("\n" + "-" * 60)
    print("Custom Problems")
    print("-" * 60)

    custom_problems = create_custom_problems()
    for name, problem in custom_problems:
        result = run_single_comparison(problem, name)
        if result is not None:
            all_results.append(result)

    # 3. Convergence experiments (full mode only)
    sample_results = {}
    order_results = {}

    if not args.quick:
        print("\n" + "-" * 60)
        print("Convergence Experiments")
        print("-" * 60)

        print("\nSample count convergence:")
        sample_counts = [50, 100, 200, 400]
        sample_results = run_sample_count_experiment(flow, sample_counts)

        print("\nSDP order convergence:")
        orders = [2, 3, 4, 5]
        order_results = run_order_experiment(flow, orders)

    total_time = time.perf_counter() - total_start

    # Print summary
    print_summary_table(all_results)
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save results and figures
    if args.save:
        save_results(all_results, outdir / 'comparison_results.json')
        plot_bounds_comparison(all_results, save_path=outdir / 'bounds_comparison.png')
        plot_gap_analysis(all_results, save_path=outdir / 'gap_analysis.png')

        # Flow system variants with different half-space angles
        flow_variants = [
            ('flow_dist_5pi_4', 5 * np.pi / 4, 'Flow (θ=5π/4, lower-left)'),
            ('flow_dist_3pi_2', 3 * np.pi / 2, 'Flow (θ=3π/2, bottom)'),
            ('flow_dist_7pi_4', 7 * np.pi / 4, 'Flow (θ=7π/4, lower-right)'),
        ]

        print("\n" + "-" * 60)
        print("Generating Flow Variant Figures")
        print("-" * 60)

        for variant_name, angle, description in flow_variants:
            print(f"\n  {description}...")
            # Use T=5.0 for sampling comparison (original time horizon)
            flow_variant = create_flow_system(time_horizon=5.0, half_space_angle=angle)

            # Sampling comparison figure
            plot_sampling_comparison(
                flow_variant,
                problem_type='flow',
                n_ball_trajectories=10,
                save_path=outdir / f'{variant_name}_sampling_comparison.png'
            )

            # 3D distance function plot (spline-verify) with T=2.0
            # Uses 500 samples and smoothing=0.001 for better accuracy
            flow_variant_short = create_flow_system(time_horizon=2.0, half_space_angle=angle)
            plot_distance_function_3d(
                flow_variant_short,
                problem_name=description,
                save_path=outdir / f'{variant_name}_distance_function.png'
            )
            plt.close('all')

            # 1D slice plot
            plot_distance_function_1d_slice(
                flow_variant_short,
                problem_name=description,
                save_path=outdir / f'{variant_name}_distance_1d_slice.png'
            )
            plt.close('all')

        # Moon system
        print("\n" + "-" * 60)
        print("Generating Moon System Figures")
        print("-" * 60)

        # Use T=5.0 for sampling comparison (original time horizon)
        moon_full = create_moon_system(time_horizon=5.0)
        plot_sampling_comparison(
            moon_full,
            problem_type='moon',
            n_ball_trajectories=10,
            save_path=outdir / 'moon_sampling_comparison.png'
        )

        # 3D distance function plot (spline-verify) with T=2.0
        # Uses 500 samples and smoothing=0.001 for better accuracy
        print("\n  Moon distance function...")
        moon_short = create_moon_system(time_horizon=2.0)
        plot_distance_function_3d(
            moon_short,
            problem_name='Moon System',
            save_path=outdir / 'moon_distance_function.png'
        )
        plt.close('all')

        # 1D slice plot for moon
        plot_distance_function_1d_slice(
            moon_short,
            problem_name='Moon System',
            save_path=outdir / 'moon_distance_1d_slice.png'
        )
        plt.close('all')

        if not args.quick and sample_results:
            plot_convergence(
                sample_results, order_results,
                save_path=outdir / 'convergence.png'
            )

    # Generate visualization options for comparison
    if args.viz_options:
        print("\n" + "-" * 60)
        print("Generating Visualization Options")
        print("-" * 60)

        outdir_viz = outdir / 'viz_options'
        outdir_viz.mkdir(parents=True, exist_ok=True)

        # Use flow_dist_5pi_4 as the example
        flow_5pi4 = create_flow_system(time_horizon=2.0, half_space_angle=5 * np.pi / 4)

        print("\nFlow system (θ=5π/4):")
        plot_option1_clean_contour(flow_5pi4, save_path=outdir_viz / 'option1_clean_contour.png')
        plt.close('all')

        plot_option2_contour_ms_line(flow_5pi4, save_path=outdir_viz / 'option2_contour_ms_line.png')
        plt.close('all')

        plot_option3_1d_slice(flow_5pi4, save_path=outdir_viz / 'option3_1d_slice.png')
        plt.close('all')

        plot_option4_3d_surface(flow_5pi4, save_path=outdir_viz / 'option4_3d_surface.png')
        plt.close('all')

        plot_option5_samples_comparison(flow_5pi4, save_path=outdir_viz / 'option5_samples.png')
        plt.close('all')

        print(f"\nAll visualization options saved to: {outdir_viz}")

    print("\nComparison complete!")


if __name__ == '__main__':
    main()
