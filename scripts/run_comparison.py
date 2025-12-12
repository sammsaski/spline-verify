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


def plot_distance_function(
    problem: UnsafeSupport,
    problem_name: str,
    n_samples: int = 200,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot the distance function F_T(x0) over the initial set.

    Creates a visualization showing:
    - Contour plot of minimum distance to unsafe set from each initial condition
    - Comparison between spline-verify (computed) and M-S bounds

    Args:
        problem: UnsafeSupport problem definition
        problem_name: Name for the title
        n_samples: Number of samples for computing distance function
        save_path: Optional path to save figure
    """
    from spline_verify.dynamics import ODEDynamics
    from spline_verify.geometry import Ball, LevelSet

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create grid over initial set
    center = problem.initial_center
    radius = problem.initial_radius
    n_grid = 50

    x1_range = np.linspace(center[0] - radius, center[0] + radius, n_grid)
    x2_range = np.linspace(center[1] - radius, center[1] + radius, n_grid)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Compute distance function at each grid point
    F_T = np.full_like(X1, np.nan)

    # Create dynamics
    dynamics = ODEDynamics(f=problem.dynamics, _n_dims=problem.n_vars)

    # Create unsafe set level function
    def unsafe_distance(x):
        """Compute distance from point x to unsafe set."""
        x = np.asarray(x)
        # Distance to ball boundary
        dist_to_center = np.linalg.norm(x - problem.unsafe_center)
        ball_dist = dist_to_center - problem.unsafe_radius

        if ball_dist > 0:
            # Outside ball
            return ball_dist

        # Inside ball - check constraints
        for g in problem.unsafe_constraints:
            constraint_val = g(x)
            if constraint_val < 0:
                # Outside constraint region
                return -constraint_val

        # Inside unsafe set
        return ball_dist  # Negative

    print(f"  Computing distance function for {problem_name}...")
    for i in range(n_grid):
        for j in range(n_grid):
            x0 = np.array([X1[i, j], X2[i, j]])

            # Check if inside initial ball
            if np.linalg.norm(x0 - center) > radius:
                continue

            # Simulate trajectory
            try:
                sol = solve_ivp(
                    problem.dynamics, [0, problem.time_horizon], x0,
                    t_eval=np.linspace(0, problem.time_horizon, 200),
                    method='RK45'
                )
                if sol.success:
                    # Compute minimum distance along trajectory
                    min_dist = np.inf
                    for k in range(len(sol.t)):
                        pt = sol.y[:, k]
                        d = unsafe_distance(pt)
                        if d < min_dist:
                            min_dist = d
                    F_T[i, j] = min_dist
            except Exception:
                pass

    # Plot contour
    levels = np.linspace(np.nanmin(F_T), np.nanmax(F_T), 20)
    if np.nanmin(F_T) < 0:
        # Include zero level for unsafe boundary
        levels = np.sort(np.unique(np.concatenate([levels, [0]])))

    cs = ax.contourf(X1, X2, F_T, levels=levels, cmap='RdYlGn', alpha=0.8)
    cbar = plt.colorbar(cs, ax=ax, label='Min Distance to Unsafe Set')

    # Add zero contour if exists
    if np.nanmin(F_T) < 0 and np.nanmax(F_T) > 0:
        ax.contour(X1, X2, F_T, levels=[0], colors='black', linewidths=2)

    # Plot initial set boundary
    theta = np.linspace(0, 2*np.pi, 100)
    init_x = center[0] + radius * np.cos(theta)
    init_y = center[1] + radius * np.sin(theta)
    ax.plot(init_x, init_y, 'b-', linewidth=2.5, label='Initial set')

    # Plot unsafe set
    half_angle = getattr(problem, 'half_space_angle', None) or (5 * np.pi / 4)
    angle_deg = np.degrees(half_angle)
    wedge_start = angle_deg - 90
    wedge_end = angle_deg + 90

    wedge = Wedge(problem.unsafe_center, problem.unsafe_radius,
                 wedge_start, wedge_end,
                 facecolor='red', alpha=0.5, edgecolor='red', linewidth=2,
                 label='Unsafe set')
    ax.add_patch(wedge)

    # Labels
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title(f'{problem_name}: Distance Function $F_T(x_0)$', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved distance function plot to {save_path}")

    return fig


def plot_distance_function_moon(
    problem: UnsafeSupport,
    problem_name: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot the distance function F_T(x0) for moon system.

    Args:
        problem: UnsafeSupport problem definition (moon)
        problem_name: Name for the title
        save_path: Optional path to save figure
    """
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create grid over initial set
    center = problem.initial_center
    radius = problem.initial_radius
    n_grid = 50

    x1_range = np.linspace(center[0] - radius, center[0] + radius, n_grid)
    x2_range = np.linspace(center[1] - radius, center[1] + radius, n_grid)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    # Moon parameters
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

    def moon_distance(x):
        """Compute distance from point x to moon unsafe set."""
        x = np.asarray(x)
        # Distance to outer circle
        dist_to_outer = np.linalg.norm(x - outer_center) - outer_radius
        # Distance to inner circle (negative means inside)
        dist_to_inner = np.linalg.norm(x - inner_center) - inner_radius

        if dist_to_outer > 0:
            # Outside outer circle
            return dist_to_outer
        elif dist_to_inner < 0:
            # Inside inner circle (excluded from moon)
            return -dist_to_inner
        else:
            # Inside moon (outer but not inner)
            return min(dist_to_outer, 0)  # Negative

    # Compute distance function
    F_T = np.full_like(X1, np.nan)

    print(f"  Computing distance function for {problem_name}...")
    for i in range(n_grid):
        for j in range(n_grid):
            x0 = np.array([X1[i, j], X2[i, j]])

            # Check if inside initial ball
            if np.linalg.norm(x0 - center) > radius:
                continue

            # Simulate trajectory
            try:
                sol = solve_ivp(
                    problem.dynamics, [0, problem.time_horizon], x0,
                    t_eval=np.linspace(0, problem.time_horizon, 200),
                    method='RK45'
                )
                if sol.success:
                    min_dist = np.inf
                    for k in range(len(sol.t)):
                        pt = sol.y[:, k]
                        d = moon_distance(pt)
                        if d < min_dist:
                            min_dist = d
                    F_T[i, j] = min_dist
            except Exception:
                pass

    # Plot contour
    levels = np.linspace(np.nanmin(F_T), np.nanmax(F_T), 20)
    if np.nanmin(F_T) < 0 and np.nanmax(F_T) > 0:
        levels = np.sort(np.unique(np.concatenate([levels, [0]])))

    cs = ax.contourf(X1, X2, F_T, levels=levels, cmap='RdYlGn', alpha=0.8)
    plt.colorbar(cs, ax=ax, label='Min Distance to Unsafe Set')

    if np.nanmin(F_T) < 0 and np.nanmax(F_T) > 0:
        ax.contour(X1, X2, F_T, levels=[0], colors='black', linewidths=2)

    # Plot initial set boundary
    theta = np.linspace(0, 2*np.pi, 100)
    init_x = center[0] + radius * np.cos(theta)
    init_y = center[1] + radius * np.sin(theta)
    ax.plot(init_x, init_y, 'b-', linewidth=2.5, label='Initial set')

    # Plot moon unsafe set
    n_pts = 200
    theta_fine = np.linspace(0, 2*np.pi, n_pts)

    outer_arc = []
    for t in theta_fine:
        pt = outer_center + outer_radius * np.array([np.cos(t), np.sin(t)])
        if np.linalg.norm(pt - inner_center) >= inner_radius:
            outer_arc.append(pt)

    inner_arc = []
    for t in theta_fine:
        pt = inner_center + inner_radius * np.array([np.cos(t), np.sin(t)])
        if np.linalg.norm(pt - outer_center) <= outer_radius:
            inner_arc.append(pt)

    if outer_arc and inner_arc:
        outer_arc = np.array(outer_arc)
        inner_arc = np.array(inner_arc)
        crescent_pts = np.vstack([outer_arc, inner_arc[::-1]])
        crescent = Polygon(crescent_pts, facecolor='red', alpha=0.5,
                          edgecolor='red', linewidth=2, label='Unsafe set (moon)')
        ax.add_patch(crescent)

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title(f'{problem_name}: Distance Function $F_T(x_0)$', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved distance function plot to {save_path}")

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
            flow_variant = create_flow_system(time_horizon=5.0, half_space_angle=angle)

            # Sampling comparison figure
            plot_sampling_comparison(
                flow_variant,
                problem_type='flow',
                n_ball_trajectories=10,
                save_path=outdir / f'{variant_name}_sampling_comparison.png'
            )

            # Distance function figure
            plot_distance_function(
                flow_variant,
                problem_name=description,
                save_path=outdir / f'{variant_name}_distance_function.png'
            )
            plt.close('all')

        # Moon system
        print("\n" + "-" * 60)
        print("Generating Moon System Figures")
        print("-" * 60)

        moon_full = create_moon_system(time_horizon=5.0)
        plot_sampling_comparison(
            moon_full,
            problem_type='moon',
            n_ball_trajectories=10,
            save_path=outdir / 'moon_sampling_comparison.png'
        )

        # Distance function for moon (need to handle differently)
        print("\n  Moon distance function...")
        plot_distance_function_moon(
            moon_full,
            problem_name='Moon System',
            save_path=outdir / 'moon_distance_function.png'
        )
        plt.close('all')

        if not args.quick and sample_results:
            plot_convergence(
                sample_results, order_results,
                save_path=outdir / 'convergence.png'
            )

    print("\nComparison complete!")


if __name__ == '__main__':
    main()
