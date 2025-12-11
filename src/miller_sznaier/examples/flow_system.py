"""Flow system example from Miller & Sznaier's paper.

This reproduces the main example (Fig. 5) from:
    "Bounding the Distance to Unsafe Sets with Convex Optimization"

The system is a Van der Pol-like oscillator:
    ẋ₁ = x₂
    ẋ₂ = -x₁ + (1/3)x₁³ - x₂

With:
    Initial set: Circle centered at (1.5, 0) with radius 0.4
    Unsafe set: Half-disk at (0, -0.7) with radius 0.5

Expected result from paper: L2 distance ≈ 0.2831
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..problem import create_flow_system, create_moon_system, UnsafeSupport
from ..distance_estimator import DistanceEstimator, CVXPY_AVAILABLE


def simulate_trajectories(
    problem: UnsafeSupport,
    n_trajectories: int = 50,
    dt: float = 0.01
) -> list[np.ndarray]:
    """Simulate multiple trajectories from the initial set.

    Args:
        problem: Problem specification.
        n_trajectories: Number of trajectories to simulate.
        dt: Time step.

    Returns:
        List of trajectory arrays.
    """
    trajectories = []
    n_steps = int(problem.time_horizon / dt) + 1

    np.random.seed(42)
    for _ in range(n_trajectories):
        # Sample from initial set
        direction = np.random.randn(problem.n_vars)
        direction /= np.linalg.norm(direction)
        r = np.random.uniform(0, problem.initial_radius)
        x0 = problem.initial_center + r * direction

        # Simulate
        traj = np.zeros((n_steps, problem.n_vars))
        traj[0] = x0
        x = x0.copy()
        t = 0.0

        for i in range(1, n_steps):
            dx = problem.dynamics(t, x)
            x = x + dt * dx
            t += dt
            traj[i] = x

        trajectories.append(traj)

    return trajectories


def plot_flow_system(
    problem: UnsafeSupport,
    trajectories: list[np.ndarray],
    title: str = "Flow System Trajectories",
    save_path: Optional[Path] = None,
    show_distance_contours: bool = True,
) -> plt.Figure:
    """Plot the Flow system with trajectories.

    Args:
        problem: Problem specification.
        trajectories: List of trajectory arrays.
        title: Plot title.
        save_path: Path to save figure.
        show_distance_contours: If True, show distance contours.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot initial set
    theta = np.linspace(0, 2*np.pi, 100)
    x_init = problem.initial_center[0] + problem.initial_radius * np.cos(theta)
    y_init = problem.initial_center[1] + problem.initial_radius * np.sin(theta)
    ax.fill(x_init, y_init, alpha=0.3, color='blue', label='Initial Set')
    ax.plot(x_init, y_init, 'b-', linewidth=2)

    # Plot unsafe set (half-disk)
    x_unsafe = problem.unsafe_center[0] + problem.unsafe_radius * np.cos(theta)
    y_unsafe = problem.unsafe_center[1] + problem.unsafe_radius * np.sin(theta)

    # Apply half-space constraint if present
    if problem.unsafe_constraints:
        # Filter points by half-space
        mask = np.array([problem.in_unsafe_set(np.array([x, y]))
                        for x, y in zip(x_unsafe, y_unsafe)])
        # Fill only the valid region
        ax.fill(x_unsafe[mask], y_unsafe[mask], alpha=0.3, color='red', label='Unsafe Set')
    else:
        ax.fill(x_unsafe, y_unsafe, alpha=0.3, color='red', label='Unsafe Set')
    ax.plot(x_unsafe, y_unsafe, 'r--', linewidth=1, alpha=0.5)

    # Plot trajectories
    for i, traj in enumerate(trajectories):
        color = plt.cm.viridis(i / len(trajectories))
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=0.5, linewidth=0.5)

        # Mark start and end
        if i == 0:
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=4, label='Start')
            ax.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=4, label='End')
        else:
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=4)
            ax.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=4)

    # Distance contours
    if show_distance_contours:
        x_grid = np.linspace(-3, 3, 100)
        y_grid = np.linspace(-2, 2, 100)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Distance to unsafe set center
        dist = np.sqrt((X - problem.unsafe_center[0])**2 +
                      (Y - problem.unsafe_center[1])**2) - problem.unsafe_radius
        dist = np.maximum(dist, 0)

        contours = ax.contour(X, Y, dist, levels=[0.1, 0.2, 0.3, 0.5, 1.0],
                             colors='gray', alpha=0.5, linestyles='dotted')
        ax.clabel(contours, inline=True, fontsize=8)

    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def run_flow_example(
    order: int = 4,
    n_trajectories: int = 50,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """Run the complete Flow system example.

    Args:
        order: Moment relaxation order for SDP.
        n_trajectories: Number of trajectories to simulate.
        save_dir: Directory to save figures.
        verbose: Whether to print results.

    Returns:
        Dict with results and figures.
    """
    results = {}

    # Create Flow system
    problem = create_flow_system()

    if verbose:
        print("=" * 60)
        print("Flow System Example (Miller & Sznaier)")
        print("=" * 60)
        print(f"Initial set: Ball at {problem.initial_center}, r={problem.initial_radius}")
        print(f"Unsafe set: Half-disk at {problem.unsafe_center}, r={problem.unsafe_radius}")
        print(f"Time horizon: T={problem.time_horizon}")
        print()

    # Simulate trajectories
    if verbose:
        print("Simulating trajectories...")
    trajectories = simulate_trajectories(problem, n_trajectories)
    results['trajectories'] = trajectories

    # Compute minimum distance from trajectories
    min_dist = float('inf')
    for traj in trajectories:
        for x in traj:
            diff = x - problem.unsafe_center
            dist = max(0, np.linalg.norm(diff) - problem.unsafe_radius)
            min_dist = min(min_dist, dist)

    results['sampled_min_dist'] = min_dist
    if verbose:
        print(f"Minimum distance from sampling: {min_dist:.6f}")

    # Run SDP estimation if cvxpy available
    if CVXPY_AVAILABLE:
        if verbose:
            print(f"\nRunning SDP estimation (order={order})...")

        estimator = DistanceEstimator(order=order, verbose=False)
        sdp_result = estimator.estimate(problem, compute_upper_bound=False)
        results['sdp_result'] = sdp_result

        if verbose:
            print(f"SDP lower bound: {sdp_result.lower_bound:.6f}")
            print(f"Solve time: {sdp_result.solve_time:.3f}s")
            print(f"Solver status: {sdp_result.status}")

        # Compare with paper's result
        paper_bound = 0.2831
        if verbose:
            print(f"\nPaper's bound: {paper_bound:.4f}")
            print(f"Our bound: {sdp_result.lower_bound:.4f}")
    else:
        if verbose:
            print("\nNote: cvxpy not available, skipping SDP estimation")
            print("Install with: pip install cvxpy scs")

    # Create visualization
    if save_dir:
        save_path = Path(save_dir) / "flow_trajectories.png"
    else:
        save_path = None

    fig = plot_flow_system(
        problem, trajectories,
        title="Flow System: Trajectories and Unsafe Set",
        save_path=save_path
    )
    results['figure'] = fig

    if verbose:
        print()
        print("=" * 60)

    return results


def main():
    """Command-line interface for Flow example."""
    parser = argparse.ArgumentParser(description="Flow system distance estimation example")
    parser.add_argument('--order', type=int, default=4, help='Moment relaxation order')
    parser.add_argument('--n-traj', type=int, default=50, help='Number of trajectories')
    parser.add_argument('--save', action='store_true', help='Save figures')
    parser.add_argument('--outdir', type=str, default='./figs/miller_sznaier',
                       help='Output directory')
    args = parser.parse_args()

    save_dir = Path(args.outdir) if args.save else None

    run_flow_example(
        order=args.order,
        n_trajectories=args.n_traj,
        save_dir=save_dir,
        verbose=True,
    )


if __name__ == '__main__':
    main()
