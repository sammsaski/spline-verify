"""Relay feedback verification example (switching system with sliding mode).

The relay feedback system is a classic example of Filippov dynamics:
    dx/dt = -sign(x)

This is a 1D system where:
    dx/dt = -1 if x > 0 (drives toward origin from right)
    dx/dt = +1 if x < 0 (drives toward origin from left)

At x = 0, the system exhibits sliding mode behavior: the trajectory stays
on the switching surface (the origin in this case).

This example demonstrates:
1. Filippov sliding mode handling
2. Automatic region classification
3. Safety verification for switching systems

Usage:
    python relay_feedback.py                        # Run without visualization
    python relay_feedback.py --save                 # Save plots to current directory
    python relay_feedback.py --save --outdir ./figs # Save plots to specific directory
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spline_verify.dynamics import SwitchingDynamics, FilippovSolver
from spline_verify.geometry import HyperRectangle
from spline_verify.geometry.sampling import sample_set, SamplingStrategy
from spline_verify.verification import (
    SwitchingVerifier,
    VerificationStatus,
)


def example_safe():
    """SAFE example: relay drives system to origin, avoiding unsafe region."""
    print("=" * 60)
    print("Relay Feedback - SAFE case")
    print("=" * 60)

    # Create 1D relay feedback dynamics
    dynamics = SwitchingDynamics.relay_feedback()

    # Initial set: region in positive x (1D)
    initial_set = HyperRectangle(
        lower=np.array([0.5]),
        upper=np.array([1.5])
    )

    # Unsafe set: region x < -3 (far from where trajectories go)
    unsafe_set = HyperRectangle(
        lower=np.array([-10.0]),
        upper=np.array([-3.0])
    )

    T = 2.0  # Time horizon

    # Verify with SwitchingVerifier
    verifier = SwitchingVerifier(n_samples=100, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(result.summary())

    if 'switching_info' in result.details:
        info = result.details['switching_info']
        print(f"\nSwitching Analysis:")
        print(f"  Number of regions: {info['n_regions']}")
        print(f"  Region counts: {info['region_counts']}")

    print(f"\nExpected: SAFE (relay drives to origin, doesn't reach x < -3)")
    print(f"Got:      {result.status.name}")

    return result


def example_unsafe():
    """UNSAFE example: trajectories pass through unsafe region at origin."""
    print("\n" + "=" * 60)
    print("Relay Feedback - UNSAFE case")
    print("=" * 60)

    dynamics = SwitchingDynamics.relay_feedback()

    # Initial set: region containing points on both sides of origin
    initial_set = HyperRectangle(
        lower=np.array([-0.5]),
        upper=np.array([0.5])
    )

    # Unsafe set: small region around origin (which relay reaches)
    unsafe_set = HyperRectangle(
        lower=np.array([-0.1]),
        upper=np.array([0.1])
    )

    T = 2.0

    verifier = SwitchingVerifier(n_samples=100, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(result.summary())
    print(f"\nExpected: UNSAFE (trajectories converge to origin)")
    print(f"Got:      {result.status.name}")

    return result


def demonstrate_sliding_mode():
    """Demonstrate sliding mode behavior in relay feedback."""
    print("\n" + "=" * 60)
    print("Demonstrating Sliding Mode Behavior")
    print("=" * 60)

    dynamics = SwitchingDynamics.relay_feedback()
    solver = FilippovSolver(dynamics)

    # Initial condition in positive region
    x0 = np.array([1.0])
    T = 3.0

    print(f"Initial condition: x = {x0[0]}")
    bundle = solver.solve(x0, (0, T))
    traj = bundle.primary

    print(f"Final state: x = {traj.final_state[0]:.6f}")
    print(f"Trajectory points: {len(traj)}")

    # Check for sliding (x near 0 for extended period)
    x_values = traj.states[:, 0]
    near_zero = np.abs(x_values) < 0.05
    sliding_fraction = np.sum(near_zero) / len(x_values)
    print(f"Fraction of time near x=0: {sliding_fraction:.2%}")

    return traj


def visualize_safe_case(dynamics, initial_set, unsafe_set, T, result):
    """Visualize the safe case verification result."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    solver = FilippovSolver(dynamics)

    # Plot trajectories over time
    samples = sample_set(initial_set, 20, SamplingStrategy.LATIN_HYPERCUBE, seed=42)

    for x0 in samples:
        bundle = solver.solve(x0, (0, T))
        traj = bundle.primary
        ax1.plot(traj.times, traj.states[:, 0], 'b-', alpha=0.4)
        ax1.plot(0, x0[0], 'go', markersize=6)

    # Plot initial set bounds
    ax1.axhline(y=initial_set.lower[0], color='green', linestyle='-', linewidth=2,
                label='Initial set')
    ax1.axhline(y=initial_set.upper[0], color='green', linestyle='-', linewidth=2)

    # Plot unsafe region
    ax1.axhspan(unsafe_set.lower[0], unsafe_set.upper[0], alpha=0.3, color='red',
                label='Unsafe region')

    # Plot switching surface
    ax1.axhline(y=0, color='orange', linestyle='--', linewidth=2,
                label='Switching surface (x=0)')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('x')
    ax1.set_title(f'Relay Feedback - Safe Case\nResult: {result.status.name}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot objective function F_T over initial set
    n_vis = 100
    x0_vis = np.linspace(initial_set.lower[0], initial_set.upper[0], n_vis)
    F_T_values = []

    for x_init in x0_vis:
        bundle = solver.solve(np.array([x_init]), (0, T))
        min_dist = bundle.min_distance_to_set(unsafe_set.distance)
        F_T_values.append(min_dist)

    ax2.plot(x0_vis, F_T_values, 'b-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', label='Unsafe threshold')
    ax2.axhline(y=result.error_bound, color='orange', linestyle='--',
                label=f'Error bound = {result.error_bound:.3f}')
    ax2.fill_between(x0_vis, 0, result.error_bound, alpha=0.2, color='orange',
                     label='Uncertainty region')

    ax2.set_xlabel('Initial x')
    ax2.set_ylabel('F_T (min distance to unsafe)')
    ax2.set_title('Objective Function over Initial Set')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_unsafe_case(dynamics, initial_set, unsafe_set, T, result):
    """Visualize the unsafe case verification result."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    solver = FilippovSolver(dynamics)

    # Plot trajectories over time
    samples = sample_set(initial_set, 30, SamplingStrategy.LATIN_HYPERCUBE, seed=42)

    for x0 in samples:
        bundle = solver.solve(x0, (0, T))
        traj = bundle.primary
        ax1.plot(traj.times, traj.states[:, 0], 'b-', alpha=0.4)
        ax1.plot(0, x0[0], 'go', markersize=6)

    # Plot initial set bounds
    ax1.axhline(y=initial_set.lower[0], color='green', linestyle='-', linewidth=2,
                label='Initial set')
    ax1.axhline(y=initial_set.upper[0], color='green', linestyle='-', linewidth=2)

    # Plot unsafe region (highlighted since it's reached)
    ax1.axhspan(unsafe_set.lower[0], unsafe_set.upper[0], alpha=0.4, color='red',
                label='Unsafe region')

    # Plot switching surface
    ax1.axhline(y=0, color='orange', linestyle='--', linewidth=2,
                label='Switching surface (x=0)')

    # Mark counterexample if available
    if result.counterexample is not None:
        ax1.plot(0, result.counterexample[0], 'r*', markersize=15,
                 label=f'Counterexample x0={result.counterexample[0]:.2f}')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('x')
    ax1.set_title(f'Relay Feedback - Unsafe Case\nResult: {result.status.name}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot objective function F_T over initial set
    n_vis = 100
    x0_vis = np.linspace(initial_set.lower[0], initial_set.upper[0], n_vis)
    F_T_values = []

    for x_init in x0_vis:
        bundle = solver.solve(np.array([x_init]), (0, T))
        min_dist = bundle.min_distance_to_set(unsafe_set.distance)
        F_T_values.append(min_dist)

    ax2.plot(x0_vis, F_T_values, 'b-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Unsafe threshold')

    # Highlight where F_T <= 0
    ax2.fill_between(x0_vis, F_T_values, 0,
                     where=[v <= 0 for v in F_T_values],
                     alpha=0.3, color='red', label='Reaches unsafe')

    ax2.set_xlabel('Initial x')
    ax2.set_ylabel('F_T (min distance to unsafe)')
    ax2.set_title('Objective Function over Initial Set\n(negative = reaches unsafe)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_sliding_trajectory(traj):
    """Plot the sliding mode trajectory."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Time evolution
    ax.plot(traj.times, traj.states[:, 0], 'b-', linewidth=2, label='x(t)')
    ax.axhline(y=0, color='orange', linestyle='--', linewidth=2,
               label='Switching surface (x=0)')

    ax.plot(0, traj.states[0, 0], 'go', markersize=12, label='Start')
    ax.plot(traj.times[-1], traj.states[-1, 0], 'rs', markersize=12, label='End')

    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_title('Relay Feedback: Trajectory converging to origin\n(dx/dt = -sign(x))')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Relay feedback verification example")
    parser.add_argument('--save', action='store_true',
                        help='Save plots to files')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory for saved plots (default: current dir)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Run examples
    sliding_traj = demonstrate_sliding_mode()
    result_safe = example_safe()
    result_unsafe = example_unsafe()

    print("\n" + "=" * 60)
    print("All relay feedback examples completed!")
    print("=" * 60)

    # Save figures if requested
    if args.save:
        try:
            dynamics = SwitchingDynamics.relay_feedback()

            # Safe case visualization
            initial_safe = HyperRectangle(np.array([0.5]), np.array([1.5]))
            unsafe_safe = HyperRectangle(np.array([-10.0]), np.array([-3.0]))
            fig1 = visualize_safe_case(
                dynamics, initial_safe, unsafe_safe, 2.0, result_safe
            )

            # Unsafe case visualization
            initial_unsafe = HyperRectangle(np.array([-0.5]), np.array([0.5]))
            unsafe_unsafe = HyperRectangle(np.array([-0.1]), np.array([0.1]))
            fig2 = visualize_unsafe_case(
                dynamics, initial_unsafe, unsafe_unsafe, 2.0, result_unsafe
            )

            # Sliding mode visualization
            fig3 = visualize_sliding_trajectory(sliding_traj)

            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            fig1.savefig(outdir / 'relay_feedback_safe.png', dpi=150, bbox_inches='tight')
            fig2.savefig(outdir / 'relay_feedback_unsafe.png', dpi=150, bbox_inches='tight')
            fig3.savefig(outdir / 'relay_feedback_sliding.png', dpi=150, bbox_inches='tight')
            print(f"\nFigures saved to {outdir.absolute()}")

        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
