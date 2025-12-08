"""Linear system verification example.

This example demonstrates verification on a simple linear ODE:
    dx/dt = Ax

where A is a stable matrix (eigenvalues with negative real parts).

Example 1: UNSAFE case
- Initial set: box around origin
- Unsafe set: small ball at origin
- Result: UNSAFE (trajectories converge to origin)

Example 2: SAFE case
- Initial set: box away from origin
- Unsafe set: small ball at origin
- Result: SAFE (trajectories converge but start far enough away)

Usage:
    python linear_system.py                        # Run without visualization
    python linear_system.py --save                 # Save plots to current directory
    python linear_system.py --save --outdir ./figs # Save plots to specific directory
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spline_verify.dynamics import ODEDynamics
from spline_verify.geometry import HyperRectangle, Ball
from spline_verify.geometry.sampling import sample_set, SamplingStrategy
from spline_verify.utils.visualization import plot_set
from spline_verify.verification import SafetyVerifier, VerificationStatus
from spline_verify.verification.objective import ObjectiveSampler


def example_unsafe():
    """Example where system is UNSAFE - trajectories pass through unsafe set."""
    print("=" * 60)
    print("Example 1: Linear system - UNSAFE case")
    print("=" * 60)

    # Stable linear system: dx/dt = Ax
    # Eigenvalues: -1 +/- i (stable spiral)
    A = np.array([
        [-1, 1],
        [-1, -1]
    ])
    dynamics = ODEDynamics.from_matrix(A)

    # Initial set: box containing origin
    initial_set = HyperRectangle(
        lower=np.array([-1.0, -1.0]),
        upper=np.array([1.0, 1.0])
    )

    # Unsafe set: small ball at origin
    unsafe_set = Ball(
        center=np.array([0.0, 0.0]),
        radius=0.1
    )

    # Time horizon
    T = 5.0

    # Create verifier and run
    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(result.summary())
    print(f"\nExpected: UNSAFE (trajectories converge to origin)")
    print(f"Got:      {result.status.name}")

    assert result.status == VerificationStatus.UNSAFE, \
        f"Expected UNSAFE, got {result.status.name}"

    return result


def example_safe():
    """Example where system is SAFE - trajectories don't reach unsafe set."""
    print("\n" + "=" * 60)
    print("Example 2: Linear system - SAFE case")
    print("=" * 60)

    # Same stable linear system
    A = np.array([
        [-1, 1],
        [-1, -1]
    ])
    dynamics = ODEDynamics.from_matrix(A)

    # Initial set: box far from origin
    initial_set = HyperRectangle(
        lower=np.array([2.0, 2.0]),
        upper=np.array([3.0, 3.0])
    )

    # Unsafe set: small ball at origin
    # The key is that for this time horizon, trajectories don't reach it
    unsafe_set = Ball(
        center=np.array([0.0, 0.0]),
        radius=0.1
    )

    # Short time horizon so trajectories don't reach origin
    T = 1.0

    verifier = SafetyVerifier(n_samples=300, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(result.summary())
    print(f"\nExpected: SAFE (trajectories don't reach origin in time T={T})")
    print(f"Got:      {result.status.name}")

    # For short time horizon, should be safe
    # (may return UNKNOWN if margin is small)
    assert result.status in [VerificationStatus.SAFE, VerificationStatus.UNKNOWN], \
        f"Expected SAFE or UNKNOWN, got {result.status.name}"

    return result


def visualize_example(dynamics, initial_set, unsafe_set, T, result, title):
    """Visualize the verification result."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot some trajectories
    from spline_verify.geometry.sampling import sample_set, SamplingStrategy

    samples = sample_set(initial_set, 20, SamplingStrategy.LATIN_HYPERCUBE, seed=42)

    for x0 in samples:
        bundle = dynamics.simulate(x0, (0, T))
        traj = bundle.primary
        ax1.plot(traj.states[:, 0], traj.states[:, 1], 'b-', alpha=0.3)
        ax1.plot(x0[0], x0[1], 'go', markersize=4)

    # Plot sets
    from spline_verify.utils.visualization import plot_set
    plot_set(initial_set, ax1, color='green', alpha=0.2, label='Initial set')
    plot_set(unsafe_set, ax1, color='red', alpha=0.3, label='Unsafe set')

    # Plot minimizer
    ax1.plot(result.minimizer[0], result.minimizer[1], 'k*',
             markersize=15, label=f'Min F_T={result.min_objective:.3f}')

    ax1.set_xlabel('x[0]')
    ax1.set_ylabel('x[1]')
    ax1.set_title(f'{title}\nResult: {result.status.name}')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot F_T over initial set
    from spline_verify.verification.objective import ObjectiveSampler
    sampler = ObjectiveSampler(dynamics, initial_set, unsafe_set, T)
    points, values = sampler.sample(500, seed=42)

    scatter = ax2.scatter(points[:, 0], points[:, 1], c=values, cmap='viridis', s=10)
    plt.colorbar(scatter, ax=ax2, label='F_T (distance to unsafe)')
    ax2.set_xlabel('x[0]')
    ax2.set_ylabel('x[1]')
    ax2.set_title('Objective function F_T over initial set')
    ax2.set_aspect('equal')

    plt.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Linear system verification example")
    parser.add_argument('--save', action='store_true',
                        help='Save plots to files')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory for saved plots (default: current dir)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Run examples
    result_unsafe = example_unsafe()
    result_safe = example_safe()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

    # Save figures if requested
    if args.save:
        try:
            A = np.array([[-1, 1], [-1, -1]])
            dynamics = ODEDynamics.from_matrix(A)

            # Visualize unsafe case
            initial_unsafe = HyperRectangle(np.array([-1., -1.]), np.array([1., 1.]))
            unsafe_set = Ball(np.array([0., 0.]), 0.1)
            fig1 = visualize_example(dynamics, initial_unsafe, unsafe_set, 5.0,
                                    result_unsafe, "Linear System - Unsafe Case")

            # Visualize safe case
            initial_safe = HyperRectangle(np.array([2., 2.]), np.array([3., 3.]))
            fig2 = visualize_example(dynamics, initial_safe, unsafe_set, 1.0,
                                    result_safe, "Linear System - Safe Case")

            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            fig1.savefig(outdir / 'linear_system_unsafe.png', dpi=150, bbox_inches='tight')
            fig2.savefig(outdir / 'linear_system_safe.png', dpi=150, bbox_inches='tight')
            print(f"\nFigures saved to {outdir.absolute()}")

        except Exception as e:
            print(f"Visualization failed: {e}")
