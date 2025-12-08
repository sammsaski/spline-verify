"""Bouncing ball example (switching system placeholder for Phase 4).

The bouncing ball is a canonical switching system:
- Mode 1: Free fall (y >= 0): dy/dt = v, dv/dt = -g
- Mode 2: Bounce (y = 0, v < 0): velocity reverses with restitution

This is a placeholder demonstrating the switching dynamics API.
Full Filippov handling will be implemented in Phase 4.

Usage:
    python bouncing_ball.py                        # Run without visualization
    python bouncing_ball.py --save                 # Save plots to current directory
    python bouncing_ball.py --save --outdir ./figs # Save plots to specific directory
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spline_verify.dynamics.switching import SwitchingDynamics
from spline_verify.geometry import HyperRectangle, Ball
from spline_verify.verification import SafetyVerifier, VerificationStatus


def simulate_bouncing_ball():
    """Demonstrate bouncing ball simulation."""
    print("=" * 60)
    print("Bouncing Ball Simulation (Phase 4 Placeholder)")
    print("=" * 60)

    # Create bouncing ball dynamics
    dynamics = SwitchingDynamics.bouncing_ball(gravity=9.81, restitution=0.9)

    # Initial condition: ball at height 1m, velocity 0
    x0 = np.array([1.0, 0.0])  # [height, velocity]

    # Simulate for 3 seconds
    T = 3.0
    bundle = dynamics.simulate(x0, (0, T))
    traj = bundle.primary

    print(f"Simulated {len(traj)} time points")
    print(f"Initial state: height={x0[0]:.2f}m, velocity={x0[1]:.2f}m/s")
    print(f"Final state: height={traj.final_state[0]:.3f}m, velocity={traj.final_state[1]:.3f}m/s")

    # Note: Current simple implementation doesn't handle bounces correctly
    # Full implementation in Phase 4 will properly handle:
    # - Event detection for ground contact
    # - Velocity reversal with restitution
    # - Filippov sliding modes

    return traj


def example_safe_height():
    """Example: verify ball stays above minimum height."""
    print("\n" + "=" * 60)
    print("Bouncing Ball - Safety Verification")
    print("=" * 60)

    dynamics = SwitchingDynamics.bouncing_ball(gravity=9.81)

    # Initial set: ball starts between heights 0.8 and 1.2, small velocity
    initial_set = HyperRectangle(
        lower=np.array([0.8, -0.5]),
        upper=np.array([1.2, 0.5])
    )

    # Unsafe set: ball going too fast downward (dangerous impact)
    # We consider velocity < -5 m/s as unsafe
    unsafe_set = HyperRectangle(
        lower=np.array([-10.0, -100.0]),
        upper=np.array([10.0, -5.0])
    )

    T = 1.0  # Short time horizon

    verifier = SafetyVerifier(n_samples=100, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(result.summary())
    print("\nNote: This uses placeholder switching simulation.")
    print("Full Filippov handling will be implemented in Phase 4.")

    return result


def visualize_trajectory(traj):
    """Plot bouncing ball trajectory."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Height vs time
    ax1.plot(traj.times, traj.states[:, 0], 'b-')
    ax1.axhline(y=0, color='k', linestyle='--', label='Ground')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Bouncing Ball Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Velocity vs time
    ax2.plot(traj.times, traj.states[:, 1], 'r-')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_phase_portrait(traj):
    """Plot phase portrait (height vs velocity)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(traj.states[:, 0], traj.states[:, 1], 'b-', alpha=0.7)
    ax.plot(traj.states[0, 0], traj.states[0, 1], 'go', markersize=10, label='Start')
    ax.plot(traj.states[-1, 0], traj.states[-1, 1], 'rs', markersize=10, label='End')

    ax.axvline(x=0, color='k', linestyle='--', label='Ground')
    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Bouncing Ball Phase Portrait')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Bouncing ball example (Phase 4 placeholder)")
    parser.add_argument('--save', action='store_true',
                        help='Save plots to files')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory for saved plots (default: current dir)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Run simulation
    traj = simulate_bouncing_ball()

    # Run safety verification
    result = example_safe_height()

    print("\n" + "=" * 60)
    print("Bouncing ball example completed!")
    print("Note: Full switching system support is Phase 4 work.")
    print("=" * 60)

    # Save figures if requested
    if args.save:
        try:
            fig1 = visualize_trajectory(traj)
            fig2 = visualize_phase_portrait(traj)

            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            fig1.savefig(outdir / 'bouncing_ball_trajectory.png', dpi=150, bbox_inches='tight')
            fig2.savefig(outdir / 'bouncing_ball_phase.png', dpi=150, bbox_inches='tight')
            print(f"\nFigures saved to {outdir.absolute()}")

        except Exception as e:
            print(f"Visualization failed: {e}")
