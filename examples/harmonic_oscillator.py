"""Harmonic oscillator verification example.

The harmonic oscillator has dynamics:
    dx/dt = y
    dy/dt = -omega^2 * x

Solutions are circles/ellipses in phase space with radius sqrt(x^2 + y^2/omega^2).
The radius is conserved, so trajectories never leave their initial orbit.

This allows us to construct SAFE and UNSAFE examples with known answers.

Usage:
    python harmonic_oscillator.py                        # Run without visualization
    python harmonic_oscillator.py --save                 # Save plots to current directory
    python harmonic_oscillator.py --save --outdir ./figs # Save plots to specific directory
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spline_verify.dynamics import ODEDynamics
from spline_verify.geometry import HyperRectangle, Ball, HalfSpace
from spline_verify.geometry.sampling import sample_set, SamplingStrategy
from spline_verify.utils.visualization import plot_set
from spline_verify.verification import SafetyVerifier, VerificationStatus


def example_safe():
    """SAFE example: initial orbits don't reach unsafe region."""
    print("=" * 60)
    print("Harmonic Oscillator - SAFE case")
    print("=" * 60)

    # Unit frequency harmonic oscillator
    dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

    # Initial set: small box near (1, 0)
    # Orbits starting here have radius ~1, so max |x| ~ 1
    initial_set = HyperRectangle(
        lower=np.array([0.8, -0.2]),
        upper=np.array([1.2, 0.2])
    )

    # Unsafe set: region x < -2
    # Since orbits have radius ~1, they never reach x < -2
    unsafe_set = HalfSpace(
        a=np.array([-1.0, 0.0]),  # Points in -x direction
        b=-2.0  # -x <= -2, i.e., x >= 2 is safe
    )
    # Note: HalfSpace is {x : a @ x <= b}
    # We want x < -2, so a = [1, 0], b = -2 means x <= -2

    # Actually let's use a ball instead for easier distance computation
    unsafe_set = Ball(
        center=np.array([-3.0, 0.0]),
        radius=0.5
    )

    # Time horizon: one full period
    T = 2 * np.pi

    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(result.summary())

    # Analytical check: max x coordinate for orbit starting at (1, 0) is 1
    # Distance to unsafe set at (-3, 0) with radius 0.5 is: |1 - (-3)| - 0.5 = 3.5
    # So minimum distance should be around 3.5 - 1 = 2.5 (from opposite side of orbit)
    # Actually min distance = 3 - 0.5 - 1 = 1.5 (when x = -1 on orbit)

    print(f"\nAnalytical min distance: ~1.5")
    print(f"Computed min F_T: {result.min_objective:.3f}")
    print(f"\nExpected: SAFE")
    print(f"Got:      {result.status.name}")

    assert result.status == VerificationStatus.SAFE, \
        f"Expected SAFE, got {result.status.name}"

    return result


def example_unsafe():
    """UNSAFE example: orbit passes through unsafe set."""
    print("\n" + "=" * 60)
    print("Harmonic Oscillator - UNSAFE case")
    print("=" * 60)

    dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

    # Initial set containing points with large enough radius
    initial_set = HyperRectangle(
        lower=np.array([1.5, -0.2]),
        upper=np.array([2.0, 0.2])
    )

    # Unsafe set: ball at (-1.5, 0)
    # Orbits starting at x=1.5 to 2.0 have radius 1.5 to 2.0
    # So they swing to x = -1.5 to -2.0, hitting the unsafe ball
    unsafe_set = Ball(
        center=np.array([-1.8, 0.0]),
        radius=0.3
    )

    T = np.pi  # Half period to reach opposite side

    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(result.summary())
    print(f"\nExpected: UNSAFE (orbits pass through unsafe ball)")
    print(f"Got:      {result.status.name}")

    # Should be unsafe since orbit from (2, 0) reaches (-2, 0)
    assert result.status == VerificationStatus.UNSAFE, \
        f"Expected UNSAFE, got {result.status.name}"

    return result


def verify_conservation():
    """Verify that the integrator conserves energy (orbit radius)."""
    print("\n" + "=" * 60)
    print("Verifying energy conservation")
    print("=" * 60)

    dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

    x0 = np.array([1.0, 0.0])
    T = 4 * np.pi  # Two full periods

    bundle = dynamics.simulate(x0, (0, T))
    traj = bundle.primary

    # Compute energy: E = (x^2 + y^2) / 2
    energies = 0.5 * (traj.states[:, 0]**2 + traj.states[:, 1]**2)

    initial_energy = energies[0]
    energy_drift = np.max(np.abs(energies - initial_energy))

    print(f"Initial energy: {initial_energy:.6f}")
    print(f"Max energy drift: {energy_drift:.6e}")
    print(f"Relative drift: {energy_drift / initial_energy:.6e}")

    assert energy_drift < 1e-4, f"Energy drift too large: {energy_drift}"
    print("Energy conservation: OK")


def visualize(dynamics, initial_set, unsafe_set, T, result, title):
    """Visualize the verification result."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot some orbits
    samples = sample_set(initial_set, 20, SamplingStrategy.LATIN_HYPERCUBE, seed=42)

    for x0 in samples:
        bundle = dynamics.simulate(x0, (0, T))
        traj = bundle.primary
        ax.plot(traj.states[:, 0], traj.states[:, 1], 'b-', alpha=0.5)

    # Plot sets
    plot_set(initial_set, ax, color='green', alpha=0.3, label='Initial set')
    plot_set(unsafe_set, ax, color='red', alpha=0.3, label='Unsafe set')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title}\nResult: {result.status.name}')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Harmonic oscillator verification example")
    parser.add_argument('--save', action='store_true',
                        help='Save plots to files')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory for saved plots (default: current dir)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    verify_conservation()
    result_safe = example_safe()
    result_unsafe = example_unsafe()

    print("\n" + "=" * 60)
    print("All harmonic oscillator examples completed!")
    print("=" * 60)

    # Save figures if requested
    if args.save:
        try:
            dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

            # Safe case
            initial_safe = HyperRectangle(np.array([0.8, -0.2]), np.array([1.2, 0.2]))
            unsafe_safe = Ball(np.array([-3.0, 0.0]), 0.5)
            fig1 = visualize(dynamics, initial_safe, unsafe_safe, 2*np.pi,
                            result_safe, "Harmonic Oscillator - Safe")

            # Unsafe case
            initial_unsafe = HyperRectangle(np.array([1.5, -0.2]), np.array([2.0, 0.2]))
            unsafe_unsafe = Ball(np.array([-1.8, 0.0]), 0.3)
            fig2 = visualize(dynamics, initial_unsafe, unsafe_unsafe, np.pi,
                            result_unsafe, "Harmonic Oscillator - Unsafe")

            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            fig1.savefig(outdir / 'harmonic_oscillator_safe.png', dpi=150, bbox_inches='tight')
            fig2.savefig(outdir / 'harmonic_oscillator_unsafe.png', dpi=150, bbox_inches='tight')
            print(f"\nFigures saved to {outdir.absolute()}")

        except Exception as e:
            print(f"Visualization failed: {e}")
