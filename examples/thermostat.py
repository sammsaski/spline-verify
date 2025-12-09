"""Thermostat verification example (switching system with hysteresis).

The thermostat is a classic hybrid system with hysteresis switching:
    dT/dt = -alpha * (T - T_env) + beta * heater_on

The heater switches ON when T < T_low and OFF when T > T_high,
creating a hysteresis band [T_low, T_high].

This example demonstrates:
1. Hysteresis switching (different thresholds for on/off)
2. State-dependent mode changes
3. Safety verification with temperature bounds

Usage:
    python thermostat.py                        # Run without visualization
    python thermostat.py --save                 # Save plots to current directory
    python thermostat.py --save --outdir ./figs # Save plots to specific directory
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


def example_safe_temperature_band():
    """SAFE example: temperature stays within acceptable band."""
    print("=" * 60)
    print("Thermostat - SAFE case (temperature regulation)")
    print("=" * 60)

    # Create thermostat dynamics with hysteresis band [18, 22]
    # T_ambient = 10 (cold environment), heater provides warming
    dynamics = SwitchingDynamics.thermostat(
        T_low=18.0,
        T_high=22.0,
        T_ambient=10.0,
        cooling_rate=0.1,  # Heat loss rate
        heating_power=2.0,   # Heater power
    )

    # Initial set: temperature in [19, 21] (within hysteresis band)
    # Note: thermostat is 1D, but we add a dummy dimension for compatibility
    initial_set = HyperRectangle(
        lower=np.array([19.0]),
        upper=np.array([21.0])
    )

    # Unsafe set: temperature below 15 (too cold) or above 25 (too hot)
    # We'll check the "too cold" case
    unsafe_set = HyperRectangle(
        lower=np.array([0.0]),
        upper=np.array([15.0])
    )

    T = 10.0  # Time horizon to test regulation

    # Verify
    verifier = SwitchingVerifier(n_samples=50, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(result.summary())

    if 'switching_info' in result.details:
        info = result.details['switching_info']
        print(f"\nSwitching Analysis:")
        print(f"  Number of regions: {info['n_regions']}")
        print(f"  Region counts: {info['region_counts']}")

    print(f"\nExpected: SAFE (temperature stays in regulation band)")
    print(f"Got:      {result.status.name}")

    return result


def example_verify_overheat():
    """Verify that temperature doesn't overheat."""
    print("\n" + "=" * 60)
    print("Thermostat - Verify no overheating")
    print("=" * 60)

    dynamics = SwitchingDynamics.thermostat(
        T_low=18.0,
        T_high=22.0,
        T_ambient=10.0,
        cooling_rate=0.1,
        heating_power=2.0,
    )

    # Initial set: start in the band
    initial_set = HyperRectangle(
        lower=np.array([19.0]),
        upper=np.array([21.0])
    )

    # Unsafe set: overheating (T > 25)
    unsafe_set = HyperRectangle(
        lower=np.array([25.0]),
        upper=np.array([100.0])
    )

    T = 10.0

    verifier = SwitchingVerifier(n_samples=50, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(result.summary())
    print(f"\nExpected: SAFE (heater turns off before overheating)")
    print(f"Got:      {result.status.name}")

    return result


def demonstrate_hysteresis():
    """Demonstrate hysteresis behavior in thermostat."""
    print("\n" + "=" * 60)
    print("Demonstrating Hysteresis Behavior")
    print("=" * 60)

    dynamics = SwitchingDynamics.thermostat(
        T_low=18.0,
        T_high=22.0,
        T_ambient=10.0,
        cooling_rate=0.1,
        heating_power=2.0,
    )

    solver = FilippovSolver(dynamics)

    # Start cold (heater will turn on)
    T0_cold = np.array([17.0])
    bundle_cold = solver.solve(T0_cold, (0, 20.0))
    traj_cold = bundle_cold.primary

    # Start hot (heater will turn off)
    T0_hot = np.array([23.0])
    bundle_hot = solver.solve(T0_hot, (0, 20.0))
    traj_hot = bundle_hot.primary

    print(f"Starting cold (T={T0_cold[0]}):")
    print(f"  Final temperature: {traj_cold.final_state[0]:.2f}")
    print(f"  Min temperature: {traj_cold.states[:, 0].min():.2f}")
    print(f"  Max temperature: {traj_cold.states[:, 0].max():.2f}")

    print(f"\nStarting hot (T={T0_hot[0]}):")
    print(f"  Final temperature: {traj_hot.final_state[0]:.2f}")
    print(f"  Min temperature: {traj_hot.states[:, 0].min():.2f}")
    print(f"  Max temperature: {traj_hot.states[:, 0].max():.2f}")

    return traj_cold, traj_hot


def visualize_thermostat(traj_cold, traj_hot, T_low, T_high):
    """Plot thermostat trajectories with hysteresis band."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot trajectories
    ax.plot(traj_cold.times, traj_cold.states[:, 0], 'b-', linewidth=2,
            label=f'Starting cold (T={traj_cold.states[0, 0]:.0f})')
    ax.plot(traj_hot.times, traj_hot.states[:, 0], 'r-', linewidth=2,
            label=f'Starting hot (T={traj_hot.states[0, 0]:.0f})')

    # Plot hysteresis band
    ax.axhline(y=T_low, color='orange', linestyle='--', linewidth=1.5,
               label=f'T_low = {T_low}')
    ax.axhline(y=T_high, color='purple', linestyle='--', linewidth=1.5,
               label=f'T_high = {T_high}')
    ax.axhspan(T_low, T_high, alpha=0.2, color='green', label='Hysteresis band')

    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    ax.set_title('Thermostat Hysteresis Behavior\n(heater ON below T_low, OFF above T_high)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_verification(dynamics, initial_set, unsafe_set, T, result, title):
    """Visualize verification result with sample trajectories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    solver = FilippovSolver(dynamics)

    # Sample and simulate
    samples = sample_set(initial_set, 20, SamplingStrategy.LATIN_HYPERCUBE, seed=42)

    for T0 in samples:
        bundle = solver.solve(T0, (0, T))
        traj = bundle.primary
        ax1.plot(traj.times, traj.states[:, 0], 'b-', alpha=0.3)

    # Plot initial set bounds
    ax1.axhline(y=initial_set.lower[0], color='green', linestyle='-', linewidth=2,
                label='Initial set')
    ax1.axhline(y=initial_set.upper[0], color='green', linestyle='-', linewidth=2)

    # Plot unsafe set
    ax1.axhspan(unsafe_set.lower[0], unsafe_set.upper[0], alpha=0.3, color='red',
                label='Unsafe region')

    # Plot hysteresis band (assuming standard thermostat parameters)
    ax1.axhline(y=18.0, color='orange', linestyle='--', alpha=0.7)
    ax1.axhline(y=22.0, color='purple', linestyle='--', alpha=0.7)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature')
    ax1.set_title(f'{title}\nResult: {result.status.name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot objective function samples
    # Sample F_T over initial set
    n_vis = 100
    T0_vis = np.linspace(initial_set.lower[0], initial_set.upper[0], n_vis)
    F_T_values = []

    for temp in T0_vis:
        bundle = solver.solve(np.array([temp]), (0, T))
        min_dist = bundle.min_distance_to_set(unsafe_set.distance)
        F_T_values.append(min_dist)

    ax2.plot(T0_vis, F_T_values, 'b-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', label='Unsafe threshold')
    ax2.axhline(y=result.error_bound, color='orange', linestyle='--',
                label=f'Error bound = {result.error_bound:.3f}')
    ax2.fill_between(T0_vis, 0, result.error_bound, alpha=0.2, color='orange',
                     label='Uncertainty region')

    ax2.set_xlabel('Initial Temperature')
    ax2.set_ylabel('F_T (min distance to unsafe)')
    ax2.set_title('Objective Function over Initial Set')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Thermostat verification example")
    parser.add_argument('--save', action='store_true',
                        help='Save plots to files')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output directory for saved plots (default: current dir)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Run demonstrations
    traj_cold, traj_hot = demonstrate_hysteresis()

    # Run verification examples
    result_cold = example_safe_temperature_band()
    result_hot = example_verify_overheat()

    print("\n" + "=" * 60)
    print("All thermostat examples completed!")
    print("=" * 60)

    # Save figures if requested
    if args.save:
        try:
            # Hysteresis demonstration
            fig1 = visualize_thermostat(traj_cold, traj_hot, T_low=18.0, T_high=22.0)

            # Verification visualization
            dynamics = SwitchingDynamics.thermostat(
                T_low=18.0, T_high=22.0, T_ambient=10.0, cooling_rate=0.1, heating_power=2.0
            )
            initial_set = HyperRectangle(np.array([19.0]), np.array([21.0]))
            unsafe_cold = HyperRectangle(np.array([0.0]), np.array([15.0]))
            fig2 = visualize_verification(
                dynamics, initial_set, unsafe_cold, 10.0,
                result_cold, "Thermostat - Avoid Freezing"
            )

            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            fig1.savefig(outdir / 'thermostat_hysteresis.png', dpi=150, bbox_inches='tight')
            fig2.savefig(outdir / 'thermostat_verification.png', dpi=150, bbox_inches='tight')
            print(f"\nFigures saved to {outdir.absolute()}")

        except Exception as e:
            print(f"Visualization failed: {e}")
