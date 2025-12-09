"""Vehicle Braking Safety Verification Example.

Demonstrates spline-verify on emergency braking collision avoidance.

Dynamics:
    dx/dt = v                              (position derivative)
    dv/dt = a                              (velocity derivative)
    da/dt = -k_response * (a - a_target)   (actuator dynamics)

    where a_target = -k_brake (full braking)

State variables:
    x: distance to obstacle (m)
    v: velocity (m/s)
    a: acceleration (m/s^2)

Safety property:
    x > x_unsafe (don't collide with obstacle)

Two scenarios:
    1. SAFE: Starting far enough to stop in time
    2. UNSAFE: Starting too close/fast to avoid collision

Usage:
    python examples/vehicle_braking.py --save --outdir ./examples/figs/case_study
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spline_verify.dynamics import ODEDynamics
from spline_verify.geometry import HyperRectangle
from spline_verify.verification import SafetyVerifier
from spline_verify.verification.objective import ObjectiveSampler


# ============================================================
# Dynamics Definition
# ============================================================

def create_braking_dynamics(
    k_brake: float = 9.0,
    k_response: float = 10.0,
) -> ODEDynamics:
    """Create vehicle braking dynamics.

    Note: x is distance to obstacle, v is speed (positive).
    Vehicle approaches obstacle, so dx/dt = -v.
    Braking causes deceleration (negative a), reducing v.

    Args:
        k_brake: Maximum braking deceleration (m/s^2), ~0.9g
        k_response: Brake actuator response rate (1/s)

    Returns:
        ODEDynamics for the 3D braking system.
    """
    def dynamics(t: float, state: np.ndarray) -> np.ndarray:
        x, v, a = state
        # Target acceleration: full braking (negative to reduce speed)
        a_target = -k_brake
        # Derivatives
        dx = -v  # Position decreases as vehicle approaches obstacle
        dv = a   # Velocity changes with acceleration
        da = -k_response * (a - a_target)  # Actuator dynamics
        return np.array([dx, dv, da])

    return ODEDynamics(f=dynamics, _n_dims=3)


def create_coasting_dynamics() -> ODEDynamics:
    """Create dynamics with no braking (coasting/failure).

    Note: x is distance to obstacle, v is speed (positive).
    Vehicle approaches obstacle, so dx/dt = -v.
    """
    def dynamics(t: float, state: np.ndarray) -> np.ndarray:
        x, v, a = state
        # Vehicle moving toward obstacle: position decreases
        dx = -v  # Negative because approaching
        dv = a
        da = -5.0 * a  # Acceleration decays (no brake input)
        return np.array([dx, dv, da])

    return ODEDynamics(f=dynamics, _n_dims=3)


# ============================================================
# Verification Examples
# ============================================================

def example_safe(save: bool = False, outdir: Path = None):
    """Safe scenario: Vehicle can stop before collision.

    Initial conditions: Far from obstacle, moderate speed
    - x in [80, 90] m (distance to obstacle)
    - v in [25, 28] m/s (~60 mph)
    - a in [-0.5, 0.5] m/s^2 (near zero)

    Unsafe: x < 5 m (collision zone)
    """
    print("\n" + "=" * 60)
    print("Vehicle Braking - SAFE Case")
    print("=" * 60)

    # Create dynamics with full braking
    dynamics = create_braking_dynamics(k_brake=9.0, k_response=10.0)

    # Initial set: far enough to stop
    initial_set = HyperRectangle(
        lower=np.array([80.0, 25.0, -0.5]),
        upper=np.array([90.0, 28.0, 0.5])
    )

    # Unsafe set: collision zone (x < 5)
    # Use very negative lower bound for x, and full range for v, a
    unsafe_set = HyperRectangle(
        lower=np.array([-1000.0, -100.0, -100.0]),
        upper=np.array([5.0, 100.0, 100.0])
    )

    # Time horizon: 6 seconds (enough to come to stop from ~28 m/s)
    T = 6.0

    # Run verification
    print(f"Initial set: x in [80, 90] m, v in [25, 28] m/s")
    print(f"Unsafe set: x < 5 m")
    print(f"Time horizon: {T} s")
    print(f"Expected result: SAFE")
    print("-" * 40)

    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(f"Verification result: {result.status.name}")
    print(f"Min objective: {result.min_objective:.4f}")
    print(f"Error bound: {result.error_bound:.4f}")
    print(f"Safety margin: {result.safety_margin:.4f}")

    if save and outdir:
        fig = plot_braking_trajectories(
            dynamics, initial_set, unsafe_set, T,
            title="Vehicle Braking - Safe Case (Full Braking)"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / 'vehicle_braking_safe.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outdir / 'vehicle_braking_safe.png'}")

    return result


def example_unsafe(save: bool = False, outdir: Path = None):
    """Unsafe scenario: Vehicle cannot avoid collision.

    Initial conditions: Close to obstacle, high speed, no braking
    - x in [25, 30] m (close to obstacle)
    - v in [28, 32] m/s (~70 mph)
    - a in [-0.5, 0.5] m/s^2

    Using coasting dynamics (brake failure)
    """
    print("\n" + "=" * 60)
    print("Vehicle Braking - UNSAFE Case (Brake Failure)")
    print("=" * 60)

    # Create dynamics with no braking (failure)
    dynamics = create_coasting_dynamics()

    # Initial set: already very close to unsafe boundary
    initial_set = HyperRectangle(
        lower=np.array([6.0, 30.0, -0.5]),
        upper=np.array([8.0, 35.0, 0.5])
    )

    # Unsafe set: collision zone (position only matters, but need full dims)
    # When x < 5, vehicle has collided
    unsafe_set = HyperRectangle(
        lower=np.array([-1000.0, -100.0, -100.0]),
        upper=np.array([5.0, 100.0, 100.0])
    )

    # Time horizon: even 0.1s is enough to enter unsafe (30m/s * 0.1s = 3m)
    T = 0.2

    print(f"Initial set: x in [6, 8] m, v in [30, 35] m/s")
    print(f"Unsafe set: x < 5 m")
    print(f"Time horizon: {T} s")
    print(f"Dynamics: No braking (brake failure)")
    print(f"Expected result: UNSAFE")
    print("-" * 40)

    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(f"Verification result: {result.status.name}")
    print(f"Min objective: {result.min_objective:.4f}")
    print(f"Error bound: {result.error_bound:.4f}")
    print(f"Safety margin: {result.safety_margin:.4f}")

    if save and outdir:
        fig = plot_braking_trajectories(
            dynamics, initial_set, unsafe_set, T,
            title="Vehicle Braking - Unsafe Case (Brake Failure)"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / 'vehicle_braking_unsafe.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outdir / 'vehicle_braking_unsafe.png'}")

    return result


# ============================================================
# Visualization
# ============================================================

def plot_braking_trajectories(
    dynamics: ODEDynamics,
    initial_set: HyperRectangle,
    unsafe_set: HyperRectangle,
    T: float,
    title: str = "Vehicle Braking Trajectories",
    n_trajectories: int = 20,
) -> plt.Figure:
    """Plot sample trajectories showing position, velocity, and acceleration."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sample initial conditions
    rng = np.random.default_rng(42)
    x0_samples = rng.uniform(initial_set.lower, initial_set.upper, size=(n_trajectories, 3))

    # Colors by initial position
    colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))

    for i, x0 in enumerate(x0_samples):
        bundle = dynamics.simulate(x0, (0, T))
        traj = bundle.primary
        times = traj.times
        states = traj.states

        # Position vs time
        axes[0].plot(times, states[:, 0], color=colors[i], alpha=0.7, linewidth=1.5)
        # Velocity vs time
        axes[1].plot(times, states[:, 1], color=colors[i], alpha=0.7, linewidth=1.5)
        # Acceleration vs time
        axes[2].plot(times, states[:, 2], color=colors[i], alpha=0.7, linewidth=1.5)

    # Mark unsafe region on position plot
    axes[0].axhline(y=unsafe_set.upper[0], color='red', linestyle='--', linewidth=2,
                    label=f'Unsafe boundary (x = {unsafe_set.upper[0]} m)')
    axes[0].axhspan(-10, unsafe_set.upper[0], alpha=0.2, color='red', label='Collision zone')

    # Labels and formatting
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Position x (m)', fontsize=12)
    axes[0].set_title('Distance to Obstacle', fontsize=14)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=-5)

    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Velocity v (m/s)', fontsize=12)
    axes[1].set_title('Vehicle Speed', fontsize=14)
    axes[1].axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Stopped')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('Acceleration a (m/s²)', fontsize=12)
    axes[2].set_title('Braking Deceleration', fontsize=14)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_objective_landscape(
    dynamics: ODEDynamics,
    initial_set: HyperRectangle,
    unsafe_set: HyperRectangle,
    T: float,
    save: bool = False,
    outdir: Path = None,
) -> plt.Figure:
    """Plot F_T objective function over (x0, v0) with a fixed at 0."""
    print("\nGenerating objective landscape (x0, v0)...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample objective function on a grid (fixing a=0)
    n_grid = 30
    x_range = np.linspace(initial_set.lower[0], initial_set.upper[0], n_grid)
    v_range = np.linspace(initial_set.lower[1], initial_set.upper[1], n_grid)
    X, V = np.meshgrid(x_range, v_range)

    F_T = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):
            x0 = np.array([X[i, j], V[i, j], 0.0])  # a = 0
            bundle = dynamics.simulate(x0, (0, T))
            traj = bundle.primary
            # Compute minimum distance to unsafe set
            distances = []
            for state in traj.states:
                dist = unsafe_set.distance(state)
                distances.append(dist)
            F_T[i, j] = min(distances)

    # Contour plot
    levels = np.linspace(F_T.min(), F_T.max(), 25)
    contour = ax.contourf(X, V, F_T, levels=levels, cmap='RdYlGn')
    ax.contour(X, V, F_T, levels=levels, colors='white', linewidths=0.3, alpha=0.5)

    # Zero level contour (boundary between safe/unsafe)
    if F_T.min() < 0 < F_T.max():
        ax.contour(X, V, F_T, levels=[0], colors='black', linewidths=3, linestyles='--')

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, label='$F_T(x_0)$ = min distance to collision')
    cbar.ax.tick_params(labelsize=10)

    # Labels
    ax.set_xlabel('Initial Position $x_0$ (m)', fontsize=14)
    ax.set_ylabel('Initial Velocity $v_0$ (m/s)', fontsize=14)
    ax.set_title('Objective Function Landscape\n(Full Braking, $a_0 = 0$)', fontsize=16)

    # Mark safe/unsafe regions
    ax.text(0.05, 0.95, '$F_T > 0$: Safe\n$F_T < 0$: Collision',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save and outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / 'vehicle_braking_objective.png', dpi=200, bbox_inches='tight')
        print(f"Saved: {outdir / 'vehicle_braking_objective.png'}")

    return fig


# ============================================================
# Main
# ============================================================

def main(save: bool = False, outdir: str = './examples/figs/case_study'):
    """Run vehicle braking verification examples."""
    outdir_path = Path(outdir)

    print("\n" + "=" * 60)
    print("VEHICLE BRAKING SAFETY VERIFICATION")
    print("=" * 60)
    print("\nThis example demonstrates safety verification for")
    print("emergency braking collision avoidance scenarios.")
    print("\nPhysics:")
    print("  - 3D state: position, velocity, acceleration")
    print("  - First-order brake actuator dynamics")
    print("  - Max deceleration: ~0.9g (9 m/s²)")

    # Run safe example
    result_safe = example_safe(save=save, outdir=outdir_path)

    # Run unsafe example
    result_unsafe = example_unsafe(save=save, outdir=outdir_path)

    # Generate objective landscape for safe case
    if save:
        dynamics = create_braking_dynamics(k_brake=9.0, k_response=10.0)
        initial_set = HyperRectangle(
            lower=np.array([40.0, 20.0, -0.5]),
            upper=np.array([100.0, 35.0, 0.5])
        )
        unsafe_set = HyperRectangle(
            lower=np.array([-1000.0, -100.0, -100.0]),
            upper=np.array([5.0, 100.0, 100.0])
        )
        T = 6.0
        fig = plot_objective_landscape(dynamics, initial_set, unsafe_set, T,
                                       save=save, outdir=outdir_path)
        plt.close(fig)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Safe case:   {result_safe.status.name} (expected: SAFE)")
    print(f"Unsafe case: {result_unsafe.status.name} (expected: UNSAFE)")

    if save:
        print(f"\nFigures saved to: {outdir_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vehicle braking safety verification')
    parser.add_argument('--save', action='store_true', help='Save figures')
    parser.add_argument('--outdir', type=str, default='./examples/figs/case_study',
                        help='Output directory for figures')
    args = parser.parse_args()

    main(save=args.save, outdir=args.outdir)
