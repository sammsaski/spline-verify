"""Quadrotor Altitude Control Safety Verification Example.

Demonstrates spline-verify on quadrotor altitude control to prevent
ground collision.

Dynamics (simplified 2D vertical motion):
    dz/dt = vz                           (altitude derivative)
    dvz/dt = (u - 1) * g                 (thrust minus gravity)

    where u is normalized thrust (u=1 hovers, u=0 free fall, u=2 max thrust)

State variables:
    z: altitude (m)
    vz: vertical velocity (m/s)

Safety property:
    z > z_ground (don't hit the ground)

Three scenarios:
    1. SAFE (hover): Controlled hover maintains altitude
    2. UNSAFE (fall): Motor failure leads to ground collision
    3. Phase portrait: Shows safe/unsafe regions in state space

Usage:
    python examples/quadrotor.py --save --outdir ./examples/figs/case_study
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


# ============================================================
# Constants
# ============================================================

G = 9.81  # Gravitational acceleration (m/s^2)


# ============================================================
# Dynamics Definition
# ============================================================

def create_quadrotor_dynamics(
    thrust: float = 1.0,
    drag: float = 0.0,
) -> ODEDynamics:
    """Create quadrotor vertical dynamics.

    Args:
        thrust: Normalized thrust (0=off, 1=hover, 2=max)
        drag: Aerodynamic drag coefficient

    Returns:
        ODEDynamics for the 2D altitude system.
    """
    def dynamics(t: float, state: np.ndarray) -> np.ndarray:
        z, vz = state
        # Vertical dynamics: thrust - gravity - drag
        dz = vz
        dvz = (thrust - 1.0) * G - drag * vz
        return np.array([dz, dvz])

    return ODEDynamics(dynamics, _n_dims=2)


def create_pd_controlled_dynamics(
    z_target: float = 2.0,
    kp: float = 2.0,
    kd: float = 1.5,
) -> ODEDynamics:
    """Create quadrotor with PD altitude controller.

    Args:
        z_target: Target altitude (m)
        kp: Proportional gain
        kd: Derivative gain

    Returns:
        ODEDynamics with closed-loop control.
    """
    def dynamics(t: float, state: np.ndarray) -> np.ndarray:
        z, vz = state
        # PD controller: u = 1 + kp*(z_target - z) - kd*vz
        # Clamp thrust to [0, 2]
        u = 1.0 + kp * (z_target - z) - kd * vz
        u = np.clip(u, 0.0, 2.0)

        dz = vz
        dvz = (u - 1.0) * G
        return np.array([dz, dvz])

    return ODEDynamics(dynamics, _n_dims=2)


# ============================================================
# Verification Examples
# ============================================================

def example_safe_hover(save: bool = False, outdir: Path = None):
    """Safe scenario: PD-controlled hover maintains altitude.

    Initial conditions:
    - z in [1.0, 2.0] m (above ground)
    - vz in [-0.5, 0.5] m/s (near hover)

    Unsafe: z < 0.1 m (ground collision)
    """
    print("\n" + "=" * 60)
    print("Quadrotor - SAFE Case (PD-Controlled Hover)")
    print("=" * 60)

    # Create PD-controlled dynamics
    dynamics = create_pd_controlled_dynamics(z_target=2.0, kp=2.0, kd=1.5)

    # Initial set: near hover altitude
    initial_set = HyperRectangle(
        lower=np.array([1.0, -0.5]),
        upper=np.array([2.0, 0.5])
    )

    # Unsafe set: ground collision (z < 0.1)
    unsafe_set = HyperRectangle(
        lower=np.array([-100.0, -100.0]),
        upper=np.array([0.1, 100.0])
    )

    # Time horizon
    T = 5.0

    print(f"Initial set: z in [1.0, 2.0] m, vz in [-0.5, 0.5] m/s")
    print(f"Unsafe set: z < 0.1 m (ground)")
    print(f"Control: PD controller targeting z = 2.0 m")
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
        fig = plot_quadrotor_trajectories(
            dynamics, initial_set, unsafe_set, T,
            title="Quadrotor Altitude - Safe Case (PD Control)"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / 'quadrotor_hover_safe.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outdir / 'quadrotor_hover_safe.png'}")

    return result


def example_unsafe_fall(save: bool = False, outdir: Path = None):
    """Unsafe scenario: Motor failure causes free fall.

    Initial conditions:
    - z in [3.0, 4.0] m (moderate altitude)
    - vz in [-1.0, 0.0] m/s (already descending)

    Dynamics: Zero thrust (motor failure)
    """
    print("\n" + "=" * 60)
    print("Quadrotor - UNSAFE Case (Motor Failure)")
    print("=" * 60)

    # Create free-fall dynamics (thrust = 0)
    dynamics = create_quadrotor_dynamics(thrust=0.0, drag=0.1)

    # Initial set: at altitude, descending
    initial_set = HyperRectangle(
        lower=np.array([3.0, -1.0]),
        upper=np.array([4.0, 0.0])
    )

    # Unsafe set: ground collision
    unsafe_set = HyperRectangle(
        lower=np.array([-100.0, -100.0]),
        upper=np.array([0.1, 100.0])
    )

    # Time horizon: long enough to hit ground
    T = 2.0

    print(f"Initial set: z in [3.0, 4.0] m, vz in [-1.0, 0.0] m/s")
    print(f"Unsafe set: z < 0.1 m (ground)")
    print(f"Control: None (motor failure, free fall)")
    print(f"Time horizon: {T} s")
    print(f"Expected result: UNSAFE")
    print("-" * 40)

    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(f"Verification result: {result.status.name}")
    print(f"Min objective: {result.min_objective:.4f}")
    print(f"Error bound: {result.error_bound:.4f}")
    print(f"Safety margin: {result.safety_margin:.4f}")

    if save and outdir:
        fig = plot_quadrotor_trajectories(
            dynamics, initial_set, unsafe_set, T,
            title="Quadrotor Altitude - Unsafe Case (Motor Failure)"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / 'quadrotor_fall_unsafe.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outdir / 'quadrotor_fall_unsafe.png'}")

    return result


# ============================================================
# Visualization
# ============================================================

def plot_quadrotor_trajectories(
    dynamics: ODEDynamics,
    initial_set: HyperRectangle,
    unsafe_set: HyperRectangle,
    T: float,
    title: str = "Quadrotor Trajectories",
    n_trajectories: int = 20,
) -> plt.Figure:
    """Plot sample trajectories showing altitude and velocity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sample initial conditions
    rng = np.random.default_rng(42)
    x0_samples = rng.uniform(initial_set.lower, initial_set.upper, size=(n_trajectories, 2))

    # Colors by initial altitude
    colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))

    for i, x0 in enumerate(x0_samples):
        bundle = dynamics.simulate(x0, (0, T))
        traj = bundle.primary
        times = traj.times
        states = traj.states

        # Altitude vs time
        axes[0].plot(times, states[:, 0], color=colors[i], alpha=0.7, linewidth=1.5)
        # Velocity vs time
        axes[1].plot(times, states[:, 1], color=colors[i], alpha=0.7, linewidth=1.5)

    # Mark unsafe region on altitude plot
    ground_level = unsafe_set.upper[0]
    axes[0].axhline(y=ground_level, color='red', linestyle='--', linewidth=2,
                    label=f'Ground (z = {ground_level} m)')
    axes[0].axhspan(-1, ground_level, alpha=0.2, color='red', label='Crash zone')

    # Labels and formatting
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Altitude z (m)', fontsize=12)
    axes[0].set_title('Altitude vs Time', fontsize=14)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=-0.5)

    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Vertical Velocity $v_z$ (m/s)', fontsize=12)
    axes[1].set_title('Vertical Velocity vs Time', fontsize=14)
    axes[1].axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Hover (vz=0)')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_phase_portrait(save: bool = False, outdir: Path = None) -> plt.Figure:
    """Plot phase portrait showing safe/unsafe regions in (z, vz) space."""
    print("\nGenerating phase portrait...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create dynamics for different scenarios
    dynamics_hover = create_pd_controlled_dynamics(z_target=2.0)
    dynamics_fall = create_quadrotor_dynamics(thrust=0.0, drag=0.1)

    # Sample trajectories for phase portrait
    n_trajs = 15

    # Controlled trajectories (blue)
    rng = np.random.default_rng(42)
    for _ in range(n_trajs):
        z0 = rng.uniform(0.5, 4.0)
        vz0 = rng.uniform(-2.0, 2.0)
        x0 = np.array([z0, vz0])
        bundle = dynamics_hover.simulate(x0, (0, 5.0))
        traj = bundle.primary
        ax.plot(traj.states[:, 0], traj.states[:, 1], 'b-', alpha=0.5, linewidth=1)
        ax.plot(x0[0], x0[1], 'bo', markersize=4)

    # Free-fall trajectories (red)
    for _ in range(n_trajs):
        z0 = rng.uniform(1.0, 5.0)
        vz0 = rng.uniform(-1.0, 1.0)
        x0 = np.array([z0, vz0])
        bundle = dynamics_fall.simulate(x0, (0, 2.0))
        traj = bundle.primary
        ax.plot(traj.states[:, 0], traj.states[:, 1], 'r-', alpha=0.5, linewidth=1)
        ax.plot(x0[0], x0[1], 'ro', markersize=4)

    # Mark unsafe region
    ax.axvspan(-1, 0.1, alpha=0.3, color='red', label='Crash zone (z < 0.1)')

    # Mark target altitude
    ax.axvline(x=2.0, color='green', linestyle='--', alpha=0.5, label='Target altitude')

    # Add vector field for controlled dynamics
    z_range = np.linspace(0.2, 5.0, 15)
    vz_range = np.linspace(-3.0, 3.0, 15)
    Z, VZ = np.meshgrid(z_range, vz_range)

    # Compute derivatives
    DZ = VZ.copy()
    DVZ = np.zeros_like(Z)
    for i in range(len(z_range)):
        for j in range(len(vz_range)):
            state = np.array([Z[j, i], VZ[j, i]])
            deriv = dynamics_hover.f(0, state)
            DVZ[j, i] = deriv[1]

    # Normalize for visualization
    magnitude = np.sqrt(DZ**2 + DVZ**2)
    DZ_norm = DZ / (magnitude + 0.1)
    DVZ_norm = DVZ / (magnitude + 0.1)

    ax.quiver(Z, VZ, DZ_norm, DVZ_norm, magnitude, cmap='Greys', alpha=0.4, scale=25)

    # Labels
    ax.set_xlabel('Altitude z (m)', fontsize=14)
    ax.set_ylabel('Vertical Velocity $v_z$ (m/s)', fontsize=14)
    ax.set_title('Quadrotor Phase Portrait\nBlue: PD Control | Red: Free Fall', fontsize=16)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save and outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / 'quadrotor_phase.png', dpi=200, bbox_inches='tight')
        print(f"Saved: {outdir / 'quadrotor_phase.png'}")

    return fig


# ============================================================
# Main
# ============================================================

def main(save: bool = False, outdir: str = './examples/figs/case_study'):
    """Run quadrotor verification examples."""
    outdir_path = Path(outdir)

    print("\n" + "=" * 60)
    print("QUADROTOR ALTITUDE SAFETY VERIFICATION")
    print("=" * 60)
    print("\nThis example demonstrates safety verification for")
    print("quadrotor altitude control to prevent ground collision.")
    print("\nPhysics:")
    print("  - 2D state: altitude z, vertical velocity vz")
    print("  - Normalized thrust: 0 (off), 1 (hover), 2 (max)")
    print(f"  - Gravity: {G} m/s^2")

    # Run safe example
    result_safe = example_safe_hover(save=save, outdir=outdir_path)

    # Run unsafe example
    result_unsafe = example_unsafe_fall(save=save, outdir=outdir_path)

    # Generate phase portrait
    if save:
        fig = plot_phase_portrait(save=save, outdir=outdir_path)
        plt.close(fig)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Safe case (hover):  {result_safe.status.name} (expected: SAFE)")
    print(f"Unsafe case (fall): {result_unsafe.status.name} (expected: UNSAFE)")

    if save:
        print(f"\nFigures saved to: {outdir_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quadrotor altitude safety verification')
    parser.add_argument('--save', action='store_true', help='Save figures')
    parser.add_argument('--outdir', type=str, default='./examples/figs/case_study',
                        help='Output directory for figures')
    args = parser.parse_args()

    main(save=args.save, outdir=args.outdir)
