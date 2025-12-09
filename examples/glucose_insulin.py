"""Glucose-Insulin Regulation Safety Verification Example.

Demonstrates spline-verify on blood glucose regulation to prevent
hypoglycemia and hyperglycemia.

Dynamics (simplified Bergman-inspired model):
    dG/dt = -k_g * G - k_i * I * G + G_input + G_basal
    dI/dt = -k_decay * I + k_secrete * max(0, G - G_thresh)

State variables:
    G: blood glucose concentration (mg/dL)
    I: plasma insulin concentration (mU/L)

Safety property:
    G_low < G < G_high (avoid hypo/hyperglycemia)

Two scenarios:
    1. SAFE (fasting): Normal fasting glucose regulation
    2. UNSAFE (meal spike): Large meal without insulin causes hyperglycemia

Usage:
    python examples/glucose_insulin.py --save --outdir ./examples/figs/case_study
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
# Physiological Parameters
# ============================================================

# Default parameters (simplified Bergman minimal model)
DEFAULT_PARAMS = {
    'k_g': 0.01,        # Glucose clearance rate (1/min)
    'k_i': 0.0001,      # Insulin effectiveness (1/(mU/L)/min)
    'k_decay': 0.1,     # Insulin degradation rate (1/min)
    'k_secrete': 0.005, # Insulin secretion rate (mU/L per mg/dL per min)
    'G_thresh': 80.0,   # Glucose threshold for secretion (mg/dL)
    'G_basal': 1.0,     # Basal hepatic glucose production (mg/dL/min)
}

# Safety thresholds
G_HYPOGLYCEMIA = 70.0   # Dangerous low (mg/dL)
G_HYPERGLYCEMIA = 180.0  # Dangerous high (mg/dL)
G_SEVERE_HYPER = 200.0  # Severe hyperglycemia (mg/dL)


# ============================================================
# Dynamics Definition
# ============================================================

def create_glucose_dynamics(
    G_input: float = 0.0,
    params: dict = None,
) -> ODEDynamics:
    """Create glucose-insulin regulation dynamics.

    Args:
        G_input: Glucose input rate from meal (mg/dL/min)
        params: Model parameters (uses defaults if None)

    Returns:
        ODEDynamics for the 2D glucose-insulin system.
    """
    p = params or DEFAULT_PARAMS

    def dynamics(t: float, state: np.ndarray) -> np.ndarray:
        G, I = state

        # Glucose dynamics:
        # - Clearance: -k_g * G
        # - Insulin-mediated uptake: -k_i * I * G
        # - Meal input: G_input
        # - Basal production: G_basal
        dG = -p['k_g'] * G - p['k_i'] * I * G + G_input + p['G_basal']

        # Insulin dynamics:
        # - Degradation: -k_decay * I
        # - Secretion in response to high glucose: k_secrete * max(0, G - G_thresh)
        secretion = p['k_secrete'] * max(0, G - p['G_thresh'])
        dI = -p['k_decay'] * I + secretion

        return np.array([dG, dI])

    return ODEDynamics(dynamics, _n_dims=2)


def create_meal_dynamics(
    meal_glucose: float = 30.0,
    meal_duration: float = 60.0,
    params: dict = None,
) -> ODEDynamics:
    """Create dynamics with time-varying meal input.

    Args:
        meal_glucose: Total glucose from meal (mg/dL equivalent)
        meal_duration: Duration of glucose absorption (min)
        params: Model parameters

    Returns:
        ODEDynamics with meal absorption.
    """
    p = params or DEFAULT_PARAMS
    # Simple triangular meal absorption profile
    peak_rate = 2 * meal_glucose / meal_duration

    def dynamics(t: float, state: np.ndarray) -> np.ndarray:
        G, I = state

        # Meal absorption (triangular profile)
        if t < meal_duration / 2:
            G_input = peak_rate * (2 * t / meal_duration)
        elif t < meal_duration:
            G_input = peak_rate * (2 - 2 * t / meal_duration)
        else:
            G_input = 0.0

        # Glucose dynamics
        dG = -p['k_g'] * G - p['k_i'] * I * G + G_input + p['G_basal']

        # Insulin dynamics
        secretion = p['k_secrete'] * max(0, G - p['G_thresh'])
        dI = -p['k_decay'] * I + secretion

        return np.array([dG, dI])

    return ODEDynamics(dynamics, _n_dims=2)


# ============================================================
# Verification Examples
# ============================================================

def example_safe_fasting(save: bool = False, outdir: Path = None):
    """Safe scenario: Fasting glucose stays in normal range.

    Initial conditions (normal fasting):
    - G in [85, 100] mg/dL
    - I in [8, 12] mU/L

    Unsafe: G < 70 (hypoglycemia) or G > 180 (hyperglycemia)
    """
    print("\n" + "=" * 60)
    print("Glucose-Insulin - SAFE Case (Fasting)")
    print("=" * 60)

    # Create fasting dynamics (no meal input)
    dynamics = create_glucose_dynamics(G_input=0.0)

    # Initial set: normal fasting values
    initial_set = HyperRectangle(
        lower=np.array([85.0, 8.0]),
        upper=np.array([100.0, 12.0])
    )

    # Unsafe set: hypoglycemia (G < 70)
    # We check for hypoglycemia first as it's more dangerous
    unsafe_set = HyperRectangle(
        lower=np.array([0.0, 0.0]),
        upper=np.array([G_HYPOGLYCEMIA, 100.0])
    )

    # Time horizon: 2 hours of fasting
    T = 120.0

    print(f"Initial set: G in [85, 100] mg/dL, I in [8, 12] mU/L")
    print(f"Unsafe set: G < {G_HYPOGLYCEMIA} mg/dL (hypoglycemia)")
    print(f"Condition: Fasting (no meal)")
    print(f"Time horizon: {T} min ({T/60:.1f} hours)")
    print(f"Expected result: SAFE")
    print("-" * 40)

    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(f"Verification result: {result.status.name}")
    print(f"Min objective: {result.min_objective:.4f}")
    print(f"Error bound: {result.error_bound:.4f}")
    print(f"Safety margin: {result.safety_margin:.4f}")

    if save and outdir:
        fig = plot_glucose_trajectories(
            dynamics, initial_set, T,
            title="Glucose Regulation - Safe Case (Fasting)"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / 'glucose_safe.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outdir / 'glucose_safe.png'}")

    return result


def example_unsafe_meal(save: bool = False, outdir: Path = None):
    """Unsafe scenario: Large meal causes hyperglycemia spike.

    Initial conditions:
    - G in [90, 100] mg/dL
    - I in [5, 8] mU/L (slightly low insulin)

    Large meal without additional insulin injection
    """
    print("\n" + "=" * 60)
    print("Glucose-Insulin - UNSAFE Case (Meal Spike)")
    print("=" * 60)

    # Create dynamics with large meal (severe glucose load)
    dynamics = create_meal_dynamics(meal_glucose=300.0, meal_duration=60.0)

    # Initial set: normal glucose, low insulin
    initial_set = HyperRectangle(
        lower=np.array([90.0, 5.0]),
        upper=np.array([100.0, 8.0])
    )

    # Unsafe set: severe hyperglycemia (G > 200)
    unsafe_set = HyperRectangle(
        lower=np.array([G_SEVERE_HYPER, 0.0]),
        upper=np.array([500.0, 100.0])
    )

    # Time horizon: 90 minutes post-meal
    T = 90.0

    print(f"Initial set: G in [90, 100] mg/dL, I in [5, 8] mU/L")
    print(f"Unsafe set: G > {G_SEVERE_HYPER} mg/dL (severe hyperglycemia)")
    print(f"Condition: Large meal (300 mg/dL glucose load)")
    print(f"Time horizon: {T} min ({T/60:.1f} hours)")
    print(f"Expected result: UNSAFE")
    print("-" * 40)

    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    print(f"Verification result: {result.status.name}")
    print(f"Min objective: {result.min_objective:.4f}")
    print(f"Error bound: {result.error_bound:.4f}")
    print(f"Safety margin: {result.safety_margin:.4f}")

    if save and outdir:
        fig = plot_glucose_trajectories(
            dynamics, initial_set, T,
            title="Glucose Regulation - Unsafe Case (Meal Spike)"
        )
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / 'glucose_unsafe.png', dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outdir / 'glucose_unsafe.png'}")

    return result


# ============================================================
# Visualization
# ============================================================

def plot_glucose_trajectories(
    dynamics: ODEDynamics,
    initial_set: HyperRectangle,
    T: float,
    title: str = "Glucose-Insulin Trajectories",
    n_trajectories: int = 20,
) -> plt.Figure:
    """Plot sample trajectories showing glucose and insulin over time."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sample initial conditions
    rng = np.random.default_rng(42)
    x0_samples = rng.uniform(initial_set.lower, initial_set.upper, size=(n_trajectories, 2))

    # Colors by initial glucose
    norm_G = (x0_samples[:, 0] - initial_set.lower[0]) / (initial_set.upper[0] - initial_set.lower[0])
    colors = plt.cm.viridis(norm_G)

    for i, x0 in enumerate(x0_samples):
        bundle = dynamics.simulate(x0, (0, T))
        traj = bundle.primary
        times = traj.times
        states = traj.states

        # Glucose vs time
        axes[0].plot(times, states[:, 0], color=colors[i], alpha=0.7, linewidth=1.5)
        # Insulin vs time
        axes[1].plot(times, states[:, 1], color=colors[i], alpha=0.7, linewidth=1.5)

    # Mark safety boundaries on glucose plot
    axes[0].axhline(y=G_HYPOGLYCEMIA, color='blue', linestyle='--', linewidth=2,
                    label=f'Hypoglycemia ({G_HYPOGLYCEMIA} mg/dL)')
    axes[0].axhline(y=G_HYPERGLYCEMIA, color='orange', linestyle='--', linewidth=2,
                    label=f'Hyperglycemia ({G_HYPERGLYCEMIA} mg/dL)')
    axes[0].axhline(y=G_SEVERE_HYPER, color='red', linestyle='--', linewidth=2,
                    label=f'Severe hyper ({G_SEVERE_HYPER} mg/dL)')

    # Shade dangerous regions
    axes[0].axhspan(0, G_HYPOGLYCEMIA, alpha=0.15, color='blue', label='_')
    axes[0].axhspan(G_SEVERE_HYPER, 300, alpha=0.15, color='red', label='_')

    # Labels and formatting
    axes[0].set_xlabel('Time (min)', fontsize=12)
    axes[0].set_ylabel('Blood Glucose G (mg/dL)', fontsize=12)
    axes[0].set_title('Glucose Concentration', fontsize=14)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(50, max(250, axes[0].get_ylim()[1]))

    axes[1].set_xlabel('Time (min)', fontsize=12)
    axes[1].set_ylabel('Plasma Insulin I (mU/L)', fontsize=12)
    axes[1].set_title('Insulin Concentration', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_phase_portrait(save: bool = False, outdir: Path = None) -> plt.Figure:
    """Plot G-I phase portrait showing safe zone and trajectories."""
    print("\nGenerating phase portrait...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create fasting and meal dynamics
    dynamics_fasting = create_glucose_dynamics(G_input=0.0)
    dynamics_meal = create_meal_dynamics(meal_glucose=50.0, meal_duration=60.0)

    # Sample trajectories
    n_trajs = 12
    rng = np.random.default_rng(42)

    # Fasting trajectories (blue)
    for _ in range(n_trajs):
        G0 = rng.uniform(70, 150)
        I0 = rng.uniform(5, 20)
        x0 = np.array([G0, I0])
        bundle = dynamics_fasting.simulate(x0, (0, 180.0))
        traj = bundle.primary
        ax.plot(traj.states[:, 0], traj.states[:, 1], 'b-', alpha=0.5, linewidth=1)
        ax.plot(x0[0], x0[1], 'bo', markersize=4)

    # Post-meal trajectories (red)
    for _ in range(n_trajs):
        G0 = rng.uniform(90, 120)
        I0 = rng.uniform(5, 15)
        x0 = np.array([G0, I0])
        bundle = dynamics_meal.simulate(x0, (0, 120.0))
        traj = bundle.primary
        ax.plot(traj.states[:, 0], traj.states[:, 1], 'r-', alpha=0.5, linewidth=1)
        ax.plot(x0[0], x0[1], 'ro', markersize=4)

    # Mark safe zone
    ax.axvspan(G_HYPOGLYCEMIA, G_HYPERGLYCEMIA, alpha=0.2, color='green', label='Safe zone')

    # Mark dangerous zones
    ax.axvspan(0, G_HYPOGLYCEMIA, alpha=0.2, color='blue', label='Hypoglycemia')
    ax.axvspan(G_SEVERE_HYPER, 300, alpha=0.2, color='red', label='Severe hyperglycemia')

    # Mark thresholds
    ax.axvline(x=G_HYPOGLYCEMIA, color='blue', linestyle='--', linewidth=2)
    ax.axvline(x=G_HYPERGLYCEMIA, color='orange', linestyle='--', linewidth=2)
    ax.axvline(x=G_SEVERE_HYPER, color='red', linestyle='--', linewidth=2)

    # Add equilibrium point annotation
    # At equilibrium: dG/dt = 0, dI/dt = 0
    # G_eq ~ G_basal / k_g = 100 (approx)
    ax.plot(100, 10, 'g*', markersize=15, label='Equilibrium (approx)', zorder=5)

    # Labels
    ax.set_xlabel('Blood Glucose G (mg/dL)', fontsize=14)
    ax.set_ylabel('Plasma Insulin I (mU/L)', fontsize=14)
    ax.set_title('Glucose-Insulin Phase Portrait\nBlue: Fasting | Red: Post-meal', fontsize=16)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim(40, 280)
    ax.set_ylim(0, 35)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save and outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / 'glucose_phase.png', dpi=200, bbox_inches='tight')
        print(f"Saved: {outdir / 'glucose_phase.png'}")

    return fig


# ============================================================
# Main
# ============================================================

def main(save: bool = False, outdir: str = './examples/figs/case_study'):
    """Run glucose-insulin verification examples."""
    outdir_path = Path(outdir)

    print("\n" + "=" * 60)
    print("GLUCOSE-INSULIN REGULATION SAFETY VERIFICATION")
    print("=" * 60)
    print("\nThis example demonstrates safety verification for")
    print("blood glucose regulation to prevent dangerous levels.")
    print("\nPhysiology:")
    print("  - 2D state: glucose G (mg/dL), insulin I (mU/L)")
    print("  - Simplified Bergman minimal model")
    print(f"  - Safe range: {G_HYPOGLYCEMIA} < G < {G_HYPERGLYCEMIA} mg/dL")
    print("\nTime scale: Minutes (physiological processes)")

    # Run safe example
    result_safe = example_safe_fasting(save=save, outdir=outdir_path)

    # Run unsafe example
    result_unsafe = example_unsafe_meal(save=save, outdir=outdir_path)

    # Generate phase portrait
    if save:
        fig = plot_phase_portrait(save=save, outdir=outdir_path)
        plt.close(fig)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Safe case (fasting):  {result_safe.status.name} (expected: SAFE)")
    print(f"Unsafe case (meal):   {result_unsafe.status.name} (expected: UNSAFE)")

    if save:
        print(f"\nFigures saved to: {outdir_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glucose-insulin safety verification')
    parser.add_argument('--save', action='store_true', help='Save figures')
    parser.add_argument('--outdir', type=str, default='./examples/figs/case_study',
                        help='Output directory for figures')
    args = parser.parse_args()

    main(save=args.save, outdir=args.outdir)
