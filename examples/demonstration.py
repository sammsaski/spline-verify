"""Demonstration of spline-verify method for research presentations.

This script generates 5 high-impact figures explaining the safety verification
pipeline using spline approximation. Each figure is designed to be standalone
and suitable for inclusion in research presentations or papers.

Usage:
    python examples/demonstration.py --save --outdir ./presentation_figs
    python examples/demonstration.py --save  # saves to ./examples/figs/demo/

Figure List:
    1. demo_01_problem_setup.png       - The safety verification problem
    2. demo_02_trajectory_to_distance.png - From trajectory to F_T
    3. demo_03_why_approximate.png     - Why we need spline approximation (with 3D surface)
    4. demo_04_switching_example.png   - Switching systems extension
    5. demo_05_pipeline_overview.png   - Complete pipeline diagram
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Patch, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

from spline_verify.dynamics import ODEDynamics
from spline_verify.dynamics.switching import SwitchingDynamics
from spline_verify.geometry import HyperRectangle, Ball
from spline_verify.geometry.sampling import sample_set, SamplingStrategy
from spline_verify.splines.multivariate import ScatteredDataSpline
from spline_verify.verification import SafetyVerifier, VerificationStatus
from spline_verify.verification.objective import ObjectiveSampler
from spline_verify.utils.visualization import plot_set

# ============================================================================
# Presentation Style Configuration
# ============================================================================

COLORS = {
    'initial': '#2ecc71',      # Green
    'unsafe': '#e74c3c',       # Red
    'trajectory': '#3498db',   # Blue
    'spline': '#9b59b6',       # Purple
    'safe': '#00AA00',         # Dark green (for binary)
    'unsafe_dark': '#CC0000',  # Dark red (for binary)
    'unknown': '#f39c12',      # Orange
    'highlight': '#f1c40f',    # Yellow
    'grid': '#bdc3c7',         # Light gray
}

FONTSIZE = {
    'title': 16,
    'label': 14,
    'tick': 12,
    'legend': 12,
    'annotation': 11,
}


def setup_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to axes."""
    if title:
        ax.set_title(title, fontsize=FONTSIZE['title'], fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONTSIZE['label'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE['label'])
    ax.tick_params(labelsize=FONTSIZE['tick'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'])


# ============================================================================
# Demo System Setup - Used consistently throughout all figures
# ============================================================================

def create_demo_system() -> Tuple[ODEDynamics, HyperRectangle, Ball, float]:
    """Create a simple 2D linear system for demonstration.

    Uses a stable spiral system: eigenvalues -0.5 +/- i
    This gives clear, intuitive trajectory behavior.
    """
    A = np.array([
        [-0.5, 1.0],
        [-1.0, -0.5]
    ])
    dynamics = ODEDynamics.from_matrix(A)

    initial_set = HyperRectangle(
        lower=np.array([1.0, 0.5]),
        upper=np.array([2.0, 1.5])
    )

    unsafe_set = Ball(
        center=np.array([0.0, 0.0]),
        radius=0.3
    )

    T = 6.0

    return dynamics, initial_set, unsafe_set, T


# ============================================================================
# Figure 1: Safety Verification Problem (KEEP - enhanced from demo_01)
# ============================================================================

def figure_01_problem_setup(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
    """Generate the problem setup figure showing the safety verification problem."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample a few trajectories to show the challenge
    samples = sample_set(initial_set, 8, SamplingStrategy.LATIN_HYPERCUBE, seed=42)

    for i, x0 in enumerate(samples):
        bundle = dynamics.simulate(x0, (0, T))
        traj = bundle.primary

        # Color gradient along trajectory
        points = traj.states.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, T)
        lc = LineCollection(segments, cmap='Blues', norm=norm, alpha=0.7, linewidth=2)
        lc.set_array(traj.times[:-1])
        ax.add_collection(lc)

        # Mark start point
        ax.plot(x0[0], x0[1], 'o', color=COLORS['initial'], markersize=8,
                markeredgecolor='white', markeredgewidth=1.5)

    # Plot sets
    plot_set(initial_set, ax, color=COLORS['initial'], alpha=0.3, label='Initial Set $X_0$')
    plot_set(unsafe_set, ax, color=COLORS['unsafe'], alpha=0.4, label='Unsafe Set $X_u$')

    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, T))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time $t$', shrink=0.8)
    cbar.ax.tick_params(labelsize=FONTSIZE['tick'])

    # Annotations
    ax.annotate('Trajectories spiral\ntoward origin',
                xy=(0.5, 0.3), xytext=(1.5, -0.8),
                fontsize=FONTSIZE['annotation'],
                arrowprops=dict(arrowstyle='->', color='gray'),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.annotate('Question: Do any trajectories\nenter the unsafe set?',
                xy=(0.0, 0.0), xytext=(-1.5, 1.0),
                fontsize=FONTSIZE['annotation'] + 1,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['unsafe'], lw=2),
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    setup_axes(ax,
               title='Safety Verification Problem',
               xlabel='$x_1$', ylabel='$x_2$')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=FONTSIZE['legend'])
    ax.set_xlim(-2, 2.5)
    ax.set_ylim(-1.5, 2)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 2: From Trajectory to Distance Function (KEEP - enhanced from demo_04)
# ============================================================================

def figure_02_trajectory_to_distance(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
    """Show how F_T is computed for a single trajectory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pick a representative initial condition
    x0 = np.array([1.5, 1.0])

    bundle = dynamics.simulate(x0, (0, T))
    traj = bundle.primary

    # Compute distance to unsafe set at each point
    distances = np.array([unsafe_set.distance(state) for state in traj.states])
    min_idx = np.argmin(distances)
    F_T = distances[min_idx]

    # Left plot: trajectory in state space with distance arrows
    ax1.plot(traj.states[:, 0], traj.states[:, 1],
             color=COLORS['trajectory'], linewidth=2.5, label='Trajectory')
    ax1.plot(x0[0], x0[1], 'o', color=COLORS['initial'], markersize=12,
             markeredgecolor='white', markeredgewidth=2, label=f'$x_0$ = ({x0[0]}, {x0[1]})')

    # Plot unsafe set
    plot_set(unsafe_set, ax1, color=COLORS['unsafe'], alpha=0.4, label='Unsafe Set')

    # Show distance at key points
    key_indices = [0, len(traj)//4, len(traj)//2, min_idx, -1]
    for idx in key_indices:
        state = traj.states[idx]
        dist = distances[idx]

        direction = -state / (np.linalg.norm(state) + 1e-10)
        nearest = unsafe_set.center + unsafe_set.radius * (-direction)

        if idx == min_idx:
            color = COLORS['highlight']
            lw = 2.5
            ax1.plot(state[0], state[1], '*', color=COLORS['highlight'],
                     markersize=20, markeredgecolor='black', markeredgewidth=1,
                     label=f'Min distance = {F_T:.3f}', zorder=10)
        else:
            color = 'gray'
            lw = 1
            ax1.plot(state[0], state[1], 's', color='gray', markersize=6, alpha=0.7)

        ax1.plot([state[0], nearest[0]], [state[1], nearest[1]],
                 '--', color=color, linewidth=lw, alpha=0.8)

    setup_axes(ax1, title='Trajectory with Distance to Unsafe Set',
               xlabel='$x_1$', ylabel='$x_2$')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=FONTSIZE['legend'])

    # Right plot: distance vs time
    ax2.plot(traj.times, distances, color=COLORS['trajectory'], linewidth=2.5)
    ax2.axhline(y=0, color=COLORS['unsafe'], linestyle='--', linewidth=2,
                label='Unsafe boundary')
    ax2.fill_between(traj.times, 0, distances, alpha=0.2, color=COLORS['trajectory'])

    # Mark minimum
    ax2.plot(traj.times[min_idx], F_T, '*', color=COLORS['highlight'],
             markersize=20, markeredgecolor='black', markeredgewidth=1,
             label=f'$F_T(x_0) = {F_T:.3f}$', zorder=10)
    ax2.axhline(y=F_T, color=COLORS['highlight'], linestyle=':', linewidth=1.5, alpha=0.7)

    # Annotation
    ax2.annotate(f'$F_T(x_0) = \\min_{{t \\in [0,T]}} d(x(t), X_u)$\n$= {F_T:.3f}$',
                 xy=(traj.times[min_idx], F_T),
                 xytext=(traj.times[min_idx] + 1, F_T + 0.3),
                 fontsize=FONTSIZE['annotation'] + 1,
                 arrowprops=dict(arrowstyle='->', color='black'),
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    setup_axes(ax2, title='Distance to Unsafe Set Over Time',
               xlabel='Time $t$', ylabel='Distance $d(x(t), X_u)$')
    ax2.legend(loc='upper right', fontsize=FONTSIZE['legend'])
    ax2.set_xlim(0, T)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 3: Why Approximate? The Computational Challenge (NEW)
# ============================================================================

def figure_03_why_approximate() -> plt.Figure:
    """Explain why we need spline approximation - the computational challenge.

    Uses a system where some initial conditions are safe and some are unsafe
    to show the value of the distance function visualization.
    """
    # Use a system with mixed safe/unsafe outcomes
    # Van der Pol-like dynamics that spiral inward
    def vdp_dynamics(t, x):
        mu = 0.5
        return np.array([x[1], mu * (1 - x[0]**2) * x[1] - x[0]])

    dynamics = ODEDynamics(vdp_dynamics, _n_dims=2)
    initial_set = HyperRectangle(lower=np.array([-0.5, -0.5]), upper=np.array([0.5, 0.5]))
    unsafe_set = Ball(center=np.array([1.5, 0.0]), radius=0.4)
    T = 5.0

    # Create figure with 2 panels (no middle gap)
    fig = plt.figure(figsize=(14, 6))

    # Left panel: The challenge - many sample points
    ax1 = fig.add_subplot(1, 2, 1)

    # Show a dense grid of points in the initial set
    n_grid = 12
    x1 = np.linspace(initial_set.lower[0], initial_set.upper[0], n_grid)
    x2 = np.linspace(initial_set.lower[1], initial_set.upper[1], n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])

    # Plot initial set
    plot_set(initial_set, ax1, color=COLORS['initial'], alpha=0.2)

    # Plot many points
    ax1.scatter(grid_points[:, 0], grid_points[:, 1],
                c=COLORS['trajectory'], s=40, alpha=0.7, edgecolors='white')

    # Add annotation
    ax1.text(0.0, -0.75, 'Infinitely many initial conditions\nto check!',
             fontsize=FONTSIZE['annotation'] + 1, style='italic',
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax1.set_title('The Challenge',
                  fontsize=FONTSIZE['title'] + 2, fontweight='bold')
    ax1.set_xlabel('$x_1$', fontsize=FONTSIZE['label'])
    ax1.set_ylabel('$x_2$', fontsize=FONTSIZE['label'])
    ax1.set_xlim(-0.8, 0.8)
    ax1.set_ylim(-1.0, 0.7)
    ax1.set_aspect('equal')
    ax1.tick_params(labelsize=FONTSIZE['tick'])
    ax1.grid(True, alpha=0.3)

    # Right panel: 3D distance function surface showing the solution
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Sample and fit spline
    sampler = ObjectiveSampler(dynamics, initial_set, unsafe_set, T)
    points, values = sampler.sample(200, seed=42)

    spline = ScatteredDataSpline(kernel='thin_plate_spline', smoothing=0.01)
    spline.fit(points, values)

    # Create smooth surface
    n_grid = 40
    x1 = np.linspace(initial_set.lower[0], initial_set.upper[0], n_grid)
    x2 = np.linspace(initial_set.lower[1], initial_set.upper[1], n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    grid_pts = np.column_stack([X1.ravel(), X2.ravel()])
    Z = np.array([spline.evaluate(p) for p in grid_pts]).reshape(X1.shape)

    # Binary colormap for safe/unsafe
    cmap_binary = ListedColormap([COLORS['unsafe_dark'], COLORS['safe']])
    colors = (Z > 0).astype(float)

    # Plot 3D surface
    surf = ax2.plot_surface(X1, X2, Z, facecolors=cmap_binary(colors),
                            alpha=0.9, linewidth=0, antialiased=True, shade=True)

    # Mark minimum
    min_idx = np.argmin(Z)
    min_i, min_j = np.unravel_index(min_idx, Z.shape)
    ax2.scatter([X1[min_i, min_j]], [X2[min_i, min_j]], [Z[min_i, min_j]],
                c='black', s=150, marker='*', edgecolors='white', linewidths=1, zorder=10)

    ax2.set_xlabel('$x_{0,1}$', fontsize=FONTSIZE['label'])
    ax2.set_ylabel('$x_{0,2}$', fontsize=FONTSIZE['label'])
    ax2.set_zlabel('$\\tilde{F}_T(x_0)$', fontsize=FONTSIZE['label'])
    ax2.set_title('Our Solution: Approximate $F_T$\nand Find Its Minimum',
                  fontsize=FONTSIZE['title'] + 2, fontweight='bold')

    # Add legend
    legend_elements = [
        Patch(facecolor=COLORS['safe'], label='Safe ($F > 0$)'),
        Patch(facecolor=COLORS['unsafe_dark'], label='Unsafe ($F \\leq 0$)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=FONTSIZE['legend'])

    ax2.view_init(elev=25, azim=-60)

    # Add arrow annotation between panels
    fig.text(0.5, 0.5, '→', fontsize=40, ha='center', va='center',
             fontweight='bold', color=COLORS['trajectory'])
    fig.text(0.5, 0.4, 'Sample + Fit Spline\n+ Minimize', fontsize=FONTSIZE['annotation'] + 1,
             ha='center', va='center', style='italic')

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 4: Switching Systems Example (REWORK)
# ============================================================================

def figure_04_switching_example() -> plt.Figure:
    """Show a clean, intuitive switching system example."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Relay feedback phase portrait
    ax1.set_title('Relay Feedback System\n$\\dot{x} = -\\text{sign}(x)$',
                  fontsize=FONTSIZE['title'], fontweight='bold')

    # Draw vector field
    x_range = np.linspace(-2, 2, 20)
    for x in x_range:
        if abs(x) > 0.1:
            # Draw arrow showing direction
            dx = -np.sign(x) * 0.15
            ax1.annotate('', xy=(x + dx, 0), xytext=(x, 0),
                        arrowprops=dict(arrowstyle='->', color=COLORS['trajectory'],
                                       lw=1.5, alpha=0.6))

    # Simulate a few trajectories
    dynamics = SwitchingDynamics.relay_feedback()
    T = 3.0

    for x0_val in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
        bundle = dynamics.simulate(np.array([x0_val]), (0, T))
        traj = bundle.primary
        color = 'blue' if x0_val > 0 else 'red'
        ax1.plot(traj.times, traj.states[:, 0], color=color, linewidth=2, alpha=0.7)

    # Mark switching surface
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.fill_between([0, T], [-0.05], [0.05], color='yellow', alpha=0.4)

    ax1.text(T/2, 0.15, 'Sliding Mode\n(x = 0)', ha='center', fontsize=FONTSIZE['annotation'],
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax1.set_xlabel('Time $t$', fontsize=FONTSIZE['label'])
    ax1.set_ylabel('State $x$', fontsize=FONTSIZE['label'])
    ax1.tick_params(labelsize=FONTSIZE['tick'])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-2, 2)

    # Right: Thermostat temperature control
    ax2.set_title('Thermostat with Hysteresis',
                  fontsize=FONTSIZE['title'], fontweight='bold')

    dynamics = SwitchingDynamics.thermostat(T_low=18.0, T_high=22.0, T_ambient=10.0)
    T_sim = 25.0

    # Cold start
    bundle_cold = dynamics.simulate(np.array([15.0]), (0, T_sim))
    traj_cold = bundle_cold.primary
    ax2.plot(traj_cold.times, traj_cold.states[:, 0], 'b-', linewidth=2.5,
             label='Start cold (15°C)')

    # Hot start
    bundle_hot = dynamics.simulate(np.array([25.0]), (0, T_sim))
    traj_hot = bundle_hot.primary
    ax2.plot(traj_hot.times, traj_hot.states[:, 0], 'r-', linewidth=2.5,
             label='Start hot (25°C)')

    # Thresholds
    ax2.axhline(y=18, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=22, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.fill_between([0, T_sim], 18, 22, color='green', alpha=0.15,
                      label='Target range [18°C, 22°C]')

    ax2.text(T_sim * 0.75, 17, '$T_{low}$ = 18°C\n(heater ON)', fontsize=FONTSIZE['annotation'] - 1)
    ax2.text(T_sim * 0.75, 23, '$T_{high}$ = 22°C\n(heater OFF)', fontsize=FONTSIZE['annotation'] - 1)

    ax2.set_xlabel('Time', fontsize=FONTSIZE['label'])
    ax2.set_ylabel('Temperature (°C)', fontsize=FONTSIZE['label'])
    ax2.tick_params(labelsize=FONTSIZE['tick'])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='right', fontsize=FONTSIZE['legend'] - 1)
    ax2.set_xlim(0, T_sim)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 5: Pipeline Overview (REWORK - cleaner design with better labels)
# ============================================================================

def figure_05_pipeline_overview() -> plt.Figure:
    """Create clean pipeline flowchart with improved labels."""
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Pipeline boxes - main flow (5 steps)
    box_height = 1.8
    box_width = 2.8
    y_main = 5.5
    spacing = 3.0

    steps = [
        (2.0, 'Sample\nInitial States\n$x_0 \\in X_0$', COLORS['initial'], '1'),
        (2.0 + spacing, 'Simulate\nTrajectories\n$\\phi(t, x_0)$', COLORS['trajectory'], '2'),
        (2.0 + 2*spacing, 'Compute\nDistance\n$F_T(x_0)$', '#17a2b8', '3'),  # Cyan
        (2.0 + 3*spacing, 'Fit Spline to\n$F_T(x_0)$', COLORS['spline'], '4'),
        (2.0 + 4*spacing, 'Minimize\nSpline\n$\\tilde{F}_T$', COLORS['highlight'], '5'),
    ]

    for x, text, color, num in steps:
        # Main box with rounded corners
        box = FancyBboxPatch((x - box_width/2, y_main - box_height/2),
                              box_width, box_height,
                              boxstyle="round,pad=0.08,rounding_size=0.2",
                              facecolor=color, edgecolor='#333333', linewidth=2.5,
                              alpha=0.85)
        ax.add_patch(box)

        # Text with better formatting
        ax.text(x, y_main, text, ha='center', va='center',
                fontsize=FONTSIZE['annotation'] + 2, fontweight='bold',
                linespacing=1.2)

        # Step number badge
        ax.text(x - box_width/2 + 0.2, y_main + box_height/2 - 0.2, num,
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='#333333', edgecolor='none'))

    # Arrows between steps (thicker, more prominent)
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + box_width/2
        x2 = steps[i+1][0] - box_width/2
        ax.annotate('', xy=(x2 - 0.05, y_main), xytext=(x1 + 0.05, y_main),
                    arrowprops=dict(arrowstyle='-|>', lw=3, color='#333333',
                                   mutation_scale=20))

    # Decision diamond
    decision_x = 2.0 + 4*spacing
    decision_y = 2.2

    # Draw diamond shape
    diamond_size = 1.6
    diamond = plt.Polygon([
        (decision_x, decision_y + diamond_size),
        (decision_x + diamond_size, decision_y),
        (decision_x, decision_y - diamond_size),
        (decision_x - diamond_size, decision_y),
    ], facecolor='white', edgecolor='#333333', linewidth=2.5)
    ax.add_patch(diamond)
    ax.text(decision_x, decision_y, 'Compare\nwith $\\varepsilon$',
            ha='center', va='center', fontsize=FONTSIZE['annotation'])

    # Arrow from minimize to decision
    ax.annotate('', xy=(decision_x, decision_y + diamond_size + 0.1),
                xytext=(decision_x, y_main - box_height/2),
                arrowprops=dict(arrowstyle='-|>', lw=2.5, color='#333333',
                               mutation_scale=18))

    # Result boxes (larger, clearer)
    results_y = 0.8
    result_width = 2.8
    result_height = 1.3
    result_data = [
        (decision_x - 3.0, 'SAFE', 'min $\\tilde{F}_T > \\varepsilon$', COLORS['safe']),
        (decision_x, 'UNKNOWN', '$0 <$ min $\\leq \\varepsilon$', COLORS['unknown']),
        (decision_x + 3.0, 'UNSAFE', 'min $\\tilde{F}_T \\leq 0$', COLORS['unsafe']),
    ]

    for x, label, sublabel, color in result_data:
        box = FancyBboxPatch((x - result_width/2, results_y - result_height/2),
                              result_width, result_height,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor=color, edgecolor='#333333', linewidth=2.5)
        ax.add_patch(box)
        ax.text(x, results_y + 0.15, label, ha='center', va='center',
                fontsize=FONTSIZE['annotation'] + 2, fontweight='bold', color='white')
        ax.text(x, results_y - 0.3, sublabel, ha='center', va='center',
                fontsize=FONTSIZE['annotation'] - 2, color='white')

    # Arrows from decision to results
    for x, _, _, _ in result_data:
        ax.annotate('', xy=(x, results_y + result_height/2 + 0.05),
                    xytext=(decision_x, decision_y - diamond_size - 0.1),
                    arrowprops=dict(arrowstyle='-|>', lw=2, color='#666666',
                                   connectionstyle='arc3,rad=0.15',
                                   mutation_scale=15))

    # Title
    ax.text(9, 8.3, 'Spline-Verify: Safety Verification Pipeline', ha='center',
            fontsize=FONTSIZE['title'] + 6, fontweight='bold')

    # Subtitle
    ax.text(9, 7.6, 'Approximate $F_T(x_0)$ with a spline and find its minimum',
            ha='center', fontsize=FONTSIZE['annotation'] + 1, style='italic', color='#555555')

    plt.tight_layout()
    return fig


# ============================================================================
# Main Generation Function
# ============================================================================

def generate_all_figures(outdir: Path, verbose: bool = True):
    """Generate all demonstration figures."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Create demo system
    dynamics, initial_set, unsafe_set, T = create_demo_system()

    figures = [
        ('demo_01_problem_setup.png',
         lambda: figure_01_problem_setup(dynamics, initial_set, unsafe_set, T)),
        ('demo_02_trajectory_to_distance.png',
         lambda: figure_02_trajectory_to_distance(dynamics, initial_set, unsafe_set, T)),
        ('demo_03_why_approximate.png',
         figure_03_why_approximate),
        ('demo_04_switching_example.png',
         figure_04_switching_example),
        ('demo_05_pipeline_overview.png',
         figure_05_pipeline_overview),
    ]

    for filename, fig_func in figures:
        if verbose:
            print(f"Generating {filename}...", end=' ', flush=True)
        try:
            fig = fig_func()
            fig.savefig(outdir / filename, dpi=200, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            if verbose:
                print("done")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")
            raise

    print(f"\nGenerated {len(figures)} figures in {outdir.absolute()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate demonstration figures for spline-verify presentations"
    )
    parser.add_argument('--save', action='store_true',
                        help='Save figures to files')
    parser.add_argument('--outdir', type=str, default='./examples/figs/demo',
                        help='Output directory for saved figures (default: ./examples/figs/demo)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.save:
        generate_all_figures(Path(args.outdir))
    else:
        print("Use --save to generate figures.")
        print("Example: python examples/demonstration.py --save --outdir ./presentation_figs")
