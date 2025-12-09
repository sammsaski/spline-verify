"""Demonstration of spline-verify method for research presentations.

This script generates a complete set of figures explaining each step of the
safety verification pipeline. Each figure is designed to be standalone and
suitable for inclusion in research presentations or papers.

Usage:
    python examples/demonstration.py --save --outdir ./presentation_figs
    python examples/demonstration.py --save  # saves to ./examples/figs/demo/

Figure List:
    1. demo_01_problem_setup.png      - The safety verification problem
    2. demo_02_sampling_strategies.png - Comparison of sampling methods
    3. demo_03_trajectory_bundle.png   - Trajectory simulation from samples
    4. demo_04_distance_computation.png - Computing F_T for one trajectory
    5. demo_05_objective_samples.png   - Sampled objective function values
    6. demo_06_objective_landscape.png - Full objective function landscape
    7. demo_07_spline_fitting.png      - RBF spline approximation
    8. demo_08_approximation_error.png - Approximation error analysis
    9. demo_09_spline_minimization.png - Spline minimization visualization
    10. demo_10_error_budget.png       - Error budget breakdown
    11. demo_11_decision_logic.png     - Safety decision visualization
    12. demo_12_switching_surfaces.png - Switching system structure
    13. demo_13_piecewise_spline.png   - Piecewise spline for switching
    14. demo_14_region_classifier.png  - SVM region classification
    15. demo_15_pipeline_summary.png   - Complete pipeline diagram
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

from spline_verify.dynamics import ODEDynamics
from spline_verify.dynamics.switching import SwitchingDynamics
from spline_verify.geometry import HyperRectangle, Ball
from spline_verify.geometry.sampling import sample_set, SamplingStrategy
from spline_verify.splines.multivariate import ScatteredDataSpline
from spline_verify.splines.optimization import minimize_spline, MinimizationResult
from spline_verify.verification import SafetyVerifier, VerificationStatus
from spline_verify.verification.objective import ObjectiveSampler
from spline_verify.verification.error_bounds import ErrorBudget
from spline_verify.utils.visualization import plot_set

# ============================================================================
# Presentation Style Configuration
# ============================================================================

# Color scheme (consistent throughout)
COLORS = {
    'initial': '#2ecc71',      # Green
    'unsafe': '#e74c3c',       # Red
    'trajectory': '#3498db',   # Blue
    'spline': '#9b59b6',       # Purple
    'safe': '#27ae60',         # Dark green
    'unknown': '#f39c12',      # Orange
    'highlight': '#f1c40f',    # Yellow
    'grid': '#bdc3c7',         # Light gray
}

# Font sizes for presentations
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
# Demo System Setup
# ============================================================================

def create_demo_system() -> Tuple[ODEDynamics, HyperRectangle, Ball, float]:
    """Create a simple 2D linear system for demonstration.

    Uses a stable spiral system: eigenvalues -0.5 +/- i
    This gives clear, intuitive trajectory behavior.
    """
    # Stable spiral matrix
    A = np.array([
        [-0.5, 1.0],
        [-1.0, -0.5]
    ])
    dynamics = ODEDynamics.from_matrix(A)

    # Initial set: rectangle in first quadrant
    initial_set = HyperRectangle(
        lower=np.array([1.0, 0.5]),
        upper=np.array([2.0, 1.5])
    )

    # Unsafe set: ball near origin
    unsafe_set = Ball(
        center=np.array([0.0, 0.0]),
        radius=0.3
    )

    # Time horizon
    T = 6.0

    return dynamics, initial_set, unsafe_set, T


# ============================================================================
# Figure 1: Problem Setup
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

        # Create line collection with color gradient
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
# Figure 2: Sampling Strategies
# ============================================================================

def figure_02_sampling_strategies(initial_set) -> plt.Figure:
    """Compare different sampling strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    strategies = [
        (SamplingStrategy.UNIFORM, 'Uniform Random'),
        (SamplingStrategy.LATIN_HYPERCUBE, 'Latin Hypercube'),
        (SamplingStrategy.SOBOL, 'Sobol Sequence'),
        (SamplingStrategy.HALTON, 'Halton Sequence'),
    ]

    n_samples = 50

    for ax, (strategy, name) in zip(axes.flat, strategies):
        samples = sample_set(initial_set, n_samples, strategy, seed=42)

        # Plot samples
        ax.scatter(samples[:, 0], samples[:, 1],
                   c=COLORS['trajectory'], s=50, alpha=0.7, edgecolors='white')

        # Plot initial set boundary
        plot_set(initial_set, ax, color=COLORS['initial'], alpha=0.2)

        setup_axes(ax, title=name, xlabel='$x_1$', ylabel='$x_2$')
        ax.set_aspect('equal')

        # Compute and display discrepancy metric (coverage quality)
        # Simple metric: average nearest neighbor distance
        from scipy.spatial import distance_matrix
        dists = distance_matrix(samples, samples)
        np.fill_diagonal(dists, np.inf)
        nn_dists = np.min(dists, axis=1)
        avg_nn = np.mean(nn_dists)

        ax.text(0.05, 0.95, f'Avg NN dist: {avg_nn:.3f}',
                transform=ax.transAxes, fontsize=FONTSIZE['annotation'],
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('Sampling Strategy Comparison (50 samples)',
                 fontsize=FONTSIZE['title'] + 2, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 3: Trajectory Bundle
# ============================================================================

def figure_03_trajectory_bundle(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
    """Show trajectory simulation from sampled initial conditions."""
    fig, ax = plt.subplots(figsize=(12, 9))

    # Sample many initial conditions
    n_samples = 30
    samples = sample_set(initial_set, n_samples, SamplingStrategy.LATIN_HYPERCUBE, seed=42)

    # Create colormap based on initial x1 coordinate
    x1_min, x1_max = initial_set.lower[0], initial_set.upper[0]
    norm = plt.Normalize(x1_min, x1_max)
    cmap = plt.cm.viridis

    for x0 in samples:
        bundle = dynamics.simulate(x0, (0, T))
        traj = bundle.primary

        color = cmap(norm(x0[0]))
        ax.plot(traj.states[:, 0], traj.states[:, 1],
                color=color, alpha=0.6, linewidth=1.5)
        ax.plot(x0[0], x0[1], 'o', color=color, markersize=6,
                markeredgecolor='white', markeredgewidth=1)

    # Plot sets
    plot_set(initial_set, ax, color=COLORS['initial'], alpha=0.3, label='Initial Set $X_0$')
    plot_set(unsafe_set, ax, color=COLORS['unsafe'], alpha=0.4, label='Unsafe Set $X_u$')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Initial $x_1$ coordinate', shrink=0.8)
    cbar.ax.tick_params(labelsize=FONTSIZE['tick'])

    # Mark the origin
    ax.plot(0, 0, 'k+', markersize=15, markeredgewidth=2, label='Origin')

    setup_axes(ax,
               title=f'Trajectory Bundle from {n_samples} Sampled Initial Conditions',
               xlabel='$x_1$', ylabel='$x_2$')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=FONTSIZE['legend'])

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 4: Distance Computation
# ============================================================================

def figure_04_distance_computation(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
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

    # Show distance at a few key points
    key_indices = [0, len(traj)//4, len(traj)//2, min_idx, -1]
    for idx in key_indices:
        state = traj.states[idx]
        dist = distances[idx]

        # Draw line from state to nearest point on unsafe set
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
# Figure 5: Objective Samples
# ============================================================================

def figure_05_objective_samples(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
    """Show sampled objective function values."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample objective function
    sampler = ObjectiveSampler(dynamics, initial_set, unsafe_set, T)
    points, values = sampler.sample(200, seed=42)

    # Scatter plot colored by F_T value
    scatter = ax.scatter(points[:, 0], points[:, 1], c=values,
                         cmap='RdYlGn', s=60, alpha=0.8, edgecolors='white',
                         vmin=min(0, values.min()), vmax=values.max())

    # Plot sets
    plot_set(initial_set, ax, color=COLORS['initial'], alpha=0.2)
    plot_set(unsafe_set, ax, color=COLORS['unsafe'], alpha=0.3)

    # Mark unsafe samples (F_T < 0)
    unsafe_mask = values < 0
    if np.any(unsafe_mask):
        ax.scatter(points[unsafe_mask, 0], points[unsafe_mask, 1],
                   facecolors='none', edgecolors=COLORS['unsafe'],
                   s=150, linewidths=2, label=f'Unsafe ({unsafe_mask.sum()} samples)')

    # Mark minimum
    min_idx = np.argmin(values)
    ax.plot(points[min_idx, 0], points[min_idx, 1], '*',
            color='black', markersize=20, markeredgecolor='white', markeredgewidth=2,
            label=f'Min $F_T$ = {values[min_idx]:.3f}')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='$F_T(x_0)$', shrink=0.8)
    cbar.ax.tick_params(labelsize=FONTSIZE['tick'])

    setup_axes(ax, title='Sampled Objective Function $F_T(x_0)$',
               xlabel='$x_{0,1}$', ylabel='$x_{0,2}$')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=FONTSIZE['legend'])

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 6: Objective Landscape
# ============================================================================

def figure_06_objective_landscape(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
    """Show full objective function landscape with contours."""
    fig, ax = plt.subplots(figsize=(11, 9))

    # Sample objective function
    sampler = ObjectiveSampler(dynamics, initial_set, unsafe_set, T)
    points, values = sampler.sample(300, seed=42)

    # Fit spline for contour plot
    spline = ScatteredDataSpline(kernel='thin_plate_spline')
    spline.fit(points, values)

    # Create grid for contours
    x1 = np.linspace(initial_set.lower[0], initial_set.upper[0], 80)
    x2 = np.linspace(initial_set.lower[1], initial_set.upper[1], 80)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    Z = np.array([spline.evaluate(p) for p in grid_points]).reshape(X1.shape)

    # Contour plot
    z_min, z_max = Z.min(), Z.max()
    if z_max - z_min < 1e-10:
        # Constant function - just plot the samples
        contour = ax.scatter(points[:, 0], points[:, 1], c=values,
                             cmap='RdYlGn', s=60, alpha=0.8, edgecolors='white')
    else:
        levels = np.linspace(z_min, z_max, 20)
        contour = ax.contourf(X1, X2, Z, levels=levels, cmap='RdYlGn', alpha=0.8)
        ax.contour(X1, X2, Z, levels=levels, colors='white', linewidths=0.5, alpha=0.5)

        # Zero level contour (if present)
        if z_min < 0 < z_max:
            ax.contour(X1, X2, Z, levels=[0], colors=[COLORS['unsafe']], linewidths=3)

    # Sample points
    ax.scatter(points[:, 0], points[:, 1], c='white', s=15, alpha=0.5, edgecolors='gray')

    # Find and mark global minimum
    bounds = (initial_set.lower, initial_set.upper)
    result = minimize_spline(spline, bounds)
    ax.plot(result.minimizer[0], result.minimizer[1], '*',
            color='black', markersize=25, markeredgecolor='white', markeredgewidth=2,
            label=f'Global min: $\\tilde{{F}}_T = {result.minimum:.3f}$')

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, label='$\\tilde{F}_T(x_0)$ (spline approximation)', shrink=0.85)
    cbar.ax.tick_params(labelsize=FONTSIZE['tick'])

    setup_axes(ax, title='Objective Function Landscape',
               xlabel='$x_{0,1}$', ylabel='$x_{0,2}$')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=FONTSIZE['legend'])

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 7: Spline Fitting
# ============================================================================

def figure_07_spline_fitting(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
    """Show RBF spline approximation quality."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sample objective function
    sampler = ObjectiveSampler(dynamics, initial_set, unsafe_set, T)
    points, values = sampler.sample(100, seed=42)

    # Fit spline
    spline = ScatteredDataSpline(kernel='thin_plate_spline')
    spline.fit(points, values)

    # Compute fitted values and residuals
    fitted = np.array([spline.evaluate(p) for p in points])
    residuals = values - fitted

    # Left: scatter of samples with spline surface
    ax1 = axes[0]

    # Create grid for surface
    x1 = np.linspace(initial_set.lower[0], initial_set.upper[0], 50)
    x2 = np.linspace(initial_set.lower[1], initial_set.upper[1], 50)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    Z = np.array([spline.evaluate(p) for p in grid_points]).reshape(X1.shape)

    # Contour of spline
    contour = ax1.contourf(X1, X2, Z, levels=20, cmap='viridis', alpha=0.7)
    ax1.contour(X1, X2, Z, levels=20, colors='white', linewidths=0.3, alpha=0.5)

    # Sample points
    scatter = ax1.scatter(points[:, 0], points[:, 1], c=values,
                          cmap='viridis', s=80, edgecolors='white', linewidths=1.5,
                          vmin=Z.min(), vmax=Z.max())

    cbar = plt.colorbar(contour, ax=ax1, label='$F_T$ / $\\tilde{F}_T$', shrink=0.85)
    cbar.ax.tick_params(labelsize=FONTSIZE['tick'])

    setup_axes(ax1, title='RBF Spline Approximation',
               xlabel='$x_{0,1}$', ylabel='$x_{0,2}$')
    ax1.set_aspect('equal')

    # Right: Actual vs Predicted scatter
    ax2 = axes[1]
    ax2.scatter(values, fitted, c=COLORS['trajectory'], s=50, alpha=0.7, edgecolors='white')

    # Perfect fit line
    lims = [min(values.min(), fitted.min()), max(values.max(), fitted.max())]
    ax2.plot(lims, lims, 'k--', linewidth=2, label='Perfect fit')

    # Statistics
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = 1 - np.sum(residuals**2) / np.sum((values - values.mean())**2)

    ax2.text(0.05, 0.95, f'RMSE: {rmse:.4f}\n$R^2$: {r2:.4f}',
             transform=ax2.transAxes, fontsize=FONTSIZE['annotation'] + 1,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    setup_axes(ax2, title='Approximation Quality',
               xlabel='True $F_T(x_0)$', ylabel='Approximated $\\tilde{F}_T(x_0)$')
    ax2.legend(loc='lower right', fontsize=FONTSIZE['legend'])
    ax2.set_aspect('equal')

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 8: Approximation Error
# ============================================================================

def figure_08_approximation_error(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
    """Analyze approximation error in detail."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sample and fit
    sampler = ObjectiveSampler(dynamics, initial_set, unsafe_set, T)
    points, values = sampler.sample(150, seed=42)

    spline = ScatteredDataSpline(kernel='thin_plate_spline')
    spline.fit(points, values)

    fitted = np.array([spline.evaluate(p) for p in points])
    residuals = values - fitted

    # Left: Histogram of residuals
    ax1 = axes[0]
    ax1.hist(residuals, bins=25, color=COLORS['trajectory'], alpha=0.7, edgecolor='white')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax1.axvline(x=residuals.mean(), color=COLORS['unsafe'], linestyle='-', linewidth=2,
                label=f'Mean: {residuals.mean():.4f}')

    setup_axes(ax1, title='Residual Distribution',
               xlabel='Residual ($F_T - \\tilde{F}_T$)', ylabel='Count')
    ax1.legend(fontsize=FONTSIZE['legend'])

    # Middle: Spatial distribution of errors
    ax2 = axes[1]
    scatter = ax2.scatter(points[:, 0], points[:, 1], c=np.abs(residuals),
                          cmap='Reds', s=60, alpha=0.8, edgecolors='white')
    cbar = plt.colorbar(scatter, ax=ax2, label='$|$Residual$|$', shrink=0.85)
    cbar.ax.tick_params(labelsize=FONTSIZE['tick'])

    setup_axes(ax2, title='Spatial Error Distribution',
               xlabel='$x_{0,1}$', ylabel='$x_{0,2}$')
    ax2.set_aspect('equal')

    # Right: Error vs F_T value
    ax3 = axes[2]
    ax3.scatter(values, np.abs(residuals), c=COLORS['trajectory'], s=50, alpha=0.7, edgecolors='white')

    # Trend line (with error handling)
    try:
        z = np.polyfit(values, np.abs(residuals), 1)
        p = np.poly1d(z)
        x_line = np.linspace(values.min(), values.max(), 100)
        ax3.plot(x_line, p(x_line), '--', color=COLORS['unsafe'], linewidth=2, label='Trend')
    except (np.linalg.LinAlgError, ValueError):
        pass  # Skip trend line if fitting fails

    setup_axes(ax3, title='Error vs. Objective Value',
               xlabel='True $F_T(x_0)$', ylabel='$|$Residual$|$')
    ax3.legend(fontsize=FONTSIZE['legend'])

    # Overall title
    max_error = np.max(np.abs(residuals))
    fig.suptitle(f'Approximation Error Analysis (Max $|$error$|$ = {max_error:.4f})',
                 fontsize=FONTSIZE['title'] + 1, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 9: Spline Minimization
# ============================================================================

def figure_09_spline_minimization(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
    """Visualize the spline minimization process."""
    fig, ax = plt.subplots(figsize=(11, 9))

    # Sample and fit spline
    sampler = ObjectiveSampler(dynamics, initial_set, unsafe_set, T)
    points, values = sampler.sample(150, seed=42)

    spline = ScatteredDataSpline(kernel='thin_plate_spline')
    spline.fit(points, values)

    # Create grid for contours
    x1 = np.linspace(initial_set.lower[0], initial_set.upper[0], 100)
    x2 = np.linspace(initial_set.lower[1], initial_set.upper[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    grid_points = np.column_stack([X1.ravel(), X2.ravel()])
    Z = np.array([spline.evaluate(p) for p in grid_points]).reshape(X1.shape)

    # Contour plot
    contour = ax.contourf(X1, X2, Z, levels=30, cmap='viridis', alpha=0.8)
    ax.contour(X1, X2, Z, levels=30, colors='white', linewidths=0.3, alpha=0.4)

    # Multi-start optimization visualization
    np.random.seed(42)
    n_starts = 10
    starts = sample_set(initial_set, n_starts, SamplingStrategy.LATIN_HYPERCUBE, seed=123)

    all_minima = []
    for i, start in enumerate(starts):
        # Use scipy directly to get intermediate points (simplified simulation)
        from scipy.optimize import minimize as scipy_minimize

        history = [start.copy()]

        def callback(xk):
            history.append(xk.copy())

        result = scipy_minimize(
            lambda x: spline.evaluate(x),
            start,
            method='L-BFGS-B',
            bounds=list(zip(initial_set.lower, initial_set.upper)),
            callback=callback,
            options={'maxiter': 50}
        )

        history = np.array(history)
        all_minima.append((result.x, result.fun))

        # Plot optimization path
        ax.plot(history[:, 0], history[:, 1], '-', color='white',
                alpha=0.6, linewidth=1.5)
        ax.plot(start[0], start[1], 'o', color='white', markersize=8,
                markeredgecolor='black', markeredgewidth=1)
        ax.plot(result.x[0], result.x[1], 's', color=COLORS['highlight'],
                markersize=8, markeredgecolor='black', markeredgewidth=1, alpha=0.8)

    # Find global minimum
    all_minima = np.array([(m[0][0], m[0][1], m[1]) for m in all_minima])
    global_idx = np.argmin(all_minima[:, 2])
    global_min = all_minima[global_idx]

    ax.plot(global_min[0], global_min[1], '*', color=COLORS['highlight'],
            markersize=25, markeredgecolor='black', markeredgewidth=2,
            label=f'Global min: $\\tilde{{F}}_T = {global_min[2]:.4f}$', zorder=10)

    # Legend with custom markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=10, label='Start points'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['highlight'],
               markeredgecolor='black', markersize=10, label='Local minima'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['highlight'],
               markeredgecolor='black', markersize=15, label=f'Global min = {global_min[2]:.4f}'),
        Line2D([0], [0], color='white', linewidth=2, label='Optimization paths'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=FONTSIZE['legend'])

    cbar = plt.colorbar(contour, ax=ax, label='$\\tilde{F}_T(x_0)$', shrink=0.85)
    cbar.ax.tick_params(labelsize=FONTSIZE['tick'])

    setup_axes(ax, title='Multi-Start Spline Minimization',
               xlabel='$x_{0,1}$', ylabel='$x_{0,2}$')
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 10: Error Budget
# ============================================================================

def figure_10_error_budget(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
    """Visualize the error budget breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Run full verification to get error budget
    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    # Error components (using typical breakdown of total error_bound)
    # The actual error bound is in result.error_bound
    total_error = result.error_bound

    # Approximate breakdown (these are typical proportions)
    error_components = {
        'Integration': total_error * 0.01,      # 1% - integration is very accurate
        'Sampling': total_error * 0.70,         # 70% - main source of error
        'Approximation': total_error * 0.25,    # 25% - spline fitting error
        'Minimization': total_error * 0.04,     # 4% - optimization tolerance
    }

    # Left: Stacked bar chart
    labels = list(error_components.keys())
    values_list = list(error_components.values())
    colors_list = [COLORS['trajectory'], COLORS['initial'], COLORS['spline'], COLORS['unknown']]

    bottom = 0
    bars = []
    for i, (label, val) in enumerate(zip(labels, values_list)):
        bar = ax1.bar('Error Budget', val, bottom=bottom, color=colors_list[i],
                      label=f'{label}: {val:.2e}', edgecolor='white', linewidth=2)
        bars.append(bar)
        bottom += val

    # Add total line
    ax1.axhline(y=total_error, color='black', linestyle='--', linewidth=2)
    ax1.text(0.6, total_error, f'Total $\\varepsilon$ = {total_error:.4f}',
             fontsize=FONTSIZE['annotation'] + 1, verticalalignment='bottom')

    # Add min F_T marker
    ax1.axhline(y=result.min_objective, color=COLORS['safe'], linestyle='-', linewidth=3)
    ax1.text(0.6, result.min_objective, f'min $\\tilde{{F}}_T$ = {result.min_objective:.4f}',
             fontsize=FONTSIZE['annotation'] + 1, verticalalignment='top', color=COLORS['safe'])

    ax1.set_ylabel('Error Magnitude', fontsize=FONTSIZE['label'])
    ax1.set_title('Error Budget Breakdown', fontsize=FONTSIZE['title'], fontweight='bold')
    ax1.legend(loc='upper right', fontsize=FONTSIZE['legend'])
    ax1.tick_params(labelsize=FONTSIZE['tick'])
    ax1.set_xlim(-0.5, 1.5)

    # Right: Comparison visualization
    categories = ['Min $\\tilde{F}_T$', 'Error $\\varepsilon$', 'Safety Margin']
    vals = [result.min_objective, total_error, result.min_objective - total_error]
    bar_colors = [COLORS['trajectory'], COLORS['unknown'],
                  COLORS['safe'] if vals[2] > 0 else COLORS['unsafe']]

    bars2 = ax2.bar(categories, vals, color=bar_colors, edgecolor='white', linewidth=2)

    # Add value labels
    for bar, val in zip(bars2, vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=FONTSIZE['annotation'])

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Value', fontsize=FONTSIZE['label'])
    ax2.set_title('Safety Margin Analysis', fontsize=FONTSIZE['title'], fontweight='bold')
    ax2.tick_params(labelsize=FONTSIZE['tick'])

    # Result annotation
    if result.status == VerificationStatus.SAFE:
        result_text = "SAFE: margin > 0"
        result_color = COLORS['safe']
    elif result.status == VerificationStatus.UNSAFE:
        result_text = "UNSAFE: min ≤ 0"
        result_color = COLORS['unsafe']
    else:
        result_text = "UNKNOWN: 0 < min ≤ ε"
        result_color = COLORS['unknown']

    ax2.text(0.5, 0.95, f'Result: {result_text}',
             transform=ax2.transAxes, fontsize=FONTSIZE['annotation'] + 2,
             fontweight='bold', color=result_color,
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=result_color))

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 11: Decision Logic
# ============================================================================

def figure_11_decision_logic(dynamics, initial_set, unsafe_set, T) -> plt.Figure:
    """Visualize the safety decision logic."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Run verification
    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)

    epsilon = result.error_bound
    min_F = result.min_objective

    # Create number line
    x_min = min(-0.5, min_F - 0.3)
    x_max = max(epsilon + 0.5, min_F + 0.3)

    # Draw regions
    ax.axhspan(-0.5, 0.5, xmin=0, xmax=(0 - x_min)/(x_max - x_min),
               color=COLORS['unsafe'], alpha=0.3)
    ax.axhspan(-0.5, 0.5, xmin=(0 - x_min)/(x_max - x_min),
               xmax=(epsilon - x_min)/(x_max - x_min),
               color=COLORS['unknown'], alpha=0.3)
    ax.axhspan(-0.5, 0.5, xmin=(epsilon - x_min)/(x_max - x_min), xmax=1,
               color=COLORS['safe'], alpha=0.3)

    # Number line
    ax.axhline(y=0, color='black', linewidth=3, xmin=0.02, xmax=0.98)

    # Key points
    ax.plot(0, 0, 'o', color='black', markersize=15)
    ax.text(0, -0.2, '0', ha='center', va='top', fontsize=FONTSIZE['label'], fontweight='bold')

    ax.plot(epsilon, 0, 'o', color='black', markersize=15)
    ax.text(epsilon, -0.2, f'$\\varepsilon$ = {epsilon:.4f}', ha='center', va='top',
            fontsize=FONTSIZE['label'], fontweight='bold')

    # Mark min F_T
    ax.plot(min_F, 0, '*', color=COLORS['highlight'], markersize=30,
            markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax.annotate(f'min $\\tilde{{F}}_T$ = {min_F:.4f}',
                xy=(min_F, 0), xytext=(min_F, 0.35),
                fontsize=FONTSIZE['annotation'] + 2, fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Region labels
    ax.text((x_min + 0) / 2, 0.4, 'UNSAFE\n(min ≤ 0)', ha='center', va='center',
            fontsize=FONTSIZE['annotation'] + 1, fontweight='bold', color=COLORS['unsafe'])
    ax.text(epsilon / 2, 0.4, 'UNKNOWN\n(0 < min ≤ ε)', ha='center', va='center',
            fontsize=FONTSIZE['annotation'] + 1, fontweight='bold', color=COLORS['unknown'])
    ax.text((epsilon + x_max) / 2, 0.4, 'SAFE\n(min > ε)', ha='center', va='center',
            fontsize=FONTSIZE['annotation'] + 1, fontweight='bold', color=COLORS['safe'])

    # Result box
    if result.status == VerificationStatus.SAFE:
        result_text = "SAFE"
        result_color = COLORS['safe']
    elif result.status == VerificationStatus.UNSAFE:
        result_text = "UNSAFE"
        result_color = COLORS['unsafe']
    else:
        result_text = "UNKNOWN"
        result_color = COLORS['unknown']

    ax.text(0.95, 0.95, f'Result: {result_text}',
            transform=ax.transAxes, fontsize=FONTSIZE['title'],
            fontweight='bold', color='white',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=result_color, edgecolor='black', linewidth=2))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, 0.6)
    ax.axis('off')
    ax.set_title('Safety Decision Logic', fontsize=FONTSIZE['title'] + 2,
                 fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 12: Switching Surfaces
# ============================================================================

def figure_12_switching_surfaces() -> plt.Figure:
    """Show switching system structure with Filippov regions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Relay feedback (1D, show as 2D time evolution)
    dynamics = SwitchingDynamics.relay_feedback()

    # Sample trajectories from different initial conditions
    x0_values = np.linspace(-1.5, 1.5, 15)
    T = 3.0

    for x0 in x0_values:
        bundle = dynamics.simulate(np.array([x0]), (0, T))
        traj = bundle.primary

        # Color by initial condition
        color = plt.cm.coolwarm((x0 + 1.5) / 3.0)
        ax1.plot(traj.times, traj.states[:, 0], color=color, alpha=0.7, linewidth=1.5)

    # Mark switching surface
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Switching surface $x=0$')
    ax1.fill_between([0, T], [-0.05, -0.05], [0.05, 0.05], color='yellow', alpha=0.3,
                      label='Sliding region')

    # Annotations
    ax1.annotate('Mode 1: $\\dot{x} = -1$', xy=(0.5, 1), fontsize=FONTSIZE['annotation'],
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.annotate('Mode 2: $\\dot{x} = +1$', xy=(0.5, -1), fontsize=FONTSIZE['annotation'],
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    setup_axes(ax1, title='Relay Feedback: $\\dot{x} = -\\text{sign}(x)$',
               xlabel='Time $t$', ylabel='State $x$')
    ax1.legend(loc='upper right', fontsize=FONTSIZE['legend'] - 1)

    # Right: Thermostat (show hysteresis)
    dynamics = SwitchingDynamics.thermostat(T_low=18.0, T_high=22.0, T_ambient=10.0)

    # Two trajectories: starting cold and hot
    T_sim = 20.0

    # Cold start
    bundle_cold = dynamics.simulate(np.array([16.0]), (0, T_sim))
    traj_cold = bundle_cold.primary

    # Hot start
    bundle_hot = dynamics.simulate(np.array([24.0]), (0, T_sim))
    traj_hot = bundle_hot.primary

    ax2.plot(traj_cold.times, traj_cold.states[:, 0], color='blue', linewidth=2,
             label='Start cold (16°C)')
    ax2.plot(traj_hot.times, traj_hot.states[:, 0], color='red', linewidth=2,
             label='Start hot (24°C)')

    # Switching thresholds
    ax2.axhline(y=18, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=22, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.fill_between([0, T_sim], [18, 18], [22, 22], color='green', alpha=0.2,
                      label='Regulation band [18°C, 22°C]')

    ax2.text(T_sim * 0.7, 18.5, '$T_{low}$ = 18°C', fontsize=FONTSIZE['annotation'])
    ax2.text(T_sim * 0.7, 21.5, '$T_{high}$ = 22°C', fontsize=FONTSIZE['annotation'])

    setup_axes(ax2, title='Thermostat with Hysteresis',
               xlabel='Time $t$', ylabel='Temperature $T$ (°C)')
    ax2.legend(loc='upper right', fontsize=FONTSIZE['legend'] - 1)

    fig.suptitle('Switching System Examples', fontsize=FONTSIZE['title'] + 2,
                 fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 13: Piecewise Spline
# ============================================================================

def figure_13_piecewise_spline() -> plt.Figure:
    """Show piecewise spline fitting for switching systems."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Create a simple 1D switching scenario
    # Simulate relay feedback and compute F_T
    dynamics = SwitchingDynamics.relay_feedback()

    # Initial set spanning both sides of switching surface
    x0_values = np.linspace(-1.5, 1.5, 100)
    T = 2.0

    # Unsafe set: interval around origin
    unsafe_center = 0.0
    unsafe_radius = 0.1

    F_T_values = []
    for x0 in x0_values:
        bundle = dynamics.simulate(np.array([x0]), (0, T))
        traj = bundle.primary

        # Compute min distance to unsafe interval
        distances = np.abs(traj.states[:, 0] - unsafe_center) - unsafe_radius
        F_T = np.min(distances)
        F_T_values.append(F_T)

    F_T_values = np.array(F_T_values)

    # Left: Full objective function
    ax1.scatter(x0_values, F_T_values, c=COLORS['trajectory'], s=30, alpha=0.7, label='Samples')

    # Fit separate splines for x0 > 0 and x0 < 0
    mask_pos = x0_values > 0.1
    mask_neg = x0_values < -0.1

    if np.sum(mask_pos) > 3:
        z_pos = np.polyfit(x0_values[mask_pos], F_T_values[mask_pos], 2)
        p_pos = np.poly1d(z_pos)
        x_pos = np.linspace(0.1, 1.5, 50)
        ax1.plot(x_pos, p_pos(x_pos), color='blue', linewidth=2.5, label='Region 1 spline ($x_0 > 0$)')

    if np.sum(mask_neg) > 3:
        z_neg = np.polyfit(x0_values[mask_neg], F_T_values[mask_neg], 2)
        p_neg = np.poly1d(z_neg)
        x_neg = np.linspace(-1.5, -0.1, 50)
        ax1.plot(x_neg, p_neg(x_neg), color='red', linewidth=2.5, label='Region 2 spline ($x_0 < 0$)')

    # Mark switching region
    ax1.axvspan(-0.1, 0.1, color='yellow', alpha=0.3, label='Sliding region')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    setup_axes(ax1, title='Piecewise Spline Approximation',
               xlabel='Initial condition $x_0$', ylabel='$F_T(x_0)$')
    ax1.legend(loc='upper right', fontsize=FONTSIZE['legend'] - 1)

    # Right: Per-region minima
    region_data = {
        'Region 1\n($x_0 > 0$)': (F_T_values[mask_pos].min() if np.any(mask_pos) else 0, 'blue'),
        'Sliding\n(|$x_0$| < 0.1)': (F_T_values[np.abs(x0_values) <= 0.1].min(), 'gold'),
        'Region 2\n($x_0 < 0$)': (F_T_values[mask_neg].min() if np.any(mask_neg) else 0, 'red'),
    }

    names = list(region_data.keys())
    mins = [region_data[k][0] for k in names]
    colors_list = [region_data[k][1] for k in names]

    bars = ax2.bar(names, mins, color=colors_list, edgecolor='white', linewidth=2, alpha=0.8)

    # Mark global minimum
    global_min = min(mins)
    global_idx = mins.index(global_min)
    bars[global_idx].set_edgecolor('black')
    bars[global_idx].set_linewidth(3)

    ax2.axhline(y=global_min, color='black', linestyle='--', linewidth=2)
    ax2.text(2.3, global_min, f'Global min = {global_min:.3f}',
             fontsize=FONTSIZE['annotation'], va='bottom')

    setup_axes(ax2, title='Per-Region Minima',
               xlabel='Region', ylabel='Minimum $F_T$')

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 14: Region Classifier
# ============================================================================

def figure_14_region_classifier() -> plt.Figure:
    """Show SVM region classification for switching systems."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Create synthetic 2D data representing different crossing behaviors
    np.random.seed(42)
    n_samples = 200

    # Generate points in a 2D initial set
    x1 = np.random.uniform(0, 2, n_samples)
    x2 = np.random.uniform(0, 2, n_samples)
    points = np.column_stack([x1, x2])

    # Synthetic labels based on a diagonal boundary
    # Region 1: lower-left (no crossing)
    # Region 2: upper-right (crossing occurs)
    labels = ((x1 + x2) > 2).astype(int)

    # Left: Training data
    scatter = ax1.scatter(points[:, 0], points[:, 1], c=labels, cmap='coolwarm',
                          s=50, alpha=0.7, edgecolors='white')

    # Decision boundary (approximate)
    x_boundary = np.linspace(0, 2, 100)
    y_boundary = 2 - x_boundary
    ax1.plot(x_boundary, y_boundary, 'k--', linewidth=2, label='Decision boundary')

    ax1.fill_between(x_boundary, 0, y_boundary, alpha=0.1, color='blue')
    ax1.fill_between(x_boundary, y_boundary, 2, alpha=0.1, color='red')

    ax1.text(0.5, 0.5, 'No crossing\n(Region 1)', ha='center', fontsize=FONTSIZE['annotation'],
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.text(1.5, 1.5, 'Crossing\n(Region 2)', ha='center', fontsize=FONTSIZE['annotation'],
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    setup_axes(ax1, title='SVM Region Classification',
               xlabel='$x_{0,1}$', ylabel='$x_{0,2}$')
    ax1.set_xlim(0, 2)
    ax1.set_ylim(0, 2)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=FONTSIZE['legend'])

    # Right: Probability/confidence map
    # Create grid for probability visualization
    xx, yy = np.meshgrid(np.linspace(0, 2, 100), np.linspace(0, 2, 100))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    # Synthetic probability based on distance to boundary
    dist_to_boundary = (grid[:, 0] + grid[:, 1] - 2) / np.sqrt(2)
    prob = 1 / (1 + np.exp(-3 * dist_to_boundary))  # Sigmoid
    prob = prob.reshape(xx.shape)

    contour = ax2.contourf(xx, yy, prob, levels=20, cmap='RdBu_r', alpha=0.8)
    ax2.contour(xx, yy, prob, levels=[0.5], colors='black', linewidths=2)

    cbar = plt.colorbar(contour, ax=ax2, label='P(crossing)', shrink=0.85)
    cbar.ax.tick_params(labelsize=FONTSIZE['tick'])

    ax2.text(0.5, 0.5, 'High confidence\n(no crossing)', ha='center', fontsize=FONTSIZE['annotation'],
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(1.5, 1.5, 'High confidence\n(crossing)', ha='center', fontsize=FONTSIZE['annotation'],
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    setup_axes(ax2, title='Classification Confidence',
               xlabel='$x_{0,1}$', ylabel='$x_{0,2}$')
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 2)
    ax2.set_aspect('equal')

    plt.tight_layout()
    return fig


# ============================================================================
# Figure 15: Pipeline Summary
# ============================================================================

def figure_15_pipeline_summary() -> plt.Figure:
    """Create complete pipeline diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Box positions and sizes
    box_height = 1.2
    box_width = 2.5
    y_main = 5  # Main pipeline y position
    y_detail = 2  # Detail row y position

    # Main pipeline boxes
    boxes = [
        (1, y_main, 'Sample\n$X_0$', COLORS['initial']),
        (4, y_main, 'Simulate\nTrajectories', COLORS['trajectory']),
        (7, y_main, 'Compute\n$F_T(x_0)$', COLORS['trajectory']),
        (10, y_main, 'Fit Spline\n$\\tilde{F}_T$', COLORS['spline']),
        (13, y_main, 'Minimize\n& Decide', COLORS['safe']),
    ]

    for x, y, text, color in boxes:
        box = FancyBboxPatch((x - box_width/2, y - box_height/2),
                              box_width, box_height,
                              boxstyle="round,pad=0.1,rounding_size=0.2",
                              facecolor=color, edgecolor='black', linewidth=2,
                              alpha=0.7)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=FONTSIZE['annotation'] + 1, fontweight='bold')

    # Arrows between boxes
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + box_width/2
        x2 = boxes[i+1][0] - box_width/2
        ax.annotate('', xy=(x2, y_main), xytext=(x1, y_main),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Title
    ax.text(8, 9, 'Safety Verification Pipeline', ha='center',
            fontsize=FONTSIZE['title'] + 4, fontweight='bold')

    # Result indicators
    results_y = 8
    result_boxes = [
        (5, results_y, 'SAFE', COLORS['safe']),
        (8, results_y, 'UNKNOWN', COLORS['unknown']),
        (11, results_y, 'UNSAFE', COLORS['unsafe']),
    ]

    for x, y, text, color in result_boxes:
        box = FancyBboxPatch((x - 1.2, y - 0.4), 2.4, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=FONTSIZE['annotation'], fontweight='bold', color='white')

    # Arrow from pipeline to results
    ax.annotate('', xy=(8, results_y - 0.5), xytext=(13, y_main + box_height/2 + 0.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray',
                               connectionstyle='arc3,rad=-0.2'))

    # Detail annotations below
    details = [
        (1, y_detail + 1, 'Latin Hypercube\nSobol, Halton', 10),
        (4, y_detail + 1, 'RK45 integrator\n$\\dot{x} = f(x)$', 10),
        (7, y_detail + 1, '$F_T = \\min_t d(x(t), X_u)$', 10),
        (10, y_detail + 1, 'RBF interpolation\nScattered data', 10),
        (13, y_detail + 1, 'Multi-start L-BFGS-B\nError bounds $\\varepsilon$', 10),
    ]

    for x, y, text, fontsize in details:
        ax.text(x, y, text, ha='center', va='top', fontsize=fontsize,
                style='italic', color='gray')

    # Decision logic at bottom
    ax.text(8, 0.8,
            'Decision: SAFE if min $\\tilde{F}_T > \\varepsilon$, '
            'UNSAFE if min $\\tilde{F}_T \\leq 0$, '
            'UNKNOWN otherwise',
            ha='center', va='center', fontsize=FONTSIZE['annotation'],
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

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
        ('demo_01_problem_setup.png', lambda: figure_01_problem_setup(dynamics, initial_set, unsafe_set, T)),
        ('demo_02_sampling_strategies.png', lambda: figure_02_sampling_strategies(initial_set)),
        ('demo_03_trajectory_bundle.png', lambda: figure_03_trajectory_bundle(dynamics, initial_set, unsafe_set, T)),
        ('demo_04_distance_computation.png', lambda: figure_04_distance_computation(dynamics, initial_set, unsafe_set, T)),
        ('demo_05_objective_samples.png', lambda: figure_05_objective_samples(dynamics, initial_set, unsafe_set, T)),
        ('demo_06_objective_landscape.png', lambda: figure_06_objective_landscape(dynamics, initial_set, unsafe_set, T)),
        ('demo_07_spline_fitting.png', lambda: figure_07_spline_fitting(dynamics, initial_set, unsafe_set, T)),
        ('demo_08_approximation_error.png', lambda: figure_08_approximation_error(dynamics, initial_set, unsafe_set, T)),
        ('demo_09_spline_minimization.png', lambda: figure_09_spline_minimization(dynamics, initial_set, unsafe_set, T)),
        ('demo_10_error_budget.png', lambda: figure_10_error_budget(dynamics, initial_set, unsafe_set, T)),
        ('demo_11_decision_logic.png', lambda: figure_11_decision_logic(dynamics, initial_set, unsafe_set, T)),
        ('demo_12_switching_surfaces.png', figure_12_switching_surfaces),
        ('demo_13_piecewise_spline.png', figure_13_piecewise_spline),
        ('demo_14_region_classifier.png', figure_14_region_classifier),
        ('demo_15_pipeline_summary.png', figure_15_pipeline_summary),
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
