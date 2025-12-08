"""Visualization utilities for trajectories and verification results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon

if TYPE_CHECKING:
    from ..dynamics.trajectory import Trajectory
    from ..geometry.sets import Set, HyperRectangle, Ball
    from ..verification.verifier import VerificationResult


def plot_trajectory(
    trajectory: Trajectory,
    ax: plt.Axes | None = None,
    dims: tuple[int, int] = (0, 1),
    **kwargs
) -> plt.Axes:
    """Plot a trajectory in 2D.

    Args:
        trajectory: Trajectory to plot.
        ax: Matplotlib axes (creates new if None).
        dims: Which dimensions to plot (for n_dims > 2).
        **kwargs: Additional arguments to plt.plot.

    Returns:
        The matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if trajectory.n_dims == 1:
        ax.plot(trajectory.times, trajectory.states[:, 0], **kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('State')
    else:
        d0, d1 = dims
        ax.plot(trajectory.states[:, d0], trajectory.states[:, d1], **kwargs)
        ax.set_xlabel(f'x[{d0}]')
        ax.set_ylabel(f'x[{d1}]')

        # Mark start and end
        ax.plot(trajectory.states[0, d0], trajectory.states[0, d1],
                'go', markersize=8, label='Start')
        ax.plot(trajectory.states[-1, d0], trajectory.states[-1, d1],
                'rs', markersize=8, label='End')

    return ax


def plot_set(
    s: Set,
    ax: plt.Axes | None = None,
    dims: tuple[int, int] = (0, 1),
    color: str = 'red',
    alpha: float = 0.3,
    label: str | None = None,
    **kwargs
) -> plt.Axes:
    """Plot a 2D projection of a set.

    Args:
        s: Set to plot.
        ax: Matplotlib axes.
        dims: Dimensions to project onto.
        color: Fill color.
        alpha: Transparency.
        label: Legend label.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    d0, d1 = dims

    # Import set types
    from ..geometry.sets import HyperRectangle, Ball

    if isinstance(s, HyperRectangle):
        rect = Rectangle(
            (s.lower[d0], s.lower[d1]),
            s.upper[d0] - s.lower[d0],
            s.upper[d1] - s.lower[d1],
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            label=label,
            **kwargs
        )
        ax.add_patch(rect)

    elif isinstance(s, Ball):
        if s.n_dims == 2 or (d0 < 2 and d1 < 2):
            circle = Circle(
                (s.center[d0], s.center[d1]),
                s.radius,
                facecolor=color,
                alpha=alpha,
                edgecolor=color,
                label=label,
                **kwargs
            )
            ax.add_patch(circle)
        else:
            # For higher dimensions, show as a circle with projected radius
            circle = Circle(
                (s.center[d0], s.center[d1]),
                s.radius,
                facecolor=color,
                alpha=alpha,
                edgecolor=color,
                label=label,
                **kwargs
            )
            ax.add_patch(circle)

    else:
        # Generic: sample boundary points
        if hasattr(s, 'sample'):
            points = s.sample(500)
            ax.scatter(points[:, d0], points[:, d1], c=color, alpha=alpha,
                      s=1, label=label)

    return ax


def plot_objective_function(
    points: np.ndarray,
    values: np.ndarray,
    approximation=None,
    ax: plt.Axes | None = None,
    dims: tuple[int, int] = (0, 1),
    n_grid: int = 50,
    **kwargs
) -> plt.Axes:
    """Plot the sampled objective function.

    Args:
        points: Sample points, shape (n, d).
        values: Objective values at samples.
        approximation: Optional spline approximation to plot as contours.
        ax: Matplotlib axes.
        dims: Dimensions to plot.
        n_grid: Grid resolution for contours.
        **kwargs: Additional arguments.

    Returns:
        The matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    d0, d1 = dims

    if points.shape[1] == 1:
        # 1D case
        ax.scatter(points[:, 0], values, c='blue', alpha=0.5, label='Samples')
        if approximation is not None:
            x_grid = np.linspace(points.min(), points.max(), 100)
            y_grid = np.array([approximation.evaluate(np.array([x])) for x in x_grid])
            ax.plot(x_grid, y_grid, 'r-', label='Approximation')
        ax.set_xlabel('x')
        ax.set_ylabel('F_T(x)')
    else:
        # 2D scatter plot
        scatter = ax.scatter(
            points[:, d0], points[:, d1],
            c=values, cmap='viridis', alpha=0.7
        )
        plt.colorbar(scatter, ax=ax, label='F_T')

        if approximation is not None:
            # Contour plot of approximation
            x_min, x_max = points[:, d0].min(), points[:, d0].max()
            y_min, y_max = points[:, d1].min(), points[:, d1].max()

            x_grid = np.linspace(x_min, x_max, n_grid)
            y_grid = np.linspace(y_min, y_max, n_grid)
            X, Y = np.meshgrid(x_grid, y_grid)

            Z = np.zeros_like(X)
            for i in range(n_grid):
                for j in range(n_grid):
                    point = np.zeros(points.shape[1])
                    point[d0] = X[i, j]
                    point[d1] = Y[i, j]
                    Z[i, j] = approximation.evaluate(point)

            ax.contour(X, Y, Z, levels=10, colors='red', alpha=0.5)

        ax.set_xlabel(f'x[{d0}]')
        ax.set_ylabel(f'x[{d1}]')

    return ax


def plot_verification_result(
    result: VerificationResult,
    trajectories: list = None,
    initial_set: Set = None,
    unsafe_set: Set = None,
    ax: plt.Axes | None = None,
    dims: tuple[int, int] = (0, 1)
) -> plt.Axes:
    """Plot verification result with trajectories and sets.

    Args:
        result: Verification result.
        trajectories: Optional list of trajectories to plot.
        initial_set: Initial set to plot.
        unsafe_set: Unsafe set to plot.
        ax: Matplotlib axes.
        dims: Dimensions to plot.

    Returns:
        The matplotlib axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot sets
    if initial_set is not None:
        plot_set(initial_set, ax, dims, color='green', alpha=0.2, label='Initial set')

    if unsafe_set is not None:
        plot_set(unsafe_set, ax, dims, color='red', alpha=0.3, label='Unsafe set')

    # Plot trajectories
    if trajectories is not None:
        for i, traj in enumerate(trajectories):
            label = 'Trajectories' if i == 0 else None
            plot_trajectory(traj, ax, dims, color='blue', alpha=0.3, label=label)

    # Mark minimizer
    d0, d1 = dims
    ax.plot(result.minimizer[d0], result.minimizer[d1],
            'k*', markersize=15, label=f'Minimizer (F={result.min_objective:.3f})')

    # Mark counterexample if present
    if result.counterexample is not None:
        ax.plot(result.counterexample[d0], result.counterexample[d1],
                'rx', markersize=15, mew=3, label='Counterexample')

    # Title with result
    status_colors = {
        'SAFE': 'green',
        'UNSAFE': 'red',
        'UNKNOWN': 'orange'
    }
    ax.set_title(
        f'Verification Result: {result.status.name}',
        color=status_colors.get(result.status.name, 'black'),
        fontsize=14,
        fontweight='bold'
    )

    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    return ax
