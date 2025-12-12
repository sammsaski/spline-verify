"""Problem definition for distance estimation.

Defines the UnsafeSupport class that specifies the distance estimation
problem: initial set, unsafe set, dynamics, and distance function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class UnsafeSupport:
    """Problem definition for distance estimation.

    Mirrors the MATLAB `unsafe_support` class from Miller & Sznaier's
    implementation.

    The problem is to find:
        P* = inf_{t, x0, y} c(x(t | x0), y)
        subject to: ẋ = f(t,x), x(0) = x0 ∈ X0, y ∈ Xu

    Where:
    - x(t | x0) is the trajectory starting from x0
    - X0 is the initial set
    - Xu is the unsafe set
    - c(x, y) is the distance function (typically squared Euclidean)

    Attributes:
        n_vars: State dimension.
        time_horizon: Time horizon T.
        dynamics: Dynamics function f(t, x) -> dx/dt.
        initial_center: Center of initial set.
        initial_radius: Radius of initial set (ball constraint).
        initial_constraints: List of polynomial constraints g(x) >= 0.
        unsafe_center: Center of unsafe set.
        unsafe_radius: Radius of unsafe set (ball constraint).
        unsafe_constraints: List of polynomial constraints h(y) >= 0.
        distance_type: Type of distance ('euclidean' or 'squared_euclidean').
    """
    n_vars: int
    time_horizon: float
    dynamics: Callable[[float, np.ndarray], np.ndarray]

    # Initial set X0 (ball by default)
    initial_center: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    initial_radius: float = 1.0
    initial_constraints: list[Callable] = field(default_factory=list)

    # Unsafe set Xu (ball by default)
    unsafe_center: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    unsafe_radius: float = 0.5
    unsafe_constraints: list[Callable] = field(default_factory=list)

    # Distance function
    distance_type: str = 'squared_euclidean'

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute distance between trajectory point x and unsafe point y."""
        diff = x - y
        if self.distance_type == 'squared_euclidean':
            return np.dot(diff, diff)
        elif self.distance_type == 'euclidean':
            return np.sqrt(np.dot(diff, diff))
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

    def in_initial_set(self, x: np.ndarray) -> bool:
        """Check if point x is in the initial set."""
        # Check ball constraint
        diff = x - self.initial_center
        if np.dot(diff, diff) > self.initial_radius**2:
            return False
        # Check additional constraints
        for g in self.initial_constraints:
            if g(x) < 0:
                return False
        return True

    def in_unsafe_set(self, y: np.ndarray) -> bool:
        """Check if point y is in the unsafe set."""
        # Check ball constraint
        diff = y - self.unsafe_center
        if np.dot(diff, diff) > self.unsafe_radius**2:
            return False
        # Check additional constraints
        for h in self.unsafe_constraints:
            if h(y) < 0:
                return False
        return True


def create_flow_system(
    initial_center: np.ndarray = None,
    initial_radius: float = 0.4,
    unsafe_center: np.ndarray = None,
    unsafe_radius: float = 0.5,
    time_horizon: float = 5.0,
) -> UnsafeSupport:
    """Create the Flow system from Miller & Sznaier's paper (Eq. 5, Fig. 5).

    Dynamics (Van der Pol-like):
        ẋ₁ = x₂
        ẋ₂ = -x₁ + (1/3)x₁³ - x₂

    Default initial set: Circle centered at (1.5, 0) with radius 0.4
    Default unsafe set: Half-disk at (0, -0.7) with radius 0.5
        The half-plane constraint is: cos(5π/4)*x1 + sin(5π/4)*x2 <= 0
        This keeps the lower-left half of the ball.

    Args:
        initial_center: Center of initial ball. Default: [1.5, 0]
        initial_radius: Radius of initial ball.
        unsafe_center: Center of unsafe ball. Default: [0, -0.7]
        unsafe_radius: Radius of unsafe ball.
        time_horizon: Time horizon T.

    Returns:
        UnsafeSupport problem specification.
    """
    if initial_center is None:
        initial_center = np.array([1.5, 0.0])
    if unsafe_center is None:
        unsafe_center = np.array([0.0, -0.7])

    def flow_dynamics(t: float, x: np.ndarray) -> np.ndarray:
        """Van der Pol-like flow dynamics."""
        return np.array([
            x[1],
            -x[0] + (1/3) * x[0]**3 - x[1]
        ])

    # Half-plane constraint for unsafe set (half-disk)
    # From MATLAB: c2f = w_c(1)*(y(1) - Cu(1)) + w_c(2) * (y(2) - Cu(2)) >= 0
    # where w_c = [cos(theta_c); sin(theta_c)] and theta_c = 5*pi/4
    #
    # This means: cos(5π/4)*(y1-c1) + sin(5π/4)*(y2-c2) >= 0
    # = -0.707*(y1-c1) - 0.707*(y2-c2) >= 0
    # = -(y1-c1) - (y2-c2) >= 0
    # = y1 + y2 <= c1 + c2 = 0 + (-0.7) = -0.7
    # This selects the lower-left region (below the line y1 + y2 = -0.7)
    half_space_angle = 5 * np.pi / 4  # 225°
    cos_theta = np.cos(half_space_angle)
    sin_theta = np.sin(half_space_angle)

    unsafe_constraints = []

    def half_plane_constraint(y):
        """Returns >= 0 if point is in the half-disk (satisfies half-plane constraint).

        The half-plane constraint from MATLAB is:
            cos(5π/4)*(y1-c1) + sin(5π/4)*(y2-c2) >= 0
        For a point to be IN the unsafe set, this must be satisfied.
        Returns >= 0 if IN the unsafe half-disk, < 0 if OUT.
        """
        dy = y - unsafe_center
        val = cos_theta * dy[0] + sin_theta * dy[1]
        return val

    unsafe_constraints.append(half_plane_constraint)

    return UnsafeSupport(
        n_vars=2,
        time_horizon=time_horizon,
        dynamics=flow_dynamics,
        initial_center=initial_center,
        initial_radius=initial_radius,
        unsafe_center=unsafe_center,
        unsafe_radius=unsafe_radius,
        unsafe_constraints=unsafe_constraints,
        distance_type='squared_euclidean',
    )


def create_moon_system(
    initial_center: np.ndarray = None,
    initial_radius: float = 0.4,
    time_horizon: float = 5.0,
) -> UnsafeSupport:
    """Create the Moon unsafe set variant from Miller & Sznaier's MATLAB code.

    Same dynamics as Flow, but with a moon-shaped unsafe set.

    From MATLAB flow_dist_moon.m:
    - Moon is constructed from two circles with h_in=0.4, h_out=1.0
    - Transformed: rotated by -π/10, scaled by 0.8, centered at [0.4, -0.4]
    - Outer circle: center=[0.4, -0.4], radius=0.8
    - Inner circle (excluded): center≈[0.66, 0.40], radius=1.16

    The moon shape is: inside outer circle AND outside inner circle.

    Args:
        initial_center: Center of initial ball. Default: [1.5, 0]
        initial_radius: Radius of initial ball.
        time_horizon: Time horizon T.

    Returns:
        UnsafeSupport problem specification.
    """
    if initial_center is None:
        initial_center = np.array([1.5, 0.0])

    def flow_dynamics(t: float, x: np.ndarray) -> np.ndarray:
        return np.array([
            x[1],
            -x[0] + (1/3) * x[0]**3 - x[1]
        ])

    # Moon parameters from MATLAB flow_dist_moon.m
    h_in = 0.4
    h_out = 1.0
    moon_center = np.array([0.4, -0.4])
    moon_theta = -np.pi / 10
    moon_scale = 0.8

    # Rotation matrix
    moon_rot = np.array([
        [np.cos(moon_theta), -np.sin(moon_theta)],
        [np.sin(moon_theta), np.cos(moon_theta)]
    ])

    # Inner and outer circle parameters (before transformation)
    c_in = np.array([0.0, 0.5 * (1.0/h_in - h_in)])
    r_in = 0.5 * (1.0/h_in + h_in)

    c_out = np.array([0.0, 0.5 * (1.0/h_out - h_out)])
    r_out = 0.5 * (1.0/h_out + h_out)

    # Apply transformation: rotate, scale, translate
    inner_center = moon_rot @ c_in * moon_scale + moon_center
    outer_center = moon_rot @ c_out * moon_scale + moon_center

    inner_radius = moon_scale * r_in
    outer_radius = moon_scale * r_out

    # Moon constraint: in outer ball AND outside inner ball
    # From MATLAB:
    #   con_inner = |y - c_in_scale|^2 - r_in_scale^2 >= 0  (outside inner)
    #   con_outer = -|y - c_out_scale|^2 + r_out_scale^2 >= 0  (inside outer)
    def moon_constraint(y):
        """Returns >= 0 if outside inner circle (required to be in moon)."""
        diff = y - inner_center
        return np.dot(diff, diff) - inner_radius**2

    return UnsafeSupport(
        n_vars=2,
        time_horizon=time_horizon,
        dynamics=flow_dynamics,
        initial_center=initial_center,
        initial_radius=initial_radius,
        unsafe_center=outer_center,
        unsafe_radius=outer_radius,
        unsafe_constraints=[moon_constraint],
        distance_type='squared_euclidean',
    )


def create_twist_system(
    initial_center: np.ndarray = None,
    initial_radius: float = 0.5,
    unsafe_center: np.ndarray = None,
    unsafe_radius: float = 0.5,
    time_horizon: float = 5.0,
    half_space: bool = True,
) -> UnsafeSupport:
    """Create the 3D Twist system from Miller & Sznaier's paper (Eq. 34-35).

    Dynamics:
        ẋ = Ax + B(x⊗x)

    Where A and B are specific matrices creating a twisted trajectory
    in 3D space.

    Default initial set: Sphere centered at origin with radius 0.5
    Default unsafe set: Half-sphere at (1, 0, 0) with radius 0.5

    Args:
        initial_center: Center of initial ball. Default: [0, 0, 0]
        initial_radius: Radius of initial ball.
        unsafe_center: Center of unsafe ball. Default: [1, 0, 0]
        unsafe_radius: Radius of unsafe ball.
        time_horizon: Time horizon T.
        half_space: If True, restrict unsafe to x1 >= 0.

    Returns:
        UnsafeSupport problem specification.
    """
    if initial_center is None:
        initial_center = np.array([0.0, 0.0, 0.0])
    if unsafe_center is None:
        unsafe_center = np.array([1.0, 0.0, 0.0])

    # System matrices from Eq. 34-35
    A = np.array([
        [0, 0, 0],
        [0, -1, 1],
        [0, -1, -1]
    ])

    # B is a 3x9 matrix for the Kronecker product x ⊗ x
    # For simplicity, we use a simplified version
    B = np.zeros((3, 9))
    B[0, 0] = 1.0  # x1^2 term
    B[0, 4] = 0.5  # x2^2 term
    B[0, 8] = 0.5  # x3^2 term

    def twist_dynamics(t: float, x: np.ndarray) -> np.ndarray:
        """Twist dynamics with quadratic terms."""
        # Kronecker product x ⊗ x
        x_kron = np.outer(x, x).flatten()
        return A @ x + B @ x_kron

    # Half-space constraint for unsafe set
    unsafe_constraints = []
    if half_space:
        def half_space_constraint(y):
            return y[0]  # x1 >= 0

        unsafe_constraints.append(half_space_constraint)

    return UnsafeSupport(
        n_vars=3,
        time_horizon=time_horizon,
        dynamics=twist_dynamics,
        initial_center=initial_center,
        initial_radius=initial_radius,
        unsafe_center=unsafe_center,
        unsafe_radius=unsafe_radius,
        unsafe_constraints=unsafe_constraints,
        distance_type='squared_euclidean',
    )
