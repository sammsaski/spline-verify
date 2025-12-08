"""Set representations for initial and unsafe sets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize


class Set(ABC):
    """Abstract base class for set representations.

    A Set supports:
    - Membership testing (contains)
    - Distance computation (distance)
    - Sampling (sample)
    """

    @property
    @abstractmethod
    def n_dims(self) -> int:
        """Dimension of the ambient space."""
        ...

    @abstractmethod
    def contains(self, x: np.ndarray) -> bool:
        """Check if point x is in the set.

        Args:
            x: Point to test, shape (n_dims,).

        Returns:
            True if x is in the set.
        """
        ...

    @abstractmethod
    def distance(self, x: np.ndarray) -> float:
        """Compute distance from point x to the set.

        Args:
            x: Point, shape (n_dims,).

        Returns:
            Distance from x to the closest point in the set.
            Returns 0 if x is in the set.
        """
        ...

    @abstractmethod
    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Sample n points uniformly from the set.

        Args:
            n: Number of points to sample.
            seed: Random seed for reproducibility.

        Returns:
            Array of sampled points, shape (n, n_dims).
        """
        ...

    def distance_function(self) -> Callable[[np.ndarray], float]:
        """Return a callable that computes distance to this set."""
        return self.distance


@dataclass
class HyperRectangle(Set):
    """Axis-aligned hyperrectangle (box) defined by lower and upper bounds.

    The set is {x : lower[i] <= x[i] <= upper[i] for all i}.
    """
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)

        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"lower and upper must have same shape: "
                f"{self.lower.shape} vs {self.upper.shape}"
            )

        if np.any(self.lower > self.upper):
            raise ValueError("lower bounds must be <= upper bounds")

    @property
    def n_dims(self) -> int:
        return len(self.lower)

    @property
    def center(self) -> np.ndarray:
        """Center of the rectangle."""
        return (self.lower + self.upper) / 2

    @property
    def widths(self) -> np.ndarray:
        """Width in each dimension."""
        return self.upper - self.lower

    @property
    def volume(self) -> float:
        """Volume of the rectangle."""
        return float(np.prod(self.widths))

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x)
        return bool(np.all(x >= self.lower) and np.all(x <= self.upper))

    def distance(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        # Distance to box is distance to nearest point on box surface
        clamped = np.clip(x, self.lower, self.upper)
        return float(np.linalg.norm(x - clamped))

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(self.lower, self.upper, size=(n, self.n_dims))

    @classmethod
    def from_center_width(
        cls,
        center: np.ndarray,
        widths: np.ndarray | float
    ) -> HyperRectangle:
        """Create rectangle from center and width."""
        center = np.asarray(center)
        if isinstance(widths, (int, float)):
            widths = np.full_like(center, widths)
        widths = np.asarray(widths)
        half_widths = widths / 2
        return cls(lower=center - half_widths, upper=center + half_widths)


@dataclass
class Ball(Set):
    """Ball (sphere) defined by center and radius.

    The set is {x : ||x - center|| <= radius}.
    """
    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        if self.radius < 0:
            raise ValueError("radius must be non-negative")

    @property
    def n_dims(self) -> int:
        return len(self.center)

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x)
        return bool(np.linalg.norm(x - self.center) <= self.radius)

    def distance(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        dist_to_center = np.linalg.norm(x - self.center)
        return float(max(0, dist_to_center - self.radius))

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Sample uniformly from the ball using rejection sampling for low dims."""
        rng = np.random.default_rng(seed)

        if self.n_dims <= 4:
            # Rejection sampling is efficient for low dimensions
            samples = []
            while len(samples) < n:
                batch_size = 2 * n
                candidates = rng.uniform(-1, 1, size=(batch_size, self.n_dims))
                norms = np.linalg.norm(candidates, axis=1)
                valid = candidates[norms <= 1]
                samples.extend(valid[:n - len(samples)])
            samples = np.array(samples[:n])
        else:
            # For higher dimensions, use the algorithm:
            # sample direction uniformly, then scale by r^(1/d) * radius
            directions = rng.normal(size=(n, self.n_dims))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            radii = rng.uniform(0, 1, size=(n, 1)) ** (1 / self.n_dims)
            samples = directions * radii

        return self.center + self.radius * samples


@dataclass
class HalfSpace(Set):
    """Half-space defined by a linear inequality: {x : a @ x <= b}.

    Attributes:
        a: Normal vector (pointing outward from the half-space).
        b: Offset constant.
    """
    a: np.ndarray
    b: float

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=float)
        # Normalize for easier distance computation
        norm = np.linalg.norm(self.a)
        if norm == 0:
            raise ValueError("Normal vector a cannot be zero")
        self.a = self.a / norm
        self.b = self.b / norm

    @property
    def n_dims(self) -> int:
        return len(self.a)

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x)
        return bool(np.dot(self.a, x) <= self.b)

    def distance(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        # Signed distance (positive if outside)
        signed_dist = np.dot(self.a, x) - self.b
        return float(max(0, signed_dist))

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        raise NotImplementedError(
            "Cannot sample from unbounded half-space. "
            "Use intersection with a bounded set."
        )


@dataclass
class ConvexPolytope(Set):
    """Convex polytope defined by linear inequalities: {x : Ax <= b}.

    Attributes:
        A: Matrix of normal vectors, shape (m, n) for m constraints in R^n.
        b: Vector of offsets, shape (m,).
    """
    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float)

        if self.A.ndim != 2:
            raise ValueError(f"A must be 2D, got {self.A.ndim}D")
        if self.b.ndim != 1:
            raise ValueError(f"b must be 1D, got {self.b.ndim}D")
        if self.A.shape[0] != len(self.b):
            raise ValueError(
                f"A and b dimension mismatch: A has {self.A.shape[0]} rows, "
                f"b has {len(self.b)} elements"
            )

    @property
    def n_dims(self) -> int:
        return self.A.shape[1]

    @property
    def n_constraints(self) -> int:
        return self.A.shape[0]

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x)
        return bool(np.all(self.A @ x <= self.b + 1e-10))

    def distance(self, x: np.ndarray) -> float:
        """Compute distance by finding closest point in polytope via QP."""
        x = np.asarray(x)

        if self.contains(x):
            return 0.0

        # Minimize ||y - x||^2 subject to Ay <= b
        # This is a QP: minimize (y-x)'(y-x) s.t. Ay <= b
        from scipy.optimize import minimize

        def objective(y):
            return np.sum((y - x) ** 2)

        def grad(y):
            return 2 * (y - x)

        constraints = {'type': 'ineq', 'fun': lambda y: self.b - self.A @ y}

        # Start from x projected onto feasible region (crude)
        x0 = x.copy()
        result = minimize(
            objective, x0, method='SLSQP',
            jac=grad, constraints=constraints
        )

        return float(np.sqrt(result.fun)) if result.success else float('inf')

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Sample using hit-and-run MCMC."""
        rng = np.random.default_rng(seed)

        # Find a feasible starting point via LP
        from scipy.optimize import linprog

        # Find interior point: minimize t s.t. Ax + t*1 <= b, t >= 0
        # Reformulate: [A, 1] @ [x; t] <= b
        c = np.zeros(self.n_dims + 1)
        c[-1] = 1  # minimize t

        A_lp = np.hstack([self.A, np.ones((self.n_constraints, 1))])
        bounds = [(None, None)] * self.n_dims + [(0, None)]

        result = linprog(c, A_ub=A_lp, b_ub=self.b, bounds=bounds, method='highs')
        if not result.success or result.x[-1] > 1e-6:
            raise RuntimeError("Could not find interior point of polytope")

        current = result.x[:-1]
        samples = [current.copy()]

        # Hit-and-run sampling
        for _ in range(n * 10):  # Oversample for mixing
            direction = rng.normal(size=self.n_dims)
            direction /= np.linalg.norm(direction)

            # Find range of valid t: A(current + t*direction) <= b
            # A @ current + t * (A @ direction) <= b
            # t * (A @ direction) <= b - A @ current
            Ad = self.A @ direction
            slack = self.b - self.A @ current

            t_min, t_max = -np.inf, np.inf
            for i in range(self.n_constraints):
                if Ad[i] > 1e-10:
                    t_max = min(t_max, slack[i] / Ad[i])
                elif Ad[i] < -1e-10:
                    t_min = max(t_min, slack[i] / Ad[i])

            if t_max > t_min:
                t = rng.uniform(t_min, t_max)
                current = current + t * direction
                samples.append(current.copy())

        # Take every k-th sample after burn-in
        samples = np.array(samples)
        burn_in = len(samples) // 4
        samples = samples[burn_in:]
        indices = np.linspace(0, len(samples) - 1, n, dtype=int)
        return samples[indices]


@dataclass
class LevelSet(Set):
    """Level set of a function: {x : g(x) <= 0}.

    Attributes:
        g: Function g(x) -> float.
        n_dims: Dimension of the ambient space.
        bounds: Optional bounding box for sampling.
    """
    g: Callable[[np.ndarray], float]
    _n_dims: int
    bounds: HyperRectangle | None = None

    @property
    def n_dims(self) -> int:
        return self._n_dims

    def contains(self, x: np.ndarray) -> bool:
        return bool(self.g(np.asarray(x)) <= 0)

    def distance(self, x: np.ndarray) -> float:
        """Approximate distance using the level set value.

        Note: This is exact only if g is a signed distance function.
        For general g, this is an approximation.
        """
        x = np.asarray(x)
        val = self.g(x)
        return float(max(0, val))

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Sample via rejection sampling within bounds."""
        if self.bounds is None:
            raise ValueError(
                "Cannot sample from LevelSet without bounds. "
                "Set bounds attribute first."
            )

        rng = np.random.default_rng(seed)
        samples = []

        while len(samples) < n:
            batch_size = 10 * n
            candidates = self.bounds.sample(batch_size)
            for x in candidates:
                if self.contains(x):
                    samples.append(x)
                    if len(samples) >= n:
                        break

        return np.array(samples[:n])


@dataclass
class UnionSet(Set):
    """Union of multiple sets: S1 ∪ S2 ∪ ... ∪ Sk.

    A point is in the union if it's in any of the component sets.
    Distance is the minimum distance to any component set.
    """
    sets: list[Set]

    def __post_init__(self) -> None:
        if len(self.sets) == 0:
            raise ValueError("UnionSet must contain at least one set")

        dims = [s.n_dims for s in self.sets]
        if len(set(dims)) > 1:
            raise ValueError(f"All sets must have same dimension, got {dims}")

    @property
    def n_dims(self) -> int:
        return self.sets[0].n_dims

    def contains(self, x: np.ndarray) -> bool:
        return any(s.contains(x) for s in self.sets)

    def distance(self, x: np.ndarray) -> float:
        return float(min(s.distance(x) for s in self.sets))

    def sample(self, n: int, seed: int | None = None) -> np.ndarray:
        """Sample by randomly choosing which set to sample from."""
        rng = np.random.default_rng(seed)

        # Distribute samples among sets (could weight by volume if known)
        samples_per_set = [n // len(self.sets)] * len(self.sets)
        for i in range(n % len(self.sets)):
            samples_per_set[i] += 1

        all_samples = []
        for s, ns in zip(self.sets, samples_per_set):
            if ns > 0:
                all_samples.append(s.sample(ns, seed=rng.integers(0, 2**31)))

        result = np.vstack(all_samples)
        rng.shuffle(result)
        return result
