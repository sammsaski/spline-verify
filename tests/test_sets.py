"""Tests for set representations."""

import numpy as np
import pytest

from spline_verify.geometry.sets import (
    HyperRectangle, Ball, HalfSpace, ConvexPolytope, LevelSet, UnionSet
)


class TestHyperRectangle:
    """Tests for HyperRectangle."""

    def test_basic_creation(self):
        """Test rectangle creation."""
        rect = HyperRectangle(
            lower=np.array([0, 0]),
            upper=np.array([1, 1])
        )

        assert rect.n_dims == 2
        assert rect.volume == 1.0
        np.testing.assert_array_equal(rect.center, [0.5, 0.5])

    def test_contains(self):
        """Test point containment."""
        rect = HyperRectangle(np.array([0, 0]), np.array([1, 1]))

        assert rect.contains(np.array([0.5, 0.5]))
        assert rect.contains(np.array([0, 0]))  # Boundary
        assert rect.contains(np.array([1, 1]))  # Boundary
        assert not rect.contains(np.array([1.5, 0.5]))
        assert not rect.contains(np.array([-0.1, 0.5]))

    def test_distance(self):
        """Test distance computation."""
        rect = HyperRectangle(np.array([0, 0]), np.array([1, 1]))

        # Inside: distance = 0
        assert rect.distance(np.array([0.5, 0.5])) == 0

        # Outside: distance to nearest edge
        assert abs(rect.distance(np.array([2, 0.5])) - 1.0) < 1e-10
        assert abs(rect.distance(np.array([2, 2])) - np.sqrt(2)) < 1e-10

    def test_sampling(self):
        """Test uniform sampling."""
        rect = HyperRectangle(np.array([0, 0]), np.array([1, 1]))

        samples = rect.sample(100, seed=42)

        assert samples.shape == (100, 2)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_from_center_width(self):
        """Test creation from center and width."""
        rect = HyperRectangle.from_center_width(
            center=np.array([0.5, 0.5]),
            widths=np.array([1.0, 2.0])
        )

        np.testing.assert_array_equal(rect.lower, [0, -0.5])
        np.testing.assert_array_equal(rect.upper, [1, 1.5])


class TestBall:
    """Tests for Ball."""

    def test_basic_creation(self):
        """Test ball creation."""
        ball = Ball(center=np.array([0, 0]), radius=1.0)

        assert ball.n_dims == 2
        assert ball.radius == 1.0

    def test_contains(self):
        """Test point containment."""
        ball = Ball(np.array([0, 0]), 1.0)

        assert ball.contains(np.array([0, 0]))
        assert ball.contains(np.array([0.5, 0.5]))
        assert ball.contains(np.array([1, 0]))  # On boundary
        assert not ball.contains(np.array([1.1, 0]))

    def test_distance(self):
        """Test distance computation."""
        ball = Ball(np.array([0, 0]), 1.0)

        assert ball.distance(np.array([0, 0])) == 0
        assert ball.distance(np.array([0.5, 0])) == 0  # Inside
        assert abs(ball.distance(np.array([2, 0])) - 1.0) < 1e-10

    def test_sampling(self):
        """Test sampling."""
        ball = Ball(np.array([0, 0]), 1.0)

        samples = ball.sample(100, seed=42)

        assert samples.shape == (100, 2)
        distances = np.linalg.norm(samples, axis=1)
        assert np.all(distances <= 1.0)


class TestHalfSpace:
    """Tests for HalfSpace."""

    def test_basic(self):
        """Test half-space creation."""
        # x <= 1
        hs = HalfSpace(a=np.array([1, 0]), b=1.0)

        assert hs.n_dims == 2
        assert hs.contains(np.array([0, 0]))
        assert hs.contains(np.array([1, 5]))  # On boundary
        assert not hs.contains(np.array([2, 0]))

    def test_distance(self):
        """Test distance computation."""
        hs = HalfSpace(a=np.array([1, 0]), b=1.0)

        assert hs.distance(np.array([0, 0])) == 0  # Inside
        assert abs(hs.distance(np.array([2, 0])) - 1.0) < 1e-10


class TestConvexPolytope:
    """Tests for ConvexPolytope."""

    def test_box_as_polytope(self):
        """Test box represented as polytope."""
        # Unit square: 0 <= x,y <= 1
        # x >= 0: -x <= 0
        # x <= 1: x <= 1
        # y >= 0: -y <= 0
        # y <= 1: y <= 1
        A = np.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]
        ])
        b = np.array([0, 1, 0, 1])

        poly = ConvexPolytope(A, b)

        assert poly.n_dims == 2
        assert poly.contains(np.array([0.5, 0.5]))
        assert not poly.contains(np.array([1.5, 0.5]))

    def test_distance(self):
        """Test distance to polytope."""
        A = np.array([
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]
        ])
        b = np.array([0, 1, 0, 1])

        poly = ConvexPolytope(A, b)

        assert poly.distance(np.array([0.5, 0.5])) == 0
        assert abs(poly.distance(np.array([2, 0.5])) - 1.0) < 0.1


class TestLevelSet:
    """Tests for LevelSet."""

    def test_circle_level_set(self):
        """Test circle as level set."""
        # Circle: x^2 + y^2 - 1 <= 0
        def g(x):
            return x[0]**2 + x[1]**2 - 1

        ls = LevelSet(g=g, _n_dims=2)

        assert ls.contains(np.array([0, 0]))
        assert ls.contains(np.array([1, 0]))  # On boundary
        assert not ls.contains(np.array([1, 1]))

    def test_sampling_with_bounds(self):
        """Test sampling from level set with bounds."""
        def g(x):
            return x[0]**2 + x[1]**2 - 1

        ls = LevelSet(
            g=g,
            _n_dims=2,
            bounds=HyperRectangle(np.array([-1, -1]), np.array([1, 1]))
        )

        samples = ls.sample(50, seed=42)
        assert samples.shape == (50, 2)

        # All samples should be in the circle
        for x in samples:
            assert ls.contains(x)


class TestUnionSet:
    """Tests for UnionSet."""

    def test_two_balls(self):
        """Test union of two balls."""
        ball1 = Ball(np.array([0, 0]), 1.0)
        ball2 = Ball(np.array([3, 0]), 1.0)

        union = UnionSet([ball1, ball2])

        assert union.contains(np.array([0, 0]))  # In ball1
        assert union.contains(np.array([3, 0]))  # In ball2
        assert not union.contains(np.array([1.5, 0]))  # Between them

    def test_distance(self):
        """Test distance to union."""
        ball1 = Ball(np.array([0, 0]), 1.0)
        ball2 = Ball(np.array([3, 0]), 1.0)

        union = UnionSet([ball1, ball2])

        # Point between balls
        dist = union.distance(np.array([1.5, 0]))
        assert abs(dist - 0.5) < 1e-10  # Distance to nearest ball
