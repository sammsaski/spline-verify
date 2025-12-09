# spline-verify

Safety verification for cyber-physical systems via spline approximation of objective functions.

## Overview

Instead of computing reachable sets (which is computationally expensive and often undecidable), we:

1. Sample initial conditions from the initial set
2. Simulate trajectories and compute distance to unsafe set (objective function F_T)
3. Fit a spline approximation to F_T
4. Minimize the spline to find the worst-case initial condition
5. Compare the minimum against error bounds to determine safety

## Installation

### Using conda (recommended)

```bash
# Create and activate environment
conda create -n spline-verify python=3.10
conda activate spline-verify

# Install dependencies
pip install numpy scipy matplotlib scikit-learn pytest

# Install package in development mode
pip install -e .
```

### Using pip only

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package with dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from spline_verify.dynamics import ODEDynamics
from spline_verify.geometry import HyperRectangle, Ball
from spline_verify.verification import SafetyVerifier

# Define dynamics: harmonic oscillator
dynamics = ODEDynamics.harmonic_oscillator(omega=1.0)

# Initial set: small box near (1, 0)
initial_set = HyperRectangle(
    lower=np.array([0.9, -0.1]),
    upper=np.array([1.1, 0.1])
)

# Unsafe set: ball far from orbits
unsafe_set = Ball(center=np.array([5.0, 0.0]), radius=0.5)

# Time horizon
T = 2 * np.pi

# Verify safety
verifier = SafetyVerifier(n_samples=200, seed=42)
result = verifier.verify(dynamics, initial_set, unsafe_set, T)

print(result.summary())
# Verification Result: SAFE
# Minimum F̃_T:      3.456789
# Error bound ε:    0.123456
# Safety margin:    3.333333
```

## Project Structure

```
spline-verify/
├── pyproject.toml
├── README.md                    # This file
├── CLAUDE.md                    # Development notes and progress tracking
├── src/spline_verify/
│   ├── __init__.py
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── base.py              # DynamicsModel protocol, TrajectoryBundle
│   │   ├── integrators.py       # Numerical integrators (Euler, RK4, RK45, Adams)
│   │   ├── ode.py               # ODE dynamics implementation
│   │   ├── switching.py         # SwitchingDynamics, FilippovSolver
│   │   └── trajectory.py        # Trajectory data structure
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── sampling.py          # Initial set sampling strategies
│   │   └── sets.py              # Set representations (HyperRectangle, Ball, etc.)
│   ├── splines/
│   │   ├── __init__.py
│   │   ├── approximation.py     # 1D spline approximation
│   │   ├── multivariate.py      # Scattered data spline fitting (RBF)
│   │   ├── optimization.py      # Spline minimization
│   │   └── piecewise.py         # PiecewiseSplineApproximation for switching
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── error_bounds.py      # Error budget analysis
│   │   ├── objective.py         # Objective function F_T computation
│   │   ├── switching.py         # SwitchingVerifier, region classifier
│   │   └── verifier.py          # Main SafetyVerifier class
│   └── utils/
│       ├── __init__.py
│       └── visualization.py     # Plotting utilities
├── tests/
│   ├── test_integrators.py      # Integrator tests
│   ├── test_sets.py             # Geometry tests
│   ├── test_splines.py          # Spline fitting tests
│   ├── test_verification.py     # ODE verification tests
│   └── test_switching.py        # Switching system tests
└── examples/
    ├── linear_system.py         # Linear ODE examples (SAFE/UNSAFE)
    ├── harmonic_oscillator.py   # Periodic orbit examples
    ├── bouncing_ball.py         # Bouncing ball (event-driven switching)
    ├── relay_feedback.py        # Relay feedback (sliding mode)
    └── thermostat.py            # Thermostat (hysteresis switching)
```

## Verification Results

The verifier returns one of three statuses:

- **SAFE**: Proven safe (min F_T > error_bound)
- **UNSAFE**: Found counterexample (trajectory reaches unsafe set)
- **UNKNOWN**: Cannot certify (0 < min F_T ≤ error_bound)

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_verification.py -v
```

## Running Examples

All examples support optional figure saving for headless environments:

```bash
# Linear system examples (run only)
python examples/linear_system.py

# Save figures to current directory
python examples/linear_system.py --save

# Save figures to specific directory
python examples/linear_system.py --save --outdir ./figs

# Harmonic oscillator examples
python examples/harmonic_oscillator.py --save --outdir ./figs

# Bouncing ball (switching system placeholder)
python examples/bouncing_ball.py --save --outdir ./figs
```

## Core Components

### Dynamics (`spline_verify.dynamics`)

- **ODEDynamics**: Define ODEs via vector field functions or matrices
- **Integrators**: Euler, RK4, RK45 (adaptive), Adams-Bashforth
- **Trajectory**: Stores (t, x) pairs with interpolation support

### Geometry (`spline_verify.geometry`)

- **HyperRectangle**: Axis-aligned boxes
- **Ball**: Euclidean balls (spheres)
- **ConvexPolytope**: Ax ≤ b constraints
- **HalfSpace**: Single linear constraint
- **LevelSet**: g(x) ≤ 0 constraint
- **UnionSet**: Union of multiple sets

### Splines (`spline_verify.splines`)

- **ScatteredDataSpline**: RBF-based multivariate interpolation
- **SplineApproximation**: 1D B-spline fitting
- **minimize_spline**: Multi-start L-BFGS-B optimization

### Verification (`spline_verify.verification`)

- **SafetyVerifier**: Main verification pipeline
- **ObjectiveSampler**: Sample F_T over initial set
- **ErrorBudget**: Track error sources

## Switching Systems (Phase 4)

Full support for switching/hybrid systems with Filippov handling:

```python
from spline_verify.dynamics import SwitchingDynamics, FilippovSolver
from spline_verify.verification import SwitchingVerifier

# Create switching dynamics (relay feedback: dx/dt = -sign(x))
dynamics = SwitchingDynamics.relay_feedback()

# Or bouncing ball, thermostat, etc.
# dynamics = SwitchingDynamics.bouncing_ball(gravity=9.81, restitution=0.9)
# dynamics = SwitchingDynamics.thermostat(T_low=18.0, T_high=22.0, T_ambient=10.0)

# Verify with switching-aware verifier
verifier = SwitchingVerifier(n_samples=100, seed=42)
result = verifier.verify(dynamics, initial_set, unsafe_set, T)
```

**Features:**
- `SwitchingSurface`: Explicit boundary functions with normals
- `FilippovSolver`: Handles sliding modes and set-valued dynamics
- `SwitchingRegionClassifier`: SVM-based classification of initial conditions
- `PiecewiseSplineApproximation`: Per-region spline fitting for discontinuous F_T
- Automatic crossing label extraction from trajectory bundles

## References

- Schumaker, "Spline Functions: Basic Theory" (Cambridge, 2007)
- Filippov, "Differential Equations with Discontinuous Righthand Sides" (Springer, 1988)
- de Boor, "A Practical Guide to Splines" (Springer, 2001)
