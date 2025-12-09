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
├── examples/
│   ├── EXAMPLES.md              # Detailed examples documentation
│   ├── linear_system.py         # Linear ODE examples (SAFE/UNSAFE)
│   ├── harmonic_oscillator.py   # Periodic orbit examples
│   ├── bouncing_ball.py         # Bouncing ball (event-driven switching)
│   ├── relay_feedback.py        # Relay feedback (sliding mode)
│   ├── thermostat.py            # Thermostat (hysteresis switching)
│   ├── vehicle_braking.py       # Emergency braking (case study)
│   ├── quadrotor.py             # Altitude control (case study)
│   ├── glucose_insulin.py       # Blood glucose regulation (case study)
│   └── demonstration.py         # Full pipeline visualization
└── tests/
    ├── test_integrators.py      # Integrator tests
    ├── test_sets.py             # Geometry tests
    ├── test_splines.py          # Spline fitting tests
    ├── test_verification.py     # ODE verification tests
    ├── test_switching.py        # Switching system tests
    └── test_benchmarks.py       # Benchmark infrastructure tests
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
# ODE examples
python examples/linear_system.py --save --outdir ./examples/figs
python examples/harmonic_oscillator.py --save --outdir ./examples/figs

# Switching system examples
python examples/bouncing_ball.py --save --outdir ./examples/figs
python examples/relay_feedback.py --save --outdir ./examples/figs
python examples/thermostat.py --save --outdir ./examples/figs

# Case studies (realistic applications)
python examples/vehicle_braking.py --save --outdir ./examples/figs/case_study
python examples/quadrotor.py --save --outdir ./examples/figs/case_study
python examples/glucose_insulin.py --save --outdir ./examples/figs/case_study

# Demonstration tutorial (generates all pipeline figures)
python examples/demonstration.py --save --outdir ./examples/figs/demo
```

See [examples/EXAMPLES.md](examples/EXAMPLES.md) for detailed documentation of all examples.

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

## Case Studies

Three realistic case studies demonstrate spline-verify on practical safety-critical systems:

### Vehicle Braking (Collision Avoidance)

```python
# 3D state: position, velocity, acceleration
# Verifies emergency braking can prevent collision
python examples/vehicle_braking.py --save --outdir ./examples/figs/case_study
```

- **SAFE case**: Vehicle at 80-90m, speed 25-28 m/s, T=6s → Successfully brakes
- **UNSAFE case**: Brake failure at close range → Collision unavoidable

### Quadrotor Altitude Control

```python
# 2D state: altitude, vertical velocity
# Verifies altitude control prevents ground collision
python examples/quadrotor.py --save --outdir ./examples/figs/case_study
```

- **SAFE case**: PD-controlled hover maintains altitude above ground
- **UNSAFE case**: Motor failure leads to free-fall ground impact

### Glucose-Insulin Regulation

```python
# 2D state: blood glucose, plasma insulin
# Verifies glucose stays in safe physiological range
python examples/glucose_insulin.py --save --outdir ./examples/figs/case_study
```

- **SAFE case**: Fasting glucose regulation prevents hypoglycemia
- **UNSAFE case**: Large meal without insulin causes severe hyperglycemia

## Benchmarking

The benchmark infrastructure validates verification accuracy and analyzes performance:

```python
from spline_verify.benchmarks import BenchmarkRunner, GroundTruthValidator

# Run ground truth validation against analytical solutions
validator = GroundTruthValidator()
results = validator.validate_linear_systems()

# Analyze scalability across dimensions and sample counts
from spline_verify.benchmarks import ScalabilityAnalyzer
analyzer = ScalabilityAnalyzer()
results = analyzer.analyze_dimension_scaling(max_dim=8)
```

Run all benchmarks:
```bash
pytest tests/test_benchmarks.py -v
```

## References

- Schumaker, "Spline Functions: Basic Theory" (Cambridge, 2007)
- Filippov, "Differential Equations with Discontinuous Righthand Sides" (Springer, 1988)
- de Boor, "A Practical Guide to Splines" (Springer, 2001)
