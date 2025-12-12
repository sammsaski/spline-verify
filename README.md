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

# Optional: Install SDP solvers for Miller-Sznaier comparison
pip install -e ".[sdp]"

# Or install everything
pip install -e ".[all]"
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
│   │   ├── bspline.py           # Tensor-product B-splines for 1D/2D
│   │   ├── multivariate.py      # Scattered data spline fitting (RBF)
│   │   ├── optimization.py      # Spline minimization
│   │   └── piecewise.py         # PiecewiseSplineApproximation for switching
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── error_bounds.py      # Error budget analysis
│   │   ├── objective.py         # Objective function F_T computation
│   │   ├── switching.py         # SwitchingVerifier, region classifier
│   │   └── verifier.py          # Main SafetyVerifier class
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   ├── runner.py            # BenchmarkRunner infrastructure
│   │   ├── ground_truth.py      # Ground truth validation
│   │   ├── scalability.py       # Dimension/sample scaling analysis
│   │   ├── sensitivity.py       # Parameter sensitivity analysis
│   │   └── ablation.py          # Ablation study framework
│   └── utils/
│       ├── __init__.py
│       └── visualization.py     # Plotting utilities
├── src/miller_sznaier/          # Miller & Sznaier SDP-based comparison
│   ├── __init__.py
│   ├── problem.py               # UnsafeSupport problem definition
│   ├── distance_estimator.py    # SDP-based distance estimation
│   └── examples/
│       ├── flow_system.py       # Flow system example
│       └── comparison.py        # Method comparison framework
├── scripts/
│   ├── README.md                # Experiment documentation
│   ├── run_benchmarks.py        # Full benchmark suite
│   └── run_comparison.py        # Spline-verify vs Miller-Sznaier comparison
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
- **GriddedBSpline**: Tensor-product B-splines for 1D/2D data
- **SplineApproximation**: 1D B-spline fitting
- **minimize_spline**: Multi-start L-BFGS-B optimization

The verifier supports multiple spline methods:
```python
# RBF-based (default, works for any dimension)
verifier = SafetyVerifier(spline_method='rbf')

# B-spline (1D/2D only, better for gridded data)
verifier = SafetyVerifier(spline_method='bspline', n_grid_points=50)

# Auto-select (B-spline for 1D/2D, RBF for higher dimensions)
verifier = SafetyVerifier(spline_method='auto')
```

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

### Ablation Studies

Systematic ablation studies isolate each pipeline component's contribution:

```python
from spline_verify.benchmarks.ablation import AblationStudy

study = AblationStudy()

# Compare integrators: Euler vs RK4 vs RK45
integrator_results = study.run_integrator_ablation()

# Compare sampling: uniform vs Latin hypercube vs Sobol
sampling_results = study.run_sampling_ablation()

# Compare spline methods: RBF vs B-spline (1D/2D only)
spline_results = study.run_spline_ablation()

# Study convergence with sample count
sample_results = study.run_sample_count_ablation()

# Run all ablation studies (~10 min runtime)
all_results = study.run_full_ablation()
```

Run all benchmarks:
```bash
pytest tests/test_benchmarks.py -v
```

## Miller & Sznaier Comparison

For comparison, we implement the SDP-based distance estimation from Miller & Sznaier's
"Bounding the Distance to Unsafe Sets with Convex Optimization" (IEEE TAC 2023).

**Note**: Requires optional SDP dependencies: `pip install cvxpy scs`

```python
from miller_sznaier import DistanceEstimator
from miller_sznaier.problem import create_flow_system

# Create the Flow system example from the paper
problem = create_flow_system(time_horizon=5.0)

# Estimate distance bounds via SDP
estimator = DistanceEstimator(order=4)
result = estimator.estimate(problem, compute_upper_bound=True)

print(f"Lower bound (certified): {result.lower_bound:.6f}")
print(f"Upper bound (sampled):   {result.upper_bound:.6f}")
```

### Side-by-Side Comparison

```python
from miller_sznaier.examples.comparison import compare_methods, run_comparison_suite

# Compare spline-verify vs Miller-Sznaier on the same problem
result = compare_methods(problem, problem_name="flow_2d")
print(result.summary())

# Run comparison on multiple standard problems
results = run_comparison_suite(verbose=True)
```

**Key differences:**
- **Spline-Verify**: Sample-based spline approximation → upper bounds
- **Miller-Sznaier**: Occupation measure SDP relaxation → lower bounds
- The gap between bounds indicates estimation uncertainty

### Comparison Visualizations

Generate comprehensive comparison figures:

```bash
python scripts/run_comparison.py --save --outdir ./results/comparison
```

This generates:
- **3D Distance Function Plots** (`*_distance_function.png`): Spline approximation F̃_T(x₀) as a surface over initial states
  - Green surface: SAFE (all F > 0)
  - Red/green binary coloring: UNSAFE regions visible
  - Blue horizontal plane: M-S lower bound
- **1D Slice Plots** (`*_distance_1d_slice.png`): Cross-section at x₂=center showing spline curve vs M-S bound
- **Sampling Comparison** (`*_sampling_comparison.png`): Initial set representations (M-S ball vs spline-verify box)

Examples include Flow system variants (θ=5π/4, 3π/2, 7π/4) and Moon system from the M-S paper.

## Running Tests

```bash
# Activate conda environment
conda activate spline-verify

# Run all tests (113 tests)
pytest tests/ -v

# Run specific test files
pytest tests/test_verification.py -v
pytest tests/test_switching.py -v
pytest tests/test_benchmarks.py -v
```

## References

- Schumaker, "Spline Functions: Basic Theory" (Cambridge, 2007)
- Filippov, "Differential Equations with Discontinuous Righthand Sides" (Springer, 1988)
- de Boor, "A Practical Guide to Splines" (Springer, 2001)
- Miller & Sznaier, "Bounding the Distance to Unsafe Sets with Convex Optimization" (IEEE TAC, 2023)
