# Experiment Scripts

This folder contains scripts for running the benchmark and comparison experiments that evaluate the spline-verify algorithm.

## Overview

Two main experimental pipelines are provided:

1. **`run_benchmarks.py`** - Comprehensive evaluation of spline-verify across multiple dimensions
2. **`run_comparison.py`** - Side-by-side comparison with Miller-Sznaier's SDP-based method

---

## Benchmark Suite (`run_benchmarks.py`)

### Purpose

Systematically evaluate spline-verify's accuracy, scalability, and robustness across different problem configurations. This provides the primary metrics for measuring algorithm success.

### Experiments

#### 1. Ground Truth Validation

**Goal**: Verify correctness against problems with known analytical solutions.

**Method**: Run verification on linear systems where the minimum distance to unsafe sets can be computed analytically.

**Metrics**:
- Accuracy (% of correct SAFE/UNSAFE classifications)
- Agreement between computed and analytical minimum distances

**Systems tested**:
- Stable linear ODEs with exponential decay
- Harmonic oscillators with periodic orbits
- Systems with known eigenvalue structure

#### 2. Scalability Analysis

**Goal**: Measure how runtime and accuracy scale with problem complexity.

**Parameters varied**:
| Parameter | Range | Default |
|-----------|-------|---------|
| State dimension | 2D - 6D | 2D |
| Sample count | 50 - 400 | 200 |

**Metrics**:
- Runtime (seconds)
- Verification status consistency
- Minimum objective value convergence

#### 3. Sensitivity Analysis

**Goal**: Understand how algorithm parameters affect results.

**Sampling strategy comparison**:
- Uniform random sampling
- Latin Hypercube Sampling (LHS)
- Sobol sequences (quasi-random)
- Halton sequences (quasi-random)

**Spline kernel comparison**:
- Thin-plate spline (TPS)
- Gaussian RBF
- Multiquadric RBF
- Inverse multiquadric RBF

**Metrics**:
- Minimum objective value (lower = more conservative)
- Error bound magnitude
- Runtime

#### 4. Ablation Studies

**Goal**: Isolate each pipeline component's contribution to overall performance.

**Components ablated**:

| Component | Variants Tested |
|-----------|-----------------|
| Integrator | Euler, RK4, RK45 (adaptive), Adams-Bashforth |
| Sampling | Uniform, Latin Hypercube, Sobol, Halton |
| Spline method | RBF-TPS, RBF-Gaussian, B-spline (1D/2D only) |
| Sample count | 50, 100, 200, 400 |
| Optimization | Multi-start L-BFGS-B, Differential Evolution |

**Test problems**:
- `diagonal_4d`: 4D linear system with diagonal decay
- `van_der_pol_2d`: 2D Van der Pol oscillator
- `lorenz_3d`: 3D Lorenz-like chaotic system

**Metrics**:
- Baseline vs. ablated performance comparison
- Impact score (% change from baseline)

### Usage

```bash
# Quick mode (~2 minutes)
python scripts/run_benchmarks.py --quick

# Full mode (~10 minutes)
python scripts/run_benchmarks.py --full

# Save results and figures
python scripts/run_benchmarks.py --full --save --outdir ./results/benchmarks
```

### Output

When `--save` is specified:
- `benchmark_results.json` - All numerical results
- `scalability.png` - Dimension and sample count scaling plots
- `ablation.png` - Component ablation comparison charts

---

## Method Comparison (`run_comparison.py`)

### Purpose

Compare spline-verify against Miller & Sznaier's convex optimization approach to validate our method and understand the trade-offs.

### Background

**Miller & Sznaier (IEEE TAC 2023)** solve the same safety verification problem using:
- Occupation measures over trajectories
- Monge-Kantorovich optimal transport relaxation
- Semidefinite Programming (SDP) hierarchy

**Key difference**:
- **Spline-verify**: Sampling + spline approximation → **upper bounds** on minimum distance
- **Miller-Sznaier**: SDP relaxation → **lower bounds** on minimum distance

When both bounds bracket the true minimum, we have certified bounds on the safety margin.

### Test Problems

| Problem | Dimension | Description | Time Horizon |
|---------|-----------|-------------|--------------|
| Flow 2D | 2D | Van der Pol-like dynamics from their paper | 2.0s |
| Twist 3D | 3D | 3D nonlinear system from their paper | 1.0s |
| Harmonic | 2D | Simple harmonic oscillator | 3.0s |
| Linear stable | 2D | Exponentially stable linear system | 3.0s |

### Metrics

For each problem, we report:

| Metric | Description |
|--------|-------------|
| Spline min | Minimum of fitted spline (upper bound on true min) |
| Error bound ε | Spline approximation error bound |
| Spline time | Total spline-verify runtime |
| SDP lower | Lower bound from SDP relaxation |
| SDP upper | Upper bound from trajectory sampling |
| SDP time | SDP solve time |
| Gap | Distance between upper and lower bounds |

### Interpretation

**Success criteria**:
1. **Correct classification**: Both methods agree on SAFE/UNSAFE
2. **Tight bounds**: Gap between upper and lower bounds is small
3. **Efficiency**: Spline-verify is faster for large problems

**Expected results**:
- Spline-verify provides tighter upper bounds (sampling-based)
- Miller-Sznaier provides certified lower bounds (SDP-based)
- Spline-verify scales better to higher dimensions
- Miller-Sznaier has exponential cost in dimension (SDP size)

### Usage

```bash
# Quick comparison (requires cvxpy and scs)
pip install cvxpy scs
python scripts/run_comparison.py --quick

# Full comparison with all problems
python scripts/run_comparison.py --full

# Save results
python scripts/run_comparison.py --full --save --outdir ./results/comparison
```

### Output

When `--save` is specified:

**Numerical Results:**
- `comparison_results.json` - All numerical results

**Visualization Figures:**

| Figure | Description |
|--------|-------------|
| `bounds_comparison.png` | Bar chart comparing spline-verify min vs M-S lower bound |
| `gap_analysis.png` | Gap analysis and runtime speedup |
| `flow_dist_*_distance_function.png` | 3D surface of spline approximation F̃_T(x₀) |
| `flow_dist_*_distance_1d_slice.png` | 1D slice showing spline vs M-S bound |
| `flow_dist_*_sampling_comparison.png` | M-S ball vs spline-verify box sampling |
| `moon_distance_function.png` | Moon system 3D distance surface |
| `moon_distance_1d_slice.png` | Moon system 1D slice |
| `moon_sampling_comparison.png` | Moon system sampling comparison |

**3D Distance Function Visualization:**
- Green surface: SAFE (all F > 0)
- Red/green binary: UNSAFE regions visible
- Blue horizontal plane: M-S lower bound
- Black star: Minimizer location

**1D Slice Visualization:**
- Blue curve: Spline approximation along x₁ axis
- Red dashed line: M-S lower bound
- Green dots: Sample points near slice
- Red star: Slice minimum

### Hyperparameters

The comparison script uses these defaults for visualization:
- **n_samples=500**: Provides reliable accuracy
- **smoothing=0.01**: Prevents RBF oscillation artifacts
- **n_grid=50**: Grid resolution for surface plots

---

## Running All Experiments

For a complete evaluation:

```bash
# Create output directories
mkdir -p results/benchmarks results/comparison

# Run full benchmark suite
python scripts/run_benchmarks.py --full --save --outdir ./results/benchmarks

# Run method comparison (requires cvxpy)
python scripts/run_comparison.py --full --save --outdir ./results/comparison
```

Total runtime: ~15-20 minutes

---

## Key Success Metrics

### Primary Metrics (for reporting)

1. **Ground Truth Accuracy**: % of correct classifications on linear systems
   - Target: >95%

2. **Scalability Coefficient**: Runtime growth rate with dimension
   - Target: Polynomial (not exponential)

3. **Bound Tightness**: Gap between spline minimum and error bound
   - Target: Error bound < 10% of minimum for safe cases

4. **Comparison Agreement**: % agreement with Miller-Sznaier on SAFE/UNSAFE
   - Target: 100% (methods should agree on clear-cut cases)

### Secondary Metrics

- Runtime vs. sample count (should be ~linear)
- Sensitivity to sampling strategy (should be <5% variance)
- Integrator impact on accuracy (RK45 should match RK4)

---

## Troubleshooting

### Import Errors

Ensure the package is installed in development mode:
```bash
pip install -e .
```

### Missing SDP Solver

For comparison experiments:
```bash
pip install cvxpy scs
# Or for better performance:
pip install cvxpy mosek
```

### Integration Failures

Some stiff systems (like Van der Pol at long time horizons) may fail to integrate. The comparison script handles these gracefully and skips problematic configurations.

### Memory Issues

For high-dimensional problems (>6D), reduce sample count:
```bash
python scripts/run_benchmarks.py --quick  # Uses fewer samples
```
