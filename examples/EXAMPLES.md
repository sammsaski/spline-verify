# Examples Documentation

This document describes each example in the `spline-verify` project, including the mathematical setup, expected results, and analysis of the generated figures.

## Table of Contents

1. [Running Examples](#running-examples)
2. [ODE Examples (Phases 1-3)](#ode-examples-phases-1-3)
   - [Linear System](#linear-system)
   - [Harmonic Oscillator](#harmonic-oscillator)
3. [Switching System Examples (Phase 4)](#switching-system-examples-phase-4)
   - [Bouncing Ball](#bouncing-ball)
   - [Relay Feedback](#relay-feedback)
   - [Thermostat](#thermostat)
4. [Case Studies (Phase 5.5)](#case-studies-phase-55)
   - [Vehicle Braking](#vehicle-braking)
   - [Quadrotor Altitude Control](#quadrotor-altitude-control)
   - [Glucose-Insulin Regulation](#glucose-insulin-regulation)
5. [Demonstration Tutorial](#demonstration-tutorial)
6. [Figure Gallery](#figure-gallery)

---

## Running Examples

All examples support command-line arguments for saving figures:

```bash
# Run example without saving figures
python examples/<example_name>.py

# Save figures to current directory
python examples/<example_name>.py --save

# Save figures to specific directory
python examples/<example_name>.py --save --outdir ./examples/figs
```

To generate all figures:

```bash
# ODE examples (Phase 1-3)
mkdir -p examples/figs
python examples/linear_system.py --save --outdir ./examples/figs
python examples/harmonic_oscillator.py --save --outdir ./examples/figs

# Switching system examples (Phase 4)
python examples/bouncing_ball.py --save --outdir ./examples/figs
python examples/relay_feedback.py --save --outdir ./examples/figs
python examples/thermostat.py --save --outdir ./examples/figs

# Case studies (Phase 5.5)
mkdir -p examples/figs/case_study
python examples/vehicle_braking.py --save --outdir ./examples/figs/case_study
python examples/quadrotor.py --save --outdir ./examples/figs/case_study
python examples/glucose_insulin.py --save --outdir ./examples/figs/case_study
```

Figures are saved in `examples/figs/` and `examples/figs/case_study/` directories.

---

## ODE Examples (Phases 1-3)

These examples demonstrate the core verification pipeline for ordinary differential equations.

### Linear System

**File:** `linear_system.py`

#### Mathematical Setup

The linear system has dynamics:

```
dx/dt = Ax
```

where `A` is a stable matrix with eigenvalues having negative real parts:

```
A = [[-1,  1],
     [-1, -1]]
```

Eigenvalues: `-1 ± i` (stable spiral toward origin)

#### Example 1: UNSAFE Case

| Parameter | Value |
|-----------|-------|
| Initial set | Box `[-1, 1] × [-1, 1]` (contains origin) |
| Unsafe set | Ball at origin, radius 0.1 |
| Time horizon | T = 5.0 |
| Expected result | **UNSAFE** |

**Analysis:** Since all trajectories spiral toward the origin and the initial set contains points that start arbitrarily close to the origin, the system is clearly unsafe. The verifier correctly identifies this and returns a counterexample.

#### Example 2: SAFE Case

| Parameter | Value |
|-----------|-------|
| Initial set | Box `[2, 3] × [2, 3]` (far from origin) |
| Unsafe set | Ball at origin, radius 0.1 |
| Time horizon | T = 1.0 (short) |
| Expected result | **SAFE** |

**Analysis:** Although trajectories eventually converge to the origin, the short time horizon means they don't travel far enough to reach the unsafe set. The verifier proves safety by showing the minimum distance to the unsafe set remains positive with margin.

#### Generated Figures

- `linear_system_unsafe.png`: Shows spiraling trajectories from the initial box, with the objective function F_T (distance to unsafe) visualized over the initial set. The minimum is at/near 0, confirming unsafety.
- `linear_system_safe.png`: Shows partial spiral trajectories that don't reach the origin, with F_T remaining positive throughout.

---

### Harmonic Oscillator

**File:** `harmonic_oscillator.py`

#### Mathematical Setup

The harmonic oscillator has dynamics:

```
dx/dt = y
dy/dt = -ω²x
```

With ω = 1.0, solutions trace circles in phase space:

```
x(t) = x₀ cos(t) + y₀ sin(t)
y(t) = -x₀ sin(t) + y₀ cos(t)
```

**Key property:** The energy `E = (x² + y²)/2` is conserved, so orbits are closed circles with radius `r = √(x₀² + y₀²)`.

#### Energy Conservation Test

Before verification, we test that the numerical integrator preserves energy. Over two full periods (T = 4π), the relative energy drift should be < 10⁻⁴.

#### Example 1: SAFE Case

| Parameter | Value |
|-----------|-------|
| Initial set | Box `[0.8, 1.2] × [-0.2, 0.2]` |
| Unsafe set | Ball at (-3, 0), radius 0.5 |
| Time horizon | T = 2π (one period) |
| Expected result | **SAFE** |

**Analysis:** Initial conditions near (1, 0) have radius ≈ 1. The orbit swings to x = -1 at most, which is far from the unsafe ball centered at x = -3. Minimum distance is approximately:

```
min_distance ≈ |-1 - (-3)| - 0.5 = 2 - 0.5 = 1.5
```

#### Example 2: UNSAFE Case

| Parameter | Value |
|-----------|-------|
| Initial set | Box `[1.5, 2.0] × [-0.2, 0.2]` |
| Unsafe set | Ball at (-1.8, 0), radius 0.3 |
| Time horizon | T = π (half period) |
| Expected result | **UNSAFE** |

**Analysis:** Initial conditions at x = 1.5 to 2.0 have radius 1.5 to 2.0. After half a period, they swing to x = -1.5 to -2.0, directly hitting the unsafe ball centered at -1.8.

#### Generated Figures

- `harmonic_oscillator_safe.png`: Shows circular orbits that avoid the unsafe region in the far left.
- `harmonic_oscillator_unsafe.png`: Shows orbits that pass through the unsafe ball on the opposite side of the origin.

---

## Switching System Examples (Phase 4)

These examples demonstrate the full Filippov handling for hybrid/switching systems.

### Bouncing Ball

**File:** `bouncing_ball.py`

#### Mathematical Setup

The bouncing ball is an event-driven switching system:

**Mode 1 (Free fall):** When y ≥ 0
```
dy/dt = v
dv/dt = -g
```

**Mode 2 (Bounce):** When y = 0 and v < 0
```
v⁺ = -e × v⁻
```
where `e` is the coefficient of restitution (0.9 by default).

#### Simulation Demo

| Parameter | Value |
|-----------|-------|
| Initial state | Height = 1.0m, Velocity = 0 m/s |
| Gravity | g = 9.81 m/s² |
| Restitution | e = 0.9 |
| Time horizon | T = 3.0s |

**Analysis:** The ball drops, bounces, and loses energy with each bounce. The phase portrait shows characteristic parabolic arcs between bounces.

#### Safety Verification

| Parameter | Value |
|-----------|-------|
| Initial set | Height `[0.8, 1.2]`, Velocity `[-0.5, 0.5]` |
| Unsafe set | Velocity < -5 m/s (dangerous impact) |
| Time horizon | T = 1.0s |
| Expected result | **SAFE** |

**Analysis:** From the given initial heights, the ball cannot accelerate to -5 m/s within 1 second of free fall. The verifier confirms this.

#### Generated Figures

- `bouncing_ball_trajectory.png`: Height and velocity vs. time, showing multiple bounces with decreasing amplitude.
- `bouncing_ball_phase.png`: Phase portrait (height vs. velocity) showing the characteristic bouncing pattern.

---

### Relay Feedback

**File:** `relay_feedback.py`

#### Mathematical Setup

The relay feedback system is a classic example of Filippov dynamics:

```
dx/dt = -sign(x)
```

This is a 1D switching system where:
- `dx/dt = -1` if x > 0 (drives toward origin from right)
- `dx/dt = +1` if x < 0 (drives toward origin from left)

At x = 0, the system exhibits **sliding mode** behavior: both vector fields point inward, so the trajectory stays on the switching surface.

#### Sliding Mode Demonstration

| Parameter | Value |
|-----------|-------|
| Initial state | x = 1.0 |
| Time horizon | T = 3.0 |

**Analysis:** The trajectory decreases linearly from x = 1 at rate -1 until reaching x = 0 at t = 1. After that, it enters sliding mode and remains at x = 0. The demonstration shows ~99% of trajectory time is spent near x = 0.

#### Example 1: SAFE Case

| Parameter | Value |
|-----------|-------|
| Initial set | `[0.5, 1.5]` |
| Unsafe set | `[-10, -3]` |
| Time horizon | T = 2.0 |
| Expected result | **SAFE** |

**Analysis:** Starting from positive x, the relay drives toward the origin. It never goes below x = -3 because:
1. Trajectories converge to x = 0
2. The system would have to "overshoot" by 3 units, which is impossible

The objective function F_T (minimum distance to unsafe) is uniformly positive (~3.0) across the initial set.

#### Example 2: UNSAFE Case

| Parameter | Value |
|-----------|-------|
| Initial set | `[-0.5, 0.5]` (contains origin) |
| Unsafe set | `[-0.1, 0.1]` (small region around origin) |
| Time horizon | T = 2.0 |
| Expected result | **UNSAFE** |

**Analysis:** Since all trajectories converge to x = 0 and the unsafe set contains the origin, any trajectory will eventually enter the unsafe region. The verifier finds counterexamples at x₀ ≈ 0.04.

#### Generated Figures

- `relay_feedback_safe.png`: Two panels showing (1) trajectories over time all converging to 0 but staying away from x < -3, and (2) the objective function F_T over the initial set showing uniform positive values.

- `relay_feedback_unsafe.png`: Two panels showing (1) trajectories converging to the origin (inside unsafe region), with counterexample marked, and (2) the objective function showing where F_T ≤ 0 (unsafe initial conditions).

- `relay_feedback_sliding.png`: Single trajectory demonstrating convergence to and sliding along x = 0.

---

### Thermostat

**File:** `thermostat.py`

#### Mathematical Setup

The thermostat is a hybrid system with **hysteresis switching**:

```
dT/dt = -α(T - T_env) + β × heater_on
```

**Switching logic:**
- Heater turns ON when T < T_low (18°C)
- Heater turns OFF when T > T_high (22°C)
- Between T_low and T_high, heater state persists (hysteresis)

| Parameter | Value |
|-----------|-------|
| T_low | 18.0°C |
| T_high | 22.0°C |
| T_ambient | 10.0°C |
| Cooling rate α | 0.1 |
| Heating power β | 2.0 |

#### Hysteresis Demonstration

Two initial conditions demonstrate the hysteresis behavior:

1. **Starting cold (T = 17°C):** Heater turns on, temperature rises until T_high, then heater cycles.
2. **Starting hot (T = 23°C):** Heater stays off, temperature falls until T_low, then heater cycles.

Both trajectories eventually oscillate within the hysteresis band [18, 22].

#### Example 1: Avoid Freezing (SAFE)

| Parameter | Value |
|-----------|-------|
| Initial set | Temperature `[19, 21]` (within band) |
| Unsafe set | Temperature `[0, 15]` (too cold) |
| Time horizon | T = 10.0 |
| Expected result | **SAFE** |

**Analysis:** Starting within the regulation band, the thermostat maintains temperature above T_low = 18. It cannot drop to 15°C because the heater activates at 18°C.

#### Example 2: Avoid Overheating (SAFE)

| Parameter | Value |
|-----------|-------|
| Initial set | Temperature `[19, 21]` |
| Unsafe set | Temperature `[25, 100]` (too hot) |
| Time horizon | T = 10.0 |
| Expected result | **SAFE** |

**Analysis:** The heater turns off at T_high = 22°C, so temperature cannot exceed 22°C. The unsafe region at T > 25 is never reached.

#### Generated Figures

- `thermostat_hysteresis.png`: Temperature trajectories from cold and hot initial conditions, showing convergence to the hysteresis band.

- `thermostat_verification.png`: Two panels showing (1) sample trajectories staying within safe temperature bounds, and (2) the objective function F_T over initial temperatures.

---

## Case Studies (Phase 5.5)

These case studies demonstrate spline-verify on realistic physical systems with practical safety requirements.

### Vehicle Braking

**File:** `vehicle_braking.py`

#### Mathematical Setup

Emergency braking collision avoidance with first-order actuator dynamics:

```
dx/dt = -v          (position decreases as vehicle approaches obstacle)
dv/dt = a           (velocity changes with acceleration)
da/dt = -k_response × (a - a_target)  (brake actuator response)
```

| Parameter | Value |
|-----------|-------|
| k_brake | 9.0 m/s² (max deceleration ~0.9g) |
| k_response | 10.0 (brake actuator response rate) |

#### Example 1: SAFE Case

| Parameter | Value |
|-----------|-------|
| Initial set | x ∈ [80, 90] m, v ∈ [25, 28] m/s, a ≈ 0 |
| Unsafe set | x < 5 m (collision zone) |
| Time horizon | T = 6.0 s |
| Expected result | **SAFE** |

**Analysis:** With sufficient initial distance (80-90m) and effective braking, all vehicles stop before reaching the 5m collision zone.

#### Example 2: UNSAFE Case (Brake Failure)

| Parameter | Value |
|-----------|-------|
| Initial set | x ∈ [6, 8] m, v ∈ [30, 35] m/s, a ≈ 0 |
| Unsafe set | x < 5 m |
| Time horizon | T = 0.2 s |
| Control | No braking (brake failure) |
| Expected result | **UNSAFE** |

**Analysis:** With brake failure and high initial velocity at close range, collision is unavoidable.

#### Generated Figures

- `vehicle_braking_safe.png`: Trajectories showing successful braking before obstacle
- `vehicle_braking_unsafe.png`: Trajectories entering collision zone
- `vehicle_braking_objective.png`: F_T landscape over (x₀, v₀) showing safe/unsafe regions

---

### Quadrotor Altitude Control

**File:** `quadrotor.py`

#### Mathematical Setup

Simplified 2D quadrotor altitude dynamics:

```
dz/dt = vz
dvz/dt = (u_thrust - 1) × g - drag × vz
```

where `g = 9.81 m/s²`, `u_thrust` is normalized thrust (0 = off, 1 = hover, 2 = max), and drag provides damping.

#### Example 1: SAFE Case (PD-Controlled Hover)

| Parameter | Value |
|-----------|-------|
| Initial set | z ∈ [1, 2] m, vz ∈ [-0.5, 0.5] m/s |
| Unsafe set | z < 0.1 m (ground collision) |
| Control | PD controller targeting z = 2.0 m |
| Time horizon | T = 5.0 s |
| Expected result | **SAFE** |

**Analysis:** The PD controller maintains altitude above ground with comfortable margin.

#### Example 2: UNSAFE Case (Motor Failure)

| Parameter | Value |
|-----------|-------|
| Initial set | z ∈ [3, 4] m, vz ∈ [-1, 0] m/s |
| Unsafe set | z < 0.1 m |
| Control | None (motor failure, free fall) |
| Time horizon | T = 2.0 s |
| Expected result | **UNSAFE** |

**Analysis:** Under free fall from 3-4m altitude, the quadrotor impacts the ground within 2 seconds.

#### Generated Figures

- `quadrotor_hover_safe.png`: Altitude trajectories maintained by PD controller
- `quadrotor_fall_unsafe.png`: Free-fall trajectories hitting ground
- `quadrotor_phase.png`: Phase portrait (z vs vz) showing safe and unsafe dynamics

---

### Glucose-Insulin Regulation

**File:** `glucose_insulin.py`

#### Mathematical Setup

Simplified Bergman-inspired glucose-insulin model:

```
dG/dt = -k_g × G - k_i × I × G + G_input + G_basal
dI/dt = -k_decay × I + k_secrete × max(0, G - G_thresh)
```

| Parameter | Value |
|-----------|-------|
| k_g | 0.01 min⁻¹ (glucose clearance) |
| k_i | 0.0001 (mU/L)⁻¹ min⁻¹ (insulin effectiveness) |
| k_decay | 0.1 min⁻¹ (insulin degradation) |
| k_secrete | 0.005 mU/L per mg/dL per min |
| G_thresh | 80 mg/dL (secretion threshold) |
| G_basal | 1.0 mg/dL/min (hepatic production) |

**Safety thresholds:**
- Hypoglycemia: G < 70 mg/dL (dangerous low)
- Hyperglycemia: G > 180 mg/dL (dangerous high)
- Severe hyperglycemia: G > 200 mg/dL

#### Example 1: SAFE Case (Fasting)

| Parameter | Value |
|-----------|-------|
| Initial set | G ∈ [85, 100] mg/dL, I ∈ [8, 12] mU/L |
| Unsafe set | G < 70 mg/dL (hypoglycemia) |
| Condition | Fasting (no meal) |
| Time horizon | T = 120 min (2 hours) |
| Expected result | **SAFE** |

**Analysis:** During fasting, basal glucose production and insulin secretion maintain glucose above hypoglycemic levels.

#### Example 2: UNSAFE Case (Large Meal)

| Parameter | Value |
|-----------|-------|
| Initial set | G ∈ [90, 100] mg/dL, I ∈ [5, 8] mU/L |
| Unsafe set | G > 200 mg/dL (severe hyperglycemia) |
| Condition | Large meal (300 mg/dL glucose load) |
| Time horizon | T = 90 min (1.5 hours) |
| Expected result | **UNSAFE** |

**Analysis:** A large glucose load with insufficient initial insulin causes dangerous hyperglycemia spikes.

#### Generated Figures

- `glucose_safe.png`: Glucose and insulin time series during safe fasting
- `glucose_unsafe.png`: Glucose spike exceeding safety threshold
- `glucose_phase.png`: G-I phase portrait showing safe zone and meal response

---

## Demonstration Tutorial

**File:** `demonstration.py`

### Purpose

The demonstration tutorial generates a complete set of 15 figures explaining each step of the spline-verify safety verification pipeline. These figures are designed for research presentations and papers.

### Running the Demonstration

```bash
# Generate all 15 presentation figures
python examples/demonstration.py --save --outdir ./examples/figs/demo
```

### Figure List

The demonstration generates figures in `examples/figs/demo/`:

#### Section 1: Problem Setup
| Figure | Description |
|--------|-------------|
| `demo_01_problem_setup.png` | The safety verification problem: initial set, unsafe set, sample trajectories |
| `demo_02_sampling_strategies.png` | Comparison of sampling methods (Uniform, Latin Hypercube, Sobol, Halton) |

#### Section 2: Trajectory Simulation
| Figure | Description |
|--------|-------------|
| `demo_03_trajectory_bundle.png` | Multiple trajectories from sampled initial conditions with color gradient |
| `demo_04_distance_computation.png` | Computing F_T for one trajectory: distance to unsafe set over time |

#### Section 3: Objective Function
| Figure | Description |
|--------|-------------|
| `demo_05_objective_samples.png` | Sampled objective function values colored by F_T |
| `demo_06_objective_landscape.png` | Full objective function landscape with contours and global minimum |

#### Section 4: Spline Approximation
| Figure | Description |
|--------|-------------|
| `demo_07_spline_fitting.png` | RBF spline approximation with actual vs predicted comparison |
| `demo_08_approximation_error.png` | Error analysis: residual distribution, spatial error, error vs F_T |
| `demo_09_spline_minimization.png` | Multi-start optimization with paths and local/global minima |

#### Section 5: Error Bounds & Decision
| Figure | Description |
|--------|-------------|
| `demo_10_error_budget.png` | Error budget breakdown (integration, sampling, approximation, minimization) |
| `demo_11_decision_logic.png` | Safety decision visualization: SAFE/UNSAFE/UNKNOWN regions |

#### Section 6: Switching Systems
| Figure | Description |
|--------|-------------|
| `demo_12_switching_surfaces.png` | Relay feedback and thermostat switching behavior |
| `demo_13_piecewise_spline.png` | Piecewise spline fitting for switching systems |
| `demo_14_region_classifier.png` | SVM region classification with decision boundaries |

#### Section 7: Summary
| Figure | Description |
|--------|-------------|
| `demo_15_pipeline_summary.png` | Complete pipeline diagram: Sample → Simulate → Compute F_T → Fit Spline → Minimize → Decide |

### Design Principles

All demonstration figures follow consistent styling:
- **Color scheme**: Green (initial), Red (unsafe), Blue (trajectories), Purple (spline)
- **Large fonts**: Readable in presentations (14+ pt for labels)
- **High resolution**: 200 DPI for print quality
- **Standalone**: Each figure understandable without context
- **Progressive**: Figures build on each other, showing pipeline flow

---

## Figure Gallery

### ODE Examples

| Figure | Description |
|--------|-------------|
| `linear_system_unsafe.png` | Spiraling trajectories entering unsafe ball at origin |
| `linear_system_safe.png` | Trajectories that don't reach origin in short time |
| `harmonic_oscillator_safe.png` | Circular orbits avoiding distant unsafe ball |
| `harmonic_oscillator_unsafe.png` | Orbits passing through unsafe ball on opposite side |

### Switching System Examples

| Figure | Description |
|--------|-------------|
| `bouncing_ball_trajectory.png` | Height/velocity time series with bounces |
| `bouncing_ball_phase.png` | Phase portrait of bouncing dynamics |
| `relay_feedback_safe.png` | Relay feedback avoiding far unsafe region |
| `relay_feedback_unsafe.png` | Relay feedback entering origin unsafe region |
| `relay_feedback_sliding.png` | Sliding mode behavior demonstration |
| `thermostat_hysteresis.png` | Temperature regulation with hysteresis |
| `thermostat_verification.png` | Thermostat safety verification |

### Case Study Figures (in `examples/figs/case_study/`)

| Figure | Description |
|--------|-------------|
| `vehicle_braking_safe.png` | Vehicle braking successfully avoiding collision |
| `vehicle_braking_unsafe.png` | Vehicle collision under brake failure |
| `vehicle_braking_objective.png` | F_T landscape over initial (x, v) |
| `quadrotor_hover_safe.png` | Quadrotor maintaining altitude with PD control |
| `quadrotor_fall_unsafe.png` | Quadrotor free-fall ground collision |
| `quadrotor_phase.png` | Phase portrait of quadrotor altitude dynamics |
| `glucose_safe.png` | Blood glucose regulation during fasting |
| `glucose_unsafe.png` | Dangerous glucose spike from large meal |
| `glucose_phase.png` | Glucose-insulin phase portrait with safe zone |

### Comparison Figures (in `results/comparison/`)

Generated by `python scripts/run_comparison.py --save`:

| Figure | Description |
|--------|-------------|
| `bounds_comparison.png` | Spline-verify min vs M-S lower bound across problems |
| `gap_analysis.png` | Gap and runtime speedup analysis |
| `flow_dist_5pi_4_distance_function.png` | 3D surface of F̃_T for Flow θ=5π/4 (SAFE) |
| `flow_dist_5pi_4_distance_1d_slice.png` | 1D slice for Flow θ=5π/4 |
| `flow_dist_3pi_2_distance_function.png` | 3D surface of F̃_T for Flow θ=3π/2 (SAFE) |
| `flow_dist_3pi_2_distance_1d_slice.png` | 1D slice for Flow θ=3π/2 |
| `flow_dist_7pi_4_distance_function.png` | 3D surface of F̃_T for Flow θ=7π/4 (UNSAFE) |
| `flow_dist_7pi_4_distance_1d_slice.png` | 1D slice for Flow θ=7π/4 |
| `moon_distance_function.png` | 3D surface of F̃_T for Moon system (SAFE) |
| `moon_distance_1d_slice.png` | 1D slice for Moon system |
| `*_sampling_comparison.png` | M-S ball vs spline-verify box sampling |

### Demonstration Figures

| Figure | Description |
|--------|-------------|
| `demo_01_problem_setup.png` | Safety verification problem setup |
| `demo_02_sampling_strategies.png` | Sampling method comparison |
| `demo_03_trajectory_bundle.png` | Trajectory simulation from samples |
| `demo_04_distance_computation.png` | F_T computation for one trajectory |
| `demo_05_objective_samples.png` | Sampled objective function |
| `demo_06_objective_landscape.png` | Objective function landscape |
| `demo_07_spline_fitting.png` | RBF spline approximation |
| `demo_08_approximation_error.png` | Approximation error analysis |
| `demo_09_spline_minimization.png` | Multi-start minimization |
| `demo_10_error_budget.png` | Error budget breakdown |
| `demo_11_decision_logic.png` | Safety decision logic |
| `demo_12_switching_surfaces.png` | Switching system structure |
| `demo_13_piecewise_spline.png` | Piecewise spline for switching |
| `demo_14_region_classifier.png` | SVM region classification |
| `demo_15_pipeline_summary.png` | Complete pipeline diagram |

---

## Interpreting Verification Results

Each verification run outputs:

```
Verification Result: SAFE/UNSAFE/UNKNOWN
========================================
Minimum F̃_T:      <value>      # Minimum of spline approximation
Error bound ε:    <value>      # Total error from all sources
Safety margin:    <value>      # F̃_T - ε (positive = proven safe)
```

**Decision logic:**
- **SAFE**: `min F̃_T > ε` — proven safe with positive margin
- **UNSAFE**: `min F̃_T ≤ 0` — found trajectory entering unsafe set
- **UNKNOWN**: `0 < min F̃_T ≤ ε` — cannot certify either way

For switching systems, additional information is provided:

```
Switching Analysis:
  Number of regions: <n>
  Region counts: {<label>: <count>, ...}
```

This shows how initial conditions were classified by crossing behavior.

---

## Adding New Examples

To add a new example:

1. Create a new Python file in `examples/`
2. Import the appropriate dynamics and verifier:
   - ODE: `ODEDynamics`, `SafetyVerifier`
   - Switching: `SwitchingDynamics`, `SwitchingVerifier`
3. Define initial set, unsafe set, and time horizon
4. Run verification and print results
5. Add visualization functions with `--save` support
6. Document in this file

Example template:

```python
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spline_verify.dynamics import ODEDynamics  # or SwitchingDynamics
from spline_verify.geometry import HyperRectangle, Ball
from spline_verify.verification import SafetyVerifier  # or SwitchingVerifier

def example():
    dynamics = ODEDynamics.from_function(lambda t, x: ...)
    initial_set = HyperRectangle(lower=..., upper=...)
    unsafe_set = Ball(center=..., radius=...)
    T = ...

    verifier = SafetyVerifier(n_samples=200, seed=42)
    result = verifier.verify(dynamics, initial_set, unsafe_set, T)
    print(result.summary())
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--outdir', type=str, default='.')
    args = parser.parse_args()

    result = example()

    if args.save:
        # Generate and save figures
        ...
```
