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
4. [Figure Gallery](#figure-gallery)

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
mkdir -p examples/figs
python examples/linear_system.py --save --outdir ./examples/figs
python examples/harmonic_oscillator.py --save --outdir ./examples/figs
python examples/bouncing_ball.py --save --outdir ./examples/figs
python examples/relay_feedback.py --save --outdir ./examples/figs
python examples/thermostat.py --save --outdir ./examples/figs
```

All figures are saved in `examples/figs/` directory.

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
