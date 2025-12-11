"""Benchmarking infrastructure for spline-verify.

This module provides tools for:
- Running systematic benchmarks (BenchmarkRunner)
- Validating against ground truth (analytical solutions)
- Analyzing scalability (dimension, samples, time horizon)
- Sensitivity analysis (parameter sweeps)
- Ablation studies (component-wise impact analysis)
"""

from .runner import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    TimingResult,
)
from .ground_truth import (
    GroundTruthSystem,
    LinearGroundTruth,
    HarmonicOscillatorGroundTruth,
    validate_against_ground_truth,
    get_all_ground_truth_systems,
)
from .scalability import (
    ScalabilityAnalysis,
    ScalabilityResult,
    run_dimension_scaling,
    run_sample_scaling,
    run_time_horizon_scaling,
    run_full_scalability_analysis,
)
from .sensitivity import (
    SensitivityAnalysis,
    ParameterSweep,
    ParameterSweepResult,
    run_parameter_sweep,
    create_sampling_sweep,
    create_spline_sweep,
    create_comprehensive_sweep,
    run_sampling_sensitivity,
    run_spline_sensitivity,
)
from .ablation import (
    AblationResult,
    AblationStudy,
    create_diagonal_decay_4d,
    create_coupled_oscillator_4d,
    create_van_der_pol_2d,
    create_lorenz_like_3d,
    create_high_dim_diagonal_6d,
    get_ablation_problems,
    run_full_ablation_study,
    # Harder benchmarks
    get_hard_benchmark_problems,
    create_high_dim_coupled_8d,
    create_high_dim_diagonal_10d,
    create_stiff_vanderpol_2d,
    create_rossler_3d,
    create_double_pendulum_4d,
    create_lotka_volterra_4d,
    create_neural_network_6d,
)

__all__ = [
    # Runner
    'BenchmarkResult',
    'BenchmarkRunner',
    'BenchmarkSuite',
    'TimingResult',
    # Ground truth
    'GroundTruthSystem',
    'LinearGroundTruth',
    'HarmonicOscillatorGroundTruth',
    'validate_against_ground_truth',
    'get_all_ground_truth_systems',
    # Scalability
    'ScalabilityAnalysis',
    'ScalabilityResult',
    'run_dimension_scaling',
    'run_sample_scaling',
    'run_time_horizon_scaling',
    'run_full_scalability_analysis',
    # Sensitivity
    'SensitivityAnalysis',
    'ParameterSweep',
    'ParameterSweepResult',
    'run_parameter_sweep',
    'create_sampling_sweep',
    'create_spline_sweep',
    'create_comprehensive_sweep',
    'run_sampling_sensitivity',
    'run_spline_sensitivity',
    # Ablation
    'AblationResult',
    'AblationStudy',
    'create_diagonal_decay_4d',
    'create_coupled_oscillator_4d',
    'create_van_der_pol_2d',
    'create_lorenz_like_3d',
    'create_high_dim_diagonal_6d',
    'get_ablation_problems',
    'run_full_ablation_study',
]
