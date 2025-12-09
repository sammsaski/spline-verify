"""Benchmarking infrastructure for spline-verify.

This module provides tools for:
- Running systematic benchmarks (BenchmarkRunner)
- Validating against ground truth (analytical solutions)
- Analyzing scalability (dimension, samples, time horizon)
- Sensitivity analysis (parameter sweeps)
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
]
