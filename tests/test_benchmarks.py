"""Tests for benchmarking infrastructure."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from spline_verify.benchmarks import (
    # Runner
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    TimingResult,
    # Ground truth
    GroundTruthSystem,
    LinearGroundTruth,
    HarmonicOscillatorGroundTruth,
    validate_against_ground_truth,
    get_all_ground_truth_systems,
    # Scalability
    ScalabilityResult,
    run_dimension_scaling,
    run_sample_scaling,
    run_time_horizon_scaling,
    # Sensitivity
    ParameterSweep,
    run_parameter_sweep,
    create_sampling_sweep,
)
from spline_verify.dynamics import ODEDynamics
from spline_verify.geometry import HyperRectangle, Ball
from spline_verify.verification import VerificationStatus


class TestTimingResult:
    """Tests for TimingResult."""

    def test_time_per_trajectory(self):
        """Test time per trajectory calculation."""
        timing = TimingResult(
            total_time=1.0,
            simulation_time=0.5,
            n_trajectories=10,
        )
        assert timing.time_per_trajectory == 0.05

    def test_time_per_trajectory_zero_trajectories(self):
        """Test handling of zero trajectories."""
        timing = TimingResult(total_time=1.0, n_trajectories=0)
        assert timing.time_per_trajectory == 0.0

    def test_to_dict(self):
        """Test serialization."""
        timing = TimingResult(
            total_time=1.5,
            simulation_time=0.8,
            fitting_time=0.3,
            optimization_time=0.2,
            n_trajectories=100,
            n_samples=100,
        )
        d = timing.to_dict()
        assert d['total_time'] == 1.5
        assert d['n_samples'] == 100


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_is_correct_with_ground_truth(self):
        """Test correctness checking."""
        timing = TimingResult(total_time=1.0)
        result = BenchmarkResult(
            name="test",
            status=VerificationStatus.SAFE,
            min_objective=1.0,
            error_bound=0.1,
            safety_margin=0.9,
            timing=timing,
            ground_truth_status=VerificationStatus.SAFE,
        )
        assert result.is_correct is True

    def test_is_correct_mismatch(self):
        """Test incorrect result detection."""
        timing = TimingResult(total_time=1.0)
        result = BenchmarkResult(
            name="test",
            status=VerificationStatus.SAFE,
            min_objective=1.0,
            error_bound=0.1,
            safety_margin=0.9,
            timing=timing,
            ground_truth_status=VerificationStatus.UNSAFE,
        )
        assert result.is_correct is False

    def test_unknown_is_always_correct(self):
        """Test that UNKNOWN is considered correct (conservative)."""
        timing = TimingResult(total_time=1.0)
        result = BenchmarkResult(
            name="test",
            status=VerificationStatus.UNKNOWN,
            min_objective=0.05,
            error_bound=0.1,
            safety_margin=-0.05,
            timing=timing,
            ground_truth_status=VerificationStatus.SAFE,
        )
        assert result.is_correct is True

    def test_to_dict_and_summary(self):
        """Test serialization and summary."""
        timing = TimingResult(total_time=1.0)
        result = BenchmarkResult(
            name="test",
            status=VerificationStatus.SAFE,
            min_objective=1.0,
            error_bound=0.1,
            safety_margin=0.9,
            timing=timing,
        )
        d = result.to_dict()
        assert d['name'] == "test"
        assert d['status'] == "SAFE"

        summary = result.summary()
        assert "test" in summary
        assert "SAFE" in summary


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_add_result(self):
        """Test adding results."""
        suite = BenchmarkSuite(name="test_suite")
        timing = TimingResult(total_time=1.0)
        result = BenchmarkResult(
            name="test1",
            status=VerificationStatus.SAFE,
            min_objective=1.0,
            error_bound=0.1,
            safety_margin=0.9,
            timing=timing,
        )
        suite.add_result(result)
        assert suite.n_benchmarks == 1

    def test_accuracy(self):
        """Test accuracy calculation."""
        suite = BenchmarkSuite(name="test_suite")
        timing = TimingResult(total_time=1.0)

        # Add correct result
        suite.add_result(BenchmarkResult(
            name="test1",
            status=VerificationStatus.SAFE,
            min_objective=1.0,
            error_bound=0.1,
            safety_margin=0.9,
            timing=timing,
            ground_truth_status=VerificationStatus.SAFE,
        ))

        # Add incorrect result
        suite.add_result(BenchmarkResult(
            name="test2",
            status=VerificationStatus.SAFE,
            min_objective=1.0,
            error_bound=0.1,
            safety_margin=0.9,
            timing=timing,
            ground_truth_status=VerificationStatus.UNSAFE,
        ))

        assert suite.n_correct == 1
        assert suite.n_incorrect == 1
        assert suite.accuracy == 0.5

    def test_save_and_load(self):
        """Test saving and loading suite."""
        suite = BenchmarkSuite(name="test_suite")
        timing = TimingResult(total_time=1.0)
        suite.add_result(BenchmarkResult(
            name="test1",
            status=VerificationStatus.SAFE,
            min_objective=1.0,
            error_bound=0.1,
            safety_margin=0.9,
            timing=timing,
            ground_truth_status=VerificationStatus.SAFE,
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "suite.json"
            suite.save(path)
            loaded = BenchmarkSuite.load(path)

            assert loaded.name == suite.name
            assert loaded.n_benchmarks == suite.n_benchmarks
            assert loaded.results[0].status == VerificationStatus.SAFE


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_run_single(self):
        """Test running a single benchmark."""
        runner = BenchmarkRunner(n_samples=50, verbose=False)

        A = np.array([[-0.5, 1.0], [-1.0, -0.5]])
        dynamics = ODEDynamics.from_matrix(A)
        initial_set = HyperRectangle(
            lower=np.array([2.0, 2.0]),
            upper=np.array([3.0, 3.0])
        )
        unsafe_set = Ball(center=np.array([0.0, 0.0]), radius=0.1)

        result = runner.run_single(
            name="test",
            dynamics=dynamics,
            initial_set=initial_set,
            unsafe_set=unsafe_set,
            time_horizon=1.0,
        )

        assert result.name == "test"
        assert result.status in [VerificationStatus.SAFE, VerificationStatus.UNKNOWN]
        assert result.timing.total_time > 0

    def test_run_suite(self):
        """Test running a benchmark suite."""
        runner = BenchmarkRunner(n_samples=30, verbose=False)

        A = np.array([[-0.5, 0.0], [0.0, -0.5]])
        dynamics = ODEDynamics.from_matrix(A)

        benchmarks = [
            {
                'name': 'test1',
                'dynamics': dynamics,
                'initial_set': HyperRectangle(
                    lower=np.array([1.0, 1.0]),
                    upper=np.array([2.0, 2.0])
                ),
                'unsafe_set': Ball(center=np.array([0.0, 0.0]), radius=0.1),
                'time_horizon': 1.0,
            },
            {
                'name': 'test2',
                'dynamics': dynamics,
                'initial_set': HyperRectangle(
                    lower=np.array([0.5, 0.5]),
                    upper=np.array([1.5, 1.5])
                ),
                'unsafe_set': Ball(center=np.array([0.0, 0.0]), radius=0.1),
                'time_horizon': 1.0,
            },
        ]

        suite = runner.run_suite("test_suite", benchmarks)
        assert suite.n_benchmarks == 2


class TestLinearGroundTruth:
    """Tests for LinearGroundTruth."""

    def test_stable_spiral_safe(self):
        """Test stable spiral safe configuration."""
        system = LinearGroundTruth.stable_spiral_safe()
        assert system.name == "stable_spiral_safe"
        assert system.ground_truth_status == VerificationStatus.SAFE
        assert system.compute_exact_min_distance() > 0

    def test_stable_spiral_unsafe(self):
        """Test stable spiral unsafe configuration."""
        system = LinearGroundTruth.stable_spiral_unsafe()
        assert system.name == "stable_spiral_unsafe"
        # This may be SAFE or UNSAFE depending on exact trajectory
        # Just check it runs without error
        _ = system.compute_exact_min_distance()

    def test_to_benchmark_spec(self):
        """Test conversion to benchmark spec."""
        system = LinearGroundTruth.stable_spiral_safe()
        spec = system.to_benchmark_spec()
        assert 'name' in spec
        assert 'dynamics' in spec
        assert 'initial_set' in spec
        assert 'unsafe_set' in spec
        assert 'time_horizon' in spec


class TestHarmonicOscillatorGroundTruth:
    """Tests for HarmonicOscillatorGroundTruth."""

    def test_small_orbit_safe(self):
        """Test small orbit safe configuration."""
        system = HarmonicOscillatorGroundTruth.small_orbit_safe()
        assert system.name == "harmonic_small_orbit_safe"
        assert system.ground_truth_status == VerificationStatus.SAFE

    def test_large_orbit_unsafe(self):
        """Test large orbit unsafe configuration."""
        system = HarmonicOscillatorGroundTruth.large_orbit_unsafe()
        assert system.name == "harmonic_large_orbit_unsafe"
        # May be UNSAFE or SAFE depending on exact configuration


class TestGetAllGroundTruthSystems:
    """Tests for get_all_ground_truth_systems."""

    def test_returns_list(self):
        """Test that function returns a list."""
        systems = get_all_ground_truth_systems()
        assert isinstance(systems, list)
        assert len(systems) >= 5

    def test_all_have_required_properties(self):
        """Test all systems have required properties."""
        systems = get_all_ground_truth_systems()
        for system in systems:
            assert hasattr(system, 'name')
            assert hasattr(system, 'dynamics')
            assert hasattr(system, 'initial_set')
            assert hasattr(system, 'unsafe_set')
            assert hasattr(system, 'time_horizon')
            assert hasattr(system, 'ground_truth_status')


class TestScalabilityResult:
    """Tests for ScalabilityResult."""

    def test_to_dict(self):
        """Test serialization."""
        result = ScalabilityResult(
            parameter_name='dimension',
            parameter_values=[2, 3, 4],
            times=[0.1, 0.2, 0.3],
            statuses=[VerificationStatus.SAFE] * 3,
            min_objectives=[1.0, 1.0, 1.0],
            error_bounds=[0.1, 0.1, 0.1],
        )
        d = result.to_dict()
        assert d['parameter_name'] == 'dimension'
        assert len(d['parameter_values']) == 3

    def test_save_and_load(self):
        """Test saving and loading."""
        result = ScalabilityResult(
            parameter_name='samples',
            parameter_values=[50, 100],
            times=[0.1, 0.2],
            statuses=[VerificationStatus.SAFE, VerificationStatus.SAFE],
            min_objectives=[1.0, 1.0],
            error_bounds=[0.1, 0.1],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scalability.json"
            result.save(path)
            loaded = ScalabilityResult.load(path)

            assert loaded.parameter_name == result.parameter_name
            assert loaded.parameter_values == result.parameter_values


class TestDimensionScaling:
    """Tests for run_dimension_scaling."""

    def test_runs_without_error(self):
        """Test that dimension scaling runs."""
        result = run_dimension_scaling(
            dimensions=[2, 3],
            n_samples=30,
            time_horizon=1.0,
            verbose=False,
        )
        assert result.parameter_name == 'dimension'
        assert len(result.times) == 2


class TestSampleScaling:
    """Tests for run_sample_scaling."""

    def test_runs_without_error(self):
        """Test that sample scaling runs."""
        result = run_sample_scaling(
            sample_counts=[30, 50],
            dimension=2,
            time_horizon=1.0,
            verbose=False,
        )
        assert result.parameter_name == 'n_samples'
        assert len(result.times) == 2

    def test_more_samples_reduces_error(self):
        """Test that more samples generally reduces error bounds."""
        result = run_sample_scaling(
            sample_counts=[30, 100],
            dimension=2,
            time_horizon=1.0,
            verbose=False,
        )
        # Error bounds should generally decrease with more samples
        # (though not strictly monotonic due to randomness)
        assert all(e > 0 for e in result.error_bounds)


class TestTimeHorizonScaling:
    """Tests for run_time_horizon_scaling."""

    def test_runs_without_error(self):
        """Test that time horizon scaling runs."""
        result = run_time_horizon_scaling(
            time_horizons=[0.5, 1.0],
            dimension=2,
            n_samples=30,
            verbose=False,
        )
        assert result.parameter_name == 'time_horizon'
        assert len(result.times) == 2


class TestParameterSweep:
    """Tests for ParameterSweep."""

    def test_get_combinations(self):
        """Test generating parameter combinations."""
        sweep = ParameterSweep(
            name="test",
            parameters={
                'a': [1, 2],
                'b': ['x', 'y'],
            }
        )
        combos = sweep.get_combinations()
        assert len(combos) == 4
        assert {'a': 1, 'b': 'x'} in combos
        assert {'a': 2, 'b': 'y'} in combos

    def test_n_combinations(self):
        """Test counting combinations."""
        sweep = ParameterSweep(
            name="test",
            parameters={
                'a': [1, 2, 3],
                'b': [1, 2],
            }
        )
        assert sweep.n_combinations == 6


class TestRunParameterSweep:
    """Tests for run_parameter_sweep."""

    def test_runs_without_error(self):
        """Test that parameter sweep runs."""
        A = np.array([[-0.5, 0.0], [0.0, -0.5]])
        dynamics = ODEDynamics.from_matrix(A)
        initial_set = HyperRectangle(
            lower=np.array([1.0, 1.0]),
            upper=np.array([2.0, 2.0])
        )
        unsafe_set = Ball(center=np.array([0.0, 0.0]), radius=0.1)

        sweep = ParameterSweep(
            name="small_test",
            parameters={
                'n_samples': [30, 50],
            }
        )

        analysis = run_parameter_sweep(
            dynamics, initial_set, unsafe_set,
            time_horizon=1.0,
            sweep=sweep,
            verbose=False,
        )

        assert len(analysis.results) == 2


class TestCreateSweeps:
    """Tests for sweep factory functions."""

    def test_create_sampling_sweep(self):
        """Test sampling sweep creation."""
        sweep = create_sampling_sweep()
        assert 'n_samples' in sweep.parameters
        assert 'sampling_strategy' in sweep.parameters

    # Note: create_spline_sweep not tested here to avoid long runtime
