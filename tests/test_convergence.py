import unittest
import math

from salsbury_md_analysis.convergence import (
    ConvergenceAnalysisError,
    _exact_observation_accounting,
    _series_diagnostic,
    autocorrelation_adjusted_mean_uncertainty,
    autocorrelation_sequence,
    effective_sample_size,
)


class ConvergenceTests(unittest.TestCase):
    def test_impossible_block_contract_fails_technically_not_scientifically(self):
        settings = {
            "block_size_frames": 1000,
            "minimum_blocks": 4,
            "include_partial_final_block": True,
            "minimum_effective_sample_size": 20.0,
            "maximum_split_mean_difference_in_sd": 1.0,
        }
        with self.assertRaisesRegex(
            ConvergenceAnalysisError,
            "500 selected observations cannot yield 4 blocks of 1000",
        ):
            _series_diagnostic([float(value) for value in range(500)], settings)

    def test_compatible_selected_observation_blocks_are_reported(self):
        settings = {
            "block_size_frames": 50,
            "minimum_blocks": 4,
            "include_partial_final_block": True,
            "minimum_effective_sample_size": 20.0,
            "maximum_split_mean_difference_in_sd": 1.0,
        }
        report = _series_diagnostic(
            [float(value % 7) for value in range(500)], settings
        )
        self.assertEqual(len(report["block_means"]), 10)
        self.assertTrue(report["passes_minimum_blocks"])
        self.assertEqual(
            report["block_contract"]["minimum_required_observations"], 151
        )

    def test_exact_frame_and_metric_value_accounting_are_distinct(self):
        upstream = {"systems": [{"replicas": [{"segments": [{
            "segment_id": "s1", "source_frame_count": 10,
            "evaluated_frame_count": 4, "timeseries": [{}, {}, {}, {}],
        }]}]}]}
        report = _exact_observation_accounting(upstream, 2)
        self.assertEqual(report["source_physical_frame_count"], 10)
        self.assertEqual(report["selected_physical_frame_count"], 4)
        self.assertEqual(report["symmetry_expanded_observation_count"], 4)
        self.assertEqual(report["metric_value_observation_count"], 8)
        self.assertTrue(report["subsampling_triggered"])

    def test_autocorrelation_sequence_is_explicit_and_bounded(self):
        report = autocorrelation_sequence([0.0, 1.0, 0.0, 1.0], maximum_lag=2)
        self.assertEqual(report["maximum_lag"], 2)
        self.assertEqual(report["autocorrelation"][0], 1.0)
        self.assertLess(report["autocorrelation"][1], 0.0)
        constant = autocorrelation_sequence([2.0, 2.0, 2.0])
        self.assertTrue(constant["constant_series"])
        self.assertIsNone(constant["autocorrelation"][1])

    def test_fft_autocorrelation_matches_direct_overlap_normalization(self):
        values = [
            math.sin(index / 17.0) + 0.2 * math.cos(index / 5.0)
            for index in range(1024)
        ]
        report = autocorrelation_sequence(values)
        self.assertEqual(report["algorithm"], "fft_overlap_normalized_v1")
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        for lag in (0, 1, 7, 127, 511, 1023):
            expected = (
                sum(
                    (values[index] - mean) * (values[index + lag] - mean)
                    for index in range(len(values) - lag)
                )
                / (len(values) - lag)
                / variance
            )
            self.assertAlmostEqual(report["autocorrelation"][lag], expected, places=9)

    def test_ess_distinguishes_constant_and_alternating_series(self):
        constant = effective_sample_size([1.0, 1.0, 1.0, 1.0])
        self.assertIsNone(constant["effective_sample_size"])
        alternating = effective_sample_size([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(alternating["effective_sample_size"], 6.0)
        self.assertEqual(alternating["positive_lag_count"], 0)

    def test_autocorrelation_adjusted_uncertainty_is_exploratory(self):
        report = autocorrelation_adjusted_mean_uncertainty(
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        )
        self.assertEqual(report["status"], "exploratory_autocorrelation_adjusted")
        self.assertFalse(report["acceptance_gate"])
        self.assertIsNotNone(report["standard_error"])
        self.assertLess(report["interval_lower"], report["mean"])
        self.assertGreater(report["interval_upper"], report["mean"])


if __name__ == "__main__":
    unittest.main()
