import math
import unittest

from salsbury_md_analysis.moments import (
    CoordinateMoments,
    DisplacementCovariance,
    sample_summary,
)


class MomentTests(unittest.TestCase):
    def test_streaming_mean_and_population_rmsf(self):
        moments = CoordinateMoments(1)
        moments.update([(-1, 0, 0)])
        moments.update([(1, 0, 0)])
        self.assertEqual(moments.mean_coordinate(0), (0.0, 0.0, 0.0))
        self.assertEqual(moments.rmsf(0), 1.0)

    def test_merged_moments_match_direct_updates(self):
        left = CoordinateMoments(1)
        right = CoordinateMoments(1)
        direct = CoordinateMoments(1)
        for value in (0.0, 2.0):
            left.update([(value, 0, 0)])
            direct.update([(value, 0, 0)])
        for value in (4.0, 6.0):
            right.update([(value, 0, 0)])
            direct.update([(value, 0, 0)])
        left.merge(right)
        self.assertEqual(left.mean_coordinate(0), direct.mean_coordinate(0))
        self.assertAlmostEqual(left.rmsf(0), direct.rmsf(0))

    def test_sample_summary_does_not_invent_one_sample_uncertainty(self):
        one = sample_summary([2.0])
        self.assertIsNone(one["sample_sd"])
        summary = sample_summary([1.0, 2.0])
        self.assertAlmostEqual(summary["sample_sd"], math.sqrt(0.5))
        self.assertAlmostEqual(summary["sem"], 0.5)

    def test_displacement_correlation_preserves_sign_and_null_variance(self):
        covariance = DisplacementCovariance(3)
        for value in (-1.0, 0.0, 1.0):
            covariance.update([
                (value, 0, 0),
                (2 * value, 0, 0),
                (0, 0, 0),
            ])
        matrix = covariance.correlation_matrix(1.0e-12)
        self.assertAlmostEqual(matrix[0][1], 1.0)
        self.assertIsNone(matrix[0][2])
        self.assertIsNone(matrix[2][2])

    def test_merged_displacement_covariance_matches_direct_stream(self):
        left = DisplacementCovariance(2)
        right = DisplacementCovariance(2)
        direct = DisplacementCovariance(2)
        frames = [(-2.0, 2.0), (-1.0, 1.0), (1.0, -1.0), (2.0, -2.0)]
        for index, (first, second) in enumerate(frames):
            coordinates = [(first, 0, 0), (second, 0, 0)]
            direct.update(coordinates)
            (left if index < 2 else right).update(coordinates)
        left.merge(right)
        merged_matrix = left.correlation_matrix(1.0e-12)
        direct_matrix = direct.correlation_matrix(1.0e-12)
        for merged_row, direct_row in zip(merged_matrix, direct_matrix):
            for merged, expected in zip(merged_row, direct_row):
                self.assertAlmostEqual(merged, expected)


if __name__ == "__main__":
    unittest.main()
