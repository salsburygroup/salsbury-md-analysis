import math
import unittest

import numpy as np

from salsbury_md_analysis.information_dynamics import (
    coskewness_tensor,
    displacement_propagator,
    lagged_cross_correlation,
    transfer_entropy_matrix,
)


class InformationDynamicsTests(unittest.TestCase):
    def test_transfer_entropy_detects_declared_one_way_binary_driver(self):
        generator = np.random.default_rng(20260811)
        source = generator.integers(0, 2, size=5000)
        target = np.zeros_like(source)
        target[1:] = source[:-1]
        report = transfer_entropy_matrix(
            [np.column_stack((source, target)).tolist()],
            lag_frames=1,
            bin_count=2,
            minimum_pairs=100,
        )
        matrix = report["transfer_entropy_nats"]
        self.assertGreater(matrix[0][1], 0.60)
        self.assertLess(matrix[1][0], 0.01)

    def test_transfer_entropy_pair_count_is_segment_safe(self):
        report = transfer_entropy_matrix(
            [
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            ],
            lag_frames=1,
            bin_count=2,
            minimum_pairs=4,
        )
        self.assertEqual(report["pair_count"], 4)

    def test_lagged_cross_correlation_is_directional_and_segment_safe(self):
        generator = np.random.default_rng(17)
        source = generator.normal(size=1000)
        target = np.zeros_like(source)
        target[1:] = source[:-1]
        report = lagged_cross_correlation(
            [np.column_stack((source, target)).tolist()],
            lag_frames=1,
            minimum_pairs=100,
        )
        matrix = report["lagged_cross_correlation"]
        self.assertGreater(matrix[0][1], 0.99)
        self.assertLess(abs(matrix[1][0]), 0.1)
        self.assertEqual(report["pair_count"], 999)

    def test_coskewness_and_displacement_propagator_are_finite_and_gated(self):
        values = [[float(index), float(index * index)] for index in range(1, 30)]
        tensor = coskewness_tensor(values, maximum_tensor_elements=8)
        self.assertEqual(tensor["feature_count"], 2)
        self.assertGreater(tensor["coskewness"][1][1][1], 0.5)

        coordinates = [
            [
                [[float(frame), 0.0, 0.0], [0.0, float(frame), 0.0]]
                for frame in range(4)
            ],
            [
                [[float(frame + 10), 0.0, 0.0], [0.0, float(frame + 10), 0.0]]
                for frame in range(3)
            ],
        ]
        propagator = displacement_propagator(coordinates, lag_frames=1)
        self.assertEqual(propagator["pair_count"], 5)
        self.assertEqual(propagator["atom_count"], 2)
        self.assertTrue(math.isfinite(
            propagator["normalized_displacement_dot_matrix"][0][1]
        ))


if __name__ == "__main__":
    unittest.main()
