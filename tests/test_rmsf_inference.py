import unittest

from salsbury_md_analysis.rmsf_inference import RMSFInferenceError, rmsf_permutation_test


class RMSFInferenceTests(unittest.TestCase):
    def test_exact_permutation_is_unit_level_and_max_t_adjusted(self):
        report = rmsf_permutation_test(
            [[1.0, 1.0], [1.1, 1.0], [0.9, 1.0]],
            [[4.0, 1.0], [4.1, 1.0], [3.9, 1.0]],
        )
        self.assertEqual(report["method"], "exact")
        self.assertEqual(report["evaluated_partition_count"], 20)
        self.assertLess(report["two_sided_pointwise_p_values"][0], 0.2)
        self.assertEqual(report["two_sided_pointwise_p_values"][1], 1.0)
        self.assertGreaterEqual(
            report["max_t_familywise_p_values"][0], report["two_sided_pointwise_p_values"][0]
        )

    def test_frame_pseudoreplication_is_blocked_by_minimum_unit_gate(self):
        with self.assertRaises(RMSFInferenceError):
            rmsf_permutation_test([[1.0, 2.0]], [[2.0, 3.0], [2.1, 3.1]])


if __name__ == "__main__":
    unittest.main()
