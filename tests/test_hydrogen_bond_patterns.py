import unittest

import numpy as np

from salsbury_md_analysis.hydrogen_bond_patterns import (
    encode_bond_patterns,
    jaccard_distance_matrix,
    pam_jaccard,
)


class HydrogenBondPatternTests(unittest.TestCase):
    def test_encoding_is_deterministic_and_jaccard_handles_empty_frames(self):
        identifiers, patterns = encode_bond_patterns([{"b", "a"}, set(), {"b"}])
        self.assertEqual(identifiers, ["a", "b"])
        distances = jaccard_distance_matrix(patterns)
        self.assertAlmostEqual(distances[0, 2], 0.5)
        self.assertEqual(distances[1, 1], 0.0)
        self.assertTrue(np.allclose(distances, distances.T))

    def test_pam_jaccard_separates_two_binary_patterns(self):
        _, patterns = encode_bond_patterns([{"a"}, {"a"}, {"b"}, {"b"}])
        report = pam_jaccard(patterns, 2)
        self.assertEqual(sorted(report["cluster_sizes"]), [2, 2])
        self.assertEqual(report["within_cluster_distance_sum"], 0.0)


if __name__ == "__main__":
    unittest.main()
