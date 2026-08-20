import unittest

import numpy as np

from salsbury_md_analysis.representative_structures import representative_structures


class RepresentativeStructureTests(unittest.TestCase):
    def test_observed_representatives_and_mean_remain_distinct(self):
        frames = np.asarray([
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[9.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        ])
        report = representative_structures(frames)
        self.assertEqual(report["closest_to_mean_frame_index"], 1)
        self.assertEqual(report["medoid_frame_index"], 1)
        self.assertTrue(np.allclose(report["arithmetic_mean_coordinates"], [[10.0 / 3, 0, 0], [13.0 / 3, 0, 0]]))
        self.assertIn(1, report["central_frame_indices"])


if __name__ == "__main__":
    unittest.main()
