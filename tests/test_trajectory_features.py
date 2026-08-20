import unittest

import numpy as np

from salsbury_md_analysis.trajectory_features import (
    TrajectoryFeatureError,
    center_of_mass,
    center_of_geometry,
    flatten_coordinates,
    group_distance_statistics,
    minimum_mean_group_distance,
    principal_axes,
)


class TrajectoryFeatureTests(unittest.TestCase):
    def test_cartesian_and_center_of_mass_features_match_direct_definitions(self):
        coordinates = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0)]
        self.assertEqual(
            flatten_coordinates(coordinates, [2, 0]),
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(center_of_mass(coordinates, [1.0, 3.0, 2.0], [0, 1]), (1.5, 0.0, 0.0))
        self.assertEqual(center_of_geometry(coordinates, [0, 1]), (1.0, 0.0, 0.0))

    def test_group_distance_families_remain_distinct(self):
        coordinates = [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
        ]
        statistics = group_distance_statistics(coordinates, [0, 1], [2, 3])
        self.assertAlmostEqual(statistics["minimum_distance_angstrom"], 1.0)
        self.assertAlmostEqual(statistics["mean_distance_angstrom"], 2.5)
        self.assertAlmostEqual(statistics["maximum_distance_angstrom"], 5.0)
        self.assertEqual(statistics["closest_atom_indices"], [0, 2])
        self.assertEqual(statistics["farthest_atom_indices"], [0, 3])

        minimum_mean = minimum_mean_group_distance(coordinates, [0, 1], [2, 3])
        self.assertAlmostEqual(minimum_mean["minimum_mean_distance_angstrom"], 1.0)
        self.assertEqual(minimum_mean["selected_candidate_atom_index"], 2)

    def test_principal_axes_are_orthonormal_right_handed_and_translation_invariant(self):
        coordinates = [(-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        translated = [(x + 7.0, y - 4.0, z + 2.0) for x, y, z in coordinates]
        first = principal_axes(coordinates, [12.0, 12.0, 12.0], [0, 1, 2])
        second = principal_axes(translated, [12.0, 12.0, 12.0], [0, 1, 2])
        axes = np.asarray(first["principal_axes"])
        self.assertTrue(np.allclose(axes @ axes.T, np.eye(3), atol=1.0e-12))
        self.assertGreater(np.linalg.det(axes), 0.0)
        self.assertTrue(np.allclose(first["principal_moments"], second["principal_moments"]))
        self.assertTrue(np.allclose(np.abs(first["principal_axes"]), np.abs(second["principal_axes"])))

    def test_empty_and_too_small_selections_fail_closed(self):
        with self.assertRaises(TrajectoryFeatureError):
            flatten_coordinates([(0.0, 0.0, 0.0)], [])
        with self.assertRaises(TrajectoryFeatureError):
            principal_axes([(0.0, 0.0, 0.0)] * 3, [1.0] * 3, [0, 1])


if __name__ == "__main__":
    unittest.main()
