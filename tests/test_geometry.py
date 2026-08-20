import math
import unittest

from salsbury_md_analysis.geometry import (
    GeometryError,
    apply_transform,
    best_fit_transform,
    distance3,
    mass_weighted_radius_of_gyration,
    rmsd,
)


class GeometryTests(unittest.TestCase):
    def test_quaternion_fit_removes_known_rotation_and_translation(self):
        mobile = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        reference = [(3, -2, 5), (3, -1, 5), (2, -2, 5), (3, -2, 6)]
        transform = best_fit_transform(mobile, reference)
        fitted = apply_transform(mobile, transform)
        self.assertLess(transform.fitted_rmsd_angstrom, 1.0e-12)
        self.assertLess(rmsd(fitted, reference), 1.0e-12)

    def test_one_atom_fit_is_translation_only(self):
        transform = best_fit_transform([(4, 5, 6)], [(1, 2, 3)])
        self.assertEqual(apply_transform([(4, 5, 6)], transform), ((1.0, 2.0, 3.0),))
        self.assertEqual(transform.fitted_rmsd_angstrom, 0.0)

    def test_mass_weighted_radius_of_gyration(self):
        value = mass_weighted_radius_of_gyration([(-1, 0, 0), (1, 0, 0)], [12, 12])
        self.assertAlmostEqual(value, 1.0)
        unequal = mass_weighted_radius_of_gyration([(0, 0, 0), (2, 0, 0)], [1, 3])
        self.assertAlmostEqual(unequal, math.sqrt(0.75))

    def test_three_dimensional_distance_is_shared_across_analyses(self):
        self.assertEqual(distance3((1, 2, 3), (4, 6, 3)), 5.0)

    def test_nonfinite_coordinates_fail_closed(self):
        with self.assertRaises(GeometryError):
            best_fit_transform([(float("nan"), 0, 0)], [(0, 0, 0)])


if __name__ == "__main__":
    unittest.main()
