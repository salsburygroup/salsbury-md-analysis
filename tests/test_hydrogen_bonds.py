import unittest

from salsbury_md_analysis.hydrogen_bonds import hydrogen_bond_present


class HydrogenBondTests(unittest.TestCase):
    def test_distance_and_angle_definition(self):
        present, distance, angle = hydrogen_bond_present(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.8, 0.0, 0.0),
            maximum_distance=3.5, minimum_angle=150.0,
        )
        self.assertTrue(present)
        self.assertAlmostEqual(distance, 2.8)
        self.assertAlmostEqual(angle, 180.0)
        absent, _, bent_angle = hydrogen_bond_present(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0),
            maximum_distance=3.5, minimum_angle=150.0,
        )
        self.assertFalse(absent)
        self.assertAlmostEqual(bent_angle, 90.0)

    def test_periodic_geometry_reimages_acceptor(self):
        cell = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))
        present, distance, angle = hydrogen_bond_present(
            (8.0, 0.0, 0.0), (9.0, 0.0, 0.0), (0.8, 0.0, 0.0),
            maximum_distance=3.5, minimum_angle=150.0, cell=cell,
        )
        self.assertTrue(present)
        self.assertAlmostEqual(distance, 2.8)
        self.assertAlmostEqual(angle, 180.0)


if __name__ == "__main__":
    unittest.main()
