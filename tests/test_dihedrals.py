import unittest

from salsbury_md_analysis.atom_mapping import AtomRecord
from salsbury_md_analysis.dihedrals import _torsion_specs, circular_summary, dihedral_degrees


class DihedralTests(unittest.TestCase):
    def test_signed_dihedral_and_circular_wrap(self):
        angle = dihedral_degrees(
            (1.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),
        )
        self.assertAlmostEqual(abs(angle), 90.0)
        summary = circular_summary([179.0, -179.0], 36)
        self.assertGreater(abs(summary["mean_angle_degrees"]), 178.0)
        self.assertGreater(summary["mean_resultant_length"], 0.99)
        self.assertEqual(sum(row["count"] for row in summary["histogram"]), 2)

    def test_peptide_omega_convention_distinguishes_cis_and_trans(self):
        cis = dihedral_degrees(
            (0.0, 1.0, 0.0), (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
        )
        trans = dihedral_degrees(
            (0.0, 1.0, 0.0), (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0), (1.0, -1.0, 0.0),
        )
        self.assertAlmostEqual(cis, 0.0)
        self.assertAlmostEqual(abs(trans), 180.0)

    def test_arg_sidechain_defines_all_five_conventional_chis(self):
        names = ["N", "CA", "C", "CB", "CG", "CD", "NE", "CZ", "NH1"]
        atoms = [
            AtomRecord(index, index + 1, name, "", "ARG", "A", 1, "", name[0])
            for index, name in enumerate(names)
        ]
        coordinates = [(float(index), float(index % 2), float(index % 3)) for index in range(len(names))]
        settings = {
            "angle_types": ["chi1", "chi2", "chi3", "chi4", "chi5"],
            "maximum_reference_peptide_bond_angstrom": 2.0,
        }
        specs = _torsion_specs(atoms, coordinates, settings)
        self.assertEqual([spec["angle_type"] for spec in specs], ["chi1", "chi2", "chi3", "chi4", "chi5"])

    def test_dna_backbone_glycosidic_and_sugar_torsions_are_defined(self):
        atom_rows = []
        for residue_number, residue_name in ((1, "DG"), (2, "DC")):
            names = [
                "P", "O5'", "C5'", "C4'", "C3'", "O3'", "O4'", "C1'", "C2'",
                *(("N9", "C4") if residue_name == "DG" else ("N1", "C2")),
            ]
            for name in names:
                atom_rows.append((residue_number, residue_name, name))
        atoms = [
            AtomRecord(
                index, index + 1, name, "", residue_name, "D", residue_number,
                "", name[0],
            )
            for index, (residue_number, residue_name, name) in enumerate(atom_rows)
        ]
        coordinates = []
        for index, (residue_number, _, name) in enumerate(atom_rows):
            if name == "O3'" and residue_number == 1:
                coordinates.append((0.0, 0.0, 0.0))
            elif name == "P" and residue_number == 2:
                coordinates.append((1.6, 0.0, 0.0))
            else:
                coordinates.append((float(index), float(index % 3), float((index * 2) % 5)))
        settings = {
            "angle_types": [
                "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                "chi", "nu0", "nu1", "nu2", "nu3", "nu4",
            ],
            "maximum_reference_peptide_bond_angstrom": 1.8,
            "maximum_reference_phosphodiester_bond_angstrom": 2.2,
        }
        specs = _torsion_specs(atoms, coordinates, settings)
        angle_types = {spec["angle_type"] for spec in specs}
        self.assertTrue({"alpha", "beta", "gamma", "delta", "epsilon", "zeta"}.issubset(angle_types))
        self.assertTrue({"chi", "nu0", "nu1", "nu2", "nu3", "nu4"}.issubset(angle_types))


if __name__ == "__main__":
    unittest.main()
