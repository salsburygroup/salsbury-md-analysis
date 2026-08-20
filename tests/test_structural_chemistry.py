import unittest

import numpy as np

from salsbury_md_analysis.atom_mapping import AtomRecord
from salsbury_md_analysis.structural_chemistry import chemical_integrity_snapshot, reference_chirality_signs


def atom(index, name, residue, residue_number, element):
    return AtomRecord(index, index + 1, name, "", residue, "A", residue_number, "", element)


class StructuralChemistryTests(unittest.TestCase):
    def test_peptide_link_chirality_clash_and_declared_link_are_reported(self):
        atoms = [
            atom(0, "N", "ALA", 1, "N"), atom(1, "CA", "ALA", 1, "C"),
            atom(2, "C", "ALA", 1, "C"), atom(3, "CB", "ALA", 1, "C"),
            atom(4, "N", "GLY", 2, "N"), atom(5, "CA", "GLY", 2, "C"),
            atom(6, "SG", "CYS", 3, "S"), atom(7, "SG", "CYS", 4, "S"),
        ]
        reference = [(0, 0, 0), (1, 0, 0), (2, 0.5, 0), (1, 1, 1), (3.2, 0.5, 0), (4.2, 0.5, 0), (10, 0, 0), (12, 0, 0)]
        signs = reference_chirality_signs(atoms, reference)
        frame = list(reference)
        frame[3] = (1, 1, -1)
        frame[4] = (6, 0.5, 0)
        report = chemical_integrity_snapshot(
            atoms, frame,
            maximum_peptide_bond_angstrom=1.8,
            maximum_trans_omega_deviation_degrees=40.0,
            minimum_ca_chirality_volume_angstrom3=0.1,
            steric_clash_scale=0.5,
            reference_chirality=signs,
            covalent_bonds=[(2, 4), (6, 7)],
            declared_covalent_links=[{"link_id": "disulfide", "atom_indices": [6, 7], "minimum_distance_angstrom": 1.9, "maximum_distance_angstrom": 2.2}],
        )
        self.assertEqual(report["peptide_break_count"], 1)
        self.assertEqual(report["chirality_outlier_count"], 1)
        self.assertEqual(report["declared_covalent_link_outlier_count"], 0)

    def test_kdtree_clash_search_matches_reference_pair_rules(self):
        atoms = [
            atom(0, "C1", "A", 1, "C"),
            atom(1, "C2", "A", 1, "C"),
            atom(2, "O1", "B", 2, "O"),
            atom(3, "N1", "C", 3, "N"),
        ]
        coordinates = [(0, 0, 0), (0.2, 0, 0), (1.0, 0, 0), (5.0, 0, 0)]
        report = chemical_integrity_snapshot(
            atoms, coordinates,
            maximum_peptide_bond_angstrom=2.0,
            maximum_trans_omega_deviation_degrees=45.0,
            minimum_ca_chirality_volume_angstrom3=0.01,
            steric_clash_scale=0.5,
            reference_chirality={},
            covalent_bonds=[(0, 1)],
        )
        self.assertEqual(report["steric_clash_count"], 2)
        self.assertEqual(
            [row["atom_indices"] for row in report["steric_clash_examples"]],
            [[0, 2], [1, 2]],
        )

    def test_residue_order_or_shared_chain_does_not_invent_peptide_bond(self):
        atoms = [
            atom(0, "CA", "GLN", 633, "C"), atom(1, "C", "GLN", 633, "C"),
            atom(2, "N", "GLU", 641, "N"), atom(3, "CA", "GLU", 641, "C"),
        ]
        coordinates = [(0, 0, 0), (1, 0, 0), (20, 0, 0), (21, 0, 0)]
        report = chemical_integrity_snapshot(
            atoms, coordinates,
            maximum_peptide_bond_angstrom=1.8,
            maximum_trans_omega_deviation_degrees=40.0,
            minimum_ca_chirality_volume_angstrom3=0.1,
            steric_clash_scale=0.5,
            reference_chirality={},
            covalent_bonds=[],
        )
        self.assertEqual(report["peptide_link_count"], 0)
        self.assertEqual(report["peptide_break_count"], 0)
        self.assertEqual(report["omega_outlier_count"], 0)

    def test_explicit_cn_bond_defines_peptide_even_across_numbering_gap(self):
        atoms = [
            atom(0, "CA", "GLN", 633, "C"), atom(1, "C", "GLN", 633, "C"),
            atom(2, "N", "GLU", 641, "N"), atom(3, "CA", "GLU", 641, "C"),
        ]
        coordinates = [(0, 0, 0), (1, 0, 0), (20, 0, 0), (21, 0, 0)]
        report = chemical_integrity_snapshot(
            atoms, coordinates,
            maximum_peptide_bond_angstrom=1.8,
            maximum_trans_omega_deviation_degrees=40.0,
            minimum_ca_chirality_volume_angstrom3=0.1,
            steric_clash_scale=0.5,
            reference_chirality={},
            covalent_bonds=[(1, 2)],
        )
        self.assertEqual(report["peptide_link_count"], 1)
        self.assertEqual(report["peptide_break_count"], 1)

    def test_nonfinite_chemical_coordinates_fail_closed(self):
        atoms = [atom(0, "C", "A", 1, "C")]
        with self.assertRaisesRegex(ValueError, "finite atom-by-three"):
            chemical_integrity_snapshot(
                atoms, [(np.nan, 0, 0)],
                maximum_peptide_bond_angstrom=2.0,
                maximum_trans_omega_deviation_degrees=45.0,
                minimum_ca_chirality_volume_angstrom3=0.01,
                steric_clash_scale=0.5,
                reference_chirality={},
            )


if __name__ == "__main__":
    unittest.main()
