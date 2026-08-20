import unittest

from salsbury_md_analysis.atom_mapping import AtomRecord
from salsbury_md_analysis.conformational_views import (
    plan_comparative_conformational_views,
    plan_conformational_views,
)


def atom(index, name, residue, chain, number, element):
    return AtomRecord(
        atom_index=index,
        serial=index + 1,
        atom_name=name,
        altloc="",
        residue_name=residue,
        chain_id=chain,
        residue_number=number,
        insertion_code="",
        element=element,
    )


class ConformationalViewTests(unittest.TestCase):
    def test_canonical_rna_names_are_not_misclassified_as_modifications(self):
        atoms = []
        coordinates = []
        for number, residue in enumerate(("RA", "RC", "RG", "RU"), start=1):
            base = len(atoms)
            atoms.extend([
                atom(base, "P", residue, "R", number, "P"),
                atom(base + 1, "C1'", residue, "R", number, "C"),
                atom(base + 2, "O4'", residue, "R", number, "O"),
            ])
            coordinates.extend([
                (float(number), 0.0, 0.0),
                (float(number), 1.0, 0.0),
                (float(number), 2.0, 0.0),
            ])

        report = plan_conformational_views(atoms, coordinates)

        self.assertEqual(report["system_classification"], "nucleic_acid_only")
        self.assertEqual(report["composition"]["modified_nucleotide_count"], 0)

    def test_comparative_plan_harmonizes_modified_position_from_lesion(self):
        def reference(modified):
            atoms = [
                atom(0, "N", "ALA", "P", 1, "N"),
                atom(1, "CA", "ALA", "P", 1, "C"),
                atom(2, "C", "ALA", "P", 1, "C"),
                atom(3, "O", "ALA", "P", 1, "O"),
            ]
            coordinates = [
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0), (2.0, 1.0, 0.0),
            ]
            for number in (10, 11, 12, 13):
                residue = "8OG" if modified and number == 11 else "DG"
                base = len(atoms)
                atoms.extend([
                    atom(base, "P", residue, "D", number, "P"),
                    atom(base + 1, "C1'", residue, "D", number, "C"),
                    atom(base + 2, "O4'", residue, "D", number, "O"),
                ])
                x = 2.0 if number < 13 else 30.0
                coordinates.extend([(x, 3.0, 0.0), (x, 4.0, 0.0), (x, 5.0, 0.0)])
            return atoms, coordinates

        control = reference(False)
        lesion = reference(True)
        report = plan_comparative_conformational_views([
            ("control", *control),
            ("lesion", *lesion),
        ])
        views = {view["view_id"]: view for view in report["views"]}
        interface = views["chemical_interface"]
        self.assertEqual(
            [row["residue_number"] for row in interface["focus_nucleic_residue_keys"]],
            [10, 11, 12],
        )
        self.assertEqual(
            interface["modified_nucleotide_residue_keys"],
            [{"chain_id": "D", "residue_number": 11, "insertion_code": ""}],
        )
        self.assertEqual(interface["common_atom_count"], 13)
        self.assertNotIn(
            13,
            [row["residue_number"] for row in interface["selection"]["residue_keys"]],
        )

    def test_protein_only_gets_global_and_trace_views(self):
        atoms = [
            atom(0, "N", "ALA", "A", 1, "N"),
            atom(1, "CA", "ALA", "A", 1, "C"),
            atom(2, "C", "ALA", "A", 1, "C"),
            atom(3, "O", "ALA", "A", 1, "O"),
            atom(4, "CB", "ALA", "A", 1, "C"),
        ]
        report = plan_conformational_views(
            atoms, [(float(index), 0.0, 0.0) for index in range(len(atoms))]
        )
        self.assertEqual(report["system_classification"], "protein_only")
        self.assertEqual(
            [view["view_id"] for view in report["views"]],
            ["global_common_heavy", "macromolecular_trace"],
        )

    def test_nucleic_acid_ligand_records_both_chemical_and_macromolecular_class(self):
        atoms = [
            atom(0, "P", "DG", "D", 1, "P"),
            atom(1, "C1'", "DG", "D", 1, "C"),
            atom(2, "O4'", "DG", "D", 1, "O"),
            atom(3, "C1", "LIG", "L", 1, "C"),
            atom(4, "O1", "LIG", "L", 1, "O"),
        ]
        report = plan_conformational_views(
            atoms, [(float(index), 0.0, 0.0) for index in range(len(atoms))]
        )
        self.assertEqual(
            report["system_classification"],
            "nucleic_acid_other_solute_complex",
        )
        self.assertEqual(
            report["macromolecular_classification"], "nucleic_acid_only"
        )
        self.assertEqual(report["composition"]["other_solute_residue_count"], 1)

    def test_ligand_bound_and_apo_proteins_share_comparative_basis(self):
        apo_atoms = [
            atom(0, "N", "ALA", "A", 1, "N"),
            atom(1, "CA", "ALA", "A", 1, "C"),
            atom(2, "C", "ALA", "A", 1, "C"),
            atom(3, "O", "ALA", "A", 1, "O"),
        ]
        bound_atoms = apo_atoms + [
            atom(4, "C1", "LIG", "L", 1, "C"),
            atom(5, "O1", "LIG", "L", 1, "O"),
        ]
        apo_coordinates = [
            (float(index), 0.0, 0.0) for index in range(len(apo_atoms))
        ]
        bound_coordinates = apo_coordinates + [(5.0, 0.0, 0.0), (6.0, 0.0, 0.0)]

        report = plan_comparative_conformational_views([
            ("apo", apo_atoms, apo_coordinates),
            ("bound", bound_atoms, bound_coordinates),
        ])

        self.assertEqual(report["system_classification"], "protein_only")
        self.assertEqual(
            report["system_classifications_by_reference"],
            {
                "apo": "protein_only",
                "bound": "protein_other_solute_complex",
            },
        )

    def test_modified_dna_complex_gets_outcome_independent_interface_view(self):
        atoms = [
            atom(0, "N", "ALA", "P", 1, "N"),
            atom(1, "CA", "ALA", "P", 1, "C"),
            atom(2, "C", "ALA", "P", 1, "C"),
            atom(3, "O", "ALA", "P", 1, "O"),
            atom(4, "CB", "ALA", "P", 1, "C"),
            atom(5, "N", "GLY", "P", 2, "N"),
            atom(6, "CA", "GLY", "P", 2, "C"),
            atom(7, "C", "GLY", "P", 2, "C"),
            atom(8, "O", "GLY", "P", 2, "O"),
            atom(9, "P", "DG", "D", 10, "P"),
            atom(10, "C1'", "DG", "D", 10, "C"),
            atom(11, "O4'", "DG", "D", 10, "O"),
            atom(12, "P", "8OG", "D", 11, "P"),
            atom(13, "C1'", "8OG", "D", 11, "C"),
            atom(14, "O4'", "8OG", "D", 11, "O"),
            atom(15, "P", "DC", "D", 12, "P"),
            atom(16, "C1'", "DC", "D", 12, "C"),
            atom(17, "O4'", "DC", "D", 12, "O"),
            atom(18, "MG", "MG", "I", 1, "MG"),
        ]
        coordinates = [
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0), (1.0, 1.0, 0.0),
            (30.0, 0.0, 0.0), (31.0, 0.0, 0.0), (32.0, 0.0, 0.0),
            (32.0, 1.0, 0.0),
            (2.0, 3.0, 0.0), (2.0, 4.0, 0.0), (2.0, 5.0, 0.0),
            (3.0, 3.0, 0.0), (3.0, 4.0, 0.0), (3.0, 5.0, 0.0),
            (4.0, 3.0, 0.0), (4.0, 4.0, 0.0), (4.0, 5.0, 0.0),
            (3.0, 4.0, 1.0),
        ]
        report = plan_conformational_views(atoms, coordinates)
        self.assertEqual(
            report["system_classification"], "protein_nucleic_acid_complex"
        )
        views = {view["view_id"]: view for view in report["views"]}
        self.assertEqual(
            set(views),
            {"global_common_heavy", "chemical_interface", "macromolecular_trace"},
        )
        interface = views["chemical_interface"]
        self.assertEqual(
            interface["modified_nucleotide_residue_keys"],
            [{"chain_id": "D", "residue_number": 11, "insertion_code": ""}],
        )
        self.assertEqual(len(interface["focus_nucleic_residue_keys"]), 3)
        self.assertEqual(
            interface["contacting_protein_residue_keys"],
            [{"chain_id": "P", "residue_number": 1, "insertion_code": ""}],
        )
        self.assertIn("ions are excluded", interface["bound_ion_policy"])

    def test_equivalent_protein_dna_dimer_gets_symmetry_expanded_member_view(self):
        atoms = []
        coordinates = []
        for member_index, (protein_chain, dna_chain, offset) in enumerate(
            (("A", "C", 0.0), ("B", "D", 30.0))
        ):
            base = len(atoms)
            atoms.extend([
                atom(base, "N", "ALA", protein_chain, 1, "N"),
                atom(base + 1, "CA", "ALA", protein_chain, 1, "C"),
                atom(base + 2, "C", "ALA", protein_chain, 1, "C"),
                atom(base + 3, "O", "ALA", protein_chain, 1, "O"),
                atom(base + 4, "CB", "ALA", protein_chain, 1, "C"),
                atom(base + 5, "P", "DG", dna_chain, 10, "P"),
                atom(base + 6, "C1'", "DG", dna_chain, 10, "C"),
                atom(base + 7, "O4'", "DG", dna_chain, 10, "O"),
            ])
            coordinates.extend([
                (offset, 0.0, 0.0), (offset + 1.0, 0.0, 0.0),
                (offset + 2.0, 0.0, 0.0), (offset + 2.0, 1.0, 0.0),
                (offset + 1.0, 1.0, 0.0), (offset + 2.0, 3.0, 0.0),
                (offset + 2.0, 4.0, 0.0), (offset + 2.0, 5.0, 0.0),
            ])
        report = plan_conformational_views(atoms, coordinates)
        views = {view["view_id"]: view for view in report["views"]}
        self.assertIn("oligomer_member_common_heavy", views)
        self.assertIn("oligomer_member_interface_common_heavy", views)
        oligomer = views["oligomer_member_common_heavy"]["symmetry_expansion"]
        self.assertEqual(oligomer["member_count"], 2)
        self.assertEqual(oligomer["analysis_atom_count_per_member"], 8)
        self.assertEqual(
            oligomer["observation_contract"]["member_observation_multiplier"], 2
        )
        self.assertFalse(
            oligomer["observation_contract"]["member_observations_are_independent_replicas"]
        )
        self.assertEqual(
            [row["nucleic_chain_ids"] for row in oligomer["members"]],
            [["C"], ["D"]],
        )
        interface = views["oligomer_member_interface_common_heavy"]
        self.assertFalse(
            interface["observation_contract"][
                "member_observations_are_independent_replicas"
            ]
        )


if __name__ == "__main__":
    unittest.main()
