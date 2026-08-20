import unittest

from salsbury_md_analysis.atom_mapping import AtomMappingError, AtomRecord
from salsbury_md_analysis.selections import (
    build_common_correspondences,
    build_correspondence,
    select_atoms,
)


def _atom(index, name, residue="ALA", element=None):
    return AtomRecord(
        atom_index=index,
        serial=index + 1,
        atom_name=name,
        altloc="",
        residue_name=residue,
        chain_id="A",
        residue_number=1,
        insertion_code="",
        element=element or name[0],
    )


class SelectionTests(unittest.TestCase):
    def test_portable_presets_and_exact_names(self):
        atoms = [
            _atom(0, "N"), _atom(1, "CA"), _atom(2, "CB"), _atom(3, "H"),
            _atom(4, "O", residue="HOH"), _atom(5, "H1", residue="HOH"),
            _atom(6, "OH2", residue="TIP"), _atom(7, "H1", residue="TIP"),
            _atom(8, "C1'", residue="DG"), _atom(9, "P", residue="DG"),
        ]
        self.assertEqual(
            [atom.atom_name for atom in select_atoms(atoms, {"preset": "backbone"}, "fit")],
            ["N", "CA"],
        )
        self.assertEqual(
            [atom.atom_name for atom in select_atoms(atoms, {"preset": "heavy"}, "heavy")],
            ["N", "CA", "CB", "O", "OH2", "C1'", "P"],
        )
        self.assertEqual(
            [atom.atom_name for atom in select_atoms(atoms, {"preset": "solute_heavy"}, "solute")],
            ["N", "CA", "CB", "C1'", "P"],
        )
        self.assertEqual(
            [atom.atom_name for atom in select_atoms(atoms, {"preset": "complex_trace"}, "trace")],
            ["CA", "C1'"],
        )
        self.assertEqual(
            [atom.atom_name for atom in select_atoms(atoms, {"preset": "macromolecular_backbone"}, "macro")],
            ["N", "CA", "C1'", "P"],
        )
        self.assertEqual(
            [atom.atom_name for atom in select_atoms(atoms, {"atom_names": ["ca"]}, "named")],
            ["CA"],
        )
        self.assertEqual(
            [
                atom.atom_name
                for atom in select_atoms(
                    atoms,
                    {
                        "residue_keys": [
                            {"chain_id": "A", "residue_number": 1, "insertion_code": ""}
                        ],
                        "heavy_only": True,
                    },
                    "residue-heavy",
                )
            ],
            ["N", "CA", "CB", "O", "OH2", "C1'", "P"],
        )

    def test_generic_solute_and_payload_treat_common_water_and_ions_consistently(self):
        atoms = [
            _atom(0, "CA", residue="ALA", element="C"),
            _atom(1, "O", residue="OPC", element="O"),
            _atom(2, "FE", residue="FE2", element="FE"),
            _atom(3, "MN", residue="MN2", element="MN"),
            _atom(4, "C1", residue="LIG", element="C"),
        ]

        self.assertEqual(
            [atom.residue_name for atom in select_atoms(
                atoms, {"preset": "solute_heavy"}, "solute"
            )],
            ["ALA", "LIG"],
        )
        self.assertEqual(
            [atom.residue_name for atom in select_atoms(
                atoms, {"preset": "molecular_payload"}, "payload"
            )],
            ["ALA", "FE2", "MN2", "LIG"],
        )

    def test_correspondence_is_reference_ordered_and_signed(self):
        reference = [_atom(0, "N"), _atom(1, "CA"), _atom(2, "C")]
        target = [_atom(0, "C"), _atom(1, "N"), _atom(2, "CA")]
        mapping = build_correspondence(
            reference, target, {"preset": "all"}, "all", "strict", 1.0
        )
        self.assertEqual(mapping.reference_indices, (0, 1, 2))
        self.assertEqual(mapping.target_indices, (1, 2, 0))
        self.assertEqual(len(mapping.mapping_signature_sha256), 64)

    def test_coverage_gate_and_position_mutation_warning(self):
        reference = [_atom(0, "N"), _atom(1, "CA")]
        with self.assertRaises(AtomMappingError):
            build_correspondence(
                reference, [_atom(0, "N")], {"preset": "all"}, "all", "strict", 1.0
            )
        mutation = build_correspondence(
            reference,
            [_atom(0, "N", residue="GLY"), _atom(1, "CA", residue="GLY")],
            {"preset": "all"},
            "all",
            "position",
            1.0,
        )
        self.assertEqual(mutation.residue_name_mismatch_count, 2)

    def test_multi_target_mapping_uses_one_global_intersection(self):
        reference = [_atom(0, "N"), _atom(1, "CA"), _atom(2, "C")]
        target_one = [_atom(0, "N"), _atom(1, "CA")]
        target_two = [_atom(0, "CA"), _atom(1, "C")]
        mappings = build_common_correspondences(
            reference,
            [target_one, target_two],
            {"preset": "all"},
            "all",
            "strict",
            1.0 / 3.0,
        )
        self.assertEqual([mapping.reference_indices for mapping in mappings], [(1,), (1,)])
        self.assertEqual([mapping.target_indices for mapping in mappings], [(1,), (0,)])
        self.assertTrue(all(mapping.reference_coverage == 1.0 / 3.0 for mapping in mappings))
        with self.assertRaises(AtomMappingError):
            build_common_correspondences(
                reference,
                [target_one, target_two],
                {"preset": "all"},
                "all",
                "strict",
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
