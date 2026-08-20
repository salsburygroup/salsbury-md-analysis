import unittest

from pathlib import Path

from salsbury_md_analysis.atom_mapping import AtomRecord
from salsbury_md_analysis.secondary_structure import (
    _frame_pdb_payload,
    _protein_residue_keys,
    build_mkdssp_command,
    parse_dssp_text,
)


class SecondaryStructureTests(unittest.TestCase):
    def test_mixed_protein_dna_filter_retains_only_protein_residues(self):
        atoms = [
            AtomRecord(0, 1, "N", "", "ALA", "A", 1, "", "N"),
            AtomRecord(1, 2, "CA", "", "ALA", "A", 1, "", "C"),
            AtomRecord(2, 3, "C", "", "ALA", "A", 1, "", "C"),
            AtomRecord(3, 4, "N1", "", "DG", "D", 2, "", "N"),
            AtomRecord(4, 5, "C1'", "", "DG", "D", 2, "", "C"),
        ]
        self.assertEqual(
            _protein_residue_keys(atoms),
            {("A", 1, "", "ALA")},
        )

    def test_dssp_pdb_normalization_is_reversible_and_excludes_heteroatoms(self):
        lines = [
            "ATOM  A0000  CA  THR L   1H      0.000   0.000   0.000  1.00  0.00           C  ",
            "HETATM00002  O   HOH L9999       1.000   1.000   1.000  1.00  0.00           O  ",
        ]
        atoms = [
            AtomRecord(0, 100000, "CA", "", "THR", "L", 1, "H", "C"),
            AtomRecord(1, 2, "O", "", "HOH", "L", 9999, "", "O"),
        ]
        text, mapping = _frame_pdb_payload(
            lines, atoms, [(2.0, 3.0, 4.0), (5.0, 6.0, 7.0)]
        )
        atom_lines = [line for line in text.splitlines() if line.startswith("ATOM")]
        self.assertEqual(len(atom_lines), 1)
        self.assertEqual(atom_lines[0][6:11], "    1")
        self.assertEqual(atom_lines[0][22:26], "   1")
        self.assertEqual(atom_lines[0][26:27], " ")
        self.assertEqual(mapping[("L", "1")]["original_residue_token"], "1H")

    def test_dssp_pdb_normalization_populates_missing_element_column(self):
        lines = [
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00"
        ]
        atoms = [AtomRecord(0, 1, "N", "", "ALA", "A", 1, "", "N")]
        text, _ = _frame_pdb_payload(lines, atoms, [(1.0, 2.0, 3.0)])
        atom_line = next(line for line in text.splitlines() if line.startswith("ATOM"))
        self.assertGreaterEqual(len(atom_line), 78)
        self.assertEqual(atom_line[76:78], " N")

    def test_builds_current_mkdssp_positional_command(self):
        self.assertEqual(
            build_mkdssp_command(
                "/opt/bin/mkdssp", Path("input.pdb"), Path("output.dssp"),
                "mkdssp version 4.6.1",
            ),
            [
                "/opt/bin/mkdssp", "--output-format", "dssp",
                "input.pdb", "output.dssp",
            ],
        )

    def test_builds_legacy_mkdssp_command(self):
        self.assertEqual(
            build_mkdssp_command(
                "mkdssp", Path("input.pdb"), Path("output.dssp"),
                "mkdssp version 3.0.0",
            ),
            ["mkdssp", "-i", "input.pdb", "-o", "output.dssp"],
        )

    def test_classic_dssp_table_parser(self):
        text = (
            "HEADER\n"
            "  #  RESIDUE AA STRUCTURE\n"
            "    1    1 A A  H              0   0\n"
            "    2    2 A G                 0   0\n"
        )
        rows = parse_dssp_text(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["secondary_structure_code"], "H")
        self.assertEqual(rows[1]["secondary_structure_code"], "C")

    def test_dssp_46_ppii_code_is_preserved(self):
        text = (
            "  #  RESIDUE AA STRUCTURE\n"
            "    1    1 A A  P              0   0\n"
        )
        rows = parse_dssp_text(text)
        self.assertEqual(rows[0]["secondary_structure_code"], "P")


if __name__ == "__main__":
    unittest.main()
