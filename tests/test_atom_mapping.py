import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from salsbury_md_analysis.atom_mapping import AtomMappingError, map_common_atoms
from salsbury_md_analysis.atom_mapping import decode_pdb_hybrid36, read_pdb_atoms
from salsbury_md_analysis.cli import main


def _pdb_atom(serial, name, residue, chain, residue_number, element, x=0.0):
    return (
        f"ATOM  {serial:5d} {name:>4s} {residue:>3s} {chain:1s}{residue_number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}\n"
    )


def _write_pdb(path, atoms):
    path.write_text("".join(atoms) + "END\n", encoding="utf-8")


class AtomMappingTests(unittest.TestCase):
    def test_hybrid36_pdb_identifiers_decode_across_decimal_boundaries(self):
        self.assertEqual(decode_pdb_hybrid36(" 9999", "serial"), 9999)
        self.assertEqual(decode_pdb_hybrid36("A000", "residue number"), 10000)
        self.assertEqual(decode_pdb_hybrid36("A0000", "serial"), 100000)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hybrid.pdb"
            path.write_text(
                "HETATMA0000  O   HOH 3A000      19.836  52.972  27.293  1.00  0.00           O  \nEND\n",
                encoding="utf-8",
            )
            atom = read_pdb_atoms(path)[0]
        self.assertEqual(atom.serial, 100000)
        self.assertEqual(atom.residue_number, 10000)
        self.assertEqual(atom.chain_id, "3")

    def test_vmd_lowercase_hexadecimal_identifiers_decode(self):
        self.assertEqual(decode_pdb_hybrid36("186a0", "serial"), 100000)
        self.assertEqual(decode_pdb_hybrid36("1a2b", "residue number"), 6699)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vmd-hex.pdb"
            path.write_text(
                "ATOM  186a0  OH2 TIP W1a2b      19.836  52.972  27.293  1.00  0.00           O  \nEND\n",
                encoding="utf-8",
            )
            atom = read_pdb_atoms(path)[0]
        self.assertEqual(atom.serial, 100000)
        self.assertEqual(atom.residue_number, 6699)

    def test_ca_preset_excludes_ambiguous_solvent_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atoms = [
                _pdb_atom(1, "CA", "ALA", "A", 1, "C"),
                _pdb_atom(2, "O", "HOH", "W", 1, "O"),
                _pdb_atom(3, "O", "HOH", "W", 1, "O"),
            ]
            reference = root / "reference.pdb"
            target = root / "target.pdb"
            _write_pdb(reference, atoms)
            _write_pdb(target, atoms)
            report = map_common_atoms(
                reference,
                [target],
                policy="strict",
                selection="ca",
                minimum_reference_coverage=1.0,
            )
        self.assertEqual(report["common_atom_count"], 1)
        self.assertEqual(report["selection"], "ca")
    def test_position_policy_maps_mutation_and_strict_policy_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pdb"
            target = root / "target.pdb"
            _write_pdb(reference, [
                _pdb_atom(1, "CA", "ALA", "A", 1, "C"),
                _pdb_atom(2, "CA", "GLY", "A", 2, "C"),
            ])
            _write_pdb(target, [
                _pdb_atom(1, "CA", "VAL", "A", 1, "C"),
                _pdb_atom(2, "CA", "GLY", "A", 2, "C"),
            ])
            position = map_common_atoms(reference, [target], "position", "all", 1.0)
            self.assertEqual(position["technical_status"], "complete")
            self.assertEqual(position["common_atom_count"], 2)
            self.assertEqual(position["warning_count"], 1)
            strict = map_common_atoms(reference, [target], "strict", "all", 1.0)
            self.assertEqual(strict["technical_status"], "failed")
            self.assertEqual(strict["common_atom_count"], 1)
            self.assertEqual(strict["reference_coverage"], 0.5)

    def test_signature_is_deterministic_and_hashes_are_optional(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pdb"
            target = root / "target.pdb"
            atoms = [_pdb_atom(1, "CA", "ALA", "A", 1, "C")]
            _write_pdb(reference, atoms)
            _write_pdb(target, atoms)
            first = map_common_atoms(reference, [target], "strict", "all", 1.0)
            second = map_common_atoms(reference, [target], "strict", "all", 1.0)
            self.assertEqual(first["mapping_signature_sha256"], second["mapping_signature_sha256"])
            self.assertIsNone(first["reference"]["sha256"])
            hashed = map_common_atoms(reference, [target], "strict", "all", 1.0, True)
            self.assertEqual(len(hashed["reference"]["sha256"]), 64)

    def test_selection_and_multi_target_intersection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pdb"
            target_one = root / "target-one.pdb"
            target_two = root / "target-two.pdb"
            atoms = [
                _pdb_atom(1, "N", "ALA", "A", 1, "N"),
                _pdb_atom(2, "CA", "ALA", "A", 1, "C"),
                _pdb_atom(3, "CB", "ALA", "A", 1, "C"),
                _pdb_atom(4, "H", "ALA", "A", 1, "H"),
            ]
            _write_pdb(reference, atoms)
            _write_pdb(target_one, atoms)
            _write_pdb(target_two, atoms[:2])
            backbone = map_common_atoms(
                reference, [target_one, target_two], "strict", "backbone", 1.0
            )
            self.assertEqual(backbone["technical_status"], "complete")
            self.assertEqual(backbone["common_atom_count"], 2)
            heavy = map_common_atoms(reference, [target_one], "strict", "heavy", 1.0)
            self.assertEqual(heavy["common_atom_count"], 3)
            all_atoms = map_common_atoms(reference, [target_one, target_two], "strict", "all", 1.0)
            self.assertEqual(all_atoms["technical_status"], "failed")
            self.assertEqual(len(all_atoms["excluded_reference_atoms"]), 2)

    def test_duplicate_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pdb"
            target = root / "target.pdb"
            duplicate = [
                _pdb_atom(1, "CA", "ALA", "A", 1, "C"),
                _pdb_atom(2, "CA", "ALA", "A", 1, "C"),
            ]
            _write_pdb(reference, duplicate)
            _write_pdb(target, duplicate[:1])
            with self.assertRaises(AtomMappingError) as context:
                map_common_atoms(reference, [target], "strict", "all", 0.0)
            self.assertIn("duplicate identities", str(context.exception))

    def test_gro_topologies_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.gro"
            target = root / "target.gro"
            content = "synthetic\n1\n    1ALA     CA    1   0.000   0.000   0.000\n1.0 1.0 1.0\n"
            reference.write_text(content, encoding="utf-8")
            target.write_text(content, encoding="utf-8")
            report = map_common_atoms(reference, [target], "strict", "all", 1.0)
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["common_atom_count"], 1)

    def test_cli_returns_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pdb"
            target = root / "target.pdb"
            atoms = [_pdb_atom(1, "CA", "ALA", "A", 1, "C")]
            _write_pdb(reference, atoms)
            _write_pdb(target, atoms)
            output = io.StringIO()
            with redirect_stdout(output):
                status = main([
                    "map-common-atoms", str(reference), str(target),
                    "--policy", "strict", "--selection", "all",
                    "--minimum-reference-coverage", "1.0", "--hash-content",
                ])
            report = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(report["scientific_status"], "not evaluated")
            self.assertEqual(report["common_atom_count"], 1)


if __name__ == "__main__":
    unittest.main()
