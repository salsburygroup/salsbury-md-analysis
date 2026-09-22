import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.atom_mapping import AtomRecord, read_topology_atoms
from salsbury_md_analysis.coordinates import iter_coordinate_frames
from salsbury_md_analysis.secondary_structure import (
    SecondaryStructureAnalysisError,
    _frame_pdb_payload,
    _mkdssp_environment,
    _pdb_template_lines,
    _protein_residue_keys,
    _validate_dssp_residue_coverage,
    build_mkdssp_command,
    parse_dssp_text,
)

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / 'tutorials/nemo_zinc_finger_workstation/data/nemo_zinc_finger.pdb'


def normalized_fixture():
    _, atoms = read_topology_atoms(PDB)
    frame = next(iter_coordinate_frames(PDB, 'angstrom'))
    return _frame_pdb_payload(_pdb_template_lines(PDB), atoms, frame.coordinates_angstrom)


def residue(names, residue_name='GLU'):
    atoms = [AtomRecord(i, i + 1, name, '', residue_name, 'A', 9, 'B',
                        'O' if name.startswith('O') else 'N')
             for i, name in enumerate(names)]
    lines = [f'ATOM  {i+1:5d}  {name:<3s} {residue_name:3s} A   9B      1.000   2.000   3.000  1.00  0.00           O  '
             for i, name in enumerate(names)]
    return lines, atoms, [(1.0, 2.0, 3.0)] * len(names)


class DsspCompatibilityTests(unittest.TestCase):
    def test_fixture_names_are_normalized_without_changing_input_or_coordinates(self):
        before = PDB.read_bytes()
        text, mapping = normalized_fixture()
        lines = [line for line in text.splitlines() if line.startswith('ATOM')]
        source = [line for line in before.decode().splitlines()
                  if line.startswith('ATOM') and line[17:20].strip() != 'ZN2']
        self.assertEqual(len(mapping), 28)
        self.assertEqual(len(lines), 422)
        self.assertEqual([line[30:54] for line in lines], [line[30:54] for line in source])
        self.assertEqual(mapping[('A', '22')]['residue_name'], 'HSD')
        self.assertEqual(mapping[('A', '22')]['dssp_residue_name'], 'HIS')
        self.assertEqual(mapping[('A', '28')]['atom_name_aliases'], [
            {'atom_index': atom_index, 'original_atom_name': old, 'dssp_atom_name': new}
            for atom_index, old, new in (
                (next(i for i, line in enumerate(source) if line[22:26].strip() == '28' and line[12:16].strip() == 'OT1'), 'OT1', 'O'),
                (next(i for i, line in enumerate(source) if line[22:26].strip() == '28' and line[12:16].strip() == 'OT2'), 'OT2', 'OXT'),
            )
        ])
        self.assertNotIn('REMARK', text)
        self.assertEqual(PDB.read_bytes(), before)

    def test_supported_histidine_aliases_keep_original_identity(self):
        for name in ('HSD', 'HSE', 'HSP', 'HID', 'HIE', 'HIP'):
            with self.subTest(name=name):
                text, mapping = _frame_pdb_payload(*residue(['N'], name))
                self.assertEqual(text.splitlines()[0][17:20], 'HIS')
                self.assertEqual(mapping[('A', '1')]['residue_name'], name)
                self.assertEqual(mapping[('A', '1')]['original_residue_token'], '9B')

    def test_terminal_caps_are_not_expected_amino_acid_assignments(self):
        atoms = [AtomRecord(0, 1, 'N', '', 'ALA', 'A', 1, '', 'N'),
                 AtomRecord(1, 2, 'C', '', 'ACE', 'A', 0, '', 'C'),
                 AtomRecord(2, 3, 'N', '', 'NME', 'A', 2, '', 'N')]
        self.assertEqual(_protein_residue_keys(atoms), {('A', 1, '', 'ALA')})

    def test_ambiguous_terminal_aliases_fail(self):
        for names in (['OT1'], ['OT2'], ['OT1', 'OT2', 'O'], ['OT1', 'OT2', 'OXT']):
            with self.subTest(names=names):
                with self.assertRaisesRegex(SecondaryStructureAnalysisError, 'ambiguous terminal oxygen'):
                    _frame_pdb_payload(*residue(names))

    def test_standard_names_are_not_rewritten(self):
        text, mapping = _frame_pdb_payload(*residue(['O', 'OXT']))
        self.assertEqual([line[12:16].strip() for line in text.splitlines() if line.startswith('ATOM')], ['O', 'OXT'])
        self.assertEqual(mapping[('A', '1')]['atom_name_aliases'], [])

    def test_coverage_accepts_all_expected_residues(self):
        _, mapping = normalized_fixture()
        rows = [{'chain_id': chain, 'dssp_residue_token': token} for chain, token in mapping]
        _validate_dssp_residue_coverage(rows, mapping)

    def test_missing_coverage_reports_original_residues(self):
        _, mapping = normalized_fixture()
        rows = [{'chain_id': chain, 'dssp_residue_token': token}
                for chain, token in mapping if token not in ('22', '28')]
        with self.assertRaisesRegex(SecondaryStructureAnalysisError, 'omitted 2 of 28.*A/22/HSD.*A/28/GLU'):
            _validate_dssp_residue_coverage(rows, mapping)

    def test_duplicate_or_unknown_assignments_fail(self):
        _, mapping = normalized_fixture()
        rows = [{'chain_id': chain, 'dssp_residue_token': token} for chain, token in mapping]
        with self.assertRaisesRegex(SecondaryStructureAnalysisError, 'duplicate residue'):
            _validate_dssp_residue_coverage(rows + [rows[0]], mapping)
        with self.assertRaisesRegex(SecondaryStructureAnalysisError, 'absent from the reversible input map'):
            _validate_dssp_residue_coverage(rows + [{'chain_id': 'Z', 'dssp_residue_token': '1'}], mapping)

    @unittest.skipUnless(shutil.which('mkdssp'), 'optional real DSSP executable unavailable')
    def test_real_dssp_covers_all_28_nemo_residues(self):
        binary = shutil.which('mkdssp')
        before = hashlib.sha256(PDB.read_bytes()).hexdigest()
        version = subprocess.check_output([binary, '--version'], text=True)
        env, _, _ = _mkdssp_environment(binary)
        text, mapping = normalized_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / 'frame.pdb'
            output_path = Path(temporary) / 'frame.dssp'
            input_path.write_text(text)
            result = subprocess.run(build_mkdssp_command(binary, input_path, output_path, version),
                                    env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = parse_dssp_text(output_path.read_text())
            _validate_dssp_residue_coverage(rows, mapping)
            self.assertEqual(len(rows), 28)
        self.assertEqual(hashlib.sha256(PDB.read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()
