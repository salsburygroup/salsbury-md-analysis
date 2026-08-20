import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.rmsf_visualization import (
    RMSFVisualizationError,
    build_rmsf_bfactor_pdb,
    export_rmsf_visualization,
)


def _atom(serial, name, residue, x, bfactor=0.0):
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA A{residue:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{bfactor:6.2f}           C\n"
    )


class RMSFVisualizationTests(unittest.TestCase):
    def test_residue_mean_populates_pdb_bfactor_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pdb"
            reference.write_text(
                _atom(1, "CA", 1, 0.0) + _atom(2, "CB", 1, 1.0)
                + _atom(3, "CA", 2, 2.0) + "END\n",
                encoding="utf-8",
            )
            result = build_rmsf_bfactor_pdb(reference, [{
                "chain_id": "A", "residue_number": 1, "insertion_code": "",
                "residue_name": "ALA", "atom_name": "CA", "altloc": "",
                "frame_pooled_rmsf_angstrom": 2.5,
            }])
        atom_lines = [line for line in result["pdb_text"].splitlines() if line.startswith("ATOM")]
        self.assertEqual([float(line[60:66]) for line in atom_lines], [2.5, 2.5, 0.0])
        self.assertEqual(result["mapped_output_atom_count"], 2)

    def test_export_writes_bfactor_pdb_and_vmd_cartoon(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pdb"
            reference.write_text(_atom(1, "CA", 1, 0.0) + "END\n", encoding="utf-8")
            report = root / "rmsf.json"
            report.write_text(json.dumps({
                "reference": {"path": str(reference), "format": "pdb"},
                "systems": [{
                    "system_id": "system",
                    "atom_statistics": [{
                        "chain_id": "A", "residue_number": 1, "insertion_code": "",
                        "residue_name": "ALA", "atom_name": "CA", "altloc": "",
                        "frame_pooled_rmsf_angstrom": 1.25,
                    }],
                }],
            }), encoding="utf-8")
            result = export_rmsf_visualization(report, "system", root / "view")
            pdb = Path(result["pdb_path"]).read_text(encoding="utf-8")
            script = Path(result["vmd_script_path"]).read_text(encoding="utf-8")
        self.assertAlmostEqual(float(next(line for line in pdb.splitlines() if line.startswith("ATOM"))[60:66]), 1.25)
        self.assertIn("NewCartoon", script)
        self.assertIn("mol color Beta", script)

    def test_refuses_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.pdb"
            reference.write_text(_atom(1, "CA", 1, 0.0) + "END\n", encoding="utf-8")
            report = root / "rmsf.json"
            report.write_text(json.dumps({
                "reference": {"path": str(reference)},
                "systems": [{"system_id": "s", "atom_statistics": [{
                    "chain_id": "A", "residue_number": 1, "insertion_code": "",
                    "residue_name": "ALA", "atom_name": "CA", "altloc": "",
                    "frame_pooled_rmsf_angstrom": 0.0,
                }]}],
            }), encoding="utf-8")
            export_rmsf_visualization(report, "s", root / "view")
            with self.assertRaises(RMSFVisualizationError):
                export_rmsf_visualization(report, "s", root / "view")


if __name__ == "__main__":
    unittest.main()

