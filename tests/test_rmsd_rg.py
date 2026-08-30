import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from salsbury_md_analysis.atom_mapping import AtomRecord
from salsbury_md_analysis.cli import main
from salsbury_md_analysis.rmsd_rg import (
    _mapping_bundles,
    replica_rmsd_rg_project,
    replica_rmsd_rg_project_safe,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "manifest_fixture"


def _pdb_atom(serial, name, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _write_project(root: Path, frame_stride: int = 1, element: str = "C") -> Path:
    if element == "C":
        atoms = [
            _pdb_atom(1, "C", 0, 0, 0, "C"),
            _pdb_atom(2, "N", 1, 0, 0, "N"),
            _pdb_atom(3, "O", 0, 2, 0, "O"),
        ]
        frames = (
            "3\nf0\nC 5 -3 2\nN 5 -2 2\nO 3 -3 2\n"
            "3\nf1\nC -4 7 1\nN -4 8 1\nO -6 7 1\n"
            "3\nf2\nC 2 2 2\nN 2 3 2\nO 0 2 2\n"
        )
    else:
        atoms = [_pdb_atom(1, "X", 0, 0, 0, element)]
        frames = "1\nf0\nX 0 0 0\n"
    (root / "reference.pdb").write_text("".join(atoms) + "END\n", encoding="utf-8")
    (root / "trajectory.xyz").write_text(frames, encoding="utf-8")
    system = {
        "systems": [{
            "system_id": "system",
            "replicas": [{
                "replica_id": "replica-1",
                "topology": "reference.pdb",
                "segments": [{
                    "segment_id": "segment-1",
                    "trajectory": "trajectory.xyz",
                    "timing": {"first_frame_time": 10, "frame_interval": 2, "unit": "ps"},
                }],
            }],
        }]
    }
    (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
    project = {
        "project_id": "rmsd-rg-test",
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "reject",
        "reference_structure": "reference.pdb",
        "common_atom_policy": "strict",
        "selections": {
            "alignment": {"preset": "all"},
            "analysis": {"preset": "all"},
        },
        "definitions": {
            "replica_rmsd_rg": {
                "alignment_selection": "alignment",
                "rmsd_selection": "analysis",
                "rg_selection": "analysis",
                "minimum_reference_coverage": 1.0,
                "frame_stride": frame_stride,
            }
        },
        "requested_modules": ["replica_rmsd_rg"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


def _replace_with_periodic_gro(project_path: Path, policy: str) -> None:
    root = project_path.parent
    gro = (
        "periodic fixture\n"
        "3\n"
        f"{1:5d}{'ALA':<5}{'C':>5}{1:5d}{0.95:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{1:5d}{'ALA':<5}{'N':>5}{2:5d}{0.05:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        f"{1:5d}{'ALA':<5}{'O':>5}{3:5d}{0.95:8.3f}{0.2:8.3f}{0.0:8.3f}\n"
        "1.0 1.0 1.0\n"
    )
    (root / "trajectory.gro").write_text(gro, encoding="utf-8")
    system_path = root / "system.json"
    system = json.loads(system_path.read_text(encoding="utf-8"))
    system["systems"][0]["replicas"][0]["segments"][0]["trajectory"] = "trajectory.gro"
    if policy in {"make_whole", "unwrap_continuous"}:
        (root / "bonds.json").write_text(
            json.dumps({
                "format": "salsbury-bonds-v1", "atom_count": 3,
                "index_base": 0, "bonds": [[0, 1], [0, 2]],
            }),
            encoding="utf-8",
        )
        system["systems"][0]["replicas"][0]["connectivity"] = "bonds.json"
    system_path.write_text(json.dumps(system), encoding="utf-8")
    project = json.loads(project_path.read_text(encoding="utf-8"))
    project["periodic_coordinate_policy"] = policy
    if policy in {"make_whole", "unwrap_continuous"}:
        project["periodic_reconstruction"] = {
            "maximum_bond_length_angstrom": 3.0,
            "cycle_closure_tolerance_angstrom": 1.0e-6,
        }
        if policy == "unwrap_continuous":
            project["periodic_reconstruction"]["maximum_anchor_displacement_angstrom"] = 2.0
    project_path.write_text(json.dumps(project), encoding="utf-8")


class RMSDRGTests(unittest.TestCase):
    def test_rg_selection_is_topology_local_not_global_common_atom_intersection(self):
        reference = [
            AtomRecord(index, index + 1, name, "", "ALA", "A", 1, "", element)
            for index, (name, element) in enumerate(
                (("C", "C"), ("N", "N"), ("O", "O"), ("K", "K"))
            )
        ]
        target = reference[:3]
        mappings = _mapping_bundles(
            reference,
            [target],
            {
                "alignment": {"atom_names": ["C", "N", "O"]},
                "analysis": {"atom_names": ["C", "N", "O"]},
                "solute": {"preset": "all"},
            },
            {
                "alignment_selection": "alignment",
                "rmsd_selection": "analysis",
                "rg_selection": "solute",
                "minimum_reference_coverage": 0.95,
            },
            "strict",
        )
        self.assertEqual(mappings[0]["rmsd"].reference_selected_count, 3)
        self.assertEqual(mappings[0]["rg"].target_selected_count, 3)
        self.assertEqual(mappings[0]["rg"].reference_coverage, 1.0)

    def test_teaching_fixture_is_technical_only(self):
        report = replica_rmsd_rg_project(EXAMPLE / "project.json", hash_content=True)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["scientific_status"], "not evaluated")
        segment = report["systems"][0]["replicas"][0]["segments"][0]
        self.assertEqual(segment["observed_frame_count"], 2)
        self.assertEqual(segment["timeseries"][0]["rmsd_angstrom"], 0.0)
        self.assertEqual(segment["timeseries"][1]["radius_of_gyration_angstrom"], 0.0)

    def test_rotation_translation_are_removed_and_rg_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = replica_rmsd_rg_project(_write_project(Path(temporary)))
        self.assertEqual(report["technical_status"], "complete")
        rows = report["systems"][0]["replicas"][0]["segments"][0]["timeseries"]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["rmsd_angstrom"] < 1.0e-12 for row in rows))
        self.assertAlmostEqual(
            rows[0]["radius_of_gyration_angstrom"],
            rows[2]["radius_of_gyration_angstrom"],
        )

    def test_frame_stride_is_explicit_and_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = replica_rmsd_rg_project(
                _write_project(Path(temporary), frame_stride=2)
            )
        segment = report["systems"][0]["replicas"][0]["segments"][0]
        self.assertEqual(segment["observed_frame_count"], 3)
        self.assertEqual(segment["decoded_frame_count"], 1)
        self.assertEqual([row["frame_index"] for row in segment["timeseries"]], [0])
        self.assertEqual([row["time"] for row in segment["timeseries"]], [10.0])
        self.assertIn("FRAME_SUBSAMPLING", {issue["code"] for issue in report["issues"]})

    def test_unknown_element_mass_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = replica_rmsd_rg_project_safe(
                _write_project(Path(temporary), element="X")
            )
        self.assertEqual(report["technical_status"], "failed")
        self.assertTrue(
            any("atomic mass" in issue["message"] for issue in report["issues"])
        )

    def test_periodic_coordinates_reject_by_default_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary))
            _replace_with_periodic_gro(path, "reject")
            report = replica_rmsd_rg_project_safe(path)
        self.assertEqual(report["technical_status"], "failed")
        self.assertTrue(
            any("periodic_coordinate_policy=reject" in issue["message"] for issue in report["issues"])
        )

    def test_wrapped_periodic_diagnostic_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary))
            _replace_with_periodic_gro(path, "allow_wrapped_diagnostic")
            report = replica_rmsd_rg_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertIn(
            "PERIODIC_COORDINATES_NOT_UNWRAPPED",
            {issue["code"] for issue in report["issues"]},
        )

    def test_connectivity_aware_make_whole_recovers_boundary_crossing_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary))
            _replace_with_periodic_gro(path, "make_whole")
            report = replica_rmsd_rg_project(path, hash_content=True)
        self.assertEqual(report["technical_status"], "complete")
        row = report["systems"][0]["replicas"][0]["segments"][0]["timeseries"][0]
        self.assertLess(row["rmsd_angstrom"], 1.0e-12)
        self.assertNotIn(
            "PERIODIC_COORDINATES_NOT_UNWRAPPED",
            {issue["code"] for issue in report["issues"]},
        )

    def test_cli_emits_machine_readable_report(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["rmsd-rg", str(EXAMPLE / "project.json")])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["module_id"], "replica_rmsd_rg")


if __name__ == "__main__":
    unittest.main()
