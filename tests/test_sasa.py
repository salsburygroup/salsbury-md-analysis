import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.sasa import (
    SASAAnalysisError,
    shrake_rupley_sasa,
    solvent_accessible_surface_area_project,
)


def _pdb_atom(serial, name, residue, residue_number, x, element):
    return (
        f"ATOM  {serial:5d} {name:^4s} {residue:>3s} A{residue_number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _write_project(root: Path, sphere_point_count=240) -> Path:
    topology = (
        _pdb_atom(1, "C", "ALA", 1, 0.0, "C")
        + _pdb_atom(2, "O", "GLY", 2, 8.0, "O")
        + "END\n"
    )
    (root / "topology.pdb").write_text(topology, encoding="utf-8")
    trajectory = (
        "2\nframe 0\nC 0 0 0\nO 8 0 0\n"
        "2\nframe 1\nC 0 0 0\nO 7 0 0\n"
    )
    (root / "trajectory.xyz").write_text(trajectory, encoding="utf-8")
    system = {
        "systems": [{
            "system_id": "test-system",
            "replicas": [{
                "replica_id": "r1",
                "topology": "topology.pdb",
                "segments": [{
                    "segment_id": "s1",
                    "trajectory": "trajectory.xyz",
                    "timing": {"first_frame_time": 0, "frame_interval": 2, "unit": "ps"},
                }],
            }],
        }]
    }
    (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
    project = {
        "project_id": "sasa-test",
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "reject",
        "selections": {
            "alignment": {"preset": "all"},
            "analysis": {"preset": "all"},
            "surface": {"preset": "all"},
            "occluders": {"preset": "all"},
        },
        "definitions": {
            "solvent_accessible_surface_area": {
                "surface_selection": "surface",
                "occluder_selection": "occluders",
                "probe_radius_angstrom": 1.4,
                "sphere_point_count": sphere_point_count,
                "frame_stride": 1,
                "maximum_surface_atoms": 10,
                "maximum_observations": 100,
                "output_detail": "full_atom_timeseries",
            }
        },
        "requested_modules": ["solvent_accessible_surface_area"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class SASATests(unittest.TestCase):
    def test_isolated_atom_has_exact_probe_inflated_sphere_area(self):
        observed = shrake_rupley_sasa(
            [(0.0, 0.0, 0.0)], ["C"], sphere_point_count=240
        )[0]
        self.assertAlmostEqual(observed, 4.0 * math.pi * (1.70 + 1.40) ** 2)

    def test_neighbor_occlusion_reduces_but_does_not_eliminate_area(self):
        isolated = 2.0 * 4.0 * math.pi * (1.70 + 1.40) ** 2
        observed = sum(shrake_rupley_sasa(
            [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)],
            ["C", "C"],
            sphere_point_count=960,
        ))
        self.assertGreater(observed, 0.0)
        self.assertLess(observed, isolated)

    def test_unknown_element_fails_closed(self):
        with self.assertRaises(SASAAnalysisError):
            shrake_rupley_sasa([(0.0, 0.0, 0.0)], ["XX"], sphere_point_count=24)

    def test_unknown_element_outside_surface_and_occluders_is_ignored(self):
        observed = shrake_rupley_sasa(
            [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)],
            ["C", "XX"],
            surface_atom_indices=[0],
            occluder_atom_indices=[0],
            sphere_point_count=24,
        )
        self.assertEqual(len(observed), 1)
        self.assertAlmostEqual(observed[0], 4.0 * math.pi * (1.70 + 1.40) ** 2)

    def test_declared_element_radius_override_is_explicit_and_effective(self):
        observed = shrake_rupley_sasa(
            [(0.0, 0.0, 0.0)], ["MG"], sphere_point_count=24,
            element_radii_overrides_angstrom={"MG": 1.73},
        )[0]
        self.assertAlmostEqual(observed, 4.0 * math.pi * (1.73 + 1.40) ** 2)

    def test_project_reports_atom_residue_and_total_sasa(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = solvent_accessible_surface_area_project(_write_project(Path(temporary)))
        self.assertEqual(report["technical_status"], "complete")
        replica = report["replicas"][0]
        self.assertEqual(replica["evaluated_frame_count"], 2)
        self.assertEqual(len(replica["frames"][0]["per_atom"]), 2)
        self.assertEqual(len(replica["frames"][0]["per_residue"]), 2)
        self.assertEqual(report["observation_count"], 4)
        self.assertEqual(report["warning_count"], 1)
        self.assertEqual(
            report["issues"][0]["code"],
            "SASA_RESOLUTION_SENSITIVITY_REQUIRED",
        )

    def test_project_refuses_scientifically_coarse_sphere_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary), sphere_point_count=24)
            with self.assertRaisesRegex(
                SASAAnalysisError,
                "sphere_point_count must be an integer of at least 240",
            ):
                solvent_accessible_surface_area_project(path)

    def test_bounded_output_preserves_exact_summaries_and_scott_histogram(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary))
            project = json.loads(path.read_text(encoding="utf-8"))
            project["definitions"]["solvent_accessible_surface_area"][
                "output_detail"
            ] = "bounded_summary_v1"
            path.write_text(json.dumps(project), encoding="utf-8")
            report = solvent_accessible_surface_area_project(path)
        replica = report["replicas"][0]
        self.assertNotIn("frames", replica)
        self.assertEqual(len(replica["total_sasa_timeseries"]), 2)
        self.assertEqual(len(replica["per_atom_summaries"]), 2)
        self.assertEqual(
            replica["per_atom_summaries"][0]["summary_angstrom2"]["count"], 2
        )
        self.assertEqual(replica["total_sasa_distribution"]["binning"]["rule"], "scott")

    def test_project_executes_and_reports_automatic_resource_subsampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary))
            project = json.loads(path.read_text(encoding="utf-8"))
            project["definitions"]["solvent_accessible_surface_area"]["frame_selection"] = {
                "mode": "auto_resource_budget_v1",
                "target_wall_seconds": 2.0,
                "estimated_seconds_per_frame": 1.0,
                "minimum_frames_per_replica": 1,
                "sensitivity_check_policy": "recommend",
                "calibration_id": "test-calibration",
            }
            path.write_text(json.dumps(project), encoding="utf-8")
            report = solvent_accessible_surface_area_project(path)
        self.assertEqual(report["frame_selection"]["source_frame_count"], 2)
        self.assertEqual(report["frame_selection"]["selected_frame_count"], 1)
        self.assertEqual(
            report["frame_selection"]["resolved_mode"],
            "integer_stride_per_replica_v1",
        )
        self.assertTrue(
            report["frame_selection"]["resource_estimate"]["subsampling_triggered"]
        )
        self.assertIn("FRAME_SUBSAMPLING", {
            issue["code"] for issue in report["issues"]
        })

    def test_cli_emits_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary))
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["sasa", str(path)])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["module_id"], "solvent_accessible_surface_area")


if __name__ == "__main__":
    unittest.main()
