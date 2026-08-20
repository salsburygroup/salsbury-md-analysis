import json
import math
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.rdf import (
    RDFAnalysisError,
    periodic_cell_geometry,
    radial_distribution_functions_project,
)


def _atom(serial: int, name: str, x: float) -> str:
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA A   1    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
    )


def _write_project(root: Path, maximum_radius: float = 5.0) -> Path:
    cell = "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n"
    topology = cell + _atom(1, "C1", 1.0) + _atom(2, "C2", 3.0) + "END\n"
    (root / "topology.pdb").write_text(topology, encoding="ascii")
    trajectory = cell
    for model, second_x in ((1, 3.2), (2, 18.8)):
        trajectory += f"MODEL     {model:4d}\n"
        trajectory += _atom(1, "C1", 1.0) + _atom(2, "C2", second_x)
        trajectory += "ENDMDL\n"
    trajectory += "END\n"
    (root / "trajectory.pdb").write_text(trajectory, encoding="ascii")
    (root / "system.json").write_text(json.dumps({
        "systems": [{"system_id": "box", "replicas": [{
            "replica_id": "r1",
            "topology": "topology.pdb",
            "segments": [{
                "segment_id": "s1",
                "trajectory": "trajectory.pdb",
                "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
            }],
        }]}],
    }), encoding="utf-8")
    project = {
        "project_id": "rdf-test",
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "allow_wrapped_diagnostic",
        "selections": {
            "alignment": {"preset": "all"},
            "analysis": {"preset": "all"},
        },
        "definitions": {"radial_distribution_functions": {
            "frame_stride": 1,
            "maximum_observations": 10,
            "features": [{
                "feature_id": "carbon-pair",
                "question": "How is the declared carbon pair distributed?",
                "group_a_atom_indices": [0],
                "group_b_atom_indices": [1],
                "minimum_radius_angstrom": 0.0,
                "maximum_radius_angstrom": maximum_radius,
                "bin_width_angstrom": 1.0,
            }],
        }},
        "requested_modules": ["radial_distribution_functions"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class RDFTests(unittest.TestCase):
    def test_triclinic_geometry_reports_volume_and_safe_radius(self):
        volume, radius = periodic_cell_geometry(
            ((10.0, 0.0, 0.0), (2.0, 8.0, 0.0), (0.0, 0.0, 6.0))
        )
        self.assertAlmostEqual(volume, 480.0)
        self.assertGreater(radius, 2.9)

    def test_project_uses_minimum_image_and_uniform_shell_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = radial_distribution_functions_project(
                _write_project(Path(temporary))
            )
        feature = report["feature_reports"][0]
        self.assertEqual(feature["evaluated_frame_count"], 2)
        self.assertAlmostEqual(feature["mean_cell_volume_angstrom3"], 8000.0)
        occupied = feature["bins"][2]
        self.assertEqual(occupied["observed_pair_count"], 2)
        expected = 2 * (4.0 * math.pi * (3.0 ** 3 - 2.0 ** 3) / 3.0) / 8000.0
        self.assertAlmostEqual(occupied["uniform_expected_pair_count"], expected)
        self.assertAlmostEqual(occupied["g_r"], 2 / expected)

    def test_project_refuses_radius_beyond_half_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary), maximum_radius=11.0)
            with self.assertRaisesRegex(RDFAnalysisError, "safe half-cell radius"):
                radial_distribution_functions_project(path)


if __name__ == "__main__":
    unittest.main()
