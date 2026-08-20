import json
import math
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.ion_geometry import (
    coordination_geometry_score,
    ion_coordination_geometry_project,
)


def _atom(serial, name, residue, resid, point, element):
    x, y, z = point
    return (
        f"HETATM{serial:5d} {name:^4s} {residue:>3s} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _write_project(root: Path) -> Path:
    scale = 2.0 / math.sqrt(3.0)
    ligands = [
        (scale, scale, scale), (scale, -scale, -scale),
        (-scale, scale, -scale), (-scale, -scale, scale),
    ]
    points = [(0.0, 0.0, 0.0)] + ligands + [(5.0, 0.0, 0.0)]
    topology = _atom(1, "MG", "MG", 1, points[0], "MG")
    topology += "".join(
        _atom(index + 2, f"O{index + 1}", "HOH", index + 2, point, "O")
        for index, point in enumerate(ligands)
    )
    topology += _atom(6, "CA", "CA", 6, points[5], "CA") + "END\n"
    (root / "topology.pdb").write_text(topology, encoding="ascii")
    frames = [
        points,
        [points[0], (ligands[0][0] - 0.2, *ligands[0][1:])] + points[2:],
    ]
    trajectory = ""
    for frame_index, frame in enumerate(frames):
        trajectory += f"6\nframe {frame_index}\n"
        trajectory += "".join(f"X {x} {y} {z}\n" for x, y, z in frame)
    (root / "trajectory.xyz").write_text(trajectory, encoding="ascii")
    (root / "system.json").write_text(json.dumps({
        "systems": [{"system_id": "ions", "replicas": [{
            "replica_id": "r1", "topology": "topology.pdb",
            "segments": [{
                "segment_id": "s1", "trajectory": "trajectory.xyz",
                "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
            }],
        }]}],
    }), encoding="utf-8")
    project = {
        "project_id": "ion-test", "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json", "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD", "coordinate_unit": "angstrom",
        "time_unit": "ps", "periodic_coordinate_policy": "reject",
        "selections": {"alignment": {"preset": "all"}, "analysis": {"preset": "all"}},
        "definitions": {"ion_coordination_geometry": {
            "frame_stride": 1, "maximum_frames": 10,
            "ion_sites": [{
                "site_id": "bound-mg", "ion_atom_index": 0,
                "candidate_ligand_atom_indices": [1, 2, 3, 4],
                "coordination_cutoff_angstrom": 2.5,
                "geometry_templates": ["tetrahedral", "square_planar"],
            }, {
                "site_id": "paired-ca", "ion_atom_index": 5,
                "candidate_ligand_atom_indices": [1],
                "coordination_cutoff_angstrom": 4.5,
                "geometry_templates": [],
            }],
            "ion_pairs": [{
                "pair_id": "mg-ca", "first_ion_atom_index": 0,
                "second_ion_atom_index": 5,
            }],
            "block_count": 2, "histogram_rule": "scott",
            "histogram_padding_fraction": 0.05,
            "minimum_histogram_bins": 2, "maximum_histogram_bins": 20,
        }},
        "requested_modules": ["ion_coordination_geometry"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class IonGeometryTests(unittest.TestCase):
    def test_ideal_tetrahedral_score_is_zero_and_wrong_template_is_not(self):
        vectors = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        tetrahedral = coordination_geometry_score(vectors, "tetrahedral")
        rotated_permuted = coordination_geometry_score(
            [(-y, x, z) for x, y, z in (vectors[2], vectors[0], vectors[3], vectors[1])],
            "tetrahedral",
        )
        square = coordination_geometry_score(vectors, "square_planar")
        self.assertAlmostEqual(tetrahedral["rms_pair_angle_deviation_degrees"], 0.0)
        self.assertAlmostEqual(tetrahedral["optimal_shape_rms_unit_vector"], 0.0)
        self.assertAlmostEqual(tetrahedral["optimal_shape_angular_rms_degrees"], 0.0)
        self.assertAlmostEqual(rotated_permuted["optimal_shape_rms_unit_vector"], 0.0)
        self.assertGreater(square["rms_pair_angle_deviation_degrees"], 10.0)
        self.assertGreater(square["optimal_shape_rms_unit_vector"], 0.1)

    def test_project_reports_bound_ligands_pair_distance_and_scott_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = ion_coordination_geometry_project(_write_project(Path(temporary)))
        self.assertEqual(report["technical_status"], "complete")
        first = report["frame_reports"][0]
        self.assertEqual(first["ion_sites"][0]["coordination_number"], 4)
        self.assertEqual(len(first["ion_sites"][0]["bound_ligands"]), 4)
        self.assertAlmostEqual(first["ion_pairs"][0]["distance_angstrom"], 5.0)
        self.assertTrue(first["ion_pairs"][0]["bound_site_pair_evaluated"])
        self.assertEqual(first["ion_pairs"][0]["shared_bound_ligand_count"], 1)
        self.assertGreater(
            first["ion_pairs"][0]["shared_bound_ligands"][0][
                "ion_ligand_ion_angle_degrees"
            ],
            90.0,
        )
        geometry = first["ion_sites"][0]["geometry_scores"]
        self.assertEqual({row["template"] for row in geometry}, {"tetrahedral", "square_planar"})
        occupancy = report["replica_reports"][0]["ligand_occupancies"]
        self.assertTrue(all(row["bound_fraction"] == 1.0 for row in occupancy))
        distance_distribution = next(
            row for row in report["distribution_reports"]
            if row["metric_id"] == "ion_site:bound-mg:nearest_candidate_distance_angstrom"
        )
        self.assertEqual(distance_distribution["binning"]["rule"], "scott")


if __name__ == "__main__":
    unittest.main()
