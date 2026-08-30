import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from salsbury_md_analysis.trajectory_features import (
    TrajectoryFeatureError,
    center_of_mass,
    center_of_geometry,
    flatten_coordinates,
    group_distance_statistics,
    minimum_mean_group_distance,
    principal_axes,
    trajectory_features_project,
)
from salsbury_md_analysis.columnar_artifacts import iter_columnar_records


def _pdb_atom(serial, name, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


class TrajectoryFeatureTests(unittest.TestCase):
    def test_cartesian_and_center_of_mass_features_match_direct_definitions(self):
        coordinates = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0)]
        self.assertEqual(
            flatten_coordinates(coordinates, [2, 0]),
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(center_of_mass(coordinates, [1.0, 3.0, 2.0], [0, 1]), (1.5, 0.0, 0.0))
        self.assertEqual(center_of_geometry(coordinates, [0, 1]), (1.0, 0.0, 0.0))

    def test_group_distance_families_remain_distinct(self):
        coordinates = [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
        ]
        statistics = group_distance_statistics(coordinates, [0, 1], [2, 3])
        self.assertAlmostEqual(statistics["minimum_distance_angstrom"], 1.0)
        self.assertAlmostEqual(statistics["mean_distance_angstrom"], 2.5)
        self.assertAlmostEqual(statistics["maximum_distance_angstrom"], 5.0)
        self.assertEqual(statistics["closest_atom_indices"], [0, 2])
        self.assertEqual(statistics["farthest_atom_indices"], [0, 3])

        minimum_mean = minimum_mean_group_distance(coordinates, [0, 1], [2, 3])
        self.assertAlmostEqual(minimum_mean["minimum_mean_distance_angstrom"], 1.0)
        self.assertEqual(minimum_mean["selected_candidate_atom_index"], 2)

    def test_principal_axes_are_orthonormal_right_handed_and_translation_invariant(self):
        coordinates = [(-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        translated = [(x + 7.0, y - 4.0, z + 2.0) for x, y, z in coordinates]
        first = principal_axes(coordinates, [12.0, 12.0, 12.0], [0, 1, 2])
        second = principal_axes(translated, [12.0, 12.0, 12.0], [0, 1, 2])
        axes = np.asarray(first["principal_axes"])
        self.assertTrue(np.allclose(axes @ axes.T, np.eye(3), atol=1.0e-12))
        self.assertGreater(np.linalg.det(axes), 0.0)
        self.assertTrue(np.allclose(first["principal_moments"], second["principal_moments"]))
        self.assertTrue(np.allclose(np.abs(first["principal_axes"]), np.abs(second["principal_axes"])))

    def test_empty_and_too_small_selections_fail_closed(self):
        with self.assertRaises(TrajectoryFeatureError):
            flatten_coordinates([(0.0, 0.0, 0.0)], [])
        with self.assertRaises(TrajectoryFeatureError):
            principal_axes([(0.0, 0.0, 0.0)] * 3, [1.0] * 3, [0, 1])

    def test_project_writes_columnar_records_when_artifact_root_is_declared(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reference.pdb").write_text(
                _pdb_atom(1, "C", 0, 0, 0, "C")
                + _pdb_atom(2, "N", 1, 0, 0, "N")
                + "END\n",
                encoding="utf-8",
            )
            (root / "trajectory.xyz").write_text(
                "2\nf0\nC 0 0 0\nN 1 0 0\n"
                "2\nf1\nC 0 0 0\nN 2 0 0\n",
                encoding="utf-8",
            )
            (root / "system.json").write_text(json.dumps({
                "systems": [{
                    "system_id": "system",
                    "replicas": [{
                        "replica_id": "r1",
                        "topology": "reference.pdb",
                        "segments": [{
                            "segment_id": "s1",
                            "trajectory": "trajectory.xyz",
                            "timing": {
                                "first_frame_time": 0,
                                "frame_interval": 1,
                                "unit": "ps",
                            },
                        }],
                    }],
                }],
            }), encoding="utf-8")
            project = root / "project.json"
            project.write_text(json.dumps({
                "project_id": "trajectory-feature-columnar-test",
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
                "definitions": {"trajectory_features": {
                    "frame_stride": 1,
                    "maximum_feature_values": 10,
                    "features": [{
                        "feature_id": "distance",
                        "kind": "pair_distance",
                        "atom_indices": [0, 1],
                    }],
                }},
                "requested_modules": ["trajectory_features"],
                "protected_locations": ["/protected/example"],
            }), encoding="utf-8")
            artifact_root = root / "artifacts"
            with patch.dict(os.environ, {
                "SALSBURY_MD_ANALYSIS_COLUMNAR_ARTIFACT_ROOT": str(
                    artifact_root
                )
            }):
                report = trajectory_features_project(project)
            feature = report["segments"][0]["features"][0]
            self.assertIsNone(feature["records"])
            rows = list(iter_columnar_records(feature["columnar_artifact"]))
            self.assertEqual([row["values"] for row in rows], [[1.0], [2.0]])


if __name__ == "__main__":
    unittest.main()
