import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.observables import (
    minimum_group_distance,
    native_contact_pairs,
    optional_observables_project,
)


def _atom(serial, name, x):
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA A   1    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
    )


def _native_project(root: Path) -> Path:
    topology = _atom(1, "C1", 0.0) + _atom(2, "C2", 2.0) + _atom(3, "C3", 8.0) + "END\n"
    (root / "reference.pdb").write_text(topology, encoding="ascii")
    (root / "trajectory.xyz").write_text(
        "3\nframe 0\nC 0 0 0\nC 2 0 0\nC 8 0 0\n"
        "3\nframe 1\nC 0 0 0\nC 4 0 0\nC 8 0 0\n",
        encoding="ascii",
    )
    (root / "system.json").write_text(json.dumps({
        "systems": [{"system_id": "native", "replicas": [{
            "replica_id": "r1", "topology": "reference.pdb",
            "segments": [{
                "segment_id": "s1", "trajectory": "trajectory.xyz",
                "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
            }],
        }]}],
    }), encoding="utf-8")
    project = {
        "project_id": "native-test", "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json", "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD", "coordinate_unit": "angstrom",
        "time_unit": "ps", "periodic_coordinate_policy": "reject",
        "reference_structure": "reference.pdb", "common_atom_policy": "strict",
        "selections": {"alignment": {"preset": "all"}, "analysis": {"preset": "all"}},
        "definitions": {"optional_observables": {
            "frame_stride": 1, "maximum_observations": 10,
            "features": [{
                "feature_id": "native-q", "question": "Is the reference contact retained?",
                "kind": "native_contact_fraction", "atom_indices": [0, 1, 2],
                "reference_cutoff_angstrom": 3.0,
                "observation_cutoff_angstrom": 3.0,
                "minimum_atom_index_separation": 1,
            }],
        }},
        "requested_modules": ["optional_observables"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class ObservableTests(unittest.TestCase):
    def test_group_minimum_distance_retains_pair_identity(self):
        coordinates = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        distance, pair = minimum_group_distance(coordinates, [0, 1], [2])
        self.assertAlmostEqual(distance, 2.0)
        self.assertEqual(pair, (0, 2))

    def test_periodic_group_distance_uses_triclinic_minimum_image(self):
        cell = ((10.0, 0.0, 0.0), (4.0, 8.0, 0.0), (0.0, 0.0, 10.0))
        coordinates = [(9.5, 0.0, 0.0), (0.5, 0.0, 0.0)]
        distance, pair = minimum_group_distance(coordinates, [0], [1], cell)
        self.assertAlmostEqual(distance, 1.0)
        self.assertEqual(pair, (0, 1))

    def test_native_pairs_and_project_fraction_are_reference_defined(self):
        self.assertEqual(
            native_contact_pairs(
                [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (8.0, 0.0, 0.0)],
                [0, 1, 2], 3.0, 1,
            ),
            [(0, 1)],
        )
        with tempfile.TemporaryDirectory() as temporary:
            report = optional_observables_project(_native_project(Path(temporary)))
        feature = report["feature_reports"][0]
        self.assertEqual(feature["native_pair_count"], 1)
        self.assertEqual(
            [row["native_contact_fraction"] for row in feature["timeseries"]],
            [1.0, 0.0],
        )
        self.assertEqual(
            feature["native_pair_occupancies"][0]["contact_occupancy_fraction"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
