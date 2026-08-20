import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.feature_matrix import load_feature_matrix
from salsbury_md_analysis.tica import TICAAnalysisError, fit_tica, project_tica


def _pdb_atom(serial, name, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:^4s} ALA A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _write_project(root: Path) -> Path:
    atoms = [
        _pdb_atom(1, "C", 0, 0, 0, "C"),
        _pdb_atom(2, "N", 1, 0, 0, "N"),
        _pdb_atom(3, "O", 0, 1, 0, "O"),
        _pdb_atom(4, "CB", 0, 2, 0, "C"),
        _pdb_atom(5, "CG", 1, 2, 0, "C"),
    ]
    (root / "reference.pdb").write_text("".join(atoms) + "END\n", encoding="utf-8")
    frames = []
    for index in range(24):
        slow = math.sin(2.0 * math.pi * index / 24.0)
        faster = 0.7 * math.sin(2.0 * math.pi * index / 6.0)
        frames.append(
            f"5\nframe {index}\n"
            "C 0 0 0\nN 1 0 0\nO 0 1 0\n"
            f"C {slow} 2 0\nC 1 {2 + faster} 0\n"
        )
    (root / "trajectory.xyz").write_text("".join(frames), encoding="utf-8")
    system = {
        "systems": [{
            "system_id": "system",
            "replicas": [{
                "replica_id": "r1",
                "topology": "reference.pdb",
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
        "project_id": "tica-test",
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "reject",
        "reference_structure": "reference.pdb",
        "reference_system": "system",
        "common_atom_policy": "strict",
        "selections": {
            "alignment": {"atom_names": ["C", "N", "O"]},
            "analysis": {"atom_names": ["CB", "CG"]},
        },
        "definitions": {
            "common_pca": {
                "alignment_selection": "alignment",
                "analysis_selection": "analysis",
                "minimum_reference_coverage": 1.0,
                "frame_stride": 1,
                "maximum_features": 20,
                "component_count": 2,
                "minimum_evaluated_frames_per_replica": 5,
                "basis_weighting": "frame",
            },
            "time_lagged_independent_component_analysis": {
                "feature_source": "common_pca",
                "component_indices": [1, 2],
                "lag_frames": 1,
                "component_count": 1,
                "covariance_regularization": 1.0e-8,
                "covariance_eigenvalue_cutoff": 1.0e-10,
                "minimum_pairs_per_segment": 10,
                "maximum_features": 10,
            },
        },
        "requested_modules": ["common_pca", "time_lagged_independent_component_analysis"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class TICATests(unittest.TestCase):
    def test_lag_pairs_never_join_segment_boundaries(self):
        model = fit_tica(
            [
                [[0.0], [1.0], [2.0]],
                [[100.0], [101.0], [102.0]],
            ],
            lag_frames=1,
            component_count=1,
            covariance_regularization=1.0e-8,
        )
        self.assertEqual(model["pair_count"], 4)

    def test_slowest_ar_process_is_the_leading_mode(self):
        generator = np.random.default_rng(20260811)
        observations = np.zeros((2000, 2), dtype=float)
        for index in range(1, len(observations)):
            observations[index, 0] = 0.96 * observations[index - 1, 0] + generator.normal(scale=0.25)
            observations[index, 1] = 0.20 * observations[index - 1, 1] + generator.normal(scale=1.0)
        model = fit_tica(
            [observations.tolist()],
            lag_frames=1,
            component_count=2,
            covariance_regularization=1.0e-10,
        )
        self.assertGreater(abs(model["eigenvalues"][0]), 0.9)
        self.assertGreater(abs(model["eigenvalues"][0]), abs(model["eigenvalues"][1]))
        self.assertGreater(
            abs(model["eigenvectors"][0][0]),
            abs(model["eigenvectors"][0][1]),
        )

    def test_projection_is_centered_and_dimension_checked(self):
        model = fit_tica(
            [[[-2.0], [-1.0], [0.0], [1.0], [2.0]]],
            lag_frames=1,
            component_count=1,
            covariance_regularization=1.0e-8,
        )
        scores = project_tica([[model["mean"][0]]], model["mean"], model["eigenvectors"])
        self.assertAlmostEqual(scores[0][0], 0.0)
        with self.assertRaises(TICAAnalysisError):
            project_tica([[1.0, 2.0]], model["mean"], model["eigenvectors"])

    def test_zero_variance_fails_closed(self):
        with self.assertRaises(TICAAnalysisError):
            fit_tica(
                [[[1.0], [1.0], [1.0]]],
                lag_frames=1,
                component_count=1,
            )

    def test_cli_runs_segment_safe_tica_from_common_pca_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary))
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["tica", str(path)])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0, report)
        self.assertEqual(report["module_id"], "time_lagged_independent_component_analysis")
        self.assertEqual(report["pair_count"], 23)
        self.assertEqual(report["lag_time"], 2.0)
        self.assertEqual(len(report["components"]), 1)
        self.assertEqual(len(report["segments"][0]["projections"]), 24)

    def test_tica_projections_are_available_as_clustering_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary))
            project = json.loads(path.read_text(encoding="utf-8"))
            project["definitions"]["time_lagged_independent_component_analysis"][
                "component_count"
            ] = 2
            path.write_text(json.dumps(project), encoding="utf-8")
            report, metadata, vectors, contract = load_feature_matrix(
                path,
                {"feature_source": "tica", "component_indices": [1, 2]},
                hash_content=False,
                error_type=ValueError,
            )
        self.assertEqual(report["module_id"], "time_lagged_independent_component_analysis")
        self.assertEqual(len(metadata), 24)
        self.assertEqual(len(vectors), 24)
        self.assertEqual(len(vectors[0]), 2)
        self.assertEqual(contract["source"], "tica")
        self.assertEqual(contract["columns"][0]["label"], "tIC1")


if __name__ == "__main__":
    unittest.main()
