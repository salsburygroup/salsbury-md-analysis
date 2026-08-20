import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.representative_frames import (
    representative_frames_project,
    select_state_representatives,
)


def _project(path: Path, source: str, representatives_per_state: int = 1) -> Path:
    path.write_text(json.dumps({
        "definitions": {
            "representative_frames": {
                "source": source,
                "representatives_per_state": representatives_per_state,
                "maximum_states": 10,
                "maximum_candidates": 100,
            }
        }
    }), encoding="utf-8")
    return path


def _lineage() -> dict:
    return {
        "technical_status": "complete",
        "project_manifest_sha256": "a" * 64,
        "system_manifest_path": "/data/system.json",
        "system_manifest_sha256": "b" * 64,
        "input_content_signature_sha256": "c" * 64,
        "issues": [],
    }


class RepresentativeFrameTests(unittest.TestCase):
    def test_nearest_selection_is_state_sorted_and_identity_tie_broken(self):
        candidates = [
            {
                "system_id": "z", "replica_id": "r1", "segment_id": "s1",
                "source_frame_index": 5, "sample_index": 5,
                "cluster_id": 1, "distance": 0.5,
            },
            {
                "system_id": "a", "replica_id": "r1", "segment_id": "s1",
                "source_frame_index": 7, "sample_index": 7,
                "cluster_id": 1, "distance": 0.5,
            },
            {
                "system_id": "a", "replica_id": "r1", "segment_id": "s1",
                "source_frame_index": 1, "sample_index": 1,
                "cluster_id": 2, "distance": 0.1,
            },
        ]
        selected = select_state_representatives(
            candidates,
            state_field="cluster_id",
            distance_field="distance",
            representatives_per_state=2,
            maximum_states=2,
            maximum_candidates=3,
        )
        self.assertEqual(
            [(row["state_id"], row["system_id"]) for row in selected],
            [(1, "a"), (1, "z"), (2, "a")],
        )

    def test_kmeans_project_reports_observed_frame_locators_only(self):
        upstream = {
            **_lineage(),
            "assignments": [
                {
                    "system_id": "sys", "replica_id": "r1", "segment_id": "s1",
                    "source_frame_index": 0, "time": 0.0, "time_unit": "ps",
                    "cluster_id": 1, "squared_distance_in_clustering_space": 0.2,
                },
                {
                    "system_id": "sys", "replica_id": "r1", "segment_id": "s1",
                    "source_frame_index": 1, "time": 1.0, "time_unit": "ps",
                    "cluster_id": 1, "squared_distance_in_clustering_space": 0.1,
                },
                {
                    "system_id": "sys", "replica_id": "r1", "segment_id": "s1",
                    "source_frame_index": 2, "time": 2.0, "time_unit": "ps",
                    "cluster_id": 2, "squared_distance_in_clustering_space": 0.3,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = _project(
                Path(temporary) / "project.json", "clustering_kmeans"
            )
            with patch(
                "salsbury_md_analysis.representative_frames.clustering_kmeans_project",
                return_value=upstream,
            ):
                report = representative_frames_project(path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["representative_count"], 2)
        self.assertEqual(
            report["observation_accounting"]["selected_physical_frame_count"],
            3,
        )
        self.assertEqual(
            report["observation_accounting"]["symmetry_expanded_observation_count"],
            3,
        )
        self.assertEqual(
            report["observation_accounting"]["representative_physical_frame_count"],
            2,
        )
        self.assertEqual(
            [row["source_frame_index"] for row in report["representatives"]],
            [1, 2],
        )
        self.assertEqual(report["coordinate_files_written"], 0)

    def test_pca_basin_project_uses_declared_basin_roots(self):
        upstream = {
            **_lineage(),
            "landscape": {"basins": [
                {
                    "basin_id": 1,
                    "root_x_center_angstrom": -1.0,
                    "root_y_center_angstrom": -1.0,
                },
                {
                    "basin_id": 2,
                    "root_x_center_angstrom": 2.0,
                    "root_y_center_angstrom": 2.0,
                },
            ]},
            "frame_assignments": [
                {
                    "system_id": "sys", "replica_id": "r1", "segment_id": "s1",
                    "source_frame_index": 0, "sample_index": 10,
                    "pc_x_angstrom": -1.1, "pc_y_angstrom": -1.0, "basin_id": 1,
                },
                {
                    "system_id": "sys", "replica_id": "r1", "segment_id": "s1",
                    "source_frame_index": 1, "sample_index": 11,
                    "pc_x_angstrom": -1.5, "pc_y_angstrom": -1.0, "basin_id": 1,
                },
                {
                    "system_id": "sys", "replica_id": "r1", "segment_id": "s1",
                    "source_frame_index": 2, "sample_index": 12,
                    "pc_x_angstrom": 2.1, "pc_y_angstrom": 2.1, "basin_id": 2,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = _project(Path(temporary) / "project.json", "pca_fes_basins")
            with patch(
                "salsbury_md_analysis.representative_frames.pca_fes_basins_project",
                return_value=upstream,
            ):
                report = representative_frames_project(path)
        self.assertEqual(
            [row["sample_index"] for row in report["representatives"]],
            [10, 12],
        )

    def test_cli_emits_machine_readable_report(self):
        upstream = {
            **_lineage(),
            "assignments": [{
                "system_id": "sys", "replica_id": "r1", "segment_id": "s1",
                "source_frame_index": 0, "sample_index": 0,
                "cluster_id": 1, "squared_distance_in_clustering_space": 0.0,
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = _project(
                Path(temporary) / "project.json", "clustering_kmeans"
            )
            output = io.StringIO()
            with patch(
                "salsbury_md_analysis.representative_frames.clustering_kmeans_project",
                return_value=upstream,
            ), redirect_stdout(output):
                status = main(["representative-frames", str(path)])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["module_id"], "representative_frames")


if __name__ == "__main__":
    unittest.main()
