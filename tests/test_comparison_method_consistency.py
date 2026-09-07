import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.campaign_planning import _consistent_comparison_clustering_tasks
from salsbury_md_analysis.planning_report import _sampling_row


class ComparisonConsistencyTests(unittest.TestCase):
    def test_rechecks_current_projection_tasks_and_preserves_unmatched_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            views = ["global_common_heavy", "system_A__global_common_heavy", "system_B__global_common_heavy", "system_A__protein_dna_interface"]
            paths = []
            for view in views:
                path = root / f"project-{view}.json"
                path.write_text(json.dumps({"requested_modules": ["alternative_clustering"],
                                            "definitions": {"alternative_clustering": {"algorithms": ["pam", "ward"]}}}))
                paths.append(path)
            tasks = [{"task_id": f"view:{view}:alternative_clustering:{algorithm}"}
                     for view in views for algorithm in ("pam", "ward")]
            partial = [row for row in tasks if row["task_id"] != "view:global_common_heavy:alternative_clustering:ward"]
            retained, skips = _consistent_comparison_clustering_tasks(partial, paths)
            self.assertEqual(len(skips), 3)
            self.assertTrue(all(row["task_id"].endswith(":pam") or "protein_dna_interface" in row["task_id"] for row in retained))
            complete, skips = _consistent_comparison_clustering_tasks(tasks, paths)
            self.assertEqual(complete, tasks)
            self.assertEqual(skips, {})

    def test_sampling_report_distinguishes_source_exhaustion_from_failure(self):
        task = {"source_frames_per_replica": [80, 100],
                "selected_physical_frames_per_replica": [80, 100],
                "source_limited_below_declared_minimum": True}
        self.assertEqual(_sampling_row(task)["sampling_floor_status"], "source_exhausted")
        task["selected_physical_frames_per_replica"] = [40, 50]
        self.assertEqual(_sampling_row(task)["sampling_floor_status"], "below_floor")
        task["scientific_sampling_assessment"] = {"keep_enabled": True, "raw_coverage_status": "source_limited_below_standard"}
        self.assertEqual(_sampling_row(task)["sampling_floor_status"], "source_limited")
        task["scientific_sampling_assessment"]["keep_enabled"] = False
        task["source_limited_below_declared_minimum"] = False
        self.assertEqual(_sampling_row(task)["sampling_floor_status"], "below_floor")
