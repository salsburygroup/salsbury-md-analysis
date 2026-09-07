import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.execution_adapters import _apply_task_dependency_graph


class ContextCompletionContractTests(unittest.TestCase):
    def test_context_array_reports_are_resolved_without_shell_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            script = root / "run_automatic_context_stage_0_array.slurm"
            script.write_text('''#!/bin/bash
PROJECTS=(
  'project-a.json'
  'project-b.json'
)
COMMANDS=(
  'ion-atmosphere'
  'ion-geometry'
)
OUTPUTS=(
  'results/per-system/a/ion-atmosphere'
  'results/per-system/b/ion-geometry'
)
OUTPUT="${OUTPUTS[$SLURM_ARRAY_TASK_ID]}"
FINAL="$OUTPUT/report.json"
''')
            phases = _apply_task_dependency_graph(root, [{"phase_id": "chemical", "tasks": [
                {"script": script.name, "array_task_id": i, "cpu_slots": 1} for i in range(2)
            ]}])
            tasks = [task for phase in phases for task in phase["tasks"]]
            by_index = {task["array_task_id"]: task for task in tasks}
            self.assertEqual(by_index[0]["completion_reports"], ["results/per-system/a/ion-atmosphere/report.json"])
            self.assertEqual(by_index[1]["completion_reports"], ["results/per-system/b/ion-geometry/report.json"])
