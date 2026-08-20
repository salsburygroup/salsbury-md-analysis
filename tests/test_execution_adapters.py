import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.execution_adapters import (
    ExecutionAdapterError,
    apply_slurm_profile,
    build_local_execution_plan,
    load_slurm_profile,
    run_local_workflow,
)


class ExecutionAdapterTests(unittest.TestCase):
    def test_deac_profile_is_valid_and_injects_scheduler_contract(self):
        repository = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repository / "profiles/slurm/deac.json")
        self.assertEqual(profile["account"], "salsburygrp")
        self.assertEqual(profile["unix_group"], "salsburyGrp")
        self.assertEqual(profile["partitions"]["analysis"], "small")
        self.assertIn(
            "/software/salsbury-md-analysis/environments/v76/",
            profile["environment"]["python_executable"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "slurm-profile.json").write_text("{}\n", encoding="utf-8")
            (root / "run_stage_0_array.slurm").write_text(
                "#!/usr/bin/env bash\n"
                "#SBATCH --time=01:00:00\n"
                "#SBATCH --cpus-per-task=1\n"
                "set -euo pipefail\n"
                "PYTHON_DEFAULT=/old/python\n"
                "PACKAGE_ROOT_DEFAULT=/old/source\n",
                encoding="utf-8",
            )
            (root / "submit.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nsbatch worker.slurm\n",
                encoding="utf-8",
            )
            scheduler = apply_slurm_profile(root, profile, {
                "phases": [{"phase_id": "analysis", "tasks": [{
                    "script": "run_stage_0_array.slurm",
                    "array_task_id": 0,
                    "requested_wall_minutes": 47,
                    "requested_memory_gib": 7.1,
                    "planner_task_ids": ["direct:structural_integrity_qc"],
                    "cpu_slots": 1,
                    "planned_wall_hours": 0.5,
                    "planned_peak_memory_gib": 4,
                    "resource_request_source": "unit_test",
                    "wall_request_limited_by_campaign_cap": False,
                    "memory_request_limited_by_campaign_cap": False,
                }]}],
            })
            worker = (root / "run_stage_0_array.slurm").read_text(encoding="utf-8")
            submit = (root / "submit.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --account=salsburygrp", worker)
        self.assertIn("#SBATCH --partition=small", worker)
        self.assertIn("#SBATCH --qos=normal", worker)
        self.assertIn("umask 0002", worker)
        self.assertIn("SALSBURY_CLUSTER_PROFILE=deac", worker)
        self.assertIn("SALSBURY_UNIX_GROUP=salsburyGrp", worker)
        self.assertIn("SALSBURY_GROUP_STORAGE_ROOT=/deac/phy/salsburyGrp", worker)
        self.assertIn("python3.12", worker)
        self.assertIn("#SBATCH --time=00:47:00", worker)
        self.assertIn("#SBATCH --mem=8G", worker)
        self.assertEqual(
            scheduler["scripts"]["run_stage_0_array.slurm"]["selected_partition"],
            "small",
        )
        self.assertIn("/opt/scyld/slurm/bin/sbatch worker.slurm", submit)

    def test_profile_rejects_shell_text_in_submit_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.json"
            path.write_text(json.dumps({
                "slurm_profile_schema": "salsbury-slurm-profile-v1",
                "profile_id": "unsafe",
                "cluster_name": "test",
                "submit_command": "sbatch --help",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ExecutionAdapterError, "one executable"):
                load_slurm_profile(path)

    def test_profile_rejects_additional_resource_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "override.json"
            path.write_text(json.dumps({
                "slurm_profile_schema": "salsbury-slurm-profile-v1",
                "profile_id": "override",
                "cluster_name": "test",
                "additional_sbatch_directives": ["#SBATCH --mem=999G"],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ExecutionAdapterError, "managed resources"):
                load_slurm_profile(path)

    def test_planner_rows_drive_task_resources_and_large_partition(self):
        repository = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repository / "profiles/slurm/deac.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = (
                "#!/usr/bin/env bash\n#SBATCH --time=06:00:00\n"
                "#SBATCH --cpus-per-task=1\n#SBATCH --mem=32G\n"
                "set -euo pipefail\n"
            )
            (root / "run_preflight.slurm").write_text(common, encoding="utf-8")
            (root / "run_finalize_reporting.slurm").write_text(
                common, encoding="utf-8"
            )
            (root / "run_stage_0_array.slurm").write_text(
                common + 'COMMANDS=(\n  "structural-qc"\n)\n', encoding="utf-8"
            )
            context = root / "run_automatic_context_stage_0_array.slurm"
            context.write_text(
                common
                + "PROJECTS=(\n  'project-chemical_a.json'\n)\n"
                + "COMMANDS=(\n  'ion-atmosphere'\n)\n",
                encoding="utf-8",
            )
            view = root / "run_view_global_common_heavy_stage_1.slurm"
            view.write_text(
                common + 'COMMANDS=(\n  "alternative-clustering"\n)\n',
                encoding="utf-8",
            )
            (root / "campaign-resource-plan.json").write_text(json.dumps({
                "tasks": [
                    {
                        "task_id": "direct:structural_integrity_qc",
                        "module_id": "structural_integrity_qc",
                        "estimated_wall_hours_at_effective_cpu_cap": 0.5,
                        "estimated_peak_memory_gib_at_selected_observations": 4,
                    },
                    {
                        "task_id": "view:global_common_heavy:alternative_clustering:pam",
                        "workflow_id": "global_common_heavy",
                        "module_id": "alternative_clustering",
                        "estimated_wall_hours_at_effective_cpu_cap": 0.5,
                        "estimated_peak_memory_gib_at_selected_observations": 60,
                    },
                    {
                        "task_id": "view:global_common_heavy:alternative_clustering:gmm",
                        "workflow_id": "global_common_heavy",
                        "module_id": "alternative_clustering",
                        "estimated_wall_hours_at_effective_cpu_cap": 0.6,
                        "estimated_peak_memory_gib_at_selected_observations": 100,
                    },
                    {
                        "task_id": "context:chemical_a:ion_atmosphere",
                        "workflow_id": "chemical_a",
                        "module_id": "ion_atmosphere",
                        "estimated_wall_hours_at_effective_cpu_cap": 0.2,
                        "estimated_peak_memory_gib_at_selected_observations": 12,
                    },
                ]
            }), encoding="utf-8")
            execution = {
                "maximum_parallel_cpus": 2,
                "maximum_hours_per_cpu": 24,
                "maximum_memory_gib": 128,
            }
            reporting = {
                "resource_table_enabled": False,
                "finding_picker_enabled": False,
            }
            plan = build_local_execution_plan(
                root, execution, reporting, profile["resource_policy"]
            )
            tasks = {
                (task["script"], task.get("array_task_id")): task
                for phase in plan["phases"] for task in phase["tasks"]
            }
            direct = tasks[("run_stage_0_array.slurm", 0)]
            chemistry = tasks[(context.name, 0)]
            alternative = tasks[(view.name, 0)]
            self.assertEqual(direct["requested_memory_gib"], 7)
            self.assertEqual(
                chemistry["planner_task_ids"],
                ["context:chemical_a:ion_atmosphere"],
            )
            self.assertEqual(alternative["planned_wall_hours"], 1.1)
            self.assertEqual(alternative["planned_peak_memory_gib"], 100)
            scheduler = apply_slurm_profile(root, profile, plan)
            view_text = view.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=large", view_text)
        self.assertIn("#SBATCH --mem=128G", view_text)
        self.assertEqual(
            scheduler["scripts"][view.name]["selected_partition_role"],
            "large_memory",
        )

    def test_local_runner_preserves_dependencies_and_stops_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            first = root / "first.slurm"
            second = root / "second.slurm"
            never = root / "never.slurm"
            first.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nprintf ready > marker\n",
                encoding="utf-8",
            )
            second.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\ntest -f marker\nexit 7\n",
                encoding="utf-8",
            )
            never.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nprintf bad > should-not-exist\n",
                encoding="utf-8",
            )
            plan = {
                "local_execution_plan_schema": "salsbury-local-execution-plan-v1",
                "maximum_parallel_cpus": 2,
                "maximum_campaign_wall_hours": 1,
                "phases": [
                    {"phase_id": "first", "tasks": [{
                        "script": "first.slurm", "array_task_id": None, "cpu_slots": 1,
                    }]},
                    {"phase_id": "failure", "tasks": [{
                        "script": "second.slurm", "array_task_id": None, "cpu_slots": 1,
                    }]},
                    {"phase_id": "never", "tasks": [{
                        "script": "never.slurm", "array_task_id": None, "cpu_slots": 1,
                    }]},
                ],
            }
            (root / "local-execution-plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            report = run_local_workflow(root)
            retained = list((root / "local-execution-status").glob("*.json"))
            self.assertEqual(report["technical_status"], "failed")
            self.assertEqual(report["remaining_phases_not_run"], ["never"])
            self.assertFalse((root / "should-not-exist").exists())
            self.assertEqual(len(retained), 1)
            self.assertEqual(
                json.loads(retained[0].read_text())["technical_status"], "failed"
            )

    def test_local_runner_executes_arrays_and_reuses_complete_finalizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            array = root / "array.slurm"
            finalizer = root / "finalizer.slurm"
            array.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "printf '%s' \"$SLURM_CPUS_PER_TASK\" > \"array-$SLURM_ARRAY_TASK_ID\"\n",
                encoding="utf-8",
            )
            finalizer.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "printf '{\"technical_status\":\"complete\"}\\n' > final.json\n",
                encoding="utf-8",
            )
            plan = {
                "local_execution_plan_schema": "salsbury-local-execution-plan-v1",
                "maximum_parallel_cpus": 2,
                "maximum_campaign_wall_hours": 1,
                "phases": [
                    {"phase_id": "array", "tasks": [
                        {"script": "array.slurm", "array_task_id": index, "cpu_slots": 1}
                        for index in range(2)
                    ]},
                    {"phase_id": "final", "tasks": [{
                        "script": "finalizer.slurm", "array_task_id": None,
                        "cpu_slots": 1, "completion_reports": ["final.json"],
                    }]},
                ],
            }
            (root / "local-execution-plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            first = run_local_workflow(root)
            second = run_local_workflow(root)
            retained = list((root / "local-execution-status").glob("*.json"))
        self.assertEqual(first["technical_status"], "complete")
        self.assertEqual(second["technical_status"], "complete")
        self.assertEqual(
            second["phase_reports"][-1]["tasks"][0]["status"],
            "reused_complete",
        )
        self.assertEqual(len(retained), 2)

    def test_local_runner_reserves_memory_across_parallel_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            worker = root / "worker.slurm"
            worker.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "mkdir resource-lock\n"
                "test \"$SLURM_MEM_PER_NODE\" = 2048\n"
                "sleep 0.1\n"
                "rmdir resource-lock\n",
                encoding="utf-8",
            )
            plan = {
                "local_execution_plan_schema": "salsbury-local-execution-plan-v2",
                "maximum_parallel_cpus": 2,
                "maximum_parallel_memory_gib": 3,
                "maximum_campaign_wall_hours": 1,
                "phases": [{"phase_id": "memory", "tasks": [
                    {
                        "script": "worker.slurm", "array_task_id": None,
                        "cpu_slots": 1, "requested_memory_gib": 2,
                        "requested_wall_minutes": 1,
                    },
                    {
                        "script": "worker.slurm", "array_task_id": None,
                        "cpu_slots": 1, "requested_memory_gib": 2,
                        "requested_wall_minutes": 1,
                    },
                ]}],
            }
            (root / "local-execution-plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            report = run_local_workflow(root)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["maximum_parallel_memory_gib"], 3)


if __name__ == "__main__":
    unittest.main()
