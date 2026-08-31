import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.execution_adapters import (
    ExecutionAdapterError,
    _active_python_executable,
    _append_afterany_dependencies,
    _render_resource_bounded_submit,
    apply_slurm_profile,
    build_local_execution_plan,
    load_slurm_profile,
    run_local_workflow,
)


class ExecutionAdapterTests(unittest.TestCase):
    def test_distributed_replica_launcher_spans_configured_nodes(self):
        repository = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repository / "profiles/slurm/deac.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_path = root / "slurm-profile.json"
            profile_path.write_text("{}\n", encoding="utf-8")
            script = _render_resource_bounded_submit(
                root,
                profile,
                profile_path,
                [{
                    "lane_index": 0,
                    "items": [{
                        "submission_index": 0,
                        "phase_id": "analysis",
                        "task_id": "replica-qc",
                        "script": "run_qc.slurm",
                        "cpu_slots": 63,
                        "memory_gib": 362.0,
                        "node_count": 2,
                        "workers_per_node": 40,
                        "distributed_worker_count": 63,
                        "distributed_replica_execution": True,
                        "slurm_time": "01:00:00",
                        "slurm_memory": "181G",
                        "depends_on_task_ids": [],
                        "wait_for_task_ids": [],
                    }],
                }],
            )
        self.assertIn("--nodes=2", script)
        self.assertIn("--ntasks=63", script)
        self.assertIn("--ntasks-per-node=40", script)
        self.assertIn("--cpus-per-task=1", script)
        self.assertIn("--mem=181G", script)
        self.assertIn("SMA_DISTRIBUTED_REPLICA_WORKERS=1", script)

    def test_resource_barrier_does_not_turn_an_input_gate_into_a_success_chain(self):
        self.assertEqual(
            _append_afterany_dependencies(
                '--parsable --dependency="afterok:$INPUT_JOB"',
                ["FIRST_TIER", "SECOND_TIER"],
            ),
            '--parsable --dependency="afterok:$INPUT_JOB,'
            'afterany:${FIRST_TIER}:${SECOND_TIER}"',
        )
        self.assertEqual(
            _append_afterany_dependencies("--parsable", ["FIRST_TIER"]),
            '--parsable --dependency="afterany:${FIRST_TIER}"',
        )

    def test_active_python_path_preserves_virtual_environment_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base-python"
            base.write_text("", encoding="utf-8")
            virtual_environment_python = root / "venv" / "bin" / "python"
            virtual_environment_python.parent.mkdir(parents=True)
            virtual_environment_python.symlink_to(base)
            with patch(
                "salsbury_md_analysis.execution_adapters.os.sys.executable",
                str(virtual_environment_python),
            ):
                self.assertEqual(
                    _active_python_executable(),
                    str(virtual_environment_python.absolute()),
                )
                self.assertNotEqual(
                    _active_python_executable(),
                    str(virtual_environment_python.resolve()),
                )

    def test_deac_profile_is_valid_and_injects_scheduler_contract(self):
        repository = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repository / "profiles/slurm/deac.json")
        self.assertEqual(profile["account"], "salsburygrp")
        self.assertEqual(profile["unix_group"], "salsburyGrp")
        self.assertEqual(profile["partitions"]["analysis"], "small")
        self.assertEqual(profile["partitions"]["long_wall"], "large")
        self.assertEqual(profile["node_policy"]["cpus_per_node"], 44)
        self.assertEqual(profile["node_policy"]["memory_gib_per_node"], 185.0)
        self.assertEqual(
            profile["partition_maximum_wall_minutes"]["small"], 1440.0,
        )
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
                "maximum_parallel_cpus": 1,
                "maximum_parallel_memory_gib": 8,
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
        self.assertIn("#SBATCH --nodes=1", worker)
        self.assertEqual(
            scheduler["scripts"]["run_stage_0_array.slurm"]["selected_partition"],
            "small",
        )
        self.assertIn("SUBMIT_COMMAND=/opt/scyld/slurm/bin/sbatch", submit)
        self.assertIn('"$ROOT"/run_stage_0_array.slurm', submit)

    def test_deac_profile_routes_requests_over_small_limit_to_long_wall(self):
        repository = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repository / "profiles/slurm/deac.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "slurm-profile.json").write_text("{}\n", encoding="utf-8")
            worker = root / "run_stage_0_array.slurm"
            worker.write_text(
                "#!/usr/bin/env bash\n"
                "#SBATCH --time=1-00:01:00\n"
                "#SBATCH --cpus-per-task=1\n"
                "#SBATCH --mem=8G\n"
                "set -euo pipefail\n",
                encoding="utf-8",
            )
            scheduler = apply_slurm_profile(root, profile, {
                "maximum_parallel_cpus": 1,
                "maximum_parallel_memory_gib": 8,
                "phases": [{"phase_id": "analysis", "tasks": [{
                    "script": worker.name,
                    "array_task_id": 0,
                    "requested_wall_minutes": 1441,
                    "requested_memory_gib": 8,
                    "planner_task_ids": ["direct:test"],
                    "cpu_slots": 1,
                    "planned_wall_hours": 24.0,
                    "planned_peak_memory_gib": 8,
                    "resource_request_source": "unit_test",
                    "wall_request_limited_by_campaign_cap": False,
                    "memory_request_limited_by_campaign_cap": False,
                }]}],
            })
            text = worker.read_text(encoding="utf-8")
        request = scheduler["scripts"][worker.name]
        self.assertIn("#SBATCH --partition=large", text)
        self.assertEqual(request["selected_partition_role"], "long_wall")
        self.assertTrue(request["long_wall_routed"])
        self.assertEqual(request["selected_partition_maximum_wall_minutes"], 259200.0)

    def test_profile_fails_closed_when_long_request_has_no_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            profile_path = Path(temporary) / "profile.json"
            profile_path.write_text(json.dumps({
                "slurm_profile_schema": "salsbury-slurm-profile-v1",
                "profile_id": "bounded",
                "cluster_name": "bounded",
                "partitions": {"analysis": "short"},
                "partition_maximum_wall_minutes": {"short": 60},
            }), encoding="utf-8")
            profile = load_slurm_profile(profile_path)
            root = Path(temporary) / "run"
            root.mkdir()
            (root / "slurm-profile.json").write_text("{}\n", encoding="utf-8")
            worker = root / "run_stage_0_array.slurm"
            worker.write_text(
                "#!/usr/bin/env bash\n#SBATCH --time=02:00:00\n"
                "#SBATCH --cpus-per-task=1\n#SBATCH --mem=2G\n"
                "set -euo pipefail\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ExecutionAdapterError, "partitions.long_wall is unset"
            ):
                apply_slurm_profile(root, profile, {
                    "maximum_parallel_cpus": 1,
                    "maximum_parallel_memory_gib": 2,
                    "phases": [{"phase_id": "analysis", "tasks": [{
                        "script": worker.name,
                        "array_task_id": 0,
                        "requested_wall_minutes": 120,
                        "requested_memory_gib": 2,
                        "planner_task_ids": ["direct:test"],
                        "cpu_slots": 1,
                        "planned_wall_hours": 2,
                        "planned_peak_memory_gib": 2,
                        "resource_request_source": "unit_test",
                        "wall_request_limited_by_campaign_cap": False,
                        "memory_request_limited_by_campaign_cap": False,
                    }]}],
                })

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
                        "estimated_scheduler_memory_gib_at_selected_observations": 7,
                        "estimated_scheduler_memory_gib_per_node_at_selected_observations": 7,
                    },
                    {
                        "task_id": "view:global_common_heavy:alternative_clustering:pam",
                        "workflow_id": "global_common_heavy",
                        "module_id": "alternative_clustering",
                        "estimated_wall_hours_at_effective_cpu_cap": 0.5,
                        "estimated_peak_memory_gib_at_selected_observations": 60,
                        "estimated_scheduler_memory_gib_at_selected_observations": 91,
                        "estimated_scheduler_memory_gib_per_node_at_selected_observations": 91,
                    },
                    {
                        "task_id": "view:global_common_heavy:alternative_clustering:gmm",
                        "workflow_id": "global_common_heavy",
                        "module_id": "alternative_clustering",
                        "estimated_wall_hours_at_effective_cpu_cap": 0.6,
                        "estimated_peak_memory_gib_at_selected_observations": 100,
                        "estimated_scheduler_memory_gib_at_selected_observations": 151,
                        "estimated_scheduler_memory_gib_per_node_at_selected_observations": 151,
                    },
                    {
                        "task_id": "context:chemical_a:ion_atmosphere",
                        "workflow_id": "chemical_a",
                        "module_id": "ion_atmosphere",
                        "estimated_wall_hours_at_effective_cpu_cap": 0.2,
                        "estimated_peak_memory_gib_at_selected_observations": 12,
                        "estimated_scheduler_memory_gib_at_selected_observations": 19,
                        "estimated_scheduler_memory_gib_per_node_at_selected_observations": 19,
                    },
                ]
            }), encoding="utf-8")
            execution = {
                "maximum_parallel_cpus": 2,
                "maximum_hours_per_cpu": 24,
                "maximum_memory_gib": 256,
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
                direct["resource_request_source"],
                "campaign_planner_final_memory_reservation_passthrough",
            )
            self.assertEqual(
                chemistry["planner_task_ids"],
                ["context:chemical_a:ion_atmosphere"],
            )
            self.assertEqual(alternative["planned_wall_hours"], 1.1)
            self.assertEqual(alternative["planned_peak_memory_gib"], 100)
            scheduler = apply_slurm_profile(root, profile, plan)
            view_text = view.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=large", view_text)
        self.assertIn("#SBATCH --mem=151G", view_text)
        self.assertEqual(
            scheduler["scripts"][view.name]["selected_partition_role"],
            "large_memory",
        )

    def test_execution_adapter_does_not_repeat_planner_memory_adjustment(self):
        repository = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repository / "profiles/slurm/deac.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = (
                "#!/usr/bin/env bash\n#SBATCH --time=01:00:00\n"
                "#SBATCH --cpus-per-task=1\n#SBATCH --mem=2G\nset -euo pipefail\n"
            )
            (root / "run_preflight.slurm").write_text(
                common, encoding="utf-8"
            )
            (root / "run_finalize_reporting.slurm").write_text(
                common, encoding="utf-8"
            )
            worker = root / "run_stage_0_array.slurm"
            worker.write_text(
                common + "COMMANDS=(\n  'structural-qc'\n)\n",
                encoding="utf-8",
            )
            (root / "campaign-resource-plan.json").write_text(json.dumps({
                "tasks": [{
                    "task_id": "direct:structural_integrity_qc",
                    "module_id": "structural_integrity_qc",
                    "estimated_wall_hours_at_effective_cpu_cap": 0.5,
                    "estimated_peak_memory_gib_at_selected_observations": 10,
                    "estimated_scheduler_memory_gib_at_selected_observations": 16,
                    "estimated_scheduler_memory_gib_per_node_at_selected_observations": 16,
                }],
            }), encoding="utf-8")
            execution = {
                "maximum_parallel_cpus": 1,
                "maximum_hours_per_cpu": 24,
                "maximum_memory_gib": 32,
            }
            reporting = {
                "resource_table_enabled": False,
                "finding_picker_enabled": False,
            }
            deliberately_different_adapter_policy = {
                **profile["resource_policy"],
                "memory_safety_factor": 9.0,
                "memory_overhead_gib": 8.0,
            }
            plan = build_local_execution_plan(
                root,
                execution,
                reporting,
                deliberately_different_adapter_policy,
                profile["node_policy"],
            )
            task = next(
                task
                for phase in plan["phases"]
                for task in phase["tasks"]
                if task["script"] == worker.name
            )
        self.assertEqual(task["planned_peak_memory_gib"], 10)
        self.assertEqual(task["requested_memory_gib"], 16)
        self.assertEqual(
            task["resource_request_source"],
            "campaign_planner_final_memory_reservation_passthrough",
        )

    def test_memory_limited_replica_workers_keep_planned_execution_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = (
                "#!/usr/bin/env bash\n#SBATCH --time=01:00:00\n"
                "#SBATCH --cpus-per-task=12\n#SBATCH --mem=2G\nset -euo pipefail\n"
            )
            (root / "run_preflight.slurm").write_text(common, encoding="utf-8")
            (root / "run_finalize_reporting.slurm").write_text(
                common, encoding="utf-8"
            )
            worker = root / "run_stage_0_array.slurm"
            worker.write_text(
                common + "COMMANDS=(\n  'structural-qc'\n)\n",
                encoding="utf-8",
            )
            (root / "campaign-resource-plan.json").write_text(json.dumps({
                "tasks": [{
                    "task_id": "direct:structural_integrity_qc",
                    "module_id": "structural_integrity_qc",
                    "effective_cpu_cap": 12,
                    "estimated_wall_hours_at_effective_cpu_cap": 2.0,
                    "estimated_peak_memory_gib_at_selected_observations": 18.0,
                    "estimated_scheduler_memory_gib_at_selected_observations": 28.0,
                    "estimated_scheduler_memory_gib_per_node_at_selected_observations": 28.0,
                    "parallel_node_layout_at_selected_observations": {
                        "active_worker_count": 6,
                        "execution_cpu_slots": 6,
                        "node_count": 1,
                        "workers_per_node": 6,
                        "distributed_replica_execution": False,
                    },
                }],
            }), encoding="utf-8")
            plan = build_local_execution_plan(
                root,
                {
                    "maximum_parallel_cpus": 12,
                    "maximum_hours_per_cpu": 8,
                    "maximum_memory_gib": 64,
                },
                {
                    "resource_table_enabled": False,
                    "finding_picker_enabled": False,
                },
                node_policy={
                    "cpus_per_node": 44,
                    "memory_gib_per_node": 185.0,
                },
            )

        task = next(
            task
            for phase in plan["phases"]
            for task in phase["tasks"]
            if task["script"] == worker.name
        )
        self.assertEqual(task["cpu_slots"], 6)
        self.assertEqual(task["distributed_worker_count"], 6)
        self.assertFalse(task["distributed_replica_execution"])

    def test_generated_plan_uses_only_true_task_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = (
                "#!/usr/bin/env bash\n#SBATCH --time=01:00:00\n"
                "#SBATCH --cpus-per-task=1\n#SBATCH --mem=2G\n"
                "set -euo pipefail\n"
            )
            (root / "run_preflight.slurm").write_text(worker, encoding="utf-8")
            (root / "run_stage_0_array.slurm").write_text(
                worker + "COMMANDS=(\n  'structural-qc'\n  'common-pca'\n)\n",
                encoding="utf-8",
            )
            (root / "run_stage_1_array.slurm").write_text(
                worker + "COMMANDS=(\n  'pca-fes-basins'\n)\n",
                encoding="utf-8",
            )
            (root / "run_finalize_reporting.slurm").write_text(
                worker, encoding="utf-8"
            )
            (root / "analysis-config.json").write_text(json.dumps({
                "modules": {
                    "structural_integrity_qc": {"depends_on": []},
                    "common_pca": {"depends_on": []},
                    "pca_fes_basins": {"depends_on": ["common_pca"]},
                }
            }), encoding="utf-8")
            plan = build_local_execution_plan(root, {
                "maximum_parallel_cpus": 2,
                "maximum_hours_per_cpu": 8,
                "maximum_memory_gib": 8,
            }, {
                "resource_table_enabled": False,
                "finding_picker_enabled": False,
            })

        tasks = {
            task.get("module_id") or task.get("source_phase_id"): task
            for phase in plan["phases"] for task in phase["tasks"]
        }
        preflight = next(
            task for phase in plan["phases"] for task in phase["tasks"]
            if task["script"] == "run_preflight.slurm"
        )
        qc = tasks["structural_integrity_qc"]
        pca = tasks["common_pca"]
        fes = tasks["pca_fes_basins"]
        finalizer = next(
            task for phase in plan["phases"] for task in phase["tasks"]
            if task["script"] == "run_finalize_reporting.slurm"
        )
        self.assertEqual(plan["dependency_model"], "task_dag_v1")
        self.assertEqual(qc["depends_on_task_ids"], [preflight["task_id"]])
        self.assertEqual(pca["depends_on_task_ids"], [preflight["task_id"]])
        self.assertTrue(
            qc["ensemble_parallelism_contract"][
                "replica_shard_may_finalize_primary_result"
            ]
        )
        self.assertFalse(
            pca["ensemble_parallelism_contract"][
                "replica_shard_may_finalize_primary_result"
            ]
        )
        self.assertEqual(
            pca["ensemble_parallelism_contract"]["primary_estimator_scope"],
            "declared_view_all_systems_and_replicas",
        )
        self.assertNotIn(qc["task_id"], fes["depends_on_task_ids"])
        self.assertEqual(
            set(fes["depends_on_task_ids"]),
            {preflight["task_id"]},
        )
        self.assertEqual(fes["wait_for_task_ids"], [pca["task_id"]])
        self.assertEqual(finalizer["depends_on_task_ids"], [])
        self.assertEqual(
            set(finalizer["wait_for_task_ids"]),
            {
                preflight["task_id"], qc["task_id"], pca["task_id"],
                fes["task_id"],
            },
        )

    def test_coordinate_cache_gates_only_cache_backed_view_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = (
                "#!/usr/bin/env bash\n#SBATCH --time=01:00:00\n"
                "#SBATCH --cpus-per-task=1\n#SBATCH --mem=2G\n"
                "set -euo pipefail\n"
            )
            (root / "run_coordinate_cache.slurm").write_text(worker, encoding="utf-8")
            (root / "run_preflight.slurm").write_text(worker, encoding="utf-8")
            (root / "run_stage_0_array.slurm").write_text(
                worker + "COMMANDS=(\n  'structural-qc'\n)\n", encoding="utf-8"
            )
            cache_manifest = root / "coordinate-cache/system-cache.json"
            preflight_report = root / "preflight-system-cache.report.json"
            (root / "run_view_preflight_0.slurm").write_text(
                worker + f"MANIFEST={cache_manifest}\nFINAL={preflight_report}\n",
                encoding="utf-8",
            )
            (root / "project-cached.json").write_text(json.dumps({
                "system_manifest": "coordinate-cache/system-cache.json",
                "definitions": {"common_pca": {}},
            }), encoding="utf-8")
            (root / "run_view_cached_stage_0.slurm").write_text(
                worker + f"PROJECT={root / 'project-cached.json'}\n"
                + "COMMANDS=(\n  'common-pca'\n)\n",
                encoding="utf-8",
            )
            (root / "run_finalize_reporting.slurm").write_text(worker, encoding="utf-8")
            (root / "analysis-config.json").write_text(json.dumps({
                "modules": {
                    "structural_integrity_qc": {"depends_on": []},
                    "common_pca": {"depends_on": []},
                }
            }), encoding="utf-8")
            plan = build_local_execution_plan(root, {
                "maximum_parallel_cpus": 2,
                "maximum_hours_per_cpu": 8,
                "maximum_memory_gib": 8,
            }, {"resource_table_enabled": False, "finding_picker_enabled": False})

        tasks = {
            (task["script"], task.get("array_task_id")): task
            for phase in plan["phases"] for task in phase["tasks"]
        }
        cache = tasks[("run_coordinate_cache.slurm", None)]
        preflight = tasks[("run_preflight.slurm", None)]
        qc = tasks[("run_stage_0_array.slurm", 0)]
        view_preflight = tasks[("run_view_preflight_0.slurm", None)]
        view_pca = tasks[("run_view_cached_stage_0.slurm", 0)]
        self.assertEqual(preflight["depends_on_task_ids"], [])
        self.assertEqual(
            set(qc["depends_on_task_ids"]),
            {preflight["task_id"], cache["task_id"]},
        )
        self.assertEqual(
            view_preflight["depends_on_task_ids"], [cache["task_id"]]
        )
        self.assertEqual(
            view_pca["depends_on_task_ids"], [view_preflight["task_id"]]
        )

    def test_experimental_aggregators_wait_without_false_success_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = (
                "#!/usr/bin/env bash\n#SBATCH --time=01:00:00\n"
                "#SBATCH --cpus-per-task=1\n#SBATCH --mem=2G\n"
                "set -euo pipefail\n"
            )
            project = root / "project.json"
            project.write_text(json.dumps({
                "system_manifest": "system.json",
                "definitions": {
                    "hydrogen_bond_discovery": {},
                    "ion_atmosphere": {},
                    "interaction_fingerprints": {
                        "source_modules": [
                            "hydrogen_bond_discovery", "ion_atmosphere",
                        ]
                    },
                },
            }), encoding="utf-8")
            (root / "run_preflight.slurm").write_text(worker, encoding="utf-8")
            (root / "run_stage_0_array.slurm").write_text(
                worker + f"PROJECT={project}\nCOMMANDS=(\n"
                "  'hydrogen-bond-discovery'\n  'ion-atmosphere'\n)\n",
                encoding="utf-8",
            )
            (root / "run_stage_1_array.slurm").write_text(
                worker + f"PROJECT={project}\nCOMMANDS=(\n"
                "  'interaction-fingerprints'\n)\n",
                encoding="utf-8",
            )
            (root / "run_finalize_reporting.slurm").write_text(worker, encoding="utf-8")
            (root / "analysis-config.json").write_text(json.dumps({
                "modules": {
                    "hydrogen_bond_discovery": {"depends_on": []},
                    "ion_atmosphere": {"depends_on": []},
                    "interaction_fingerprints": {"depends_on": []},
                }
            }), encoding="utf-8")
            plan = build_local_execution_plan(root, {
                "maximum_parallel_cpus": 2,
                "maximum_hours_per_cpu": 8,
                "maximum_memory_gib": 8,
            }, {"resource_table_enabled": False, "finding_picker_enabled": False})

        tasks = {
            task.get("module_id"): task
            for phase in plan["phases"] for task in phase["tasks"]
            if task.get("module_id")
        }
        fingerprints = tasks["interaction_fingerprints"]
        self.assertEqual(fingerprints["depends_on_task_ids"], [
            next(
                task for phase in plan["phases"] for task in phase["tasks"]
                if task["script"] == "run_preflight.slurm"
            )["task_id"]
        ])
        self.assertEqual(set(fingerprints["wait_for_task_ids"]), {
            tasks["hydrogen_bond_discovery"]["task_id"],
            tasks["ion_atmosphere"]["task_id"],
        })

    def test_reporting_components_do_not_wait_for_structural_qc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = (
                "#!/usr/bin/env bash\n#SBATCH --time=01:00:00\n"
                "#SBATCH --cpus-per-task=1\n#SBATCH --mem=2G\nset -euo pipefail\n"
            )
            (root / "run_preflight.slurm").write_text(worker, encoding="utf-8")
            (root / "run_stage_0_array.slurm").write_text(
                worker + "COMMANDS=(\n  'structural-qc'\n  'rmsf'\n  'common-pca'\n)\n",
                encoding="utf-8",
            )
            (root / "run_reporting_rmsf_permutation_inference.slurm").write_text(worker, encoding="utf-8")
            (root / "run_reporting_integrated_comparison.slurm").write_text(worker, encoding="utf-8")
            (root / "run_finalize_reporting.slurm").write_text(worker, encoding="utf-8")
            (root / "analysis-config.json").write_text(json.dumps({
                "modules": {
                    "structural_integrity_qc": {"depends_on": []},
                    "pooled_rmsf": {"depends_on": []},
                    "common_pca": {"depends_on": []},
                }
            }), encoding="utf-8")
            plan = build_local_execution_plan(root, {
                "maximum_parallel_cpus": 3,
                "maximum_hours_per_cpu": 8,
                "maximum_memory_gib": 12,
            }, {"resource_table_enabled": False, "finding_picker_enabled": False})

        tasks = {
            task.get("module_id") or task["script"]: task
            for phase in plan["phases"] for task in phase["tasks"]
        }
        qc = tasks["structural_integrity_qc"]
        rmsf = tasks["pooled_rmsf"]
        permutation = tasks["rmsf_permutation_inference"]
        integrated = tasks["integrated_comparison"]
        self.assertEqual(permutation["depends_on_task_ids"], [rmsf["task_id"]])
        self.assertNotIn(qc["task_id"], permutation["depends_on_task_ids"])
        self.assertNotIn(qc["task_id"], integrated["wait_for_task_ids"])
        self.assertIn(rmsf["task_id"], integrated["wait_for_task_ids"])
        self.assertIn(tasks["common_pca"]["task_id"], integrated["wait_for_task_ids"])

    def test_mixed_resource_array_is_split_into_scheduler_tiers(self):
        repository = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repository / "profiles/slurm/deac.json")
        profile["node_policy"]["memory_gib_per_node"] = 512.0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "slurm-profile.json").write_text("{}\n", encoding="utf-8")
            script = "run_view_global_common_heavy_stage_1.slurm"
            (root / script).write_text(
                "#!/usr/bin/env bash\n"
                "#SBATCH --time=24:00:00\n"
                "#SBATCH --cpus-per-task=1\n"
                "#SBATCH --mem=389G\n"
                "set -euo pipefail\n",
                encoding="utf-8",
            )
            (root / "submit-conformational-views.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "ROOT=/analysis\n"
                "VIEW_JOB=$(sbatch --parsable --dependency=\"afterok:$UPSTREAM_JOB\" "
                f"--array=0-6%7 \"$ROOT/{script}\")\n"
                "VIEW_JOB=\"${VIEW_JOB%%;*}\"\n"
                "printf 'Submitted %s.\\n' \"$VIEW_JOB\"\n"
                "NEXT_JOB=$(sbatch --parsable --dependency=\"afterok:$VIEW_JOB\" "
                "\"$ROOT/run_finalize_reporting.slurm\")\n",
                encoding="utf-8",
            )

            def task(index, memory, minutes):
                return {
                    "script": script,
                    "array_task_id": index,
                    "requested_wall_minutes": minutes,
                    "requested_memory_gib": memory,
                    "planner_task_ids": [f"view:task:{index}"],
                    "cpu_slots": 1,
                    "planned_wall_hours": minutes / 60.0,
                    "planned_peak_memory_gib": memory,
                    "resource_request_source": "unit_test",
                    "wall_request_limited_by_campaign_cap": False,
                    "memory_request_limited_by_campaign_cap": False,
                }

            scheduler = apply_slurm_profile(root, profile, {
                "maximum_parallel_cpus": 7,
                "maximum_parallel_memory_gib": 512,
                "phases": [{
                    "phase_id": "conformational",
                    "tasks": [
                        task(0, 3, 30), task(1, 3, 30), task(2, 3, 30),
                        task(3, 7, 60), task(4, 3, 30), task(5, 5, 45),
                        task(6, 389, 120),
                    ],
                }],
            })
            submit = (root / "submit-conformational-views.sh").read_text(
                encoding="utf-8"
            )
            syntax = subprocess.run(
                ["bash", "-n", str(root / "submit-conformational-views.sh")],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertIn("--array=0-2,4%4 --time=00:30:00 --mem=3G", submit)
        self.assertIn("--array=3%1 --time=01:00:00 --mem=7G", submit)
        self.assertIn("--array=5%1 --time=00:45:00 --mem=5G", submit)
        self.assertIn(
            "--array=6%1 --time=02:00:00 --mem=389G --partition=large",
            submit,
        )
        self.assertIn(
            'VIEW_JOB="${VIEW_JOB_TIER_0}:${VIEW_JOB_TIER_1}:'
            '${VIEW_JOB_TIER_2}:${VIEW_JOB_TIER_3}"',
            submit,
        )
        self.assertIn('--dependency="afterok:$VIEW_JOB"', submit)
        tiers = scheduler["submission_resource_tiers"][script]
        self.assertEqual([tier["array_task_ids"] for tier in tiers], [
            [0, 1, 2, 4], [3], [5], [6],
        ])
        self.assertEqual(sum(tier["submission_parallelism"] for tier in tiers), 7)
        self.assertEqual(tiers[-1]["selected_partition_role"], "large_memory")

    def test_tiered_array_preserves_a_single_cpu_throttle_with_dependencies(self):
        repository = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repository / "profiles/slurm/deac.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "slurm-profile.json").write_text("{}\n", encoding="utf-8")
            script = "run_stage_0_array.slurm"
            (root / script).write_text(
                "#!/usr/bin/env bash\n#SBATCH --time=01:00:00\n"
                "#SBATCH --cpus-per-task=1\n#SBATCH --mem=128G\n"
                "set -euo pipefail\n",
                encoding="utf-8",
            )
            (root / "submit.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nROOT=/analysis\n"
                "STAGE_JOB=$(sbatch --parsable --dependency=\"afterok:$PRE_JOB\" "
                f"--array=0-1%1 \"$ROOT/{script}\")\n"
                "STAGE_JOB=\"${STAGE_JOB%%;*}\"\n",
                encoding="utf-8",
            )

            def task(index, memory):
                return {
                    "script": script, "array_task_id": index,
                    "requested_wall_minutes": 60,
                    "requested_memory_gib": memory,
                    "planner_task_ids": [f"direct:{index}"], "cpu_slots": 1,
                    "planned_wall_hours": 1,
                    "planned_peak_memory_gib": memory,
                    "resource_request_source": "unit_test",
                    "wall_request_limited_by_campaign_cap": False,
                    "memory_request_limited_by_campaign_cap": False,
                }

            scheduler = apply_slurm_profile(root, profile, {
                "maximum_parallel_cpus": 1,
                "maximum_parallel_memory_gib": 128,
                "phases": [{"phase_id": "analysis", "tasks": [
                    task(0, 4), task(1, 128),
                ]}],
            })
            submit = (root / "submit.sh").read_text(encoding="utf-8")
            syntax = subprocess.run(
                ["bash", "-n", str(root / "submit.sh")],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertIn("--mem=128G --partition=large --array=1", submit)
        self.assertIn("--mem=4G --partition=small --array=0", submit)
        self.assertIn('--dependency="afterany:${JOB_T0000}"', submit)
        self.assertEqual(
            [lane["memory_gib"] for lane in scheduler["resource_lanes"]],
            [128.0],
        )

    def test_canonical_slurm_launcher_enforces_aggregate_memory_waves(self):
        repository = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repository / "profiles/slurm/deac.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "slurm-profile.json").write_text("{}\n", encoding="utf-8")
            script = "run_stage_0_array.slurm"
            (root / script).write_text(
                "#!/usr/bin/env bash\n#SBATCH --time=01:00:00\n"
                "#SBATCH --cpus-per-task=1\n#SBATCH --mem=100G\n"
                "set -euo pipefail\n",
                encoding="utf-8",
            )
            (root / "submit.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8"
            )

            def task(index, memory):
                return {
                    "script": script,
                    "array_task_id": index,
                    "requested_wall_minutes": 60,
                    "requested_memory_gib": memory,
                    "planner_task_ids": [f"direct:{index}"],
                    "cpu_slots": 1,
                    "planned_wall_hours": 1,
                    "planned_peak_memory_gib": memory,
                    "resource_request_source": "unit_test",
                    "wall_request_limited_by_campaign_cap": False,
                    "memory_request_limited_by_campaign_cap": False,
                }

            scheduler = apply_slurm_profile(root, profile, {
                "maximum_parallel_cpus": 3,
                "maximum_parallel_memory_gib": 185,
                "phases": [{"phase_id": "analysis", "tasks": [
                    task(0, 100), task(1, 90), task(2, 80),
                ]}],
            })
            submit = (root / "submit.sh").read_text(encoding="utf-8")
            syntax = subprocess.run(
                ["bash", "-n", str(root / "submit.sh")],
                check=False,
                capture_output=True,
                text=True,
            )
            preview_run = subprocess.run(
                [str(root / "submit.sh"), "--preview"],
                check=False,
                capture_output=True,
                text=True,
            )
            preview = json.loads(preview_run.stdout)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertEqual(preview_run.returncode, 0, preview_run.stderr)
        self.assertEqual(
            [lane["memory_gib"] for lane in scheduler["resource_lanes"]],
            [100.0, 80.0],
        )
        self.assertTrue(all(
            lane["memory_gib"] <= 185.0
            for lane in scheduler["resource_lanes"]
        ))
        self.assertIn('--dependency="afterany:${JOB_T0000}"', submit)
        self.assertFalse(preview["execution_started"])
        self.assertFalse(preview["jobs_submitted"])
        self.assertEqual(preview["task_count"], 3)
        self.assertEqual(preview["dependency_wave_count"], 1)
        self.assertEqual(preview["maximum_parallel_cpus_configured"], 3)
        self.assertEqual(preview["maximum_parallel_cpus_in_generated_waves"], 2)
        self.assertEqual(preview["maximum_parallel_memory_gib_configured"], 185)
        self.assertEqual(preview["maximum_parallel_memory_gib_in_generated_waves"], 180)
        self.assertEqual(preview["planned_node_count"], 1)
        self.assertEqual(
            preview["planned_node_reservations"][0]["reserved_memory_gib"],
            180.0,
        )
        self.assertEqual(preview["per_node_padding_validation"], "complete")
        self.assertEqual(
            preview["planner_estimated_dependency_critical_path_hours"], 2,
        )
        self.assertEqual(
            preview["scheduler_time_limit_reservation_critical_path_hours"], 2,
        )
        self.assertEqual(preview["warning_count"], 1)
        self.assertEqual(
            preview["warnings"][0]["code"],
            "REQUESTED_CPUS_EXCEED_GENERATED_PARALLELISM",
        )
        self.assertEqual(
            scheduler["submission_preview_file"],
            "slurm-submission-preview.json",
        )

    def test_task_dag_uses_afterany_resource_barriers_and_afterok_inputs(self):
        repository = Path(__file__).resolve().parents[1]
        profile = load_slurm_profile(repository / "profiles/slurm/deac.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "slurm-profile.json").write_text("{}\n", encoding="utf-8")
            for name in ("a.slurm", "b.slurm", "c.slurm"):
                (root / name).write_text(
                    "#!/usr/bin/env bash\n#SBATCH --time=01:00:00\n"
                    "#SBATCH --cpus-per-task=1\n#SBATCH --mem=100G\n"
                    "set -euo pipefail\n",
                    encoding="utf-8",
                )
            (root / "submit.sh").write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8"
            )

            def task(task_id, script, requirements, waits=()):
                return {
                    "task_id": task_id,
                    "depends_on_task_ids": requirements,
                    "wait_for_task_ids": list(waits),
                    "script": script,
                    "array_task_id": None,
                    "requested_wall_minutes": 60,
                    "requested_memory_gib": 100,
                    "planner_task_ids": [],
                    "cpu_slots": 1,
                    "planned_wall_hours": 1,
                    "planned_peak_memory_gib": 100,
                    "resource_request_source": "unit_test",
                    "wall_request_limited_by_campaign_cap": False,
                    "memory_request_limited_by_campaign_cap": False,
                }

            scheduler = apply_slurm_profile(root, profile, {
                "dependency_model": "task_dag_v1",
                "maximum_parallel_cpus": 2,
                "maximum_parallel_memory_gib": 100,
                "phases": [
                    {"phase_id": "level0", "tasks": [
                        task("a", "a.slurm", []), task("b", "b.slurm", []),
                    ]},
                    {"phase_id": "level1", "tasks": [
                        task("c", "c.slurm", ["a"], ["b"]),
                    ]},
                ],
            })
            submit = (root / "submit.sh").read_text(encoding="utf-8")
            syntax = subprocess.run(
                ["bash", "-n", str(root / "submit.sh")], check=False,
                capture_output=True, text=True,
            )

        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertIn('--dependency="afterany:${JOB_T0000}"', submit)
        self.assertIn(
            '--kill-on-invalid-dep=yes '
            '--dependency="afterany:${JOB_T0001},afterok:${JOB_T0000}"',
            submit,
        )
        self.assertEqual(scheduler["dependency_model"], "task_dag_v1")
        self.assertEqual(
            scheduler["submission_preview"]["scientific_dependency_edge_count"], 1,
        )
        self.assertEqual(
            scheduler["submission_preview"]["completion_wait_edge_count"], 1,
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

    def test_task_dag_local_runner_continues_unrelated_work_after_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            scripts = {
                "fails.slurm": "#!/usr/bin/env bash\nexit 9\n",
                "independent.slurm": (
                    "#!/usr/bin/env bash\nprintf complete > independent-marker\n"
                ),
                "dependent.slurm": (
                    "#!/usr/bin/env bash\nprintf bad > dependent-marker\n"
                ),
            }
            for name, content in scripts.items():
                (root / name).write_text(content, encoding="utf-8")

            def task(task_id, script, requirements, waits=()):
                return {
                    "task_id": task_id,
                    "depends_on_task_ids": requirements,
                    "wait_for_task_ids": list(waits),
                    "script": script,
                    "array_task_id": None,
                    "cpu_slots": 1,
                    "requested_memory_gib": 1,
                    "requested_wall_minutes": 1,
                }

            plan = {
                "local_execution_plan_schema": "salsbury-local-execution-plan-v4",
                "dependency_model": "task_dag_v1",
                "maximum_parallel_cpus": 1,
                "maximum_parallel_memory_gib": 1,
                "maximum_campaign_wall_hours": 1,
                "phases": [
                    {"phase_id": "level0", "tasks": [
                        task("failed", "fails.slurm", []),
                    ]},
                    {"phase_id": "level1", "tasks": [
                        task("independent", "independent.slurm", [], ["failed"]),
                        task("dependent", "dependent.slurm", ["failed"]),
                    ]},
                ],
            }
            (root / "local-execution-plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            report = run_local_workflow(root)
            independent_exists = (root / "independent-marker").exists()
            dependent_exists = (root / "dependent-marker").exists()

        second = {row["task_id"]: row for row in report["phase_reports"][1]["tasks"]}
        self.assertEqual(report["technical_status"], "failed")
        self.assertEqual(report["remaining_phases_not_run"], [])
        self.assertEqual(second["independent"]["status"], "complete")
        self.assertEqual(second["dependent"]["status"], "skipped_dependency")
        self.assertTrue(independent_exists)
        self.assertFalse(dependent_exists)

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
