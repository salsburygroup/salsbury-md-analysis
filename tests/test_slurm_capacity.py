import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.resource_planning import plan_campaign_resource_budget
from salsbury_md_analysis.slurm_capacity import (
    SlurmCapacityError,
    advise_slurm_capacity,
    render_capacity_markdown,
)


class FakeSlurm:
    def __init__(self):
        self.commands = []

    def __call__(self, command):
        command = list(command)
        self.commands.append(command)
        if command[0].endswith("scontrol") and command[1:3] == ["show", "partition"]:
            partition = command[3]
            return subprocess.CompletedProcess(
                command, 0,
                stdout=(
                    f"PartitionName={partition} MaxNodes=2 MaxTime=1-00:00:00 "
                    "TotalCPUs=8 State=UP\n"
                ),
                stderr="",
            )
        if command[0].endswith("scontrol") and command[1:4] == ["show", "nodes", "-o"]:
            return subprocess.CompletedProcess(
                command, 0,
                stdout=(
                    "NodeName=n1 CPUTot=4 CPUAlloc=1 RealMemory=65536 "
                    "AllocMem=8192 FreeMem=50000 State=MIX Partitions=small\n"
                    "NodeName=n2 CPUTot=4 CPUAlloc=0 RealMemory=65536 "
                    "AllocMem=0 FreeMem=62000 State=IDLE Partitions=small\n"
                ),
                stderr="",
            )
        if command[0].endswith("sacctmgr") and "assoc" in command:
            return subprocess.CompletedProcess(
                command, 0,
                stdout="test|acct|fred||normal|||||100|1000|\n",
                stderr="",
            )
        if command[0].endswith("sacctmgr") and "qos" in command:
            return subprocess.CompletedProcess(
                command, 0,
                stdout="normal||||100|1000|\n",
                stderr="",
            )
        if command[0].endswith("squeue") and "--start" in command:
            return subprocess.CompletedProcess(
                command, 0,
                stdout="123|PENDING|2026-08-27T10:00:00|Resources\n",
                stderr="",
            )
        if command[0].endswith("squeue"):
            return subprocess.CompletedProcess(
                command, 0,
                stdout=(
                    "100|other|other|small|R|4|16G|02:00:00|01:00:00|"
                    "2026-08-26T12:00:00|n1|100\n"
                    "101|fred|acct|small|PD|1|8G|01:00:00|01:00:00|"
                    "2026-08-26T13:00:00|Priority|90\n"
                ),
                stderr="",
            )
        raise AssertionError(command)


class SlurmCapacityTests(unittest.TestCase):
    def _prepared_campaign(self, root: Path) -> Path:
        prepared = root / "prepared"
        prepared.mkdir()
        tasks = []
        for index in range(3):
            tasks.append({
                "task_id": f"task-{index}",
                "module_id": f"module-{index}",
                "dependency_stage": 0,
                "effective_cpu_cap": 1,
                "execution_bundle_id": f"bundle-{index}",
                "source_frames_per_replica": [1000],
                "minimum_frames_per_replica": 10,
                "maximum_frames_per_replica": 1000,
                "cpu_seconds_per_physical_frame": 5.0,
                "fixed_cpu_hours": 0.0,
                "estimated_peak_memory_gib": 2.0,
                "measured_memory_cost_model": {
                    "calibration_observations": 10,
                    "calibration_memory_gib": 2.0 + index,
                    "memory_exponent": 0.5,
                    "minimum_observation_scale": 0.1,
                },
                "priority_weight": 1.0,
            })
        plan = plan_campaign_resource_budget(
            tasks,
            maximum_parallel_cpus=2,
            maximum_wall_hours=1.0,
            maximum_memory_gib=64.0,
            planning_utilization=0.85,
            pilot_budget_fraction=0.05,
            finalization_headroom_fraction=0.05,
        )
        (prepared / "campaign-resource-plan.json").write_text(
            json.dumps(plan), encoding="utf-8"
        )
        (prepared / "scheduler-resource-requests.json").write_text(
            json.dumps({
                "scheduler_resource_requests_schema": (
                    "salsbury-scheduler-resource-requests-v1"
                ),
                "tasks": [
                    {"planner_task_ids": [f"task-{index}"]}
                    for index in range(3)
                ],
            }),
            encoding="utf-8",
        )
        (prepared / "slurm-profile.json").write_text(json.dumps({
            "slurm_profile_schema": "salsbury-slurm-profile-v1",
            "profile_id": "test",
            "cluster_name": "test",
            "submit_command": "/opt/slurm/bin/sbatch",
            "status_command": "/opt/slurm/bin/squeue",
            "cancel_command": "/opt/slurm/bin/scancel",
            "account": "acct",
            "qos": "normal",
            "partitions": {"default": "small", "analysis": "small"},
            "partition_maximum_wall_minutes": {"small": 1440},
            "resource_policy": {
                "minimum_wall_minutes": 30,
                "walltime_safety_factor": 1.5,
                "walltime_overhead_minutes": 15,
                "minimum_memory_gib": 2,
                "memory_safety_factor": 1.5,
                "memory_overhead_gib": 1,
                "large_memory_threshold_gib": 32,
            },
        }), encoding="utf-8")
        return prepared

    def test_live_advice_uses_workflow_ceiling_and_never_submits(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_campaign(Path(temporary))
            runner = FakeSlurm()
            report = advise_slurm_capacity(
                prepared,
                wall_hours=8.0,
                live=True,
                slurm_user="fred",
                job_ids=["123"],
                runner=runner,
            )
        self.assertEqual(
            report["cpu_capacity"]["workflow_useful_parallel_cpu_ceiling"], 3
        )
        self.assertEqual(
            report["cpu_capacity"]["live_scheduler_simultaneous_cpu_ceiling"], 8
        )
        self.assertEqual(
            report["cpu_capacity"]["recommended_maximum_parallel_cpus"], 3
        )
        self.assertEqual(report["replanned_campaign"]["raw_capacity_cpu_hours"], 24.0)
        self.assertEqual(
            report["queue_forecast"]["forecast_quality"],
            "scheduler_projected_for_submitted_job_ids",
        )
        self.assertEqual(
            report["queue_forecast"]["live_scheduler"]["queue"]
            ["submitted_job_start_estimates"][0]["expected_start"],
            "2026-08-27T10:00:00",
        )
        flattened = [value for command in runner.commands for value in command]
        self.assertNotIn("scancel", " ".join(flattened))
        self.assertFalse(any(command[0].endswith("sbatch") for command in runner.commands))
        self.assertFalse(report["jobs_submitted"])

    def test_longer_cpu_hour_envelope_recomputes_sampling_and_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_campaign(Path(temporary))
            short = advise_slurm_capacity(prepared, wall_hours=1.0, live=False)
            long = advise_slurm_capacity(prepared, wall_hours=8.0, live=False)
        short_tasks = short["replanned_campaign"]["tasks"]
        long_tasks = long["replanned_campaign"]["tasks"]
        self.assertGreaterEqual(
            sum(row["selected_physical_frame_count"] for row in long_tasks),
            sum(row["selected_physical_frame_count"] for row in short_tasks),
        )
        self.assertGreaterEqual(
            long["memory_capacity"]["planned_working_set_gib"],
            short["memory_capacity"]["planned_working_set_gib"],
        )
        self.assertEqual(
            long["cpu_capacity"]["recommended_maximum_parallel_cpus"], 3
        )

    def test_cpu_ceiling_is_optional_policy_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_campaign(Path(temporary))
            report = advise_slurm_capacity(
                prepared, wall_hours=2.0, cpu_ceiling=1, live=False
            )
        self.assertEqual(
            report["cpu_capacity"]["recommended_maximum_parallel_cpus"], 1
        )

    def test_scheduler_manifest_excludes_nonexecuting_planner_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_campaign(Path(temporary))
            (prepared / "scheduler-resource-requests.json").write_text(
                json.dumps({
                    "scheduler_resource_requests_schema": (
                        "salsbury-scheduler-resource-requests-v1"
                    ),
                    "tasks": [
                        {"planner_task_ids": ["task-0"]},
                        {"planner_task_ids": ["task-1"]},
                    ],
                }),
                encoding="utf-8",
            )
            report = advise_slurm_capacity(
                prepared, wall_hours=2.0, live=False
            )
        selection = report["replanned_campaign"]["task_selection"]
        self.assertEqual(selection["executable_task_count"], 2)
        self.assertEqual(
            selection["excluded_nonexecuting_planner_task_ids"], ["task-2"]
        )
        self.assertEqual(
            report["cpu_capacity"]["workflow_useful_parallel_cpu_ceiling"], 2
        )
        self.assertNotEqual(report["memory_capacity"]["largest_task_id"], "task-2")

    def test_invalid_job_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_campaign(Path(temporary))
            with self.assertRaisesRegex(SlurmCapacityError, "invalid Slurm job id"):
                advise_slurm_capacity(
                    prepared,
                    wall_hours=2.0,
                    live=True,
                    slurm_user="fred",
                    job_ids=["123;scancel"],
                    runner=FakeSlurm(),
                )

    def test_markdown_is_compact_and_explicitly_non_submitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = self._prepared_campaign(Path(temporary))
            report = advise_slurm_capacity(prepared, wall_hours=2.0, live=False)
        text = render_capacity_markdown(report)
        self.assertIn("Useful workflow maximum: 3 CPUs", text)
        self.assertIn("Estimated selected CPU-hours:", text)
        self.assertIn("Scheduler memory needed with safety margin:", text)
        self.assertIn("No job was submitted", text)


if __name__ == "__main__":
    unittest.main()
