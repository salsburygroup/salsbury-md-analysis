"""Regressions discovered while preparing bounded live Slurm validation."""
import tempfile
import unittest
from pathlib import Path
from salsbury_md_analysis.execution_adapters import (
    _fit_walltime_requests_to_campaign, _slurm_resource_epochs,
    _render_resource_bounded_submit, load_slurm_profile,
)


class LiveSlurmRegressions(unittest.TestCase):
    @staticmethod
    def _task(name, hours, deps=()):
        return {"task_id": name, "script": name + ".slurm", "cpu_slots": 1,
                "planned_wall_hours": hours, "requested_memory_gib": 1,
                "minimum_requested_wall_minutes": hours * 60,
                "preferred_requested_wall_minutes": hours * 60,
                "requested_wall_minutes": hours * 60,
                "depends_on_task_ids": list(deps)}

    def test_timeout_fit_uses_real_dag_not_phase_barriers(self):
        plan = {"maximum_parallel_cpus": 2, "maximum_parallel_memory_gib": 8,
                "maximum_campaign_wall_hours": 3, "node_policy": {},
                "phases": [
                    {"phase_id": "first", "tasks": [self._task("slow", 2), self._task("fast", .1)]},
                    {"phase_id": "second", "tasks": [self._task("consumer", 2, ("fast",))]},
                ]}
        allocation = _fit_walltime_requests_to_campaign(plan)
        self.assertTrue(allocation["submission_time_feasible"])
        self.assertAlmostEqual(allocation["selected_scheduler_reservation_critical_path_hours"], 2.1)

    def test_wrapper_submission_keeps_profile_account_and_qos(self):
        profile = load_slurm_profile(Path(__file__).resolve().parents[1] / "profiles/slurm/deac.json")
        plan = {"maximum_parallel_cpus": 1, "maximum_parallel_memory_gib": 8,
                "phases": [{"phase_id": "p", "tasks": [self._task("task", .1)]}]}
        epochs = _slurm_resource_epochs(plan, profile["partitions"],
            profile["partition_maximum_wall_minutes"], profile["partition_maximum_nodes"],
            profile["resource_policy"], {})
        with tempfile.TemporaryDirectory() as directory:
            script = _render_resource_bounded_submit(Path(directory), profile,
                Path(directory) / "profile.json", epochs, True, True)
        line = next(line for line in script.splitlines() if line.startswith("JOB_T0000="))
        self.assertIn("--account=salsburygrp", line)
        self.assertIn("--qos=normal", line)
