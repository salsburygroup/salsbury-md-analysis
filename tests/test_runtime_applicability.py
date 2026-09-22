import copy
import unittest

from salsbury_md_analysis.runtime_applicability import apply_runtime_applicability


class RuntimeApplicabilityTests(unittest.TestCase):
    def task(self, module="replica_rmsd_rg"):
        return {
            "module_id": module, "source_frames_per_replica": [1000],
            "fixed_cpu_hours": 4.0, "cpu_seconds_per_physical_frame": 20.0,
            "runtime_baseline_terms": {"fixed_cpu_hours": 0.002,
                                       "cpu_seconds_per_physical_frame": 0.1},
            "runtime_workload_scaling": {"observed_atom_count": 423,
                "reference_atom_count": 85199, "resolved_multiplier": 0.01,
                "dimension": "maximum topology atom count proxy"},
            "estimated_peak_memory_gib": 8.0, "minimum_frames_per_replica": 100,
            "maximum_frames_per_replica": 1000,
            "censored_wall_lower_bound_points": [{"planning_wall_hours_lower_bound": 2.0}],
        }

    def calibration(self):
        return {
            "complete_measurement_count": 3,
            "conservative_cpu_seconds_per_frame": 20.0,
            "conservative_affine_cpu_seconds_per_frame": 10.0,
            "conservative_fixed_cpu_seconds": 9600.0,
            "maximum_completed_wall_seconds_per_frame": 5.0,
            "censored_wall_lower_bound_points": [{
                "planning_wall_seconds_lower_bound": 3600.0,
                "selected_source_physical_frames": 2000,
                "allocated_cpu_count": 6,
            }],
        }

    def test_consistent_work_scale_and_separate_setup(self):
        task = self.task()
        self.assertTrue(apply_runtime_applicability(task, self.calibration(), time_safety_factor=1.5))
        self.assertAlmostEqual(task["fixed_cpu_hours"], 0.04 + 100 / 3600)
        self.assertAlmostEqual(task["cpu_seconds_per_physical_frame"], 0.15)
        self.assertAlmostEqual(task["censored_wall_lower_bound_points"][0]["planning_wall_hours_lower_bound"], 0.015)
        self.assertEqual(task["runtime_applicability"]["baseline_setup_cpu_hours"], 0.002)

    def test_source_read_cost_grows_even_when_selected_frames_are_fixed(self):
        short = self.task(); long = self.task()
        short["selected_physical_frame_count"] = long["selected_physical_frame_count"] = 100
        long["source_frames_per_replica"] = [20000]
        for task in (short, long):
            apply_runtime_applicability(task, self.calibration(), time_safety_factor=1.5)
        self.assertAlmostEqual(long["fixed_cpu_hours"] - short["fixed_cpu_hours"],
                               19000 * 0.1 / 3600)
        self.assertEqual(short["cpu_seconds_per_physical_frame"],
                         long["cpu_seconds_per_physical_frame"])

    def test_dssp_retains_external_process_cost_per_frame(self):
        task = self.task("secondary_structure")
        apply_runtime_applicability(task, self.calibration(), time_safety_factor=1.5)
        self.assertEqual(task["cpu_seconds_per_physical_frame"], 7.5)

    def test_idempotent_without_double_scaling(self):
        task = self.task()
        apply_runtime_applicability(task, self.calibration(), time_safety_factor=1.5)
        before = copy.deepcopy(task)
        apply_runtime_applicability(task, self.calibration(), time_safety_factor=1.5)
        self.assertEqual(task, before)

    def test_sampling_and_memory_are_unchanged(self):
        task = self.task(); before = copy.deepcopy(task)
        apply_runtime_applicability(task, self.calibration(), time_safety_factor=1.5)
        for key in ("estimated_peak_memory_gib", "minimum_frames_per_replica",
                    "maximum_frames_per_replica", "source_frames_per_replica"):
            self.assertEqual(task[key], before[key])

    def test_large_unknown_multireplica_and_extrapolation_keep_legacy(self):
        for change in ("large", "tiny", "multiple", "long", "short", "missing", "unsupported", "unknown_reference", "nan"):
            with self.subTest(change=change):
                task = self.task()
                if change == "large": task["runtime_workload_scaling"]["observed_atom_count"] = 104300
                if change == "tiny": task["runtime_workload_scaling"]["observed_atom_count"] = 200
                if change == "multiple": task["source_frames_per_replica"] = [1000, 1000]
                if change == "long": task["source_frames_per_replica"] = [20001]
                if change == "short": task["source_frames_per_replica"] = [999]
                if change == "missing": task.pop("runtime_baseline_terms")
                if change == "unsupported": task["module_id"] = "hydrogen_bond_discovery"
                if change == "unknown_reference": task["runtime_workload_scaling"]["reference_atom_count"] = 100
                if change == "nan": task["runtime_workload_scaling"]["resolved_multiplier"] = float("inf")
                before = copy.deepcopy(task)
                self.assertFalse(apply_runtime_applicability(task, self.calibration(), time_safety_factor=1.5))
                self.assertEqual(task, before)

    def test_censored_only_cannot_lower_cost(self):
        task=self.task(); before=copy.deepcopy(task); c=self.calibration()
        c["complete_measurement_count"] = 0
        self.assertFalse(apply_runtime_applicability(task,c,time_safety_factor=1.5))
        self.assertEqual(task,before)

    def test_missing_dssp_wall_measurement_keeps_legacy(self):
        task=self.task("secondary_structure"); before=copy.deepcopy(task)
        c=self.calibration(); c.pop("maximum_completed_wall_seconds_per_frame")
        self.assertFalse(apply_runtime_applicability(task,c,time_safety_factor=1.5))
        self.assertEqual(task,before)

    def test_invalid_safety_rejected(self):
        for safety in (0,-1,float("inf"),float("nan")):
            with self.assertRaises(ValueError):
                apply_runtime_applicability(self.task(),self.calibration(),time_safety_factor=safety)

    def test_zero_catalog_intercept_does_not_erase_setup(self):
        task=self.task(); c=self.calibration(); c["conservative_fixed_cpu_seconds"]=0
        apply_runtime_applicability(task,c,time_safety_factor=1.5)
        self.assertEqual(task["fixed_cpu_hours"],0.002 + 100 / 3600)


if __name__ == "__main__":
    unittest.main()
