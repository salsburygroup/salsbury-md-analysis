import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from salsbury_md_analysis.fixed_sampling import (
    FixedSamplingError, _digest, _validate_document, export_fixed_sampling_schedule,
    freeze_task_sampling, sampling_fields, verify_fixed_plan,
)
from salsbury_md_analysis.quickstart import prepare_standard_analysis, QuickstartError
from salsbury_md_analysis.comparative_quickstart import prepare_comparative_analysis
from salsbury_md_analysis.resource_planning import plan_campaign_resource_budget, ResourcePlanningError
from salsbury_md_analysis.scientific_sampling import profile_contract, scientific_sampling_profile
from tests.test_quickstart import _write_inputs, _write_dcd, _write_ion_inputs


def _rdf_task():
    return {
        "task_id":"rdf", "module_id":"radial_distribution_functions",
        "task_scope":"automatic_chemical_context", "workflow_id":"chemical_A",
        "source_frames_per_replica":[10_000]*4, "system_ids_per_replica":["A"]*4,
        "minimum_frames_per_replica":200, "maximum_frames_per_replica":10_000,
        "scientific_minimum_frames_per_replica":200,
        "scientific_sampling_requirements":profile_contract(scientific_sampling_profile("radial_distribution_functions")),
        "effective_cpu_cap":1, "dependency_stage":1,
        "cpu_seconds_per_physical_frame":1, "fixed_cpu_hours":0,
        "estimated_peak_memory_gib":2, "member_observation_multiplier":1,
    }


class FixedSamplingTests(unittest.TestCase):
    def test_ion_worker_selection_must_match_frozen_task(self):
        for scope in ("automatic_chemical_context", "base_automatic_chemistry"):
            name = "project-chemical_A.json" if scope == "automatic_chemical_context" else "project.json"
            task = {
                "module_id": "ion_atmosphere", "task_scope": scope,
                "workflow_id": "chemical_A", "integer_stride": 498,
                "source_frames_per_replica": [100_000] * 6,
                "selected_physical_frames_per_replica": [200] * 6,
            }
            definition = {"frame_stride": 1, "maximum_frames": 600_000}
            document = {
                "projects": {name: {"requested_modules": ["ion_atmosphere"],
                                     "sampling_fields": {"ion_atmosphere": definition}}},
                "tasks": {"ions": task}, "source_identity": {"test": True},
                "replicas": [{"test": True}],
            }
            with self.subTest(scope=scope):
                with self.assertRaisesRegex(FixedSamplingError, "worker mismatch: ions.*planned stride 498, worker stride 1"):
                    _validate_document(document)
                definition.update({"frame_selection": {"mode": "integer_stride_per_replica_v1", "stride": 498},
                                   "maximum_frames": 1200})
                _validate_document(document)
                definition["maximum_frames"] = 1199
                with self.assertRaisesRegex(FixedSamplingError, "worker mismatch"):
                    _validate_document(document)

    @patch("salsbury_md_analysis.comparative_quickstart._discover_dssp_executable", return_value=None)
    def test_five_system_fixed_schedule_with_strided_cache(self, _dssp):
        for protected in (True, False):
            with self.subTest(protected=protected):
                self._five_system_fixed_schedule(protected)

    @patch("salsbury_md_analysis.comparative_quickstart._discover_dssp_executable", return_value=None)
    def test_five_system_wall_only_stage_failure_reaches_native_validation(self, _dssp):
        self._five_system_fixed_schedule(False, stage_wall_failure=True)

    @patch("salsbury_md_analysis.comparative_quickstart._discover_dssp_executable", return_value=None)
    def test_historical_all_projection_weighted_kmeans_schedule_is_preserved(self, _dssp):
        self._five_system_fixed_schedule(False, legacy_imwk=True)

    def _five_system_fixed_schedule(self, protected, stage_wall_failure=False, legacy_imwk=False):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pdb, psf, trajectories = _write_ion_inputs(root)
            trajectories.append(root / "fourth.dcd")
            for path in trajectories:
                _write_dcd(path, 9, 2_000)
            request = root / "request.json"
            request.write_text(json.dumps({
                "request_schema":"salsbury-comparative-analysis-input-v1",
                "systems":[{"system_id": f"system_{i}", "pdb":str(pdb), "psf":str(psf),
                            "trajectories": [str(p) for p in trajectories], "frame_interval_ps":10}
                           for i in range(5)],
            }))
            config = root / "config.json"
            settings = {"config_schema":"salsbury-analysis-config-v1",
                        "clustering":{"feature_space":"common_pca"},
                        "planning":{"stride_mode":"uniform_cache_stride" if protected else "balanced_per_method",
                                    "module_selection":"protected_core_only" if protected else "all_enabled"},
                        "execution":{"maximum_parallel_cpus":44, "maximum_memory_gib":185,
                                     "overall_stride_candidates":[2], "submission_adapter":"slurm",
                                     "coordinate_cache_materialization":"planned_strided" if protected else "lossless",
                                     "slurm_profile":str(Path(__file__).resolve().parents[1] / "profiles/slurm/generic-template.json")}}
            config.write_text(json.dumps(settings))
            kwargs = dict(request_path=request, config_path=config, project_id="five_fixed", target_wall_hours=168)
            first, second = root / "first", root / "second"
            try:
                prepare_comparative_analysis(**kwargs, output_directory=first)
            except QuickstartError as exc:
                self.fail(str(getattr(exc, "plan", {}).get("optional_scientific_minimum_failures", str(exc))))
            before = json.loads((first / "campaign-resource-plan.json").read_text())
            if protected:
                self.assertEqual(before["global_stride_coupling"]["selected_coordinate_cache_integer_stride"], 2)
            schedule = first / "fixed-sampling-schedule.json"
            frozen = json.loads(schedule.read_text())
            if legacy_imwk:
                for row in frozen["tasks"].values():
                    if row["module_id"] != "clustering_imwkmeans":
                        continue
                    parent = frozen["tasks"][f"view:{row['workflow_id']}:common_pca"]
                    row["task_scope"] = "conformational_view"
                    for key in ("integer_stride", "source_frames_per_replica", "selected_physical_frames_per_replica"):
                        row[key] = copy.deepcopy(parent[key])
                    definition = frozen["projects"][f"project-{row['workflow_id']}.json"]["sampling_fields"]["clustering_imwkmeans"]
                    definition.pop("fit_stride", None)
                frozen["content_sha256"] = _digest({k:v for k,v in frozen.items() if k != "content_sha256"})
                schedule.write_text(json.dumps(frozen))
            settings["planning"]["stride_mode"] = "balanced_per_method"
            settings["sampling"] = {"fixed_schedule_file":str(schedule)}
            settings["execution"]["maximum_parallel_cpus"] = 24
            config.write_text(json.dumps(settings))
            def conservative_stage_estimate(*args, **kwargs):
                result = plan_campaign_resource_budget(*args, **kwargs)
                result["feasibility_status"] = "infeasible"
                result["infeasibility_reasons"] = ["minimum calibrated critical path exceeds the campaign science wall-time budget"]
                result["estimated_selected_wall_hours_lower_bound"] = 1000.0
                result["minimum_wall_hours_lower_bound"] = 1000.0
                return result
            if stage_wall_failure:
                with patch("salsbury_md_analysis.campaign_planning.plan_campaign_resource_budget", side_effect=conservative_stage_estimate):
                    prepare_comparative_analysis(**kwargs, output_directory=second)
            else:
                prepare_comparative_analysis(**kwargs, output_directory=second)
            after = json.loads((second / "campaign-resource-plan.json").read_text())
            self.assertTrue(verify_fixed_plan(after, frozen)["sampling_preserved"])
            if protected:
                self.assertEqual(after["global_stride_coupling"]["selected_coordinate_cache_integer_stride"], 2)
            for filename, project in frozen["projects"].items():
                current = json.loads((second/filename).read_text())
                self.assertEqual(sampling_fields(current["definitions"]), project["sampling_fields"])
            self.assertTrue((second / "submit.sh").exists())
            self.assertEqual(after["feasibility_status"], "feasible")
            native = after["native_schedule_validation"]
            self.assertEqual(native["status"], "complete")
            self.assertEqual(after["schedule_model"], "native_dependency_resource_tokens")
            self.assertAlmostEqual(after["stages"][0]["native_schedule"]["planned_wall_hours"],
                                   native["estimated_execution_hours"])
            for row in after["tasks"]:
                if row.get("future_cache_preflight"):
                    stride = after.get("global_stride_coupling", {}).get("selected_coordinate_cache_integer_stride", 1)
                    raw_counts = row["future_cache_preflight"]["raw_frames_per_replica"]
                    workload = row["resource_model"]["workload"]
                    self.assertEqual(workload["selected_frames_per_replica"], [n // stride for n in raw_counts])
                    # Raw counts must not already have been cache-strided.
                    self.assertTrue(all(n == 2000 for n in raw_counts))
            preview = json.loads((second / "slurm-submission-preview.json").read_text())
            # Science planning keeps its bounded envelope. Submission may
            # overlap independent jobs across additional physical nodes.
            self.assertLessEqual(preview["planner_estimated_dependency_critical_path_hours"], native["estimated_execution_hours"] + 1e-9)
            self.assertTrue(preview["submission_permitted"])
            self.assertEqual(preview["resource_token_edge_count"], 0)
            # A frozen historical block size must still be feasible; freezing
            # does not authorize changing scientific validity requirements.
            if not protected:
                frozen["projects"]["project.json"]["sampling_fields"]["convergence_uncertainty"]["block_size_frames"] = 100_000
                frozen["content_sha256"] = _digest({k:v for k,v in frozen.items() if k != "content_sha256"})
                invalid = root / "invalid-blocks.json"
                invalid.write_text(json.dumps(frozen))
                settings["sampling"]["fixed_schedule_file"] = str(invalid)
                config.write_text(json.dumps(settings))
                with self.assertRaisesRegex(QuickstartError, "convergence block contract is impossible"):
                    prepare_comparative_analysis(**kwargs, output_directory=root / "bad-blocks")
                self.assertFalse((root / "bad-blocks" / "submit.sh").exists())

    def test_pinned_resource_plan_never_upgrades_or_downsamples(self):
        task = _rdf_task()
        frozen = {"tasks":{"rdf":{
            key:copy.deepcopy(task[key]) for key in (
                "module_id", "task_scope", "workflow_id", "source_frames_per_replica", "member_observation_multiplier"
            )}}, "content_sha256":"test"}
        frozen["tasks"]["rdf"].update(integer_stride=50, selected_physical_frames_per_replica=[200]*4)
        pinned = freeze_task_sampling([task], frozen)
        before = copy.deepcopy(task)
        for hours, feasible in ((48, True), (.01, False)):
            result = plan_campaign_resource_budget(pinned, maximum_parallel_cpus=4, maximum_wall_hours=hours, maximum_memory_gib=8)
            self.assertEqual(result["feasibility_status"] == "feasible", feasible)
            self.assertTrue(verify_fixed_plan(result, frozen)["sampling_preserved"])
        self.assertEqual(task, before)
        frozen["tasks"]["rdf"].update(integer_stride=51, selected_physical_frames_per_replica=[196]*4)
        with self.assertRaisesRegex(ResourcePlanningError, "scientific sampling floor"):
            plan_campaign_resource_budget(freeze_task_sampling([task], frozen), maximum_parallel_cpus=4, maximum_wall_hours=48, maximum_memory_gib=8)

    def test_preparation_round_trip_preserves_sampling_and_rejects_low_budget(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pdb, psf, trajectories = _write_inputs(root)
            for path in trajectories:
                _write_dcd(path, 4, 1_000)
            config = root/"config.json"
            settings = {
                "config_schema":"salsbury-analysis-config-v1",
                "planning":{"module_selection":"protected_core_only"},
                "clustering":{"feature_space":"common_pca"},
                "execution":{"coordinate_cache_materialization":"lossless", "maximum_parallel_cpus":4, "maximum_memory_gib":185},
            }
            config.write_text(json.dumps(settings))
            kwargs = dict(pdb_path=pdb, psf_path=psf, trajectories=trajectories, project_id="fixed", frame_interval_ps=10, config_path=config)
            first = root/"first"
            prepare_standard_analysis(**kwargs, output_directory=first, target_wall_hours=48)
            schedule_path = root/"frozen.json"
            original_files = {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in first.iterdir() if path.is_file()}
            export_fixed_sampling_schedule(first, schedule_path)
            frozen = json.loads(schedule_path.read_text())
            settings["sampling"] = {"fixed_schedule_file":str(schedule_path)}
            config.write_text(json.dumps(settings))
            second = root/"second"
            prepare_standard_analysis(**kwargs, output_directory=second, target_wall_hours=168)
            plan = json.loads((second/"campaign-resource-plan.json").read_text())
            self.assertTrue(plan["fixed_sampling"]["sampling_preserved"])
            self.assertEqual(plan["planning_algorithm"], "fixed_sampling_resource_only_v1")
            for filename, project in frozen["projects"].items():
                current = json.loads((second/filename).read_text())
                self.assertEqual(sampling_fields(current["definitions"]), project["sampling_fields"])
            self.assertEqual(original_files, {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in first.iterdir() if path.is_file()})
            with self.assertRaisesRegex(QuickstartError, "No feasible fixed-sampling plan"):
                prepare_standard_analysis(**kwargs, output_directory=root/"too-short", target_wall_hours=.0001)
            self.assertFalse((root/"too-short"/"submit.sh").exists())
            pristine = copy.deepcopy(frozen)
            frozen["tasks"][next(iter(frozen["tasks"]))]["integer_stride"] = 999
            schedule_path.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(QuickstartError, "content hash mismatch"):
                prepare_standard_analysis(**kwargs, output_directory=root/"tampered", target_wall_hours=48)
            frozen = copy.deepcopy(pristine)
            frozen["source_identity"]["systems"][0]["replicas"][0]["segments"][0]["trajectory"] += ".other"
            frozen["content_sha256"] = _digest({k:v for k,v in frozen.items() if k != "content_sha256"})
            schedule_path.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(QuickstartError, "source manifest paths"):
                prepare_standard_analysis(**kwargs, output_directory=root/"changed-path", target_wall_hours=48)
            frozen = copy.deepcopy(pristine)
            frozen["replicas"][0]["replica_id"] = "changed"
            frozen["content_sha256"] = _digest({k:v for k,v in frozen.items() if k != "content_sha256"})
            schedule_path.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(QuickstartError, "replica identities"):
                prepare_standard_analysis(**kwargs, output_directory=root/"changed-source", target_wall_hours=48)


if __name__ == "__main__":
    unittest.main()
