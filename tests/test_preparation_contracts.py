import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.campaign_planning import (
    _apply_direct_project_sampling,
    _convergence_selected_series,
)
from salsbury_md_analysis.comparative_quickstart import prepare_comparative_analysis
from salsbury_md_analysis.convergence import _series_diagnostic
from salsbury_md_analysis.convergence_contracts import configure_convergence_blocks
from salsbury_md_analysis.manifests import ManifestValidationError
from tests.test_quickstart import _write_dcd, _write_inputs


def _definition():
    return {
        "source_module": "replica_rmsd_rg",
        "metrics": ["rmsd_angstrom", "radius_of_gyration_angstrom"],
        "block_size_frames": 1000,
        "include_partial_final_block": True,
        "minimum_blocks": 4,
        "effective_sample_size_reference": 20.0,
        "split_mean_difference_reference_in_sd": 1.0,
    }


def _c031_dimensions():
    """Header-count fixture matching C031's 20 replicas and 44 segments."""
    rows = []
    for system_id in ("TBA_K", "TBAF6_K", "TBAE3F3_K", "TBAE6_K", "TBAE6_ddC16"):
        counts = [1000] * 4 + [2000] * 3 if system_id == "TBAE6_ddC16" else [10000]
        for replica in range(1, 5):
            rows.append({
                "system_id": system_id, "replica_id": f"rep{replica}",
                "source_frame_count": sum(counts),
                "segments": [
                    {"segment_id": f"chunk_{index:03d}", "source_frame_count": count}
                    for index, count in enumerate(counts, 1)
                ],
            })
    return {"replicas": rows}


def _five_system_request(root):
    pdb, psf, _ = _write_inputs(root)
    systems = {}
    for replica in _c031_dimensions()["replicas"]:
        system_id = replica["system_id"]
        system = systems.setdefault(system_id, {
            "system_id": system_id, "pdb": str(pdb), "psf": str(psf), "replicas": [],
        })
        segments = []
        offset = 0
        for segment in replica["segments"]:
            path = root / f"{system_id}-{replica['replica_id']}-{segment['segment_id']}.dcd"
            _write_dcd(path, 4, segment["source_frame_count"])
            segments.append({
                "segment_id": segment["segment_id"], "trajectory": str(path),
                "frame_interval_ps": 100.0, "first_frame_time_ps": (offset + 1) * 100.0,
                "continuous_with_previous": bool(offset),
                "dcd_header_step_policy": "reset_per_segment",
            })
            offset += segment["source_frame_count"]
        system["replicas"].append({"replica_id": replica["replica_id"], "segments": segments})
    request = root / "request.json"
    request.write_text(json.dumps({
        "request_schema": "salsbury-comparative-analysis-input-v2", "systems": list(systems.values()),
    }))
    return request


class PreparationContractTests(unittest.TestCase):
    def test_c031_generated_blocks_use_shortest_selected_segment(self):
        project = {"definitions": {
            "replica_rmsd_rg": {"frame_stride": 1},
            "convergence_uncertainty": _definition(),
        }}
        plan = {"dimensions": _c031_dimensions(), "method_plans": [{
            "module_id": "replica_rmsd_rg", "frame_stride": 1,
            "campaign_resource_allocation": {"selected_physical_frames_per_replica": [10000] * 20},
        }]}
        source_before = copy.deepcopy(plan["dimensions"])
        _apply_direct_project_sampling(project, plan)
        settings = project["definitions"]["convergence_uncertainty"]
        self.assertEqual(settings["block_size_frames"], 100)
        self.assertEqual(settings["minimum_blocks"], 4)
        self.assertEqual(plan["dimensions"], source_before)
        self.assertEqual(plan["convergence_preparation_contract"]["series_count"], 44)
        self.assertEqual(plan["convergence_preparation_contract"]["minimum_selected_observations_per_segment"], 1000)
        diagnostic = _series_diagnostic([float(i % 11) for i in range(1000)], settings)
        self.assertTrue(diagnostic["block_contract_satisfied"])
        self.assertEqual(len(diagnostic["block_means"]), 10)

    def test_final_stride_and_partial_block_rule_are_honored(self):
        for partial in (True, False):
            for count in (4, 10, 1000):
                definition = _definition()
                definition["include_partial_final_block"] = partial
                receipt = configure_convergence_blocks(definition, [{
                    "system_id": "A", "replica_id": "r1", "segment_id": "s1",
                    "selected_observation_count": count,
                }])
                self.assertLessEqual(receipt["minimum_required_observations"], count)
                self.assertEqual(definition["minimum_blocks"], 4)

    def test_explicit_valid_block_size_is_preserved_and_invalid_is_rejected(self):
        for size, accepted in ((250, True), (333, True), (334, False), (1000, False)):
            settings = _definition()
            settings["block_size_frames"] = size
            series = [{"system_id": "TBAE6_ddC16", "replica_id": "rep1", "segment_id": "chunk_001", "selected_observation_count": 1000}]
            if accepted:
                configure_convergence_blocks(settings, series, explicit_block_size=True)
                self.assertEqual(settings["block_size_frames"], size)
            else:
                with self.assertRaisesRegex(ValueError, "during preparation for TBAE6_ddC16/rep1/chunk_001: 1000 selected"):
                    configure_convergence_blocks(settings, series, explicit_block_size=True)
                self.assertEqual(settings["block_size_frames"], size)

    def test_infeasible_generated_contract_does_not_weaken_minimum_blocks(self):
        settings = _definition()
        with self.assertRaisesRegex(ValueError, "sampling and minimum_blocks have not been changed"):
            configure_convergence_blocks(settings, [{"selected_observation_count": 3}])
        self.assertEqual(settings["minimum_blocks"], 4)

    def test_two_level_selection_uses_cache_offsets_and_method_stride(self):
        project = {"definitions": {
            "replica_rmsd_rg": {"frame_stride": 1}, "convergence_uncertainty": _definition(),
        }}
        plan = {"dimensions": _c031_dimensions(), "method_plans": [{
            "module_id": "replica_rmsd_rg", "frame_stride": 2,
            "campaign_resource_allocation": {"selected_physical_frames_per_replica": [500] * 20},
        }], "campaign_resource_plan": {"global_stride_coupling": {"selected_coordinate_cache_integer_stride": 10}}}
        _apply_direct_project_sampling(project, plan)
        self.assertEqual(project["definitions"]["convergence_uncertainty"]["block_size_frames"], 5)
        self.assertEqual(plan["convergence_preparation_contract"]["minimum_selected_observations_per_segment"], 50)

    def test_cache_stride_phase_is_preserved_across_uneven_segments(self):
        project = {"definitions": {"replica_rmsd_rg": {"frame_stride": 2}}}
        plan = {"dimensions": {"replicas": [{
            "system_id": "A", "replica_id": "r1", "segments": [
                {"segment_id": "s1", "source_frame_count": 31},
                {"segment_id": "s2", "source_frame_count": 69},
            ],
        }]}, "campaign_resource_plan": {"global_stride_coupling": {
            "selected_coordinate_cache_integer_stride": 3,
        }}}
        rows = _convergence_selected_series(project, plan, [16])
        # Cache frames 0,3,...,96 give 11 and 22 frames by source segment;
        # downstream stride 2 keeps five and 11, not a separate cache floor.
        self.assertEqual([row["selected_observation_count"] for row in rows], [5, 11])

    @patch("salsbury_md_analysis.comparative_quickstart._discover_dssp_executable", return_value=None)
    @patch("salsbury_md_analysis.comparative_quickstart._discover_dssr_executable", return_value=None)
    def test_five_system_prepare_and_invalid_selection_fail_before_launcher(self, *_tools):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _five_system_request(root)
            input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
            config = root / "config.json"
            settings = {"config_schema": "salsbury-analysis-config-v1", "execution": {"maximum_parallel_cpus": 44, "maximum_hours_per_cpu": 48, "maximum_memory_gib": 185, "coordinate_cache_materialization": "lossless"}}
            config.write_text(json.dumps(settings))
            good = root / "good"
            report = prepare_comparative_analysis(request_path=request, output_directory=good, project_id="C031-test", config_path=config)
            self.assertEqual(report["technical_status"], "complete")
            project = json.loads((good / "project.json").read_text())
            self.assertEqual(project["definitions"]["convergence_uncertainty"]["block_size_frames"], 100)
            self.assertEqual(project["definitions"]["convergence_uncertainty"]["minimum_blocks"], 4)
            self.assertTrue((good / "run-local.sh").exists())
            self.assertFalse(list((good / "results").iterdir()))

            # A saved-plan edit must be caught before dispatch, even though
            # the future coordinate cache has not yet been materialized.
            from salsbury_md_analysis.execution_adapters import validate_worker_projects, ExecutionAdapterError
            native = json.loads((good / "local-execution-plan.json").read_text())
            view_path = good / "project-global_common_heavy.json"
            view = json.loads(view_path.read_text())
            original_view = view_path.read_bytes()
            for module, field in (("representative_frames", "maximum_candidates"), ("grouped_ml", "maximum_observations")):
                edited = copy.deepcopy(view)
                if module not in edited["requested_modules"]:
                    edited["requested_modules"].append(module)
                edited["definitions"][module][field] = 1
                view_path.write_text(json.dumps(edited))
                with self.assertRaisesRegex(ExecutionAdapterError, "selected pooled observations.*" + field):
                    validate_worker_projects(good, native)
            view_path.write_bytes(original_view)
            validate_worker_projects(good, native)

            settings["modules"] = {"solvent_accessible_surface_area": {"options": {"surface_selection": "E7_heavy"}}}
            config.write_text(json.dumps(settings))
            bad = root / "invalid-sasa"
            with self.assertRaisesRegex(ManifestValidationError, "surface_selection='E7_heavy'.*reference system 'TBA_K'"):
                prepare_comparative_analysis(request_path=request, output_directory=bad, project_id="C031-test-invalid", config_path=config)
            self.assertFalse((bad / "run-local.sh").exists())
            self.assertFalse((bad / "campaign-resource-plan.json").exists())

            settings["modules"] = {"convergence_uncertainty": {"options": {"block_size_frames": 1000}}}
            config.write_text(json.dumps(settings))
            bad_blocks = root / "invalid-blocks"
            with self.assertRaisesRegex(ValueError, "TBAE6_ddC16/rep1/chunk_001: 1000 selected observations cannot yield 4 blocks of 1000"):
                prepare_comparative_analysis(request_path=request, output_directory=bad_blocks, project_id="C031-test-blocks", config_path=config)
            self.assertFalse((bad_blocks / "run-local.sh").exists())
            for path, expected_hash in input_hashes.items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected_hash)
