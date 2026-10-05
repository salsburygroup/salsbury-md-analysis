import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.cache_routing import (
    CacheRoutingError, _cache_relative_sampling, cache_routing_plan,
)
from salsbury_md_analysis.frame_sampling import integer_stride_indices
from salsbury_md_analysis.upstream_cache import planned_cache_modules


class CacheRuntimeContractTests(unittest.TestCase):
    def test_exact_two_level_source_identities_and_floor_counts(self):
        for count in (100, 101, 1000, 100003):
            for cache_stride in (1, 2, 5):
                for multiplier in (1, 7, 13, 40):
                    raw_stride = cache_stride * multiplier
                    with self.subTest(count=count, cache=cache_stride, raw=raw_stride):
                        definition = {
                            "frame_stride": 1,
                            "frame_selection": {"mode": "integer_stride_per_replica_v1",
                                                "stride": raw_stride},
                            "projection_frame_stride": raw_stride,
                            "component_count": 10,
                        }
                        _cache_relative_sampling(definition, "individual_pca", cache_stride, [])
                        self.assertEqual(definition["frame_selection"]["stride"], multiplier)
                        self.assertEqual(definition["projection_frame_stride"], multiplier)
                        self.assertEqual(definition["component_count"], 10)
                        cached_source_indices = sorted(integer_stride_indices(count, cache_stride))
                        selected = integer_stride_indices(len(cached_source_indices), multiplier)
                        self.assertEqual({cached_source_indices[i] for i in selected},
                                         integer_stride_indices(count, raw_stride))

    def test_nonrepresentable_or_phase_changed_sampling_is_rejected(self):
        for stride in (1, 3, 0, True):
            with self.subTest(stride=stride), self.assertRaises(CacheRoutingError):
                _cache_relative_sampling({"frame_stride": stride}, "rmsd", 2, [])
        with self.assertRaisesRegex(CacheRoutingError, "different cache phase"):
            _cache_relative_sampling({"frame_stride": 4}, "rmsd", 2,
                                     [{"segments": [{"first_retained_source_frame_index": 1}]}])
        with self.assertRaisesRegex(CacheRoutingError, "not exactly representable"):
            _cache_relative_sampling({"frame_stride": 1, "frame_selection": {
                "mode": "uniform_per_replica_budget_v1", "maximum_frames_per_replica": 10}},
                "dccm", 2, [])

    def test_cached_derived_consumers_follow_producer_contract(self):
        project = {"requested_modules": ["replica_rmsd_rg", "convergence_uncertainty",
                                          "dccm", "correlation_networks"],
                   "definitions": {"replica_rmsd_rg": {"alignment_selection": "solute"},
                                   "dccm": {"analysis_selection": "solute"}},
                   "selections": {"solute": {"preset": "solute_heavy"}}}
        self.assertEqual(set(cache_routing_plan(project)["cache_project_modules"]),
                         set(project["requested_modules"]))
        project["selections"]["solute"] = {"preset": "heavy"}
        self.assertEqual(cache_routing_plan(project)["cache_project_modules"], [])

    def test_explicit_cache_contract_survives_temporary_replica_project(self):
        with patch.dict(os.environ, {
            "SALSBURY_MD_ANALYSIS_REQUIRED_CACHE_MODULES": '["common_pca"]',
            "SALSBURY_MD_ANALYSIS_PREPARED_ROOT": "/missing/prepared",
            "SALSBURY_MD_ANALYSIS_PREPARED_COMMAND": "pca-fes-basins",
        }, clear=True):
            self.assertEqual(planned_cache_modules(Path("/temporary/shard.json")), ["common_pca"])

    def test_legacy_required_cache_uses_scope_not_unrelated_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [
                {"task_id": "consumer", "command": "pca-fes-basins",
                 "project_filename": "project.json", "scope_id": "view:a",
                 "wait_for_task_ids": ["a", "b"]},
                {"task_id": "a", "module_id": "common_pca", "scope_id": "view:a"},
                {"task_id": "b", "module_id": "dccm", "scope_id": "view:b"},
            ]
            (root / "local-execution-plan.json").write_text(json.dumps({"phases": [{"tasks": tasks}]}))
            with patch.dict(os.environ, {"SALSBURY_MD_ANALYSIS_PREPARED_ROOT": str(root),
                                         "SALSBURY_MD_ANALYSIS_PREPARED_COMMAND": "pca-fes-basins"}, clear=True):
                self.assertEqual(planned_cache_modules(root / "project.json"), ["common_pca"])


if __name__ == "__main__":
    unittest.main()
