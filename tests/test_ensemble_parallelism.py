import unittest

from salsbury_md_analysis.ensemble_parallelism import (
    EnsembleParallelismError,
    annotate_task_parallelism,
    get_ensemble_parallelism_contract,
)
from salsbury_md_analysis.registry import list_modules


class EnsembleParallelismTests(unittest.TestCase):
    def test_every_registered_module_has_an_explicit_contract(self):
        for module in list_modules():
            contract = get_ensemble_parallelism_contract(module.module_id)
            self.assertEqual(contract.module_id, module.module_id)

    def test_global_estimators_cannot_be_finalized_per_replica(self):
        expected_scopes = {
            "pooled_rmsf": "system_all_declared_replicas",
            "dccm": "system_all_declared_replicas",
            "common_pca": "declared_view_all_systems_and_replicas",
            "time_lagged_independent_component_analysis": (
                "declared_view_all_systems_and_replicas"
            ),
            "clustering_kmeans": "declared_view_all_systems_and_replicas",
            "alternative_clustering": "declared_view_all_systems_and_replicas",
            "markov_state_models": "declared_view_all_systems_and_replicas",
        }
        for module_id, scope in expected_scopes.items():
            contract = get_ensemble_parallelism_contract(module_id)
            self.assertFalse(contract.replica_shard_may_finalize_primary_result)
            self.assertEqual(contract.primary_estimator_scope, scope)
            with self.assertRaises(EnsembleParallelismError):
                annotate_task_parallelism({
                    "module_id": module_id,
                    "replica_partition_id": "r1",
                })

    def test_replica_local_qc_and_unwrapping_are_valid_partitions(self):
        for module_id in ("structural_integrity_qc", "coordinate_cache"):
            task = annotate_task_parallelism({
                "module_id": module_id,
                "replica_partition_id": "r1",
            })
            contract = task["ensemble_parallelism_contract"]
            self.assertTrue(
                contract["replica_shard_may_finalize_primary_result"]
            )

    def test_common_pca_contract_requires_global_center_and_basis(self):
        contract = get_ensemble_parallelism_contract("common_pca")
        self.assertIn("pooled observation set", contract.reduction_rule)
        self.assertIn("basis construction", contract.reduction_rule)

    def test_pald_has_no_out_of_sample_assignment_phase(self):
        phases = get_ensemble_parallelism_contract(
            "pald_community_analysis"
        ).allowed_parallel_phases
        self.assertIn("global_community_analysis_on_declared_pooled_sample", phases)
        self.assertFalse(any("assignment_after" in phase for phase in phases))


if __name__ == "__main__":
    unittest.main()
