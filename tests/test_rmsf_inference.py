import unittest

from salsbury_md_analysis.rmsf_inference import (
    RMSFInferenceError,
    rmsf_permutation_test,
    rmsf_replica_permutation_comparisons,
)


class RMSFInferenceTests(unittest.TestCase):
    def test_comparative_report_infers_replicas_as_exchangeable_units(self):
        def system(system_id, offset):
            return {
                "system_id": system_id,
                "replicas": [{
                    "replica_id": f"replica-{index + 1}",
                    "technical_status": "complete",
                    "atom_statistics": [{
                        "common_atom_index": atom_index,
                        "chain_id": "A", "residue_id": 1,
                        "residue_name": "ALA", "atom_name": atom_name,
                        "rmsf_angstrom": value + offset + index * 0.1,
                    } for atom_index, (atom_name, value) in enumerate((
                        ("CA", 1.0), ("CB", 2.0),
                    ))],
                } for index in range(3)],
            }
        report = rmsf_replica_permutation_comparisons(
            {"technical_status": "complete", "systems": [
                system("control", 0.0), system("lesion", 1.0),
            ]},
            {"mode": "all_pairs"},
        )
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(
            report["exchangeable_unit"],
            "independently_declared_simulation_replica",
        )
        self.assertEqual(report["comparisons"][0]["comparison_status"], "complete")
        self.assertEqual(report["comparisons"][0]["result"]["group_a_unit_count"], 3)

    def test_exact_permutation_is_unit_level_and_max_t_adjusted(self):
        report = rmsf_permutation_test(
            [[1.0, 1.0], [1.1, 1.0], [0.9, 1.0]],
            [[4.0, 1.0], [4.1, 1.0], [3.9, 1.0]],
        )
        self.assertEqual(report["method"], "exact")
        self.assertEqual(report["evaluated_partition_count"], 20)
        self.assertLess(report["two_sided_pointwise_p_values"][0], 0.2)
        self.assertEqual(report["two_sided_pointwise_p_values"][1], 1.0)
        self.assertGreaterEqual(
            report["max_t_familywise_p_values"][0], report["two_sided_pointwise_p_values"][0]
        )

    def test_frame_pseudoreplication_is_blocked_by_minimum_unit_gate(self):
        with self.assertRaises(RMSFInferenceError):
            rmsf_permutation_test([[1.0, 2.0]], [[2.0, 3.0], [2.1, 3.1]])


if __name__ == "__main__":
    unittest.main()
