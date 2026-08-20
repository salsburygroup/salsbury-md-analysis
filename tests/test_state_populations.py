import unittest

from salsbury_md_analysis.state_populations import summarize_state_populations


class StatePopulationTests(unittest.TestCase):
    def test_system_replica_coverage_and_pairwise_differences_are_explicit(self):
        rows = [
            {"system_id": "bound", "replica_id": "r1", "cluster_id": 1},
            {"system_id": "bound", "replica_id": "r1", "cluster_id": 1},
            {"system_id": "bound", "replica_id": "r1", "cluster_id": None},
            {"system_id": "unbound", "replica_id": "r1", "cluster_id": 1},
            {"system_id": "unbound", "replica_id": "r1", "cluster_id": 2},
            {"system_id": "unbound", "replica_id": "r2", "cluster_id": 2},
        ]
        report = summarize_state_populations(rows, "cluster_id")
        self.assertEqual(report["state_ids"], [1, 2])
        bound, unbound = report["system_populations"]
        self.assertAlmostEqual(bound["assigned_coverage_fraction"], 2 / 3)
        self.assertEqual(unbound["state_populations"][1]["count"], 2)
        comparison = report["pairwise_system_differences"][0]
        self.assertAlmostEqual(
            comparison["state_fraction_differences"][0][
                "left_minus_right_fraction_of_all_evaluated"
            ],
            1 / 3,
        )
        self.assertEqual(len(report["replica_populations"]), 3)


if __name__ == "__main__":
    unittest.main()
