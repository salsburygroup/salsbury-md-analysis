import unittest

from salsbury_md_analysis.grouped_regularized_classification import (
    GroupedRegularizedClassificationError,
    nested_grouped_classification,
)


class GroupedRegularizedClassificationTests(unittest.TestCase):
    def _settings(self):
        return {
            "estimators": ["logistic_l2", "elastic_net"],
            "inverse_regularization_strengths": [0.1, 1.0],
            "elastic_net_l1_ratios": [0.25, 0.75],
            "class_weight": "balanced",
            "maximum_iterations": 5000,
            "minimum_outer_groups": 4,
            "minimum_inner_folds": 2,
            "random_seed": 17,
        }

    def test_nested_group_holdout_separates_systems_without_frame_splitting(self):
        vectors = []
        labels = []
        groups = []
        for label in (0, 1):
            for replica in range(3):
                for frame in range(6):
                    vectors.append([
                        float(label),
                        float((frame + replica) % 2),
                        float(label if frame != 0 else 1 - label),
                    ])
                    labels.append(label)
                    groups.append((f"system-{label}", f"replica-{replica}"))
        report = nested_grouped_classification(
            vectors, labels, groups, ["diagnostic", "noise", "partial"],
            self._settings(),
        )
        self.assertEqual(report["outer_fold_count"], 6)
        self.assertGreater(report["pooled_outer_held_out_metrics"]["accuracy"], 0.9)
        self.assertEqual(
            sorted(tuple(row["held_out_group"]) for row in report["fold_reports"]),
            sorted(set(groups)),
        )
        self.assertTrue(all(
            any(candidate["eligible"] for candidate in row["inner_tuning"])
            for row in report["fold_reports"]
        ))
        self.assertIn(
            report["final_model"]["selected_parameters"]["estimator"],
            {"logistic_l2", "elastic_net"},
        )
        self.assertIsNotNone(report["pooled_without_top_feature_metrics"])

    def test_one_group_per_class_fails_closed(self):
        with self.assertRaises(GroupedRegularizedClassificationError):
            nested_grouped_classification(
                [[0.0], [0.1], [1.0], [1.1]], [0, 0, 1, 1],
                [("a",)] * 2 + [("b",)] * 2, ["bond"], self._settings(),
            )


if __name__ == "__main__":
    unittest.main()
