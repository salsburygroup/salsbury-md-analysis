import copy
import unittest
import numpy as np
from salsbury_md_analysis.clustering_presentation import (
    evaluate_partition, report_models, select_primary_partitions, apply_primary_findings,
)


class ClusteringPresentationTests(unittest.TestCase):
    def setUp(self):
        self.vectors = np.array([[i * .01, 0] for i in range(20)] + [[10+i*.01, 0] for i in range(20)])
        self.metadata = [{"system_id":"A","replica_id":"r1","frame_index":i} for i in range(40)]
        self.labels = [0]*20+[1]*20

    def model(self, method, score=None, scope="global"):
        evaluation = evaluate_partition(self.vectors, self.metadata, self.labels)
        if score is not None: evaluation["score"] = score
        return report_models({"module_id":method,"selected_model":{"presentation_evaluation":evaluation}},
                             f"/study/results/conformational-views/{scope}/{method}/report.json")[0]

    def test_common_geometry_and_sample_are_identical_between_methods(self):
        first = evaluate_partition(self.vectors, self.metadata, self.labels, maximum_observations=20)
        second = evaluate_partition(self.vectors, self.metadata, [1-v for v in self.labels], maximum_observations=20)
        self.assertEqual(first, second)
        self.assertEqual(first["evaluated_observation_count"], 20)
        self.assertGreater(first["score"], .9)

    def test_only_one_primary_per_scope(self):
        selection = select_primary_partitions([self.model("clustering_kmeans", .7), self.model("clustering_imwkmeans", .6), self.model("clustering_kmeans", .5, "interface")])
        self.assertEqual(len(selection["groups"]), 2)
        for group in selection["groups"]:
            self.assertEqual(sum(r["presentation_role"]=="primary" for r in group["candidates"]), 1)

    def test_identical_numbers_with_different_feature_definitions_are_not_comparable(self):
        first = evaluate_partition(self.vectors, self.metadata, self.labels, feature_definition={"feature_source": "common_pca"})
        second = evaluate_partition(self.vectors, self.metadata, self.labels, feature_definition={"feature_source": "tica"})
        self.assertNotEqual(first["geometry_and_identity_sha256"], second["geometry_and_identity_sha256"])

    def test_incompatible_geometry_abstains(self):
        a, b = self.model("clustering_kmeans"), self.model("clustering_imwkmeans")
        b["evaluation"]["geometry_and_identity_sha256"] = "different"
        group = select_primary_partitions([a,b])["groups"][0]
        self.assertEqual(group["status"], "abstained")
        self.assertIsNone(group["primary_candidate_id"])

    def test_exact_ties_are_declared_not_overclaimed(self):
        group = select_primary_partitions([self.model("clustering_kmeans"), self.model("clustering_imwkmeans")])["groups"][0]
        self.assertEqual(group["status"], "tied")
        self.assertEqual(len(group["tied_candidate_ids"]), 2)

    def test_noise_missing_identity_and_legacy_abstain(self):
        labels = self.labels.copy(); labels[0] = -1
        self.assertEqual(evaluate_partition(self.vectors, self.metadata, labels)["status"], "ineligible")
        self.assertEqual(evaluate_partition(self.vectors, [{}]*40, self.labels)["status"], "ineligible")
        legacy = self.model("clustering_kmeans"); legacy["evaluation"] = None
        self.assertEqual(select_primary_partitions([legacy])["groups"][0]["status"], "abstained")

    def test_diagnostic_and_losing_partition_cannot_headline(self):
        a, b = self.model("clustering_kmeans", .7), self.model("clustering_imwkmeans", .6)
        selection = select_primary_partitions([a,b])
        rows = [{"module_id":r["module_id"], "report_path":r["report_path"], "comparison_family":"state_population", "presentation_eligible":True} for r in (a,b)]
        rows.append({"module_id":"clustering_kmeans", "comparison_family":"clustering_kmeans:model_selection", "presentation_eligible":True})
        result = apply_primary_findings(rows, selection)
        self.assertTrue(result[0]["presentation_eligible"])
        self.assertFalse(result[1]["presentation_eligible"])
        self.assertFalse(result[2]["presentation_eligible"])


if __name__ == "__main__": unittest.main()
