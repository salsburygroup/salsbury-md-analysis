import unittest
from types import SimpleNamespace
from unittest.mock import patch

from salsbury_md_analysis.correlation_networks import (
    CorrelationNetworkError,
    _dccm_observation_accounting,
    correlation_network,
    correlation_profile_clustering,
)


class CorrelationNetworkTests(unittest.TestCase):
    def test_dccm_frame_accounting_is_exact_and_not_member_multiplied(self):
        report = _dccm_observation_accounting({
            "frame_selection": {
                "source_frame_count": 24,
                "selected_frame_count": 12,
            },
            "systems": [
                {"replicas": [
                    {"segments": [{"segment_id": "s1", "evaluated_frame_count": 5}]},
                    {"segments": [{"segment_id": "s1", "evaluated_frame_count": 7}]},
                ]},
            ],
        })
        self.assertEqual(report["selected_physical_frame_count"], 12)
        self.assertEqual(report["symmetry_expanded_observation_count"], 12)
        self.assertTrue(report["subsampling_triggered"])

    def test_dccm_frame_accounting_rejects_partial_evaluation(self):
        with self.assertRaises(CorrelationNetworkError):
            _dccm_observation_accounting({
                "frame_selection": {
                    "source_frame_count": 24,
                    "selected_frame_count": 12,
                },
                "systems": [
                    {"replicas": [
                        {"segments": [
                            {"segment_id": "s1", "evaluated_frame_count": 11}
                        ]}
                    ]},
                ],
            })

    def test_threshold_network_preserves_sign_components_and_strength(self):
        matrix = [
            [1.0, 0.8, 0.1, None],
            [0.8, 1.0, -0.7, 0.0],
            [0.1, -0.7, 1.0, 0.2],
            [None, 0.0, 0.2, 1.0],
        ]
        report = correlation_network(matrix, 0.6, include_negative=True)
        self.assertEqual(report["edge_count"], 2)
        self.assertEqual(report["undefined_pair_count"], 1)
        self.assertEqual(report["connected_components"], [[0, 1, 2], [3]])
        self.assertAlmostEqual(report["node_absolute_strengths"][1], 1.5)
        self.assertEqual({edge["sign"] for edge in report["edges"]}, {"positive", "negative"})

    def test_negative_edges_can_be_excluded(self):
        matrix = [[1.0, -0.9], [-0.9, 1.0]]
        report = correlation_network(matrix, 0.5, include_negative=False)
        self.assertEqual(report["edge_count"], 0)
        self.assertEqual(report["connected_components"], [[0], [1]])

    def test_correlation_profile_hdbscan_labels_are_canonical(self):
        captured = {}

        class FakeHDBSCAN:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def fit_predict(self, values):
                captured["shape"] = values.shape
                return [8, 8, -1, 3]

        matrix = [
            [1.0, 0.8, 0.1, 0.0],
            [0.8, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, -0.7],
            [0.0, 0.1, -0.7, 1.0],
        ]
        with patch(
            "salsbury_md_analysis.correlation_networks.importlib.import_module",
            return_value=SimpleNamespace(HDBSCAN=FakeHDBSCAN),
        ):
            report = correlation_profile_clustering(
                matrix, 2, 1, input_mode="absolute_similarity"
            )
        self.assertEqual(report["labels"], [0, 0, -1, 1])
        self.assertEqual(report["cluster_sizes"], [2, 1])
        self.assertEqual(report["noise_count"], 1)
        self.assertEqual(captured["metric"], "precomputed")
        self.assertEqual(captured["shape"], (4, 4))


if __name__ == "__main__":
    unittest.main()
