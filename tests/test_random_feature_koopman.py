import math
import unittest

import numpy as np

from salsbury_md_analysis.random_feature_koopman import (
    fit_random_feature_koopman,
    random_fourier_features,
)


class RandomFeatureKoopmanTests(unittest.TestCase):
    def settings(self):
        return {
            "source_module": "time_lagged_independent_component_analysis",
            "component_indices": [1, 2],
            "lag_frames": 2,
            "component_count": 2,
            "random_feature_counts": [12, 16],
            "bandwidth_scales": [0.75, 1.0],
            "random_seeds": [0, 7, 19, 41],
            "cross_validation_folds": 3,
            "covariance_regularization": 1.0e-6,
            "covariance_eigenvalue_cutoff": 1.0e-10,
            "minimum_pairs_per_segment": 10,
            "maximum_bandwidth_observations": 200,
            "maximum_feature_matrix_elements": 100_000,
            "maximum_seed_vamp_e_relative_range": 100.0,
            "minimum_seed_subspace_similarity": 0.0,
        }

    def trajectories(self):
        segments = []
        for phase in (0.0, 0.35):
            t = np.linspace(0.0, 10.0 * math.pi, 180)
            slow = np.sin(t + phase)
            curved = slow * slow + 0.08 * np.cos(3.0 * t)
            segments.append(np.column_stack([slow, curved]))
        return segments

    def test_random_map_is_reproducible_for_one_seed(self):
        values = self.trajectories()[0]
        first = random_fourier_features(
            values, feature_count=16, bandwidth=1.2, seed=7
        )
        second = random_fourier_features(
            values, feature_count=16, bandwidth=1.2, seed=7
        )
        for left, right in zip(first, second):
            self.assertTrue(np.array_equal(left, right))

    def test_grid_records_all_seeds_and_selects_only_a_stable_candidate(self):
        result = fit_random_feature_koopman(self.trajectories(), self.settings())
        self.assertEqual(len(result["hyperparameter_candidates"]), 4)
        self.assertEqual(result["selection_status"], "selected_stable_candidate")
        selected = result["selected_hyperparameters"]
        self.assertEqual(selected["primary_prespecified_seed"], 0)
        candidate = next(
            row for row in result["hyperparameter_candidates"]
            if row["random_feature_count"] == selected["random_feature_count"]
            and row["bandwidth_scale"] == selected["bandwidth_scale"]
        )
        self.assertEqual(
            [row["random_seed"] for row in candidate["seed_runs"]],
            [0, 7, 19, 41],
        )
        self.assertEqual(
            candidate["seed_stability_gate"]["configured_random_seeds"],
            [0, 7, 19, 41],
        )
        self.assertEqual(len(result["selected_primary_projections"]), 2)

    def test_subspace_gate_can_withhold_all_candidates(self):
        settings = self.settings()
        settings["minimum_seed_subspace_similarity"] = 1.0
        result = fit_random_feature_koopman(self.trajectories(), settings)
        self.assertEqual(result["selection_status"], "no_stable_candidate")
        self.assertEqual(result["eligible_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
