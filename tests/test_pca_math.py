import math
import unittest

import numpy as np

from salsbury_md_analysis.pca_math import (
    CartesianCovariance,
    PCAError,
    mixture_covariance,
    principal_components,
    project,
    randomized_truncated_pca,
)


class PCAMathTests(unittest.TestCase):
    def test_streaming_covariance_and_projection_recover_correlated_axis(self):
        state = CartesianCovariance(2)
        for vector in ((-1.0, -2.0), (0.0, 0.0), (1.0, 2.0)):
            state.update(vector)
        solution = principal_components(state.population_covariance(), 1)
        component = solution.components[0]
        self.assertAlmostEqual(solution.total_variance_angstrom2, 10.0 / 3.0)
        self.assertAlmostEqual(component.eigenvalue_angstrom2, 10.0 / 3.0)
        self.assertAlmostEqual(component.explained_variance_fraction, 1.0)
        self.assertAlmostEqual(component.vector[0], 1.0 / math.sqrt(5.0))
        self.assertAlmostEqual(component.vector[1], 2.0 / math.sqrt(5.0))
        scores = [project(vector, state.mean(), solution.components)[0] for vector in (
            (-1.0, -2.0), (0.0, 0.0), (1.0, 2.0)
        )]
        self.assertAlmostEqual(scores[0], -math.sqrt(5.0))
        self.assertAlmostEqual(scores[1], 0.0)
        self.assertAlmostEqual(scores[2], math.sqrt(5.0))

    def test_merge_matches_direct_population_covariance(self):
        left = CartesianCovariance(2)
        right = CartesianCovariance(2)
        direct = CartesianCovariance(2)
        for vector in ((0.0, 1.0), (2.0, 3.0)):
            left.update(vector)
            direct.update(vector)
        for vector in ((4.0, 5.0), (6.0, 7.0), (8.0, 9.0)):
            right.update(vector)
            direct.update(vector)
        left.merge(right)
        self.assertEqual(left.count, direct.count)
        for observed, expected in zip(left.mean(), direct.mean()):
            self.assertAlmostEqual(observed, expected)
        for observed_row, expected_row in zip(
            left.population_covariance(), direct.population_covariance()
        ):
            for observed, expected in zip(observed_row, expected_row):
                self.assertAlmostEqual(observed, expected)

    def test_equal_state_mixture_differs_from_frame_weighting(self):
        short = CartesianCovariance(1)
        long = CartesianCovariance(1)
        for value in (-1.0, 1.0):
            short.update((value,))
        for value in (10.0, 10.0, 10.0, 10.0):
            long.update((value,))
        mean, covariance = mixture_covariance((short, long), (0.5, 0.5))
        self.assertAlmostEqual(mean[0], 5.0)
        self.assertAlmostEqual(covariance[0][0], 25.5)

    def test_randomized_truncated_solver_matches_dense_weighted_pca(self):
        generator = np.random.default_rng(20260812)
        latent = generator.normal(size=(120, 3))
        loadings = generator.normal(size=(3, 30))
        samples = latent @ loadings + 0.01 * generator.normal(size=(120, 30))
        weights = np.linspace(1.0, 2.0, len(samples))
        weights /= weights.sum()
        mean, randomized, diagnostics = randomized_truncated_pca(
            samples,
            weights,
            3,
            oversampling=8,
            power_iterations=4,
            random_seed=17,
            maximum_relative_residual=1.0e-6,
        )
        centered = samples - np.asarray(mean)
        covariance = (centered * weights[:, np.newaxis]).T @ centered
        dense = principal_components(covariance, 3)
        for observed, expected in zip(randomized.components, dense.components):
            self.assertAlmostEqual(
                observed.eigenvalue_angstrom2,
                expected.eigenvalue_angstrom2,
                places=7,
            )
        observed_vectors = np.column_stack(
            [component.vector for component in randomized.components]
        )
        expected_vectors = np.column_stack(
            [component.vector for component in dense.components]
        )
        singular_values = np.linalg.svd(
            observed_vectors.T @ expected_vectors, compute_uv=False
        )
        self.assertGreater(min(singular_values), 0.999999)
        self.assertLess(diagnostics["maximum_relative_residual"], 1.0e-6)
        repeated = randomized_truncated_pca(
            samples, weights, 3, oversampling=8, power_iterations=4, random_seed=17,
            maximum_relative_residual=1.0e-6,
        )[1]
        self.assertEqual(randomized, repeated)

    def test_randomized_solver_refines_without_relaxing_residual_gate(self):
        generator = np.random.default_rng(841)
        left, _ = np.linalg.qr(generator.normal(size=(120, 40)))
        right, _ = np.linalg.qr(generator.normal(size=(300, 40)))
        singular_values = np.geomspace(10.0, 1.0, 40)
        samples = (
            (left * singular_values) @ right.T
            + 0.001 * generator.normal(size=(120, 300))
        )
        weights = np.ones(120) / 120
        _, result, diagnostics = randomized_truncated_pca(
            samples,
            weights,
            5,
            oversampling=4,
            power_iterations=4,
            power_iteration_schedule=(4, 8, 12),
            random_seed=17,
            maximum_relative_residual=1.0e-3,
        )
        attempts = diagnostics["refinement_attempts"]
        self.assertEqual([row["power_iterations"] for row in attempts], [4, 8])
        self.assertGreater(attempts[0]["maximum_relative_residual"], 1.0e-3)
        self.assertLess(attempts[1]["maximum_relative_residual"], 1.0e-3)
        self.assertEqual(diagnostics["power_iterations"], 8)
        self.assertTrue(all(component.converged for component in result.components))

    def test_randomized_solver_fails_after_bounded_refinement(self):
        generator = np.random.default_rng(841)
        samples = generator.normal(size=(40, 60))
        with self.assertRaisesRegex(PCAError, r"q=0:.*q=1:"):
            randomized_truncated_pca(
                samples,
                np.ones(40),
                3,
                oversampling=2,
                power_iterations=0,
                power_iteration_schedule=(0, 1),
                maximum_relative_residual=1.0e-12,
            )


if __name__ == "__main__":
    unittest.main()
