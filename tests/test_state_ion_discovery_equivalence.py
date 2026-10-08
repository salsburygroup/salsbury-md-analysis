"""Exact regression checks against the original greedy discovery definition."""
import unittest
from unittest.mock import patch
from scipy.spatial import cKDTree
import numpy as np
from salsbury_md_analysis.state_ion_stability import _discover_centers


def reference(observations, radius, maximum_sites):
    if not observations:
        return []
    points = np.asarray([row[2] for row in observations], dtype=np.float64)
    remaining = list(range(len(points)))
    centers = []
    while remaining and len(centers) < maximum_sites:
        best = min(remaining, key=lambda i: (-sum(float(np.linalg.norm(points[i]-points[j])) <= radius for j in remaining), i))
        selected = [j for j in remaining if float(np.linalg.norm(points[best]-points[j])) <= radius]
        centers.append(np.mean(points[selected], axis=0))
        remaining = [j for j in remaining if j not in set(selected)]
    return centers


class DiscoveryEquivalence(unittest.TestCase):
    def check(self, points, radius=1.5, maximum_sites=32):
        observations = [(i, str(i), p) for i,p in enumerate(points)]
        expected = reference(observations, radius, maximum_sites)
        actual = _discover_centers(observations, radius, maximum_sites)
        self.assertEqual(len(expected),len(actual))
        for a,b in zip(expected,actual):
            np.testing.assert_array_equal(a,b)

    def test_random_clouds_and_clusters(self):
        for seed in range(20):
            rng=np.random.default_rng(seed)
            for scale in [0.1,1.,10.]:
                with self.subTest(seed=seed,scale=scale):
                    self.check(rng.normal(size=(45,3))*scale,maximum_sites=12)

    def test_exact_and_adjacent_thresholds(self):
        radius=1.5
        a=np.nextafter(radius,0.); b=np.nextafter(radius,np.inf)
        points=np.array([[0.,0.,0.],[radius,0,0],[a,0,0],[b,0,0],[-radius,0,0],[0,.9,1.2],[0,0,0],[20,0,0]])
        self.check(points,radius)

    def test_ties_duplicates_sparse_dense_and_truncation(self):
        for n in [0,1,20]:
            for points in [np.zeros((n,3)),np.arange(n*3).reshape((n,3))*5.]:
                for maximum in [1,3,32]:
                    self.check(points,maximum_sites=maximum)

    def test_underflow_and_fallback_boundary(self):
        self.check(np.array([[0., 0., 0.], [1e-170, 0., 0.]]), 1e-200)
        threshold = float(np.sqrt(np.finfo(np.float64).tiny))
        for radius in [np.nextafter(threshold, 0.), threshold,
                       np.nextafter(threshold, np.inf), threshold * 2]:
            with self.subTest(radius=radius):
                self.check(np.array([[0., 0., 0.], [radius, 0., 0.],
                                     [np.nextafter(radius, 0.), 0., 0.],
                                     [np.nextafter(radius, np.inf), 0., 0.],
                                     [radius * .6, radius * .8, 0.],
                                     [radius * 3, 0., 0.]]), radius)

    def test_no_neighbor_updates_after_terminal_selection(self):
        for points, maximum in [(np.zeros((20, 3)), 32),
                                (np.zeros((20, 3)), 1),
                                (np.arange(60).reshape(20, 3) * 5., 1)]:
            real_tree = cKDTree(points)
            with patch('salsbury_md_analysis.state_ion_stability.cKDTree') as factory:
                factory.return_value.query_ball_point.side_effect = real_tree.query_ball_point
                self.check(points, maximum_sites=maximum)
                self.assertEqual(factory.return_value.query_ball_point.call_count, 21)

    def test_coordinate_translation_and_permutation(self):
        rng=np.random.default_rng(140)
        points=rng.normal(size=(60,3))
        for offset in [0.,1.e6,-1.e6]:
            self.check(points[rng.permutation(60)]+offset)

if __name__ == '__main__':
    unittest.main()
