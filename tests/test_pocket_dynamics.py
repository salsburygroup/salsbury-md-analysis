import unittest
from collections import Counter

import numpy as np

from salsbury_md_analysis.pocket_dynamics import (
    _discover_frequency_region_flats,
    _flat_pocket_voxels,
    _frame_pockets,
    _representative_frame_rows,
    _track_instances,
)


class PocketDynamicsTests(unittest.TestCase):
    def _settings(self):
        return {
            "minimum_clearance_angstrom": 1.0,
            "maximum_surface_distance_angstrom": 4.0,
            "minimum_seed_clearance_angstrom": 2.0,
            "minimum_seed_separation_angstrom": 4.0,
            "pocket_growth_radius_angstrom": 4.0,
            "neighborhood_radius_angstrom": 4.0,
            "minimum_nearby_atoms": 4,
            "minimum_nearby_residues": 3,
            "minimum_occupied_directions": 4,
            "maximum_directional_imbalance": 0.25,
            "minimum_pocket_voxels": 1,
            "maximum_pockets_per_frame": 5,
            "grid_spacing_angstrom": 2.0,
            "residue_jaccard_threshold": 0.5,
            "maximum_centroid_distance_angstrom": 3.0,
            "maximum_tracking_comparisons": 100,
        }

    def test_enclosed_grid_point_becomes_geometric_pocket(self):
        coordinates = np.asarray([
            [-3.0, 0.0, 0.0], [3.0, 0.0, 0.0],
            [0.0, -3.0, 0.0], [0.0, 3.0, 0.0],
            [0.0, 0.0, -3.0], [0.0, 0.0, 3.0],
        ])
        pockets = _frame_pockets(
            coordinates, np.asarray([[0.0, 0.0, 0.0]]), (1, 1, 1),
            [f"A:{index}:ALA" for index in range(6)], self._settings(),
        )
        self.assertEqual(len(pockets), 1)
        self.assertEqual(pockets[0]["volume_angstrom3"], 8.0)
        self.assertAlmostEqual(pockets[0]["mean_enclosure_score"], 1.0)

    def test_residue_overlap_tracks_repeated_instances(self):
        instances = [
            {"pocket_instance_index": 0, "centroid_angstrom": [0.0, 0.0, 0.0],
             "lining_residue_ids": ["A:1:ALA", "A:2:GLY"],
             "system_id": "a", "replica_id": "r1", "segment_id": "s1",
             "source_frame_index": 0},
            {"pocket_instance_index": 1, "centroid_angstrom": [0.5, 0.0, 0.0],
             "lining_residue_ids": ["A:1:ALA", "A:2:GLY", "A:3:SER"],
             "system_id": "a", "replica_id": "r1", "segment_id": "s1",
             "source_frame_index": 1},
        ]
        clusters, comparisons = _track_instances(instances, self._settings())
        self.assertEqual(len(clusters), 1)
        self.assertEqual(comparisons, 1)
        self.assertEqual(instances[0]["pocket_cluster_id"], instances[1]["pocket_cluster_id"])

    def test_one_cluster_cannot_receive_two_instances_from_the_same_frame(self):
        common = {
            "system_id": "a", "replica_id": "r1", "segment_id": "s1",
            "source_frame_index": 0,
            "lining_residue_ids": ["A:1:ALA", "A:2:GLY"],
        }
        instances = [
            {**common, "pocket_instance_index": 0,
             "centroid_angstrom": [0.0, 0.0, 0.0]},
            {**common, "pocket_instance_index": 1,
             "centroid_angstrom": [0.5, 0.0, 0.0]},
        ]
        clusters, _ = _track_instances(instances, self._settings())
        self.assertEqual(len(clusters), 2)

    def test_frequency_regions_use_per_system_threshold_and_connected_size(self):
        regions = _discover_frequency_region_flats(
            {
                "control": Counter({0: 8, 1: 7, 4: 1}),
                "variant": Counter({0: 1, 1: 2, 4: 9}),
            },
            {"control": 10, "variant": 10},
            (5, 1, 1),
            {
                "minimum_region_frequency_fraction": 0.5,
                "minimum_region_voxels": 2,
                "maximum_frequency_regions": 5,
            },
        )
        self.assertEqual(regions, [[0, 1]])

    def test_frequency_map_unions_overlapping_frame_pockets(self):
        pockets = [
            {"voxel_indices": [[0, 0, 0], [1, 0, 0]]},
            {"voxel_indices": [[1, 0, 0], [2, 0, 0]]},
        ]
        self.assertEqual(_flat_pocket_voxels(pockets, (3, 1, 1)), (0, 1, 2))

    def test_representatives_are_observed_frames(self):
        frames = [
            {
                "system_id": "a", "replica_id": "r1", "segment_id": "s1",
                "source_frame_index": index, "axis_kind": "sample",
                "axis_unit": "sample", "axis_value": index,
            }
            for index in range(3)
        ]
        rows = _representative_frame_rows(
            [(8.0, frames[0]), (16.0, frames[1]), (24.0, frames[2])], 3
        )
        self.assertEqual(
            [row["source_frame_index"] for row in rows], [2, 1, 0]
        )
        self.assertEqual(
            [row["selection_reason"] for row in rows],
            [
                "maximum_observed_volume", "closest_observed_to_mean_volume",
                "additional_central_observed_volume",
            ],
        )


if __name__ == "__main__":
    unittest.main()
