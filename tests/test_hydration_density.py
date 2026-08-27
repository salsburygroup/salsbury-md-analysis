import unittest

import numpy as np

from salsbury_md_analysis.hydration_density import (
    _component_record, _components,
)
from salsbury_md_analysis.interaction_fingerprints import (
    build_interaction_fingerprints,
)


class HydrationDensityTests(unittest.TestCase):
    def test_face_connected_components_and_channel_label_are_explicit(self):
        voxels = {
            (0, 3, 3), (1, 3, 3), (2, 3, 3), (3, 3, 3), (6, 6, 6)
        }
        components = _components(voxels)
        self.assertEqual([len(row) for row in components], [4, 1])
        record = _component_record(
            system_id="control", species="water", component_number=1,
            voxels=components[0], counts={voxel: 8 for voxel in components[0]},
            frame_count=10, origin=np.zeros(3), spacing=2.0,
            shape=(7, 7, 7), minimum_channel_depth=4.0,
        )
        self.assertTrue(record["touches_grid_boundary"])
        self.assertTrue(record["geometric_channel_candidate"])
        self.assertEqual(record["maximum_interior_depth_angstrom"], 6.0)
        self.assertAlmostEqual(record["maximum_voxel_frame_occupancy"], 0.8)

    def test_density_components_join_interaction_fingerprints_by_exact_frame(self):
        component_id = "control|ion:ZN|density-component-1"
        reports = {"hydration_density_channels": {
            "module_id": "hydration_density_channels",
            "technical_status": "complete",
            "density_components": [{
                "feature_id": component_id, "species": "ion:ZN",
                "centroid_angstrom": [0.0, 0.0, 0.0],
                "volume_angstrom3": 3.375,
                "geometric_channel_candidate": False,
            }],
            "frame_feature_records": [
                {"system_id": "control", "replica_id": "r1",
                 "segment_id": "s1", "source_frame_index": 0,
                 "active_feature_ids": [component_id]},
                {"system_id": "control", "replica_id": "r1",
                 "segment_id": "s1", "source_frame_index": 1,
                 "active_feature_ids": []},
            ],
        }}
        result = build_interaction_fingerprints(reports, {
            "source_modules": ["hydration_density_channels"],
            "maximum_features": 10, "maximum_pair_comparisons": 10,
            "minimum_feature_occupancy": 0.0,
            "minimum_pair_observations": 1, "minimum_cooccurrence_count": 1,
        })
        self.assertEqual(result["availability_status"], "available")
        self.assertEqual(result["feature_occupancies"][0]["occupancy_fraction"], 0.5)
        self.assertEqual(
            result["feature_dictionary"][0]["interaction_type"],
            "aligned_ion_density_component",
        )


if __name__ == "__main__":
    unittest.main()
