import unittest

from salsbury_md_analysis.interaction_fingerprints import (
    build_interaction_fingerprints,
)


class InteractionFingerprintTests(unittest.TestCase):
    def test_pairwise_complete_join_keeps_missingness_distinct_from_absence(self):
        settings = {
            "source_modules": ["hydrogen_bond_discovery", "ion_atmosphere"],
            "frame_join_policy": "pairwise_complete_observations_v1",
            "minimum_feature_occupancy": 0.0,
            "maximum_features": 20,
            "maximum_pair_comparisons": 100,
            "minimum_pair_observations": 2,
            "minimum_cooccurrence_count": 1,
        }
        identity = lambda index: {
            "system_id": "K-retained", "replica_id": "r1",
            "segment_id": "production", "source_frame_index": index,
        }
        reports = {
            "hydrogen_bond_discovery": {
                "module_id": "hydrogen_bond_discovery",
                "technical_status": "complete",
                "candidate_dictionary": [{
                    "bond_id": "SER75-to-aptamer", "donor_atom_index": 0,
                    "hydrogen_atom_index": 1, "acceptor_atom_index": 2,
                }],
                "frame_bond_matrix": [
                    {**identity(0), "binary_values": [1]},
                    {**identity(1), "binary_values": [0]},
                    {**identity(2), "binary_values": [1]},
                ],
            },
            "ion_atmosphere": {
                "module_id": "ion_atmosphere", "technical_status": "complete",
                "frame_records": [
                    {**identity(1), "species": {"K": {
                        "charge_class": "cation", "targets": {"aptamer": {
                            "ion_count_within_shell": {"4.0": 0},
                        }},
                    }}},
                    {**identity(2), "species": {"K": {
                        "charge_class": "cation", "targets": {"aptamer": {
                            "ion_count_within_shell": {"4.0": 1},
                        }},
                    }}},
                ],
            },
        }
        report = build_interaction_fingerprints(reports, settings)
        self.assertEqual(report["availability_status"], "available")
        self.assertEqual(len(report["frame_fingerprints"]), 3)
        first = report["frame_fingerprints"][0]
        self.assertEqual(first["available_source_modules"], ["hydrogen_bond_discovery"])
        self.assertEqual(len(report["cooccurrence_edges"]), 1)
        edge = report["cooccurrence_edges"][0]
        self.assertEqual(edge["pairwise_complete_frame_count"], 2)
        self.assertEqual(edge["cooccurrence_frame_count"], 1)


if __name__ == "__main__":
    unittest.main()
