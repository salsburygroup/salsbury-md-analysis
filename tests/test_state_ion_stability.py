import unittest

from salsbury_md_analysis.state_ion_stability import analyze_state_ion_stability


class StateIonStabilityTests(unittest.TestCase):
    def test_equivalent_ion_exchange_preserves_stable_site(self):
        frames = []
        for index in range(24):
            first, second = ("K-1", "K-2") if index % 2 == 0 else ("K-2", "K-1")
            frames.append({
                "system_id": "TBA",
                "state_id": 1,
                "frame_id": f"frame-{index}",
                "ions": [
                    {"ion_id": first, "element": "K", "coordinates_angstrom": [0.05 * (index % 3), 0.0, 0.0]},
                    {"ion_id": second, "element": "K", "coordinates_angstrom": [20.0 + index, 0.0, 0.0]},
                ],
            })
        report = analyze_state_ion_stability({
            "coordinates_aligned_to_polymer": True,
            "frames": frames,
            "settings": {
                "site_discovery_radius_angstrom": 0.5,
                "site_assignment_cutoff_angstrom": 0.75,
                "minimum_state_frames": 20,
                "minimum_site_occupancy_fraction": 0.8,
                "maximum_site_rmsf_angstrom": 0.3,
                "maximum_sites_per_species": 32,
            },
        })
        state = report["state_reports"][0]
        self.assertEqual(state["technical_status"], "complete")
        self.assertEqual(len(state["stable_sites"]), 1)
        site = state["stable_sites"][0]
        self.assertEqual(site["element"], "K")
        self.assertEqual(site["occupancy_fraction"], 1.0)
        assigned_ids = {
            row["ion_id"] for row in site["frame_assignments"]
        }
        self.assertEqual(assigned_ids, {"K-1", "K-2"})

    def test_alignment_provenance_is_required(self):
        report = None
        with self.assertRaisesRegex(ValueError, "aligned"):
            analyze_state_ion_stability({"frames": [{}]})


if __name__ == "__main__":
    unittest.main()
