import json
import unittest
from pathlib import Path

from salsbury_md_analysis.registry import MODULES


ROOT = Path(__file__).resolve().parents[1]


class ScientificValidationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "validation" / "scientific_validation_summary.json"
        self.summary = json.loads(self.path.read_text(encoding="utf-8"))

    def test_snapshot_covers_registry_without_claiming_support(self):
        self.assertEqual(self.summary["execution"]["registered_module_count"], 44)
        self.assertEqual(self.summary["execution"]["covered_module_count"], 44)
        self.assertLessEqual(self.summary["execution"]["covered_module_count"], len(MODULES))
        self.assertEqual(self.summary["cross_validation"]["case_count"], 29)
        self.assertEqual(self.summary["cross_validation"]["passed_case_count"], 29)
        self.assertEqual(self.summary["cross_validation"]["failed_case_count"], 0)
        self.assertIn("not established", self.summary["release_boundary"])
        self.assertEqual(
            self.summary["key_metrics"][
                "bound_ion_coordination_number_exact_match_fraction"
            ],
            1.0,
        )
        self.assertEqual(
            self.summary["key_metrics"][
                "automatic_hbond_cutoff_exact_match_fraction"
            ],
            1.0,
        )
        self.assertEqual(
            self.summary["key_metrics"][
                "automatic_hbond_cutoff_occupancy_mismatch_count"
            ],
            0,
        )

    def test_cautionary_findings_are_retained(self):
        findings = " ".join(self.summary["unresolved_scientific_findings"]).lower()
        self.assertIn("convergence", findings)
        self.assertIn("omega", findings)
        self.assertIn("sasa", findings)
        self.assertIn("not validated", findings)

    def test_public_snapshot_contains_no_private_location(self):
        text = self.path.read_text(encoding="utf-8")
        for forbidden in ("/deac/", "/users/", "Dropbox/", "/private/tmp/"):
            self.assertNotIn(forbidden, text)

        hydrogen_bond_path = (
            ROOT / "validation" / "hydrogen_bond_discovery_cross_validation.json"
        )
        hydrogen_bond_text = hydrogen_bond_path.read_text(encoding="utf-8")
        for forbidden in ("/deac/", "/users/", "Dropbox/", "/private/tmp/"):
            self.assertNotIn(forbidden, hydrogen_bond_text)

        water_path = (
            ROOT / "validation" / "water_mediated_hydrogen_bond_smoke_summary.json"
        )
        water_text = water_path.read_text(encoding="utf-8")
        water = json.loads(water_text)
        self.assertEqual(water["technical_status"], "complete")
        self.assertEqual(water["scientific_status"], "not evaluated")
        self.assertEqual(water["execution"]["water_molecule_count"], 25882)
        self.assertEqual(water["execution"]["evaluated_frame_count"], 10)
        self.assertIn("not a convergence", " ".join(water["required_caveats"]))
        for forbidden in ("/deac/", "/users/", "Dropbox/", "/private/tmp/"):
            self.assertNotIn(forbidden, water_text)


if __name__ == "__main__":
    unittest.main()
