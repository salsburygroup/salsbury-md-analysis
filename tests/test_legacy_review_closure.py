import json
import re
import unittest
from pathlib import Path

from salsbury_md_analysis.registry import MODULES


ROOT = Path(__file__).resolve().parents[1]


class LegacyReviewClosureTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "legacy_review_summary.json"
        self.raw = self.path.read_text(encoding="utf-8")
        self.summary = json.loads(self.raw)

    def test_md_destination_review_is_closed_without_support_overclaim(self):
        self.assertEqual(self.summary["reviewed_md_capability_count"], 434)
        self.assertEqual(
            self.summary["docking_capability_count_excluded_and_routed_separately"],
            62,
        )
        self.assertEqual(self.summary["public_suite_successor_count"], 280)
        self.assertEqual(self.summary["noncore_or_locked_capability_count"], 154)
        self.assertEqual(self.summary["remaining_unresolved_md_destination_count"], 0)
        self.assertTrue(self.summary["technical_destination_review_closed"])
        self.assertFalse(self.summary["scientific_support_granted_by_this_review"])
        self.assertEqual(self.summary["registered_public_module_count"], 44)
        self.assertLessEqual(self.summary["registered_public_module_count"], len(MODULES))

    def test_summary_is_integrity_pinned_and_public_safe(self):
        for key in (
            "source_crosswalk_sha256",
            "source_equivalence_matrix_sha256",
            "private_final_ledger_sha256",
        ):
            self.assertRegex(self.summary[key], r"^[0-9a-f]{64}$")
        for forbidden in ("/deac/", "apollo", "dropbox", "/users/"):
            self.assertNotIn(forbidden, self.raw.lower())
        self.assertIsNone(re.search(r"legacy-\w{12}-\d{3}", self.raw))

    def test_final_counts_cover_the_entire_md_scope(self):
        decisions = self.summary["final_decision_counts"]
        evidence = self.summary["evidence_status_counts"]
        self.assertEqual(sum(decisions.values()), 434)
        self.assertEqual(sum(evidence.values()), 434)
        self.assertEqual(
            decisions["approved_family_validated_suite_successor"]
            + decisions["approved_contract_suite_successor"],
            280,
        )


if __name__ == "__main__":
    unittest.main()
