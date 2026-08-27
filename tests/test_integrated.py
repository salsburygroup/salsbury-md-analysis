import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.finding_picker import prioritize_findings
from salsbury_md_analysis.integrated import (
    IntegratedAnalysisError,
    integrated_comparison_results,
    json_pointer,
)


class IntegratedTests(unittest.TestCase):
    def test_json_pointer_resolves_dict_list_and_escapes(self):
        document = {"rows": [{"a/b": {"~key": 7}}]}
        self.assertEqual(json_pointer(document, "/rows/0/a~1b/~0key"), 7)
        with self.assertRaises(IntegratedAnalysisError):
            json_pointer(document, "/rows/2")

    def test_completed_campaign_integration_is_canonical_picker_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis-config.json").write_text(json.dumps({
                "comparisons": {
                    "mode": "all_pairs", "reference_system_id": None,
                    "alpha": 0.05,
                },
                "reporting": {"maximum_findings": 50},
            }), encoding="utf-8")
            (root / "module-coverage.json").write_text(json.dumps({
                "comparison_system_ids": ["control", "variant"],
            }), encoding="utf-8")
            report_path = root / "results" / "rmsf" / "report.json"
            report_path.parent.mkdir(parents=True)
            identity = {
                "common_atom_index": 0, "chain_id": "A",
                "residue_name": "ALA", "residue_number": 10,
                "insertion_code": "", "atom_name": "CA",
            }
            report_path.write_text(json.dumps({
                "module_id": "pooled_rmsf", "technical_status": "complete",
                "systems": [
                    {"system_id": "control", "atom_statistics": [{
                        **identity, "frame_pooled_rmsf_angstrom": 1.0,
                    }]},
                    {"system_id": "variant", "atom_statistics": [{
                        **identity, "frame_pooled_rmsf_angstrom": 2.5,
                    }]},
                ],
            }), encoding="utf-8")

            integrated = integrated_comparison_results(root)
            self.assertEqual(integrated["technical_status"], "complete")
            self.assertEqual(integrated["reviewed_report_count"], 1)
            self.assertEqual(integrated["unreviewed_complete_report_count"], 0)
            self.assertEqual(integrated["comparison_candidate_count"], 1)
            self.assertFalse((root / "prioritized_findings.json").exists())

            integrated_path = (
                root / "results" / "integrated-comparison" / "report.json"
            )
            integrated_path.parent.mkdir()
            integrated_path.write_text(json.dumps(integrated), encoding="utf-8")
            selected = prioritize_findings(root, maximum_findings=50)
            pair_rows = [
                row for row in selected["findings"]
                if len(row["system_ids"]) == 2
            ]
            self.assertEqual(len(pair_rows), 1)
            self.assertEqual(
                pair_rows[0]["integration_report_path"],
                str(integrated_path.resolve()),
            )
            self.assertIn(str(integrated_path.resolve()), pair_rows[0]["report_paths"])
            accounting = {
                row["module_id"]: row for row in selected["module_accounting"]
            }
            self.assertEqual(
                accounting["integrated_comparison"]["disposition"],
                "interpretive_context",
            )


if __name__ == "__main__":
    unittest.main()
