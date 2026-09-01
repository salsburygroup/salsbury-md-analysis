import unittest
from pathlib import Path

from salsbury_md_analysis.finding_picker import _quality_control_records


class FindingPickerConvergenceTests(unittest.TestCase):
    def test_convergence_qc_reports_ess_counts_without_scientific_verdict(self):
        report = {
            "module_id": "convergence_uncertainty",
            "diagnostic_summary": {
                "effective_sample_size_reference": 20.0,
                "effective_sample_size_above_reference_count": 9,
                "series_count": 40,
                "by_metric": {
                    "radius_of_gyration_angstrom": {
                        "effective_sample_size_above_reference_count": 7,
                        "series_count": 20,
                    },
                    "rmsd_angstrom": {
                        "effective_sample_size_above_reference_count": 2,
                        "series_count": 20,
                    },
                },
            },
            "issues": [],
        }
        records = _quality_control_records(report, Path("report.json"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "quantitative_diagnostics")
        self.assertEqual(records[0]["severity"], "information")
        self.assertIn("9/40", records[0]["statement"])
        self.assertIn("2/20", records[0]["statement"])
        self.assertIn("7/20", records[0]["statement"])
        self.assertIn("No convergence", records[0]["statement"])
        self.assertNotIn("passed", records[0]["statement"].lower())


if __name__ == "__main__":
    unittest.main()
