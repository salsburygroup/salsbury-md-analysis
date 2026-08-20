import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.presentation import summarize_timeseries_presentations


class PresentationTests(unittest.TestCase):
    def _segments(self):
        return [
            {
                "system_id": "s1", "replica_id": "r1", "segment_id": "production",
                "records": [
                    {"source_frame_index": index, "rmsd_angstrom": value, "rg_angstrom": 10.0 + value}
                    for index, value in enumerate((0.0, 1.0, 2.0, 3.0))
                ],
            },
            {
                "system_id": "s1", "replica_id": "r2", "segment_id": "production",
                "records": [
                    {"source_frame_index": index, "rmsd_angstrom": value, "rg_angstrom": 12.0 + value}
                    for index, value in enumerate((0.5, 1.5, 2.5, 3.5))
                ],
            },
        ]

    def test_scott_histogram_is_primary_except_for_rmsd(self):
        report = summarize_timeseries_presentations(self._segments())
        by_field = {row["field"]: row for row in report["presentations"]}
        self.assertEqual(by_field["rg_angstrom"]["primary_presentation"], "histogram")
        self.assertEqual(by_field["rg_angstrom"]["histogram_rule"], "scott")
        self.assertEqual(by_field["rg_angstrom"]["histogram_status"], "complete")
        self.assertEqual(
            sum(row["count"] for row in by_field["rg_angstrom"]["distribution"]["histogram"]),
            8,
        )
        self.assertEqual(
            by_field["rmsd_angstrom"]["primary_presentation"],
            "replica_resolved_time_series",
        )
        self.assertEqual(by_field["rmsd_angstrom"]["histogram_status"], "not_applicable")

    def test_any_tokenized_rmsd_metric_is_exempt(self):
        segments = self._segments()
        for segment in segments:
            for record in segment["records"]:
                record["backbone_rmsd_angstrom"] = record["rmsd_angstrom"]
        report = summarize_timeseries_presentations(
            segments, fields=["backbone_rmsd_angstrom"]
        )
        self.assertEqual(
            report["presentations"][0]["primary_presentation"],
            "replica_resolved_time_series",
        )

    def test_constant_field_is_explicitly_not_estimable(self):
        segments = self._segments()
        for segment in segments:
            for record in segment["records"]:
                record["constant"] = 1.0
        report = summarize_timeseries_presentations(segments, fields=["constant"])
        self.assertEqual(report["presentations"][0]["histogram_status"], "not_estimable")
        self.assertIn("constant", report["presentations"][0]["not_estimable_reason"])

    def test_cli_is_json_in_json_out(self):
        with tempfile.TemporaryDirectory() as temporary:
            request = Path(temporary) / "request.json"
            request.write_text(json.dumps({"segments": self._segments()}), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["summarize-timeseries", str(request)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue())["default_histogram_rule"], "scott")


if __name__ == "__main__":
    unittest.main()
