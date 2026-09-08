import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.finding_report import (
    validate_scientific_context, write_finding_reader_reports,
)
from salsbury_md_analysis.finding_picker import prioritize_findings
from salsbury_md_analysis.analysis_config import load_analysis_config, AnalysisConfigError


class FindingReaderReportTests(unittest.TestCase):
    def fixture(self, root):
        directory = root / "presentation-artifacts"
        directory.mkdir()
        artifacts = []
        for i, (kind, suffix) in enumerate((("figure", ".svg"), ("table", ".csv"),
                                           ("structure", ".pdb"), ("figure", ".svg"))):
            path = directory / f"evidence {i}{suffix}"
            path.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text x="4" y="20">Fraction (%)</text></svg>'
                            if kind == "figure" else "system,value\na,1\nb,2\n")
            artifacts.append({"artifact_id": f"a{i}", "artifact_type": kind,
                              "title": f"State populations {i}", "relative_path": path.name,
                              "analysis_class": "free_energy_surfaces", "source_report_paths": [],
                              "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (directory / "presentation-manifest.json").write_text(json.dumps({"artifacts": artifacts}))
        candidates = [{"finding_id": str(i), "statement": f"State fraction differs observationally by +{i / 100:.2f}.",
                       "system_ids": ["a", "b"], "presentation_artifacts": artifacts[:3],
                       "presentation_artifact_match": "exact_target"} for i in range(61)]
        output = {"all_candidates": candidates, "headline_findings": candidates[:10],
                  "secondary_findings": candidates[10:50]}
        (root / "prioritized_findings.json").write_text(json.dumps(output))
        (root / "prioritized_findings.csv").write_text("finding_id\n" + "\n".join(str(i) for i in range(61)))
        for name in ("prioritized_findings_qc.md", "prioritized_findings_details.md"):
            (root / name).write_text("# Supporting evidence\n")
        return output

    def test_summary_changes_no_data_or_ranking_and_indexes_every_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = self.fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            original = copy.deepcopy(output)
            result = write_finding_reader_reports(root, output, {
                "title": "Comparison of two ensembles", "question": "How do the state populations differ?",
                "systems": {"a": {"label": "Control", "description": "Unmodified polymer"},
                            "b": {"label": "Variant", "description": "Modified polymer"}},
                "population": "All retained frames; counts vary by method.",
                "weighting": "Frame-pooled within each system.",
                "figure_captions": {"a0": "State populations within each system. Shared state definitions; bars show percentages."},
            })
            self.assertEqual(output, original)
            for path, data in before.items():
                self.assertEqual(path.read_bytes(), data)
            text = Path(result["html_path"]).read_text()
            self.assertLess(text.index("Systems and comparisons"), text.index('<section><h2>Key findings'))
            self.assertEqual(text.count('<figure id='), 1)
            self.assertIn('presentation-artifacts/evidence%200.svg', text)
            self.assertIn("bars show percentages", text)
            self.assertNotIn("observationally", text)
            self.assertNotIn("not evaluated", text)
            with (root / "finding_evidence.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)  # includes artifact absent from every headline
            checks = json.loads(Path(result["reader_report_checks_path"]).read_text())
            self.assertEqual(checks["candidate_count"], 61)
            self.assertEqual(checks["verified_artifact_count"], 4)
            self.assertFalse(checks["source_data_modified"])
            self.assertIn('<caption>Table 1.', text)
            self.assertIn('<td>Control</td>', text)

    def test_broad_matches_do_not_become_claim_supporting_panels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = self.fixture(root)
            for candidate in output["all_candidates"]:
                candidate["presentation_artifact_match"] = "source_report"
            write_finding_reader_reports(root, output)
            self.assertNotIn('<figure id=', (root / "prioritized_findings.html").read_text())
            self.assertIn('evidence%200.svg', (root / "finding_evidence.html").read_text())

    def test_multirow_comparisons_keep_state_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = self.fixture(root)
            path = root / "presentation-artifacts/evidence 1.csv"
            path.write_text("finding_number,finding_label,left_system_id,right_system_id,effect_value,finding\n"
                            "1,Finding 1,a,b,0.2,State 1 population difference\n"
                            "2,Finding 2,a,b,-0.2,State 2 population difference\n")
            mp = root / "presentation-artifacts/presentation-manifest.json"
            manifest = json.loads(mp.read_text())
            manifest["artifacts"][1]["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            mp.write_text(json.dumps(manifest))
            write_finding_reader_reports(root, output)
            text = (root / "prioritized_findings.html").read_text()
            self.assertIn("State 1 population difference", text)
            self.assertIn("State 2 population difference", text)

    def test_corrupt_missing_or_external_assets_are_not_linked_as_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = self.fixture(root)
            (root / "presentation-artifacts/evidence 0.svg").write_text("changed")
            (root / "presentation-artifacts/evidence 1.csv").unlink()
            (root / "presentation-artifacts/evidence 2.pdb").unlink()
            (root / "presentation-artifacts/evidence 2.pdb").symlink_to(root / "prioritized_findings.json")
            result = write_finding_reader_reports(root, output)
            checks = json.loads(Path(result["reader_report_checks_path"]).read_text())
            self.assertEqual(checks["verified_artifact_count"], 1)
            text = Path(result["html_path"]).read_text()
            self.assertNotIn('href="presentation-artifacts/evidence%200.svg"', text)
            self.assertTrue(checks["review_items"])

    def test_context_validation_and_html_escaping(self):
        for invalid in ({"other": 1}, {"systems": []}, {"weighting": 1},
                        {"systems": {"a": {"unknown": "x"}}}, {"methods": [3]}):
            with self.assertRaises(ValueError):
                validate_scientific_context(invalid)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = self.fixture(root)
            write_finding_reader_reports(root, output, {"question": '<script>alert(1)</script>'})
            text = (root / "prioritized_findings.html").read_text()
            self.assertNotIn('<script>alert', text)
            self.assertIn('&lt;script&gt;', text)

    def test_analysis_configuration_accepts_context_and_rejects_mistyped_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({"config_schema": "salsbury-analysis-config-v1", "reporting": {"scientific_context": {"question": "Compare conditions"}}}))
            config = load_analysis_config(path, [], [])
            self.assertEqual(config["reporting"]["scientific_context"]["question"], "Compare conditions")
            path.write_text(json.dumps({"config_schema": "salsbury-analysis-config-v1", "reporting": {"scientific_context": {"weigthing": "pooled"}}}))
            with self.assertRaises(AnalysisConfigError):
                load_analysis_config(path, [], [])

    def test_picker_writes_both_reader_and_complete_supporting_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "analysis-config.json").write_text(json.dumps({
                "reporting": {"scientific_context": {"title": "Custom ensemble report"}}}))
            report = prioritize_findings(root)
            self.assertTrue(Path(report["html_path"]).is_file())
            self.assertTrue(Path(report["details_path"]).is_file())
            self.assertIn("Module accounting", Path(report["details_path"]).read_text())
            self.assertNotIn("Module accounting", Path(report["markdown_path"]).read_text())
            self.assertIn("Custom ensemble report", Path(report["markdown_path"]).read_text())
            self.assertEqual(report["all_candidates"], [])


if __name__ == "__main__":
    unittest.main()
