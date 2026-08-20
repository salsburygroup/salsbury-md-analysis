import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.regression import run_regression_case, run_regression_case_safe


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples" / "manifest_fixture" / "regression-case.json"


class RegressionTests(unittest.TestCase):
    def test_hash_pinned_candidate_passes_without_scientific_claim(self):
        report = run_regression_case(CASE)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["scientific_status"], "not evaluated")
        self.assertEqual(report["regression_approval_status"], "candidate")
        self.assertEqual(report["passed_check_count"], report["total_check_count"])

    def test_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = json.loads(CASE.read_text(encoding="utf-8"))
            case["project_manifest"] = str(
                ROOT / "examples" / "manifest_fixture" / "project.json"
            )
            case["expected_identity"]["project_manifest_sha256"] = "0" * 64
            path = root / "regression.json"
            path.write_text(json.dumps(case), encoding="utf-8")
            report = run_regression_case_safe(path)
        self.assertEqual(report["technical_status"], "failed")
        self.assertFalse(report["identity_checks"][0]["passed"])

    def test_cli_emits_regression_report(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["run-regression", str(CASE)])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["regression_id"], "synthetic-rmsd-rg-contract-v1")


if __name__ == "__main__":
    unittest.main()
