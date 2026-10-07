import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.diagnose_state_exports import run
from tests.test_state_coordinate_exports import _export_project, _source_report


class StateExportDiagnosticTests(unittest.TestCase):
    def test_read_only_profile_records_actual_policy_and_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study"
            study.mkdir()
            project = _export_project(study)
            original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in study.iterdir()}
            output = root / "profile"
            with patch("salsbury_md_analysis.state_coordinate_exports.load_cached_project_report", return_value=_source_report()):
                receipt = run(project, output, 10)
            self.assertEqual(receipt["technical_status"], "complete")
            self.assertEqual(receipt["effective_replica_processor_policies"], {"reject": 1})
            self.assertGreaterEqual(receipt["processed_frames_by_policy"]["reject"], 6)
            self.assertEqual(json.loads((output / "timing.json").read_text()), receipt)
            self.assertTrue((output / "profile.pstats").is_file())
            self.assertEqual(set(study.iterdir()), set(original))
            for p, digest in original.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)
            with self.assertRaises(FileExistsError):
                run(project, output, 10)
            with self.assertRaisesRegex(ValueError, "outside the study"):
                run(project, study / "not-allowed", 10)

    def test_deadline_saves_partial_timing_without_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study"
            study.mkdir()
            project = _export_project(study)
            def slow(*args, **kwargs):
                kwargs["phase_observer"]("synthetic_slow_phase")
                time.sleep(1)
            with patch("scripts.diagnose_state_exports.state_coordinate_exports_project", side_effect=slow):
                receipt = run(project, root / "profile", 0.05)
            self.assertEqual(receipt["technical_status"], "timeout")
            self.assertIn("synthetic_slow_phase", [p["phase"] for p in receipt["phases"]])
            self.assertFalse((study / "outputs").exists())

    def test_missing_saved_report_fails_without_compute_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study"
            study.mkdir()
            project = _export_project(study)
            with patch("salsbury_md_analysis.state_coordinate_exports.load_cached_project_report", return_value=None), patch("salsbury_md_analysis.state_coordinate_exports.clustering_kmeans_project", side_effect=AssertionError("no recompute")):
                receipt = run(project, root / "profile", 10)
            self.assertEqual(receipt["technical_status"], "error")
            self.assertIn("upstream recomputation is prohibited", receipt["error"])
            self.assertFalse((study / "outputs").exists())

    def test_invalid_saved_report_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study"
            study.mkdir()
            project = _export_project(study)
            invalid = study / "invalid-source.json"
            invalid.write_text(json.dumps({"technical_status": "failed"}))
            with patch.dict(os.environ, {"SALSBURY_MD_ANALYSIS_KMEANS_REPORT": str(invalid)}), patch("salsbury_md_analysis.state_coordinate_exports.clustering_kmeans_project", side_effect=AssertionError("no recompute")):
                receipt = run(project, root / "profile", 10)
            self.assertEqual(receipt["technical_status"], "error")
            self.assertEqual(receipt["error_type"], "StateCoordinateExportError")
            self.assertFalse((study / "outputs").exists())
