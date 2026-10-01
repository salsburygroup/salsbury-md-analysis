import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from test_tica import _write_project
from salsbury_md_analysis.accepted_artifacts import reports_complete, validate_complete_report
from salsbury_md_analysis.cli import main
from salsbury_md_analysis.context import compile_project_context_file
from salsbury_md_analysis.manifests import content_hash_session, sha256_file
from salsbury_md_analysis.report_reuse import adopt_report, receipt_path, review_report, scientific_contract
from salsbury_md_analysis.trajectory_features import trajectory_features_project
from salsbury_md_analysis.upstream_cache import load_cached_project_report


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class ReportReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old = self.root / "old"
        self.old.mkdir()
        self.project = _write_project(self.old)
        payload = json.loads(self.project.read_text())
        payload["definitions"]["trajectory_features"] = {
            "frame_stride": 1, "maximum_feature_values": 1000,
            "features": [{"feature_id": "distance", "kind": "pair_distance", "atom_indices": [3, 4]}],
        }
        write_json(self.project, payload)
        self.report = self.old / "results" / "features" / "report.json"
        self.report.parent.mkdir(parents=True)
        write_json(self.report, trajectory_features_project(self.project, hash_content=True))
        self.seal()
        self.new = self.root / "new"
        self.new.mkdir()
        write_json(self.new / "analysis-config.json", {"execution": {"submission_adapter": "local", "trajectory_mode": "continuous"}})
        for name in ("reference.pdb", "trajectory.xyz", "system.json"):
            shutil.copyfile(self.old / name, self.new / name)
        payload["project_id"] = "relocated"
        payload["analysis_output_root"] = "new-results"
        # An unrelated, science-changing PCA choice must not invalidate a direct distance.
        payload["definitions"]["common_pca"]["basis_weighting"] = "replica_equal"
        self.target = self.new / "project.json"
        write_json(self.target, payload)
        self.destination = self.new / "results" / "features" / "report.json"
        self.task = {"task_id": "features", "module_id": "trajectory_features", "project_filename": "project.json",
                     "completion_reports": ["results/features/report.json"]}
        write_json(self.new / "local-execution-plan.json", {"phases": [{"tasks": [self.task]}]})

    def seal(self):
        write_json(Path(str(self.report) + ".summary.json"), {
            "technical_status": "complete", "module_id": json.loads(self.report.read_text())["module_id"],
            "report_sha256": sha256_file(self.report),
            "report_path": str(self.report.resolve()), "report_size_bytes": self.report.stat().st_size,
            "finding_evidence": {"candidates": [], "report_path": str(self.report.resolve())},
        })

    def test_native_adoption_and_cached_use_preserve_original_bytes(self):
        before = self.report.read_bytes()
        review = review_report(self.report, self.target)
        self.assertTrue(review["eligible"], review)
        self.assertFalse(self.destination.exists())
        result = adopt_report(self.report, self.target, self.destination)
        self.assertEqual(self.destination.read_bytes(), before)
        self.assertEqual(self.report.read_bytes(), before)
        sidecar = json.loads(Path(str(self.destination) + ".summary.json").read_text())
        self.assertEqual(sidecar["report_path"], str(self.destination.resolve()))
        self.assertEqual(sidecar["finding_evidence"]["report_path"], str(self.destination.resolve()))
        self.assertTrue(reports_complete(self.new, self.task["completion_reports"], self.task), result)
        with patch.dict(os.environ, {"SALSBURY_MD_ANALYSIS_TRAJECTORY_FEATURES_REPORT": str(self.destination)}, clear=True):
            cached = load_cached_project_report("trajectory_features", self.target, hash_content=True, error_type=ValueError)
        self.assertEqual(cached["project_manifest_path"], str(self.target.resolve()))
        self.assertEqual(cached["project_manifest_sha256"], sha256_file(self.target))
        self.assertEqual(cached["reuse_provenance"]["source_report_sha256"], sha256_file(self.report))
        self.assertEqual(cached["segments"], json.loads(before)["segments"])
        from salsbury_md_analysis.finding_picker import prioritize_findings
        findings = prioritize_findings(self.new, write_outputs=False)
        self.assertEqual(findings["technical_status"], "complete")
        receipt = receipt_path(self.destination)
        saved = receipt.read_bytes()
        receipt.unlink()
        self.assertFalse(reports_complete(self.new, self.task["completion_reports"], self.task))
        with self.assertRaisesRegex(ValueError, "adoption receipt"):
            prioritize_findings(self.new, write_outputs=False)
        receipt.write_bytes(saved)
        with self.assertRaisesRegex(ValueError, "already exists"):
            adopt_report(self.report, self.target, self.destination)

    def test_cli_defaults_to_review_and_resolves_native_task(self):
        args = ["adopt-report", str(self.report), "--prepared", str(self.new), "--task", "features"]
        output = io.StringIO()
        with redirect_stdout(output): self.assertEqual(main(args), 0)
        self.assertTrue(json.loads(output.getvalue())["eligible"])
        self.assertFalse(self.destination.parent.exists())
        output = io.StringIO()
        with redirect_stdout(output): self.assertEqual(main(args + ["--apply"]), 0)
        self.assertTrue(reports_complete(self.new, self.task["completion_reports"]))

    def test_new_inputs_sampling_chemistry_timing_or_selection_rejected(self):
        baseline = json.loads(self.target.read_text())
        for kind in ("stride", "temperature", "selection", "periodic", "feature"):
            with self.subTest(kind=kind):
                p = copy.deepcopy(baseline)
                if kind == "stride": p["definitions"]["trajectory_features"]["frame_stride"] = 2
                if kind == "temperature": p["temperature_kelvin"] = 301
                if kind == "selection": p["selections"]["analysis"]["atom_names"] = ["CB"]
                if kind == "periodic": p["periodic_coordinate_policy"] = "already_unwrapped"
                if kind == "feature": p["definitions"]["trajectory_features"]["features"][0]["atom_indices"] = [0, 4]
                write_json(self.target, p)
                self.assertFalse(review_report(self.report, self.target)["eligible"])
        write_json(self.target, baseline)
        system = json.loads((self.new / "system.json").read_text())
        system["systems"][0]["replicas"][0]["segments"][0]["timing"]["first_frame_time"] = 1
        write_json(self.new / "system.json", system)
        self.assertEqual(review_report(self.report, self.target)["disposition"], "changed_calculation")
        shutil.copyfile(self.old / "system.json", self.new / "system.json")
        with (self.new / "trajectory.xyz").open("a") as stream: stream.write("\n")
        self.assertEqual(review_report(self.report, self.target)["disposition"], "changed_calculation")

    def test_mutation_after_adoption_invalidates_native_completion(self):
        adopt_report(self.report, self.target, self.destination)
        with (self.new / "trajectory.xyz").open("a") as stream: stream.write("\n")
        self.assertFalse(reports_complete(self.new, self.task["completion_reports"], self.task))
        with self.assertRaisesRegex(ValueError, "adoption"):
            validate_complete_report(self.destination, expected_project=self.target)

    def test_original_input_and_sidecar_tampering_rejected(self):
        with (self.old / "trajectory.xyz").open("a") as stream: stream.write("\n")
        result = review_report(self.report, self.target)
        self.assertFalse(result["eligible"])
        self.assertIn("signature", result["reason"])
        shutil.copyfile(self.new / "trajectory.xyz", self.old / "trajectory.xyz")
        Path(str(self.report) + ".summary.json").write_text("{}")
        self.assertFalse(review_report(self.report, self.target)["eligible"])

    def test_missing_hash_and_global_configuration_failure_remain_invalid(self):
        payload = json.loads(self.report.read_text())
        payload.pop("input_content_signature_sha256")
        write_json(self.report, payload)
        self.seal()
        self.assertIn("lacks a complete", review_report(self.report, self.target)["reason"])
        payload["input_content_signature_sha256"] = compile_project_context_file(self.project, hash_content=True)["input_content_signature_sha256"]
        source_project = json.loads(self.project.read_text())
        source_project["definitions"]["solvent_accessible_surface_area"] = {"surface_selection": "not_declared"}
        write_json(self.project, source_project)
        payload["project_manifest_sha256"] = sha256_file(self.project)
        write_json(self.report, payload)
        self.seal()
        self.assertIn("surface_selection", review_report(self.report, self.target)["reason"])

    def test_pca_and_scalar_dependency_contracts_include_upstream_choices(self):
        for path in (self.project, self.target):
            payload = json.loads(path.read_text())
            payload["definitions"]["clustering_kmeans"] = {"feature_source": "common_pca"}
            payload["definitions"]["scalar_feature_distributions"] = {"source": "trajectory_features"}
            write_json(path, payload)
        old, _ = scientific_contract("clustering_kmeans", self.project)
        new, _ = scientific_contract("clustering_kmeans", self.target)
        self.assertNotEqual(old, new)
        old, _ = scientific_contract("scalar_feature_distributions", self.project)
        payload = json.loads(self.target.read_text())
        payload["definitions"]["trajectory_features"]["frame_stride"] = 2
        write_json(self.target, payload)
        new, _ = scientific_contract("scalar_feature_distributions", self.target)
        self.assertNotEqual(old, new)

    def test_receipt_or_companion_tampering_rejected(self):
        companion = self.report.parent / "values.csv"
        companion.write_text("value\n1\n")
        payload = json.loads(self.report.read_text())
        payload["table"] = {"path": "values.csv", "sha256": sha256_file(companion)}
        write_json(self.report, payload)
        self.seal()
        adopt_report(self.report, self.target, self.destination)
        self.assertEqual((self.destination.parent / "values.csv").read_bytes(), companion.read_bytes())
        (self.destination.parent / "values.csv").write_text("value\n2\n")
        self.assertFalse(reports_complete(self.new, self.task["completion_reports"]))
        shutil.copyfile(companion, self.destination.parent / "values.csv")
        receipt = json.loads(receipt_path(self.destination).read_text())
        receipt["target_scientific_contract_sha256"] = "0" * 64
        write_json(receipt_path(self.destination), receipt)
        self.assertFalse(reports_complete(self.new, self.task["completion_reports"]))

    def test_static_temporal_mismatch_and_active_campaign_rejected(self):
        payload = json.loads(self.report.read_text())
        payload["temporal_output_policy"] = "disabled: discontinuous static ensemble"
        write_json(self.report, payload)
        self.seal()
        self.assertFalse(review_report(self.report, self.target)["eligible"])
        from salsbury_md_analysis.report_reuse import adopt_prepared_task
        with patch("salsbury_md_analysis.user_workflow.campaign_activity", return_value={
            "local_controller": "idle", "slurm_jobs": [{"job_id": "1"}], "scheduler_query": "complete",
        }):
            with self.assertRaisesRegex(ValueError, "active campaign"):
                adopt_prepared_task(self.report, self.new, "features", apply=True)

    def test_hash_session_rechecks_changed_files_and_does_not_persist(self):
        path = self.root / "hashed.txt"
        path.write_text("first")
        with content_hash_session():
            before = sha256_file(path)
            self.assertEqual(sha256_file(path), before)
            path.write_text("other")
            after = sha256_file(path)
            self.assertNotEqual(before, after)
        path.write_text("third")
        self.assertNotEqual(sha256_file(path), after)

    def test_derived_source_validation_is_not_its_own_acceptance(self):
        child = self.old / "upstream.json"
        write_json(child, {"technical_status": "failed", "module_id": "trajectory_features"})
        summary = Path(str(self.report) + ".summary.json")
        value = json.loads(summary.read_text())
        value["sidecar_schema"] = "salsbury-derived-report-sidecar-v1"
        value["source_report_records"] = [{"path": str(child), "sha256": sha256_file(child)}]
        write_json(summary, value)
        self.assertFalse(review_report(self.report, self.target)["eligible"])

    def test_changed_upstream_pca_is_classified_before_unsupported_schema(self):
        payload = json.loads(self.report.read_text())
        payload["module_id"] = "clustering_kmeans"
        for path in (self.project, self.target):
            value = json.loads(path.read_text())
            value["definitions"]["clustering_kmeans"] = {"feature_source": "common_pca"}
            write_json(path, value)
        payload["project_manifest_sha256"] = sha256_file(self.project)
        write_json(self.report, payload)
        self.seal()
        review = review_report(self.report, self.target)
        self.assertEqual(review["disposition"], "changed_calculation", review)


if __name__ == "__main__":
    unittest.main()
