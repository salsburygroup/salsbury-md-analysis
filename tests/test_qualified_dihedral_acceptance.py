"""Synthetic trust-boundary and interrupted-publication tests; no coordinates."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from salsbury_md_analysis.manifests import sha256_file, stable_json_sha256
from salsbury_md_analysis.qualified_dihedral_acceptance import (
    ANCHORS, BASIS, POLICY, PROOF, MODULE, SCHEMA, _entry,
    publish_qualified_dihedral, registered_validation, qualified_summary,
)
from salsbury_md_analysis.accepted_artifacts import validate_complete_report, reports_complete
from salsbury_md_analysis.report_reuse import runtime_report, validate_adoption, review_report


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True) + "\n")
    return path


def record(path):
    return {"path": str(path), "sha256": sha256_file(path)}


class QualifiedDihedralTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "campaign"
        self.source = self.base / "accepted-output"
        self.project = put(self.root / "project.json", {"system_manifest": "system.json", "definitions": {MODULE: {"frame_stride": 2}}})
        self.system = put(self.root / "system.json", {"systems": [{"system_id": "sample", "replicas": [{"replica_id": "replica", "segments": [{"trajectory": str(self.base / "never-read.dcd")}]}]}]})
        self.config = put(self.root / "analysis-config.json", {"execution": {"submission_adapter": "local"}})
        self.task = {"task_id": "dihedral-task", "module_id": MODULE, "project_filename": "project.json", "completion_reports": ["results/dihedrals/report.json"]}
        self.plan = put(self.root / "local-execution-plan.json", {"phases": [{"tasks": [self.task]}]})
        self.failed = put(self.root / self.task["completion_reports"][0], {"technical_status": "failed", "module_id": MODULE})
        self.failed_bytes = self.failed.read_bytes()
        self.settings = {"frame_stride": 2, "histogram_bins": 2, "angle_types": ["phi"]}
        segments = [{"system_id": "sample", "replica_id": "replica", "segment_id": str(i), "source_frame_count": 4,
                     "decoded_frame_count": 4, "evaluated_frame_count": 2, "torsion_definition_count": 1} for i in range(2)]
        series = {"system_id": "sample", "replica_id": "replica", "chain_id": "A", "residue_number": 1, "insertion_code": "", "angle_type": "phi", "count": 4}
        self.population = {"segments": [{**s, "selected_frames": [0, 2]} for s in segments],
            "series": [series], "selected_frame_count": 4, "series_count": 1, "observation_count": 4, "histogram_bins": 2}
        self.execution = {"historical_context_is_not_current_raw_validation": True,
            "historical_producer_frames": 8, "current_cache_frames_decoded": 8, "current_selected_frames": 4,
            "assembly_coordinate_frames_decoded": 0, "native_task_adopted": False}
        self.population_file = put(self.source / "population.json", {"schema": "salsbury-qualified-dihedral-population-v1", "population": self.population,
            "series_definitions": [{**series, "atom_indices": [0, 1, 2, 3]}], "degeneracy_masks": [False] * 4,
            "group_bounds": [[0, 4]]})
        self.extra = put(self.source / "nested.json", {"retained": "exact reviewed scalar evidence"})
        self.supplement = put(self.source / "supplement.json", {"nested_path": str(self.extra), "nested_sha256": sha256_file(self.extra)})
        self.report = put(self.source / "report.json", {"technical_status": "complete", "scientific_status": "not evaluated", "module_id": MODULE,
            "project_manifest_path": str(self.project), "project_manifest_sha256": sha256_file(self.project),
            "system_manifest_path": str(self.system), "system_manifest_sha256": sha256_file(self.system),
            "input_content_signature_sha256": "a" * 64, "settings": self.settings, "error_count": 0, "issues": [],
            "segment_reports": segments, "series_count": 1, "observation_count": 4,
            "circular_summaries": [{**series, "mean_angle_degrees": None, "mean_resultant_length": 0, "circular_variance": 1,
                "histogram": [{"lower_degrees": -180, "upper_degrees": 0, "count": 2, "fraction": 0.5}, {"lower_degrees": 0, "upper_degrees": 180, "count": 2, "fraction": 0.5}]}],
            "derived_execution": self.execution, "execution_resources": {"stage": "scalar assembly", "coordinate_frames_decoded": 0},
            "numerical_supplement_path": str(self.supplement), "numerical_supplement_sha256": sha256_file(self.supplement)})
        self.summary = put(Path(str(self.report) + ".summary.json"), {"module_id": MODULE, "technical_status": "complete",
            "report_path": str(self.report), "report_sha256": sha256_file(self.report), "report_size_bytes": self.report.stat().st_size,
            "resource_evidence": {"selected_source_physical_frames": 4, "symmetry_expanded_observations": 4,
                                  "execution_resources": {"wall_seconds": 0.1, "total_cpu_seconds": 0.1}},
            "finding_evidence": {"candidates": [{"presentation_eligible": True, "claim": "unendorsed helper candidate"}]}})
        anchors = {name: record(put(self.source / (name + ".json"), {"role": name, "reviewed_output": record(self.report)})) for name in ANCHORS}
        self.historical = [{"path": str(self.base / "never-read.dcd"), "sha256": "b" * 64, "role": "raw_input"},
                           {"path": str(self.base / "historical-cache.dcd"), "sha256": "c" * 64, "role": "qualified_cache"}]
        self.proof = {"schema": PROOF, "module_id": MODULE, "task_id": self.task["task_id"],
            "acceptance_anchors": anchors, "assumptions": ["accepted historical lineage; no fresh raw validation"],
            "historical_inputs": self.historical, "population": self.population, "derived_execution": self.execution,
            "report": record(self.report), "summary": record(self.summary), "numerical_supplement": record(self.supplement), "population_evidence": record(self.population_file)}
        self.proof_path = self.base / "approved-proof.json"
        self.policy_path = self.base / "approved-policy.json"
        self.seal()

    def seal(self):
        files = list(self.source.glob("*.json")) + [self.project, self.system]
        self.proof["artifacts"] = [{**record(p), "kind": "json", "size_bytes": p.stat().st_size} for p in files]
        put(self.proof_path, self.proof)
        self.policy = {"schema": POLICY, "validation_basis": BASIS, "module_id": MODULE,
            "task_id": self.task["task_id"], "task_sha256": stable_json_sha256(self.task), "plan_sha256": sha256_file(self.plan),
            "analysis_config_sha256": sha256_file(self.config), "project_manifest": record(self.project), "system_manifest": record(self.system),
            "qualification_receipt_sha256": sha256_file(self.proof_path), "acceptance_anchors": self.proof["acceptance_anchors"],
            "historical_input_signature_sha256": "a" * 64, "settings_sha256": stable_json_sha256(self.settings),
            "historical_inputs": self.historical, "assumptions": self.proof["assumptions"], "maximum_retained_artifact_bytes": 1000000}
        put(self.policy_path, self.policy)
        self.policy_sha = sha256_file(self.policy_path)

    def publish(self, apply=False):
        return publish_qualified_dihedral(self.report, self.root, self.task["task_id"], qualification_receipt=self.proof_path,
            policy_path=self.policy_path, policy_sha256=self.policy_sha, apply=apply)

    def test_native_completion_runtime_and_original_failures_preserved(self):
        with patch("salsbury_md_analysis.context.compile_project_context_file", side_effect=AssertionError("raw scan forbidden")), patch("salsbury_md_analysis.report_reuse.compile_project_context_file", side_effect=AssertionError("raw scan forbidden")):
            self.assertFalse(self.publish()["applied"])
            self.assertFalse((self.root / ".qualified-reports").exists())
            result = self.publish(True)
            path = Path(result["report_path"])
            self.assertEqual(path.read_bytes(), self.report.read_bytes())
            self.assertEqual(self.failed.read_bytes(), self.failed_bytes)
            self.assertTrue(reports_complete(self.root, self.task["completion_reports"], self.task))
            self.assertEqual(validate_complete_report(self.failed, expected_project=self.project), json.loads(self.report.read_text()))
            self.assertEqual(validate_adoption(path, self.project)["validation_basis"], BASIS)
            runtime = runtime_report(path, self.project)
            self.assertFalse(runtime["qualified_validation"]["current_raw_validation"])
            self.assertEqual(runtime["input_content_signature_sha256"], "a" * 64)
            self.assertEqual(qualified_summary(path)["finding_evidence"]["candidates"], [])
            self.assertTrue(json.loads(self.summary.read_text())["finding_evidence"]["candidates"])
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.publish(True)

    def test_cli_review_and_apply(self):
        from salsbury_md_analysis.cli import main
        args = ["adopt-qualified-dihedral", str(self.report), "--prepared", str(self.root), "--task", self.task["task_id"],
                "--qualification-receipt", str(self.proof_path), "--policy", str(self.policy_path), "--policy-sha256", self.policy_sha]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args + ["--apply"]), 0)

    def test_raw_validation_never_silently_accepts_derived_or_v1_reuse(self):
        with self.assertRaisesRegex(ValueError, "registered pinned"):
            validate_complete_report(self.report, expected_project=self.project)
        self.assertFalse(review_report(self.report, self.project)["eligible"])

    def test_every_retained_artifact_policy_and_proof_mutation_rejected(self):
        for p in [Path(r["path"]) for r in self.proof["artifacts"]] + [self.policy_path, self.proof_path, self.plan, self.config]:
            with self.subTest(path=p.name):
                saved = p.read_bytes()
                p.write_bytes(saved + b" ")
                with self.assertRaises(ValueError):
                    self.publish()
                p.write_bytes(saved)

    def test_repeated_validation_detects_source_and_published_mutations(self):
        path = Path(self.publish(True)["report_path"])
        candidates = [path, Path(str(path) + ".summary.json"), Path(str(path) + ".adoption.json"), path.parent / "proof.json",
                      path.parent / "policy.json", self.extra, self.failed]
        for p in candidates:
            with self.subTest(path=p.name):
                data = p.read_bytes()
                p.write_bytes(data + b" ")
                self.assertFalse(reports_complete(self.root, self.task["completion_reports"], self.task))
                with self.assertRaises(ValueError):
                    runtime_report(path, self.project)
                p.write_bytes(data)
        self.assertTrue(reports_complete(self.root, self.task["completion_reports"], self.task))

    def test_missing_receipt_and_unknown_schema_do_not_fall_back_to_raw(self):
        path = Path(self.publish(True)["report_path"])
        receipt = Path(str(path) + ".adoption.json")
        data = receipt.read_bytes()
        receipt.unlink()
        with patch("salsbury_md_analysis.context.compile_project_context_file", side_effect=AssertionError("raw scan forbidden")):
            self.assertFalse(reports_complete(self.root, self.task["completion_reports"], self.task))
        receipt.write_bytes(data.replace(SCHEMA.encode(), b"unknown"))
        self.assertFalse(reports_complete(self.root, self.task["completion_reports"], self.task))

    def test_population_order_counts_frames_and_recursive_closure_fail(self):
        baseline = copy.deepcopy(self.proof)
        for defect in ("order", "duplicate", "extra", "missing", "count", "frame", "anchors", "nested", "schema", "task"):
            with self.subTest(defect=defect):
                self.proof = copy.deepcopy(baseline)
                if defect == "order": self.proof["population"]["segments"].reverse()
                elif defect == "duplicate": self.proof["population"]["segments"][0]["selected_frames"] = [0, 0]
                elif defect == "extra": self.proof["population"]["series"].append(copy.deepcopy(self.proof["population"]["series"][0]))
                elif defect == "missing": self.proof["population"]["segments"].pop()
                elif defect == "count": self.proof["population"]["series"][0]["count"] = 3
                elif defect == "frame": self.proof["population"]["segments"][0]["selected_frames"] = [0, 99]
                elif defect == "anchors": self.proof["acceptance_anchors"].pop("result_review")
                elif defect == "schema": self.proof["schema"] = "unknown"
                elif defect == "task": self.proof["task_id"] = "other"
                self.seal()
                if defect == "nested":
                    self.proof["artifacts"] = [r for r in self.proof["artifacts"] if r["path"] != str(self.extra)]
                    put(self.proof_path, self.proof)
                    self.policy["qualification_receipt_sha256"] = sha256_file(self.proof_path)
                    put(self.policy_path, self.policy)
                    self.policy_sha = sha256_file(self.policy_path)
                with self.assertRaises(ValueError): self.publish()

    def test_busy_active_unknown_and_unresolved_jobs_block_publication(self):
        from salsbury_md_analysis.user_workflow import campaign_lock
        with campaign_lock(self.root):
            with self.assertRaisesRegex(ValueError, "already owns"):
                self.publish(True)
        for activity in ({"slurm_jobs": [{"id": 1}], "scheduler_query": "complete"}, {"slurm_jobs": [], "scheduler_query": "failed"}):
            with patch("salsbury_md_analysis.user_workflow.campaign_activity", return_value=activity):
                with self.assertRaisesRegex(ValueError, "active campaign"):
                    self.publish(True)
        put(self.root / "submission-intents/unknown.json", {})
        with self.assertRaisesRegex(ValueError, "unresolved"):
            self.publish(True)
        self.assertEqual(self.failed.read_bytes(), self.failed_bytes)

    def test_interruption_before_commit_preserves_failed_and_staged_evidence(self):
        with patch("salsbury_md_analysis.qualified_dihedral_acceptance.os.link", side_effect=OSError("interruption")):
            with self.assertRaises(OSError): self.publish(True)
        self.assertFalse(reports_complete(self.root, self.task["completion_reports"], self.task))
        self.assertTrue(list((self.root / ".qualified-reports/.versions").glob("*/report.json")))
        self.assertEqual(self.failed.read_bytes(), self.failed_bytes)
        self.assertTrue(self.publish(True)["applied"])

    def test_interruption_after_atomic_link_leaves_one_valid_result(self):
        link = os.link
        def interrupted(a, b):
            link(a, b)
            raise OSError("after commit")
        with patch("salsbury_md_analysis.qualified_dihedral_acceptance.os.link", side_effect=interrupted):
            with self.assertRaises(OSError): self.publish(True)
        self.assertTrue(reports_complete(self.root, self.task["completion_reports"], self.task))
        self.assertEqual(len(list((self.root / ".qualified-reports").glob("*.json"))), 1)

    def test_atomic_commit_never_overwrites_a_raced_destination(self):
        link = os.link
        def competing(a, b):
            Path(b).write_text("preserve competing evidence")
            return link(a, b)
        with patch("salsbury_md_analysis.qualified_dihedral_acceptance.os.link", side_effect=competing):
            with self.assertRaises(FileExistsError): self.publish(True)
        self.assertEqual(_entry(self.root, self.task["task_id"]).read_text(), "preserve competing evidence")
        self.assertEqual(self.failed.read_bytes(), self.failed_bytes)

    def test_resource_and_finding_consumers_use_registered_result(self):
        from salsbury_md_analysis.execution_resources import summarize_execution_resources
        from salsbury_md_analysis.finding_picker import prioritize_findings
        result = self.publish(True)
        summary = summarize_execution_resources(self.root)
        data = json.loads(Path(summary["json_path"]).read_text())
        self.assertEqual(len(data["rows"]), 1)
        self.assertEqual(data["rows"][0]["report_path"], result["report_path"])
        self.assertEqual(data["rows"][0]["validation_basis"], BASIS)
        findings = prioritize_findings(self.root, write_outputs=False)
        self.assertEqual(findings["candidate_count"], 0)

    def test_population_evidence_semantics_reject_even_with_new_pins(self):
        baseline = json.loads(self.population_file.read_text())
        for defect in ("mask", "extra_mask", "bounds", "atoms", "identity", "population"):
            with self.subTest(defect=defect):
                data = copy.deepcopy(baseline)
                if defect == "mask": data["degeneracy_masks"][0] = True
                elif defect == "extra_mask": data["degeneracy_masks"].append(False)
                elif defect == "bounds": data["group_bounds"][0] = [1, 4]
                elif defect == "atoms": data["series_definitions"][0]["atom_indices"] = [0, 1, 2, 2]
                elif defect == "identity": data["series_definitions"][0]["replica_id"] = "other"
                else: data["population"]["selected_frame_count"] += 1
                put(self.population_file, data)
                self.proof["population_evidence"] = record(self.population_file)
                self.seal()
                with self.assertRaises(ValueError): self.publish()

    def test_activity_query_race_changes_rejected_before_commit(self):
        from salsbury_md_analysis.qualified_dihedral_acceptance import _idle
        for target in (self.plan, self.config, self.extra, self.failed):
            with self.subTest(target=target.name):
                before = target.read_bytes()
                calls = []
                def change(root):
                    _idle(root)
                    calls.append(root)
                    if len(calls) == 2:
                        target.write_bytes(before + b" ")
                with patch("salsbury_md_analysis.qualified_dihedral_acceptance._idle", side_effect=change):
                    with self.assertRaises(ValueError): self.publish(True)
                self.assertFalse(_entry(self.root, self.task["task_id"]).exists())
                target.write_bytes(before)

    def test_picker_rejects_missing_summary_receipt_and_withholds_cross_reports(self):
        from salsbury_md_analysis.finding_picker import prioritize_findings, FindingPickerError
        summary = json.loads(self.summary.read_text())
        summary["finding_evidence"]["cross_report_summary"] = {"module_id": MODULE, "arbitrary": "unendorsed"}
        summary["finding_evidence"]["clustering_models"] = [{"arbitrary": "unendorsed"}]
        put(self.summary, summary)
        self.proof["summary"] = record(self.summary)
        self.seal()
        path = Path(self.publish(True)["report_path"])
        with patch("salsbury_md_analysis.finding_picker._cross_report_candidates", return_value=[]) as cross:
            self.assertEqual(prioritize_findings(self.root, write_outputs=False)["candidate_count"], 0)
            self.assertEqual(cross.call_args.args[0], [])
        for suffix in (".summary.json", ".adoption.json"):
            p = Path(str(path) + suffix)
            data = p.read_bytes()
            p.unlink()
            with self.assertRaises(FindingPickerError):
                prioritize_findings(self.root, write_outputs=False)
            p.write_bytes(data)

    def test_chained_adoption_rejected(self):
        put(Path(str(self.report) + ".adoption.json"), {"eligible": True})
        with self.assertRaisesRegex(ValueError, "chained"):
            self.publish()


if __name__ == "__main__":
    unittest.main()
