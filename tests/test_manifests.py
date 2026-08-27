import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.manifests import (
    ManifestValidationError,
    inventory_system_inputs,
    load_json,
    validate_lock,
    validate_output,
    validate_project,
    validate_regression,
    validate_system,
)


class ManifestTests(unittest.TestCase):
    def _system(self):
        return {
            "systems": [
                {
                    "system_id": "reference",
                    "replicas": [
                        {
                            "replica_id": "r1",
                            "topology": "inputs/system.pdb",
                            "segments": [
                                {
                                    "segment_id": "s1",
                                    "trajectory": "inputs/trajectory.dcd",
                                    "timing": {
                                        "first_frame_time": 0,
                                        "frame_interval": 2,
                                        "unit": "ps",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        }

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"systems": [], "systems": []}', encoding="utf-8")
            with self.assertRaises(ManifestValidationError) as context:
                load_json(path)
            self.assertIn("duplicate JSON key", str(context.exception))

    def test_system_ids_and_first_segment_continuity_fail_closed(self):
        data = self._system()
        duplicate = json.loads(json.dumps(data["systems"][0]))
        data["systems"].append(duplicate)
        data["systems"][0]["replicas"][0]["segments"][0]["continuous_with_previous"] = True
        with self.assertRaises(ManifestValidationError) as context:
            validate_system(data)
        self.assertIn("duplicate system_id", str(context.exception))
        self.assertIn("first segment", str(context.exception))

    def test_segment_requires_exactly_one_physical_or_sample_axis(self):
        data = self._system()
        segment = data["systems"][0]["replicas"][0]["segments"][0]
        segment["sample_axis"] = {"first_sample_index": 0, "sample_interval": 1}
        with self.assertRaises(ManifestValidationError) as context:
            validate_system(data)
        self.assertIn("exactly one of timing or sample_axis", str(context.exception))

        segment.pop("timing")
        validate_system(data)
        segment["sample_axis"]["sample_interval"] = 0
        with self.assertRaises(ManifestValidationError) as context:
            validate_system(data)
        self.assertIn("sample_interval must be a positive integer", str(context.exception))

    def test_project_rejects_protected_output_overlap_and_unknown_module(self):
        data = {
            "project_id": "example",
            "analysis_profile": "standard_md_v1",
            "system_manifest": "system.json",
            "analysis_output_root": "/protected/project/results",
            "sampling_mode": "UNBIASED_MD",
            "protected_locations": ["/protected/project"],
            "requested_modules": ["replica_rmsd_rg", "invented_analysis"],
        }
        with self.assertRaises(ManifestValidationError) as context:
            validate_project(data)
        self.assertIn("overlaps protected location", str(context.exception))
        self.assertIn("invented_analysis", str(context.exception))

    def test_preprocessed_make_whole_requires_hash_matched_cache_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            system = root / "system-cache.json"
            system.write_text("{}\n", encoding="utf-8")
            cache_report = root / "coordinate-cache-report.json"
            cache_report.write_text(
                '{"technical_status":"complete"}\n', encoding="utf-8"
            )
            digest = hashlib.sha256(cache_report.read_bytes()).hexdigest()
            project = {
                "project_id": "cached-example",
                "analysis_profile": "standard_md_v1",
                "system_manifest": system.name,
                "analysis_output_root": "results",
                "sampling_mode": "UNBIASED_MD",
                "protected_locations": [],
                "periodic_coordinate_policy": "preprocessed_make_whole",
                "preprocessed_coordinate_source": {
                    "cache_report": cache_report.name,
                    "cache_report_sha256": digest,
                },
            }
            validate_project(
                project, source_path=root / "project.json", check_paths=True
            )
            project["preprocessed_coordinate_source"][
                "cache_report_sha256"
            ] = "0" * 64
            with self.assertRaisesRegex(
                ManifestValidationError, "does not match cache_report"
            ):
                validate_project(
                    project, source_path=root / "project.json", check_paths=True
                )

    def test_path_check_and_inventory_hash_relative_to_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            topology = inputs / "system.pdb"
            trajectory = inputs / "trajectory.dcd"
            topology.write_bytes(b"MODEL\nENDMDL\n")
            trajectory.write_bytes(b"synthetic-test-trajectory")
            manifest = root / "system.json"
            data = self._system()
            manifest.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
            validate_system(data, source_path=manifest, check_paths=True)
            first = inventory_system_inputs(data, manifest, hash_content=True)
            second = inventory_system_inputs(data, manifest, hash_content=True)
            self.assertEqual(first, second)
            self.assertEqual(first["entry_count"], 2)
            by_role = {entry["role"]: entry for entry in first["entries"]}
            self.assertEqual(
                by_role["trajectory"]["sha256"], hashlib.sha256(trajectory.read_bytes()).hexdigest()
            )
            self.assertEqual(first["scientific_status"], "not evaluated")

    def test_force_field_parameter_files_are_validated_and_hashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "system.pdb").write_bytes(b"MODEL\nENDMDL\n")
            (inputs / "trajectory.dcd").write_bytes(b"trajectory")
            parameter = inputs / "protein.prm"
            parameter.write_bytes(b"NONBONDED\nN 0 -0.2 1.8\nEND\n")
            expected_digest = hashlib.sha256(parameter.read_bytes()).hexdigest()
            data = self._system()
            data["systems"][0]["replicas"][0]["force_field_parameters"] = {
                "format": "charmm_parameter_files_v1",
                "files": ["inputs/protein.prm"],
            }
            manifest = root / "system.json"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            validate_system(data, source_path=manifest, check_paths=True)
            inventory = inventory_system_inputs(
                data, manifest, hash_content=True
            )
        row = next(
            entry for entry in inventory["entries"]
            if entry["role"].startswith("force_field_parameter:")
        )
        self.assertEqual(
            row["sha256"], expected_digest
        )

    def test_missing_input_fails_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "system.json"
            data = self._system()
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ManifestValidationError) as context:
                inventory_system_inputs(data, manifest)
            self.assertIn("does not exist", str(context.exception))

    def test_lock_requires_real_hash_shapes(self):
        data = {
            "project_id": "example",
            "suite_commit": "not-a-commit",
            "project_commit": "a" * 40,
            "profile_id": "standard_md_v1",
            "environment_identity": "environment.yml sha256:example",
            "input_manifest_sha256": "b" * 64,
            "source_manifest_sha256": "short",
            "owner": "owner",
            "technical_status": "complete",
            "scientific_status": "not evaluated",
        }
        with self.assertRaises(ManifestValidationError) as context:
            validate_lock(data)
        self.assertIn("suite_commit", str(context.exception))
        self.assertIn("source_manifest_sha256", str(context.exception))

    def test_lock_records_optional_frame_budget_sensitivity(self):
        data = {
            "project_id": "example",
            "suite_commit": "a" * 40,
            "project_commit": "b" * 40,
            "profile_id": "standard_md_v1",
            "environment_identity": "environment.yml sha256:example",
            "input_manifest_sha256": "c" * 64,
            "source_manifest_sha256": "d" * 64,
            "owner": "owner",
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "frame_budget_sensitivity": {
                "policy": "recommend",
                "status": "skipped",
                "rationale": "All-frame estimator execution was used.",
            },
        }
        validate_lock(data)
        data["frame_budget_sensitivity"].pop("rationale")
        with self.assertRaisesRegex(ManifestValidationError, "rationale is required"):
            validate_lock(data)

    def test_publication_lock_can_turn_frame_budget_sensitivity_off(self):
        data = {
            "project_id": "publication-example",
            "suite_commit": "a" * 40,
            "project_commit": "b" * 40,
            "profile_id": "standard_md_v1",
            "environment_identity": "environment.yml sha256:example",
            "input_manifest_sha256": "c" * 64,
            "source_manifest_sha256": "d" * 64,
            "owner": "publication owner",
            "technical_status": "complete",
            "scientific_status": "reviewed",
            "frame_budget_sensitivity": {
                "policy": "off",
                "status": "not_applicable",
                "rationale": (
                    "The project owner elected not to run a B-versus-2B "
                    "frame-budget comparison."
                ),
            },
        }
        validate_lock(data)
        data["frame_budget_sensitivity"] = {
            "policy": "require",
            "status": "skipped",
            "rationale": "The optional policy was promoted to a project gate.",
        }
        with self.assertRaisesRegex(
            ManifestValidationError, "required frame_budget_sensitivity cannot be skipped"
        ):
            validate_lock(data)

    def test_completed_optional_replica_diagnostics_require_evidence(self):
        data = {
            "project_id": "example",
            "suite_commit": "a" * 40,
            "project_commit": "b" * 40,
            "profile_id": "standard_md_v1",
            "environment_identity": "environment.yml sha256:example",
            "input_manifest_sha256": "c" * 64,
            "source_manifest_sha256": "d" * 64,
            "owner": "owner",
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "replica_diagnostics": {
                "policy": "optional",
                "status": "completed",
                "additional_replicas_may_be_useful": True,
                "evidence_report_sha256": ["e" * 64],
            },
        }
        validate_lock(data)
        data["replica_diagnostics"].pop("evidence_report_sha256")
        with self.assertRaisesRegex(
            ManifestValidationError, "completed replica_diagnostics requires evidence"
        ):
            validate_lock(data)

    def test_output_rejects_unknown_and_duplicate_module(self):
        data = {
            "run_id": "run-1",
            "suite_commit": "working-tree",
            "profile_id": "standard_md_v1",
            "modules": [
                {"module_id": "invented_analysis", "status": "complete", "outputs": []},
                {"module_id": "invented_analysis", "status": "complete", "outputs": []},
            ],
            "technical_status": "complete",
            "scientific_status": "not evaluated",
        }
        with self.assertRaises(ManifestValidationError) as context:
            validate_output(data)
        self.assertIn("unknown", str(context.exception))
        self.assertIn("duplicate module_id", str(context.exception))

    def test_output_path_check_verifies_sha256(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.csv"
            output.write_text("value\n1\n", encoding="utf-8")
            data = {
                "run_id": "run-1",
                "suite_commit": "working-tree",
                "profile_id": "standard_md_v1",
                "modules": [
                    {
                        "module_id": "provenance_manifest",
                        "status": "complete",
                        "outputs": [{"path": "result.csv", "sha256": "0" * 64}],
                    }
                ],
                "technical_status": "complete",
                "scientific_status": "not evaluated",
            }
            with self.assertRaises(ManifestValidationError) as context:
                validate_output(data, source_path=root / "outputs.json", check_paths=True)
            self.assertIn("sha256 mismatch", str(context.exception))

    def test_approved_regression_requires_reviewer_and_decision(self):
        data = {
            "regression_id": "case",
            "module_id": "replica_rmsd_rg",
            "project_manifest": "project.json",
            "expected_identity": {
                "project_manifest_sha256": "a" * 64,
                "system_manifest_sha256": "b" * 64,
                "input_content_signature_sha256": "c" * 64,
            },
            "assertions": [
                {"path": ["technical_status"], "operator": "equal", "expected": "complete"}
            ],
            "approval": {
                "status": "approved",
                "owner": "owner",
                "reviewers": [],
                "decision_utc": None,
                "notes": [],
            },
        }
        with self.assertRaises(ManifestValidationError) as context:
            validate_regression(data)
        self.assertIn("reviewers", str(context.exception))
        self.assertIn("decision_utc", str(context.exception))


if __name__ == "__main__":
    unittest.main()
