import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.context import compile_project_context, compile_project_context_file
from salsbury_md_analysis.manifests import ManifestValidationError, validate_project


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "manifest_fixture"


class ProjectContextTests(unittest.TestCase):
    def test_fixture_compiles_deterministically_with_explicit_identity(self):
        project = EXAMPLE / "project.json"
        first = compile_project_context_file(project, hash_content=True)
        second = compile_project_context_file(project, hash_content=True)
        self.assertEqual(
            first["contract_signature_sha256"], second["contract_signature_sha256"]
        )
        self.assertEqual(
            first["input_content_signature_sha256"],
            second["input_content_signature_sha256"],
        )
        self.assertEqual(first["contract"]["reference_system"], "synthetic-one-atom-system")
        replica = first["contract"]["systems"][0]["replicas"][0]
        self.assertEqual(replica["replica_id"], "replica-1")
        self.assertEqual(replica["segments"][0]["segment_id"], "segment-1")
        self.assertEqual(first["warning_count"], 1)
        self.assertEqual(first["scientific_status"], "not evaluated")

    def test_context_normalizes_exact_residue_key_selections(self):
        project_path = EXAMPLE / "project.json"
        project = json.loads(project_path.read_text(encoding="utf-8"))
        project["selections"]["chemical_interface"] = {
            "residue_keys": [
                {"chain_id": "B", "residue_number": 2, "insertion_code": ""},
                {"chain_id": "A", "residue_number": 1, "insertion_code": "A"},
            ],
            "heavy_only": True,
        }
        report = compile_project_context(project, project_path)
        self.assertEqual(
            report["contract"]["selections"]["chemical_interface"],
            {
                "residue_keys": [
                    {"chain_id": "A", "residue_number": 1, "insertion_code": "A"},
                    {"chain_id": "B", "residue_number": 2, "insertion_code": ""},
                ],
                "heavy_only": True,
            },
        )

    def test_context_requires_units_and_semantic_selections(self):
        project_path = EXAMPLE / "project.json"
        project = json.loads(project_path.read_text(encoding="utf-8"))
        project.pop("coordinate_unit")
        project["selections"].pop("alignment")
        with self.assertRaises(ManifestValidationError) as context:
            compile_project_context(project, project_path)
        self.assertIn("coordinate_unit is required", str(context.exception))
        self.assertIn("alignment", str(context.exception))

    def test_multiple_systems_require_explicit_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            topology = root / "one.pdb"
            trajectory = root / "two.xyz"
            topology.write_text("ATOM      1  C   UNK A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n", encoding="utf-8")
            trajectory.write_text("1\nframe\nC 0.0 0.0 0.0\n", encoding="utf-8")
            replica = {
                "replica_id": "r1",
                "topology": "one.pdb",
                "segments": [{
                    "segment_id": "s1", "trajectory": "two.xyz",
                    "timing": {"first_frame_time": 0, "frame_interval": 1000, "unit": "fs"},
                }],
            }
            system_path = root / "system.json"
            system_path.write_text(json.dumps({"systems": [
                {"system_id": "a", "replicas": [replica]},
                {"system_id": "b", "replicas": [replica]},
            ]}), encoding="utf-8")
            project = {
                "project_id": "multi",
                "analysis_profile": "standard_md_v1",
                "system_manifest": "system.json",
                "analysis_output_root": "outputs",
                "sampling_mode": "UNBIASED_MD",
                "coordinate_unit": "angstrom",
                "time_unit": "ps",
                "periodic_coordinate_policy": "reject",
                "selections": {
                    "alignment": {"preset": "backbone"},
                    "analysis": {"preset": "heavy"},
                },
                "protected_locations": ["/protected/example"],
            }
            project_path = root / "project.json"
            project_path.write_text(json.dumps(project), encoding="utf-8")
            with self.assertRaises(ManifestValidationError) as context:
                compile_project_context(project, project_path)
            self.assertIn("reference_system is required", str(context.exception))

    def test_selection_definition_rejects_ambiguous_and_duplicate_rules(self):
        project = {
            "project_id": "invalid-selection",
            "analysis_profile": "standard_md_v1",
            "system_manifest": "system.json",
            "analysis_output_root": "outputs",
            "sampling_mode": "UNBIASED_MD",
            "periodic_coordinate_policy": "reject",
            "selections": {
                "alignment": {"preset": "backbone", "atom_names": ["CA", "CA"]},
            },
            "protected_locations": ["/protected/example"],
        }
        with self.assertRaises(ManifestValidationError) as context:
            validate_project(project)
        self.assertIn("exactly one", str(context.exception))

        project["selections"]["alignment"] = {"atom_names": ["CA", "CA"]}
        with self.assertRaises(ManifestValidationError) as context:
            validate_project(project)
        self.assertIn("contains duplicates", str(context.exception))

    def test_production_interval_is_normalized_to_declared_time_unit(self):
        project_path = EXAMPLE / "project.json"
        project = json.loads(project_path.read_text(encoding="utf-8"))
        project["reference_system"] = "synthetic-one-atom-system"
        project["production_interval"] = {"start": 0, "end": 1000, "unit": "fs"}
        report = compile_project_context(project, project_path)
        self.assertEqual(
            report["contract"]["production_interval"],
            {"start": 0.0, "end": 1.0, "unit": "ps", "declared_unit": "fs"},
        )
        timing = report["contract"]["systems"][0]["replicas"][0]["segments"][0]["timing"]
        self.assertEqual(timing["frame_interval"], 2.0)

    def test_ai_ensemble_uses_sample_indices_without_invented_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "one.pdb").write_text(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
                encoding="utf-8",
            )
            (root / "two.xyz").write_text(
                "1\nsample-0\nC 0 0 0\n1\nsample-1\nC 1 0 0\n",
                encoding="utf-8",
            )
            system = {"systems": [{"system_id": "ai", "replicas": [{
                "replica_id": "r1", "topology": "one.pdb", "segments": [{
                    "segment_id": "samples", "trajectory": "two.xyz",
                    "sample_axis": {"first_sample_index": 100, "sample_interval": 2},
                }],
            }]}]}
            (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
            project = {
                "project_id": "ai", "analysis_profile": "standard_md_v1",
                "system_manifest": "system.json", "analysis_output_root": "outputs",
                "sampling_mode": "AI_ENSEMBLE", "coordinate_unit": "angstrom",
                "periodic_coordinate_policy": "reject",
                "selections": {"alignment": {"atom_names": ["CA"]}, "analysis": {"atom_names": ["CA"]}},
                "protected_locations": ["/protected/example"],
            }
            path = root / "project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            report = compile_project_context_file(path)
            segment = report["contract"]["systems"][0]["replicas"][0]["segments"][0]
            self.assertIsNone(report["contract"]["units"]["time"])
            self.assertEqual(segment["frame_axis_kind"], "sample_index")
            self.assertEqual(segment["sample_axis"]["first_sample_index"], 100)
            self.assertNotIn("timing", segment)

            system["systems"][0]["replicas"][0]["segments"][0] = {
                "segment_id": "samples", "trajectory": "two.xyz",
                "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"},
            }
            (root / "system.json").write_text(json.dumps(system), encoding="utf-8")
            with self.assertRaises(ManifestValidationError) as context:
                compile_project_context_file(path)
            self.assertIn("physical timing must not be invented", str(context.exception))

    def test_cli_emits_machine_readable_context(self):
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["compile-context", str(EXAMPLE / "project.json")])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["contract"]["units"]["coordinates"], "angstrom")
        self.assertIsNone(report["input_content_signature_sha256"])


if __name__ == "__main__":
    unittest.main()
