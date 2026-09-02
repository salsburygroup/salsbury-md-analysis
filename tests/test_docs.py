import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_docs


class GeneratedDocumentationTests(unittest.TestCase):
    def test_json_documentation_examples_are_strict_json(self):
        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        markdown_paths = sorted(ROOT.glob("*.md"))
        markdown_paths += sorted((ROOT / "docs").rglob("*.md"))
        markdown_paths += sorted((ROOT / "tutorials").rglob("*.md"))
        for path in markdown_paths:
            text = path.read_text(encoding="utf-8")
            for index, block in enumerate(
                re.findall(r"```json\s*\n(.*?)```", text, re.DOTALL), start=1
            ):
                try:
                    json.loads(block, object_pairs_hook=reject_duplicate_keys)
                except (json.JSONDecodeError, ValueError) as exc:
                    self.fail(
                        f"{path.relative_to(ROOT)} JSON block {index} is invalid: {exc}"
                    )

    def test_documentation_index_links_every_handwritten_method_page(self):
        index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
        for path in sorted((ROOT / "docs").glob("*.md")):
            if path.name == "index.md":
                continue
            self.assertIn(path.name, index, f"unlinked documentation page: {path.name}")

    def test_module_reference_contains_full_scope(self):
        text = generate_docs.render_module_reference()
        self.assertIn("Markov-state models", text)
        self.assertIn("RMSF permutation inference", text)
        self.assertNotIn("Docking", text)
        self.assertEqual(text.count("| **experimental** |"), len(generate_docs.MODULES))
        self.assertNotIn("| **planned** |", text)

    def test_coverage_does_not_claim_implementation(self):
        text = generate_docs.render_analysis_coverage()
        self.assertIn("not an implementation claim", text)

    def test_current_public_registry_count_matches_code(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readiness = (ROOT / "PRODUCTION_READINESS.md").read_text(encoding="utf-8")
        current_count = len(generate_docs.MODULES)
        self.assertIn(f"{current_count} registered MD, core, and reporting modules", readme)
        self.assertIn(
            f"{current_count} of {current_count} MD/core/reporting modules",
            readiness,
        )

    def test_profile_reference_uses_latest_resource_catalog(self):
        text = generate_docs.render_profiles()
        self.assertIn("apollo_measured_resource_calibrations_v5.json", text)
        self.assertIn("Right-censored timeouts:", text)

    def test_methods_citation_map_names_every_registered_module(self):
        text = (ROOT / "docs" / "METHODS_AND_CITATIONS.md").read_text(
            encoding="utf-8"
        )
        for module in generate_docs.MODULES:
            self.assertIn(
                f"`{module.module_id}`", text,
                f"uncited registered module: {module.module_id}",
            )
        self.assertIn("Scite review could\nnot be completed", text)
        self.assertIn("Do not describe the reference list as\nScite-checked", text)

    def test_recovery_documentation_states_default_and_opt_out(self):
        execution = (ROOT / "docs" / "EXECUTION_ADAPTERS.md").read_text(
            encoding="utf-8"
        )
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for text in (execution, readme):
            self.assertIn("execution.autorecovery", text)
        self.assertIn('"maximum_task_attempts": 2', execution)
        self.assertIn("one worker per replica", execution)
        self.assertIn("does not require every saved frame", execution)

    def test_terminal_tutorial_and_bounded_fixture_acceptance(self):
        tutorial = (ROOT / "tutorials" / "local_and_cluster" / "README.md").read_text()
        self.assertIn("--plan-only", tutorial)
        self.assertIn("execution.autorecovery", tutorial)
        self.assertIn("./submit.sh --preview", tutorial)
        self.assertIn("scientific validity", tutorial.lower())
        acceptance = json.loads(
            (ROOT / "validation" / "trex_thrombin_technical_fixture_acceptance.json").read_text()
        )
        self.assertEqual(acceptance["technical_acceptance"], "pass")
        self.assertEqual(acceptance["simulation_jobs_submitted"], 0)
        self.assertFalse(acceptance["screen_docking_data_used"])
        self.assertEqual(acceptance["unexpected_errors"], [])
        self.assertEqual(
            set(acceptance["hydrogen_bond_endpoint_accounting"]["trex-control"]
                ["conceptual_candidate_stratum_counts"]),
            {
                "protein_to_protein",
                "protein_to_nucleic_acid",
                "nucleic_acid_to_protein",
                "nucleic_acid_to_nucleic_acid",
            },
        )
        for record in acceptance["periodic_ion_distance_checks"].values():
            self.assertTrue(record["exact_minimum_image_invariant"])
            self.assertLess(
                record["maximum_lattice_translation_distance_delta_angstrom"],
                1e-10,
            )

    def test_handwritten_validation_docs_preserve_current_scientific_boundaries(self):
        validation = (ROOT / "docs" / "SCIENTIFIC_VALIDATION.md").read_text(
            encoding="utf-8"
        )
        structural_qc = (ROOT / "docs" / "STRUCTURAL_QC.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Every module remains `experimental`", validation)
        self.assertIn("does not turn those values into", validation)
        self.assertIn("convergence or population-validity verdict", validation)
        self.assertIn("optional peptide-link continuity", structural_qc)
        self.assertNotIn("does **not** yet evaluate residue continuity", structural_qc)

    def test_general_system_guide_is_plain_language_and_does_not_overclaim(self):
        guide = (ROOT / "docs" / "GENERAL_BIOMOLECULAR_SYSTEMS.md").read_text(
            encoding="utf-8"
        )
        policy = (ROOT / "PUBLICATION_REPOSITORY_POLICY.md").read_text(
            encoding="utf-8"
        )
        normalized = " ".join(guide.split())
        self.assertIn("user should not need to write project-specific", normalized)
        self.assertIn("RA/RC/RG/RU", normalized)
        self.assertIn("does not decide which residue matters", normalized)
        self.assertIn("unknown polymer", normalized)
        self.assertIn("remains private", policy)

    def test_current_dependency_record_is_in_source_archive_manifest(self):
        setup = (ROOT / "setup.cfg").read_text(encoding="utf-8")
        match = re.search(r"^version\s*=\s*([^\s]+)\s*$", setup, re.MULTILINE)
        self.assertIsNotNone(match)
        version = match.group(1)
        development = re.fullmatch(r"0\.0\.1\.dev(\d+)", version)
        record_suffix = development.group(1) if development else version
        record_name = f"v{record_suffix}_dependency_test.json"
        record_path = ROOT / "validation" / record_name
        self.assertTrue(record_path.is_file())
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["candidate_version"], version)
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        self.assertIn(f"include validation/{record_name}", manifest.splitlines())


if __name__ == "__main__":
    unittest.main()
