import unittest
from pathlib import Path

from salsbury_md_analysis.manifests import (
    inventory_system_inputs,
    load_json,
    validate_output,
    validate_project,
    validate_regression,
    validate_system,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "manifest_fixture"


class ExampleFixtureTests(unittest.TestCase):
    def test_project_and_system_examples_validate_with_paths(self):
        project_path = EXAMPLE / "project.json"
        system_path = EXAMPLE / "system.json"
        validate_project(load_json(project_path), source_path=project_path, check_paths=True)
        validate_system(load_json(system_path), source_path=system_path, check_paths=True)

    def test_example_inventory_contains_content_hashes(self):
        system_path = EXAMPLE / "system.json"
        inventory = inventory_system_inputs(
            load_json(system_path), source_path=system_path, hash_content=True
        )
        self.assertEqual(inventory["entry_count"], 2)
        self.assertTrue(all(entry["sha256"] for entry in inventory["entries"]))
        self.assertEqual(inventory["scientific_status"], "not evaluated")

    def test_output_example_verifies_recorded_hash(self):
        output_path = EXAMPLE / "output-manifest.json"
        validate_output(load_json(output_path), source_path=output_path, check_paths=True)

    def test_regression_example_is_hash_pinned_and_valid(self):
        case_path = EXAMPLE / "regression-case.json"
        validate_regression(load_json(case_path), source_path=case_path, check_paths=True)


if __name__ == "__main__":
    unittest.main()
