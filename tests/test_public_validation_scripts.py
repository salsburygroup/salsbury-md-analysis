import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicValidationScriptTests(unittest.TestCase):
    def test_public_hydrogen_bond_validation_passes(self):
        path = (
            ROOT
            / "validation"
            / "public"
            / "run_hydrogen_bond_synthetic_validation.py"
        )
        spec = importlib.util.spec_from_file_location(
            "public_hydrogen_bond_validation", path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.run_validation()
        self.assertEqual(report["technical_status"], "complete")
        self.assertTrue(report["chemistry"]["passed"])
        self.assertTrue(report["geometry"]["passed"])
        self.assertEqual(len(report["geometry"]["cases"]), 3)


if __name__ == "__main__":
    unittest.main()
