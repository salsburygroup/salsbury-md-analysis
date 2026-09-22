import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("nemo_zinc_finger_workstation", "nemo_zinc_finger_cluster",
         "nemo_zinc_finger_deac", "workstation", "cluster")


class TutorialResourceTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("bash"), "bash is required for shell syntax checks")
    def test_tutorial_bash_syntax(self):
        for path in (ROOT / "tutorials").rglob("*.md"):
            for number, block in enumerate(re.findall(r"```bash\n(.*?)\n```", path.read_text(), re.S)):
                with self.subTest(path=path.name, block=number):
                    self.assertNotRegex(block, r"(?m)^\+  --")
                    result = subprocess.run(["bash", "-n"], input=block, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_routes_explain_resource_limits(self):
        for page in PAGES:
            text = (ROOT / "tutorials" / page / "README.md").read_text()
            self.assertIn("../RESOURCE_PLANNING.md", text)
        guide = (ROOT / "tutorials/RESOURCE_PLANNING.md").read_text()
        for value in ("maximum_total_cpu_hours", "request_replay_policy",
                      "job-specific calibrations", "new `--output`", "open(\"xb\")",
                      "Required execution overhead", "tutorial_orchestration_20260922.md"):
            self.assertIn(value, guide)

    def test_nemo_budgets_and_recovery_paths(self):
        config = json.loads((ROOT / "tutorials/nemo_zinc_finger_workstation/analysis-config.json").read_text())
        self.assertEqual(config["execution"]["maximum_hours_per_cpu"], 2.0)
        self.assertEqual(config["execution"]["maximum_parallel_cpus"], 2)
        self.assertEqual(config["default_module_policy"], "all_applicable")
        for page, hours in (("nemo_zinc_finger_cluster", 10), ("nemo_zinc_finger_deac", 16)):
            text = (ROOT / "tutorials" / page / "README.md").read_text()
            self.assertRegex(text, rf"--hours {hours}\s")
            self.assertRegex(text, rf"--wall-hours {hours}\s")
            self.assertIn('export NEMO_ANALYSIS="$NEMO_STUDY/analysis"', text)
            self.assertIn('--output "$NEMO_ANALYSIS"', text)
            self.assertEqual(text.count('"$NEMO_STUDY/analysis"'), 1)
            self.assertIn('"$CORE_CMD" run "$NEMO_ANALYSIS"', text)


if __name__ == "__main__":
    unittest.main()
