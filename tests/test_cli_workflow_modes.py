import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from salsbury_md_analysis.cli import build_parser, main
from tests.test_quickstart import _write_inputs


class CLIWorkflowModeTests(unittest.TestCase):
    def test_prepare_commands_expose_explicit_plan_only_switch(self):
        parser = build_parser()
        analysis = parser.parse_args([
            "prepare-analysis", "--pdb", "input.pdb", "--trajectory", "run.dcd",
            "--output", "out", "--project-id", "test",
            "--frame-interval-ps", "10", "--plan-only",
        ])
        comparison = parser.parse_args([
            "prepare-comparison", "comparison.json", "--output", "out",
            "--project-id", "test", "--plan-only",
        ])
        self.assertTrue(analysis.plan_only)
        self.assertTrue(comparison.plan_only)

    def test_minimums_template_command_writes_once_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "minimums.json"
            self.assertEqual(main([
                "write-scientific-minimums-template", "--output", str(path)
            ]), 0)
            first = path.read_bytes()
            self.assertEqual(main([
                "write-scientific-minimums-template", "--output", str(path)
            ]), 2)
            self.assertEqual(path.read_bytes(), first)

    def test_plan_only_returns_complete_plan_without_starting_executor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            pdb, psf, trajectories = _write_inputs(inputs)
            output = root / "prepared"
            stdout = io.StringIO()
            arguments = [
                "prepare-analysis", "--pdb", str(pdb), "--psf", str(psf),
                "--frame-interval-ps", "10", "--project-id", "plan-only-test",
                "--output", str(output), "--plan-only",
            ]
            for trajectory in trajectories:
                arguments.extend(["--trajectory", str(trajectory)])
            with redirect_stdout(stdout):
                result = main(arguments)
            report = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(report["planning_mode"], "plan_only")
        self.assertFalse(report["execution_started"])
        self.assertFalse(report["jobs_submitted"])
        self.assertTrue(report["campaign_resource_plan"]["tasks"])


if __name__ == "__main__":
    unittest.main()
