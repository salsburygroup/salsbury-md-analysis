import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.cli import build_parser, main
from salsbury_md_analysis.quickstart import QuickstartPlanningError
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

    def test_plan_only_returns_structured_no_acceptable_reduced_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            plan = {
                "feasibility_status": "infeasible",
                "warning_count": 1,
                "resource_warnings": [{
                    "severity": "warning",
                    "code": "REQUESTED_CPUS_EXCEED_USEFUL_PARALLELISM",
                    "message": "requested CPUs exceed the useful ceiling",
                }],
                "method_reduction_recommendation": {
                    "recommendation_status": "no_feasible_subset_found",
                    "recommendation_message": "No acceptable reduced plan",
                },
            }
            failure = QuickstartPlanningError(
                "No acceptable reduced plan: protected QC does not fit",
                plan=plan,
                analysis_config={"config_schema": "salsbury-analysis-config-v1"},
                output_directory=output,
            )
            stdout = io.StringIO()
            with patch(
                "salsbury_md_analysis.cli.prepare_standard_analysis",
                side_effect=failure,
            ), redirect_stdout(stdout):
                result = main([
                    "prepare-analysis", "--pdb", "input.pdb",
                    "--trajectory", "run.dcd", "--frame-interval-ps", "10",
                    "--project-id", "test", "--output", str(output),
                    "--plan-only",
                ])
            report = json.loads(stdout.getvalue())
        self.assertEqual(result, 2)
        self.assertEqual(report["planning_mode"], "plan_only")
        self.assertEqual(
            report["planning_outcome"], "no_acceptable_reduced_plan"
        )
        self.assertEqual(report["issues"][0]["code"], "NO_ACCEPTABLE_REDUCED_PLAN")
        self.assertFalse(report["execution_started"])
        self.assertFalse(report["jobs_submitted"])
        self.assertEqual(report["warning_count"], 1)
        self.assertEqual(report["campaign_resource_plan"], plan)
        self.assertEqual(
            report["planning_summary"]["reduction_status"],
            "no_feasible_subset_found",
        )
        self.assertEqual(report["planning_summary"]["configuration_patch"], {})


if __name__ == "__main__":
    unittest.main()
