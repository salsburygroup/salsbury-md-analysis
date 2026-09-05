import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.user_workflow import (
    add_parsers, initialize, diagnose, prepare_study, workflow_status,
    execute_workflow, campaign_lock, _study,
)
from test_quickstart import _write_inputs


class UserWorkflowTests(unittest.TestCase):
    def setup_study(self, root):
        pdb, psf, trajectories = _write_inputs(root)
        parser = argparse.ArgumentParser()
        add_parsers(parser.add_subparsers(dest="command"))
        args = parser.parse_args(["init", str(root / "a study with spaces"), "--pdb", str(pdb),
            "--connectivity", str(psf), "--trajectory", str(trajectories[0]),
            "--frame-interval-ps", "100", "--cpus", "2", "--hours", "8"])
        return args, initialize(args)

    def test_setup_is_editable_and_does_not_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, result = self.setup_study(Path(tmp))
            self.assertFalse(result["jobs_submitted"])
            source, study, systems, config = _study(result["study"])
            self.assertEqual(len(systems), 1)
            self.assertEqual(json.loads(config.read_text())["execution"]["maximum_total_cpu_hours"], 16)
            self.assertEqual(diagnose(source)["technical_status"], "complete")
            with self.assertRaisesRegex(ValueError, "existing"):
                initialize(args)

    def test_plan_does_not_run_or_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, result = self.setup_study(Path(tmp))
            with patch("salsbury_md_analysis.quickstart.prepare_standard_analysis", return_value={"technical_status":"complete"}) as prepare:
                plan = prepare_study(result["study"])
            prepare.assert_called_once()
            self.assertFalse(plan["jobs_submitted"])
            self.assertFalse(plan["execution_started"])

    def test_malformed_system_and_atom_mismatch_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, result = self.setup_study(Path(tmp))
            path = Path(result["study"])
            study = json.loads(path.read_text())
            study["systems"][0]["trajectories"] = "wrong"
            path.write_text(json.dumps(study))
            with self.assertRaisesRegex(ValueError, "trajectories"):
                _study(path)

    def test_resume_review_never_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "analysis-config.json").write_text(json.dumps({"execution":{"submission_adapter":"local"}}))
            (root / "local-execution-plan.json").write_text(json.dumps({"phases":[{"tasks":[
                {"task_id":"a","completion_reports":["results/a/report.json"]},
                {"task_id":"b","depends_on_task_ids":["a"],"completion_reports":["results/b/report.json"]}]}]}))
            with patch("salsbury_md_analysis.execution_adapters.run_local_workflow") as run:
                result = execute_workflow(root, resume=True, execute=False)
            run.assert_not_called()
            self.assertEqual(result["counts"], {"not_complete":1,"dependency_blocked":1})
            self.assertFalse(result["execution_started"])

    def test_campaign_lock_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with campaign_lock(Path(tmp)):
                with self.assertRaisesRegex(ValueError, "owns"):
                    with campaign_lock(Path(tmp)):
                        pass

    def test_native_windows_diagnostic(self):
        with patch("platform.system", return_value="Windows"):
            self.assertEqual(diagnose()["technical_status"], "failed")

    def test_live_slurm_jobs_are_not_reported_as_failures_from_partial_files(self):
        from types import SimpleNamespace
        from salsbury_md_analysis.user_workflow import campaign_activity
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "analysis-config.json").write_text(json.dumps({"execution":{"submission_adapter":"slurm"}}))
            (root / "slurm-profile.json").write_text("{}")
            (root / "submission-ledgers").mkdir()
            (root / "submission-ledgers/attempt.tsv").write_text("task_id\tjob_id\na\t123\n")
            (root / "local-execution-plan.json").write_text(json.dumps({"phases":[{"tasks":[{"task_id":"a","completion_reports":["missing.json"]}]}]}))
            with patch("salsbury_md_analysis.execution_adapters.load_slurm_profile", return_value={"status_command":"/site/bin/squeue"}), patch("salsbury_md_analysis.user_workflow.subprocess.run", return_value=SimpleNamespace(stdout=f"123_0|RUNNING|{root}\n")) as query:
                result = workflow_status(root)
            self.assertEqual(result["counts"], {"running":1})
            self.assertEqual(query.call_args.args[0][0], "/site/bin/squeue")

    def test_missing_scheduler_profile_is_unknown_not_idle(self):
        from salsbury_md_analysis.user_workflow import campaign_activity
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "analysis-config.json").write_text(json.dumps({"execution":{"submission_adapter":"slurm"}}))
            self.assertEqual(campaign_activity(root)["scheduler_query"], "failed")


if __name__ == "__main__":
    unittest.main()
