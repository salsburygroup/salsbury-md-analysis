"""Diagnostics use synthetic preparation fixtures, never scientific execution."""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from salsbury_md_analysis.cli import _run_preparation_command, build_parser, main
from salsbury_md_analysis.planning_diagnostics import planning_diagnostics, planning_event
from tests.test_quickstart import _write_inputs


class PlanningDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def args(self, enabled=True):
        return SimpleNamespace(diagnose_planning=enabled, output=self.root / "plan",
                               command="prepare-analysis", planning_stack_interval_seconds=60)

    def invoke(self, args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(args)
        return code, out.getvalue(), err.getvalue()

    def test_flags_and_invalid_intervals(self):
        parser = build_parser()
        commands = [
            ["prepare-analysis", "--pdb", "a.pdb", "--trajectory", "a.dcd", "--frame-interval-ps", "10"],
            ["prepare-comparison", "request.json"],
        ]
        for command in commands:
            base = command + ["--output", "out", "--project-id", "test"]
            self.assertFalse(parser.parse_args(base).diagnose_planning)
            self.assertEqual(parser.parse_args(base).planning_stack_interval_seconds, 60)
            self.assertTrue(parser.parse_args(base + ["--diagnose-planning"]).diagnose_planning)
            for interval in ("0", "-1", "nan", "inf", "text"):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parser.parse_args(base + ["--planning-stack-interval-seconds", interval])

    def test_disabled_creates_no_files_or_watchdog(self):
        with patch("faulthandler.dump_traceback_later") as timer:
            self.assertEqual(_run_preparation_command(self.args(False), lambda: 7), 7)
        timer.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_unique_sessions_events_cleanup_and_exit_codes(self):
        directories = []
        for code in (0, 2):
            with redirect_stderr(io.StringIO()):
                with planning_diagnostics(self.root / "plan", command="test") as record:
                    directories.append(record.directory)
                    planning_event("test_phase", trial="core")
                    record.exit_code = code
                    with self.assertRaisesRegex(ValueError, "already active"):
                        with planning_diagnostics(self.root / "other", command="nested"):
                            self.fail("nested diagnostics must not start")
            metadata = json.loads((record.directory / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed" if code == 0 else "failed")
            self.assertEqual(metadata["exit_code"], code)
            self.assertGreaterEqual(metadata["process_cpu_seconds"], 0)
            events = [json.loads(row) for row in (record.directory / "events.jsonl").read_text().splitlines()]
            self.assertEqual([e["event"] for e in events], ["preparation_started", "test_phase", "preparation_finished"])
            self.assertTrue(record.events.closed)
            self.assertTrue(record.stacks.closed)
        self.assertNotEqual(*directories)
        planning_event("after_close")  # No attempt to write a closed file.

    def test_setup_failure_does_not_invoke_preparation(self):
        (self.root / "plan.planning-diagnostics").write_text("preserve me")
        out = io.StringIO()
        with patch("salsbury_md_analysis.cli._prepare_analysis_command") as operation, redirect_stdout(out):
            self.assertEqual(_run_preparation_command(self.args(), operation), 2)
        operation.assert_not_called()
        self.assertEqual(json.loads(out.getvalue())["issues"][0]["code"], "PLANNING_DIAGNOSTICS_FAILED")

    def test_body_exception_keeps_original_type_and_trace(self):
        def preparation():
            raise ValueError("original preparation defect")
        with redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "original preparation defect"):
            _run_preparation_command(self.args(), preparation)
        directory = next((self.root / "plan.planning-diagnostics").iterdir())
        self.assertIn("original preparation defect", (directory / "stack-traces.txt").read_text())
        self.assertEqual(json.loads((directory / "metadata.json").read_text())["exception_type"], "ValueError")

    def test_late_disk_failure_does_not_change_operation_result(self):
        def fail_write():
            raise OSError("disk full")
        err = io.StringIO()
        with redirect_stderr(err):
            with planning_diagnostics(self.root / "plan", command="test") as record:
                record.exit_code = 0
                record.save_metadata = fail_write
        self.assertIn("Planning diagnostics incomplete", err.getvalue())
        self.assertTrue(record.stacks.closed)

    def test_watchdog_start_failure_is_setup_error_and_does_not_call_operation(self):
        out, err = io.StringIO(), io.StringIO()
        with patch("faulthandler.dump_traceback_later", side_effect=RuntimeError("watchdog unavailable")), \
                patch("salsbury_md_analysis.cli._prepare_analysis_command") as operation, \
                redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(_run_preparation_command(self.args(), operation), 2)
        operation.assert_not_called()
        self.assertEqual(json.loads(out.getvalue())["issues"][0]["code"], "PLANNING_DIAGNOSTICS_FAILED")
    def test_resource_fit_trial_boundaries_are_recorded_on_core_rejection(self):
        from salsbury_md_analysis.quickstart import QuickstartPlanningError, prepare_standard_analysis_resource_fit
        pdb, psf, trajectories = _write_inputs(self.root)
        config = self.root / "config.json"
        config.write_text(json.dumps({"config_schema": "salsbury-analysis-config-v1", "execution": {"maximum_memory_gib": 0.01}}))
        with redirect_stderr(io.StringIO()):
            with planning_diagnostics(self.root / "plan", command="test") as record:
                with self.assertRaises(QuickstartPlanningError):
                    prepare_standard_analysis_resource_fit(pdb_path=pdb, psf_path=psf,
                        trajectories=trajectories, project_id="test", frame_interval_ps=10,
                        output_directory=self.root / "plan", config_path=config)
                record.exit_code = 2
        events = [json.loads(row) for row in (record.directory / "events.jsonl").read_text().splitlines()]
        trials = [row for row in events if row["event"].startswith("resource_fit_trial")]
        self.assertEqual([row["trial"] for row in trials], ["protected-core", "protected-core"])
        self.assertEqual(trials[-1]["status"], "rejected")
        self.assertFalse((self.root / "plan/submit.sh").exists())

    def test_signal_support_optional(self):
        # register is absent on Windows. A failed registration elsewhere must
        # also leave periodic snapshots available.
        with patch("faulthandler.register", create=True, side_effect=RuntimeError("unavailable")), redirect_stderr(io.StringIO()):
            with planning_diagnostics(self.root / "plan", command="test") as record:
                self.assertFalse(record.metadata["termination_trace_enabled"])
        self.assertEqual(json.loads((record.directory / "metadata.json").read_text())["status"], "completed")

    def test_periodic_snapshot_subprocess(self):
        code = """
import time
from pathlib import Path
from salsbury_md_analysis.planning_diagnostics import planning_diagnostics
def named_planning_wait():
    time.sleep(1.2)
with planning_diagnostics(Path(__import__('sys').argv[1]), command='test', interval_seconds=1) as record:
    named_planning_wait()
    record.exit_code = 0
print(record.directory)
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.root / "plan")],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = Path(result.stdout.strip())
        self.assertIn("named_planning_wait", (directory / "stack-traces.txt").read_text())
        self.assertIn("Timeout", (directory / "stack-traces.txt").read_text())

    def test_existing_fatal_handler_is_not_replaced_or_disabled(self):
        with patch("faulthandler.is_enabled", return_value=True), patch("faulthandler.enable") as enable, patch("faulthandler.disable") as disable, redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "sentinel"):
                with planning_diagnostics(self.root / "plan", command="test") as record:
                    self.assertEqual(record.metadata["fatal_trace_destination"], "existing_process_handler")
                    raise ValueError("sentinel")
        enable.assert_not_called()
        disable.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "fatal-signal subprocess test requires POSIX")
    def test_fatal_signal_is_traced_without_suppressing_failure(self):
        code = """
import faulthandler, os, resource, signal, sys
from pathlib import Path
from salsbury_md_analysis.planning_diagnostics import planning_diagnostics
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
faulthandler.disable()
def named_fatal_probe():
    os.kill(os.getpid(), signal.SIGABRT)
with planning_diagnostics(Path(sys.argv[1]), command='fatal-test'):
    named_fatal_probe()
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.root / "plan")],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, -signal.SIGABRT, result.stderr)
        directory = next((self.root / "plan.planning-diagnostics").iterdir())
        trace = (directory / "stack-traces.txt").read_text()
        self.assertIn("Fatal Python error", trace)
        self.assertIn("named_fatal_probe", trace)
        metadata = json.loads((directory / "metadata.json").read_text())
        self.assertTrue(metadata["fatal_trace_enabled"])
        self.assertEqual(metadata["fatal_trace_destination"], "stack-traces.txt")
        self.assertEqual(metadata["status"], "running")  # interrupted, not accepted
        self.assertFalse((self.root / "plan").exists())

    def test_owned_fatal_handler_is_released_on_setup_failure(self):
        with patch("faulthandler.is_enabled", return_value=False), patch("faulthandler.enable") as enable, patch("faulthandler.disable") as disable, patch("faulthandler.dump_traceback_later", side_effect=RuntimeError("setup failed")), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "setup failed"):
                with planning_diagnostics(self.root / "plan", command="test"):
                    self.fail("setup must fail")
        enable.assert_called_once()
        disable.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "SIGTERM chaining is POSIX-specific")
    def test_sigterm_captures_stack_and_preserves_termination(self):
        code = """
import time, sys
from pathlib import Path
from salsbury_md_analysis.planning_diagnostics import planning_diagnostics
def named_planning_wait():
    print('ready', flush=True)
    time.sleep(30)
with planning_diagnostics(Path(sys.argv[1]), command='test'):
    named_planning_wait()
"""
        child = subprocess.Popen([sys.executable, "-c", code, str(self.root / "plan")],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "ready")
            child.terminate()
            _, err = child.communicate(timeout=10)
            self.assertEqual(child.returncode, -signal.SIGTERM, err)
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate()
        directory = next((self.root / "plan.planning-diagnostics").iterdir())
        self.assertIn("named_planning_wait", (directory / "stack-traces.txt").read_text())
        self.assertEqual(json.loads((directory / "metadata.json").read_text())["status"], "running")
        self.assertFalse((self.root / "plan").exists())

    @unittest.skipUnless(os.name == "posix", "SIGTERM chaining is POSIX-specific")
    def test_previous_python_signal_handler_restored(self):
        code = """
import os, signal, sys
from pathlib import Path
from salsbury_md_analysis.planning_diagnostics import planning_diagnostics
calls = []
def previous_handler(*args):
    calls.append('called')
signal.signal(signal.SIGTERM, previous_handler)
with planning_diagnostics(Path(sys.argv[1]), command='test'):
    os.kill(os.getpid(), signal.SIGTERM)
os.kill(os.getpid(), signal.SIGTERM)
assert calls == ['called', 'called'], calls
"""
        result = subprocess.run([sys.executable, "-c", code, str(self.root / "plan")],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_preparation_sampling_resources_and_input_hashes_unchanged(self):
        pdb, psf, trajectories = _write_inputs(self.root)
        paths = [pdb, psf, *trajectories]
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        request = self.root / "request.json"
        request.write_text(json.dumps({"request_schema": "salsbury-comparative-analysis-input-v1",
            "systems": [{"system_id": name, "pdb": str(pdb), "psf": str(psf),
                         "trajectories": list(map(str, trajectories)), "frame_interval_ps": 10.0}
                        for name in ("A", "B")]}))
        single = ["prepare-analysis", "--pdb", str(pdb), "--psf", str(psf), "--frame-interval-ps", "10"]
        for path in trajectories:
            single.extend(["--trajectory", str(path)])
        for command in (single, ["prepare-comparison", str(request)]):
            plans = []
            for enabled in (False, True):
                output = self.root / (command[0] + str(enabled))
                args = command + ["--output", str(output), "--project-id", "diagnostic-test", "--plan-only"]
                if enabled:
                    args += ["--diagnose-planning"]
                code, text, err = self.invoke(args)
                self.assertEqual(code, 0, text + err)
                result = json.loads(text)
                self.assertFalse(result["execution_started"])
                self.assertFalse(result["jobs_submitted"])
                plan = json.loads((output / "campaign-resource-plan.json").read_text())
                plans.append({r["task_id"]: (r["integer_stride"], r["selected_physical_frames_per_replica"],
                    r.get("estimated_peak_memory_gib"), r.get("estimated_cpu_hours")) for r in plan["tasks"]})
                if enabled:
                    self.assertIn("Planning diagnostics:", err)
                else:
                    self.assertFalse(output.with_name(output.name + ".planning-diagnostics").exists())
            self.assertEqual(*plans)
        self.assertEqual(hashes, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    def test_real_cli_invalid_input_remains_one_json_failure(self):
        code, text, _ = self.invoke(["prepare-comparison", str(self.root / "missing.json"),
            "--output", str(self.root / "plan"), "--project-id", "test", "--diagnose-planning", "--plan-only"])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(text)["technical_status"], "failed")
        directory = next((self.root / "plan.planning-diagnostics").iterdir())
        self.assertEqual(json.loads((directory / "metadata.json").read_text())["exit_code"], 2)
