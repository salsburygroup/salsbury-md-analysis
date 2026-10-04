"""Opt-in, process-local CLI diagnostics; no sampling profiler or dependencies."""
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import traceback

from . import __version__

_active = None


class PlanningDiagnosticsSetupError(RuntimeError):
    """Diagnostics could not start; preparation has not been invoked."""


def _utc():
    return datetime.now(timezone.utc).isoformat()


def planning_event(name, **fields):
    """Record a major preparation boundary only when tracing is enabled."""
    if _active is not None:
        _active.event(name, **fields)


class PlanningDiagnostics:
    def __init__(self, directory, command, interval):
        self.directory = directory
        self.started = time.monotonic()
        self.cpu_started = time.process_time()
        self.exit_code = None
        self.write_warning_reported = False
        self.events = (directory / "events.jsonl").open("x", encoding="utf-8")
        try:
            self.stacks = (directory / "stack-traces.txt").open("x", encoding="utf-8")
        except OSError:
            self.events.close()
            raise
        self.metadata = {
            "schema": "salsbury-planning-diagnostics-v1", "command": command,
            "pid": os.getpid(), "package_version": __version__,
            "python_version": sys.version, "started_at": _utc(),
            "stack_interval_seconds": interval, "status": "running",
            "termination_trace_enabled": False,
            "boundary": "Diagnostic evidence only; a stale running status does not establish a live process or a feasible plan.",
        }

    def save_metadata(self):
        temporary = self.directory / "metadata.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "metadata.json")

    def event(self, name, **fields):
        try:
            self.events.write(json.dumps({"at": _utc(), "elapsed_seconds": time.monotonic() - self.started,
                                          "event": name, **fields}, sort_keys=True) + "\n")
            self.events.flush()
        except OSError as exc:
            self.warn_write_failure(exc)

    def warn_write_failure(self, exc):
        if not self.write_warning_reported:
            print(f"Planning diagnostics incomplete ({self.directory}): {exc}", file=sys.stderr)
            self.write_warning_reported = True


@contextmanager
def planning_diagnostics(output_directory, *, command, interval_seconds=60.0):
    """Own faulthandler's watchdog for one opt-in CLI preparation invocation.

    Stack snapshots contain Python frames, not C stacks, locals or trajectory
    data. SIGTERM is chained to its previous handler on supported platforms.
    Diagnostics do not extend deadlines, change planning, or recover a run.
    """
    global _active
    interval = float(interval_seconds)
    if not math.isfinite(interval) or interval < 1.0:
        raise ValueError("planning stack interval must be finite and at least 1 second")
    if _active is not None:
        raise ValueError("planning diagnostics already active in this process")
    output = Path(output_directory).expanduser().resolve()
    parent = output.with_name(output.name + ".planning-diagnostics")
    try:
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"{os.getpid()}-", dir=parent))
        diagnostics = PlanningDiagnostics(directory, command, interval)
    except OSError as exc:
        raise PlanningDiagnosticsSetupError(str(exc)) from exc
    registered = False
    timer_started = False
    ready = False
    error = None
    try:
        diagnostics.stacks.write(f"Planning diagnostics started {diagnostics.metadata['started_at']}\n"
                                 f"PID {os.getpid()}; repeated Python stack snapshots every {interval:g} seconds\n")
        diagnostics.stacks.flush()
        # Preserve the command's termination behavior. Windows has no register;
        # its periodic snapshots still work. SIGKILL cannot be intercepted.
        if hasattr(faulthandler, "register") and hasattr(signal, "SIGTERM"):
            try:
                faulthandler.register(signal.SIGTERM, file=diagnostics.stacks,
                                      all_threads=True, chain=True)
                registered = True
            except (OSError, RuntimeError, ValueError) as exc:
                diagnostics.metadata["termination_trace_unavailable"] = type(exc).__name__
        diagnostics.metadata["termination_trace_enabled"] = registered
        diagnostics.save_metadata()
        faulthandler.dump_traceback_later(interval, repeat=True, file=diagnostics.stacks)
        timer_started = True
        _active = diagnostics
        diagnostics.event("preparation_started")
        print(f"Planning diagnostics: {directory}", file=sys.stderr, flush=True)
        ready = True
        yield diagnostics
    except BaseException as exc:
        error = type(exc).__name__
        try:
            traceback.print_exc(file=diagnostics.stacks)
            diagnostics.stacks.flush()
        except OSError as write_error:
            diagnostics.warn_write_failure(write_error)
        if not ready and isinstance(exc, (OSError, ValueError, RuntimeError)):
            raise PlanningDiagnosticsSetupError(str(exc)) from exc
        raise
    finally:
        _active = None
        if timer_started:
            faulthandler.cancel_dump_traceback_later()
        if registered:
            faulthandler.unregister(signal.SIGTERM)
        diagnostics.metadata.update({
            "finished_at": _utc(), "elapsed_seconds": time.monotonic() - diagnostics.started,
            "process_cpu_seconds": time.process_time() - diagnostics.cpu_started,
            "exit_code": diagnostics.exit_code, "exception_type": error,
            "status": "failed" if error or diagnostics.exit_code not in (None, 0) else "completed",
        })
        diagnostics.event("preparation_finished", status=diagnostics.metadata["status"],
                          exit_code=diagnostics.exit_code, exception_type=error)
        try:
            diagnostics.save_metadata()
        except OSError as exc:
            diagnostics.warn_write_failure(exc)
        for handle in (diagnostics.events, diagnostics.stacks):
            try:
                handle.close()
            except OSError as exc:
                diagnostics.warn_write_failure(exc)
