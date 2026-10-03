"""Attempt-scoped execution evidence, separate from accepted scientific reports."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def task_signature(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def persist_task_status(root, task, controller_attempt_id, record):
    """Atomically retain each event before a worker or controller continues."""
    directory = Path(root) / "local-execution-status" / "task-attempts"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        **record,
        "task_status_schema": "salsbury-task-attempt-status-v1",
        "task_id": task.get("task_id"),
        "task_sha256": task_signature(task),
        "controller_attempt_id": controller_attempt_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    final = directory / f"{uuid4().hex}.json"
    temporary = final.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, final)
    return payload


def latest_task_statuses(root, tasks):
    """Ignore incomplete writes and receipts for a different task contract."""
    signatures = {task.get("task_id"): task_signature(task) for task in tasks}
    latest = {}
    for path in (Path(root) / "local-execution-status" / "task-attempts").glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if (row.get("task_status_schema") != "salsbury-task-attempt-status-v1"
                    or row.get("task_id") not in signatures
                    or row.get("task_sha256") != signatures[row["task_id"]]
                    or not isinstance(row.get("recorded_at"), str)):
                continue
        except (OSError, ValueError, AttributeError, KeyError):
            continue
        key = row["task_id"]
        if row["recorded_at"] > latest.get(key, {}).get("recorded_at", ""):
            latest[key] = row
    return latest


def temporary_failure_evidence(root, task):
    """Read only complete failure JSON at this task's unique report locations.

    This is a legacy diagnostic fallback, never an acceptance mechanism. New
    attempt receipts take precedence, including an active retry or recovery.
    """
    evidence = []
    module_id = task.get("module_id")
    if not module_id:
        return evidence
    for name in task.get("completion_reports", []):
        final = Path(root) / name
        for path in sorted(final.parent.glob(final.name + ".tmp.*")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(row, dict) or row.get("module_id") != module_id:
                continue
            if row.get("technical_status") not in {"failed", "incomplete"}:
                continue
            issues = row.get("issues", [])
            if not isinstance(issues, list) or not any(
                isinstance(issue, dict) and issue.get("severity") == "error" for issue in issues
            ):
                continue
            evidence.append({"path": str(path), "technical_status": row["technical_status"],
                             "issues": issues, "acceptance": False})
    return evidence


def record_slurm_event(root, script, event, attempt, exit_code):
    """Bind a generated Slurm wrapper event to its prepared task contract."""
    source = Path(root) / "local-execution-plan.json"
    if not source.is_file():
        return  # Older standalone wrappers retain their JSONL event log.
    plan = json.loads(source.read_text(encoding="utf-8"))
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID") or None
    matches = [task for phase in plan.get("phases", []) for task in phase.get("tasks", [])
               if Path(task["script"]).name == Path(script).name
               and (str(task.get("array_task_id")) if task.get("array_task_id") is not None else None) == array_id]
    if len(matches) != 1:
        raise ValueError(f"cannot bind Slurm event to one prepared task: {script}, array {array_id}")
    if event == "started":
        from .execution_adapters import validate_worker_projects
        try:
            validate_worker_projects(Path(root), {"phases": [{"tasks": matches}]}, allow_future_inputs=False)
        except ValueError as exc:
            persist_task_status(root, matches[0], "slurm-" + os.environ.get("SLURM_JOB_ID", "unknown"), {
                "status": "failed", "attempt_number": int(attempt), "exit_code": 70,
                "event": "worker_preflight_failed", "error": str(exc),
            })
            raise
    status = {"started": "running", "complete": "complete", "failed": "failed",
              "output_contract_failed": "failed", "timeout_signal": "timed_out",
              "requeue_requested": "queued"}[event]
    persist_task_status(root, matches[0], "slurm-" + os.environ.get("SLURM_JOB_ID", "unknown"), {
        "status": status, "attempt_number": int(attempt), "exit_code": int(exit_code),
        "event": event, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })


if __name__ == "__main__":
    import sys
    record_slurm_event(*sys.argv[1:])
