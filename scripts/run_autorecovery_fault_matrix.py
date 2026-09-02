#!/usr/bin/env python3
"""Exercise bounded local task recovery against controlled failures."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from salsbury_md_analysis.analysis_config import default_analysis_config
from salsbury_md_analysis.execution_adapters import run_local_workflow


COMPLETE_REPORT = "{'technical_status':'complete'}"


def _task(task_id: str, script: str, report: str, dependencies=()):
    return {
        "task_id": task_id,
        "depends_on_task_ids": list(dependencies),
        "wait_for_task_ids": [],
        "script": script,
        "array_task_id": None,
        "cpu_slots": 1,
        "requested_memory_gib": 1,
        "requested_wall_minutes": 0.001 if task_id == "timeout" else 1,
        "completion_reports": [report],
    }


def _write_worker(root: Path, name: str, body: str) -> None:
    (root / name).write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")


def _status_rows(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(row["task_id"]): row
        for phase in report["phase_reports"]
        for row in phase["tasks"]
    }


def _attempt_summary(row: dict[str, object]) -> dict[str, object]:
    attempts = row.get("attempts", [])
    return {
        "status": row["status"],
        "attempt_count": row.get("attempt_count", 0),
        "attempt_statuses": [attempt["status"] for attempt in attempts],
        "exit_codes": [attempt["exit_code"] for attempt in attempts],
        "child_exit_codes": [attempt.get("child_exit_code") for attempt in attempts],
        "completion_reports_valid": [
            attempt.get("completion_reports_valid") for attempt in attempts
        ],
    }


def run_matrix() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "recovery-on"
        root.mkdir()
        (root / "logs").mkdir()
        valid = "printf '%s\\n' '{\"technical_status\":\"complete\"}'"
        _write_worker(root, "nonzero.sh", (
            "if [[ ! -f nonzero.once ]]; then touch nonzero.once; exit 7; fi\n"
            f"{valid} > nonzero.json\n"
        ))
        _write_worker(root, "timeout.sh", (
            "if [[ ! -f timeout.once ]]; then touch timeout.once; sleep 1; fi\n"
            f"{valid} > timeout.json\n"
        ))
        _write_worker(root, "oom.sh", (
            "if [[ ! -f oom.once ]]; then touch oom.once; exit 137; fi\n"
            f"{valid} > oom.json\n"
        ))
        _write_worker(root, "missing.sh", (
            "if [[ ! -f missing.once ]]; then touch missing.once; exit 0; fi\n"
            f"{valid} > missing.json\n"
        ))
        _write_worker(root, "corrupt.sh", (
            "if [[ ! -f corrupt.once ]]; then\n"
            "  touch corrupt.once\n"
            "  printf 'not-json\\n' > corrupt.json\n"
            "  exit 0\n"
            "fi\n"
            f"{valid} > corrupt.json\n"
        ))
        _write_worker(root, "persistent.sh", "exit 9\n")
        _write_worker(root, "recovered-dependent.sh", (
            f"{valid} > recovered-dependent.json\n"
        ))
        _write_worker(root, "failed-dependent.sh", (
            f"{valid} > failed-dependent.json\n"
        ))
        plan = {
            "local_execution_plan_schema": "salsbury-local-execution-plan-v6",
            "dependency_model": "task_dag_v1",
            "maximum_parallel_cpus": 6,
            "maximum_parallel_memory_gib": 6,
            "maximum_campaign_wall_hours": 1,
            "autorecovery": True,
            "maximum_task_attempts": 2,
            "phases": [
                {"phase_id": "faults", "tasks": [
                    _task("nonzero", "nonzero.sh", "nonzero.json"),
                    _task("timeout", "timeout.sh", "timeout.json"),
                    _task("oom", "oom.sh", "oom.json"),
                    _task("missing", "missing.sh", "missing.json"),
                    _task("corrupt", "corrupt.sh", "corrupt.json"),
                    _task("persistent", "persistent.sh", "persistent.json"),
                ]},
                {"phase_id": "dependencies", "tasks": [
                    _task(
                        "recovered-dependent", "recovered-dependent.sh",
                        "recovered-dependent.json", ("nonzero",),
                    ),
                    _task(
                        "failed-dependent", "failed-dependent.sh",
                        "failed-dependent.json", ("persistent",),
                    ),
                ]},
            ],
        }
        (root / "local-execution-plan.json").write_text(
            json.dumps(plan), encoding="utf-8"
        )
        enabled_report = run_local_workflow(root)
        enabled = _status_rows(enabled_report)

        off_root = Path(temporary) / "recovery-off"
        off_root.mkdir()
        (off_root / "logs").mkdir()
        _write_worker(off_root, "optout.sh", (
            "if [[ ! -f optout.once ]]; then touch optout.once; exit 7; fi\n"
            f"{valid} > optout.json\n"
        ))
        off_plan = {
            "local_execution_plan_schema": "salsbury-local-execution-plan-v6",
            "dependency_model": "task_dag_v1",
            "maximum_parallel_cpus": 1,
            "maximum_parallel_memory_gib": 1,
            "maximum_campaign_wall_hours": 1,
            "autorecovery": False,
            "maximum_task_attempts": 2,
            "phases": [{"phase_id": "optout", "tasks": [
                _task("optout", "optout.sh", "optout.json")
            ]}],
        }
        (off_root / "local-execution-plan.json").write_text(
            json.dumps(off_plan), encoding="utf-8"
        )
        optout_report = run_local_workflow(off_root)
        optout = _status_rows(optout_report)["optout"]

        for task_id in ("nonzero", "timeout", "oom", "missing", "corrupt"):
            if enabled[task_id]["status"] != "recovered_complete":
                raise RuntimeError(f"{task_id} did not recover")
        if enabled["persistent"]["status"] != "failed":
            raise RuntimeError("persistent failure did not exhaust its attempts")
        if enabled["failed-dependent"]["status"] != "skipped_dependency":
            raise RuntimeError("failed prerequisite released its dependent")
        if enabled["recovered-dependent"]["status"] != "complete":
            raise RuntimeError("recovered prerequisite did not release its dependent")
        if optout["status"] != "failed" or optout.get("attempt_count") != 1:
            raise RuntimeError("explicit recovery opt-out was not honored")

        defaults = default_analysis_config([], [])["execution"]
        return {
            "acceptance_schema": "salsbury-autorecovery-fault-matrix-v1",
            "technical_status": "complete",
            "scientific_status": "not applicable; execution-control test only",
            "default_policy": {
                "autorecovery": defaults["autorecovery"],
                "maximum_task_attempts": defaults["maximum_task_attempts"],
            },
            "failure_policy": (
                "retry only the failed task within the original campaign and "
                "resource limits; require a valid declared completion report "
                "before releasing success-dependent work"
            ),
            "faults": {
                task_id: _attempt_summary(enabled[task_id])
                for task_id in (
                    "nonzero", "timeout", "oom", "missing", "corrupt",
                    "persistent", "recovered-dependent", "failed-dependent",
                )
            },
            "optout": _attempt_summary(optout),
            "all_expected_outcomes_observed": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_matrix()
    destination = args.output.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "technical_status": result["technical_status"],
        "output": str(destination),
        "fault_count": len(result["faults"]),
        "all_expected_outcomes_observed": result[
            "all_expected_outcomes_observed"
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
