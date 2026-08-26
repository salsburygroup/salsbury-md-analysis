"""Portable local and Slurm execution adapters for generated workflows."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .analysis_config import COMMAND_MODULES
from .manifests import load_json


class ExecutionAdapterError(ValueError):
    """Raised when an execution profile or local workflow is unsafe."""


_PROFILE_SCHEMA = "salsbury-slurm-profile-v1"
_PARTITION_ROLES = {
    "default", "preflight", "coordinate_cache", "analysis",
    "conformational", "finalizer", "large_memory", "long_wall",
}
_PROFILE_FIELDS = {
    "slurm_profile_schema", "profile_id", "cluster_name", "submit_command",
    "status_command", "cancel_command", "account", "unix_group", "qos",
    "partitions", "partition_maximum_wall_minutes", "environment", "paths", "resource_policy",
    "additional_sbatch_directives",
}
_RESOURCE_POLICY_DEFAULTS = {
    "minimum_wall_minutes": 30.0,
    "walltime_safety_factor": 1.5,
    "walltime_overhead_minutes": 15.0,
    "minimum_memory_gib": 2.0,
    "memory_safety_factor": 1.5,
    "memory_overhead_gib": 1.0,
    "large_memory_threshold_gib": 96.0,
}


def _active_python_executable() -> str:
    """Return the active interpreter path without escaping a virtual environment."""

    return str(Path(os.sys.executable).absolute())


def _plain_string(value: object, label: str, *, nullable: bool = False) -> Optional[str]:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\x00" in value:
        raise ExecutionAdapterError(f"{label} must be a nonempty single-line string")
    return value.strip()


def load_slurm_profile(path: Path) -> Dict[str, object]:
    """Load and strictly validate a portable Slurm cluster profile."""

    source = path.expanduser().resolve(strict=True)
    profile = load_json(source)
    if not isinstance(profile, dict):
        raise ExecutionAdapterError("Slurm profile must be a JSON object")
    unknown = sorted(set(profile).difference(_PROFILE_FIELDS))
    if unknown:
        raise ExecutionAdapterError(
            "Slurm profile has unknown fields: " + ", ".join(unknown)
        )
    if profile.get("slurm_profile_schema") != _PROFILE_SCHEMA:
        raise ExecutionAdapterError(
            f"slurm_profile_schema must be {_PROFILE_SCHEMA}"
        )
    normalized: Dict[str, object] = {
        "slurm_profile_schema": _PROFILE_SCHEMA,
        "profile_id": _plain_string(profile.get("profile_id"), "profile_id"),
        "cluster_name": _plain_string(
            profile.get("cluster_name"), "cluster_name"
        ),
    }
    for field, default in (
        ("submit_command", "sbatch"),
        ("status_command", "squeue"),
        ("cancel_command", "scancel"),
    ):
        command = _plain_string(profile.get(field, default), field)
        assert command is not None
        if any(character.isspace() for character in command):
            raise ExecutionAdapterError(
                f"{field} must be one executable name or absolute path"
            )
        normalized[field] = command
    for field in ("account", "unix_group", "qos"):
        normalized[field] = _plain_string(
            profile.get(field), field, nullable=True
        )

    partitions = profile.get("partitions", {})
    if not isinstance(partitions, dict) or set(partitions).difference(_PARTITION_ROLES):
        raise ExecutionAdapterError("Slurm partitions mapping is invalid")
    normalized["partitions"] = {
        role: _plain_string(partitions.get(role), f"partitions.{role}", nullable=True)
        for role in sorted(_PARTITION_ROLES)
    }

    partition_limits = profile.get("partition_maximum_wall_minutes", {})
    if not isinstance(partition_limits, dict):
        raise ExecutionAdapterError(
            "partition_maximum_wall_minutes must be an object"
        )
    configured_partitions = {
        str(value) for value in normalized["partitions"].values() if value is not None
    }
    unknown_limit_partitions = sorted(
        set(partition_limits).difference(configured_partitions)
    )
    if unknown_limit_partitions:
        raise ExecutionAdapterError(
            "partition_maximum_wall_minutes contains unconfigured partitions: "
            + ", ".join(unknown_limit_partitions)
        )
    checked_partition_limits: Dict[str, float] = {}
    for partition_name, value in partition_limits.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ExecutionAdapterError(
                "partition_maximum_wall_minutes values must be finite and positive"
            )
        checked_partition_limits[str(partition_name)] = float(value)
    normalized["partition_maximum_wall_minutes"] = checked_partition_limits

    environment = profile.get("environment", {})
    allowed_environment = {
        "python_executable", "package_root", "setup_commands", "variables", "umask"
    }
    if not isinstance(environment, dict) or set(environment).difference(allowed_environment):
        raise ExecutionAdapterError("Slurm environment mapping is invalid")
    setup_commands = environment.get("setup_commands", [])
    if not isinstance(setup_commands, list):
        raise ExecutionAdapterError("environment.setup_commands must be a list")
    checked_commands = [
        _plain_string(value, "environment.setup_commands entry")
        for value in setup_commands
    ]
    variables = environment.get("variables", {})
    if not isinstance(variables, dict):
        raise ExecutionAdapterError("environment.variables must be an object")
    checked_variables: Dict[str, str] = {}
    for name, value in variables.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ExecutionAdapterError("environment variable names must be shell identifiers")
        checked = _plain_string(value, f"environment.variables.{name}")
        assert checked is not None
        checked_variables[name] = checked
    umask = _plain_string(environment.get("umask", "0022"), "environment.umask")
    assert umask is not None
    if not re.fullmatch(r"0?[0-7]{3}", umask):
        raise ExecutionAdapterError("environment.umask must be a three- or four-digit octal mask")
    normalized["environment"] = {
        "python_executable": _plain_string(
            environment.get("python_executable"),
            "environment.python_executable", nullable=True,
        ),
        "package_root": _plain_string(
            environment.get("package_root"),
            "environment.package_root", nullable=True,
        ),
        "setup_commands": checked_commands,
        "variables": checked_variables,
        "umask": umask,
    }

    paths = profile.get("paths", {})
    allowed_paths = {"group_storage_root", "scratch_root", "allowed_output_roots"}
    if not isinstance(paths, dict) or set(paths).difference(allowed_paths):
        raise ExecutionAdapterError("Slurm paths mapping is invalid")
    allowed_roots = paths.get("allowed_output_roots", [])
    if not isinstance(allowed_roots, list):
        raise ExecutionAdapterError("paths.allowed_output_roots must be a list")
    normalized["paths"] = {
        "group_storage_root": _plain_string(
            paths.get("group_storage_root"), "paths.group_storage_root", nullable=True
        ),
        "scratch_root": _plain_string(
            paths.get("scratch_root"), "paths.scratch_root", nullable=True
        ),
        "allowed_output_roots": [
            _plain_string(value, "paths.allowed_output_roots entry")
            for value in allowed_roots
        ],
    }

    policy = profile.get("resource_policy", {})
    allowed_policy = {
        "minimum_wall_minutes", "walltime_safety_factor",
        "walltime_overhead_minutes", "minimum_memory_gib",
        "memory_safety_factor", "memory_overhead_gib",
        "large_memory_threshold_gib",
    }
    if not isinstance(policy, dict) or set(policy).difference(allowed_policy):
        raise ExecutionAdapterError("Slurm resource_policy mapping is invalid")
    checked_policy: Dict[str, float] = {}
    for name, default in _RESOURCE_POLICY_DEFAULTS.items():
        value = policy.get(name, default)
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < 0.0
        ):
            raise ExecutionAdapterError(f"resource_policy.{name} must be finite and nonnegative")
        checked_policy[name] = float(value)
    normalized["resource_policy"] = checked_policy

    directives = profile.get("additional_sbatch_directives", [])
    if not isinstance(directives, list):
        raise ExecutionAdapterError("additional_sbatch_directives must be a list")
    checked_directives = []
    for directive in directives:
        checked = _plain_string(directive, "additional_sbatch_directives entry")
        assert checked is not None
        if not checked.startswith("#SBATCH --"):
            raise ExecutionAdapterError(
                "additional Slurm directives must start with '#SBATCH --'"
            )
        if re.match(
            r"#SBATCH --(?:account|partition|qos|time|mem|cpus-per-task)(?:=|\s)",
            checked,
        ):
            raise ExecutionAdapterError(
                "additional Slurm directives may not override managed resources"
            )
        checked_directives.append(checked)
    normalized["additional_sbatch_directives"] = checked_directives
    normalized["source_path"] = str(source)
    return normalized


def _script_role(name: str) -> str:
    if "coordinate_cache" in name:
        return "coordinate_cache"
    if "preflight" in name:
        return "preflight"
    if "finalize" in name:
        return "finalizer"
    if name.startswith("run_view_"):
        return "conformational"
    return "analysis"


def _profile_preamble(profile: Mapping[str, object], profile_path: Path) -> str:
    environment = profile["environment"]
    paths = profile["paths"]
    assert isinstance(environment, Mapping)
    assert isinstance(paths, Mapping)
    lines = [
        f"umask {environment['umask']}",
        f"export SALSBURY_SLURM_PROFILE={shlex.quote(str(profile_path))}",
        f"export SALSBURY_SLURM_CLUSTER={shlex.quote(str(profile['cluster_name']))}",
    ]
    if profile.get("unix_group"):
        lines.append(
            f"export SALSBURY_UNIX_GROUP={shlex.quote(str(profile['unix_group']))}"
        )
    if paths.get("group_storage_root"):
        lines.append(
            "export SALSBURY_GROUP_STORAGE_ROOT="
            + shlex.quote(str(paths["group_storage_root"]))
        )
    if paths.get("scratch_root"):
        lines.append(
            "export SALSBURY_SCRATCH_ROOT="
            + shlex.quote(str(paths["scratch_root"]))
        )
    variables = environment["variables"]
    assert isinstance(variables, Mapping)
    lines.extend(
        f"export {name}={shlex.quote(str(value))}"
        for name, value in sorted(variables.items())
    )
    setup_commands = environment["setup_commands"]
    assert isinstance(setup_commands, Sequence)
    lines.extend(str(value) for value in setup_commands)
    return "\n".join(lines)


def _bash_array_values(path: Path, name: str) -> List[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}=\(\n(.*?)^\)\n",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ExecutionAdapterError(f"cannot determine {name} values for {path.name}")
    values: List[str] = []
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        fields = shlex.split(line)
        if len(fields) != 1:
            raise ExecutionAdapterError(
                f"{path.name} has an ambiguous {name} entry"
            )
        values.append(fields[0])
    if not values:
        raise ExecutionAdapterError(f"{path.name} has no {name} values")
    return values


def _existing_wall_minutes(path: Path) -> float:
    match = re.search(
        r"^#SBATCH --time=(?:(\d+)-)?(\d+):(\d+):(\d+)\s*$",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if not match:
        return 30.0
    days, hours, minutes, seconds = (int(value or 0) for value in match.groups())
    return days * 1440.0 + hours * 60.0 + minutes + seconds / 60.0


def _existing_memory_gib(path: Path) -> float:
    match = re.search(
        r"^#SBATCH --mem=([0-9]+(?:\.[0-9]+)?)([KMGT]?)\s*$",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if not match:
        return 2.0
    value = float(match.group(1))
    scale = {"": 1.0 / 1024.0, "K": 1.0 / (1024.0 ** 2), "M": 1.0 / 1024.0,
             "G": 1.0, "T": 1024.0}
    return value * scale[match.group(2).upper()]


def _planner_rows(root: Path) -> List[Dict[str, object]]:
    path = root / "campaign-resource-plan.json"
    if not path.is_file():
        return []
    document = load_json(path)
    rows = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ExecutionAdapterError("campaign-resource-plan.json has invalid tasks")
    return [dict(row) for row in rows]


def _task_planner_rows(
    root: Path,
    task: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> List[Mapping[str, object]]:
    script = str(task["script"])
    if script == "run_coordinate_cache.slurm":
        return [row for row in rows if row.get("task_id") == "preprocessing:coordinate_cache"]
    array_id = task.get("array_task_id")
    if array_id is None:
        return []
    path = root / script
    commands = _bash_array_values(path, "COMMANDS")
    index = int(array_id)
    if index < 0 or index >= len(commands):
        raise ExecutionAdapterError(f"array index {index} is outside {script}")
    module_id = COMMAND_MODULES.get(commands[index])
    if module_id is None:
        raise ExecutionAdapterError(
            f"generated command {commands[index]!r} has no module mapping"
        )
    view_match = re.fullmatch(r"run_view_(.+)_stage_\d+\.slurm", script)
    if view_match:
        workflow_id = view_match.group(1)
        return [
            row for row in rows
            if row.get("workflow_id") == workflow_id and row.get("module_id") == module_id
        ]
    if script.startswith("run_automatic_context_stage_"):
        projects = _bash_array_values(path, "PROJECTS")
        if index >= len(projects):
            raise ExecutionAdapterError(f"PROJECTS and COMMANDS widths differ in {script}")
        name = Path(projects[index]).name
        match = re.fullmatch(r"project-(.+)\.json", name)
        if not match:
            raise ExecutionAdapterError(f"automatic-context project name is invalid: {name}")
        workflow_id = match.group(1)
        return [
            row for row in rows
            if row.get("workflow_id") == workflow_id and row.get("module_id") == module_id
        ]
    return [
        row for row in rows
        if row.get("module_id") == module_id
        and str(row.get("task_id", "")).startswith(("direct:", "base:"))
    ]


def _enrich_task_resources(
    root: Path,
    task: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    execution: Mapping[str, object],
    policy: Mapping[str, object],
) -> Dict[str, object]:
    enriched = dict(task)
    matched = _task_planner_rows(root, task, rows)
    path = root / str(task["script"])
    maximum_hours = float(execution["maximum_hours_per_cpu"])
    maximum_memory = float(execution["maximum_memory_gib"])
    if matched:
        try:
            wall_hours = sum(
                float(row["estimated_wall_hours_at_effective_cpu_cap"])
                for row in matched
            )
            memory_gib = max(
                float(row["estimated_peak_memory_gib_at_selected_observations"])
                for row in matched
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutionAdapterError(
                f"planner estimates are incomplete for {path.name}"
            ) from exc
        if not math.isfinite(wall_hours) or wall_hours < 0.0:
            raise ExecutionAdapterError(f"planner wall estimate is invalid for {path.name}")
        if not math.isfinite(memory_gib) or memory_gib <= 0.0:
            raise ExecutionAdapterError(f"planner memory estimate is invalid for {path.name}")
        if wall_hours > maximum_hours + 1e-9:
            raise ExecutionAdapterError(
                f"planner task exceeds the campaign wall limit: {path.name}"
            )
        if memory_gib > maximum_memory + 1e-9:
            raise ExecutionAdapterError(
                f"planner task exceeds the campaign memory limit: {path.name}"
            )
        safe_minutes = math.ceil(
            wall_hours * 60.0 * float(policy["walltime_safety_factor"])
            + float(policy["walltime_overhead_minutes"])
        )
        requested_wall_minutes = min(
            maximum_hours * 60.0,
            max(float(policy["minimum_wall_minutes"]), float(safe_minutes)),
        )
        safe_memory = math.ceil(
            memory_gib * float(policy["memory_safety_factor"])
            + float(policy["memory_overhead_gib"])
        )
        requested_memory_gib = min(
            maximum_memory,
            max(float(policy["minimum_memory_gib"]), float(safe_memory)),
        )
        planner_task_ids = [str(row["task_id"]) for row in matched]
        source = "campaign_planner_with_profile_safety_margin"
    else:
        wall_hours = _existing_wall_minutes(path) / 60.0
        memory_gib = _existing_memory_gib(path)
        requested_wall_minutes = min(maximum_hours * 60.0, wall_hours * 60.0)
        requested_memory_gib = min(maximum_memory, memory_gib)
        planner_task_ids = []
        source = "generated_worker_static_request_no_planner_row"
    enriched.update({
        "planner_task_ids": planner_task_ids,
        "planned_wall_hours": wall_hours,
        "planned_peak_memory_gib": memory_gib,
        "requested_wall_minutes": requested_wall_minutes,
        "requested_memory_gib": requested_memory_gib,
        "resource_request_source": source,
        "wall_request_limited_by_campaign_cap": (
            matched and requested_wall_minutes + 1e-9 < max(
                float(policy["minimum_wall_minutes"]),
                math.ceil(
                    wall_hours * 60.0 * float(policy["walltime_safety_factor"])
                    + float(policy["walltime_overhead_minutes"])
                ),
            )
        ),
        "memory_request_limited_by_campaign_cap": (
            matched and requested_memory_gib + 1e-9 < max(
                float(policy["minimum_memory_gib"]),
                math.ceil(
                    memory_gib * float(policy["memory_safety_factor"])
                    + float(policy["memory_overhead_gib"])
                ),
            )
        ),
    })
    return enriched


def _format_slurm_time(minutes: float) -> str:
    total_seconds = max(60, int(math.ceil(minutes * 60.0)))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minute, second = divmod(remainder, 60)
    prefix = f"{days}-" if days else ""
    return f"{prefix}{hours:02d}:{minute:02d}:{second:02d}"


def _replace_sbatch(text: str, name: str, value: str) -> str:
    pattern = rf"^#SBATCH --{re.escape(name)}=.*$"
    replacement = f"#SBATCH --{name}={value}"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)
    first_newline = text.find("\n")
    return text[: first_newline + 1] + replacement + "\n" + text[first_newline + 1 :]


def _script_resource_requests(plan: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    requests: Dict[str, Dict[str, object]] = {}
    for phase in plan.get("phases", []):
        for task in phase.get("tasks", []):
            script = str(task["script"])
            row = requests.setdefault(script, {
                "requested_wall_minutes": 0.0,
                "requested_memory_gib": 0.0,
                "planner_task_ids": [],
                "aggregation": "maximum across concurrently schedulable array elements",
            })
            row["requested_wall_minutes"] = max(
                float(row["requested_wall_minutes"]),
                float(task["requested_wall_minutes"]),
            )
            row["requested_memory_gib"] = max(
                float(row["requested_memory_gib"]),
                float(task["requested_memory_gib"]),
            )
            row["planner_task_ids"].extend(task.get("planner_task_ids", []))
    for row in requests.values():
        row["planner_task_ids"] = sorted(set(row["planner_task_ids"]))
    return requests


def _task_resource_requests(plan: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for phase in plan.get("phases", []):
        for task in phase.get("tasks", []):
            rows.append({
                "phase_id": phase["phase_id"],
                "script": task["script"],
                "array_task_id": task.get("array_task_id"),
                "cpu_slots": task["cpu_slots"],
                "planner_task_ids": list(task.get("planner_task_ids", [])),
                "planned_wall_hours": task["planned_wall_hours"],
                "planned_peak_memory_gib": task["planned_peak_memory_gib"],
                "requested_wall_minutes": task["requested_wall_minutes"],
                "requested_memory_gib": task["requested_memory_gib"],
                "resource_request_source": task["resource_request_source"],
                "wall_request_limited_by_campaign_cap": task[
                    "wall_request_limited_by_campaign_cap"
                ],
                "memory_request_limited_by_campaign_cap": task[
                    "memory_request_limited_by_campaign_cap"
                ],
            })
    return rows


def apply_slurm_profile(
    root: Path,
    profile: Mapping[str, object],
    execution_plan: Mapping[str, object],
) -> Dict[str, object]:
    """Apply planner-derived scheduler requests and cluster environment settings."""

    profile_path = root / "slurm-profile.json"
    environment = profile["environment"]
    partitions = profile["partitions"]
    assert isinstance(environment, Mapping)
    assert isinstance(partitions, Mapping)
    preamble = _profile_preamble(profile, profile_path)
    account = profile.get("account")
    qos = profile.get("qos")
    additional = profile["additional_sbatch_directives"]
    assert isinstance(additional, Sequence)
    resource_policy = profile["resource_policy"]
    assert isinstance(resource_policy, Mapping)
    script_requests = _script_resource_requests(execution_plan)
    partition_limits = profile["partition_maximum_wall_minutes"]
    assert isinstance(partition_limits, Mapping)

    for path in sorted(root.glob("*.slurm")):
        text = path.read_text(encoding="utf-8")
        role = _script_role(path.name)
        request = script_requests.get(path.name, {
            "requested_wall_minutes": _existing_wall_minutes(path),
            "requested_memory_gib": _existing_memory_gib(path),
            "planner_task_ids": [],
            "aggregation": "static worker fallback",
        })
        is_large = (
            float(request["requested_memory_gib"])
            >= float(resource_policy["large_memory_threshold_gib"])
            and bool(partitions.get("large_memory"))
        )
        if is_large:
            role = "large_memory"
        partition = partitions.get(role) or partitions.get("default")
        requested_wall_minutes = float(request["requested_wall_minutes"])
        partition_limit = (
            None if partition is None else partition_limits.get(str(partition))
        )
        long_wall_routed = False
        if (
            partition_limit is not None
            and requested_wall_minutes > float(partition_limit)
        ):
            long_wall_partition = partitions.get("long_wall")
            if not long_wall_partition:
                raise ExecutionAdapterError(
                    f"{path.name} requests {requested_wall_minutes:g} minutes, "
                    f"exceeding partition {partition!r} limit "
                    f"{float(partition_limit):g}, but partitions.long_wall is unset"
                )
            long_wall_limit = partition_limits.get(str(long_wall_partition))
            if (
                long_wall_limit is not None
                and requested_wall_minutes > float(long_wall_limit)
            ):
                raise ExecutionAdapterError(
                    f"{path.name} requests {requested_wall_minutes:g} minutes, "
                    f"exceeding long-wall partition {long_wall_partition!r} limit "
                    f"{float(long_wall_limit):g}"
                )
            role = "long_wall"
            partition = long_wall_partition
            long_wall_routed = True
        directives = []
        if account:
            directives.append(f"#SBATCH --account={account}")
        if partition:
            directives.append(f"#SBATCH --partition={partition}")
        if qos:
            directives.append(f"#SBATCH --qos={qos}")
        directives.extend(str(value) for value in additional)
        if directives:
            first_newline = text.find("\n")
            text = text[: first_newline + 1] + "\n".join(directives) + "\n" + text[first_newline + 1 :]
        text = _replace_sbatch(
            text, "time", _format_slurm_time(float(request["requested_wall_minutes"]))
        )
        text = _replace_sbatch(
            text, "mem", f"{int(math.ceil(float(request['requested_memory_gib'])))}G"
        )
        text = text.replace("set -euo pipefail\n", f"set -euo pipefail\n{preamble}\n", 1)
        python_path = environment.get("python_executable")
        package_root = environment.get("package_root")
        if python_path:
            text = re.sub(
                r"^PYTHON_DEFAULT=.*$", f"PYTHON_DEFAULT={shlex.quote(str(python_path))}",
                text, flags=re.MULTILINE,
            )
        if package_root:
            text = re.sub(
                r"^PACKAGE_ROOT_DEFAULT=.*$",
                f"PACKAGE_ROOT_DEFAULT={shlex.quote(str(package_root))}",
                text, flags=re.MULTILINE,
            )
        path.write_text(text, encoding="utf-8")
        request["selected_partition_role"] = role
        request["selected_partition"] = partition
        request["long_wall_routed"] = long_wall_routed
        request["selected_partition_maximum_wall_minutes"] = (
            None if partition is None else partition_limits.get(str(partition))
        )
        request["slurm_time"] = _format_slurm_time(
            float(request["requested_wall_minutes"])
        )
        request["slurm_memory"] = f"{int(math.ceil(float(request['requested_memory_gib'])))}G"

    submit_command = str(profile["submit_command"])
    for path in sorted(root.glob("submit*.sh")):
        text = path.read_text(encoding="utf-8")
        text = text.replace("set -euo pipefail\n", f"set -euo pipefail\n{preamble}\n", 1)
        text = re.sub(r"\bsbatch\b", submit_command, text)
        python_path = environment.get("python_executable")
        package_root = environment.get("package_root")
        if python_path:
            text = re.sub(
                r"^PYTHON_DEFAULT=.*$", f"PYTHON_DEFAULT={shlex.quote(str(python_path))}",
                text, flags=re.MULTILINE,
            )
        if package_root:
            text = re.sub(
                r"^PACKAGE_ROOT_DEFAULT=.*$",
                f"PACKAGE_ROOT_DEFAULT={shlex.quote(str(package_root))}",
                text, flags=re.MULTILINE,
            )
        path.write_text(text, encoding="utf-8")
    return {
        "scheduler_resource_requests_schema": "salsbury-scheduler-resource-requests-v1",
        "profile_id": profile["profile_id"],
        "cluster_name": profile["cluster_name"],
        "resource_policy": dict(resource_policy),
        "partition_maximum_wall_minutes": dict(partition_limits),
        "tasks": _task_resource_requests(execution_plan),
        "scripts": script_requests,
        "large_memory_routing": (
            "a whole worker array uses the large-memory role when the largest "
            "element request reaches the configured threshold"
        ),
        "long_wall_routing": (
            "a worker script is routed to the long-wall role when its requested "
            "wall time exceeds the configured limit of its preferred partition"
        ),
    }


def _command_count(path: Path) -> int:
    return len(_bash_array_values(path, "COMMANDS"))


def _cpu_slots(path: Path) -> int:
    match = re.search(
        r"^#SBATCH --cpus-per-task=(\d+)\s*$",
        path.read_text(encoding="utf-8"), flags=re.MULTILINE,
    )
    return int(match.group(1)) if match else 1


def _task(path: Path, array_id: Optional[int] = None) -> Dict[str, object]:
    return {
        "script": path.name,
        "array_task_id": array_id,
        "cpu_slots": _cpu_slots(path),
    }


def build_local_execution_plan(
    root: Path,
    execution: Mapping[str, object],
    reporting: Mapping[str, object],
    resource_policy: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Build one dependency/resource plan shared by local and Slurm adapters."""

    phases = []
    cache = root / "run_coordinate_cache.slurm"
    if cache.is_file():
        phases.append({"phase_id": "coordinate_cache", "tasks": [_task(cache)]})
    preflight = root / "run_preflight.slurm"
    if not preflight.is_file():
        raise ExecutionAdapterError("generated workflow lacks run_preflight.slurm")
    phases.append({"phase_id": "preflight", "tasks": [_task(preflight)]})

    staged: Dict[int, list[Dict[str, object]]] = {}
    for path in sorted(root.glob("run_stage_*_array.slurm")):
        match = re.fullmatch(r"run_stage_(\d+)_array\.slurm", path.name)
        assert match is not None
        staged.setdefault(int(match.group(1)), []).extend(
            _task(path, index) for index in range(_command_count(path))
        )
    for path in sorted(root.glob("run_automatic_context_stage_*_array.slurm")):
        match = re.fullmatch(r"run_automatic_context_stage_(\d+)_array\.slurm", path.name)
        assert match is not None
        staged.setdefault(int(match.group(1)), []).extend(
            _task(path, index) for index in range(_command_count(path))
        )
    for stage, tasks in sorted(staged.items()):
        phases.append({"phase_id": f"analysis_stage_{stage}", "tasks": tasks})

    view_preflights = [
        _task(path) for path in sorted(root.glob("run_view_preflight_*.slurm"))
    ]
    if view_preflights:
        phases.append({"phase_id": "conformational_view_preflights", "tasks": view_preflights})
    view_stages: Dict[int, list[Dict[str, object]]] = {}
    for path in sorted(root.glob("run_view_*_stage_*.slurm")):
        match = re.search(r"_stage_(\d+)\.slurm$", path.name)
        assert match is not None
        view_stages.setdefault(int(match.group(1)), []).extend(
            _task(path, index) for index in range(_command_count(path))
        )
    for stage, tasks in sorted(view_stages.items()):
        phases.append({"phase_id": f"conformational_view_stage_{stage}", "tasks": tasks})

    finalizer = root / "run_finalize_reporting.slurm"
    if not finalizer.is_file():
        raise ExecutionAdapterError("generated workflow lacks run_finalize_reporting.slurm")
    completion_reports = []
    if bool(reporting.get("resource_table_enabled")):
        completion_reports.append("final-resource-summary.json")
    if bool(reporting.get("finding_picker_enabled")):
        completion_reports.append("final-findings-summary.json")
    if bool(reporting.get("interactive_report_enabled")):
        completion_reports.extend([
            "final-interactive-report-summary.json",
            "interactive-report/manifest.json",
        ])
    if not completion_reports:
        completion_reports.append("final-reporting-disabled.json")
    final_task = _task(finalizer)
    final_task["completion_reports"] = completion_reports
    phases.append({"phase_id": "final_reporting", "tasks": [final_task]})
    maximum_cpus = int(execution["maximum_parallel_cpus"])
    if any(int(task["cpu_slots"]) > maximum_cpus for phase in phases for task in phase["tasks"]):
        raise ExecutionAdapterError("a local task requests more CPUs than the campaign limit")
    policy = dict(_RESOURCE_POLICY_DEFAULTS)
    if resource_policy is not None:
        policy.update(resource_policy)
    rows = _planner_rows(root)
    for phase in phases:
        phase["tasks"] = [
            _enrich_task_resources(root, task, rows, execution, policy)
            for task in phase["tasks"]
        ]
    return {
        "local_execution_plan_schema": "salsbury-local-execution-plan-v2",
        "maximum_parallel_cpus": maximum_cpus,
        "maximum_campaign_wall_hours": float(execution["maximum_hours_per_cpu"]),
        "maximum_parallel_memory_gib": float(execution["maximum_memory_gib"]),
        "resource_policy": policy,
        "phases": phases,
        "dependency_policy": (
            "phases are serial; tasks within one phase share atomic CPU and memory caps"
        ),
    }


def prepare_execution_artifacts(
    root: Path, analysis_config: Mapping[str, object]
) -> Dict[str, object]:
    """Generate both launchers and activate the configured execution adapter."""

    execution = analysis_config["execution"]
    reporting = analysis_config["reporting"]
    assert isinstance(execution, Mapping)
    assert isinstance(reporting, Mapping)
    adapter = str(execution.get("submission_adapter", "local"))
    if adapter == "unspecified":
        adapter = "local"
    profile = None
    if adapter == "slurm":
        profile_value = execution.get("slurm_profile")
        if not isinstance(profile_value, str) or not profile_value:
            raise ExecutionAdapterError(
                "execution.slurm_profile is required when submission_adapter is slurm"
            )
        profile = load_slurm_profile(Path(profile_value))
    plan = build_local_execution_plan(
        root,
        execution,
        reporting,
        None if profile is None else profile["resource_policy"],
    )
    (root / "local-execution-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    package_root = str(Path(__file__).resolve(strict=True).parents[1])
    # Keep the virtual-environment entry point rather than resolving its
    # symlink to a base interpreter that may not have the package dependencies.
    python = _active_python_executable()
    launcher = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={shlex.quote(str(root))}
PYTHON_DEFAULT={shlex.quote(python)}
PYTHON="${{SALSBURY_MD_ANALYSIS_PYTHON:-$PYTHON_DEFAULT}}"
PACKAGE_ROOT_DEFAULT={shlex.quote(package_root)}
PACKAGE_ROOT="${{SALSBURY_MD_ANALYSIS_PYTHONPATH:-$PACKAGE_ROOT_DEFAULT}}"
export PYTHONPATH="$PACKAGE_ROOT${{PYTHONPATH:+:$PYTHONPATH}}"
exec "$PYTHON" -m salsbury_md_analysis run-local-workflow "$ROOT"
"""
    (root / "run-local.sh").write_text(launcher, encoding="utf-8")
    os.chmod(root / "run-local.sh", 0o755)

    profile_id = None
    generated = ["local-execution-plan.json", "run-local.sh"]
    if adapter == "slurm":
        assert profile is not None
        profile_id = str(profile["profile_id"])
        retained = {key: value for key, value in profile.items() if key != "source_path"}
        (root / "slurm-profile.json").write_text(
            json.dumps(retained, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        scheduler_requests = apply_slurm_profile(root, retained, plan)
        (root / "scheduler-resource-requests.json").write_text(
            json.dumps(scheduler_requests, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        generated.extend(["slurm-profile.json", "scheduler-resource-requests.json"])
    metadata = {
        "execution_adapter_schema": "salsbury-execution-adapter-v1",
        "active_adapter": adapter,
        "slurm_profile_id": profile_id,
        "local_launcher": "run-local.sh",
        "slurm_launcher": "submit.sh",
        "shared_resource_plan": "local-execution-plan.json",
        "scheduler_resource_requests": (
            "scheduler-resource-requests.json" if adapter == "slurm" else None
        ),
        "local_resource_controls": [
            "maximum_parallel_cpus", "maximum_parallel_memory_gib",
            "per_task_wall_minutes", "complete_campaign_wall_deadline",
        ],
        "scientific_workflow_identity": (
            "both launchers execute the same generated worker scripts and output contracts"
        ),
    }
    (root / "execution-adapter.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generated.append("execution-adapter.json")
    return {
        "adapter": adapter,
        "generated_files": generated,
        "next_command": f"cd {root} && ./{'submit.sh' if adapter == 'slurm' else 'run-local.sh'}",
        "slurm_profile_id": profile_id,
    }


def _reports_complete(root: Path, names: Sequence[str]) -> bool:
    for name in names:
        path = root / name
        if not path.is_file():
            return False
        try:
            report = load_json(path)
        except (OSError, ValueError):
            return False
        if not isinstance(report, dict) or report.get("technical_status") != "complete":
            return False
    return True


class _ResourcePool:
    """Atomically reserve CPU and memory slots for local execution."""

    def __init__(self, cpu_capacity: int, memory_capacity_gib: float):
        self.cpu_capacity = cpu_capacity
        self.memory_capacity_gib = memory_capacity_gib
        self.available_cpus = cpu_capacity
        self.available_memory_gib = memory_capacity_gib
        self.condition = threading.Condition()

    def acquire(self, cpus: int, memory_gib: float) -> None:
        if cpus > self.cpu_capacity or memory_gib > self.memory_capacity_gib + 1e-9:
            raise ExecutionAdapterError(
                "a local task requests more CPU or memory than the campaign limit"
            )
        with self.condition:
            self.condition.wait_for(
                lambda: self.available_cpus >= cpus
                and self.available_memory_gib + 1e-9 >= memory_gib
            )
            self.available_cpus -= cpus
            self.available_memory_gib -= memory_gib

    def release(self, cpus: int, memory_gib: float) -> None:
        with self.condition:
            self.available_cpus += cpus
            self.available_memory_gib += memory_gib
            self.condition.notify_all()


def _run_local_task(
    root: Path,
    task: Mapping[str, object],
    phase_id: str,
    task_index: int,
    attempt_id: str,
    deadline: float,
    slots: _ResourcePool,
) -> Dict[str, object]:
    cpu_slots = int(task["cpu_slots"])
    memory_gib = float(task.get("requested_memory_gib", 1.0))
    wall_minutes = float(task.get("requested_wall_minutes", 30.0))
    completion_reports = task.get("completion_reports", [])
    if isinstance(completion_reports, list) and completion_reports and _reports_complete(
        root, [str(value) for value in completion_reports]
    ):
        return {
            "script": task["script"], "array_task_id": task.get("array_task_id"),
            "cpu_slots": cpu_slots, "requested_memory_gib": memory_gib,
            "requested_wall_minutes": wall_minutes,
            "status": "reused_complete", "exit_code": 0,
            "wall_seconds": 0.0,
        }
    slots.acquire(cpu_slots, memory_gib)
    try:
        script = root / str(task["script"])
        if not script.is_file() or script.parent != root:
            raise ExecutionAdapterError(f"local worker is missing or outside root: {script}")
        suffix = "single" if task.get("array_task_id") is None else str(task["array_task_id"])
        stem = f"{attempt_id}-{phase_id}-{task_index}-{suffix}"
        stdout_path = root / "logs" / f"{stem}.out"
        stderr_path = root / "logs" / f"{stem}.err"
        env = os.environ.copy()
        env.update({
            "SLURM_JOB_ID": stem,
            "SLURM_ARRAY_JOB_ID": f"{attempt_id}-{phase_id}",
            "SLURM_CPUS_PER_TASK": str(cpu_slots),
            "SLURM_MEM_PER_NODE": str(int(math.ceil(memory_gib * 1024.0))),
            "SLURM_TIMELIMIT": str(max(1, int(math.ceil(wall_minutes)))),
            "SLURM_CLUSTER_NAME": "local",
        })
        if task.get("array_task_id") is not None:
            env["SLURM_ARRAY_TASK_ID"] = str(task["array_task_id"])
        start = time.monotonic()
        timeout_seconds = min(deadline - start, wall_minutes * 60.0)
        if timeout_seconds <= 0.0:
            return {
                "script": task["script"], "array_task_id": task.get("array_task_id"),
                "cpu_slots": cpu_slots, "requested_memory_gib": memory_gib,
                "requested_wall_minutes": wall_minutes,
                "status": "timed_out", "exit_code": None,
                "wall_seconds": 0.0,
            }
        timed_out = False
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                ["bash", str(script)], cwd=root, env=env,
                stdout=stdout, stderr=stderr, start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    exit_code = process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    exit_code = process.wait()
        return {
            "script": task["script"], "array_task_id": task.get("array_task_id"),
            "cpu_slots": cpu_slots, "requested_memory_gib": memory_gib,
            "requested_wall_minutes": wall_minutes,
            "planner_task_ids": task.get("planner_task_ids", []),
            "status": "timed_out" if timed_out else ("complete" if exit_code == 0 else "failed"),
            "exit_code": exit_code,
            "wall_seconds": time.monotonic() - start,
            "stdout": str(stdout_path.relative_to(root)),
            "stderr": str(stderr_path.relative_to(root)),
        }
    finally:
        slots.release(cpu_slots, memory_gib)


def run_local_workflow(root: Path) -> Dict[str, object]:
    """Execute a generated workflow locally while respecting its CPU envelope."""

    resolved = root.expanduser().resolve(strict=True)
    plan_path = resolved / "local-execution-plan.json"
    plan = load_json(plan_path)
    accepted_schemas = {
        "salsbury-local-execution-plan-v1", "salsbury-local-execution-plan-v2"
    }
    if not isinstance(plan, dict) or plan.get("local_execution_plan_schema") not in accepted_schemas:
        raise ExecutionAdapterError("local execution plan is invalid")
    maximum_cpus = int(plan["maximum_parallel_cpus"])
    maximum_memory_gib = float(plan.get("maximum_parallel_memory_gib", 1.0e12))
    campaign_seconds = float(plan["maximum_campaign_wall_hours"]) * 3600.0
    if maximum_cpus <= 0 or maximum_memory_gib <= 0.0 or campaign_seconds <= 0.0:
        raise ExecutionAdapterError("local execution limits must be positive")
    (resolved / "logs").mkdir(exist_ok=True)
    status_directory = resolved / "local-execution-status"
    status_directory.mkdir(exist_ok=True)
    attempt_id = datetime.now(timezone.utc).strftime(
        "local-%Y%m%dT%H%M%S.%fZ"
    ) + f"-{os.getpid()}"
    phase_reports = []
    technical_status = "complete"
    slots = _ResourcePool(maximum_cpus, maximum_memory_gib)
    deadline = time.monotonic() + campaign_seconds
    for phase in plan.get("phases", []):
        phase_id = str(phase["phase_id"])
        tasks = phase["tasks"]
        if not isinstance(tasks, list) or not tasks:
            raise ExecutionAdapterError(f"local phase {phase_id} has no tasks")
        results = []
        with ThreadPoolExecutor(max_workers=min(len(tasks), maximum_cpus)) as executor:
            futures = {
                executor.submit(
                    _run_local_task, resolved, task, phase_id, index,
                    attempt_id, deadline, slots,
                ): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # preserve a machine-readable failed attempt
                    results.append({
                        "script": str(tasks[futures[future]].get("script", "unknown")),
                        "array_task_id": tasks[futures[future]].get("array_task_id"),
                        "cpu_slots": tasks[futures[future]].get("cpu_slots"),
                        "requested_memory_gib": tasks[futures[future]].get(
                            "requested_memory_gib", 1.0
                        ),
                        "requested_wall_minutes": tasks[futures[future]].get(
                            "requested_wall_minutes", 30.0
                        ),
                        "status": "failed",
                        "exit_code": None,
                        "error": str(exc),
                        "wall_seconds": 0.0,
                    })
        results.sort(key=lambda row: (str(row["script"]), str(row.get("array_task_id"))))
        phase_status = (
            "complete" if all(row["status"] in {"complete", "reused_complete"} for row in results)
            else "failed"
        )
        phase_reports.append({
            "phase_id": phase_id, "technical_status": phase_status, "tasks": results
        })
        if phase_status != "complete":
            technical_status = "failed"
            break
    report = {
        "local_execution_status_schema": "salsbury-local-execution-status-v1",
        "technical_status": technical_status,
        "scientific_status": "not evaluated",
        "attempt_id": attempt_id,
        "analysis_root": str(resolved),
        "maximum_parallel_cpus": maximum_cpus,
        "maximum_parallel_memory_gib": maximum_memory_gib,
        "phase_reports": phase_reports,
        "remaining_phases_not_run": [
            str(phase["phase_id"])
            for phase in plan.get("phases", [])[len(phase_reports):]
        ],
    }
    final = status_directory / f"{attempt_id}.json"
    temporary = status_directory / f".{attempt_id}.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, final)
    report["status_report"] = str(final)
    return report
