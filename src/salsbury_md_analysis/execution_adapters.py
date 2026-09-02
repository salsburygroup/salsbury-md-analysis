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
from .ensemble_parallelism import annotate_task_parallelism
from .manifests import load_json
from .resource_planning import ResourcePlanningError, pack_resource_lanes


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
    "partitions", "partition_maximum_wall_minutes", "partition_maximum_nodes",
    "environment", "paths", "resource_policy",
    "node_policy", "additional_sbatch_directives",
}
_RESOURCE_POLICY_DEFAULTS = {
    "minimum_wall_minutes": 30.0,
    "walltime_safety_factor": 1.0,
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


def _normalized_package_root(value: object) -> Optional[str]:
    """Resolve an accessible repository root to its importable ``src`` tree."""

    checked = _plain_string(
        value, "environment.package_root", nullable=True
    )
    if checked is None:
        return None
    candidate = Path(checked).expanduser()
    if (candidate / "salsbury_md_analysis").is_dir():
        return str(candidate.resolve())
    source_tree = candidate / "src"
    if (source_tree / "salsbury_md_analysis").is_dir():
        return str(source_tree.resolve())
    # Profiles are sometimes prepared on a host that cannot mount the target
    # cluster path.  Preserve such paths; the generated worker preflight still
    # verifies the import source before computation begins.
    return checked


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

    partition_node_limits = profile.get("partition_maximum_nodes", {})
    if not isinstance(partition_node_limits, dict):
        raise ExecutionAdapterError("partition_maximum_nodes must be an object")
    unknown_node_limit_partitions = sorted(
        set(partition_node_limits).difference(configured_partitions)
    )
    if unknown_node_limit_partitions:
        raise ExecutionAdapterError(
            "partition_maximum_nodes contains unconfigured partitions: "
            + ", ".join(unknown_node_limit_partitions)
        )
    checked_partition_node_limits: Dict[str, int] = {}
    for partition_name, value in partition_node_limits.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ExecutionAdapterError(
                "partition_maximum_nodes values must be positive integers"
            )
        checked_partition_node_limits[str(partition_name)] = int(value)
    normalized["partition_maximum_nodes"] = checked_partition_node_limits

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
        "package_root": _normalized_package_root(
            environment.get("package_root")
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
    if not math.isclose(
        checked_policy["walltime_safety_factor"], 1.0,
        rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise ExecutionAdapterError(
            "resource_policy.walltime_safety_factor must be 1.0 because the "
            "campaign planner already applies task-time uncertainty; use "
            "walltime_overhead_minutes for explicit scheduler-only overhead"
        )
    normalized["resource_policy"] = checked_policy

    node_policy = profile.get("node_policy", {})
    allowed_node_policy = {
        "cpus_per_node", "memory_gib_per_node", "maximum_nodes_per_campaign",
    }
    if (
        not isinstance(node_policy, dict)
        or set(node_policy).difference(allowed_node_policy)
    ):
        raise ExecutionAdapterError("Slurm node_policy mapping is invalid")
    cpus_per_node = node_policy.get("cpus_per_node")
    memory_gib_per_node = node_policy.get("memory_gib_per_node")
    maximum_nodes_per_campaign = node_policy.get("maximum_nodes_per_campaign")
    if (cpus_per_node is None) != (memory_gib_per_node is None):
        raise ExecutionAdapterError(
            "node_policy requires both cpus_per_node and memory_gib_per_node"
        )
    if cpus_per_node is not None and (
        isinstance(cpus_per_node, bool)
        or not isinstance(cpus_per_node, int)
        or cpus_per_node <= 0
    ):
        raise ExecutionAdapterError(
            "node_policy.cpus_per_node must be a positive integer"
        )
    if memory_gib_per_node is not None and (
        isinstance(memory_gib_per_node, bool)
        or not isinstance(memory_gib_per_node, (int, float))
        or not math.isfinite(float(memory_gib_per_node))
        or float(memory_gib_per_node) <= 0.0
    ):
        raise ExecutionAdapterError(
            "node_policy.memory_gib_per_node must be finite and positive"
        )
    if maximum_nodes_per_campaign is not None and (
        isinstance(maximum_nodes_per_campaign, bool)
        or not isinstance(maximum_nodes_per_campaign, int)
        or maximum_nodes_per_campaign <= 0
    ):
        raise ExecutionAdapterError(
            "node_policy.maximum_nodes_per_campaign must be a positive integer"
        )
    if maximum_nodes_per_campaign is not None and cpus_per_node is None:
        raise ExecutionAdapterError(
            "node_policy.maximum_nodes_per_campaign requires a node shape"
        )
    normalized["node_policy"] = {
        "cpus_per_node": cpus_per_node,
        "memory_gib_per_node": (
            None if memory_gib_per_node is None else float(memory_gib_per_node)
        ),
        "maximum_nodes_per_campaign": maximum_nodes_per_campaign,
    }

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
            r"#SBATCH --(?:account|partition|qos|time|mem|nodes|cpus-per-task)(?:=|\s)",
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
    if script == "run_finalize_reporting.slurm":
        text = (root / script).read_text(encoding="utf-8")
        final_modules = set()
        if "rmsf-permutation-from-report" in text:
            final_modules.add("rmsf_permutation_inference")
        return [row for row in rows if row.get("module_id") in final_modules]
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


def _structural_qc_replica_runtime_contract(
    root: Path,
    task: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    active_worker_count: int,
) -> Optional[Dict[str, object]]:
    """Validate that a planned replica-parallel QC task cannot run serially."""

    qc_rows = [
        row for row in rows
        if row.get("module_id") == "structural_integrity_qc"
        and row.get("parallel_execution_model") is not None
    ]
    if not qc_rows:
        return None
    # Hand-built adapter fixtures and legacy launchers may not have a generated
    # analysis configuration.  Current generic workflows always do.
    if not (root / "analysis-config.json").is_file():
        return None
    array_id = task.get("array_task_id")
    if array_id is None:
        raise ExecutionAdapterError(
            "replica-parallel structural QC must be an explicit worker task"
        )
    script = root / str(task["script"])
    projects = _bash_array_values(script, "PROJECTS")
    index = int(array_id)
    if index < 0 or index >= len(projects):
        raise ExecutionAdapterError(
            f"structural-QC project index {index} is outside {script.name}"
        )
    project_path = Path(projects[index]).expanduser()
    if not project_path.is_absolute():
        project_path = root / project_path
    if not project_path.is_file():
        raise ExecutionAdapterError(
            f"replica-parallel structural-QC project is missing: {project_path}"
        )
    project = load_json(project_path)
    definitions = project.get("definitions") if isinstance(project, dict) else None
    structural = (
        definitions.get("structural_qc")
        if isinstance(definitions, dict) else None
    )
    parallel = (
        structural.get("parallel_execution")
        if isinstance(structural, dict) else None
    )
    if not isinstance(parallel, dict) or parallel.get("enabled") is not True:
        raise ExecutionAdapterError(
            "planner selected replica-parallel structural QC, but its runtime "
            "project does not enable parallel_execution"
        )
    configured_workers = parallel.get("maximum_workers")
    if (
        isinstance(configured_workers, bool)
        or not isinstance(configured_workers, int)
        or configured_workers <= 0
    ):
        raise ExecutionAdapterError(
            "structural-QC runtime maximum_workers must be a positive integer"
        )
    declared_shards = max(int(row["parallel_worker_count"]) for row in qc_rows)
    planned_worker_cap = max(int(row["effective_cpu_cap"]) for row in qc_rows)
    if configured_workers != planned_worker_cap:
        raise ExecutionAdapterError(
            "structural-QC planner/runtime worker mismatch: planner cap "
            f"{planned_worker_cap}, runtime cap {configured_workers}"
        )
    if declared_shards < configured_workers:
        raise ExecutionAdapterError(
            "structural-QC runtime worker cap exceeds its replica shard count"
        )
    if active_worker_count > configured_workers:
        raise ExecutionAdapterError(
            "structural-QC scheduler worker count exceeds its runtime cap"
        )
    for field in (
        "coordinate_cache_system_manifest", "coordinate_cache_report",
    ):
        if not isinstance(parallel.get(field), str) or not parallel[field]:
            raise ExecutionAdapterError(
                f"structural-QC runtime contract lacks {field}"
            )
    return {
        "execution_model": "one_replica_shard_per_worker_v1",
        "runtime_project": str(project_path.resolve(strict=False)),
        "replica_shard_count": declared_shards,
        "configured_maximum_workers": configured_workers,
        "planned_active_worker_count": active_worker_count,
        "worker_wave_count": math.ceil(
            declared_shards / active_worker_count
        ),
    }


def _enrich_task_resources(
    root: Path,
    task: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    execution: Mapping[str, object],
    policy: Mapping[str, object],
    node_policy: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    enriched = dict(task)
    structural_qc_runtime_contract: Optional[Dict[str, object]] = None
    matched = _task_planner_rows(root, task, rows)
    path = root / str(task["script"])
    maximum_hours = float(execution["maximum_hours_per_cpu"])
    maximum_memory = float(execution["maximum_memory_gib"])
    maximum_cpus = int(execution["maximum_parallel_cpus"])
    maximum_cpus_per_node = (
        int(node_policy["cpus_per_node"])
        if isinstance(node_policy, Mapping)
        and node_policy.get("cpus_per_node") is not None
        else maximum_cpus
    )
    maximum_memory_gib_per_node = (
        float(node_policy["memory_gib_per_node"])
        if isinstance(node_policy, Mapping)
        and node_policy.get("memory_gib_per_node") is not None
        else maximum_memory
    )
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
            node_count = max(
                int(
                    row.get(
                        "parallel_node_layout_at_selected_observations", {}
                    ).get("node_count", 1)
                )
                for row in matched
            )
            distributed_replica_execution = any(
                bool(
                    row.get(
                        "parallel_node_layout_at_selected_observations", {}
                    ).get("distributed_replica_execution", False)
                )
                for row in matched
            )
            workers_per_node = max(
                int(
                    row.get(
                        "parallel_node_layout_at_selected_observations", {}
                    ).get("workers_per_node", 1)
                )
                for row in matched
            )
            distributed_worker_count = max(
                int(
                    row.get(
                        "parallel_node_layout_at_selected_observations", {}
                    ).get("active_worker_count", 1)
                )
                for row in matched
            )
            planned_execution_cpu_slots = max(
                int(
                    row.get(
                        "parallel_node_layout_at_selected_observations", {}
                    ).get(
                        "execution_cpu_slots",
                        min(maximum_cpus, int(row.get("effective_cpu_cap", 1))),
                    )
                )
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
        try:
            requested_memory_gib = max(
                float(row[
                    "estimated_scheduler_memory_gib_per_node_at_selected_observations"
                ])
                for row in matched
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutionAdapterError(
                "planner final per-node memory reservation is missing for "
                f"{path.name}"
            ) from exc
        if not math.isfinite(requested_memory_gib) or requested_memory_gib <= 0.0:
            raise ExecutionAdapterError(
                f"planner final per-node memory reservation is invalid for {path.name}"
            )
        # The planner is the sole owner of cluster memory adjustment.  The
        # execution adapter validates and emits its final reservation without
        # applying the Slurm profile factor or overhead a second time.
        aggregate_requested_memory_gib = requested_memory_gib * node_count
        if aggregate_requested_memory_gib > maximum_memory + 1e-9:
            raise ExecutionAdapterError(
                f"safety-adjusted memory request for {path.name} is "
                f"{aggregate_requested_memory_gib:g} GiB, exceeding the aggregate campaign "
                f"limit {maximum_memory:g} GiB"
            )
        if requested_memory_gib > maximum_memory_gib_per_node + 1e-9:
            raise ExecutionAdapterError(
                f"safety-adjusted memory request for {path.name} is "
                f"{requested_memory_gib:g} GiB, exceeding the per-node limit "
                f"{maximum_memory_gib_per_node:g} GiB"
            )
        planner_task_ids = [str(row["task_id"]) for row in matched]
        cpu_slots = planned_execution_cpu_slots
        structural_qc_runtime_contract = _structural_qc_replica_runtime_contract(
            root,
            task,
            matched,
            active_worker_count=cpu_slots,
        )
        source = "campaign_planner_final_memory_reservation_passthrough"
    else:
        wall_hours = _existing_wall_minutes(path) / 60.0
        memory_gib = _existing_memory_gib(path)
        requested_wall_minutes = min(maximum_hours * 60.0, wall_hours * 60.0)
        requested_memory_gib = memory_gib
        if requested_memory_gib > maximum_memory + 1e-9:
            raise ExecutionAdapterError(
                f"static memory request for {path.name} exceeds the aggregate "
                f"campaign limit"
            )
        if requested_memory_gib > maximum_memory_gib_per_node + 1e-9:
            raise ExecutionAdapterError(
                f"static memory request for {path.name} exceeds the per-node limit"
            )
        planner_task_ids = []
        cpu_slots = int(task.get("cpu_slots", 1))
        node_count = 1
        workers_per_node = cpu_slots
        distributed_worker_count = cpu_slots
        distributed_replica_execution = False
        aggregate_requested_memory_gib = requested_memory_gib
        if cpu_slots > maximum_cpus_per_node:
            raise ExecutionAdapterError(
                f"static CPU request for {path.name} exceeds the per-node limit"
            )
        source = "generated_worker_static_request_no_planner_row"
    enriched.update({
        "planner_task_ids": planner_task_ids,
        "cpu_slots": cpu_slots,
        "planned_wall_hours": wall_hours,
        "planned_peak_memory_gib": memory_gib,
        "requested_wall_minutes": requested_wall_minutes,
        "preferred_requested_wall_minutes": requested_wall_minutes,
        "minimum_requested_wall_minutes": max(
            float(policy["minimum_wall_minutes"]),
            float(math.ceil(wall_hours * 60.0)),
        ),
        "requested_memory_gib": requested_memory_gib,
        "aggregate_requested_memory_gib": aggregate_requested_memory_gib,
        "node_count": node_count,
        "workers_per_node": workers_per_node,
        "distributed_worker_count": distributed_worker_count,
        "distributed_replica_execution": distributed_replica_execution,
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
        "memory_request_limited_by_campaign_cap": False,
        "structural_qc_replica_runtime_contract": (
            structural_qc_runtime_contract
        ),
    })
    return enriched


def _walltime_path_for_phases(
    phases: Sequence[Mapping[str, object]],
    *,
    maximum_parallel_cpus: int,
    maximum_parallel_memory_gib: float,
    node_policy: Mapping[str, object],
) -> float:
    """Return the serialized scheduler-time path for dependency phases."""

    node_cpus = node_policy.get("cpus_per_node")
    node_memory = node_policy.get("memory_gib_per_node")
    configured_maximum_nodes = node_policy.get("maximum_nodes_per_campaign")
    maximum_nodes = (
        int(configured_maximum_nodes)
        if configured_maximum_nodes is not None else
        (
            max(
                math.ceil(maximum_parallel_cpus / int(node_cpus)),
                math.ceil(maximum_parallel_memory_gib / float(node_memory)),
            )
            if node_cpus is not None and node_memory is not None else None
        )
    )
    total = 0.0
    for phase_index, phase in enumerate(phases):
        items = []
        for task_index, task in enumerate(phase.get("tasks", [])):
            items.append({
                "item_id": str(task.get("task_id") or (
                    f"phase-{phase_index}:task-{task_index}"
                )),
                "cpu_slots": int(task["cpu_slots"]),
                "memory_gib": (
                    float(task["requested_memory_gib"])
                    * int(task.get("node_count", 1))
                ),
                "wall_hours": float(task["requested_wall_minutes"]) / 60.0,
                "node_count": int(task.get("node_count", 1)),
                "workers_per_node": int(task.get(
                    "workers_per_node", task["cpu_slots"]
                )),
                "distributed_replica_execution": bool(task.get(
                    "distributed_replica_execution", False
                )),
            })
        try:
            lanes = pack_resource_lanes(
                items,
                maximum_parallel_cpus=maximum_parallel_cpus,
                maximum_parallel_memory_gib=maximum_parallel_memory_gib,
                maximum_cpus_per_node=(
                    None if node_cpus is None else int(node_cpus)
                ),
                maximum_memory_gib_per_node=(
                    None if node_memory is None else float(node_memory)
                ),
                maximum_nodes=maximum_nodes,
            )
        except ResourcePlanningError as exc:
            raise ExecutionAdapterError(str(exc)) from exc
        total += max(
            (float(lane["wall_hours"]) for lane in lanes),
            default=0.0,
        )
    return total


def _fit_walltime_requests_to_campaign(
    execution_plan: Dict[str, object],
) -> Dict[str, object]:
    """Fit job kill limits inside the padded end-to-end campaign ceiling.

    Planner wall estimates already include the configured model uncertainty.
    A Slurm profile may request additional per-job timeout padding, but that
    padding is optional headroom inside the campaign limit rather than a second
    wall-time budget layered on top of it.
    """

    campaign_wall = float(execution_plan["maximum_campaign_wall_hours"])
    maximum_cpus = int(execution_plan["maximum_parallel_cpus"])
    maximum_memory = float(execution_plan["maximum_parallel_memory_gib"])
    node_policy = execution_plan.get("node_policy", {})
    assert isinstance(node_policy, Mapping)
    phases = execution_plan.get("phases", [])
    assert isinstance(phases, Sequence)
    tasks = [
        task for phase in phases for task in phase.get("tasks", [])
    ]

    def assign(scale: float) -> float:
        for task in tasks:
            minimum = float(task["minimum_requested_wall_minutes"])
            preferred = float(task["preferred_requested_wall_minutes"])
            requested = minimum + scale * max(0.0, preferred - minimum)
            task["requested_wall_minutes"] = min(
                campaign_wall * 60.0,
                max(minimum, float(math.ceil(requested))),
            )
            task["wall_request_limited_by_campaign_cap"] = (
                float(task["requested_wall_minutes"]) + 1.0e-9 < preferred
            )
        return _walltime_path_for_phases(
            phases,
            maximum_parallel_cpus=maximum_cpus,
            maximum_parallel_memory_gib=maximum_memory,
            node_policy=node_policy,
        )

    minimum_path = assign(0.0)
    preferred_path = assign(1.0)
    if preferred_path <= campaign_wall + 1.0e-9:
        scale = 1.0
        selected_path = preferred_path
        status = "preferred_padding_fits"
    elif minimum_path > campaign_wall + 1.0e-9:
        scale = 0.0
        selected_path = assign(scale)
        status = "minimum_time_limits_exceed_campaign"
    else:
        low = 0.0
        high = 1.0
        for _ in range(50):
            midpoint = (low + high) / 2.0
            path = assign(midpoint)
            if path <= campaign_wall + 1.0e-9:
                low = midpoint
            else:
                high = midpoint
        scale = low
        selected_path = assign(scale)
        status = "preferred_padding_reduced_to_fit"

    allocation = {
        "walltime_allocation_schema": "salsbury-walltime-allocation-v1",
        "contract": "padded_end_to_end_campaign_ceiling",
        "campaign_wall_limit_hours": campaign_wall,
        "minimum_scheduler_reservation_critical_path_hours": minimum_path,
        "preferred_scheduler_reservation_critical_path_hours": preferred_path,
        "selected_scheduler_reservation_critical_path_hours": selected_path,
        "preferred_padding_scale_applied": scale,
        "status": status,
        "submission_time_feasible": (
            selected_path <= campaign_wall + 1.0e-9
        ),
        "interpretation": (
            "Planner estimates already include modeled task-time uncertainty. "
            "Profile timeout padding is retained only to the extent that the "
            "serialized scheduler kill-limit path remains inside the requested "
            "padded end-to-end campaign wall limit."
        ),
    }
    execution_plan["walltime_allocation"] = allocation
    return allocation


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


def _inject_package_source_validation(text: str) -> str:
    """Make generated workers fail before importing an unintended installation."""

    marker = 'export PYTHONPATH="$PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"\n'
    if marker not in text or "SALSBURY_PACKAGE_SOURCE_VALIDATED" in text:
        return text
    validation = marker + r'''if [[ ! -d "$PACKAGE_ROOT/salsbury_md_analysis" && -d "$PACKAGE_ROOT/src/salsbury_md_analysis" ]]; then
  PACKAGE_ROOT="$PACKAGE_ROOT/src"
  export PYTHONPATH="$PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
if [[ ! -d "$PACKAGE_ROOT/salsbury_md_analysis" ]]; then
  printf 'Configured package root is not an importable source tree: %s\n' "$PACKAGE_ROOT" >&2
  exit 2
fi
SALSBURY_PACKAGE_SOURCE_ACTUAL=$("$PYTHON" -c 'import pathlib, salsbury_md_analysis; print(pathlib.Path(salsbury_md_analysis.__file__).resolve())')
case "$SALSBURY_PACKAGE_SOURCE_ACTUAL" in
  "$PACKAGE_ROOT"/salsbury_md_analysis/*) ;;
  *)
    printf 'Imported salsbury_md_analysis from %s instead of %s\n' "$SALSBURY_PACKAGE_SOURCE_ACTUAL" "$PACKAGE_ROOT" >&2
    exit 2
    ;;
esac
export SALSBURY_PACKAGE_SOURCE_VALIDATED=1
'''
    return text.replace(marker, validation, 1)


def _script_resource_requests(plan: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    requests: Dict[str, Dict[str, object]] = {}
    for phase in plan.get("phases", []):
        for task in phase.get("tasks", []):
            script = str(task["script"])
            row = requests.setdefault(script, {
                "requested_wall_minutes": 0.0,
                "requested_memory_gib": 0.0,
                "cpu_slots": 1,
                "node_count": 1,
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
            row["cpu_slots"] = max(
                int(row["cpu_slots"]), int(task["cpu_slots"])
            )
            row["node_count"] = max(
                int(row["node_count"]), int(task.get("node_count", 1))
            )
            row["planner_task_ids"].extend(task.get("planner_task_ids", []))
    for row in requests.values():
        row["planner_task_ids"] = sorted(set(row["planner_task_ids"]))
    return requests


def _partition_for_request(
    script: str,
    requested_wall_minutes: float,
    requested_memory_gib: float,
    partitions: Mapping[str, object],
    partition_limits: Mapping[str, object],
    partition_node_limits: Mapping[str, object],
    resource_policy: Mapping[str, object],
    *,
    requested_nodes: int = 1,
) -> Dict[str, object]:
    """Select a scheduler role for one task or resource-compatible task tier."""

    if (
        isinstance(requested_nodes, bool)
        or not isinstance(requested_nodes, int)
        or requested_nodes <= 0
    ):
        raise ExecutionAdapterError("requested_nodes must be a positive integer")
    role = _script_role(script)
    if (
        requested_memory_gib
        >= float(resource_policy["large_memory_threshold_gib"])
        and bool(partitions.get("large_memory"))
    ):
        role = "large_memory"
    partition = partitions.get(role) or partitions.get("default")
    partition_limit = (
        None if partition is None else partition_limits.get(str(partition))
    )
    partition_node_limit = (
        None if partition is None else partition_node_limits.get(str(partition))
    )
    long_wall_routed = False
    multi_node_routed = False
    wall_exceeds = (
        partition_limit is not None
        and requested_wall_minutes > float(partition_limit)
    )
    nodes_exceed = (
        partition_node_limit is not None
        and requested_nodes > int(partition_node_limit)
    )
    if wall_exceeds or nodes_exceed:
        long_wall_partition = partitions.get("long_wall")
        if not long_wall_partition:
            raise ExecutionAdapterError(
                f"{script} requests {requested_wall_minutes:g} minutes and "
                f"{requested_nodes} node(s), exceeding partition {partition!r} "
                "limits, but partitions.long_wall is unset"
            )
        long_wall_limit = partition_limits.get(str(long_wall_partition))
        if (
            long_wall_limit is not None
            and requested_wall_minutes > float(long_wall_limit)
        ):
            raise ExecutionAdapterError(
                f"{script} requests {requested_wall_minutes:g} minutes, "
                f"exceeding long-wall partition {long_wall_partition!r} limit "
                f"{float(long_wall_limit):g}"
            )
        long_wall_node_limit = partition_node_limits.get(
            str(long_wall_partition)
        )
        if (
            long_wall_node_limit is not None
            and requested_nodes > int(long_wall_node_limit)
        ):
            raise ExecutionAdapterError(
                f"{script} requests {requested_nodes} nodes, exceeding "
                f"partition {long_wall_partition!r} limit "
                f"{int(long_wall_node_limit)}"
            )
        role = "long_wall"
        partition = long_wall_partition
        long_wall_routed = wall_exceeds
        multi_node_routed = nodes_exceed
    return {
        "selected_partition_role": role,
        "selected_partition": partition,
        "long_wall_routed": long_wall_routed,
        "multi_node_routed": multi_node_routed,
        "selected_partition_maximum_wall_minutes": (
            None if partition is None else partition_limits.get(str(partition))
        ),
        "selected_partition_maximum_nodes": (
            None if partition is None
            else partition_node_limits.get(str(partition))
        ),
    }


def _slurm_array_expression(indices: Sequence[int]) -> str:
    """Return a compact Slurm array expression for sorted task indices."""

    ordered = sorted(set(indices))
    if not ordered:
        raise ExecutionAdapterError("a Slurm submission tier has no array tasks")
    ranges: List[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _submission_resource_tiers(
    plan: Mapping[str, object],
    partitions: Mapping[str, object],
    partition_limits: Mapping[str, object],
    partition_node_limits: Mapping[str, object],
    resource_policy: Mapping[str, object],
) -> Dict[str, List[Dict[str, object]]]:
    """Group each array's elements by their effective scheduler request."""

    tasks_by_script: Dict[str, List[Mapping[str, object]]] = {}
    for phase in plan.get("phases", []):
        for task in phase.get("tasks", []):
            if task.get("array_task_id") is None:
                continue
            tasks_by_script.setdefault(str(task["script"]), []).append(task)

    result: Dict[str, List[Dict[str, object]]] = {}
    for script, tasks in sorted(tasks_by_script.items()):
        task_ids = [int(task["array_task_id"]) for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ExecutionAdapterError(f"duplicate array task ids for {script}")
        grouped: Dict[tuple[int, str, Optional[str]], List[Mapping[str, object]]] = {}
        for task in tasks:
            wall_minutes = float(task["requested_wall_minutes"])
            memory_gib = float(task["requested_memory_gib"])
            route = _partition_for_request(
                script,
                wall_minutes,
                memory_gib,
                partitions,
                partition_limits,
                partition_node_limits,
                resource_policy,
                requested_nodes=int(task.get("node_count", 1)),
            )
            key = (
                int(math.ceil(memory_gib)),
                _format_slurm_time(wall_minutes),
                None if route["selected_partition"] is None
                else str(route["selected_partition"]),
            )
            grouped.setdefault(key, []).append(task)
        tiers: List[Dict[str, object]] = []
        for tier_index, (_, tier_tasks) in enumerate(
            sorted(
                grouped.items(),
                key=lambda item: min(int(task["array_task_id"]) for task in item[1]),
            )
        ):
            indices = sorted(int(task["array_task_id"]) for task in tier_tasks)
            requested_wall_minutes = max(
                float(task["requested_wall_minutes"]) for task in tier_tasks
            )
            requested_memory_gib = max(
                float(task["requested_memory_gib"]) for task in tier_tasks
            )
            route = _partition_for_request(
                script,
                requested_wall_minutes,
                requested_memory_gib,
                partitions,
                partition_limits,
                partition_node_limits,
                resource_policy,
                requested_nodes=max(
                    int(task.get("node_count", 1)) for task in tier_tasks
                ),
            )
            tiers.append({
                "tier_id": tier_index,
                "array_task_ids": indices,
                "array_expression": _slurm_array_expression(indices),
                "requested_wall_minutes": requested_wall_minutes,
                "requested_memory_gib": requested_memory_gib,
                "slurm_time": _format_slurm_time(requested_wall_minutes),
                "slurm_memory": f"{int(math.ceil(requested_memory_gib))}G",
                "planner_task_ids": sorted({
                    str(planner_task_id)
                    for task in tier_tasks
                    for planner_task_id in task.get("planner_task_ids", [])
                }),
                **route,
            })
        result[script] = tiers
    return result


def _tier_parallelism(counts: Sequence[int], cap: int) -> List[int]:
    """Share one original array throttle across simultaneously submitted tiers."""

    if cap < 1:
        raise ExecutionAdapterError("Slurm array parallelism must be positive")
    if len(counts) > cap:
        return [1 for _ in counts]
    allocations = [1 for _ in counts]
    remaining = cap - len(counts)
    while remaining:
        candidates = [
            index for index, count in enumerate(counts)
            if allocations[index] < count
        ]
        if not candidates:
            break
        selected = max(
            candidates,
            key=lambda index: (counts[index] - allocations[index], -index),
        )
        allocations[selected] += 1
        remaining -= 1
    return allocations


def _append_afterany_dependencies(options: str, variables: Sequence[str]) -> str:
    if not variables:
        return options
    suffix = ":".join(f"${{{variable}}}" for variable in variables)
    pattern = r'--dependency="([^"]*)"'
    if re.search(pattern, options):
        return re.sub(
            pattern,
            lambda match: (
                f'--dependency="{match.group(1)},afterany:{suffix}"'
            ),
            options,
            count=1,
        )
    return f'{options} --dependency="afterany:{suffix}"'


def _split_tiered_array_submissions(
    text: str,
    tiers_by_script: Mapping[str, Sequence[Dict[str, object]]],
) -> str:
    """Replace one mixed-resource array submission with safe tier submissions."""

    lines = text.splitlines()
    output: List[str] = []
    for line in lines:
        replacement: Optional[List[str]] = None
        for script, tiers in tiers_by_script.items():
            if len(tiers) <= 1 or f'"$ROOT/{script}"' not in line:
                continue
            match = re.fullmatch(
                r'(?P<indent>\s*)(?P<variable>[A-Za-z_][A-Za-z0-9_]*)='
                r'\$\(sbatch\s+(?P<options>.*?)\s+' +
                re.escape(f'"$ROOT/{script}"') + r'\)',
                line,
            )
            if not match:
                continue
            array_match = re.search(r'(?:^|\s)--array=([^\s]+)', match.group("options"))
            if not array_match:
                continue
            original_array = array_match.group(1)
            cap_match = re.search(r'%(\d+)$', original_array)
            cap = (
                int(cap_match.group(1)) if cap_match
                else sum(len(tier["array_task_ids"]) for tier in tiers)
            )
            options = re.sub(
                r'(?:^|\s)--array=[^\s]+', '', match.group("options"), count=1
            ).strip()
            allocations = _tier_parallelism(
                [len(tier["array_task_ids"]) for tier in tiers], cap
            )
            variable = match.group("variable")
            indent = match.group("indent")
            tier_variables: List[str] = []
            tier_lines: List[str] = []
            previous_wave: List[str] = []
            current_wave: List[str] = []
            wave_width = 0
            wave_index = 0
            for tier_index, (tier, parallelism) in enumerate(zip(tiers, allocations)):
                if wave_width + parallelism > cap:
                    previous_wave = current_wave
                    current_wave = []
                    wave_width = 0
                    wave_index += 1
                tier_variable = f"{variable}_TIER_{tier_index}"
                tier_options = _append_afterany_dependencies(options, previous_wave)
                scheduler_options = [
                    tier_options,
                    f"--array={tier['array_expression']}%{parallelism}",
                    f"--time={tier['slurm_time']}",
                    f"--mem={tier['slurm_memory']}",
                ]
                if tier.get("selected_partition"):
                    scheduler_options.append(
                        f"--partition={tier['selected_partition']}"
                    )
                tier_lines.extend([
                    f"{indent}{tier_variable}=$(sbatch "
                    + " ".join(scheduler_options)
                    + f' "$ROOT/{script}")',
                    f'{indent}{tier_variable}="${{{tier_variable}%%;*}}"',
                ])
                tier["submission_parallelism"] = parallelism
                tier["dependency_wave"] = wave_index
                tier_variables.append(tier_variable)
                current_wave.append(tier_variable)
                wave_width += parallelism
            tier_lines.append(
                f'{indent}{variable}="' + ":".join(
                    f"${{{tier_variable}}}" for tier_variable in tier_variables
                ) + '"'
            )
            replacement = tier_lines
            break
        output.extend(replacement if replacement is not None else [line])
    return "\n".join(output) + ("\n" if text.endswith("\n") else "")


def _task_resource_requests(plan: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for phase in plan.get("phases", []):
        for task in phase.get("tasks", []):
            rows.append({
                "phase_id": phase["phase_id"],
                "task_id": task.get("task_id"),
                "depends_on_task_ids": list(task.get("depends_on_task_ids", [])),
                "wait_for_task_ids": list(task.get("wait_for_task_ids", [])),
                "source_phase_id": task.get("source_phase_id"),
                "module_id": task.get("module_id"),
                "command": task.get("command"),
                "script": task["script"],
                "array_task_id": task.get("array_task_id"),
                "cpu_slots": task["cpu_slots"],
                "planner_task_ids": list(task.get("planner_task_ids", [])),
                "planned_wall_hours": task["planned_wall_hours"],
                "planned_peak_memory_gib": task["planned_peak_memory_gib"],
                "requested_wall_minutes": task["requested_wall_minutes"],
                "requested_memory_gib": task["requested_memory_gib"],
                "aggregate_requested_memory_gib": task.get(
                    "aggregate_requested_memory_gib",
                    task["requested_memory_gib"],
                ),
                "node_count": int(task.get("node_count", 1)),
                "workers_per_node": int(task.get(
                    "workers_per_node", task["cpu_slots"]
                )),
                "distributed_worker_count": int(task.get(
                    "distributed_worker_count", task["cpu_slots"]
                )),
                "distributed_replica_execution": bool(task.get(
                    "distributed_replica_execution", False
                )),
                "resource_request_source": task["resource_request_source"],
                "wall_request_limited_by_campaign_cap": task[
                    "wall_request_limited_by_campaign_cap"
                ],
                "memory_request_limited_by_campaign_cap": task[
                    "memory_request_limited_by_campaign_cap"
                ],
            })
    return rows


def _slurm_resource_epochs(
    execution_plan: Mapping[str, object],
    partitions: Mapping[str, object],
    partition_limits: Mapping[str, object],
    partition_node_limits: Mapping[str, object],
    resource_policy: Mapping[str, object],
    node_policy: Mapping[str, object],
) -> List[Dict[str, object]]:
    """Add resource-token edges to a dependency-safe task order.

    Scientific prerequisites remain ``afterok`` edges.  CPU and padded-memory
    tokens add only ``afterany`` edges, so a failed or slow task releases its
    capacity without becoming an accidental scientific gate.  Tokens are kept
    both globally and on conceptual physical nodes.  This prevents aggregate
    overcommit and node-memory fragmentation without imposing a whole-wave or
    fixed-lane barrier on unrelated work.
    """

    maximum_cpus = int(execution_plan.get("maximum_parallel_cpus", 0))
    maximum_memory = float(
        execution_plan.get("maximum_parallel_memory_gib", 0.0)
    )
    if maximum_cpus <= 0 or maximum_memory <= 0.0:
        raise ExecutionAdapterError(
            "execution plan lacks positive aggregate CPU and memory limits"
        )
    node_cpus = node_policy.get("cpus_per_node")
    node_memory = node_policy.get("memory_gib_per_node")
    configured_maximum_nodes = node_policy.get(
        "maximum_nodes_per_campaign"
    )
    maximum_nodes = (
        int(configured_maximum_nodes)
        if configured_maximum_nodes is not None else
        (
            max(
                math.ceil(maximum_cpus / int(node_cpus)),
                math.ceil(maximum_memory / float(node_memory)),
            )
            if node_cpus is not None and node_memory is not None else None
        )
    )
    items: List[Dict[str, object]] = []
    submission_index = 0
    for phase_index, phase in enumerate(execution_plan.get("phases", [])):
        phase_id = str(phase["phase_id"])
        for task_index, task in enumerate(phase.get("tasks", [])):
            script = str(task["script"])
            array_task_id = task.get("array_task_id")
            requested_wall_minutes = float(task["requested_wall_minutes"])
            requested_memory_gib = float(task["requested_memory_gib"])
            route = _partition_for_request(
                script,
                requested_wall_minutes,
                requested_memory_gib,
                partitions,
                partition_limits,
                partition_node_limits,
                resource_policy,
                requested_nodes=int(task.get("node_count", 1)),
            )
            item_id = str(task.get("task_id") or (
                f"{phase_id}:{script}:"
                f"{'single' if array_task_id is None else array_task_id}"
            ))
            item = {
                "item_id": item_id,
                "task_id": str(task.get("task_id") or item_id),
                "depends_on_task_ids": list(task.get("depends_on_task_ids", [])),
                "wait_for_task_ids": list(task.get("wait_for_task_ids", [])),
                "source_phase_id": task.get("source_phase_id"),
                "module_id": task.get("module_id"),
                "command": task.get("command"),
                "task_index": task_index,
                "script": script,
                "array_task_id": array_task_id,
                "cpu_slots": int(task["cpu_slots"]),
                "memory_gib": (
                    requested_memory_gib * int(task.get("node_count", 1))
                ),
                "wall_hours": requested_wall_minutes / 60.0,
                "planned_wall_hours": float(task["planned_wall_hours"]),
                "requested_wall_minutes": requested_wall_minutes,
                "requested_memory_gib": requested_memory_gib,
                "node_count": int(task.get("node_count", 1)),
                "workers_per_node": int(task.get("workers_per_node", task["cpu_slots"])),
                "distributed_worker_count": int(
                    task.get("distributed_worker_count", task["cpu_slots"])
                ),
                "distributed_replica_execution": bool(
                    task.get("distributed_replica_execution", False)
                ),
                "slurm_time": _format_slurm_time(requested_wall_minutes),
                "slurm_memory": f"{int(math.ceil(requested_memory_gib))}G",
                "planner_task_ids": list(task.get("planner_task_ids", [])),
                "phase_index": phase_index,
                "phase_id": phase_id,
                "submission_index": submission_index,
                **route,
            }
            items.append(item)
            submission_index += 1
    if (node_cpus is None) != (node_memory is None):
        raise ExecutionAdapterError(
            "resource-token scheduling requires both node CPU and memory limits"
        )
    memory_quantum_gib = 0.25
    memory_token_count = int(
        math.floor(maximum_memory / memory_quantum_gib + 1.0e-9)
    )
    if memory_token_count <= 0:
        raise ExecutionAdapterError(
            "aggregate memory is below the resource-token quantum"
        )
    cpu_tokens = [
        {"available": 0.0, "owner": None} for _ in range(maximum_cpus)
    ]
    memory_tokens = [
        {"available": 0.0, "owner": None}
        for _ in range(memory_token_count)
    ]
    if node_cpus is not None and node_memory is not None:
        if maximum_nodes is None:
            raise ExecutionAdapterError(
                "resource-token scheduling could not determine a node count"
            )
        per_node_memory_tokens = int(math.floor(
            float(node_memory) / memory_quantum_gib + 1.0e-9
        ))
        node_tokens = [{
            "cpu": [
                {"available": 0.0, "owner": None}
                for _ in range(int(node_cpus))
            ],
            "memory": [
                {"available": 0.0, "owner": None}
                for _ in range(per_node_memory_tokens)
            ],
        } for _ in range(maximum_nodes)]
    else:
        node_tokens = []

    def selected_token_indices(
        tokens: Sequence[Mapping[str, object]], count: int,
    ) -> List[int]:
        return sorted(
            range(len(tokens)),
            key=lambda index: (float(tokens[index]["available"]), index),
        )[:count]

    item_by_id = {str(item["item_id"]): item for item in items}
    if len(item_by_id) != len(items):
        raise ExecutionAdapterError("execution plan contains duplicate task IDs")

    declared_dependencies_by_id = {
        item_id: list(dict.fromkeys([
            *map(str, item.get("depends_on_task_ids", [])),
            *map(str, item.get("wait_for_task_ids", [])),
        ]))
        for item_id, item in item_by_id.items()
    }
    missing_dependencies = {
        dependency
        for dependencies in declared_dependencies_by_id.values()
        for dependency in dependencies
        if dependency not in item_by_id
    }
    if missing_dependencies:
        raise ExecutionAdapterError(
            "execution plan contains unknown dependencies: "
            + ", ".join(sorted(missing_dependencies))
        )

    successors: Dict[str, List[str]] = {
        item_id: [] for item_id in item_by_id
    }
    for item_id, dependencies in declared_dependencies_by_id.items():
        for dependency in dependencies:
            successors[dependency].append(item_id)
    critical_rank_cache: Dict[str, float] = {}
    critical_rank_visiting: set[str] = set()

    def critical_rank(item_id: str) -> float:
        """Return the remaining planned path through declared dependencies."""

        cached = critical_rank_cache.get(item_id)
        if cached is not None:
            return cached
        if item_id in critical_rank_visiting:
            raise ExecutionAdapterError(
                "execution plan contains a dependency cycle involving " + item_id
            )
        critical_rank_visiting.add(item_id)
        downstream = max(
            (critical_rank(successor) for successor in successors[item_id]),
            default=0.0,
        )
        rank = float(item_by_id[item_id]["planned_wall_hours"]) + downstream
        critical_rank_visiting.remove(item_id)
        critical_rank_cache[item_id] = rank
        return rank

    for item_id in item_by_id:
        critical_rank(item_id)

    planned_finishes: Dict[str, float] = {}

    def candidate_allocation(item: Mapping[str, object]) -> Dict[str, object]:
        scheduling_id = str(item["item_id"])
        cpu_count = int(item["cpu_slots"])
        memory_count = int(math.ceil(
            float(item["memory_gib"]) / memory_quantum_gib - 1.0e-12
        ))
        if cpu_count > len(cpu_tokens) or memory_count > len(memory_tokens):
            raise ExecutionAdapterError(
                f"task {item['item_id']} exceeds the aggregate resource envelope"
            )
        cpu_indices = selected_token_indices(cpu_tokens, cpu_count)
        memory_indices = selected_token_indices(memory_tokens, memory_count)
        declared_dependencies = declared_dependencies_by_id[scheduling_id]
        selected_node_tokens: List[tuple[int, List[int], List[int]]] = []
        assigned_node_indices: List[int] = []
        if node_tokens:
            task_nodes = int(item.get("node_count", 1))
            if task_nodes > len(node_tokens):
                raise ExecutionAdapterError(
                    f"task {item['item_id']} requests {task_nodes} nodes, exceeding "
                    f"the campaign limit {len(node_tokens)}"
                )
            per_node_cpus = []
            remaining_cpus = cpu_count
            workers_per_node = int(item.get("workers_per_node", cpu_count))
            for _node_offset in range(task_nodes):
                fragment_cpus = min(workers_per_node, remaining_cpus)
                remaining_cpus -= fragment_cpus
                per_node_cpus.append(fragment_cpus)
            if remaining_cpus or any(value <= 0 for value in per_node_cpus):
                raise ExecutionAdapterError(
                    f"task {item['item_id']} has an invalid distributed CPU layout"
                )
            memory_per_node = float(item["requested_memory_gib"])
            memory_per_node_count = int(math.ceil(
                memory_per_node / memory_quantum_gib - 1.0e-12
            ))
            candidate_nodes = []
            for node_index, node in enumerate(node_tokens):
                node_cpu_indices = selected_token_indices(
                    node["cpu"], max(per_node_cpus)
                )
                node_memory_indices = selected_token_indices(
                    node["memory"], memory_per_node_count
                )
                if (
                    len(node_cpu_indices) < max(per_node_cpus)
                    or len(node_memory_indices) < memory_per_node_count
                ):
                    continue
                ready = max(
                    [float(node["cpu"][index]["available"])
                     for index in node_cpu_indices]
                    + [float(node["memory"][index]["available"])
                       for index in node_memory_indices]
                    + [0.0]
                )
                candidate_nodes.append((ready, node_index))
            if len(candidate_nodes) < task_nodes:
                raise ExecutionAdapterError(
                    f"task {item['item_id']} cannot fit on the configured nodes"
                )
            assigned_node_indices = [
                node_index for _ready, node_index in sorted(candidate_nodes)[:task_nodes]
            ]
            for fragment_cpus, node_index in zip(
                per_node_cpus, assigned_node_indices
            ):
                node = node_tokens[node_index]
                selected_node_tokens.append((
                    node_index,
                    selected_token_indices(node["cpu"], fragment_cpus),
                    selected_token_indices(
                        node["memory"], memory_per_node_count
                    ),
                ))
        global_cpu_owners = {
            str(owner)
            for index in cpu_indices for owner in [cpu_tokens[index]["owner"]]
            if owner is not None
        }
        global_memory_owners = {
            str(owner)
            for index in memory_indices
            for owner in [memory_tokens[index]["owner"]]
            if owner is not None
        }
        node_owners = {
            str(owner)
            for node_index, node_cpu_indices, node_memory_indices
            in selected_node_tokens
            for tokens, indices in (
                (node_tokens[node_index]["cpu"], node_cpu_indices),
                (node_tokens[node_index]["memory"], node_memory_indices),
            )
            for index in indices
            for owner in [tokens[index]["owner"]]
            if owner is not None
        }
        resource_predecessors = sorted(
            (global_cpu_owners | global_memory_owners | node_owners)
            .difference(declared_dependencies)
        )
        start = max(
            [planned_finishes[value] for value in declared_dependencies]
            + [float(cpu_tokens[index]["available"]) for index in cpu_indices]
            + [float(memory_tokens[index]["available"]) for index in memory_indices]
            + [
                float(tokens[index]["available"])
                for node_index, node_cpu_indices, node_memory_indices
                in selected_node_tokens
                for tokens, indices in (
                    (node_tokens[node_index]["cpu"], node_cpu_indices),
                    (node_tokens[node_index]["memory"], node_memory_indices),
                )
                for index in indices
            ]
            + [0.0]
        )
        finish = start + float(item["planned_wall_hours"])
        return {
            "cpu_indices": cpu_indices,
            "memory_indices": memory_indices,
            "selected_node_tokens": selected_node_tokens,
            "assigned_node_indices": assigned_node_indices,
            "resource_predecessors": resource_predecessors,
            "start": start,
            "finish": finish,
            "memory_count": memory_count,
        }

    unscheduled = dict(item_by_id)
    scheduled_items: List[Dict[str, object]] = []
    while unscheduled:
        ready_items = [
            item for item_id, item in unscheduled.items()
            if all(
                dependency in planned_finishes
                for dependency in declared_dependencies_by_id[item_id]
            )
        ]
        if not ready_items:
            raise ExecutionAdapterError(
                "execution plan contains a dependency cycle among: "
                + ", ".join(sorted(unscheduled))
            )
        candidates = [
            (item, candidate_allocation(item)) for item in ready_items
        ]
        item, allocation = min(
            candidates,
            key=lambda row: (
                float(row[1]["start"]),
                -critical_rank(str(row[0]["item_id"])),
                -float(row[0]["memory_gib"]),
                -int(row[0]["cpu_slots"]),
                int(row[0]["submission_index"]),
            ),
        )
        scheduling_id = str(item["item_id"])
        item["original_submission_index"] = int(item["submission_index"])
        item["submission_index"] = len(scheduled_items)
        item["planned_node_indices"] = list(
            allocation["assigned_node_indices"]
        )
        resource_predecessors = list(allocation["resource_predecessors"])
        start = float(allocation["start"])
        finish = float(allocation["finish"])
        memory_count = int(allocation["memory_count"])
        item["resource_predecessor_task_ids"] = resource_predecessors
        item["planned_resource_start_hours"] = start
        item["planned_resource_finish_hours"] = finish
        item["memory_token_count"] = memory_count
        item["memory_token_quantum_gib"] = memory_quantum_gib
        for index in allocation["cpu_indices"]:
            cpu_tokens[index] = {"available": finish, "owner": scheduling_id}
        for index in allocation["memory_indices"]:
            memory_tokens[index] = {
                "available": finish, "owner": scheduling_id
            }
        for (
            node_index, node_cpu_indices, node_memory_indices
        ) in allocation["selected_node_tokens"]:
            for index in node_cpu_indices:
                node_tokens[node_index]["cpu"][index] = {
                    "available": finish, "owner": scheduling_id,
                }
            for index in node_memory_indices:
                node_tokens[node_index]["memory"][index] = {
                    "available": finish, "owner": scheduling_id,
                }
        planned_finishes[scheduling_id] = finish
        scheduled_items.append(item)
        del unscheduled[scheduling_id]

    requested_finishes: Dict[str, float] = {}
    for item in scheduled_items:
        requested_dependencies = list(dict.fromkeys([
            *map(str, item.get("depends_on_task_ids", [])),
            *map(str, item.get("wait_for_task_ids", [])),
            *map(str, item.get("resource_predecessor_task_ids", [])),
        ]))
        requested_start = max(
            [requested_finishes[value] for value in requested_dependencies]
            + [0.0]
        )
        requested_finish = requested_start + float(item["wall_hours"])
        item["scheduler_reservation_start_hours"] = requested_start
        item["scheduler_reservation_finish_hours"] = requested_finish
        requested_finishes[str(item["item_id"])] = requested_finish

    def peak(field_start: str, field_finish: str, field_value: str) -> float:
        events = []
        for item in scheduled_items:
            events.extend([
                (float(item[field_start]), 1, float(item[field_value])),
                (float(item[field_finish]), -1, float(item[field_value])),
            ])
        current = maximum = 0.0
        # End events precede starts at a shared boundary.
        for _time, direction, value in sorted(
            events, key=lambda row: (row[0], row[1])
        ):
            current += direction * value
            maximum = max(maximum, current)
        return maximum

    planned_wall = max(planned_finishes.values(), default=0.0)
    reservation_wall = max(requested_finishes.values(), default=0.0)
    peak_cpus = int(round(peak(
        "planned_resource_start_hours", "planned_resource_finish_hours",
        "cpu_slots",
    )))
    peak_memory = peak(
        "planned_resource_start_hours", "planned_resource_finish_hours",
        "memory_gib",
    )
    return [{
        "resource_epoch_index": 0,
        "phase_id": "task_dag_resource_token_schedule",
        "lanes": [],
        "scheduled_items": scheduled_items,
        "cpu_slots": peak_cpus,
        "memory_gib": peak_memory,
        "planned_wall_hours": planned_wall,
        "wall_hours": reservation_wall,
        "resource_token_policy": {
            "cpu_token_count": maximum_cpus,
            "memory_token_count": memory_token_count,
            "memory_token_quantum_gib": memory_quantum_gib,
            "contract": (
                "a task acquires fixed CPU and memory token chains; afterany "
                "predecessors release those tokens regardless of success"
            ),
        },
    }]


def _slurm_submission_preview(
    execution_plan: Mapping[str, object],
    resource_epochs: Sequence[Mapping[str, object]],
    node_policy: Mapping[str, object],
) -> Dict[str, object]:
    """Return a bounded, machine-readable contract shown before submission."""

    maximum_cpus = int(execution_plan["maximum_parallel_cpus"])
    maximum_memory = float(execution_plan["maximum_parallel_memory_gib"])
    resource_lanes = [
        lane
        for epoch in resource_epochs
        for lane in epoch.get("lanes", [])
    ]
    scheduled_items = [
        item
        for epoch in resource_epochs
        for item in epoch.get("scheduled_items", [])
    ]
    preview_items = scheduled_items or [
        item for lane in resource_lanes for item in lane.get("items", [])
    ]
    task_count = len(preview_items)
    scientific_dependency_edge_count = sum(
        len(item.get("depends_on_task_ids", []))
        for item in preview_items
    )
    completion_wait_edge_count = sum(
        len(item.get("wait_for_task_ids", []))
        for item in preview_items
    )
    resource_token_edge_count = sum(
        len(item.get("resource_predecessor_task_ids", []))
        for item in preview_items
    )
    peak_cpus = max(
        (int(epoch.get("cpu_slots", 0)) for epoch in resource_epochs),
        default=0,
    )
    peak_memory = max(
        (float(epoch.get("memory_gib", 0.0)) for epoch in resource_epochs),
        default=0.0,
    )
    planner_critical_path = sum(
        float(epoch.get("planned_wall_hours", 0.0))
        for epoch in resource_epochs
    )
    scheduler_reservation_path = sum(
        float(epoch.get("wall_hours", 0.0))
        for epoch in resource_epochs
    )
    warnings: List[Dict[str, object]] = []
    node_cpus = node_policy.get("cpus_per_node")
    node_memory = node_policy.get("memory_gib_per_node")
    node_reservations = []
    planned_node_count = len({
        int(node_index)
        for item in preview_items
        for node_index in item.get("planned_node_indices", [])
    })
    if node_cpus is not None and node_memory is not None:
        for item in preview_items:
            per_node_cpus = int(item.get(
                "workers_per_node", item["cpu_slots"]
            ))
            per_node_memory = float(item["requested_memory_gib"])
            if (
                per_node_cpus > int(node_cpus)
                or per_node_memory > float(node_memory) + 1.0e-9
            ):
                raise ExecutionAdapterError(
                    "planned padded task reservation exceeds one node"
                )
            node_reservations.append({
                "task_id": item.get("task_id"),
                "planned_node_indices": list(
                    item.get("planned_node_indices", [])
                ),
                "node_count": int(item.get("node_count", 1)),
                "reserved_cpus_per_node": per_node_cpus,
                "reserved_memory_gib_per_node": per_node_memory,
            })
    if peak_cpus < maximum_cpus:
        warnings.append({
            "severity": "warning",
            "code": "REQUESTED_CPUS_EXCEED_GENERATED_PARALLELISM",
            "message": (
                f"The campaign allows {maximum_cpus} simultaneous CPUs, but its "
                "generated dependency and resource-token schedule can use at "
                f"most {peak_cpus}; "
                f"the remaining {maximum_cpus - peak_cpus} CPUs cannot shorten "
                "this prepared schedule."
            ),
            "requested_parallel_cpus": maximum_cpus,
            "generated_parallel_cpu_ceiling": peak_cpus,
            "excess_cpus": maximum_cpus - peak_cpus,
        })
    campaign_wall_hours = (
        float(execution_plan["maximum_campaign_wall_hours"])
        if execution_plan.get("maximum_campaign_wall_hours") is not None
        else None
    )
    planner_path_feasible = (
        campaign_wall_hours is None
        or planner_critical_path <= campaign_wall_hours + 1.0e-9
    )
    scheduler_path_feasible = (
        campaign_wall_hours is None
        or scheduler_reservation_path <= campaign_wall_hours + 1.0e-9
    )
    walltime_allocation = execution_plan.get("walltime_allocation", {})
    allocation_feasible = (
        not isinstance(walltime_allocation, Mapping)
        or bool(walltime_allocation.get("submission_time_feasible", True))
    )
    generated_schedule_feasible = (
        planner_path_feasible
        and scheduler_path_feasible
        and allocation_feasible
    )
    if not planner_path_feasible:
        warnings.append({
            "severity": "error",
            "code": "GENERATED_SCHEDULE_EXCEEDS_CAMPAIGN_WALL_LIMIT",
            "message": (
                "The generated dependency/resource schedule is estimated to "
                f"require {planner_critical_path:g} hours, exceeding the "
                f"configured campaign limit of {campaign_wall_hours:g} hours. "
                "Submission is disabled; reduce optional work, adjust sampling, "
                "or increase the campaign wall limit and replan."
            ),
            "generated_schedule_wall_hours": planner_critical_path,
            "campaign_wall_limit_hours": campaign_wall_hours,
        })
    elif not scheduler_path_feasible or not allocation_feasible:
        warnings.append({
            "severity": "error",
            "code": "MINIMUM_SCHEDULER_TIMEOUTS_EXCEED_CAMPAIGN_WALL_LIMIT",
            "message": (
                "Even the planner estimates with minimum per-job scheduler "
                f"timeouts require {scheduler_reservation_path:g} hours along "
                f"the dependency path, exceeding the padded end-to-end campaign "
                f"limit of {campaign_wall_hours:g} hours. Submission is disabled; "
                "reduce optional work, adjust sampling, or increase the campaign "
                "wall limit and replan."
            ),
            "scheduler_reservation_critical_path_hours": (
                scheduler_reservation_path
            ),
            "campaign_wall_limit_hours": campaign_wall_hours,
        })
    if (
        isinstance(walltime_allocation, Mapping)
        and walltime_allocation.get("status")
        == "preferred_padding_reduced_to_fit"
    ):
        warnings.append({
            "severity": "warning",
            "code": "SCHEDULER_TIMEOUT_PADDING_REDUCED_TO_FIT_CAMPAIGN",
            "message": (
                "The preferred per-job scheduler timeout padding did not fit "
                "inside the requested padded end-to-end campaign wall limit. "
                "The launcher retained the largest uniform fraction that fits; "
                "sampling and enabled analyses were not changed."
            ),
            "preferred_padding_scale_applied": float(
                walltime_allocation.get("preferred_padding_scale_applied", 0.0)
            ),
            "preferred_scheduler_reservation_critical_path_hours": float(
                walltime_allocation.get(
                    "preferred_scheduler_reservation_critical_path_hours", 0.0
                )
            ),
            "selected_scheduler_reservation_critical_path_hours": float(
                walltime_allocation.get(
                    "selected_scheduler_reservation_critical_path_hours", 0.0
                )
            ),
            "campaign_wall_limit_hours": campaign_wall_hours,
        })
    lane_summaries = [
        {
            "resource_lane_index": int(lane.get("lane_index", index)),
            "resource_epoch_index": int(
                lane.get("resource_epoch_index", 0)
            ),
            "phase_ids": list(lane.get("phase_ids", [])),
            "task_count": len(lane.get("items", [])),
            "cpu_slots": int(lane["cpu_slots"]),
            "reserved_memory_gib": float(lane["memory_gib"]),
            "planner_estimated_wall_hours": float(
                lane.get("planned_wall_hours", 0.0)
            ),
            "scheduler_time_limit_hours": float(lane.get("wall_hours", 0.0)),
        }
        for index, lane in enumerate(resource_lanes)
    ]
    epoch_summaries = [
        {
            "resource_epoch_index": int(epoch["resource_epoch_index"]),
            "phase_id": str(epoch["phase_id"]),
            "resource_lane_count": len(epoch.get("lanes", [])),
            "task_count": len(epoch.get("scheduled_items", [])) or sum(
                len(lane.get("items", []))
                for lane in epoch.get("lanes", [])
            ),
            "cpu_slots": int(epoch.get("cpu_slots", 0)),
            "reserved_memory_gib": float(epoch.get("memory_gib", 0.0)),
            "planner_estimated_wall_hours": float(
                epoch.get("planned_wall_hours", 0.0)
            ),
            "scheduler_time_limit_hours": float(
                epoch.get("wall_hours", 0.0)
            ),
        }
        for epoch in resource_epochs
    ]
    return {
        "slurm_submission_preview_schema": "salsbury-slurm-submission-preview-v3",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "execution_started": False,
        "jobs_submitted": False,
        "task_count": task_count,
        "dependency_model": execution_plan.get("dependency_model", "legacy_phase_chain"),
        "scientific_dependency_edge_count": scientific_dependency_edge_count,
        "completion_wait_edge_count": completion_wait_edge_count,
        "resource_token_edge_count": resource_token_edge_count,
        "dependency_wave_count": len({
            int(item.get("phase_index", 0)) for item in preview_items
        }),
        "resource_wave_count": len(resource_epochs),
        "resource_epoch_count": len(resource_epochs),
        "resource_lane_count": len(resource_lanes),
        "maximum_parallel_cpus_configured": maximum_cpus,
        "maximum_parallel_cpus_in_generated_waves": peak_cpus,
        "maximum_parallel_memory_gib_configured": maximum_memory,
        "maximum_parallel_memory_gib_in_generated_waves": peak_memory,
        "node_policy": dict(node_policy),
        "planned_node_count": planned_node_count,
        "planned_node_reservations": node_reservations,
        "per_node_padding_validation": (
            "complete" if node_cpus is not None and node_memory is not None
            else "not_configured"
        ),
        "planner_estimated_dependency_critical_path_hours": (
            planner_critical_path
        ),
        "scheduler_time_limit_reservation_critical_path_hours": (
            scheduler_reservation_path
        ),
        "campaign_planner_wall_hours": campaign_wall_hours,
        "walltime_allocation": dict(walltime_allocation),
        "generated_schedule_feasibility_status": (
            "feasible" if generated_schedule_feasible else "infeasible"
        ),
        "submission_permitted": generated_schedule_feasible,
        "autorecovery": {
            "enabled": bool(execution_plan.get("autorecovery", True)),
            "maximum_task_attempts": int(
                execution_plan.get("maximum_task_attempts", 2)
            ),
            "failed_task_policy": "bounded retry inside the same allocation",
            "timeout_policy": (
                "pre-timeout signal and bounded Slurm requeue; checkpoint-aware "
                "modules resume compatible work"
            ),
            "completed_task_policy": "reuse; never resubmit",
        },
        "dependency_waves": epoch_summaries,
        "resource_epochs": epoch_summaries,
        "resource_lanes": lane_summaries,
        "resource_contract": (
            "The complete task DAG is guarded by global and per-node CPU and "
            "padded-memory tokens. "
            "A task waits only for its own success-required inputs, explicit "
            "completion-only inputs, and prior users of its resource tokens; "
            "there is no whole-depth barrier. Resource-token predecessors use "
            "afterany, so a failed task releases capacity without authorizing a "
            "scientifically dependent task."
            if execution_plan.get("dependency_model") == "task_dag_v1" else
            "Each later dependency wave waits for successful completion of every "
            "job in the preceding wave. CPU slots and safety-adjusted memory summed "
            "within every wave stay at or below the configured aggregate caps."
        ),
        "time_interpretation": (
            "The requested campaign wall time is the padded end-to-end ceiling. "
            "Planner hours are estimated execution time, while scheduler "
            "reservation hours sum per-job kill limits along the serialized "
            "dependency path. Preferred profile padding is reduced uniformly "
            "when necessary so that those kill limits remain inside the ceiling; "
            "sampling and module selection are unchanged by this adjustment."
        ),
        "preview_command": "./submit.sh --preview",
        "execution_command": "./submit.sh",
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def _render_resource_bounded_submit(
    root: Path,
    profile: Mapping[str, object],
    profile_path: Path,
    resource_epochs: Sequence[Mapping[str, object]],
    submission_permitted: bool,
    autorecovery: bool = False,
) -> str:
    """Render one launcher with scientific dependencies and resource epochs."""

    submit_command = shlex.quote(str(profile["submit_command"]))
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)',
        _profile_preamble(profile, profile_path),
        f"SUBMIT_COMMAND={submit_command}",
        'RECOVERY_RUNNER="$ROOT/run-task-with-recovery.sh"',
        'PREVIEW="$ROOT/slurm-submission-preview.json"',
        'case "${1:-}" in',
        '  --preview)',
        '    cat "$PREVIEW"',
        '    exit 0',
        '    ;;',
        '  "") ;;',
        '  *)',
        '    printf "Usage: %s [--preview]\\n" "$0" >&2',
        '    exit 2',
        '    ;;',
        'esac',
        'cat "$PREVIEW"',
    ]
    if not submission_permitted:
        lines.extend([
            'printf "Submission refused: generated schedule exceeds the reviewed campaign limit.\\n" >&2',
            'exit 3',
        ])
    lines.extend([
        'printf "Submitting the reviewed Slurm resource epochs now.\\n"',
        "",
    ])
    submitted_jobs: List[str] = []
    task_jobs: Dict[str, str] = {}
    previous_lane_jobs: Dict[int, str] = {}
    resource_lanes = [
        lane
        for epoch in resource_epochs
        for lane in epoch.get("lanes", [])
    ]
    scheduled_rows = [
        (
            int(epoch.get("resource_epoch_index", 0)),
            -1,
            item,
        )
        for epoch in resource_epochs
        for item in epoch.get("scheduled_items", [])
    ]
    ordered_items = sorted(
        scheduled_rows or [
            (
                int(lane.get("resource_epoch_index", 0)),
                int(lane.get("lane_index", lane_index)),
                item,
            )
            for lane_index, lane in enumerate(resource_lanes)
            for item in lane.get("items", [])
        ],
        key=lambda row: int(row[2].get("submission_index", 0)),
    )
    referenced_task_ids = {
        str(task_id)
        for _epoch_index, _lane_index, item in ordered_items
        for key in (
            "depends_on_task_ids", "wait_for_task_ids",
            "resource_predecessor_task_ids",
        )
        for task_id in item.get(key, [])
    }
    terminal_variables = [
        f"JOB_T{item_index:04d}"
        for item_index, (_epoch_index, _lane_index, item)
        in enumerate(ordered_items)
        if str(item.get("task_id") or item.get("item_id"))
        not in referenced_task_ids
    ]
    for item_index, (epoch_index, lane_index, item) in enumerate(ordered_items):
        if item_index == 0 or (
            item_index > 0
            and str(item.get("phase_id"))
            != str(ordered_items[item_index - 1][2].get("phase_id"))
        ):
            lines.append(
                f"# resource epoch {epoch_index + 1}: "
                f"dependency phase {item.get('phase_id')}"
            )
        if lane_index >= 0:
            lines.append(
                f"# resource lane {lane_index + 1}: "
                f"{int(item['cpu_slots'])} CPUs, "
                f"{float(item['memory_gib']):g} GiB"
            )
        else:
            lines.append(
                "# resource tokens: "
                f"{int(item['cpu_slots'])} CPUs, "
                f"{float(item['memory_gib']):g} GiB aggregate"
            )
        variable = f"JOB_T{item_index:04d}"
        options = [
            "--parsable",
            f"--nodes={int(item.get('node_count', 1))}",
        ]
        if autorecovery:
            options.extend(["--requeue", "--signal=B:USR1@60"])
        if item.get("distributed_replica_execution"):
            options.extend([
                f"--ntasks={int(item['distributed_worker_count'])}",
                f"--ntasks-per-node={int(item['workers_per_node'])}",
                "--cpus-per-task=1",
                "--export=ALL,"
                f"SMA_REPLICA_WORKERS={int(item['distributed_worker_count'])},"
                f"SMA_REPLICA_WORKERS_PER_NODE={int(item['workers_per_node'])},"
                "SMA_DISTRIBUTED_REPLICA_WORKERS=1,"
                "SMA_DISTRIBUTED_WORK_DIR=$ROOT",
            ])
        else:
            options.append(f"--cpus-per-task={int(item['cpu_slots'])}")
        options.extend([
            f"--time={item['slurm_time']}",
            f"--mem={item['slurm_memory']}",
        ])
        if item.get("selected_partition"):
            options.append(
                f"--partition={shlex.quote(str(item['selected_partition']))}"
            )
        if item.get("array_task_id") is not None:
            options.append(f"--array={int(item['array_task_id'])}")
        scientific_job_variables = []
        for required_task_id in item.get("depends_on_task_ids", []):
            required = task_jobs.get(str(required_task_id))
            if required is None:
                raise ExecutionAdapterError(
                    f"task {item.get('task_id')} is scheduled before its "
                    f"prerequisite {required_task_id}"
                )
            scientific_job_variables.append(required)
        completion_job_variables = []
        for waited_task_id in item.get("wait_for_task_ids", []):
            waited = task_jobs.get(str(waited_task_id))
            if waited is None:
                raise ExecutionAdapterError(
                    f"task {item.get('task_id')} is scheduled before its "
                    f"completion wait {waited_task_id}"
                )
            completion_job_variables.append(waited)
        resource_job_variables = []
        for predecessor_task_id in item.get(
            "resource_predecessor_task_ids", []
        ):
            predecessor = task_jobs.get(str(predecessor_task_id))
            if predecessor is None:
                raise ExecutionAdapterError(
                    f"task {item.get('task_id')} is scheduled before its "
                    f"resource predecessor {predecessor_task_id}"
                )
            resource_job_variables.append(predecessor)
        clauses = []
        lane_predecessor = (
            previous_lane_jobs.get(lane_index) if lane_index >= 0 else None
        )
        previous_epoch_variables = (
            [] if scheduled_rows else []
        )
        afterany_variables = list(dict.fromkeys([
            *([lane_predecessor] if lane_predecessor else []),
            *previous_epoch_variables,
            *completion_job_variables,
            *resource_job_variables,
        ]))
        if afterany_variables:
            clauses.append(
                "afterany:" + ":".join(
                    f"${{{name}}}" for name in afterany_variables
                )
            )
        if scientific_job_variables:
            clauses.append(
                "afterok:" + ":".join(
                    f"${{{name}}}" for name in scientific_job_variables
                )
            )
            options.append("--kill-on-invalid-dep=yes")
        if clauses:
            options.append('--dependency="' + ",".join(clauses) + '"')
        command_options = " ".join(options)
        script = shlex.quote(str(item["script"]))
        lines.extend([
            f'{variable}=$("$SUBMIT_COMMAND" {command_options} '
            f'"$RECOVERY_RUNNER" "$ROOT"/{script})',
            f'{variable}="${{{variable}%%;*}}"',
        ])
        if lane_index >= 0:
            previous_lane_jobs[lane_index] = variable
        submitted_jobs.append(variable)
        if item.get("task_id"):
            task_jobs[str(item["task_id"])] = variable
        lines.append("")
    if submitted_jobs:
        final_variables = terminal_variables or [submitted_jobs[-1]]
        lines.extend([
            'printf "Submitted %s jobs; final job IDs: %s\\n" '
            f'"{len(submitted_jobs)}" "'
            + ":".join(
                f"${{{name}}}" for name in final_variables
            )
            + '"',
        ])
    else:
        lines.append('printf "No jobs were generated.\\n"')
    return "\n".join(lines) + "\n"


def _render_task_recovery_runner(
    *, autorecovery: bool, maximum_task_attempts: int,
) -> str:
    """Render a bounded task wrapper shared by all generated Slurm jobs."""

    enabled = 1 if autorecovery else 0
    return f'''#!/usr/bin/env bash
set -uo pipefail
TASK="${{1:?worker script is required}}"
AUTORECOVERY_ENABLED={enabled}
MAXIMUM_TASK_ATTEMPTS={maximum_task_attempts}
RESTART_COUNT="${{SLURM_RESTART_COUNT:-0}}"
ROOT=$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)
STATUS_DIR="$ROOT/autorecovery-status"
mkdir -p "$STATUS_DIR"
JOB_TOKEN="${{SLURM_JOB_ID:-local}}"
if [[ -n "${{SLURM_ARRAY_TASK_ID:-}}" ]]; then
  JOB_TOKEN="$JOB_TOKEN-array-${{SLURM_ARRAY_TASK_ID}}"
fi
STATUS_FILE="$STATUS_DIR/$JOB_TOKEN.jsonl"
CURRENT_ATTEMPT=0
record_status() {{
  local event="$1"
  local exit_code="$2"
  local attempt="$3"
  printf '{{"event":"%s","task":"%s","attempt":%s,"restart_count":%s,"exit_code":%s}}\n' \
    "$event" "${{TASK##*/}}" "$attempt" "$RESTART_COUNT" "$exit_code" >> "$STATUS_FILE"
}}
handle_timeout_signal() {{
  trap - USR1 TERM
  local attempts_used=$((RESTART_COUNT + CURRENT_ATTEMPT))
  record_status timeout_signal 124 "$attempts_used"
  if [[ "$AUTORECOVERY_ENABLED" -eq 1 && "$attempts_used" -lt "$MAXIMUM_TASK_ATTEMPTS" ]]; then
    if command -v scontrol >/dev/null 2>&1 && [[ -n "${{SLURM_JOB_ID:-}}" ]]; then
      record_status requeue_requested 124 "$attempts_used"
      scontrol requeue "$SLURM_JOB_ID"
      exit 0
    fi
  fi
  exit 124
}}
trap handle_timeout_signal USR1 TERM
remaining=$((MAXIMUM_TASK_ATTEMPTS - RESTART_COUNT))
if [[ "$AUTORECOVERY_ENABLED" -ne 1 ]]; then
  remaining=1
fi
if [[ "$remaining" -lt 1 ]]; then
  remaining=1
fi
for ((attempt=1; attempt<=remaining; attempt++)); do
  CURRENT_ATTEMPT="$attempt"
  global_attempt=$((RESTART_COUNT + attempt))
  record_status started 0 "$global_attempt"
  bash "$TASK"
  exit_code=$?
  if [[ "$exit_code" -eq 0 ]]; then
    record_status complete 0 "$global_attempt"
    exit 0
  fi
  record_status failed "$exit_code" "$global_attempt"
done
exit "$exit_code"
'''


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
    node_policy = profile["node_policy"]
    assert isinstance(node_policy, Mapping)
    script_requests = _script_resource_requests(execution_plan)
    partition_limits = profile["partition_maximum_wall_minutes"]
    assert isinstance(partition_limits, Mapping)
    partition_node_limits = profile["partition_maximum_nodes"]
    assert isinstance(partition_node_limits, Mapping)
    submission_tiers = _submission_resource_tiers(
        execution_plan,
        partitions,
        partition_limits,
        partition_node_limits,
        resource_policy,
    )
    resource_epochs = _slurm_resource_epochs(
        execution_plan,
        partitions,
        partition_limits,
        partition_node_limits,
        resource_policy,
        node_policy,
    )
    submission_preview = _slurm_submission_preview(
        execution_plan, resource_epochs, node_policy
    )
    (root / "slurm-submission-preview.json").write_text(
        json.dumps(submission_preview, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for path in sorted(root.glob("*.slurm")):
        text = path.read_text(encoding="utf-8")
        request = script_requests.get(path.name, {
            "requested_wall_minutes": _existing_wall_minutes(path),
            "requested_memory_gib": _existing_memory_gib(path),
            "planner_task_ids": [],
            "aggregation": "static worker fallback",
        })
        requested_wall_minutes = float(request["requested_wall_minutes"])
        route = _partition_for_request(
            path.name,
            requested_wall_minutes,
            float(request["requested_memory_gib"]),
            partitions,
            partition_limits,
            partition_node_limits,
            resource_policy,
            requested_nodes=int(request.get("node_count", 1)),
        )
        partition = route["selected_partition"]
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
        text = _replace_sbatch(text, "nodes", "1")
        text = _replace_sbatch(
            text, "cpus-per-task", str(int(request.get("cpu_slots", 1)))
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
        text = _inject_package_source_validation(text)
        path.write_text(text, encoding="utf-8")
        request.update(route)
        request["slurm_time"] = _format_slurm_time(
            float(request["requested_wall_minutes"])
        )
        request["slurm_memory"] = f"{int(math.ceil(float(request['requested_memory_gib'])))}G"

    submit_command = str(profile["submit_command"])
    for path in sorted(root.glob("submit*.sh")):
        text = path.read_text(encoding="utf-8")
        text = _split_tiered_array_submissions(text, submission_tiers)
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
        text = _inject_package_source_validation(text)
        path.write_text(text, encoding="utf-8")
    canonical_submit = root / "submit.sh"
    if canonical_submit.is_file():
        canonical_submit.write_text(
            _render_resource_bounded_submit(
                root,
                profile,
                profile_path,
                resource_epochs,
                bool(submission_preview["submission_permitted"]),
                bool(execution_plan.get("autorecovery", True)),
            ),
            encoding="utf-8",
        )
        os.chmod(canonical_submit, 0o755)
    return {
        "scheduler_resource_requests_schema": "salsbury-scheduler-resource-requests-v7",
        "dependency_model": execution_plan.get(
            "dependency_model", "legacy_phase_chain"
        ),
        "profile_id": profile["profile_id"],
        "cluster_name": profile["cluster_name"],
        "resource_policy": dict(resource_policy),
        "node_policy": dict(node_policy),
        "partition_maximum_wall_minutes": dict(partition_limits),
        "partition_maximum_nodes": dict(partition_node_limits),
        "tasks": _task_resource_requests(execution_plan),
        "scripts": script_requests,
        "submission_resource_tiers": submission_tiers,
        "maximum_parallel_cpus": execution_plan["maximum_parallel_cpus"],
        "maximum_parallel_memory_gib": execution_plan[
            "maximum_parallel_memory_gib"
        ],
        "walltime_allocation": dict(
            execution_plan.get("walltime_allocation", {})
        ),
        "resource_waves": resource_epochs,
        "resource_epochs": resource_epochs,
        "resource_lanes": [
            lane
            for epoch in resource_epochs
            for lane in epoch.get("lanes", [])
        ],
        "resource_token_schedules": [
            {
                "resource_epoch_index": epoch.get("resource_epoch_index"),
                "policy": dict(epoch.get("resource_token_policy", {})),
                "scheduled_items": list(epoch.get("scheduled_items", [])),
            }
            for epoch in resource_epochs
            if epoch.get("scheduled_items")
        ],
        "submission_preview": submission_preview,
        "submission_preview_file": "slurm-submission-preview.json",
        "canonical_submit_script": (
            "submit.sh" if canonical_submit.is_file() else None
        ),
        "autorecovery": {
            "enabled": bool(execution_plan.get("autorecovery", True)),
            "maximum_task_attempts": int(
                execution_plan.get("maximum_task_attempts", 2)
            ),
            "task_runner": "run-task-with-recovery.sh",
        },
        "aggregate_resource_contract": (
            "the complete task DAG is guarded by global and per-node CPU and "
            "padded-memory tokens; tasks "
            "wait only for true prerequisites, explicit completion-only inputs, "
            "and prior users of the resource tokens they acquire, so unrelated "
            "long tasks do not form whole-depth barriers; resource-only edges use "
            "afterany while scientific inputs remain afterok"
            if execution_plan.get("dependency_model") == "task_dag_v1" else
            "tasks in one wave may run concurrently; every later wave waits "
            "afterany for every job in the preceding wave, and each wave stays "
            "within both campaign CPU and safety-adjusted memory limits"
        ),
        "large_memory_routing": (
            "mixed-resource arrays are submitted as resource-matched subarrays; "
            "only tiers at or above the configured threshold use the large-memory role"
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


def _script_scalar(path: Path, name: str) -> Optional[str]:
    """Read one generated, literal shell assignment without executing it."""

    match = re.search(
        rf"^{re.escape(name)}=(.+)$",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    try:
        values = shlex.split(match.group(1), posix=True)
    except ValueError as exc:
        raise ExecutionAdapterError(
            f"cannot parse {name} assignment in {path.name}"
        ) from exc
    return values[0] if len(values) == 1 else None


def _task_command(root: Path, task: Mapping[str, object]) -> Optional[str]:
    array_task_id = task.get("array_task_id")
    if array_task_id is None:
        match = re.fullmatch(
            r"run_reporting_(.+)\.slurm", str(task["script"])
        )
        if match:
            return match.group(1)
        return None
    values = _bash_array_values(root / str(task["script"]), "COMMANDS")
    index = int(array_task_id)
    if index < 0 or index >= len(values):
        raise ExecutionAdapterError(
            f"array task {index} is outside COMMANDS in {task['script']}"
        )
    return values[index]


def _view_id(script: str) -> Optional[str]:
    match = re.fullmatch(r"run_view_(.+)_stage_\d+\.slurm", script)
    return match.group(1) if match else None


def _task_project_filename(root: Path, task: Mapping[str, object]) -> Optional[str]:
    script = str(task["script"])
    path = root / script
    array_task_id = task.get("array_task_id")
    if (
        array_task_id is not None
        and re.search(r"^PROJECTS=\($", path.read_text(encoding="utf-8"), re.MULTILINE)
    ):
        projects = _bash_array_values(path, "PROJECTS")
        index = int(array_task_id)
        if index >= len(projects):
            raise ExecutionAdapterError(
                f"array task {index} is outside PROJECTS in {script}"
            )
        return projects[index]
    project = _script_scalar(path, "PROJECT")
    if project:
        candidate = Path(project)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            return str(candidate.resolve(strict=False).relative_to(root))
        except ValueError:
            return str(candidate.resolve(strict=False))
    return None


def _project_cached_modules(
    root: Path, project_filename: Optional[str], module_id: str,
) -> set[str]:
    """Resolve upstream reports that a project task can reuse after completion.

    Project runners retain a validated compute-from-project fallback when an
    upstream report is absent or unusable.  These relationships therefore
    order cache reuse, but they are not success-only scientific gates.
    """

    if not project_filename:
        return set()
    project_path = Path(project_filename)
    if not project_path.is_absolute():
        project_path = root / project_path
    if not project_path.is_file():
        return set()
    project = load_json(project_path)
    definitions = project.get("definitions") if isinstance(project, dict) else None
    if not isinstance(definitions, dict):
        return set()
    if module_id == "markov_state_models":
        return {
            candidate for candidate in (
                "pca_fes_basins", "clustering_kmeans", "clustering_hdbscan",
                "clustering_imwkmeans", "alternative_clustering",
            ) if candidate in definitions
        }
    if module_id in {"representative_frames", "state_coordinate_exports"}:
        definition = definitions.get(module_id)
        source = definition.get("source") if isinstance(definition, dict) else None
        return {str(source)} if isinstance(source, str) else set()
    if module_id == "grouped_ml":
        return {"clustering_kmeans"}
    return set()


def _view_preflight_requires_coordinate_cache(
    root: Path, task: Mapping[str, object]
) -> bool:
    """Return whether one generated view preflight validates cache-built input."""

    manifest = _script_scalar(root / str(task["script"]), "MANIFEST")
    if not manifest:
        return False
    source = Path(manifest)
    if not source.is_absolute():
        source = root / source
    cache_root = (root / "coordinate-cache").resolve(strict=False)
    try:
        source.resolve(strict=False).relative_to(cache_root)
    except ValueError:
        return False
    return True


def _apply_task_dependency_graph(
    root: Path, phases: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Assign stable task IDs and only the report-producing prerequisites used."""

    config_path = root / "analysis-config.json"
    config = load_json(config_path) if config_path.is_file() else {}
    module_config = config.get("modules", {}) if isinstance(config, dict) else {}
    if not isinstance(module_config, dict):
        module_config = {}
    tasks: List[Dict[str, object]] = []
    for phase in phases:
        source_phase = str(phase["phase_id"])
        for source_index, raw in enumerate(phase.get("tasks", [])):
            task = dict(raw)
            script = str(task["script"])
            array_id = task.get("array_task_id")
            suffix = "single" if array_id is None else str(array_id)
            task["task_id"] = f"task:{script}:{suffix}"
            task["source_phase_id"] = source_phase
            task["source_phase_task_index"] = source_index
            command = _task_command(root, task)
            task["command"] = command
            task["module_id"] = (
                command if script.startswith("run_reporting_")
                else COMMAND_MODULES.get(command) if command else None
            )
            task = annotate_task_parallelism(task)
            project_filename = _task_project_filename(root, task)
            task["project_filename"] = project_filename
            view = _view_id(script)
            task["scope_id"] = (
                f"view:{view}" if view else
                f"project:{project_filename}" if project_filename else
                "base"
            )
            tasks.append(task)

    by_script = {str(task["script"]): task for task in tasks if task.get("array_task_id") is None}
    cache_task = by_script.get("run_coordinate_cache.slurm")
    preflight_task = by_script.get("run_preflight.slurm")
    final_task = by_script.get("run_finalize_reporting.slurm")
    module_tasks: Dict[tuple[str, str], List[str]] = {}
    module_tasks_by_module: Dict[str, List[str]] = {}
    for task in tasks:
        module_id = task.get("module_id")
        if isinstance(module_id, str):
            module_tasks.setdefault(
                (str(task["scope_id"]), module_id), []
            ).append(str(task["task_id"]))
            module_tasks_by_module.setdefault(module_id, []).append(
                str(task["task_id"])
            )

    view_preflights: Dict[str, str] = {}
    for task in tasks:
        if not str(task["script"]).startswith("run_view_preflight_"):
            continue
        final_path = _script_scalar(root / str(task["script"]), "FINAL")
        if final_path:
            view_preflights[str(Path(final_path).resolve(strict=False))] = str(
                task["task_id"]
            )

    all_nonfinal = [
        str(task["task_id"]) for task in tasks if task is not final_task
    ]
    for task in tasks:
        dependencies: set[str] = set()
        completion_waits: set[str] = set()
        script = str(task["script"])
        if task is cache_task:
            pass
        elif task is preflight_task:
            # Base analyses use the original system manifest and can proceed
            # while the optional all-frame coordinate cache is being built.
            pass
        elif script.startswith("run_view_preflight_"):
            if (
                cache_task is not None
                and _view_preflight_requires_coordinate_cache(root, task)
            ):
                dependencies.add(str(cache_task["task_id"]))
        elif task.get("module_id") == "rmsf_permutation_inference":
            # Reporting jobs do not have a project filename, while the pooled
            # RMSF producer is scoped to project.json.  Matching the literal
            # ``base`` scope therefore released inference at dependency level
            # zero.  The report consumes the campaign-level pooled RMSF file,
            # so every generated pooled RMSF producer is a true prerequisite.
            dependencies.update(module_tasks_by_module.get("pooled_rmsf", []))
        elif task.get("module_id") == "integrated_comparison":
            completion_waits.update(
                str(candidate["task_id"])
                for candidate in tasks
                if isinstance(candidate.get("module_id"), str)
                and candidate.get("module_id") not in {
                    "structural_integrity_qc", "rmsf_permutation_inference",
                    "integrated_comparison",
                }
                and not str(candidate["script"]).startswith("run_reporting_")
            )
        elif task is final_task:
            completion_waits.update(all_nonfinal)
        else:
            view = _view_id(script)
            if view:
                project_filename = task.get("project_filename")
                project_path = (
                    Path(str(project_filename)) if project_filename else None
                )
                if project_path is not None and not project_path.is_absolute():
                    project_path = root / project_path
                system_manifest = None
                if project_path is not None and project_path.is_file():
                    project = load_json(project_path)
                    if isinstance(project, dict):
                        system_manifest = project.get("system_manifest")
                if system_manifest == "system.json" and preflight_task is not None:
                    dependencies.add(str(preflight_task["task_id"]))
                elif isinstance(system_manifest, str):
                    report_path = root / f"preflight-{Path(system_manifest).stem}.report.json"
                    preflight_id = view_preflights.get(
                        str(report_path.resolve(strict=False))
                    )
                    if preflight_id:
                        dependencies.add(preflight_id)
            elif preflight_task is not None:
                dependencies.add(str(preflight_task["task_id"]))

            project_filename = task.get("project_filename")
            if (
                cache_task is not None
                and isinstance(project_filename, str)
                and Path(project_filename).name == "project-cache-base.json"
            ):
                dependencies.add(str(cache_task["task_id"]))

            if (
                task.get("module_id") == "structural_integrity_qc"
                and cache_task is not None
            ):
                dependencies.add(str(cache_task["task_id"]))

            module_id = task.get("module_id")
            required_modules: set[str] = set()
            if isinstance(module_id, str):
                row = module_config.get(module_id)
                if isinstance(row, dict) and isinstance(row.get("depends_on"), list):
                    required_modules.update(map(str, row["depends_on"]))
                required_modules.update(_project_cached_modules(
                    root, task.get("project_filename"), module_id
                ))
                if module_id in {
                    "scalar_feature_distributions", "scalar_threshold_states",
                }:
                    required_modules.add("trajectory_features")
            for requirement in required_modules:
                # Generated workers validate and reuse these reports when they
                # are complete.  If the producer fails, the consumer unsets
                # the cache variable and recomputes from its project inputs.
                # Waiting for completion avoids duplicate work without making
                # an unrelated producer failure a false afterok gate.
                completion_waits.update(module_tasks.get(
                    (str(task["scope_id"]), requirement), []
                ))
        dependencies.discard(str(task["task_id"]))
        completion_waits.discard(str(task["task_id"]))
        task["depends_on_task_ids"] = sorted(dependencies)
        task["wait_for_task_ids"] = sorted(completion_waits)

    task_by_id = {str(task["task_id"]): task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ExecutionAdapterError("generated workflow contains duplicate task IDs")
    depth_cache: Dict[str, int] = {}
    visiting: set[str] = set()

    def depth(task_id: str) -> int:
        if task_id in depth_cache:
            return depth_cache[task_id]
        if task_id in visiting:
            raise ExecutionAdapterError("generated task dependency graph contains a cycle")
        visiting.add(task_id)
        task = task_by_id[task_id]
        requirements = list(task.get("depends_on_task_ids", [])) + list(
            task.get("wait_for_task_ids", [])
        )
        missing = [value for value in requirements if value not in task_by_id]
        if missing:
            raise ExecutionAdapterError(
                f"task {task_id} has unknown dependencies: {', '.join(missing)}"
            )
        value = 0 if not requirements else 1 + max(depth(str(item)) for item in requirements)
        visiting.remove(task_id)
        depth_cache[task_id] = value
        return value

    levels: Dict[int, List[Dict[str, object]]] = {}
    for task in tasks:
        level = depth(str(task["task_id"]))
        task["dependency_level"] = level
        levels.setdefault(level, []).append(task)
    return [
        {
            "phase_id": f"dependency_level_{level:03d}",
            "tasks": sorted(
                levels[level],
                key=lambda task: (
                    str(task["script"]), str(task.get("array_task_id"))
                ),
            ),
        }
        for level in sorted(levels)
    ]


def build_local_execution_plan(
    root: Path,
    execution: Mapping[str, object],
    reporting: Mapping[str, object],
    resource_policy: Optional[Mapping[str, object]] = None,
    node_policy: Optional[Mapping[str, object]] = None,
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

    reporting_components = []
    reporting_outputs = {
        "rmsf_permutation_inference": (
            "results/rmsf-permutation-inference/report.json"
        ),
        "integrated_comparison": "results/integrated-comparison/report.json",
    }
    for path in sorted(root.glob("run_reporting_*.slurm")):
        task = _task(path)
        reporting_id = _task_command(root, task)
        output = reporting_outputs.get(str(reporting_id))
        if output:
            task["completion_reports"] = [output]
        reporting_components.append(task)
    if reporting_components:
        phases.append({
            "phase_id": "independent_reporting",
            "tasks": reporting_components,
        })

    finalizer = root / "run_finalize_reporting.slurm"
    if not finalizer.is_file():
        raise ExecutionAdapterError("generated workflow lacks run_finalize_reporting.slurm")
    completion_reports = []
    if bool(reporting.get("resource_table_enabled")):
        completion_reports.append("final-resource-summary.json")
    if bool(reporting.get("finding_picker_enabled")):
        completion_reports.append("final-findings-summary.json")
    if "rmsf-permutation-from-report" in finalizer.read_text(encoding="utf-8"):
        completion_reports.append("results/rmsf-permutation-inference/report.json")
    if not completion_reports:
        completion_reports.append("final-reporting-disabled.json")
    final_task = _task(finalizer)
    final_task["completion_reports"] = completion_reports
    phases.append({"phase_id": "final_reporting", "tasks": [final_task]})
    phases = _apply_task_dependency_graph(root, phases)
    maximum_cpus = int(execution["maximum_parallel_cpus"])
    if any(int(task["cpu_slots"]) > maximum_cpus for phase in phases for task in phase["tasks"]):
        raise ExecutionAdapterError("a local task requests more CPUs than the campaign limit")
    policy = dict(_RESOURCE_POLICY_DEFAULTS)
    if resource_policy is not None:
        policy.update(resource_policy)
    rows = _planner_rows(root)
    for phase in phases:
        phase["tasks"] = [
            _enrich_task_resources(
                root, task, rows, execution, policy, node_policy
            )
            for task in phase["tasks"]
        ]
    plan: Dict[str, object] = {
        "local_execution_plan_schema": "salsbury-local-execution-plan-v6",
        "dependency_model": "task_dag_v1",
        "maximum_parallel_cpus": maximum_cpus,
        "maximum_campaign_wall_hours": float(execution["maximum_hours_per_cpu"]),
        "maximum_parallel_memory_gib": float(execution["maximum_memory_gib"]),
        "autorecovery": bool(execution.get("autorecovery", True)),
        "maximum_task_attempts": int(execution.get("maximum_task_attempts", 2)),
        "resource_policy": policy,
        "node_policy": dict(node_policy or {}),
        "phases": phases,
        "dependency_policy": (
            "depends_on_task_ids contains only success-required inputs that a task "
            "cannot reconstruct; wait_for_task_ids covers validated cache reuse, "
            "completion-only report collation, and other failure-tolerant ordering; "
            "resource waves may serialize otherwise independent tasks without "
            "creating scientific dependencies"
        ),
        "autorecovery_policy": (
            "Retry only a failed or timed-out task; keep completed outputs, "
            "checkpoint identities, scientific dependencies, and unrelated "
            "tasks unchanged. A later attempt receives twice the prior local "
            "wall allowance but remains inside the campaign deadline."
            if bool(execution.get("autorecovery", True)) else
            "Disabled explicitly; each task receives one attempt."
        ),
    }
    _fit_walltime_requests_to_campaign(plan)
    return plan


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
        None if profile is None else profile["node_policy"],
    )
    (root / "local-execution-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract_phases = []
    for phase in plan["phases"]:
        phase_id = str(phase["phase_id"])
        tasks = []
        for index, task in enumerate(phase["tasks"]):
            cpu_slots = int(task["cpu_slots"])
            memory_gib = float(task.get("requested_memory_gib", 1.0))
            wall_minutes = float(task.get("requested_wall_minutes", 30.0))
            array_task_id = task.get("array_task_id")
            environment = {
                "SLURM_CPUS_PER_TASK": str(cpu_slots),
                "SLURM_MEM_PER_NODE": str(int(math.ceil(memory_gib * 1024.0))),
                "SLURM_TIMELIMIT": str(max(1, int(math.ceil(wall_minutes)))),
            }
            if array_task_id is not None:
                environment["SLURM_ARRAY_TASK_ID"] = str(array_task_id)
            tasks.append({
                "launcher_task_id": task.get("task_id", f"{phase_id}:{index}"),
                "depends_on_task_ids": task.get("depends_on_task_ids", []),
                "wait_for_task_ids": task.get("wait_for_task_ids", []),
                "source_phase_id": task.get("source_phase_id"),
                "module_id": task.get("module_id"),
                "command": task.get("command"),
                "argv": ["bash", str(task["script"])],
                "working_directory": str(root),
                "environment": environment,
                "launcher_assigned_environment": [
                    "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_CLUSTER_NAME"
                ],
                "script": task["script"],
                "array_task_id": array_task_id,
                "cpu_slots": cpu_slots,
                "requested_memory_gib": memory_gib,
                "requested_wall_minutes": wall_minutes,
                "planner_task_ids": task.get("planner_task_ids", []),
                "completion_reports": task.get("completion_reports", []),
            })
        contract_phases.append({
            "phase_id": phase_id,
            "depends_on": [],
            "tasks": tasks,
        })
    launcher_contract = {
        "launcher_contract_schema": "salsbury-external-launcher-contract-v2",
        "dependency_model": plan.get("dependency_model"),
        "analysis_root": str(root),
        "resource_envelope": {
            "maximum_parallel_cpus": plan["maximum_parallel_cpus"],
            "maximum_parallel_memory_gib": plan["maximum_parallel_memory_gib"],
            "maximum_campaign_wall_hours": plan["maximum_campaign_wall_hours"],
        },
        "dependency_policy": (
            "depends_on_task_ids are success-required inputs without a local "
            "recompute path; phase order is topological metadata, and "
            "wait_for_task_ids delay a task for validated cache reuse or other "
            "completion without requiring upstream success"
        ),
        "task_success_policy": (
            "a task succeeds only with exit code zero; skip only tasks whose own "
            "depends_on_task_ids failed or timed out, and continue unrelated tasks"
        ),
        "autorecovery": {
            "enabled": bool(execution.get("autorecovery", True)),
            "maximum_task_attempts": int(
                execution.get("maximum_task_attempts", 2)
            ),
            "retry_scope": "failed_or_timed_out_task_only",
            "completed_tasks_are_never_rerun": True,
            "scientific_dependencies_remain_success_required": True,
        },
        "environment_contract": {
            "compatibility_note": (
                "Worker scripts use Slurm-compatible environment names even when "
                "the external launcher is not Slurm."
            ),
            "launcher_assigned_values": {
                "SLURM_JOB_ID": "unique task-attempt identifier",
                "SLURM_ARRAY_JOB_ID": "stable identifier shared by one array script",
                "SLURM_CLUSTER_NAME": "external launcher or site name",
            },
            "optional_user_overrides": [
                "SALSBURY_MD_ANALYSIS_PYTHON",
                "SALSBURY_MD_ANALYSIS_PYTHONPATH",
            ],
        },
        "phases": contract_phases,
    }
    (root / "launcher-contract.json").write_text(
        json.dumps(launcher_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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

    custom_launcher = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={shlex.quote(str(root))}
CONTRACT="$ROOT/launcher-contract.json"
LAUNCHER="${{SALSBURY_MD_ANALYSIS_CUSTOM_LAUNCHER:-}}"
if [[ -z "$LAUNCHER" ]]; then
  printf 'Set SALSBURY_MD_ANALYSIS_CUSTOM_LAUNCHER to an executable that accepts launcher-contract.json.\n' >&2
  exit 2
fi
if [[ ! -x "$LAUNCHER" ]]; then
  printf 'Custom launcher is not executable: %s\n' "$LAUNCHER" >&2
  exit 2
fi
exec "$LAUNCHER" "$CONTRACT"
"""
    (root / "run-custom.sh").write_text(custom_launcher, encoding="utf-8")
    os.chmod(root / "run-custom.sh", 0o755)

    recovery_runner = _render_task_recovery_runner(
        autorecovery=bool(execution.get("autorecovery", True)),
        maximum_task_attempts=int(execution.get("maximum_task_attempts", 2)),
    )
    (root / "run-task-with-recovery.sh").write_text(
        recovery_runner, encoding="utf-8"
    )
    os.chmod(root / "run-task-with-recovery.sh", 0o755)

    profile_id = None
    generated = [
        "local-execution-plan.json", "launcher-contract.json",
        "run-local.sh", "run-custom.sh", "run-task-with-recovery.sh",
    ]
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
        generated.extend([
            "slurm-profile.json",
            "scheduler-resource-requests.json",
            "slurm-submission-preview.json",
        ])
    metadata = {
        "execution_adapter_schema": "salsbury-execution-adapter-v1",
        "active_adapter": adapter,
        "slurm_profile_id": profile_id,
        "local_launcher": "run-local.sh",
        "slurm_launcher": "submit.sh",
        "custom_launcher": "run-custom.sh",
        "shared_resource_plan": "local-execution-plan.json",
        "external_launcher_contract": "launcher-contract.json",
        "autorecovery": {
            "enabled": bool(execution.get("autorecovery", True)),
            "maximum_task_attempts": int(
                execution.get("maximum_task_attempts", 2)
            ),
            "local_policy": (
                "failed or timed-out tasks only; later attempts receive a "
                "larger wall allowance within the unchanged campaign deadline"
            ),
            "slurm_policy": (
                "failed tasks retry inside their allocation; a pre-timeout "
                "signal requeues checkpoint-capable work within the attempt cap"
            ),
        },
        "scheduler_resource_requests": (
            "scheduler-resource-requests.json" if adapter == "slurm" else None
        ),
        "local_resource_controls": [
            "maximum_parallel_cpus", "maximum_parallel_memory_gib",
            "per_task_wall_minutes", "complete_campaign_wall_deadline",
        ],
        "scientific_workflow_identity": (
            "all launchers execute the same generated worker scripts and output contracts"
        ),
    }
    (root / "execution-adapter.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generated.append("execution-adapter.json")
    return {
        "adapter": adapter,
        "generated_files": generated,
        "next_command": (
            f"cd {root} && ./"
            f"{'submit.sh' if adapter == 'slurm' else 'run-custom.sh' if adapter == 'custom' else 'run-local.sh'}"
        ),
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
    autorecovery: bool,
    maximum_task_attempts: int,
) -> Dict[str, object]:
    cpu_slots = int(task["cpu_slots"])
    memory_gib = float(task.get("requested_memory_gib", 1.0))
    wall_minutes = float(task.get("requested_wall_minutes", 30.0))
    completion_reports = task.get("completion_reports", [])
    identity = {
        "task_id": task.get("task_id"),
        "depends_on_task_ids": list(task.get("depends_on_task_ids", [])),
        "wait_for_task_ids": list(task.get("wait_for_task_ids", [])),
    }
    if isinstance(completion_reports, list) and completion_reports and _reports_complete(
        root, [str(value) for value in completion_reports]
    ):
        return {
            **identity,
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
        task_start = time.monotonic()
        attempts = []
        exit_code = None
        timed_out = False
        attempt_limit = maximum_task_attempts if autorecovery else 1
        for attempt_number in range(1, attempt_limit + 1):
            attempt_start = time.monotonic()
            attempt_wall_minutes = wall_minutes * (2 ** (attempt_number - 1))
            timeout_seconds = min(
                deadline - attempt_start, attempt_wall_minutes * 60.0
            )
            stdout_path = root / "logs" / f"{stem}-attempt-{attempt_number:02d}.out"
            stderr_path = root / "logs" / f"{stem}-attempt-{attempt_number:02d}.err"
            if timeout_seconds <= 0.0:
                timed_out = True
                attempts.append({
                    "attempt_number": attempt_number,
                    "status": "timed_out",
                    "exit_code": None,
                    "wall_seconds": 0.0,
                    "allowed_wall_minutes": 0.0,
                    "stdout": str(stdout_path.relative_to(root)),
                    "stderr": str(stderr_path.relative_to(root)),
                })
                break
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
            attempt_status = (
                "timed_out" if timed_out else
                "complete" if exit_code == 0 else "failed"
            )
            attempts.append({
                "attempt_number": attempt_number,
                "status": attempt_status,
                "exit_code": exit_code,
                "wall_seconds": time.monotonic() - attempt_start,
                "allowed_wall_minutes": timeout_seconds / 60.0,
                "stdout": str(stdout_path.relative_to(root)),
                "stderr": str(stderr_path.relative_to(root)),
                "stdout_size_bytes": stdout_path.stat().st_size,
                "stderr_size_bytes": stderr_path.stat().st_size,
            })
            if exit_code == 0 and not timed_out:
                break
        recovered = len(attempts) > 1 and attempts[-1]["status"] == "complete"
        final_status = (
            "recovered_complete" if recovered else str(attempts[-1]["status"])
        )
        return {
            **identity,
            "script": task["script"], "array_task_id": task.get("array_task_id"),
            "cpu_slots": cpu_slots, "requested_memory_gib": memory_gib,
            "requested_wall_minutes": wall_minutes,
            "planner_task_ids": task.get("planner_task_ids", []),
            "status": final_status,
            "exit_code": exit_code,
            "wall_seconds": time.monotonic() - task_start,
            "autorecovery_enabled": autorecovery,
            "maximum_task_attempts": maximum_task_attempts,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "stdout": attempts[-1]["stdout"],
            "stderr": attempts[-1]["stderr"],
        }
    finally:
        slots.release(cpu_slots, memory_gib)


def run_local_workflow(root: Path) -> Dict[str, object]:
    """Execute a generated workflow locally while respecting its CPU envelope."""

    resolved = root.expanduser().resolve(strict=True)
    plan_path = resolved / "local-execution-plan.json"
    plan = load_json(plan_path)
    accepted_schemas = {
        "salsbury-local-execution-plan-v1",
        "salsbury-local-execution-plan-v2",
        "salsbury-local-execution-plan-v3",
        "salsbury-local-execution-plan-v4",
        "salsbury-local-execution-plan-v5",
        "salsbury-local-execution-plan-v6",
    }
    if not isinstance(plan, dict) or plan.get("local_execution_plan_schema") not in accepted_schemas:
        raise ExecutionAdapterError("local execution plan is invalid")
    maximum_cpus = int(plan["maximum_parallel_cpus"])
    maximum_memory_gib = float(plan.get("maximum_parallel_memory_gib", 1.0e12))
    campaign_seconds = float(plan["maximum_campaign_wall_hours"]) * 3600.0
    autorecovery = bool(plan.get("autorecovery", False))
    maximum_task_attempts = int(plan.get("maximum_task_attempts", 1))
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
    dependency_dag = plan.get("dependency_model") == "task_dag_v1"
    task_statuses: Dict[str, str] = {}
    if dependency_dag:
        task_ids = [
            str(task.get("task_id"))
            for phase in plan.get("phases", []) for task in phase.get("tasks", [])
        ]
        if any(value == "None" for value in task_ids) or len(set(task_ids)) != len(task_ids):
            raise ExecutionAdapterError(
                "task-DAG execution plans require unique nonempty task IDs"
            )
        known = set(task_ids)
        for phase in plan.get("phases", []):
            for task in phase.get("tasks", []):
                missing = set(map(str, task.get("depends_on_task_ids", []))).difference(known)
                missing.update(
                    set(map(str, task.get("wait_for_task_ids", []))).difference(known)
                )
                if missing:
                    raise ExecutionAdapterError(
                        f"task {task['task_id']} has unknown dependencies: "
                        + ", ".join(sorted(missing))
                    )
    for phase in plan.get("phases", []):
        phase_id = str(phase["phase_id"])
        tasks = phase["tasks"]
        if not isinstance(tasks, list) or not tasks:
            raise ExecutionAdapterError(f"local phase {phase_id} has no tasks")
        results = []
        runnable = []
        for index, task in enumerate(tasks):
            failed_requirements = [
                str(required) for required in task.get("depends_on_task_ids", [])
                if task_statuses.get(str(required)) not in {
                    "complete", "recovered_complete", "reused_complete"
                }
            ] if dependency_dag else []
            if failed_requirements:
                results.append({
                    "task_id": task.get("task_id"),
                    "depends_on_task_ids": list(task.get("depends_on_task_ids", [])),
                    "wait_for_task_ids": list(task.get("wait_for_task_ids", [])),
                    "script": task["script"],
                    "array_task_id": task.get("array_task_id"),
                    "cpu_slots": task.get("cpu_slots"),
                    "requested_memory_gib": task.get("requested_memory_gib", 1.0),
                    "requested_wall_minutes": task.get("requested_wall_minutes", 30.0),
                    "status": "skipped_dependency",
                    "failed_dependency_task_ids": failed_requirements,
                    "exit_code": None,
                    "wall_seconds": 0.0,
                })
            else:
                runnable.append((index, task))
        with ThreadPoolExecutor(max_workers=min(len(tasks), maximum_cpus)) as executor:
            futures = {
                executor.submit(
                    _run_local_task, resolved, task, phase_id, index,
                    attempt_id, deadline, slots, autorecovery,
                    maximum_task_attempts,
                ): index
                for index, task in runnable
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # preserve a machine-readable failed attempt
                    results.append({
                        "task_id": tasks[futures[future]].get("task_id"),
                        "depends_on_task_ids": list(
                            tasks[futures[future]].get("depends_on_task_ids", [])
                        ),
                        "wait_for_task_ids": list(
                            tasks[futures[future]].get("wait_for_task_ids", [])
                        ),
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
            "complete" if all(row["status"] in {
                "complete", "recovered_complete", "reused_complete",
            } for row in results)
            else "failed"
        )
        phase_reports.append({
            "phase_id": phase_id, "technical_status": phase_status, "tasks": results
        })
        if dependency_dag:
            task_statuses.update({
                str(row["task_id"]): str(row["status"])
                for row in results if row.get("task_id")
            })
        if phase_status != "complete":
            technical_status = "failed"
            if not dependency_dag:
                break
    report = {
        "local_execution_status_schema": "salsbury-local-execution-status-v2",
        "technical_status": technical_status,
        "scientific_status": "not evaluated",
        "dependency_model": plan.get("dependency_model", "legacy_phase_chain"),
        "attempt_id": attempt_id,
        "analysis_root": str(resolved),
        "maximum_parallel_cpus": maximum_cpus,
        "maximum_parallel_memory_gib": maximum_memory_gib,
        "autorecovery": {
            "enabled": autorecovery,
            "maximum_task_attempts": maximum_task_attempts,
            "recovered_task_count": sum(
                row.get("status") == "recovered_complete"
                for phase in phase_reports for row in phase["tasks"]
            ),
            "terminal_failure_count": sum(
                row.get("status") in {
                    "failed", "timed_out", "skipped_dependency",
                }
                for phase in phase_reports for row in phase["tasks"]
            ),
        },
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
