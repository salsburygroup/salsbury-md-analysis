"""Optional, read-only Slurm capacity advice for prepared campaigns."""

from __future__ import annotations

import getpass
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .execution_adapters import load_slurm_profile
from .manifests import load_json
from .resource_planning import plan_campaign_resource_budget


class SlurmCapacityError(ValueError):
    """Raised when a capacity request or scheduler response is invalid."""


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command), check=False, capture_output=True, text=True, timeout=30.0
    )


def _positive_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise SlurmCapacityError(f"{label} must be finite and positive")
    return float(value)


def _command_sibling(command: object, sibling: str) -> str:
    path = Path(str(command))
    return str(path.with_name(sibling)) if path.is_absolute() else sibling


def _run_read_only(
    runner: CommandRunner,
    command: Sequence[str],
    warnings: List[str],
) -> Optional[str]:
    try:
        result = runner(command)
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.append(f"{' '.join(command[:2])} unavailable: {exc}")
        return None
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "unknown error").strip()
        warnings.append(
            f"{' '.join(command[:2])} returned {result.returncode}: {message}"
        )
        return None
    return result.stdout


def _key_values(line: str) -> Dict[str, str]:
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", line)
    }


def _tres_cpu(value: object) -> Optional[int]:
    if not isinstance(value, str) or not value:
        return None
    match = re.search(r"(?:^|,)cpu=(\d+)(?:,|$)", value)
    return int(match.group(1)) if match else None


def _parse_partition(line: str) -> Dict[str, object]:
    values = _key_values(line)
    try:
        total_cpus = int(values["TotalCPUs"])
        max_nodes_text = values.get("MaxNodes", "UNLIMITED")
        max_nodes = None if max_nodes_text == "UNLIMITED" else int(max_nodes_text)
    except (KeyError, ValueError) as exc:
        raise SlurmCapacityError("Slurm partition response lacks valid CPU limits") from exc
    return {
        "partition": values.get("PartitionName"),
        "total_configured_cpus": total_cpus,
        "maximum_nodes_per_job": max_nodes,
        "maximum_time": values.get("MaxTime"),
        "state": values.get("State"),
    }


def _parse_node(line: str) -> Optional[Dict[str, object]]:
    values = _key_values(line)
    if "NodeName" not in values:
        return None
    try:
        total_cpus = int(values["CPUTot"])
        allocated_cpus = int(values.get("CPUAlloc", "0"))
        real_memory = int(values["RealMemory"])
        allocated_memory = int(values.get("AllocMem", "0"))
        free_memory = int(values.get("FreeMem", "0"))
    except (KeyError, ValueError):
        return None
    state = values.get("State", "UNKNOWN")
    partitions = values.get("Partitions", "")
    return {
        "node": values["NodeName"],
        "partitions": [value for value in partitions.split(",") if value],
        "state": state,
        "configured_cpus": total_cpus,
        "allocated_cpus": allocated_cpus,
        "idle_cpus": max(0, total_cpus - allocated_cpus),
        "real_memory_mib": real_memory,
        "allocated_memory_mib": allocated_memory,
        "scheduler_unallocated_memory_mib": max(0, real_memory - allocated_memory),
        "observed_free_memory_mib": max(0, free_memory),
    }


def _eligible_node(node: Mapping[str, object]) -> bool:
    state = str(node.get("state", "")).upper()
    return not any(
        blocked in state
        for blocked in ("DOWN", "DRAIN", "FAIL", "MAINT", "RESV", "UNKNOWN")
    )


def _parse_pipe_rows(text: str, fields: Sequence[str]) -> List[Dict[str, str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.rstrip("\n").split("|")
        if values and values[-1] == "":
            values.pop()
        values.extend([""] * max(0, len(fields) - len(values)))
        rows.append(dict(zip(fields, values)))
    return rows


def _bounded_cpu_limit(values: Iterable[Optional[int]]) -> Optional[int]:
    retained = [value for value in values if value is not None and value > 0]
    return min(retained) if retained else None


def _query_scheduler(
    profile: Mapping[str, object],
    *,
    slurm_user: str,
    job_ids: Sequence[str],
    runner: CommandRunner,
) -> Dict[str, object]:
    warnings: List[str] = []
    submit = str(profile["submit_command"])
    scontrol = _command_sibling(submit, "scontrol")
    sacctmgr = _command_sibling(submit, "sacctmgr")
    squeue = str(profile["status_command"])
    partitions_map = profile["partitions"]
    assert isinstance(partitions_map, Mapping)
    partitions = sorted({
        str(value) for value in partitions_map.values() if value is not None
    })

    partition_rows = []
    for partition in partitions:
        output = _run_read_only(
            runner, [scontrol, "show", "partition", partition, "-o"], warnings
        )
        if output:
            for line in output.splitlines():
                if line.strip():
                    parsed = _parse_partition(line)
                    if parsed.get("partition") == partition:
                        partition_rows.append(parsed)
                        break

    node_output = _run_read_only(
        runner, [scontrol, "show", "nodes", "-o"], warnings
    )
    nodes = []
    if node_output:
        for line in node_output.splitlines():
            parsed = _parse_node(line)
            if parsed is not None and set(parsed["partitions"]).intersection(
                partitions
            ):
                nodes.append(parsed)

    account = profile.get("account")
    association_rows: List[Dict[str, str]] = []
    association_fields = [
        "Cluster", "Account", "User", "Partition", "QOS", "GrpTRES",
        "MaxTRES", "MaxTRESPJ", "MaxTRESPN", "MaxJobs", "MaxSubmitJobs",
    ]
    if account:
        output = _run_read_only(
            runner,
            [
                sacctmgr, "-n", "-P", "show", "assoc",
                f"user={slurm_user}", f"account={account}",
                "format=" + ",".join(association_fields),
            ],
            warnings,
        )
        if output:
            association_rows = _parse_pipe_rows(output, association_fields)

    qos_rows: List[Dict[str, str]] = []
    qos_fields = [
        "Name", "GrpTRES", "MaxTRESPerUser", "MaxTRESPerJob",
        "MaxJobsPerUser", "MaxSubmitJobsPerUser",
    ]
    if profile.get("qos"):
        output = _run_read_only(
            runner,
            [
                sacctmgr, "-n", "-P", "show", "qos", str(profile["qos"]),
                "format=" + ",".join(qos_fields),
            ],
            warnings,
        )
        if output:
            qos_rows = _parse_pipe_rows(output, qos_fields)

    queue_fields = [
        "job_id", "user", "account", "partition", "state", "cpus", "memory",
        "time_limit", "time_left", "expected_start", "reason_or_nodes", "priority",
    ]
    queue_output = _run_read_only(
        runner,
        [
            squeue, "-h", "-p", ",".join(partitions), "-t", "PD,R",
            "-o", "%i|%u|%a|%P|%t|%C|%m|%l|%L|%S|%R|%Q",
        ],
        warnings,
    )
    queue_rows = (
        [] if not queue_output else _parse_pipe_rows(queue_output, queue_fields)
    )

    job_start_rows: List[Dict[str, str]] = []
    if job_ids:
        for job_id in job_ids:
            if not re.fullmatch(r"\d+(?:_[0-9]+)?", job_id):
                raise SlurmCapacityError(f"invalid Slurm job id: {job_id}")
        start_fields = ["job_id", "state", "expected_start", "reason_or_nodes"]
        output = _run_read_only(
            runner,
            [
                squeue, "--start", "-h", "-j", ",".join(job_ids),
                "-o", "%i|%T|%S|%R",
            ],
            warnings,
        )
        if output:
            job_start_rows = _parse_pipe_rows(output, start_fields)

    configured_cpus = sum(int(row["configured_cpus"]) for row in nodes)
    idle_cpus = sum(
        int(row["idle_cpus"]) for row in nodes if _eligible_node(row)
    )
    largest_node_cpus = max(
        (int(row["configured_cpus"]) for row in nodes), default=0
    )
    per_job_partition_limits = []
    for row in partition_rows:
        max_nodes = row["maximum_nodes_per_job"]
        limit = int(row["total_configured_cpus"])
        if max_nodes is not None and largest_node_cpus:
            limit = min(limit, int(max_nodes) * largest_node_cpus)
        per_job_partition_limits.append(limit)
    single_job_cpu_ceiling = max(per_job_partition_limits, default=None)

    association_cpu_ceiling = _bounded_cpu_limit(
        _tres_cpu(row.get(field))
        for row in association_rows
        for field in ("GrpTRES", "MaxTRES")
    )
    qos_cpu_ceiling = _bounded_cpu_limit(
        _tres_cpu(row.get(field))
        for row in qos_rows
        for field in ("GrpTRES", "MaxTRESPerUser")
    )
    concurrent_cpu_ceiling = _bounded_cpu_limit(
        [configured_cpus or None, association_cpu_ceiling, qos_cpu_ceiling]
    )

    return {
        "technical_status": "complete" if not warnings else "partial",
        "cluster_name": profile["cluster_name"],
        "slurm_user": slurm_user,
        "account": account,
        "qos": profile.get("qos"),
        "partitions": partition_rows,
        "nodes": nodes,
        "configured_cpu_count_in_profile_partitions": configured_cpus,
        "currently_idle_cpu_count": idle_cpus,
        "single_job_cpu_ceiling": single_job_cpu_ceiling,
        "association_cpu_ceiling": association_cpu_ceiling,
        "qos_cpu_ceiling": qos_cpu_ceiling,
        "simultaneous_cpu_ceiling": concurrent_cpu_ceiling,
        "queue": {
            "running_job_count": sum(row["state"] == "R" for row in queue_rows),
            "pending_job_count": sum(row["state"] == "PD" for row in queue_rows),
            "user_running_job_count": sum(
                row["state"] == "R" and row["user"] == slurm_user
                for row in queue_rows
            ),
            "user_pending_job_count": sum(
                row["state"] == "PD" and row["user"] == slurm_user
                for row in queue_rows
            ),
            "account_running_job_count": sum(
                row["state"] == "R" and row["account"] == str(account)
                for row in queue_rows
            ),
            "account_pending_job_count": sum(
                row["state"] == "PD" and row["account"] == str(account)
                for row in queue_rows
            ),
            "submitted_job_start_estimates": job_start_rows,
        },
        "warnings": warnings,
        "read_only_commands": ["scontrol", "sacctmgr", "squeue"],
        "jobs_submitted": False,
    }


def _executable_planner_task_ids(prepared: Path) -> Optional[set[str]]:
    """Return planner rows that have a generated scheduler execution task."""

    path = prepared / "scheduler-resource-requests.json"
    if not path.exists():
        return None
    document = load_json(path)
    rows = document.get("tasks") if isinstance(document, Mapping) else None
    if not isinstance(rows, list):
        raise SlurmCapacityError("scheduler-resource-requests.json is invalid")
    task_ids = {
        str(task_id)
        for row in rows
        if isinstance(row, Mapping)
        for task_id in row.get("planner_task_ids", [])
        if isinstance(task_id, str) and task_id
    }
    return task_ids or None


def _workflow_useful_cpu_peak(tasks: Sequence[Mapping[str, object]]) -> int:
    stages: Dict[int, Dict[str, int]] = {}
    for row in tasks:
        stage = int(row["dependency_stage"])
        bundle = str(row.get("execution_bundle_id", row["task_id"]))
        cap = max(1, int(row.get("effective_cpu_cap", 1)))
        bundle_caps = stages.setdefault(stage, {})
        bundle_caps[bundle] = max(cap, bundle_caps.get(bundle, 0))
    values = [sum(bundles.values()) for bundles in stages.values()]
    if not values:
        raise SlurmCapacityError("campaign resource plan lacks executable tasks")
    return max(values)


def _normalize_legacy_memory_models(
    tasks: Sequence[Mapping[str, object]],
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Repair known pre-v81 memory-model metadata for read-only replanning."""

    normalized: List[Dict[str, object]] = []
    adjustments: List[Dict[str, object]] = []
    for original in tasks:
        row = dict(original)
        measured_model = row.get("measured_memory_cost_model")
        if not isinstance(measured_model, Mapping):
            normalized.append(row)
            continue
        model = dict(measured_model)
        evidence = row.get("measured_resource_calibration")
        complete_count = row.get("calibration_complete_measurement_count")
        observation_count = row.get("calibration_observation_coverage")
        if isinstance(evidence, Mapping):
            if complete_count is None:
                complete_count = evidence.get("complete_measurement_count")
            if observation_count is None:
                observation_count = evidence.get("maximum_measured_observation_count")
        censored_only = (
            complete_count is not None
            and int(complete_count) == 0
            and (observation_count is None or int(observation_count) == 0)
        )
        if censored_only:
            row.pop("measured_memory_cost_model", None)
            adjustments.append({
                "task_id": str(row["task_id"]),
                "change": "removed_censored_only_observation_scaling",
            })
        elif not bool(model.get("workload_scaling_applied", False)):
            previous = float(model["calibration_memory_gib"])
            resolved = max(1.0, float(row["estimated_peak_memory_gib"]))
            model["calibration_memory_gib"] = resolved
            model["workload_scaling_applied"] = True
            row["measured_memory_cost_model"] = model
            adjustments.append({
                "task_id": str(row["task_id"]),
                "change": "applied_saved_system_workload_scaling",
                "previous_calibration_memory_gib": previous,
                "resolved_calibration_memory_gib": resolved,
            })
        normalized.append(row)
    return normalized, adjustments


def _scheduler_memory_request(
    working_set_gib: float,
    maximum_memory_gib: float,
    policy: Mapping[str, object],
) -> Dict[str, object]:
    unbounded = max(
        float(policy["minimum_memory_gib"]),
        float(math.ceil(
            working_set_gib * float(policy["memory_safety_factor"])
            + float(policy["memory_overhead_gib"])
        )),
    )
    return {
        "planned_working_set_gib": working_set_gib,
        "scheduler_request_gib": unbounded,
        "unbounded_scheduler_request_gib": unbounded,
        "fits_aggregate_campaign_memory_limit": (
            unbounded <= maximum_memory_gib + 1e-9
        ),
        "safety_margin_clipped_by_campaign_ceiling": False,
    }


def _planned_resource_wave_memory(
    plan: Mapping[str, object], tasks: Sequence[Mapping[str, object]]
) -> Dict[str, object]:
    """Summarize the exact aggregate-memory waves chosen by the replanner."""

    working_by_bundle: Dict[str, float] = {}
    for row in tasks:
        bundle = str(row.get("execution_bundle_id", row["task_id"]))
        working_by_bundle[bundle] = max(
            working_by_bundle.get(bundle, 0.0),
            float(row["estimated_peak_memory_gib_at_selected_observations"]),
        )
    stage_reports = []
    for stage in plan.get("stages", []):
        waves = []
        for wave in stage.get("resource_waves", []):
            bundle_ids = [
                str(item["item_id"]) for item in wave.get("items", [])
            ]
            waves.append({
                "wave_index": int(wave["wave_index"]),
                "cpu_slots": int(wave["cpu_slots"]),
                "scheduler_memory_gib": float(wave["memory_gib"]),
                "planned_working_set_gib": sum(
                    working_by_bundle.get(bundle_id, 0.0)
                    for bundle_id in bundle_ids
                ),
                "wall_hours": float(wave["wall_hours"]),
                "bundle_ids": bundle_ids,
            })
        stage_reports.append({
            "dependency_stage": stage["dependency_stage"],
            "resource_waves": waves,
        })
    all_waves = [
        wave for stage in stage_reports for wave in stage["resource_waves"]
    ]
    scheduler_peak = max(
        (float(wave["scheduler_memory_gib"]) for wave in all_waves),
        default=0.0,
    )
    return {
        "maximum_concurrent_working_set_gib": max(
            (float(wave["planned_working_set_gib"]) for wave in all_waves),
            default=0.0,
        ),
        "maximum_concurrent_scheduler_request_gib": scheduler_peak,
        "maximum_planned_resource_wave_memory_gib": scheduler_peak,
        "maximum_planned_resource_wave_cpus": max(
            (int(wave["cpu_slots"]) for wave in all_waves), default=0
        ),
        "resource_wave_count": len(all_waves),
        "stages": stage_reports,
    }


def _select_partition(
    profile: Mapping[str, object],
    requested_memory_gib: float,
    requested_wall_minutes: float,
) -> Optional[str]:
    partitions = profile["partitions"]
    policy = profile["resource_policy"]
    limits = profile["partition_maximum_wall_minutes"]
    assert isinstance(partitions, Mapping)
    assert isinstance(policy, Mapping)
    assert isinstance(limits, Mapping)
    role = (
        "large_memory"
        if requested_memory_gib >= float(policy["large_memory_threshold_gib"])
        else "analysis"
    )
    partition = partitions.get(role) or partitions.get("default")
    if partition is not None:
        limit = limits.get(str(partition))
        if limit is not None and requested_wall_minutes > float(limit):
            partition = partitions.get("long_wall") or partition
    return None if partition is None else str(partition)


def _placement_forecast(
    live: Mapping[str, object],
    *,
    partition: Optional[str],
    cpus: int,
    memory_gib: float,
) -> Dict[str, object]:
    required_mib = int(math.ceil(memory_gib * 1024.0))
    nodes = live.get("nodes", [])
    assert isinstance(nodes, list)
    configured = []
    available = []
    for row in nodes:
        if not isinstance(row, Mapping):
            continue
        node_partitions = row.get("partitions", [])
        if partition and node_partitions and partition not in node_partitions:
            continue
        if (
            int(row.get("configured_cpus", 0)) >= cpus
            and int(row.get("real_memory_mib", 0)) >= required_mib
        ):
            configured.append(str(row.get("node")))
            if (
                _eligible_node(row)
                and int(row.get("idle_cpus", 0)) >= cpus
                and int(row.get("scheduler_unallocated_memory_mib", 0)) >= required_mib
            ):
                available.append(str(row.get("node")))
    if available:
        status = "resources_available_now_priority_still_applies"
    elif configured:
        status = "must_wait_for_capacity_or_backfill"
    else:
        status = "request_does_not_fit_any_profile_node"
    return {
        "status": status,
        "partition": partition,
        "requested_cpus": cpus,
        "requested_memory_gib": memory_gib,
        "configured_node_count_that_can_fit": len(configured),
        "currently_available_node_count_that_can_fit": len(available),
        "currently_available_nodes": available,
        "scientific_boundary": (
            "Current fit is not a reservation or a guaranteed start time; priority, "
            "fair-share, reservations, backfill, and intervening submissions still apply."
        ),
    }


def advise_slurm_capacity(
    root: Path,
    *,
    wall_hours: float,
    maximum_memory_gib: Optional[float] = None,
    cpu_ceiling: Optional[int] = None,
    slurm_profile_path: Optional[Path] = None,
    live: bool = True,
    slurm_user: Optional[str] = None,
    job_ids: Sequence[str] = (),
    runner: Optional[CommandRunner] = None,
) -> Dict[str, object]:
    """Replan a prepared campaign and optionally inspect Slurm without submitting."""

    prepared = root.expanduser().resolve(strict=True)
    plan_path = prepared / "campaign-resource-plan.json"
    plan = load_json(plan_path)
    if not isinstance(plan, dict) or not isinstance(plan.get("tasks"), list):
        raise SlurmCapacityError("campaign-resource-plan.json is invalid")
    all_plan_tasks = [
        row for row in plan["tasks"] if isinstance(row, Mapping)
    ]
    executable_task_ids = _executable_planner_task_ids(prepared)
    selected_planning_tasks = (
        all_plan_tasks
        if executable_task_ids is None
        else [
            row for row in all_plan_tasks
            if str(row.get("task_id")) in executable_task_ids
        ]
    )
    excluded_task_ids = sorted({
        str(row.get("task_id")) for row in all_plan_tasks
        if row not in selected_planning_tasks
    })
    planning_tasks, legacy_memory_adjustments = _normalize_legacy_memory_models(
        selected_planning_tasks
    )
    if not planning_tasks:
        raise SlurmCapacityError("no executable planner tasks remain")
    hours = _positive_number(wall_hours, "wall_hours")
    original_memory = _positive_number(
        plan.get("maximum_memory_gib_input"),
        "campaign-resource-plan maximum_memory_gib_input",
    )
    memory = (
        original_memory
        if maximum_memory_gib is None
        else _positive_number(maximum_memory_gib, "maximum_memory_gib")
    )
    if cpu_ceiling is not None and (
        isinstance(cpu_ceiling, bool)
        or not isinstance(cpu_ceiling, int)
        or cpu_ceiling <= 0
    ):
        raise SlurmCapacityError("cpu_ceiling must be a positive integer")

    profile_source = slurm_profile_path or (prepared / "slurm-profile.json")
    profile = load_slurm_profile(profile_source)
    policy = profile["resource_policy"]
    assert isinstance(policy, Mapping)
    useful_peak = _workflow_useful_cpu_peak(planning_tasks)
    live_report: Optional[Dict[str, object]] = None
    scheduler_ceiling: Optional[int] = None
    if live:
        live_report = _query_scheduler(
            profile,
            slurm_user=slurm_user or getpass.getuser(),
            job_ids=job_ids,
            runner=runner or _default_runner,
        )
        value = live_report.get("simultaneous_cpu_ceiling")
        scheduler_ceiling = None if value is None else int(value)

    limits = [useful_peak]
    limiting_factors = ["workflow_dependency_graph"]
    if scheduler_ceiling is not None:
        limits.append(scheduler_ceiling)
        limiting_factors.append("live_scheduler_profile_partitions_and_account_qos")
    if cpu_ceiling is not None:
        limits.append(cpu_ceiling)
        limiting_factors.append("user_cpu_ceiling")
    resource_independent_cpu_ceiling = min(limits)
    recommended_cpus = resource_independent_cpu_ceiling
    replanning_iterations = 0
    while True:
        replanning_iterations += 1
        replanned = plan_campaign_resource_budget(
            planning_tasks,
            maximum_parallel_cpus=recommended_cpus,
            maximum_wall_hours=hours,
            maximum_memory_gib=memory,
            planning_utilization=float(plan.get("planning_utilization", 0.85)),
            pilot_budget_fraction=float(plan.get("pilot_budget_fraction", 0.05)),
            finalization_headroom_fraction=float(
                plan.get("finalization_headroom_fraction", 0.0)
            ),
            memory_safety_factor=float(policy["memory_safety_factor"]),
            memory_overhead_gib=float(policy["memory_overhead_gib"]),
            minimum_scheduler_memory_gib=float(policy["minimum_memory_gib"]),
        )
        tasks = replanned["tasks"]
        assert isinstance(tasks, list)
        concurrent = _planned_resource_wave_memory(replanned, tasks)
        planned_cpu_peak = int(concurrent["maximum_planned_resource_wave_cpus"])
        if (
            replanned["feasibility_status"] != "feasible"
            or planned_cpu_peak <= 0
            or planned_cpu_peak >= recommended_cpus
            or replanning_iterations >= 8
        ):
            break
        recommended_cpus = planned_cpu_peak
        if "aggregate_memory_resource_waves" not in limiting_factors:
            limiting_factors.append("aggregate_memory_resource_waves")
    largest = max(
        tasks,
        key=lambda row: float(
            row["estimated_peak_memory_gib_at_selected_observations"]
        ),
    )
    largest_scheduler = _scheduler_memory_request(
        float(largest["estimated_peak_memory_gib_at_selected_observations"]),
        memory,
        policy,
    )
    wall_minutes = max(
        float(row.get("estimated_wall_hours_at_effective_cpu_cap") or 0.0)
        for row in tasks
    ) * 60.0
    selected_partition = _select_partition(
        profile,
        float(largest_scheduler["scheduler_request_gib"]),
        wall_minutes,
    )
    placement = (
        None
        if live_report is None
        else _placement_forecast(
            live_report,
            partition=selected_partition,
            cpus=max(1, int(largest.get("effective_cpu_cap", 1))),
            memory_gib=float(largest_scheduler["scheduler_request_gib"]),
        )
    )

    return {
        "slurm_capacity_advice_schema": "salsbury-slurm-capacity-advice-v1",
        "technical_status": "complete",
        "scientific_status": "planning only",
        "read_only": True,
        "jobs_submitted": False,
        "prepared_campaign_root": str(prepared),
        "requested_wall_hours": hours,
        "cpu_capacity": {
            "workflow_useful_parallel_cpu_ceiling": useful_peak,
            "resource_independent_parallel_cpu_ceiling": (
                resource_independent_cpu_ceiling
            ),
            "planned_resource_wave_cpu_peak": concurrent[
                "maximum_planned_resource_wave_cpus"
            ],
            "live_scheduler_simultaneous_cpu_ceiling": scheduler_ceiling,
            "user_cpu_ceiling": cpu_ceiling,
            "recommended_maximum_parallel_cpus": recommended_cpus,
            "limiting_factors_considered": limiting_factors,
            "resource_bounded_replanning_iterations": replanning_iterations,
        },
        "replanned_campaign": {
            "task_selection": {
                "source": (
                    "scheduler-resource-requests.json"
                    if executable_task_ids is not None
                    else "campaign-resource-plan.json_fallback"
                ),
                "executable_task_count": len(planning_tasks),
                "excluded_nonexecuting_planner_task_ids": excluded_task_ids,
                "legacy_memory_model_adjustments": legacy_memory_adjustments,
            },
            "feasibility_status": replanned["feasibility_status"],
            "raw_capacity_cpu_hours": replanned["raw_capacity_cpu_hours"],
            "estimated_selected_cpu_hours": replanned["estimated_selected_cpu_hours"],
            "estimated_selected_wall_hours_lower_bound": replanned[
                "estimated_selected_wall_hours_lower_bound"
            ],
            "unused_science_cpu_hours": replanned["unused_science_cpu_hours"],
            "resource_budget_utilization": replanned[
                "resource_budget_utilization"
            ],
            "allocation_saturation": replanned["allocation_saturation"],
            "maximum_memory_gib": memory,
            "memory_feasibility": replanned["memory_feasibility"],
            "tasks": tasks,
        },
        "memory_capacity": {
            "largest_task_id": largest["task_id"],
            **largest_scheduler,
            **concurrent,
            "interpretation": (
                "Per-task memory changes with the selected observation count. "
                "The campaign memory value limits the sum of safety-adjusted "
                "requests in each planned dependency wave; it is not a per-task "
                "allowance or a single-node reservation."
            ),
        },
        "queue_forecast": {
            "live_scheduler": live_report,
            "largest_task_placement": placement,
            "forecast_quality": (
                "scheduler_projected_for_submitted_job_ids"
                if job_ids else "placement_only_before_submission"
            ),
            "interpretation": (
                "Before submission, DEAC can expose current fit and queue pressure but "
                "cannot guarantee a hypothetical start time. Supply pending job IDs to "
                "include Slurm's own expected starts."
            ),
        },
        "scientific_boundary": (
            "Resource feasibility and queue timing do not establish sampling adequacy, "
            "convergence, or scientific validity."
        ),
    }


def render_capacity_markdown(report: Mapping[str, object]) -> str:
    """Render the compact fields a person normally needs before submission."""

    cpus = report["cpu_capacity"]
    memory = report["memory_capacity"]
    campaign = report["replanned_campaign"]
    queue = report["queue_forecast"]
    assert isinstance(cpus, Mapping)
    assert isinstance(memory, Mapping)
    assert isinstance(campaign, Mapping)
    assert isinstance(queue, Mapping)
    utilization = campaign["resource_budget_utilization"]
    saturation = campaign["allocation_saturation"]
    assert isinstance(utilization, Mapping)
    assert isinstance(saturation, Mapping)
    placement = queue.get("largest_task_placement")
    placement_status = (
        "offline"
        if not isinstance(placement, Mapping)
        else str(placement.get("status"))
    )
    lines = [
        "# Slurm capacity advice",
        "",
        f"- Useful workflow maximum: {cpus['workflow_useful_parallel_cpu_ceiling']} CPUs",
        f"- Planned CPU peak after memory packing: "
        f"{cpus['planned_resource_wave_cpu_peak']} CPUs",
        f"- Recommended request: {cpus['recommended_maximum_parallel_cpus']} CPUs",
        f"- Requested duration: {report['requested_wall_hours']:g} hours",
        f"- CPU-hour envelope: {campaign['raw_capacity_cpu_hours']:g}",
        f"- Estimated selected CPU-hours: "
        f"{campaign['estimated_selected_cpu_hours']:.2f}",
        f"- Unallocated science CPU-hours: "
        f"{campaign['unused_science_cpu_hours']:.2f}",
        f"- Science CPU-hour utilization: "
        f"{100.0 * float(utilization['science_cpu_hour_fraction']):.1f}%",
        f"- Science wall-time utilization: "
        f"{100.0 * float(utilization['science_wall_time_fraction']):.1f}%",
        f"- Allocation stop reason: {saturation['stop_reason']}",
        f"- Largest planned task working set: {memory['planned_working_set_gib']:.2f} GiB",
        f"- Scheduler memory needed with safety margin: "
        f"{memory['unbounded_scheduler_request_gib']:.2f} GiB",
        f"- Aggregate campaign memory ceiling: "
        f"{campaign['maximum_memory_gib']:.2f} GiB",
        f"- Largest planned concurrent scheduler-memory wave: "
        f"{memory['maximum_concurrent_scheduler_request_gib']:.2f} GiB",
        f"- Largest-task placement: {placement_status}",
        "",
        str(saturation["unused_cpu_hour_interpretation"]),
        "",
        "No job was submitted. Queue timing before submission is an estimate, not a reservation.",
    ]
    return "\n".join(lines) + "\n"
