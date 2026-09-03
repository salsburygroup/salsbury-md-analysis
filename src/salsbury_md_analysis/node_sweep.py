"""Compare bounded cluster node allocations without submitting work.

The sweep reuses the campaign planner for every node count.  Scientific
minimums remain hard feasibility gates, while the information score measures
the additional normalized frame coverage that fits above those floors.
"""

from __future__ import annotations

import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Mapping, Optional, Sequence

from .resource_planning import (
    ResourcePlanningError,
    plan_campaign_resource_budget,
    plan_global_stride_projection_coupled_campaign_resource_budget,
)


NODE_SWEEP_SCHEMA = "salsbury-planner-node-sweep-v2"
PARETO_SELECTION_POLICIES = ("minimum_nodes", "balanced")
PARETO_OBJECTIVE_MODES = (
    "nodes_walltime_information",
    "walltime_information",
)


def _maximum_task_working_memory(task: Mapping[str, object]) -> float:
    """Return the modeled working set at the task's declared frame ceiling."""

    if task.get("parallel_execution_model") is not None:
        workers = _positive_integer(
            task.get("parallel_worker_count"),
            f"task {task.get('task_id')} parallel_worker_count",
        )
        per_worker = _positive_number(
            task.get("estimated_peak_memory_gib_per_parallel_worker"),
            f"task {task.get('task_id')} parallel worker memory",
        )
        reducer = _positive_number(
            task.get("reducer_memory_gib", per_worker),
            f"task {task.get('task_id')} reducer memory",
        )
        return max(reducer, per_worker * workers)

    source = task.get("source_frames_per_replica")
    if not isinstance(source, (list, tuple)) or not source:
        raise ResourcePlanningError(
            f"task {task.get('task_id')} has no source-frame inventory"
        )
    maximum = _positive_integer(
        task.get("maximum_frames_per_replica", max(source)),
        f"task {task.get('task_id')} maximum_frames_per_replica",
    )
    multiplier = _positive_integer(
        task.get("member_observation_multiplier", 1),
        f"task {task.get('task_id')} member_observation_multiplier",
    )
    observations = sum(
        min(
            _positive_integer(
                value, f"task {task.get('task_id')} source frame count"
            ),
            maximum,
        )
        for value in source
    ) * multiplier
    measured_model = task.get("measured_memory_cost_model")
    if isinstance(measured_model, Mapping):
        calibration_observations = _positive_number(
            measured_model.get("calibration_observations"),
            f"task {task.get('task_id')} memory calibration observations",
        )
        calibration_memory = _positive_number(
            measured_model.get("calibration_memory_gib"),
            f"task {task.get('task_id')} memory calibration GiB",
        )
        exponent = _positive_number(
            measured_model.get("memory_exponent"),
            f"task {task.get('task_id')} memory exponent",
        )
        minimum_scale = _positive_number(
            measured_model.get("minimum_observation_scale", 1.0),
            f"task {task.get('task_id')} minimum memory scale",
        )
        return max(
            1.0,
            calibration_memory * max(
                minimum_scale,
                (observations / calibration_observations) ** exponent,
            ),
        )
    power_model = task.get("power_law_cost_model")
    if isinstance(power_model, Mapping):
        calibration_observations = _positive_number(
            power_model.get("calibration_observations"),
            f"task {task.get('task_id')} power-law observations",
        )
        calibration_memory = _positive_number(
            power_model.get("calibration_memory_gib"),
            f"task {task.get('task_id')} power-law memory GiB",
        )
        exponent = _positive_number(
            power_model.get("memory_exponent"),
            f"task {task.get('task_id')} power-law memory exponent",
        )
        return calibration_memory * max(
            1.0, (observations / calibration_observations) ** exponent
        )
    return _positive_number(
        task.get("estimated_peak_memory_gib", 1.0),
        f"task {task.get('task_id')} estimated_peak_memory_gib",
    )


def maximum_useful_node_inventory(
    tasks: Sequence[Mapping[str, object]],
    *,
    cpus_per_node: int,
    memory_gib_per_node: float,
    maximum_nodes: int,
    memory_safety_factor: float = 1.0,
    memory_overhead_gib: float = 0.0,
    minimum_scheduler_memory_gib: float = 0.0,
) -> Dict[str, object]:
    """Pack the full task inventory to find the useful node-count ceiling.

    Every enabled task is evaluated at its declared maximum frame count and
    intrinsic worker cap. Tasks in one execution bundle are serial, so their
    resource fragments form a maximum envelope rather than being summed.
    Independent bundles in one dependency stage are then packed together.
    """

    per_node_cpus = _positive_integer(cpus_per_node, "cpus_per_node")
    per_node_memory = _positive_number(
        memory_gib_per_node, "memory_gib_per_node"
    )
    node_limit = _positive_integer(maximum_nodes, "maximum_nodes")
    memory_factor = _positive_number(
        memory_safety_factor, "memory_safety_factor"
    )
    if (
        isinstance(memory_overhead_gib, bool)
        or not isinstance(memory_overhead_gib, (int, float))
        or not math.isfinite(float(memory_overhead_gib))
        or float(memory_overhead_gib) < 0.0
    ):
        raise ResourcePlanningError(
            "memory_overhead_gib must be finite and nonnegative"
        )
    if (
        isinstance(minimum_scheduler_memory_gib, bool)
        or not isinstance(minimum_scheduler_memory_gib, (int, float))
        or not math.isfinite(float(minimum_scheduler_memory_gib))
        or float(minimum_scheduler_memory_gib) < 0.0
    ):
        raise ResourcePlanningError(
            "minimum_scheduler_memory_gib must be finite and nonnegative"
        )
    overhead = float(memory_overhead_gib)
    minimum_memory = float(minimum_scheduler_memory_gib)
    frame_ceiling_memory_limited_tasks: list[Dict[str, object]] = []

    def scheduler_memory(working_memory: float) -> float:
        return max(
            minimum_memory,
            float(math.ceil(working_memory * memory_factor + overhead)),
        )

    def task_fragments(task: Mapping[str, object]) -> list[Dict[str, object]]:
        task_id = str(task.get("task_id", "task"))
        cap = _positive_integer(
            task.get("intrinsic_cpu_cap", task.get("effective_cpu_cap", 1)),
            f"task {task_id} intrinsic_cpu_cap",
        )
        if task.get("parallel_execution_model") is None:
            working_memory = _maximum_task_working_memory(task)
            request = scheduler_memory(working_memory)
            if request > per_node_memory + 1.0e-12:
                if not isinstance(
                    task.get("measured_memory_cost_model"), Mapping
                ) and not isinstance(task.get("power_law_cost_model"), Mapping):
                    raise ResourcePlanningError(
                        f"task {task_id} requires {request:g} GiB with no "
                        "frame-dependent memory model, exceeding "
                        f"{per_node_memory:g} GiB per node"
                    )
                frame_ceiling_memory_limited_tasks.append({
                    "task_id": task_id,
                    "full_frame_working_memory_gib": working_memory,
                    "full_frame_scheduler_memory_gib": request,
                    "memory_gib_per_node": per_node_memory,
                    "required_action": (
                        "select fewer frames while preserving the scientific floor"
                    ),
                })
                request = per_node_memory
            return [{
                "cpu_slots": min(cap, per_node_cpus),
                "memory_gib": request,
            }]

        workers = min(
            cap,
            _positive_integer(
                task.get("parallel_worker_count"),
                f"task {task_id} parallel_worker_count",
            ),
        )
        per_worker = _positive_number(
            task.get("estimated_peak_memory_gib_per_parallel_worker"),
            f"task {task_id} parallel worker memory",
        )
        reducer = _positive_number(
            task.get("reducer_memory_gib", per_worker),
            f"task {task_id} reducer memory",
        )
        workers_per_node = 0
        for candidate in range(min(workers, per_node_cpus), 0, -1):
            if (
                scheduler_memory(max(reducer, per_worker * candidate))
                <= per_node_memory + 1.0e-12
            ):
                workers_per_node = candidate
                break
        if workers_per_node == 0:
            raise ResourcePlanningError(
                f"task {task_id} cannot place one worker and its reducer in "
                f"{per_node_memory:g} GiB"
            )
        fragments = []
        remaining = workers
        while remaining > 0:
            active = min(remaining, workers_per_node)
            fragments.append({
                "cpu_slots": active,
                "memory_gib": scheduler_memory(
                    max(reducer, per_worker * active)
                ),
            })
            remaining -= active
        return fragments

    def merge_bundle_fragments(
        rows: Sequence[Sequence[Mapping[str, object]]],
    ) -> list[Dict[str, object]]:
        ordered = [
            sorted(
                fragments,
                key=lambda row: (
                    -max(
                        int(row["cpu_slots"]) / per_node_cpus,
                        float(row["memory_gib"]) / per_node_memory,
                    ),
                    -float(row["memory_gib"]),
                    -int(row["cpu_slots"]),
                ),
            )
            for fragments in rows
        ]
        width = max(len(fragments) for fragments in ordered)
        return [{
            "cpu_slots": max(
                int(fragments[index]["cpu_slots"])
                if index < len(fragments) else 0
                for fragments in ordered
            ),
            "memory_gib": max(
                float(fragments[index]["memory_gib"])
                if index < len(fragments) else 0.0
                for fragments in ordered
            ),
        } for index in range(width)]

    stages: Dict[int, Dict[str, list[list[Dict[str, object]]]]] = {}
    stage_task_counts: Dict[int, int] = {}
    for task in tasks:
        stage = int(task.get("dependency_stage", 0))
        if stage < 0:
            raise ResourcePlanningError(
                f"task {task.get('task_id')} has a negative dependency stage"
            )
        bundle = str(
            task.get("execution_bundle_id") or task.get("task_id", "task")
        )
        stages.setdefault(stage, {}).setdefault(bundle, []).append(
            task_fragments(task)
        )
        stage_task_counts[stage] = stage_task_counts.get(stage, 0) + 1
    if not stages:
        raise ResourcePlanningError("node inventory requires at least one task")

    stage_reports = []
    for stage, bundles in sorted(stages.items()):
        inventory = []
        total_cpu = 0
        total_memory = 0.0
        for bundle_id, fragment_sets in sorted(bundles.items()):
            for fragment in merge_bundle_fragments(fragment_sets):
                row = {**fragment, "bundle_id": bundle_id}
                inventory.append(row)
                total_cpu += int(row["cpu_slots"])
                total_memory += float(row["memory_gib"])
        inventory.sort(key=lambda row: (
            -max(
                int(row["cpu_slots"]) / per_node_cpus,
                float(row["memory_gib"]) / per_node_memory,
            ),
            -float(row["memory_gib"]),
            -int(row["cpu_slots"]),
            str(row["bundle_id"]),
        ))
        nodes: list[Dict[str, object]] = []
        for fragment in inventory:
            destination = next((
                node for node in nodes
                if fragment["bundle_id"] not in node["bundle_ids"]
                and int(node["cpu_slots"]) + int(fragment["cpu_slots"])
                <= per_node_cpus
                and float(node["memory_gib"])
                + float(fragment["memory_gib"])
                <= per_node_memory + 1.0e-12
            ), None)
            if destination is None:
                destination = {
                    "cpu_slots": 0,
                    "memory_gib": 0.0,
                    "bundle_ids": set(),
                }
                nodes.append(destination)
            destination["cpu_slots"] = (
                int(destination["cpu_slots"]) + int(fragment["cpu_slots"])
            )
            destination["memory_gib"] = (
                float(destination["memory_gib"])
                + float(fragment["memory_gib"])
            )
            destination["bundle_ids"].add(fragment["bundle_id"])
        stage_reports.append({
            "dependency_stage": stage,
            "task_count": stage_task_counts[stage],
            "execution_bundle_count": len(bundles),
            "maximum_parallel_cpu_slots": total_cpu,
            "maximum_parallel_scheduler_memory_gib": total_memory,
            "packed_node_count": len(nodes),
            "cpu_only_node_lower_bound": math.ceil(total_cpu / per_node_cpus),
            "memory_only_node_lower_bound": math.ceil(
                total_memory / per_node_memory
            ),
        })
    uncapped = max(int(row["packed_node_count"]) for row in stage_reports)
    useful = min(node_limit, uncapped)
    limiting_stage = next(
        row for row in stage_reports
        if int(row["packed_node_count"]) == uncapped
    )
    return {
        "basis": "full_enabled_task_inventory_at_declared_frame_ceilings",
        "memory_basis": "safety_adjusted_scheduler_requests",
        "uncapped_task_inventory_node_ceiling": uncapped,
        "maximum_useful_nodes_within_campaign_cap": useful,
        "campaign_node_cap": node_limit,
        "campaign_cap_limits_full_parallelism": uncapped > node_limit,
        "maximum_useful_parallel_cpus_within_campaign_cap": min(
            int(limiting_stage["maximum_parallel_cpu_slots"]),
            useful * per_node_cpus,
        ),
        "limiting_dependency_stage": limiting_stage["dependency_stage"],
        "frame_ceiling_memory_limited_task_count": len(
            frame_ceiling_memory_limited_tasks
        ),
        "frame_ceiling_memory_limited_tasks": (
            frame_ceiling_memory_limited_tasks
        ),
        "dependency_stages": stage_reports,
        "interpretation": (
            "Nodes above the useful ceiling cannot expose additional task or "
            "replica-worker concurrency for this fixed task inventory. Runtime "
            "variance and scheduler queue policy can still favor fewer nodes. "
            "A modeled full-frame memory request above one-node capacity limits "
            "that non-distributed task's frame count; extra nodes do not repair it."
        ),
    }


def _plan_one_node(payload: Mapping[str, object]) -> tuple[int, Dict[str, object]]:
    """Process-safe worker for one independent node-count calculation."""

    nodes = int(payload["nodes"])
    task_rows = payload["tasks"]
    kwargs = dict(payload["kwargs"])
    kwargs.update({
        "maximum_parallel_cpus": nodes * int(payload["cpus_per_node"]),
        "maximum_memory_gib": nodes * float(payload["memory_gib_per_node"]),
        "maximum_nodes": nodes,
    })
    if int(payload["cache_count"]) == 1:
        plan = plan_global_stride_projection_coupled_campaign_resource_budget(
            task_rows,
            coordinate_cache_minimum_frames_per_replica=1,
            coordinate_cache_full_scan_fraction=float(
                payload["coordinate_cache_full_scan_fraction"]
            ),
            overall_stride_candidate_strides=payload.get(
                "overall_stride_candidate_strides"
            ),
            **kwargs,
        )
    else:
        plan = plan_campaign_resource_budget(task_rows, **kwargs)
    return nodes, plan


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ResourcePlanningError(f"{label} must be a positive integer")
    return value


def _positive_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ResourcePlanningError(f"{label} must be a finite positive number")
    return float(value)


def _selected_information(plan: Mapping[str, object]) -> Dict[str, object]:
    coupling = plan.get("global_stride_coupling")
    if isinstance(coupling, Mapping):
        selected_stride = coupling.get(
            "selected_overall_trajectory_integer_stride"
        )
        candidates = coupling.get("candidate_evaluations")
        if isinstance(candidates, list):
            selected = next((
                row for row in candidates
                if isinstance(row, Mapping)
                and row.get("overall_trajectory_integer_stride") == selected_stride
                and row.get("feasibility_status") == "feasible"
            ), None)
            if isinstance(selected, Mapping):
                return {
                    "balanced_information_utility": float(
                        selected["balanced_information_utility"]
                    ),
                    "minimum_normalized_analysis_coverage": float(
                        selected["minimum_normalized_analysis_coverage"]
                    ),
                    "mean_normalized_analysis_coverage": float(
                        selected["mean_normalized_analysis_coverage"]
                    ),
                    "selected_observation_count": float(
                        selected["selected_observation_count"]
                    ),
                    "scored_analysis_count": int(
                        selected["scored_analysis_count"]
                    ),
                }

    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise ResourcePlanningError("planned node sweep has no task rows")
    weighted = 0.0
    total_weight = 0.0
    coverages = []
    selected_observations = 0.0
    for row in tasks:
        if not isinstance(row, Mapping) or row.get("module_id") == "coordinate_cache":
            continue
        coverage = float(row.get("coverage_fraction", 0.0))
        weight = float(row.get("priority_weight", 1.0))
        weighted += weight * math.sqrt(max(0.0, min(1.0, coverage)))
        total_weight += weight
        coverages.append(coverage)
        selected_observations += float(
            row.get("selected_member_observation_count", 0)
        )
    if not coverages or total_weight <= 0.0:
        raise ResourcePlanningError("planned node sweep has no analysis tasks to score")
    return {
        "balanced_information_utility": weighted / total_weight,
        "minimum_normalized_analysis_coverage": min(coverages),
        "mean_normalized_analysis_coverage": statistics.fmean(coverages),
        "selected_observation_count": selected_observations,
        "scored_analysis_count": len(coverages),
    }


def _selected_plan_node_ceiling(
    plan: Mapping[str, object],
) -> tuple[int, list[int]]:
    """Return the maximum physical-node use in the plan's dependency stages."""

    stages = plan.get("stages")
    if not isinstance(stages, list):
        return 0, []
    rows = [
        row for row in stages
        if isinstance(row, Mapping)
        and isinstance(row.get("planned_node_count"), int)
        and not isinstance(row.get("planned_node_count"), bool)
    ]
    if not rows:
        return 0, []
    ceiling = max(int(row["planned_node_count"]) for row in rows)
    limiting = [
        int(row["dependency_stage"])
        for row in rows
        if int(row["planned_node_count"]) == ceiling
    ]
    return ceiling, limiting


def _planned_schedule_metrics(plan: Mapping[str, object]) -> Dict[str, object]:
    """Summarize elapsed and node-reservation time from planner stages."""

    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        fallback = plan.get("estimated_selected_wall_hours_lower_bound")
        wall = (
            float(fallback)
            if isinstance(fallback, (int, float))
            and not isinstance(fallback, bool)
            else None
        )
        return {
            "planned_makespan_hours": wall,
            "planned_peak_node_count": None,
            "planned_reserved_node_hours": None,
        }

    elapsed = 0.0
    reserved_node_hours = 0.0
    peak_nodes = 0
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ResourcePlanningError("planned stage must be an object")
        wall_value = stage.get(
            "estimated_wall_hours_with_resource_lanes",
            stage.get("estimated_wall_hours_with_resource_waves"),
        )
        node_value = stage.get("planned_node_count")
        if (
            isinstance(wall_value, bool)
            or not isinstance(wall_value, (int, float))
            or not math.isfinite(float(wall_value))
            or float(wall_value) < 0.0
        ):
            raise ResourcePlanningError(
                "planned stage lacks a finite nonnegative lane makespan"
            )
        if (
            isinstance(node_value, bool)
            or not isinstance(node_value, int)
            or node_value < 0
        ):
            raise ResourcePlanningError(
                "planned stage lacks a nonnegative planned node count"
            )
        stage_wall = float(wall_value)
        elapsed += stage_wall
        peak_nodes = max(peak_nodes, node_value)
        reserved_node_hours += stage_wall * node_value
    return {
        "planned_makespan_hours": elapsed,
        "planned_peak_node_count": peak_nodes,
        "planned_reserved_node_hours": reserved_node_hours,
    }


def _selected_stride_contract(plan: Mapping[str, object]) -> Dict[str, int]:
    """Return the effective raw stride for every planner balance group."""

    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise ResourcePlanningError("planned node sweep has no task rows")
    contract: Dict[str, int] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            raise ResourcePlanningError("planned task must be an object")
        if task.get("module_id") == "coordinate_cache":
            continue
        value = task.get("effective_raw_integer_stride", task.get("integer_stride"))
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ResourcePlanningError(
                f"task {task.get('task_id')} has an invalid selected stride"
            )
        group = str(task.get("balance_group") or task.get("task_id"))
        previous = contract.get(group)
        if previous is not None and previous != value:
            raise ResourcePlanningError(
                f"balance group {group} has inconsistent selected strides"
            )
        contract[group] = value
    return dict(sorted(contract.items()))


def scientific_minimum_multiples(
    tasks: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Summarize selected frames relative to registered scientific floors."""

    rows = []
    for task in tasks:
        if task.get("module_id") == "coordinate_cache":
            continue
        contract = task.get("scientific_sampling_requirements")
        if not isinstance(contract, Mapping):
            continue
        required = contract.get("minimum_frames_per_replica")
        if (
            isinstance(required, bool)
            or not isinstance(required, int)
            or required <= 0
        ):
            continue
        source = task.get("source_frames_per_replica")
        selected = task.get("selected_physical_frames_per_replica")
        if not isinstance(source, list) or not isinstance(selected, list):
            raise ResourcePlanningError(
                f"task {task.get('task_id')} lacks frame accounting"
            )
        task_required = task.get("scientific_minimum_frames_per_replica")
        if (
            isinstance(task_required, int)
            and not isinstance(task_required, bool)
            and task_required > 0
        ):
            required = max(required, task_required)
        per_system = contract.get("minimum_frames_per_system", 0)
        if (
            isinstance(per_system, bool)
            or not isinstance(per_system, int)
            or per_system < 0
        ):
            raise ResourcePlanningError(
                f"task {task.get('task_id')} has an invalid per-system minimum"
            )
        raw_ids = task.get("system_ids_per_replica")
        ids = (
            [str(value) for value in raw_ids]
            if isinstance(raw_ids, list) and len(raw_ids) == len(source)
            else ["__single_declared_system__"] * len(source)
        )
        group_sizes = {system_id: ids.count(system_id) for system_id in set(ids)}
        required_by_replica = [
            max(required, math.ceil(per_system / group_sizes[system_id]))
            for system_id in ids
        ]
        declared_minimum_count = sum(required_by_replica)
        minimum_count = sum(
            min(int(value), minimum)
            for value, minimum in zip(source, required_by_replica)
        )
        declared_overall = task.get("minimum_selected_physical_frame_count")
        if (
            isinstance(declared_overall, int)
            and not isinstance(declared_overall, bool)
        ):
            minimum_count = max(minimum_count, declared_overall)
        selected_count = sum(int(value) for value in selected)
        if minimum_count <= 0:
            continue
        rows.append({
            "task_id": str(task.get("task_id")),
            "module_id": str(task.get("module_id")),
            "selected_physical_frame_count": selected_count,
            "scientific_minimum_physical_frame_count": minimum_count,
            "multiple_of_scientific_minimum": selected_count / minimum_count,
            "source_limited": declared_minimum_count > sum(
                int(value) for value in source
            ),
        })
    if not rows:
        raise ResourcePlanningError(
            "node sweep cannot calculate scientific-minimum multiples"
        )
    multiples = [float(row["multiple_of_scientific_minimum"]) for row in rows]
    return {
        "task_count": len(rows),
        "mean_multiple_of_scientific_minimum": statistics.fmean(multiples),
        "median_multiple_of_scientific_minimum": statistics.median(multiples),
        "minimum_multiple_of_scientific_minimum": min(multiples),
        "maximum_multiple_of_scientific_minimum": max(multiples),
        "source_limited_task_count": sum(bool(row["source_limited"]) for row in rows),
        "tasks": rows,
    }


def plan_node_sweep(
    tasks: Sequence[Mapping[str, object]],
    *,
    cpus_per_node: int,
    memory_gib_per_node: float,
    maximum_nodes: int,
    maximum_wall_hours: float,
    information_plateau_fraction: float = 0.95,
    information_plateau_tolerance_fraction: float = 0.0,
    planning_utilization: float = 0.85,
    pilot_budget_fraction: float = 0.05,
    finalization_headroom_fraction: float = 0.05,
    memory_safety_factor: float = 1.5,
    memory_overhead_gib: float = 1.0,
    minimum_scheduler_memory_gib: float = 0.0,
    coordinate_cache_full_scan_fraction: float = 1.0,
    overall_stride_candidate_strides: Optional[Sequence[int]] = None,
    planning_processes: int = 1,
    pareto_selection_policy: str = "minimum_nodes",
    pareto_objectives: str = "nodes_walltime_information",
) -> Dict[str, object]:
    """Return a plan-only node curve and a selected Pareto allocation."""

    per_node_cpus = _positive_integer(cpus_per_node, "cpus_per_node")
    per_node_memory = _positive_number(
        memory_gib_per_node, "memory_gib_per_node"
    )
    node_limit = _positive_integer(maximum_nodes, "maximum_nodes")
    wall = _positive_number(maximum_wall_hours, "maximum_wall_hours")
    plateau = _positive_number(
        information_plateau_fraction, "information_plateau_fraction"
    )
    if plateau > 1.0:
        raise ResourcePlanningError(
            "information_plateau_fraction must not exceed one"
        )
    if (
        isinstance(information_plateau_tolerance_fraction, bool)
        or not isinstance(information_plateau_tolerance_fraction, (int, float))
        or not math.isfinite(float(information_plateau_tolerance_fraction))
        or float(information_plateau_tolerance_fraction) < 0.0
    ):
        raise ResourcePlanningError(
            "information_plateau_tolerance_fraction must be finite and nonnegative"
        )
    tolerance = float(information_plateau_tolerance_fraction)
    if tolerance >= 1.0:
        raise ResourcePlanningError(
            "information_plateau_tolerance_fraction must be smaller than one"
        )
    if pareto_selection_policy not in PARETO_SELECTION_POLICIES:
        choices = ", ".join(PARETO_SELECTION_POLICIES)
        raise ResourcePlanningError(
            f"pareto_selection_policy must be one of: {choices}"
        )
    if pareto_objectives not in PARETO_OBJECTIVE_MODES:
        choices = ", ".join(PARETO_OBJECTIVE_MODES)
        raise ResourcePlanningError(
            f"pareto_objectives must be one of: {choices}"
        )
    task_rows = [dict(row) for row in tasks]
    if not task_rows:
        raise ResourcePlanningError("node sweep requires at least one task")
    task_inventory_ceiling = maximum_useful_node_inventory(
        task_rows,
        cpus_per_node=per_node_cpus,
        memory_gib_per_node=per_node_memory,
        maximum_nodes=node_limit,
        memory_safety_factor=memory_safety_factor,
        memory_overhead_gib=memory_overhead_gib,
        minimum_scheduler_memory_gib=minimum_scheduler_memory_gib,
    )
    analytical_useful_nodes = int(task_inventory_ceiling[
        "maximum_useful_nodes_within_campaign_cap"
    ])
    cache_count = sum(
        row.get("module_id") == "coordinate_cache" for row in task_rows
    )
    if cache_count > 1:
        raise ResourcePlanningError(
            "node sweep accepts at most one coordinate-cache task"
        )

    process_count = _positive_integer(planning_processes, "planning_processes")
    common_kwargs = {
            "maximum_wall_hours": wall,
            "planning_utilization": planning_utilization,
            "pilot_budget_fraction": pilot_budget_fraction,
            "finalization_headroom_fraction": finalization_headroom_fraction,
            "memory_safety_factor": memory_safety_factor,
            "memory_overhead_gib": memory_overhead_gib,
            "minimum_scheduler_memory_gib": minimum_scheduler_memory_gib,
            "maximum_cpus_per_node": per_node_cpus,
            "maximum_memory_gib_per_node": per_node_memory,
    }
    def planning_payload(nodes: int) -> Dict[str, object]:
        return {
        "nodes": nodes,
        "tasks": task_rows,
        "kwargs": common_kwargs,
        "cpus_per_node": per_node_cpus,
        "memory_gib_per_node": per_node_memory,
        "cache_count": cache_count,
        "coordinate_cache_full_scan_fraction": (
            coordinate_cache_full_scan_fraction
        ),
        "overall_stride_candidate_strides": overall_stride_candidate_strides,
    }

    # Start with the largest allowed allocation. Its selected task graph and
    # exact node-aware lane schedule are the authoritative memory/concurrency
    # ceiling. The analytical inventory pack remains a conservative diagnostic
    # because it cannot reproduce every planner lane-fragmentation decision.
    _, maximum_node_plan = _plan_one_node(planning_payload(node_limit))
    planner_node_ceiling, planner_limiting_stages = (
        _selected_plan_node_ceiling(maximum_node_plan)
    )
    useful_nodes = min(
        node_limit,
        max(analytical_useful_nodes, planner_node_ceiling, 1),
    )
    task_inventory_ceiling.update({
        "analytical_full_inventory_packed_node_count": (
            task_inventory_ceiling[
                "uncapped_task_inventory_node_ceiling"
            ]
        ),
        "maximum_node_preflight_requested_nodes": node_limit,
        "maximum_node_preflight_selected_stage_node_ceiling": (
            planner_node_ceiling
        ),
        "maximum_node_preflight_limiting_dependency_stages": (
            planner_limiting_stages
        ),
        "maximum_useful_nodes_within_campaign_cap": useful_nodes,
        "uncapped_task_inventory_node_ceiling": max(
            int(task_inventory_ceiling[
                "uncapped_task_inventory_node_ceiling"
            ]),
            planner_node_ceiling,
        ),
        "campaign_cap_limits_full_parallelism": useful_nodes >= node_limit,
        "ceiling_selection_rule": (
            "larger of the analytical full-inventory pack and the exact "
            "node use in the maximum-node planner preflight"
        ),
    })
    payloads = [
        planning_payload(nodes)
        for nodes in range(1, useful_nodes + 1)
        if nodes != node_limit
    ]
    if process_count == 1:
        planned_items = [_plan_one_node(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(
            max_workers=min(process_count, max(1, len(payloads)))
        ) as executor:
            planned_items = list(executor.map(_plan_one_node, payloads))
    raw_plans_by_node = dict(planned_items)
    raw_plans_by_node[node_limit] = maximum_node_plan
    maximum_node_feasible = (
        maximum_node_plan.get("feasibility_status") == "feasible"
    )
    maximum_node_information = (
        _selected_information(maximum_node_plan)
        if maximum_node_feasible else None
    )
    maximum_node_information_value = (
        float(maximum_node_information["balanced_information_utility"])
        if maximum_node_information is not None else -math.inf
    )
    curve = []
    planned_by_node: Dict[int, Dict[str, object]] = {}
    best_plan: Optional[Dict[str, object]] = None
    best_source_nodes: Optional[int] = None
    best_information_value = -math.inf
    for nodes in range(1, node_limit + 1):
        raw_plan = raw_plans_by_node.get(nodes)
        raw_feasible = bool(
            raw_plan is not None
            and raw_plan.get("feasibility_status") == "feasible"
        )
        raw_information = (
            _selected_information(raw_plan) if raw_feasible else None
        )
        raw_value = (
            float(raw_information["balanced_information_utility"])
            if raw_information is not None else -math.inf
        )
        if raw_value > best_information_value + 1.0e-15:
            best_plan = raw_plan
            best_source_nodes = nodes
            best_information_value = raw_value
        if (
            nodes >= useful_nodes
            and maximum_node_information_value
            > best_information_value + 1.0e-15
        ):
            best_plan = maximum_node_plan
            best_source_nodes = useful_nodes
            best_information_value = maximum_node_information_value
        selected_plan = best_plan
        selected_source_nodes = best_source_nodes
        planned_by_node[nodes] = selected_plan or raw_plan
        information = (
            _selected_information(selected_plan)
            if selected_plan is not None else None
        )
        schedule = (
            _planned_schedule_metrics(selected_plan)
            if selected_plan is not None else {
                "planned_makespan_hours": None,
                "planned_peak_node_count": None,
                "planned_reserved_node_hours": None,
            }
        )
        stride_contract = (
            _selected_stride_contract(selected_plan)
            if selected_plan is not None else {}
        )
        curve.append({
            "requested_nodes": nodes,
            "requested_parallel_cpus": nodes * per_node_cpus,
            "requested_memory_gib": nodes * per_node_memory,
            "above_task_inventory_useful_node_ceiling": nodes > useful_nodes,
            "raw_planner_evaluated": raw_plan is not None,
            "raw_feasibility_status": (
                raw_plan.get("feasibility_status")
                if raw_plan is not None else
                "not_evaluated_above_task_inventory_ceiling"
            ),
            "raw_balanced_information_utility": (
                raw_information["balanced_information_utility"]
                if raw_information is not None else None
            ),
            "feasibility_status": (
                "feasible" if selected_plan is not None
                else (
                    raw_plan.get("feasibility_status")
                    if raw_plan is not None else "infeasible"
                )
            ),
            "replayed_from_node_count": selected_source_nodes,
            "balanced_information_utility": (
                information["balanced_information_utility"]
                if information is not None else None
            ),
            "minimum_normalized_analysis_coverage": (
                information["minimum_normalized_analysis_coverage"]
                if information is not None else None
            ),
            "mean_normalized_analysis_coverage": (
                information["mean_normalized_analysis_coverage"]
                if information is not None else None
            ),
            "selected_observation_count": (
                information["selected_observation_count"]
                if information is not None else None
            ),
            "scored_analysis_count": (
                information["scored_analysis_count"]
                if information is not None else None
            ),
            "estimated_selected_cpu_hours": selected_plan.get(
                "estimated_selected_cpu_hours"
            ) if selected_plan is not None else None,
            "estimated_selected_wall_hours_lower_bound": selected_plan.get(
                "estimated_selected_wall_hours_lower_bound"
            ) if selected_plan is not None else None,
            **schedule,
            "selected_stride_by_balance_group": stride_contract,
            "minimum_known_cpu_hours": selected_plan.get(
                "minimum_known_cpu_hours"
            ) if selected_plan is not None else (
                raw_plan.get("minimum_known_cpu_hours")
                if raw_plan is not None else None
            ),
            "minimum_wall_hours_lower_bound": selected_plan.get(
                "minimum_wall_hours_lower_bound"
            ) if selected_plan is not None else (
                raw_plan.get("minimum_wall_hours_lower_bound")
                if raw_plan is not None else None
            ),
            "selected_overall_trajectory_integer_stride": (
                selected_plan.get("global_stride_coupling", {}).get(
                    "selected_overall_trajectory_integer_stride"
                ) if selected_plan is not None and isinstance(
                    selected_plan.get("global_stride_coupling"), Mapping
                )
                else None
            ),
            "unexpected_error_count": 0,
        })

    feasible_rows = [
        row for row in curve
        if row["feasibility_status"] == "feasible"
        and row["balanced_information_utility"] is not None
    ]
    if not feasible_rows:
        return {
            "node_sweep_schema": NODE_SWEEP_SCHEMA,
            "technical_status": "infeasible",
            "scientific_status": "not evaluated",
            "execution_started": False,
            "jobs_submitted": False,
            "hardware": {
                "cpus_per_node": per_node_cpus,
                "memory_gib_per_node": per_node_memory,
                "maximum_nodes": node_limit,
            },
            "maximum_wall_hours": wall,
            "information_plateau_fraction": plateau,
            "task_inventory_ceiling": task_inventory_ceiling,
            "curve": curve,
            "sweet_spot": None,
        }

    best_information = max(
        float(row["balanced_information_utility"]) for row in feasible_rows
    )
    threshold = plateau * best_information
    chosen_row = next(
        row for row in feasible_rows
        if (
            float(row["balanced_information_utility"])
            + tolerance * best_information
            >= threshold
        )
    )
    chosen_nodes = int(chosen_row["requested_nodes"])
    chosen_plan = planned_by_node[chosen_nodes]
    minimums = scientific_minimum_multiples(chosen_plan["tasks"])
    for row in curve:
        value = row["balanced_information_utility"]
        row["fraction_of_best_information"] = (
            float(value) / best_information if value is not None else None
        )
        wall_value = row["planned_makespan_hours"]
        node_hour_value = row["planned_reserved_node_hours"]
        row["information_per_planned_wall_hour"] = (
            float(value) / float(wall_value)
            if value is not None and wall_value not in (None, 0.0) else None
        )
        row["information_per_reserved_node_hour"] = (
            float(value) / float(node_hour_value)
            if value is not None and node_hour_value not in (None, 0.0) else None
        )
    operational_rows = [
        row for row in feasible_rows
        if not bool(row["above_task_inventory_useful_node_ceiling"])
        and row["planned_makespan_hours"] is not None
    ]
    minimum_wall = min(
        float(row["planned_makespan_hours"]) for row in operational_rows
    )
    maximum_wall = max(
        float(row["planned_makespan_hours"]) for row in operational_rows
    )
    minimum_nodes = min(int(row["requested_nodes"]) for row in operational_rows)
    maximum_operational_nodes = max(
        int(row["requested_nodes"]) for row in operational_rows
    )
    for row in curve:
        value = row["balanced_information_utility"]
        wall_value = row["planned_makespan_hours"]
        if (
            value is None or wall_value is None
            or bool(row["above_task_inventory_useful_node_ceiling"])
        ):
            row["balanced_operational_score"] = None
            row["pareto_efficient"] = False
            continue
        information_regret = 1.0 - float(row["fraction_of_best_information"])
        wait_regret = (
            0.0 if maximum_wall == minimum_wall else
            (float(wall_value) - minimum_wall) / (maximum_wall - minimum_wall)
        )
        node_regret = (
            0.0 if maximum_operational_nodes == minimum_nodes else
            (int(row["requested_nodes"]) - minimum_nodes)
            / (maximum_operational_nodes - minimum_nodes)
        )
        operational_regrets = [information_regret, wait_regret]
        if pareto_objectives == "nodes_walltime_information":
            operational_regrets.append(node_regret)
        row["balanced_operational_score"] = 1.0 - math.sqrt(
            sum(regret ** 2 for regret in operational_regrets)
            / len(operational_regrets)
        )
        row["pareto_efficient"] = not any(
            other is not row
            and other["balanced_information_utility"] is not None
            and other["planned_makespan_hours"] is not None
            and not bool(other["above_task_inventory_useful_node_ceiling"])
            and float(other["balanced_information_utility"]) >= float(value)
            and float(other["planned_makespan_hours"]) <= float(wall_value)
            and (
                pareto_objectives == "walltime_information"
                or int(other["requested_nodes"]) <= int(row["requested_nodes"])
            )
            and (
                float(other["balanced_information_utility"]) > float(value)
                or float(other["planned_makespan_hours"]) < float(wall_value)
                or (
                    pareto_objectives == "nodes_walltime_information"
                    and int(other["requested_nodes"])
                    < int(row["requested_nodes"])
                )
            )
            for other in operational_rows
        )
    pareto_rows = [row for row in operational_rows if row["pareto_efficient"]]
    if pareto_selection_policy == "minimum_nodes":
        operational_choice = min(
            pareto_rows,
            key=lambda row: (
                int(row["requested_nodes"]),
                float(row["planned_makespan_hours"]),
                -float(row["balanced_information_utility"]),
            ),
        )
        operational_selection_rule = (
            "smallest requested node count on the Pareto front; ties prefer "
            "shorter planner makespan and then greater information"
        )
    else:
        operational_choice = max(
            pareto_rows,
            key=lambda row: (
                float(row["balanced_operational_score"]),
                -float(row["planned_makespan_hours"]),
                -int(row["requested_nodes"]),
            ),
        )
        balanced_dimensions = (
            "information, planner makespan, and requested nodes"
            if pareto_objectives == "nodes_walltime_information"
            else "information and planner makespan"
        )
        operational_selection_rule = (
            "highest equal-weight closeness to the observed ideal for "
            f"{balanced_dimensions}; each regret is range-normalized over "
            "feasible points at or below the useful-node ceiling"
        )
    operational_minimums = scientific_minimum_multiples(
        planned_by_node[int(operational_choice["requested_nodes"])]["tasks"]
    )
    threshold_sensitivity = []
    for fraction in (0.75, 0.80, 0.90, 0.95, 0.99, 1.00):
        threshold_row = next(
            row for row in feasible_rows
            if (
                float(row["balanced_information_utility"])
                + tolerance * best_information
                >= fraction * best_information
            )
        )
        threshold_minimums = scientific_minimum_multiples(
            planned_by_node[int(threshold_row["requested_nodes"])]["tasks"]
        )
        threshold_sensitivity.append({
            "fraction_of_best_threshold": fraction,
            "first_qualifying_node_count": threshold_row["requested_nodes"],
            "replayed_from_node_count": threshold_row[
                "replayed_from_node_count"
            ],
            "balanced_information_utility": threshold_row[
                "balanced_information_utility"
            ],
            "fraction_of_best_information": (
                float(threshold_row["balanced_information_utility"])
                / best_information
            ),
            "scientific_minimum_task_count": threshold_minimums["task_count"],
            "mean_multiple_of_scientific_minimum": threshold_minimums[
                "mean_multiple_of_scientific_minimum"
            ],
            "median_multiple_of_scientific_minimum": threshold_minimums[
                "median_multiple_of_scientific_minimum"
            ],
            "minimum_multiple_of_scientific_minimum": threshold_minimums[
                "minimum_multiple_of_scientific_minimum"
            ],
            "maximum_multiple_of_scientific_minimum": threshold_minimums[
                "maximum_multiple_of_scientific_minimum"
            ],
            "source_limited_task_count": threshold_minimums[
                "source_limited_task_count"
            ],
            "planned_makespan_hours": threshold_row[
                "planned_makespan_hours"
            ],
            "planned_peak_node_count": threshold_row[
                "planned_peak_node_count"
            ],
            "planned_reserved_node_hours": threshold_row[
                "planned_reserved_node_hours"
            ],
            "information_per_planned_wall_hour": threshold_row[
                "information_per_planned_wall_hour"
            ],
            "information_per_reserved_node_hour": threshold_row[
                "information_per_reserved_node_hour"
            ],
            "selected_stride_by_balance_group": threshold_row[
                "selected_stride_by_balance_group"
            ],
            "balanced_operational_score": threshold_row[
                "balanced_operational_score"
            ],
            "pareto_efficient": threshold_row["pareto_efficient"],
        })
    return {
        "node_sweep_schema": NODE_SWEEP_SCHEMA,
        "technical_status": "complete",
        "scientific_status": "resource planning only; scientific validity not evaluated",
        "execution_started": False,
        "jobs_submitted": False,
        "hardware": {
            "cpus_per_node": per_node_cpus,
            "memory_gib_per_node": per_node_memory,
            "maximum_nodes": node_limit,
            "maximum_parallel_cpus": per_node_cpus * node_limit,
            "maximum_memory_gib": per_node_memory * node_limit,
        },
        "maximum_wall_hours": wall,
        "task_inventory_ceiling": task_inventory_ceiling,
        "information_score": {
            "name": "balanced_information_utility",
            "definition": (
                "priority-weighted mean square root of normalized physical-frame "
                "coverage across analysis tasks; coordinate-cache work is excluded"
            ),
            "best_observed": best_information,
            "plateau_fraction": plateau,
            "plateau_tolerance_fraction": tolerance,
            "selection_threshold": threshold,
            "selection_rule": (
                "smallest node cap whose monotonic replay envelope reaches the "
                "declared fraction of the best score observed from one through "
                "maximum_nodes, within the declared numerical tolerance; report "
                "the resulting planner makespan and reserved node-hours so the "
                "information threshold is not interpreted without its time and "
                "allocation cost"
            ),
            "monotonic_replay_rule": (
                "a plan found at fewer nodes remains feasible under a larger node "
                "cap; raw heuristic regressions are retained but cannot lower the "
                "best attainable information envelope"
            ),
        },
        "curve": curve,
        "threshold_sensitivity": threshold_sensitivity,
        "operational_balance": {
            "pareto_filtering_enabled": True,
            "pareto_objective_mode": pareto_objectives,
            "pareto_objectives": (
                ["requested_nodes", "planned_makespan_hours", "information"]
                if pareto_objectives == "nodes_walltime_information"
                else ["planned_makespan_hours", "information"]
            ),
            "node_constraint": {
                "maximum_nodes": node_limit,
                "task_inventory_useful_node_ceiling": task_inventory_ceiling[
                    "maximum_useful_nodes_within_campaign_cap"
                ],
            },
            "selection_policy": pareto_selection_policy,
            "pareto_front_node_counts": [
                row["requested_nodes"] for row in pareto_rows
            ],
            "recommended_node_count": operational_choice["requested_nodes"],
            "recommended_curve_point": dict(operational_choice),
            "selection_rule": operational_selection_rule,
            "queue_wait_included": False,
            "wall_time_scope": (
                "predicted dependency-chain execution time after resources start; "
                "scheduler queue delay is not modeled"
            ),
            "scientific_minimum_multiples": operational_minimums,
        },
        "sweet_spot": {
            **dict(chosen_row),
            "scientific_minimum_multiples": minimums,
        },
        "interpretation": (
            "This is a bounded scheduling and frame-allocation comparison. It "
            "does not establish convergence, equilibration, kinetics, mechanism, "
            "or biological importance."
        ),
    }
