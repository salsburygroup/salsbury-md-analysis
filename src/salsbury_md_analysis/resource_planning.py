"""Conservative frame-budget estimates from an actual method/project pilot.

The planner does not infer scientific sufficiency. It converts retained
technical benchmark evidence into an explicit all-frame or balanced-subsample
execution contract for the same method, system, code, and environment class.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence

from .frame_sampling import (
    integer_stride_for_budget,
    integer_stride_selected_count,
)


class ResourcePlanningError(ValueError):
    """Raised when benchmark evidence cannot support a resource estimate."""


def pack_resource_waves(
    items: Sequence[Mapping[str, object]],
    *,
    maximum_parallel_cpus: int,
    maximum_parallel_memory_gib: float,
) -> list[Dict[str, object]]:
    """Pack independent tasks into deterministic CPU-and-memory bounded waves."""

    if maximum_parallel_cpus <= 0 or maximum_parallel_memory_gib <= 0.0:
        raise ResourcePlanningError("resource-wave limits must be positive")
    normalized = []
    for index, item in enumerate(items):
        item_id = str(item.get("item_id", f"item-{index}"))
        cpus = int(item.get("cpu_slots", 1))
        memory = float(item.get("memory_gib", 0.0))
        wall = float(item.get("wall_hours", 0.0))
        if cpus <= 0 or memory <= 0.0 or wall < 0.0:
            raise ResourcePlanningError(
                f"resource-wave item {item_id} has invalid resources"
            )
        if cpus > maximum_parallel_cpus:
            raise ResourcePlanningError(
                f"resource-wave item {item_id} requests {cpus} CPUs, exceeding "
                f"the campaign limit {maximum_parallel_cpus}"
            )
        if memory > maximum_parallel_memory_gib + 1.0e-12:
            raise ResourcePlanningError(
                f"resource-wave item {item_id} requests {memory:g} GiB, exceeding "
                f"the campaign limit {maximum_parallel_memory_gib:g} GiB"
            )
        normalized.append({
            **dict(item),
            "item_id": item_id,
            "cpu_slots": cpus,
            "memory_gib": memory,
            "wall_hours": wall,
        })
    ordered = sorted(
        normalized,
        key=lambda row: (
            -float(row["memory_gib"]),
            -int(row["cpu_slots"]),
            -float(row["wall_hours"]),
            str(row["item_id"]),
        ),
    )
    waves: list[Dict[str, object]] = []
    for item in ordered:
        selected_wave: Optional[Dict[str, object]] = None
        for wave in waves:
            if (
                int(wave["cpu_slots"]) + int(item["cpu_slots"])
                <= maximum_parallel_cpus
                and float(wave["memory_gib"]) + float(item["memory_gib"])
                <= maximum_parallel_memory_gib + 1.0e-12
            ):
                selected_wave = wave
                break
        if selected_wave is None:
            selected_wave = {
                "wave_index": len(waves),
                "cpu_slots": 0,
                "memory_gib": 0.0,
                "wall_hours": 0.0,
                "items": [],
            }
            waves.append(selected_wave)
        selected_wave["cpu_slots"] = (
            int(selected_wave["cpu_slots"]) + int(item["cpu_slots"])
        )
        selected_wave["memory_gib"] = (
            float(selected_wave["memory_gib"]) + float(item["memory_gib"])
        )
        selected_wave["wall_hours"] = max(
            float(selected_wave["wall_hours"]), float(item["wall_hours"])
        )
        selected_wave["items"].append(dict(item))  # type: ignore[union-attr]
    return waves


_ALTERNATIVE_CLUSTERING_FIT_PROFILES: Mapping[str, Mapping[str, object]] = {
    "pam": {
        "reference_fit_observation_ceiling": 6_000,
        "minimum_fit_observations": 1_000,
        "time_exponent": 2.0,
        "complexity_class": "pairwise_distance_quadratic",
    },
    "mwpam": {
        "reference_fit_observation_ceiling": 4_500,
        "minimum_fit_observations": 750,
        "time_exponent": 2.0,
        "complexity_class": "weighted_pairwise_distance_quadratic",
    },
    "gaussian_mixture": {
        "reference_fit_observation_ceiling": 30_000,
        "minimum_fit_observations": 3_000,
        "time_exponent": 1.0,
        "complexity_class": "iterative_observation_linear",
    },
    "variational_gaussian_mixture": {
        "reference_fit_observation_ceiling": 15_000,
        "minimum_fit_observations": 1_500,
        "time_exponent": 1.0,
        "complexity_class": "iterative_observation_linear_with_variational_overhead",
    },
    "affinity_propagation": {
        "reference_fit_observation_ceiling": 4_500,
        "minimum_fit_observations": 750,
        "time_exponent": 2.0,
        "complexity_class": "dense_similarity_quadratic",
    },
    "mean_shift": {
        "reference_fit_observation_ceiling": 9_000,
        "minimum_fit_observations": 1_500,
        "time_exponent": 2.0,
        "complexity_class": "bandwidth_dependent_neighbor_quadratic",
    },
    "ward": {
        "reference_fit_observation_ceiling": 10_000,
        "minimum_fit_observations": 2,
        "time_exponent": 2.0,
        "complexity_class": "full_observation_hierarchical_or_skip",
    },
    "quality_threshold": {
        "reference_fit_observation_ceiling": 10_000,
        "minimum_fit_observations": 2,
        "time_exponent": 2.0,
        "complexity_class": "full_observation_pairwise_or_skip",
    },
}


def alternative_clustering_fit_profiles() -> Dict[str, Dict[str, object]]:
    """Return the planner's explicit per-family complexity contracts."""

    return {
        algorithm: dict(profile)
        for algorithm, profile in _ALTERNATIVE_CLUSTERING_FIT_PROFILES.items()
    }


def plan_alternative_clustering_fit_strides(
    source_physical_frames_per_replica: Sequence[int],
    *,
    member_observation_multiplier: int,
    algorithms: Sequence[str],
    target_wall_hours: float,
) -> Dict[str, object]:
    """Choose an independent exact integer fit stride for every algorithm.

    The retained TREX 3,000-fit sweep calibrates the bundled implementation,
    but not the individual algorithms.  These per-family ceilings are therefore
    explicit conservative complexity profiles, scaled with the requested wall
    envelope and reported as provisional until per-algorithm pilots replace
    them. Ward and quality-threshold retain the agreed full-fit-or-skip policy.
    """

    counts = [int(value) for value in source_physical_frames_per_replica]
    if not counts or any(value <= 0 for value in counts):
        raise ResourcePlanningError(
            "alternative clustering requires positive per-replica frame counts"
        )
    if (
        isinstance(member_observation_multiplier, bool)
        or not isinstance(member_observation_multiplier, int)
        or member_observation_multiplier <= 0
    ):
        raise ResourcePlanningError(
            "member_observation_multiplier must be a positive integer"
        )
    hours = _positive_number(target_wall_hours, "target_wall_hours")
    if not algorithms or len(set(algorithms)) != len(algorithms):
        raise ResourcePlanningError(
            "alternative clustering algorithms must be a nonempty unique sequence"
        )
    unknown = sorted(set(algorithms).difference(_ALTERNATIVE_CLUSTERING_FIT_PROFILES))
    if unknown:
        raise ResourcePlanningError(
            "no alternative-clustering fit profile for: " + ", ".join(unknown)
        )

    full_observations = sum(counts) * member_observation_multiplier
    wall_scale = hours / 24.0
    plans: Dict[str, object] = {}
    for algorithm in algorithms:
        profile = _ALTERNATIVE_CLUSTERING_FIT_PROFILES[algorithm]
        exponent = float(profile["time_exponent"])
        reference_ceiling = int(profile["reference_fit_observation_ceiling"])
        ceiling = max(
            len(counts) * member_observation_multiplier * 2,
            math.floor(reference_ceiling * wall_scale ** (1.0 / exponent)),
        )
        ceiling = min(full_observations, ceiling)
        if algorithm in {"ward", "quality_threshold"} and ceiling < full_observations:
            plans[algorithm] = {
                "execution": "skip",
                "skip_reason": (
                    "full-observation fit exceeds the algorithm-specific "
                    "resource ceiling"
                ),
                "full_observation_count": full_observations,
                "fit_observation_ceiling": ceiling,
                "complexity_class": profile["complexity_class"],
                "time_exponent": exponent,
                "calibration_status": "provisional_complexity_profile_v1",
            }
            continue
        stride = 1
        while sum(
            member_observation_multiplier
            * integer_stride_selected_count(count, stride)
            for count in counts
        ) > ceiling:
            stride += 1
        selected_per_physical_replica = [
            member_observation_multiplier
            * integer_stride_selected_count(count, stride)
            for count in counts
        ]
        plans[algorithm] = {
            "execution": "run",
            "mode": "integer_stride_per_replica_member_v1",
            "strides": [stride],
            "primary_stride": stride,
            "fit_observation_ceiling": ceiling,
            "selected_fit_observations_per_physical_replica": (
                selected_per_physical_replica
            ),
            "selected_fit_observation_count": sum(
                selected_per_physical_replica
            ),
            "full_observation_count": full_observations,
            "complexity_class": profile["complexity_class"],
            "time_exponent": exponent,
            "calibration_status": "provisional_complexity_profile_v1",
        }
    return {
        "mode": "algorithm_specific_integer_stride_v1",
        "target_wall_hours": hours,
        "member_observation_multiplier": member_observation_multiplier,
        "source_physical_frames_per_replica": counts,
        "full_observation_count": full_observations,
        "algorithm_plans": plans,
        "scientific_boundary": (
            "Algorithm-specific fit ceilings are computational allocations, not "
            "evidence of clustering stability, metastability, or convergence."
        ),
    }


def plan_parallel_cpu_envelope(
    tasks: Sequence[Mapping[str, object]],
    *,
    maximum_parallel_cpus: int,
    maximum_hours_per_cpu: float,
    maximum_memory_gib: float,
) -> Dict[str, object]:
    """Plan scheduler-neutral task parallelism and a total CPU-hour envelope.

    ``dependency_stage`` is an abstract DAG level, not a scheduler directive.
    Launch adapters may translate the returned plan to Slurm, a local process
    pool, a workflow engine, or another execution environment.
    """

    if (
        isinstance(maximum_parallel_cpus, bool)
        or not isinstance(maximum_parallel_cpus, int)
        or maximum_parallel_cpus <= 0
    ):
        raise ResourcePlanningError("maximum_parallel_cpus must be a positive integer")
    hours = _positive_number(maximum_hours_per_cpu, "maximum_hours_per_cpu")
    memory = _positive_number(maximum_memory_gib, "maximum_memory_gib")
    if not tasks:
        raise ResourcePlanningError("parallel execution planning requires at least one task")
    normalized = []
    seen = set()
    for index, raw in enumerate(tasks):
        task_id = raw.get("task_id")
        stage = raw.get("dependency_stage")
        cap = raw.get("effective_cpu_cap")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ResourcePlanningError(f"task {index} has an invalid or duplicate task_id")
        if isinstance(stage, bool) or not isinstance(stage, int) or stage < 0:
            raise ResourcePlanningError(f"task {task_id} has an invalid dependency_stage")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise ResourcePlanningError(f"task {task_id} has an invalid effective_cpu_cap")
        estimated = raw.get("estimated_cpu_hours")
        if estimated is not None:
            estimated = _nonnegative_number(
                estimated, f"task {task_id} estimated_cpu_hours"
            )
        seen.add(task_id)
        normalized.append({
            **dict(raw),
            "task_id": task_id,
            "dependency_stage": stage,
            "effective_cpu_cap": cap,
            "estimated_cpu_hours": estimated,
        })
    stages = []
    for stage in sorted({int(row["dependency_stage"]) for row in normalized}):
        rows = [row for row in normalized if row["dependency_stage"] == stage]
        useful = sum(int(row["effective_cpu_cap"]) for row in rows)
        stages.append({
            "dependency_stage": stage,
            "ready_task_count": len(rows),
            "maximum_useful_parallel_cpus": useful,
            "planned_parallel_cpus": min(maximum_parallel_cpus, useful),
            "task_ids": [str(row["task_id"]) for row in rows],
        })
    useful_peak = max(int(row["maximum_useful_parallel_cpus"]) for row in stages)
    known_cpu_hours = sum(
        float(row["estimated_cpu_hours"])
        for row in normalized if row["estimated_cpu_hours"] is not None
    )
    unknown_count = sum(row["estimated_cpu_hours"] is None for row in normalized)
    total_budget = maximum_parallel_cpus * hours
    return {
        "planning_schema": "salsbury-scheduler-neutral-cpu-envelope-v1",
        "technical_status": "complete",
        "submission_adapter": "unspecified",
        "maximum_parallel_cpus_input": maximum_parallel_cpus,
        "maximum_hours_per_cpu_input": hours,
        "maximum_memory_gib_input": memory,
        "maximum_total_cpu_hours": total_budget,
        "maximum_useful_parallel_cpus_for_declared_graph": useful_peak,
        "additional_cpu_capacity_has_no_modeled_speedup": max(
            0, maximum_parallel_cpus - useful_peak
        ),
        "estimated_known_cpu_hours": known_cpu_hours,
        "tasks_without_cpu_hour_calibration": unknown_count,
        "known_cpu_hour_budget_fraction": known_cpu_hours / total_budget,
        "stages": stages,
        "tasks": normalized,
        "execution_contract": (
            "The planner describes dependencies and effective CPU caps only. It does "
            "not assume that tasks are submitted to Slurm or to any scheduler."
        ),
    }


def _positive_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ResourcePlanningError(f"{label} must be finite and positive")
    return float(value)


def _nonnegative_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ResourcePlanningError(f"{label} must be finite and nonnegative")
    return float(value)


def plan_campaign_resource_budget(
    tasks: Sequence[Mapping[str, object]],
    *,
    maximum_parallel_cpus: int,
    maximum_wall_hours: float,
    maximum_memory_gib: float,
    planning_utilization: float = 0.85,
    pilot_budget_fraction: float = 0.05,
    finalization_headroom_fraction: float = 0.0,
    memory_safety_factor: float = 1.0,
    memory_overhead_gib: float = 0.0,
    minimum_scheduler_memory_gib: float = 0.0,
) -> Dict[str, object]:
    """Allocate one hard CPU/wall envelope across an analysis campaign.

    Frame budgets are physical trajectory frames.  ``member_observation_multiplier``
    expands the reported observation count only; it never manufactures additional
    replicas or additional physical sampling.  Tasks sharing ``balance_group`` are
    advanced through the same per-replica frame budget so comparative systems cannot
    receive unequal coverage merely because one happens to be cheaper.

    The planner is scheduler-neutral.  It returns dependency-stage capacity and a
    conservative critical-path lower bound, but it does not submit or mutate work.
    """

    if (
        isinstance(maximum_parallel_cpus, bool)
        or not isinstance(maximum_parallel_cpus, int)
        or maximum_parallel_cpus <= 0
    ):
        raise ResourcePlanningError(
            "maximum_parallel_cpus must be a positive integer"
        )
    wall_hours = _positive_number(maximum_wall_hours, "maximum_wall_hours")
    memory_gib = _positive_number(maximum_memory_gib, "maximum_memory_gib")
    memory_factor = _positive_number(
        memory_safety_factor, "memory_safety_factor"
    )
    memory_overhead = _nonnegative_number(
        memory_overhead_gib, "memory_overhead_gib"
    )
    minimum_scheduler_memory = _nonnegative_number(
        minimum_scheduler_memory_gib, "minimum_scheduler_memory_gib"
    )
    utilization = _fraction(planning_utilization, "planning_utilization")
    pilot_fraction = _fraction(
        pilot_budget_fraction, "pilot_budget_fraction", allow_zero=True
    )
    finalization_fraction = _fraction(
        finalization_headroom_fraction,
        "finalization_headroom_fraction",
        allow_zero=True,
    )
    if pilot_fraction + finalization_fraction >= utilization:
        raise ResourcePlanningError(
            "pilot plus finalization headroom fractions must be smaller than "
            "planning_utilization"
        )
    if not tasks:
        raise ResourcePlanningError(
            "campaign resource planning requires at least one task"
        )

    normalized = []
    seen = set()
    for index, raw in enumerate(tasks):
        task_id = raw.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ResourcePlanningError(
                f"task {index} has an invalid or duplicate task_id"
            )
        stage = raw.get("dependency_stage")
        cap = raw.get("effective_cpu_cap", 1)
        if isinstance(stage, bool) or not isinstance(stage, int) or stage < 0:
            raise ResourcePlanningError(
                f"task {task_id} has an invalid dependency_stage"
            )
        if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
            raise ResourcePlanningError(
                f"task {task_id} has an invalid effective_cpu_cap"
            )
        raw_counts = raw.get("source_frames_per_replica")
        if (
            not isinstance(raw_counts, (list, tuple))
            or not raw_counts
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in raw_counts
            )
        ):
            raise ResourcePlanningError(
                f"task {task_id} source_frames_per_replica must contain "
                "positive integers"
            )
        source_counts = tuple(int(value) for value in raw_counts)
        minimum = raw.get("minimum_frames_per_replica")
        maximum = raw.get("maximum_frames_per_replica", max(source_counts))
        for value, label in (
            (minimum, "minimum_frames_per_replica"),
            (maximum, "maximum_frames_per_replica"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ResourcePlanningError(
                    f"task {task_id} {label} must be a positive integer"
                )
        if int(maximum) < int(minimum):
            raise ResourcePlanningError(
                f"task {task_id} maximum_frames_per_replica is smaller than "
                "its minimum"
            )
        rate = raw.get("cpu_seconds_per_physical_frame")
        if rate is not None:
            rate = _nonnegative_number(
                rate, f"task {task_id} cpu_seconds_per_physical_frame"
            )
        raw_power_model = raw.get("power_law_cost_model")
        power_model: Optional[Dict[str, float]] = None
        if raw_power_model is not None:
            if not isinstance(raw_power_model, Mapping):
                raise ResourcePlanningError(
                    f"task {task_id} power_law_cost_model must be an object"
                )
            power_model = {
                "calibration_observations": _positive_number(
                    raw_power_model.get("calibration_observations"),
                    f"task {task_id} power_law_cost_model."
                    "calibration_observations",
                ),
                "calibration_cpu_hours": _positive_number(
                    raw_power_model.get("calibration_cpu_hours"),
                    f"task {task_id} power_law_cost_model."
                    "calibration_cpu_hours",
                ),
                "time_exponent": _positive_number(
                    raw_power_model.get("time_exponent"),
                    f"task {task_id} power_law_cost_model.time_exponent",
                ),
                "calibration_memory_gib": _positive_number(
                    raw_power_model.get("calibration_memory_gib"),
                    f"task {task_id} power_law_cost_model."
                    "calibration_memory_gib",
                ),
                "memory_exponent": _positive_number(
                    raw_power_model.get("memory_exponent"),
                    f"task {task_id} power_law_cost_model.memory_exponent",
                ),
            }
        raw_measured_memory_model = raw.get("measured_memory_cost_model")
        measured_memory_model: Optional[Dict[str, float]] = None
        if raw_measured_memory_model is not None:
            if not isinstance(raw_measured_memory_model, Mapping):
                raise ResourcePlanningError(
                    f"task {task_id} measured_memory_cost_model must be an object"
                )
            measured_memory_model = {
                "calibration_observations": _positive_number(
                    raw_measured_memory_model.get("calibration_observations"),
                    f"task {task_id} measured_memory_cost_model."
                    "calibration_observations",
                ),
                "calibration_memory_gib": _positive_number(
                    raw_measured_memory_model.get("calibration_memory_gib"),
                    f"task {task_id} measured_memory_cost_model."
                    "calibration_memory_gib",
                ),
                "memory_exponent": _positive_number(
                    raw_measured_memory_model.get("memory_exponent"),
                    f"task {task_id} measured_memory_cost_model.memory_exponent",
                ),
                "minimum_observation_scale": _positive_number(
                    raw_measured_memory_model.get("minimum_observation_scale"),
                    f"task {task_id} measured_memory_cost_model."
                    "minimum_observation_scale",
                ),
            }
            if measured_memory_model["minimum_observation_scale"] > 1.0:
                raise ResourcePlanningError(
                    f"task {task_id} measured_memory_cost_model."
                    "minimum_observation_scale must not exceed one"
                )
        fixed = _nonnegative_number(
            raw.get("fixed_cpu_hours", 0.0),
            f"task {task_id} fixed_cpu_hours",
        )
        if rate is None and fixed == 0.0 and power_model is None:
            calibration_status = "project_pilot_required"
        else:
            calibration_status = str(
                raw.get("calibration_status", "calibrated")
            )
        task_memory = _positive_number(
            raw.get("estimated_peak_memory_gib", 1.0),
            f"task {task_id} estimated_peak_memory_gib",
        )
        weight = _positive_number(
            raw.get("priority_weight", 1.0),
            f"task {task_id} priority_weight",
        )
        multiplier = raw.get("member_observation_multiplier", 1)
        if (
            isinstance(multiplier, bool)
            or not isinstance(multiplier, int)
            or multiplier <= 0
        ):
            raise ResourcePlanningError(
                f"task {task_id} member_observation_multiplier must be a "
                "positive integer"
            )
        balance_group = raw.get("balance_group")
        if balance_group is not None and (
            not isinstance(balance_group, str) or not balance_group
        ):
            raise ResourcePlanningError(
                f"task {task_id} balance_group must be a nonempty string"
            )
        execution_bundle_id = raw.get("execution_bundle_id", task_id)
        if not isinstance(execution_bundle_id, str) or not execution_bundle_id:
            raise ResourcePlanningError(
                f"task {task_id} execution_bundle_id must be a nonempty string"
            )
        replica_sampling_mode = raw.get(
            "replica_sampling_mode", "balanced_pooled"
        )
        if replica_sampling_mode not in {
            "balanced_pooled", "independent_all_available"
        }:
            raise ResourcePlanningError(
                f"task {task_id} replica_sampling_mode must be balanced_pooled "
                "or independent_all_available"
            )
        seen.add(task_id)
        normalized.append({
            **dict(raw),
            "task_id": task_id,
            "dependency_stage": stage,
            "effective_cpu_cap": cap,
            "source_frames_per_replica": list(source_counts),
            "minimum_frames_per_replica": int(minimum),
            "maximum_frames_per_replica": int(maximum),
            "cpu_seconds_per_physical_frame": rate,
            "fixed_cpu_hours": fixed,
            "power_law_cost_model": power_model,
            "measured_memory_cost_model": measured_memory_model,
            "estimated_peak_memory_gib": task_memory,
            "priority_weight": weight,
            "member_observation_multiplier": int(multiplier),
            "balance_group": balance_group or task_id,
            "execution_bundle_id": execution_bundle_id,
            "calibration_status": calibration_status,
            "replica_sampling_mode": replica_sampling_mode,
        })

    raw_cpu_hours = maximum_parallel_cpus * wall_hours
    planned_cpu_hours = raw_cpu_hours * utilization
    reserved_pilot_cpu_hours = raw_cpu_hours * pilot_fraction
    reserved_finalization_cpu_hours = raw_cpu_hours * finalization_fraction
    science_cpu_hours = (
        planned_cpu_hours
        - reserved_pilot_cpu_hours
        - reserved_finalization_cpu_hours
    )
    science_wall_hours = wall_hours * (
        utilization - pilot_fraction - finalization_fraction
    )

    groups: Dict[str, list[Dict[str, object]]] = {}
    for row in normalized:
        groups.setdefault(str(row["balance_group"]), []).append(row)
    group_budgets: Dict[str, int] = {}
    group_strides: Dict[str, int] = {}
    selected: Dict[str, list[int]] = {}

    def group_selection(
        rows: Sequence[Mapping[str, object]], budget: int,
    ) -> tuple[int, Dict[str, list[int]]]:
        source_counts = [
            int(value)
            for row in rows
            for value in row["source_frames_per_replica"]  # type: ignore[union-attr]
        ]
        stride = integer_stride_for_budget(
            source_counts, budget, error_type=ResourcePlanningError
        )
        def minimum_is_satisfied(row: Mapping[str, object], stride: int) -> bool:
            source = [
                int(value)
                for value in row["source_frames_per_replica"]  # type: ignore[union-attr]
            ]
            retained = [
                integer_stride_selected_count(value, stride) for value in source
            ]
            if row["replica_sampling_mode"] == "balanced_pooled":
                # The declared pilot is a pooled technical minimum.  A single
                # common stride preserves temporal spacing and each replica's
                # proportional contribution without forcing a short conditioned
                # replica to select nearly every frame.
                requested = int(row["minimum_frames_per_replica"]) * len(source)
                return sum(retained) >= min(requested, sum(source))
            return all(
                count >= minimum or source_count < minimum
                for count, source_count in zip(retained, source)
                for minimum in [int(row["minimum_frames_per_replica"])]
            )

        # Refine the common integer stride only until each task's declared
        # minimum is met at its actual sampling scope.  Pooled methods use the
        # total retained observation count; replica-resolved methods retain the
        # historical per-replica gate.
        while stride > 1 and any(
            not minimum_is_satisfied(row, stride) for row in rows
        ):
            stride -= 1
        counts = {
            str(row["task_id"]): [
                integer_stride_selected_count(int(value), stride)
                for value in row["source_frames_per_replica"]  # type: ignore[union-attr]
            ]
            for row in rows
        }
        return stride, counts

    for group_id, rows in groups.items():
        declared_group_minimum = max(
            int(row["minimum_frames_per_replica"]) for row in rows
        )
        group_maximum = min(int(row["maximum_frames_per_replica"]) for row in rows)
        # A comparative balance group may contain a genuinely shorter system.
        # Its complete source is the attainable group minimum; the task report
        # retains ``source_limited_below_declared_minimum`` rather than treating
        # missing physical frames as an allocation error.
        group_minimum = min(declared_group_minimum, group_maximum)
        group_budgets[group_id] = group_minimum
        stride, counts = group_selection(rows, group_minimum)
        group_strides[group_id] = stride
        selected.update(counts)

    def task_cost(row: Mapping[str, object], counts: Sequence[int]) -> Optional[float]:
        power_model = row.get("power_law_cost_model")
        if isinstance(power_model, Mapping):
            observations = sum(counts) * int(row["member_observation_multiplier"])
            return float(row["fixed_cpu_hours"]) + float(
                power_model["calibration_cpu_hours"]
            ) * (
                observations / float(power_model["calibration_observations"])
            ) ** float(power_model["time_exponent"])
        rate = row["cpu_seconds_per_physical_frame"]
        if rate is None:
            return None
        return float(row["fixed_cpu_hours"]) + float(rate) * sum(counts) / 3600.0

    def task_memory(row: Mapping[str, object], counts: Sequence[int]) -> float:
        power_model = row.get("power_law_cost_model")
        observations = sum(counts) * int(row["member_observation_multiplier"])
        if isinstance(power_model, Mapping):
            return float(power_model["calibration_memory_gib"]) * max(
                1.0,
                (
                    observations / float(power_model["calibration_observations"])
                ) ** float(power_model["memory_exponent"]),
            )
        measured_model = row.get("measured_memory_cost_model")
        if isinstance(measured_model, Mapping):
            observation_scale = (
                observations / float(measured_model["calibration_observations"])
            ) ** float(measured_model["memory_exponent"])
            return max(
                1.0,
                float(measured_model["calibration_memory_gib"])
                * max(
                    float(measured_model["minimum_observation_scale"]),
                    observation_scale,
                ),
            )
        return float(row["estimated_peak_memory_gib"])

    def task_scheduler_memory(
        row: Mapping[str, object], counts: Sequence[int]
    ) -> float:
        return max(
            minimum_scheduler_memory,
            float(math.ceil(
                task_memory(row, counts) * memory_factor + memory_overhead
            )),
        )

    def known_costs(selection: Mapping[str, Sequence[int]]) -> Dict[str, float]:
        costs = {}
        for row in normalized:
            value = task_cost(row, selection[str(row["task_id"])])
            if value is not None:
                costs[str(row["task_id"])] = value
        return costs

    def schedule_summary(
        selection: Mapping[str, Sequence[int]],
    ) -> tuple[Optional[float], list[Dict[str, object]]]:
        costs = known_costs(selection)
        if len(costs) != len(normalized):
            return None, []
        stages = []
        total_wall = 0.0
        for stage in sorted({int(row["dependency_stage"]) for row in normalized}):
            rows = [row for row in normalized if int(row["dependency_stage"]) == stage]
            stage_cpu_hours = sum(costs[str(row["task_id"])] for row in rows)
            bundles: Dict[str, list[Mapping[str, object]]] = {}
            for row in rows:
                bundles.setdefault(str(row["execution_bundle_id"]), []).append(row)
            bundle_walls = {
                bundle_id: sum(
                    costs[str(row["task_id"])]
                    / min(maximum_parallel_cpus, int(row["effective_cpu_cap"]))
                    for row in bundle_rows
                )
                for bundle_id, bundle_rows in bundles.items()
            }
            bundle_resources = []
            for bundle_id, bundle_rows in bundles.items():
                bundle_resources.append({
                    "item_id": bundle_id,
                    "cpu_slots": max(
                        min(maximum_parallel_cpus, int(row["effective_cpu_cap"]))
                        for row in bundle_rows
                    ),
                    "memory_gib": max(
                        task_scheduler_memory(
                            row, selection[str(row["task_id"])]
                        )
                        for row in bundle_rows
                    ),
                    "wall_hours": bundle_walls[bundle_id],
                })
            try:
                resource_waves = pack_resource_waves(
                    bundle_resources,
                    maximum_parallel_cpus=maximum_parallel_cpus,
                    maximum_parallel_memory_gib=memory_gib,
                )
            except ResourcePlanningError:
                return None, []
            longest = max(bundle_walls.values())
            lower_bound = max(
                stage_cpu_hours / maximum_parallel_cpus,
                longest,
            )
            packed_wall = sum(
                float(wave["wall_hours"]) for wave in resource_waves
            )
            useful = sum(
                max(int(row["effective_cpu_cap"]) for row in bundle_rows)
                for bundle_rows in bundles.values()
            )
            stages.append({
                "dependency_stage": stage,
                "task_count": len(rows),
                "execution_bundle_count": len(bundles),
                "estimated_cpu_hours": stage_cpu_hours,
                "maximum_useful_parallel_cpus": useful,
                "planned_parallel_cpus": min(maximum_parallel_cpus, useful),
                "estimated_wall_hours_lower_bound": lower_bound,
                "estimated_wall_hours_with_resource_waves": packed_wall,
                "resource_waves": resource_waves,
                "task_ids": [str(row["task_id"]) for row in rows],
                "execution_bundle_wall_hours": bundle_walls,
            })
            total_wall += packed_wall
        return total_wall, stages

    calibration_required = sorted(
        str(row["task_id"])
        for row in normalized
        if row["cpu_seconds_per_physical_frame"] is None
        and row.get("power_law_cost_model") is None
        and float(row["fixed_cpu_hours"]) == 0.0
    )
    minimum_costs = known_costs(selected)
    minimum_known_cpu_hours = sum(minimum_costs.values())
    minimum_wall, minimum_stages = schedule_summary(selected)
    infeasibility_reasons = []
    alternative_method_config_names = {
        "pam": "pam",
        "mwpam": "minkowski_weighted_pam",
        "ward": "ward",
        "gaussian_mixture": "gaussian_mixture",
        "variational_gaussian_mixture": "variational_gaussian_mixture",
        "affinity_propagation": "affinity_propagation",
        "mean_shift": "mean_shift",
        "quality_threshold": "quality_threshold",
    }

    def memory_configuration_switch(row: Mapping[str, object]) -> str:
        module_id = str(row.get("module_id", row["task_id"]))
        if module_id == "coordinate_cache":
            return "execution.coordinate_cache"
        if module_id == "alternative_clustering" and row.get("algorithm_id"):
            method = alternative_method_config_names.get(
                str(row["algorithm_id"]), str(row["algorithm_id"])
            )
            return f"clustering.methods.{method}.enabled"
        dedicated = {
            "clustering_kmeans": "kmeans",
            "clustering_hdbscan": "hdbscan",
            "clustering_imwkmeans": "intelligent_minkowski_weighted_kmeans",
        }
        if module_id in dedicated:
            return f"clustering.methods.{dedicated[module_id]}.enabled"
        if module_id == "pald_community_analysis":
            return "community_analysis.pald.enabled"
        return f"modules.{module_id}.enabled"

    minimum_memory_rows = sorted(
        (
            {
                "task_id": str(row["task_id"]),
                "module_id": str(row.get("module_id", row["task_id"])),
                "workflow_id": str(row.get("workflow_id", "base")),
                "task_scope": str(row.get("task_scope", "unspecified")),
                "configuration_switch": memory_configuration_switch(row),
                "required_working_set_gib": task_memory(
                    row, selected[str(row["task_id"])]
                ),
                "required_memory_gib": task_scheduler_memory(
                    row, selected[str(row["task_id"])]
                ),
            }
            for row in normalized
        ),
        key=lambda row: (
            -float(row["required_memory_gib"]), str(row["task_id"])
        ),
    )
    minimum_required_memory_gib = max(
        float(row["required_memory_gib"])
        for row in minimum_memory_rows
    )
    oversized_memory_rows = [
        {
            **row,
            "configured_memory_gib": memory_gib,
            "shortfall_gib": (
                float(row["required_memory_gib"]) - memory_gib
            ),
        }
        for row in minimum_memory_rows
        if float(row["required_memory_gib"]) > memory_gib + 1.0e-12
    ]
    oversized_memory_tasks = [
        str(row["task_id"]) for row in oversized_memory_rows
    ]
    memory_modules_to_disable = sorted({
        str(row["module_id"]) for row in oversized_memory_rows
    })
    memory_switches_to_disable = sorted({
        str(row["configuration_switch"]) for row in oversized_memory_rows
    })
    if oversized_memory_rows:
        infeasibility_reasons.append(
            "minimum task memory exceeds the configured maximum for: "
            + ", ".join(oversized_memory_tasks)
        )
    if minimum_known_cpu_hours > science_cpu_hours:
        infeasibility_reasons.append(
            "minimum calibrated coverage exceeds the campaign science CPU-hour budget"
        )
    if minimum_wall is not None and minimum_wall > science_wall_hours:
        infeasibility_reasons.append(
            "minimum calibrated critical path exceeds the campaign science wall-time budget"
        )
    stride_maximum_overflows = sorted(
        str(row["task_id"])
        for row in normalized
        if any(
            count > int(row["maximum_frames_per_replica"])
            for count in selected[str(row["task_id"])]
        )
    )
    if stride_maximum_overflows:
        infeasibility_reasons.append(
            "no common integer stride can satisfy both the per-replica minimum "
            "and maximum frame constraints for: "
            + ", ".join(stride_maximum_overflows)
        )
    def maximum_group_budget(rows: Sequence[Mapping[str, object]]) -> int:
        return min(
            min(int(row["maximum_frames_per_replica"]) for row in rows),
            max(
                int(value)
                for row in rows
                for value in row["source_frames_per_replica"]  # type: ignore[union-attr]
            ),
        )

    def upgrade_candidates() -> tuple[
        list[Dict[str, object]], list[str], list[str]
    ]:
        """Return the next deterministic stride upgrade for every open group.

        ``required_science_wall_hours`` expresses both CPU-hour and packed-wall
        feasibility on the same absolute wall-time axis.  Ordering upgrades by
        the earliest resource frontier at which they become possible makes the
        allocation path independent of the final requested wall limit.  A
        longer envelope therefore extends the shorter plan instead of replacing
        inexpensive earlier upgrades with a newly affordable expensive one.
        """

        current_total = sum(known_costs(selected).values())
        candidates: list[Dict[str, object]] = []
        memory_blocked: list[str] = []
        at_ceiling: list[str] = []
        for group_id, rows in groups.items():
            current_budget = group_budgets[group_id]
            maximum_budget = maximum_group_budget(rows)
            if current_budget >= maximum_budget:
                at_ceiling.append(group_id)
                continue
            next_budget = min(
                maximum_budget,
                max(current_budget + 1, current_budget * 2),
            )
            proposed = {key: list(values) for key, values in selected.items()}
            next_stride, next_group_counts = group_selection(rows, next_budget)
            gain = 0.0
            for row in rows:
                task_id = str(row["task_id"])
                next_counts = next_group_counts[task_id]
                gain += float(row["priority_weight"]) * (
                    sum(next_counts) - sum(selected[task_id])
                ) / max(
                    1,
                    sum(
                        int(value)
                        for value in row["source_frames_per_replica"]  # type: ignore[union-attr]
                    ),
                )
                proposed[task_id] = next_counts
            proposed_costs = known_costs(proposed)
            proposed_total = sum(proposed_costs.values())
            delta = proposed_total - current_total
            proposed_wall, _ = schedule_summary(proposed)
            proposed_memory_fits = all(
                task_scheduler_memory(row, proposed[str(row["task_id"])])
                <= memory_gib + 1.0e-12
                for row in normalized
            )
            if proposed_wall is None or not proposed_memory_fits:
                memory_blocked.append(group_id)
                continue
            if delta < -1.0e-12:
                raise ResourcePlanningError(
                    f"allocation upgrade for {group_id} reduces modeled cost"
                )
            score = math.inf if delta <= 1.0e-12 else gain / delta
            candidates.append({
                "score": score,
                "gain": gain,
                "balance_group": group_id,
                "next_budget": next_budget,
                "next_stride": next_stride,
                "proposed": proposed,
                "proposed_cpu_hours": proposed_total,
                "proposed_wall_hours": proposed_wall,
                "required_science_wall_hours": max(
                    proposed_total / maximum_parallel_cpus,
                    proposed_wall,
                ),
            })
        return candidates, memory_blocked, at_ceiling

    allocation_order = []
    allocation_frontier_wall_hours = max(
        minimum_known_cpu_hours / maximum_parallel_cpus,
        0.0 if minimum_wall is None else minimum_wall,
    )
    allocation_stop_reason = "allocation_not_attempted"
    final_upgrade_candidates: list[Dict[str, object]] = []
    final_memory_blocked_groups: list[str] = []
    final_groups_at_ceiling: list[str] = []
    if not infeasibility_reasons and not calibration_required:
        while True:
            (
                candidates,
                memory_blocked_groups,
                groups_at_ceiling,
            ) = upgrade_candidates()
            final_upgrade_candidates = candidates
            final_memory_blocked_groups = memory_blocked_groups
            final_groups_at_ceiling = groups_at_ceiling
            if not candidates:
                allocation_stop_reason = (
                    "memory_ceiling_blocks_remaining_upgrades"
                    if memory_blocked_groups else
                    "all_eligible_frame_ceilings_reached"
                )
                break
            next_frontier = min(
                float(row["required_science_wall_hours"])
                for row in candidates
            )
            allocation_frontier_wall_hours = max(
                allocation_frontier_wall_hours, next_frontier
            )
            if (
                allocation_frontier_wall_hours
                > science_wall_hours + 1.0e-12
            ):
                allocation_stop_reason = (
                    "next_stride_upgrade_exceeds_campaign_envelope"
                )
                break
            active = [
                row for row in candidates
                if float(row["required_science_wall_hours"])
                <= allocation_frontier_wall_hours + 1.0e-12
            ]
            chosen = max(
                active,
                key=lambda row: (
                    float(row["score"]),
                    float(row["gain"]),
                    str(row["balance_group"]),
                ),
            )
            group_id = str(chosen["balance_group"])
            previous_budget = group_budgets[group_id]
            previous_stride = group_strides[group_id]
            group_budgets[group_id] = int(chosen["next_budget"])
            group_strides[group_id] = int(chosen["next_stride"])
            selected = chosen["proposed"]  # type: ignore[assignment]
            allocation_order.append({
                "balance_group": group_id,
                "previous_maximum_frames_per_replica": previous_budget,
                "new_maximum_frames_per_replica": group_budgets[group_id],
                "previous_integer_stride": previous_stride,
                "new_integer_stride": group_strides[group_id],
                "activation_science_wall_hours": float(
                    chosen["required_science_wall_hours"]
                ),
            })

    final_costs = known_costs(selected)
    final_known_cpu_hours = sum(final_costs.values())
    final_wall, final_stages = schedule_summary(selected)
    task_reports = []
    for row in normalized:
        task_id = str(row["task_id"])
        counts = selected[task_id]
        source_counts = [int(value) for value in row["source_frames_per_replica"]]  # type: ignore[union-attr]
        physical_count = sum(counts)
        source_count = sum(source_counts)
        all_frames = counts == source_counts
        selected_memory_gib = task_memory(row, counts)
        selected_scheduler_memory_gib = task_scheduler_memory(row, counts)
        task_reports.append({
            **row,
            "selected_physical_frames_per_replica": counts,
            "selected_physical_frame_count": physical_count,
            "selected_member_observation_count": (
                physical_count * int(row["member_observation_multiplier"])
            ),
            "coverage_fraction": physical_count / source_count,
            "subsampling_triggered": not all_frames,
            "frame_selection": (
                {"mode": "fixed_stride_v1"}
                if all_frames else {
                    "mode": "integer_stride_per_replica_v1",
                    "stride": group_strides[str(row["balance_group"])],
                }
            ),
            "integer_stride": group_strides[str(row["balance_group"])],
            "allocated_maximum_frames_per_replica": max(counts),
            "candidate_frame_ceiling_per_replica": group_budgets[
                str(row["balance_group"])
            ],
            "sampling_strategy": (
                "all source frames; no random draw"
                if all_frames else
                f"exact integer stride {group_strides[str(row['balance_group'])]} "
                "over every replica's concatenated timeline; frame zero retained; "
                "no random draw"
            ),
            "estimated_cpu_hours": final_costs.get(task_id),
            "estimated_peak_memory_gib_at_selected_observations": (
                selected_memory_gib
            ),
            "estimated_scheduler_memory_gib_at_selected_observations": (
                selected_scheduler_memory_gib
            ),
            "estimated_wall_hours_at_effective_cpu_cap": (
                final_costs[task_id]
                / min(maximum_parallel_cpus, int(row["effective_cpu_cap"]))
                if task_id in final_costs else None
            ),
            "source_limited_below_declared_minimum": (
                sum(source_counts)
                < int(row["minimum_frames_per_replica"]) * len(source_counts)
                if row["replica_sampling_mode"] == "balanced_pooled"
                else any(
                    source < int(row["minimum_frames_per_replica"])
                    for source in source_counts
                )
            ),
            "minimum_frame_scope": (
                "pooled_physical_frames"
                if row["replica_sampling_mode"] == "balanced_pooled"
                else "each_physical_replica"
            ),
            "minimum_selected_physical_frame_count": (
                min(
                    sum(source_counts),
                    int(row["minimum_frames_per_replica"])
                    * len(source_counts),
                )
                if row["replica_sampling_mode"] == "balanced_pooled"
                else None
            ),
            "independent_sampling_unit": (
                "original simulation replica and physical time block"
            ),
            "member_observations_are_independent_replicas": False,
        })

    if calibration_required:
        feasibility = "pilot_required"
    elif infeasibility_reasons:
        feasibility = "infeasible"
    else:
        feasibility = "feasible"
    unused_science_cpu_hours = max(
        0.0, science_cpu_hours - final_known_cpu_hours
    )
    science_cpu_hour_fraction = (
        final_known_cpu_hours / science_cpu_hours
        if science_cpu_hours > 0.0 else 0.0
    )
    science_wall_time_fraction = (
        final_wall / science_wall_hours
        if final_wall is not None and science_wall_hours > 0.0 else None
    )
    average_parallel_cpus = (
        final_known_cpu_hours / final_wall
        if final_wall is not None and final_wall > 0.0 else None
    )
    next_upgrade = (
        min(
            final_upgrade_candidates,
            key=lambda row: float(row["required_science_wall_hours"]),
        )
        if final_upgrade_candidates else None
    )
    if unused_science_cpu_hours <= 1.0e-12:
        unused_interpretation = "The science CPU-hour budget is fully allocated."
    elif (
        science_wall_time_fraction is not None
        and science_wall_time_fraction >= 0.9
    ):
        unused_interpretation = (
            "The campaign nearly saturates its science wall-time envelope. "
            "Unused CPU-hours reflect limited task parallelism, dependency stages, "
            "and per-task CPU caps; they are not omitted required analysis."
        )
    elif allocation_stop_reason == "all_eligible_frame_ceilings_reached":
        unused_interpretation = (
            "Every eligible task reached its configured frame ceiling. Unused "
            "CPU-hours reflect finite work and limited parallelism; the planner "
            "does not manufacture duplicate analysis to occupy idle cores."
        )
    elif allocation_stop_reason == "memory_ceiling_blocks_remaining_upgrades":
        unused_interpretation = (
            "The remaining stride upgrades exceed the aggregate memory policy. "
            "Unused CPU-hours cannot be spent without increasing memory or changing "
            "the enabled-method configuration."
        )
    else:
        unused_interpretation = (
            "The next deterministic stride upgrade exceeds the remaining campaign "
            "wall/CPU envelope. Unused CPU-hours are stranded by discrete stride "
            "steps and available parallelism, not silently discarded work."
        )
    return {
        "planning_schema": "salsbury-campaign-resource-plan-v1",
        "technical_status": "complete",
        "scientific_status": "planning only",
        "feasibility_status": feasibility,
        "execution_authorized": feasibility == "feasible",
        "maximum_parallel_cpus_input": maximum_parallel_cpus,
        "maximum_wall_hours_input": wall_hours,
        "maximum_memory_gib_input": memory_gib,
        "maximum_parallel_memory_gib_input": memory_gib,
        "scheduler_memory_safety_factor": memory_factor,
        "memory_overhead_gib": memory_overhead,
        "minimum_scheduler_memory_gib": minimum_scheduler_memory,
        "raw_capacity_cpu_hours": raw_cpu_hours,
        "planning_utilization": utilization,
        "planned_capacity_cpu_hours": planned_cpu_hours,
        "pilot_budget_fraction": pilot_fraction,
        "reserved_pilot_cpu_hours": reserved_pilot_cpu_hours,
        "finalization_headroom_fraction": finalization_fraction,
        "reserved_finalization_cpu_hours": reserved_finalization_cpu_hours,
        "science_budget_cpu_hours": science_cpu_hours,
        "science_budget_wall_hours": science_wall_hours,
        "minimum_known_cpu_hours": minimum_known_cpu_hours,
        "minimum_wall_hours_lower_bound": minimum_wall,
        "estimated_selected_cpu_hours": final_known_cpu_hours,
        "estimated_selected_wall_hours_lower_bound": final_wall,
        "unused_science_cpu_hours": unused_science_cpu_hours,
        "resource_budget_utilization": {
            "science_cpu_hour_fraction": science_cpu_hour_fraction,
            "science_wall_time_fraction": science_wall_time_fraction,
            "average_parallel_cpus_during_selected_schedule": (
                average_parallel_cpus
            ),
            "maximum_parallel_cpus": maximum_parallel_cpus,
        },
        "allocation_saturation": {
            "allocation_strategy": (
                "progressive_absolute_resource_frontier_v1"
            ),
            "stop_reason": allocation_stop_reason,
            "groups_total": len(groups),
            "groups_at_frame_ceiling": len(final_groups_at_ceiling),
            "groups_below_frame_ceiling": (
                len(groups) - len(final_groups_at_ceiling)
            ),
            "memory_blocked_groups": sorted(final_memory_blocked_groups),
            "next_upgrade_balance_group": (
                str(next_upgrade["balance_group"])
                if next_upgrade is not None else None
            ),
            "next_upgrade_required_science_wall_hours": (
                float(next_upgrade["required_science_wall_hours"])
                if next_upgrade is not None else None
            ),
            "unused_cpu_hour_interpretation": unused_interpretation,
            "monotonicity_contract": (
                "For identical tasks, CPU ceiling, memory ceiling, and planning "
                "fractions, increasing maximum wall time extends the deterministic "
                "allocation path and cannot reduce an earlier task's frame coverage."
            ),
        },
        "tasks_requiring_project_pilots": calibration_required,
        "infeasibility_reasons": infeasibility_reasons,
        "memory_feasibility": {
            "configured_memory_gib": memory_gib,
            "minimum_required_memory_gib": minimum_required_memory_gib,
            "minimum_required_working_set_gib": max(
                float(row["required_working_set_gib"])
                for row in minimum_memory_rows
            ),
            "recommended_memory_gib": float(
                math.ceil(minimum_required_memory_gib)
            ),
            "memory_shortfall_gib": max(
                0.0, minimum_required_memory_gib - memory_gib
            ),
            "fits_configured_memory": not oversized_memory_rows,
            "oversized_tasks": oversized_memory_rows,
            "modules_to_disable_to_fit_configured_memory": (
                memory_modules_to_disable
            ),
            "configuration_switches_to_disable_to_fit_configured_memory": (
                memory_switches_to_disable
            ),
            "recommendation": (
                "Increase execution.maximum_memory_gib to at least the reported "
                "recommended_memory_gib, or explicitly turn off every listed "
                "configuration switch and its dependent modules before replanning."
                if oversized_memory_rows else
                "All enabled task minima fit the configured memory ceiling."
            ),
            "memory_limit_semantics": (
                "maximum simultaneous safety-adjusted scheduler memory across "
                "all active campaign tasks"
            ),
        },
        "minimum_stages": minimum_stages,
        "stages": final_stages,
        "allocation_order": allocation_order,
        "tasks": task_reports,
        "execution_contract": (
            "The CPU and wall limits apply to the complete campaign. Enabled tasks "
            "receive declared minimum physical-frame coverage before additional "
            "frames are allocated. Dependency stages are packed into waves that "
            "respect both aggregate CPU and safety-adjusted memory ceilings. "
            "Infeasible or uncalibrated plans fail closed; "
            "no module is silently disabled and no scientific minimum is silently "
            "relaxed."
        ),
        "scientific_boundary": (
            "Computational affordability and deterministic coverage are not evidence "
            "of equilibration, convergence, metastability, kinetics, causality, or "
            "independent sampling."
        ),
    }


def _fraction(value: object, label: str, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ResourcePlanningError(f"{label} must be a finite fraction")
    number = float(value)
    lower_ok = number >= 0.0 if allow_zero else number > 0.0
    if not lower_ok or number > 1.0:
        raise ResourcePlanningError(f"{label} must be between zero and one")
    return number


def calibrate_from_benchmark(
    benchmark: Mapping[str, object],
    *,
    evaluated_frame_count: Optional[int] = None,
    calibration_id: Optional[str] = None,
) -> Dict[str, object]:
    """Create a linear frame-cost calibration from one completed pilot."""

    nested = benchmark.get("planner_benchmark")
    if isinstance(nested, dict):
        benchmark = nested

    if benchmark.get("technical_status") != "complete":
        raise ResourcePlanningError("calibration benchmark must be technically complete")
    module_id = benchmark.get("module_id")
    if not isinstance(module_id, str) or not module_id:
        raise ResourcePlanningError("benchmark module_id must be a nonempty string")
    coverage = benchmark.get("frame_coverage")
    if evaluated_frame_count is None and isinstance(coverage, dict):
        raw_count = coverage.get("estimator_selected_frame_count")
        if isinstance(raw_count, int) and not isinstance(raw_count, bool):
            evaluated_frame_count = raw_count
    if (
        isinstance(evaluated_frame_count, bool)
        or not isinstance(evaluated_frame_count, int)
        or evaluated_frame_count <= 0
    ):
        raise ResourcePlanningError(
            "evaluated_frame_count is required when benchmark frame_coverage is absent"
        )
    resources = benchmark.get("resources")
    if not isinstance(resources, dict):
        raise ResourcePlanningError("benchmark resources must be an object")
    wall_seconds = _positive_number(resources.get("wall_seconds"), "wall_seconds")
    maximum_rss_kib = _positive_number(
        resources.get("maximum_rss_kib"), "maximum_rss_kib"
    )
    report_size = _positive_number(
        benchmark.get("report_size_bytes"), "report_size_bytes"
    )
    identity = calibration_id or (
        f"{module_id}:{benchmark.get('project_sha256', 'unknown')}:"
        f"{benchmark.get('finished_utc', 'unknown')}"
    )
    return {
        "calibration_schema": "salsbury-frame-cost-calibration-v1",
        "calibration_id": identity,
        "module_id": module_id,
        "benchmark_project_sha256": benchmark.get("project_sha256"),
        "benchmark_report_sha256": benchmark.get("report_sha256"),
        "benchmark_environment": benchmark.get("environment"),
        "evaluated_frame_count": evaluated_frame_count,
        "wall_seconds": wall_seconds,
        "seconds_per_frame": wall_seconds / evaluated_frame_count,
        "fixed_overhead_seconds": 0.0,
        "maximum_rss_mib": maximum_rss_kib / 1024.0,
        "fixed_report_bytes": 0.0,
        "report_bytes_per_frame": report_size / evaluated_frame_count,
        "model": "linear frame extrapolation for the same method and workload dimensions",
        "limitations": [
            "A different atom, feature, candidate, water, sphere-point, or executable dimension requires a new pilot.",
            "Filesystem cache, node generation, contention, and output sparsity can change realized cost.",
            "The estimate is operational evidence, not scientific sufficiency or convergence evidence.",
        ],
    }


def calibrate_from_benchmarks(
    benchmarks: Sequence[Mapping[str, object]],
    *,
    evaluated_frame_counts: Optional[Sequence[int]] = None,
    calibration_id: Optional[str] = None,
) -> Dict[str, object]:
    """Fit fixed scan overhead plus incremental selected-frame cost."""

    if len(benchmarks) < 2:
        if not benchmarks:
            raise ResourcePlanningError("at least one benchmark is required")
        count = evaluated_frame_counts[0] if evaluated_frame_counts else None
        return calibrate_from_benchmark(
            benchmarks[0], evaluated_frame_count=count,
            calibration_id=calibration_id,
        )
    if evaluated_frame_counts is not None and len(evaluated_frame_counts) != len(benchmarks):
        raise ResourcePlanningError(
            "evaluated_frame_counts must match the number of benchmarks"
        )
    single = [
        calibrate_from_benchmark(
            benchmark,
            evaluated_frame_count=(
                evaluated_frame_counts[index]
                if evaluated_frame_counts is not None else None
            ),
        )
        for index, benchmark in enumerate(benchmarks)
    ]
    modules = {row["module_id"] for row in single}
    if len(modules) != 1:
        raise ResourcePlanningError("all calibration benchmarks must use one module")
    points = sorted(
        (
            int(row["evaluated_frame_count"]),
            float(row["wall_seconds"]),
            float(row["report_bytes_per_frame"])
            * int(row["evaluated_frame_count"]),
        )
        for row in single
    )
    if len({point[0] for point in points}) < 2:
        raise ResourcePlanningError(
            "multi-point calibration requires at least two distinct frame counts"
        )
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    slope = sum(
        (point[0] - mean_x) * (point[1] - mean_y) for point in points
    ) / denominator
    intercept = mean_y - slope * mean_x
    if slope <= 0.0:
        raise ResourcePlanningError(
            "benchmark wall time did not increase with selected frame count"
        )
    intercept = max(0.0, intercept)
    output_mean = sum(point[2] for point in points) / len(points)
    output_slope = sum(
        (point[0] - mean_x) * (point[2] - output_mean) for point in points
    ) / denominator
    output_intercept = output_mean - output_slope * mean_x
    if output_slope <= 0.0:
        output_slope = 0.0
        output_intercept = max(point[2] for point in points)
    else:
        output_intercept = max(0.0, output_intercept)
    identity = calibration_id or (
        f"{next(iter(modules))}:multi-point:"
        + ":".join(
            str(benchmark.get("report_sha256", "unknown"))[:12]
            for benchmark in benchmarks
        )
    )
    return {
        "calibration_schema": "salsbury-frame-cost-calibration-v1",
        "calibration_id": identity,
        "module_id": next(iter(modules)),
        "benchmark_project_sha256_values": [
            benchmark.get("project_sha256") for benchmark in benchmarks
        ],
        "benchmark_report_sha256_values": [
            benchmark.get("report_sha256") for benchmark in benchmarks
        ],
        "benchmark_environment_values": [
            benchmark.get("environment") for benchmark in benchmarks
        ],
        "evaluated_frame_counts": [point[0] for point in points],
        "wall_seconds_values": [point[1] for point in points],
        "fixed_overhead_seconds": intercept,
        "seconds_per_frame": slope,
        "maximum_rss_mib": max(float(row["maximum_rss_mib"]) for row in single),
        "fixed_report_bytes": output_intercept,
        "report_bytes_per_frame": output_slope,
        "model": "least-squares fixed scan overhead plus linear selected-frame cost",
        "limitations": [
            "Benchmarks must differ primarily in frame coverage for the same method and workload dimensions.",
            "A different atom, feature, candidate, water, sphere-point, or executable dimension requires a new pilot.",
            "Filesystem cache, node generation, contention, and output sparsity can change realized cost.",
            "The estimate is operational evidence, not scientific sufficiency or convergence evidence.",
        ],
    }


def calibrate_quadratic_from_benchmarks(
    benchmarks: Sequence[Mapping[str, object]],
    *,
    evaluated_fit_observation_counts: Sequence[int],
    calibration_id: Optional[str] = None,
) -> Dict[str, object]:
    """Fit fixed cost plus a quadratic sampled-observation term.

    This contract is intended for one unchanged alternative-clustering sweep.
    The all-frame assignment count and signed algorithm/parameter workload must
    be identical across pilots; only the balanced fit-observation count varies.
    """

    if len(benchmarks) < 2 or len(benchmarks) != len(evaluated_fit_observation_counts):
        raise ResourcePlanningError(
            "quadratic calibration requires at least two benchmarks and one fit count per benchmark"
        )
    rows = []
    signatures = set()
    full_counts = set()
    modules = set()
    for benchmark, fit_count in zip(benchmarks, evaluated_fit_observation_counts):
        source_benchmark = (
            benchmark["planner_benchmark"]
            if isinstance(benchmark.get("planner_benchmark"), dict)
            else benchmark
        )
        row = calibrate_from_benchmark(
            benchmark, evaluated_frame_count=fit_count
        )
        if isinstance(fit_count, bool) or not isinstance(fit_count, int) or fit_count <= 0:
            raise ResourcePlanningError("fit observation counts must be positive integers")
        signature = source_benchmark.get("workload_signature_sha256")
        if not isinstance(signature, str) or len(signature) != 64:
            raise ResourcePlanningError(
                "each quadratic benchmark requires workload_signature_sha256"
            )
        full_count = source_benchmark.get("full_assignment_observation_count")
        if isinstance(full_count, bool) or not isinstance(full_count, int) or full_count <= 0:
            raise ResourcePlanningError(
                "each quadratic benchmark requires full_assignment_observation_count"
            )
        signatures.add(signature)
        full_counts.add(full_count)
        modules.add(row["module_id"])
        rows.append({
            **row,
            "fit_observation_count": fit_count,
            "squared_fit_observation_count": fit_count * fit_count,
            "report_size_bytes": float(source_benchmark["report_size_bytes"]),
        })
    if len(signatures) != 1 or len(full_counts) != 1 or len(modules) != 1:
        raise ResourcePlanningError(
            "quadratic benchmarks must share module, workload signature, and full assignment count"
        )
    if len(set(evaluated_fit_observation_counts)) < 2:
        raise ResourcePlanningError(
            "quadratic calibration requires at least two distinct fit counts"
        )

    x_values = [float(row["squared_fit_observation_count"]) for row in rows]

    def fit(values: Sequence[float], label: str) -> tuple[float, float]:
        mean_x = sum(x_values) / len(x_values)
        mean_y = sum(values) / len(values)
        denominator = sum((value - mean_x) ** 2 for value in x_values)
        slope = sum(
            (x_value - mean_x) * (y_value - mean_y)
            for x_value, y_value in zip(x_values, values)
        ) / denominator
        intercept = mean_y - slope * mean_x
        if label == "wall time" and slope <= 0.0:
            raise ResourcePlanningError(
                "benchmark wall time did not increase with squared fit observation count"
            )
        if slope <= 0.0:
            return max(values), 0.0
        return max(0.0, intercept), slope

    wall_intercept, wall_slope = fit(
        [float(row["wall_seconds"]) for row in rows], "wall time"
    )
    memory_intercept, memory_slope = fit(
        [float(row["maximum_rss_mib"]) for row in rows], "peak memory"
    )
    identity = calibration_id or (
        f"{next(iter(modules))}:quadratic:"
        + ":".join(
            str(benchmark.get("report_sha256", "unknown"))[:12]
            for benchmark in benchmarks
        )
    )
    return {
        "calibration_schema": "salsbury-observation-cost-calibration-v1",
        "calibration_id": identity,
        "module_id": next(iter(modules)),
        "workload_signature_sha256": next(iter(signatures)),
        "full_assignment_observation_count": next(iter(full_counts)),
        "evaluated_fit_observation_counts": list(evaluated_fit_observation_counts),
        "wall_seconds_values": [float(row["wall_seconds"]) for row in rows],
        "maximum_rss_mib_values": [float(row["maximum_rss_mib"]) for row in rows],
        "fixed_overhead_seconds": wall_intercept,
        "seconds_per_squared_fit_observation": wall_slope,
        "fixed_memory_mib": memory_intercept,
        "memory_mib_per_squared_fit_observation": memory_slope,
        "maximum_calibrated_fit_observations": max(evaluated_fit_observation_counts),
        "maximum_report_size_bytes": max(
            float(row["report_size_bytes"]) for row in rows
        ),
        "model": (
            "least-squares fixed full-assignment overhead plus quadratic sampled-fit cost"
        ),
        "limitations": [
            "The calibration applies only to the identical signed algorithm and parameter sweep.",
            "PaLD is cubic and must be calibrated and planned separately.",
            "A different feature count, full-assignment count, implementation, environment, or algorithm grid requires a new calibration.",
            "The estimate is operational evidence, not scientific sufficiency or convergence evidence.",
        ],
    }


def recommend_quadratic_observation_budget(
    calibration: Mapping[str, object],
    *,
    total_source_observations: int,
    replica_count: int,
    target_wall_seconds: float = 14_400.0,
    target_memory_mib: float = 16_384.0,
    time_safety_factor: float = 1.5,
    memory_safety_factor: float = 1.25,
    minimum_observations_per_replica: int = 100,
    sensitivity_check_policy: str = "off",
    maximum_extrapolation_factor: float = 2.0,
) -> Dict[str, object]:
    """Choose one balanced fit budget for a calibrated quadratic sweep."""

    for value, label in (
        (total_source_observations, "total_source_observations"),
        (replica_count, "replica_count"),
        (minimum_observations_per_replica, "minimum_observations_per_replica"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ResourcePlanningError(f"{label} must be a positive integer")
    target_wall = _positive_number(target_wall_seconds, "target_wall_seconds")
    target_memory = _positive_number(target_memory_mib, "target_memory_mib")
    time_safety = _positive_number(time_safety_factor, "time_safety_factor")
    memory_safety = _positive_number(memory_safety_factor, "memory_safety_factor")
    extrapolation = _positive_number(
        maximum_extrapolation_factor, "maximum_extrapolation_factor"
    )
    wall_overhead = _nonnegative_number(
        calibration.get("fixed_overhead_seconds"), "fixed_overhead_seconds"
    )
    wall_slope = _positive_number(
        calibration.get("seconds_per_squared_fit_observation"),
        "seconds_per_squared_fit_observation",
    )
    memory_overhead = _nonnegative_number(
        calibration.get("fixed_memory_mib"), "fixed_memory_mib"
    )
    memory_slope = _nonnegative_number(
        calibration.get("memory_mib_per_squared_fit_observation"),
        "memory_mib_per_squared_fit_observation",
    )
    maximum_calibrated = calibration.get("maximum_calibrated_fit_observations")
    if (
        isinstance(maximum_calibrated, bool)
        or not isinstance(maximum_calibrated, int)
        or maximum_calibrated <= 0
    ):
        raise ResourcePlanningError(
            "maximum_calibrated_fit_observations must be a positive integer"
        )
    calibration_id = calibration.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ResourcePlanningError("calibration_id must be a nonempty string")
    if sensitivity_check_policy not in {"off", "recommend", "require"}:
        raise ResourcePlanningError(
            "sensitivity_check_policy must be off, recommend, or require"
        )

    usable_wall = target_wall / time_safety - wall_overhead
    time_capacity = math.floor(math.sqrt(usable_wall / wall_slope)) if usable_wall > 0 else 0
    usable_memory = target_memory / memory_safety - memory_overhead
    if usable_memory < 0:
        memory_capacity = 0
    elif memory_slope == 0.0:
        memory_capacity = total_source_observations
    else:
        memory_capacity = math.floor(math.sqrt(usable_memory / memory_slope))
    extrapolation_capacity = math.floor(maximum_calibrated * extrapolation)
    capacity = min(
        total_source_observations,
        time_capacity,
        memory_capacity,
        extrapolation_capacity,
    )
    per_replica = capacity // replica_count
    if per_replica < minimum_observations_per_replica:
        raise ResourcePlanningError(
            "resource envelope cannot fit the minimum quadratic-clustering sample per replica"
        )
    selected = min(total_source_observations, per_replica * replica_count)
    all_observations_fit = selected >= total_source_observations
    if all_observations_fit:
        fit_sampling = None
        budgets = []
    else:
        budgets = [per_replica]
        if sensitivity_check_policy in {"recommend", "require"}:
            lower = max(minimum_observations_per_replica, per_replica // 2)
            budgets = sorted(set([lower, per_replica]))
        estimated_source_per_replica = math.ceil(
            total_source_observations / replica_count
        )
        strides = []
        for budget in budgets:
            stride = max(1, math.ceil(estimated_source_per_replica / budget))
            while integer_stride_selected_count(
                estimated_source_per_replica, stride
            ) > budget:
                stride += 1
            strides.append(stride)
        primary_stride = strides[budgets.index(per_replica)]
        fit_sampling = {
            "mode": "integer_stride_per_replica_member_v1",
            "strides": sorted(set(strides), reverse=True),
            "primary_stride": primary_stride,
        }
    estimated_selected_wall = time_safety * (
        wall_overhead + wall_slope * selected * selected
    )
    estimated_selected_memory = memory_safety * (
        memory_overhead + memory_slope * selected * selected
    )
    estimated_full_wall = time_safety * (
        wall_overhead
        + wall_slope * total_source_observations * total_source_observations
    )
    estimated_full_memory = memory_safety * (
        memory_overhead
        + memory_slope * total_source_observations * total_source_observations
    )
    return {
        "planning_schema": "salsbury-quadratic-observation-resource-plan-v1",
        "module_id": calibration.get("module_id"),
        "calibration_id": calibration_id,
        "total_source_observations": total_source_observations,
        "replica_count": replica_count,
        "target_wall_seconds": target_wall,
        "target_memory_mib": target_memory,
        "time_safety_factor": time_safety,
        "memory_safety_factor": memory_safety,
        "maximum_extrapolation_factor": extrapolation,
        "time_limited_fit_observation_capacity": time_capacity,
        "memory_limited_fit_observation_capacity": memory_capacity,
        "extrapolation_limited_fit_observation_capacity": extrapolation_capacity,
        "resolved_fit_observation_count": selected,
        "resolved_maximum_observations_per_replica": per_replica,
        "resolved_fit_fraction": selected / total_source_observations,
        "all_observations_fit": all_observations_fit,
        "fit_sampling": fit_sampling,
        "sensitivity_check_policy": sensitivity_check_policy,
        "estimated_selected_wall_seconds": estimated_selected_wall,
        "estimated_selected_peak_memory_mib": estimated_selected_memory,
        "estimated_full_wall_seconds": estimated_full_wall,
        "estimated_full_peak_memory_mib": estimated_full_memory,
        "interpretation": (
            "Operational sampled-fit estimate; every source observation remains eligible for post-fit assignment."
        ),
    }


def recommend_frame_budget(
    calibration: Mapping[str, object],
    *,
    total_source_frames: int,
    replica_count: int,
    target_wall_seconds: float = 14_400.0,
    target_memory_mib: float = 16_384.0,
    time_safety_factor: float = 1.5,
    memory_safety_factor: float = 1.25,
    minimum_frames_per_replica: int = 100,
    sensitivity_check_policy: str = "off",
) -> Dict[str, object]:
    """Recommend all frames or an automatic balanced resource envelope."""

    for value, label in (
        (total_source_frames, "total_source_frames"),
        (replica_count, "replica_count"),
        (minimum_frames_per_replica, "minimum_frames_per_replica"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ResourcePlanningError(f"{label} must be a positive integer")
    target_wall = _positive_number(target_wall_seconds, "target_wall_seconds")
    target_memory = _positive_number(target_memory_mib, "target_memory_mib")
    time_safety = _positive_number(time_safety_factor, "time_safety_factor")
    memory_safety = _positive_number(memory_safety_factor, "memory_safety_factor")
    rate = _positive_number(calibration.get("seconds_per_frame"), "seconds_per_frame")
    overhead_value = calibration.get("fixed_overhead_seconds", 0.0)
    if (
        isinstance(overhead_value, bool)
        or not isinstance(overhead_value, (int, float))
        or not math.isfinite(float(overhead_value))
        or float(overhead_value) < 0.0
    ):
        raise ResourcePlanningError(
            "fixed_overhead_seconds must be finite and nonnegative"
        )
    overhead = float(overhead_value)
    pilot_memory = _positive_number(calibration.get("maximum_rss_mib"), "maximum_rss_mib")
    bytes_per_frame = _nonnegative_number(
        calibration.get("report_bytes_per_frame"), "report_bytes_per_frame"
    )
    fixed_report_bytes = _nonnegative_number(
        calibration.get("fixed_report_bytes", 0.0), "fixed_report_bytes"
    )
    calibration_id = calibration.get("calibration_id")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ResourcePlanningError("calibration_id must be a nonempty string")
    if sensitivity_check_policy not in {"off", "recommend", "require"}:
        raise ResourcePlanningError(
            "sensitivity_check_policy must be off, recommend, or require"
        )

    estimated_memory = pilot_memory * memory_safety
    full_wall = (overhead + rate * total_source_frames) * time_safety
    full_output = fixed_report_bytes + bytes_per_frame * total_source_frames
    memory_fits = estimated_memory <= target_memory
    full_fits = memory_fits and full_wall <= target_wall
    if full_fits:
        selected_frames = total_source_frames
        per_replica_budget = None
        resolved_mode = "fixed_stride_v1"
        selection = {"mode": "fixed_stride_v1"}
        reason = None
    elif not memory_fits:
        selected_frames = 0
        per_replica_budget = None
        resolved_mode = "dimension_reduction_required"
        selection = None
        reason = (
            "pilot peak memory with safety factor exceeds the target; frame "
            "subsampling is not assumed to fix a frame-independent memory cost"
        )
    else:
        usable = target_wall / time_safety - overhead
        capacity = math.floor(usable / rate) if usable > 0.0 else 0
        per_replica_budget = capacity // replica_count
        if per_replica_budget < minimum_frames_per_replica:
            raise ResourcePlanningError(
                "target wall-time envelope cannot fit the minimum per-replica pilot"
            )
        quotient, remainder = divmod(total_source_frames, replica_count)
        inferred_counts = [
            quotient + (1 if index < remainder else 0)
            for index in range(replica_count)
        ]
        integer_stride = integer_stride_for_budget(
            inferred_counts,
            per_replica_budget,
            error_type=ResourcePlanningError,
        )
        selected_frames = sum(
            integer_stride_selected_count(count, integer_stride)
            for count in inferred_counts
        )
        resolved_mode = "integer_stride_per_replica_v1"
        reason = "estimated full-frame wall time exceeds the target envelope"
        selection = {
            "mode": "integer_stride_per_replica_v1",
            "stride": integer_stride,
        }
    return {
        "planning_schema": "salsbury-frame-resource-plan-v1",
        "module_id": calibration.get("module_id"),
        "calibration_id": calibration_id,
        "total_source_frames": total_source_frames,
        "replica_count": replica_count,
        "target_wall_seconds": target_wall,
        "target_memory_mib": target_memory,
        "time_safety_factor": time_safety,
        "memory_safety_factor": memory_safety,
        "estimated_full_wall_seconds": full_wall,
        "estimated_fixed_overhead_seconds": overhead,
        "estimated_full_peak_memory_mib": estimated_memory,
        "estimated_full_report_bytes": full_output,
        "all_frames_fit": full_fits,
        "resolved_mode": resolved_mode,
        "resolved_selected_frame_count": selected_frames,
        "resolved_maximum_frames_per_replica": per_replica_budget,
        "resolved_coverage_fraction": selected_frames / total_source_frames,
        "subsampling_reason": reason,
        "sensitivity_check_policy": sensitivity_check_policy,
        "frame_selection": selection,
        "interpretation": (
            "Operational estimate only; execution reports actual coverage and resources."
        ),
    }
