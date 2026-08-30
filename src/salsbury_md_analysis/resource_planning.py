"""Conservative frame-budget estimates from an actual method/project pilot.

The planner does not infer scientific sufficiency. It converts retained
technical benchmark evidence into an explicit all-frame or balanced-subsample
execution contract for the same method, system, code, and environment class.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Dict, Mapping, Optional, Sequence

from .frame_sampling import (
    integer_stride_for_budget,
    integer_stride_selected_count,
)
from .scientific_sampling import (
    POLICY_ID,
    ScientificSamplingError,
    assess_raw_sampling,
    profile_from_contract,
    scientific_sampling_profile,
)


class ResourcePlanningError(ValueError):
    """Raised when benchmark evidence cannot support a resource estimate."""


def workflow_useful_parallel_cpu_ceiling(
    tasks: Sequence[Mapping[str, object]],
    *,
    maximum_cpus_per_node: Optional[int] = None,
) -> int:
    """Return the dependency-stage CPU peak without a user or cluster cap."""

    stages: Dict[int, Dict[str, int]] = {}
    for row in tasks:
        stage = int(row.get("dependency_stage", 0))
        bundle = str(
            row.get("execution_bundle_id") or row.get("task_id", "task")
        )
        cap = max(
            1,
            int(row.get("intrinsic_cpu_cap", row.get("effective_cpu_cap", 1))),
        )
        if maximum_cpus_per_node is not None:
            cap = min(cap, maximum_cpus_per_node)
        stage_bundles = stages.setdefault(stage, {})
        stage_bundles[bundle] = max(cap, stage_bundles.get(bundle, 0))
    if not stages:
        raise ResourcePlanningError("workflow has no tasks for CPU capacity")
    return max(sum(bundles.values()) for bundles in stages.values())


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


def pack_resource_lanes(
    items: Sequence[Mapping[str, object]],
    *,
    maximum_parallel_cpus: int,
    maximum_parallel_memory_gib: float,
    maximum_cpus_per_node: Optional[int] = None,
    maximum_memory_gib_per_node: Optional[float] = None,
    maximum_nodes: Optional[int] = None,
) -> list[Dict[str, object]]:
    """Assign tasks to independent, aggregate-resource-bounded serial lanes.

    One task may run at a time in each lane.  A lane therefore reserves the
    maximum CPU and memory request of any task assigned to it, while its wall
    estimate is the sum of its tasks.  The sum of the lane reservations never
    exceeds the campaign caps.  Unlike whole-wave barriers, a slow task delays
    only later work in its own lane; scientific prerequisites remain separate
    task-level dependencies.

    Input order is retained within every lane.  Callers must provide tasks in
    dependency-safe order.
    """

    if maximum_parallel_cpus <= 0 or maximum_parallel_memory_gib <= 0.0:
        raise ResourcePlanningError("resource-lane limits must be positive")
    node_limits_supplied = (
        maximum_cpus_per_node is not None
        or maximum_memory_gib_per_node is not None
        or maximum_nodes is not None
    )
    if node_limits_supplied and (
        maximum_cpus_per_node is None
        or maximum_memory_gib_per_node is None
    ):
        raise ResourcePlanningError(
            "node-aware packing requires both maximum_cpus_per_node and "
            "maximum_memory_gib_per_node"
        )
    if maximum_cpus_per_node is not None and (
        isinstance(maximum_cpus_per_node, bool)
        or not isinstance(maximum_cpus_per_node, int)
        or maximum_cpus_per_node <= 0
    ):
        raise ResourcePlanningError(
            "maximum_cpus_per_node must be a positive integer"
        )
    if maximum_memory_gib_per_node is not None and (
        isinstance(maximum_memory_gib_per_node, bool)
        or not isinstance(maximum_memory_gib_per_node, (int, float))
        or not math.isfinite(float(maximum_memory_gib_per_node))
        or float(maximum_memory_gib_per_node) <= 0.0
    ):
        raise ResourcePlanningError(
            "maximum_memory_gib_per_node must be finite and positive"
        )
    if maximum_nodes is not None and (
        isinstance(maximum_nodes, bool)
        or not isinstance(maximum_nodes, int)
        or maximum_nodes <= 0
    ):
        raise ResourcePlanningError("maximum_nodes must be a positive integer")
    if maximum_cpus_per_node is not None and maximum_nodes is None:
        assert maximum_memory_gib_per_node is not None
        maximum_nodes = max(
            math.ceil(maximum_parallel_cpus / maximum_cpus_per_node),
            math.ceil(
                maximum_parallel_memory_gib
                / float(maximum_memory_gib_per_node)
            ),
        )
    normalized: list[Dict[str, object]] = []
    for index, item in enumerate(items):
        item_id = str(item.get("item_id", f"item-{index}"))
        cpus = int(item.get("cpu_slots", 1))
        memory = float(item.get("memory_gib", 0.0))
        wall = float(item.get("wall_hours", 0.0))
        if cpus <= 0 or memory <= 0.0 or wall < 0.0:
            raise ResourcePlanningError(
                f"resource-lane item {item_id} has invalid resources"
            )
        if cpus > maximum_parallel_cpus:
            raise ResourcePlanningError(
                f"resource-lane item {item_id} requests {cpus} CPUs, exceeding "
                f"the campaign limit {maximum_parallel_cpus}"
            )
        if memory > maximum_parallel_memory_gib + 1.0e-12:
            raise ResourcePlanningError(
                f"resource-lane item {item_id} requests {memory:g} GiB, "
                f"exceeding the campaign limit {maximum_parallel_memory_gib:g} GiB"
            )
        if (
            maximum_cpus_per_node is not None
            and cpus > maximum_cpus_per_node
        ):
            raise ResourcePlanningError(
                f"resource-lane item {item_id} requests {cpus} CPUs, exceeding "
                f"the per-node limit {maximum_cpus_per_node}"
            )
        if (
            maximum_memory_gib_per_node is not None
            and memory > float(maximum_memory_gib_per_node) + 1.0e-12
        ):
            raise ResourcePlanningError(
                f"resource-lane item {item_id} requests {memory:g} GiB after "
                "scheduler padding, exceeding the per-node limit "
                f"{float(maximum_memory_gib_per_node):g} GiB"
            )
        normalized.append({
            **dict(item),
            "item_id": item_id,
            "cpu_slots": cpus,
            "memory_gib": memory,
            "wall_hours": wall,
            "_input_index": index,
        })

    # Establish lane envelopes with the largest requests first.  This prevents
    # many small lanes from consuming aggregate capacity that a later large
    # task needs.  Items are restored to caller order within each lane below.
    ordered = sorted(
        normalized,
        key=lambda row: (
            -float(row["memory_gib"]),
            -int(row["cpu_slots"]),
            -float(row["wall_hours"]),
            str(row["item_id"]),
        ),
    )
    lanes: list[Dict[str, object]] = []

    def totals() -> tuple[int, float]:
        return (
            sum(int(lane["cpu_slots"]) for lane in lanes),
            sum(float(lane["memory_gib"]) for lane in lanes),
        )

    for item in ordered:
        total_cpu, total_memory = totals()
        can_open = (
            total_cpu + int(item["cpu_slots"]) <= maximum_parallel_cpus
            and total_memory + float(item["memory_gib"])
            <= maximum_parallel_memory_gib + 1.0e-12
        )
        candidates: list[tuple[float, float, int, Dict[str, object]]] = []
        for lane in lanes:
            new_cpu = max(int(lane["cpu_slots"]), int(item["cpu_slots"]))
            new_memory = max(
                float(lane["memory_gib"]), float(item["memory_gib"])
            )
            cpu_delta = new_cpu - int(lane["cpu_slots"])
            memory_delta = new_memory - float(lane["memory_gib"])
            if (
                total_cpu + cpu_delta <= maximum_parallel_cpus
                and total_memory + memory_delta
                <= maximum_parallel_memory_gib + 1.0e-12
            ):
                candidates.append((
                    float(lane["wall_hours"]) + float(item["wall_hours"]),
                    memory_delta + float(cpu_delta),
                    int(lane["lane_index"]),
                    lane,
                ))
        # Prefer a new lane when it reduces the predicted makespan.  Otherwise
        # grow the least-loaded compatible lane by the smallest reservation.
        selected: Optional[Dict[str, object]] = None
        if can_open:
            existing_finish = min(
                (candidate[0] for candidate in candidates), default=math.inf
            )
            if float(item["wall_hours"]) < existing_finish - 1.0e-12:
                selected = {
                    "lane_index": len(lanes),
                    "cpu_slots": int(item["cpu_slots"]),
                    "memory_gib": float(item["memory_gib"]),
                    "wall_hours": 0.0,
                    "items": [],
                }
                lanes.append(selected)
        if selected is None:
            if not candidates:
                if not can_open:
                    raise ResourcePlanningError(
                        f"resource-lane item {item['item_id']} cannot be assigned "
                        "within the aggregate CPU and memory caps"
                    )
                selected = {
                    "lane_index": len(lanes),
                    "cpu_slots": int(item["cpu_slots"]),
                    "memory_gib": float(item["memory_gib"]),
                    "wall_hours": 0.0,
                    "items": [],
                }
                lanes.append(selected)
            else:
                selected = min(candidates, key=lambda value: value[:3])[3]
                selected["cpu_slots"] = max(
                    int(selected["cpu_slots"]), int(item["cpu_slots"])
                )
                selected["memory_gib"] = max(
                    float(selected["memory_gib"]), float(item["memory_gib"])
                )
        selected["wall_hours"] = (
            float(selected["wall_hours"]) + float(item["wall_hours"])
        )
        selected["items"].append(dict(item))  # type: ignore[union-attr]

    def node_assignment(
        candidate_lanes: Sequence[Mapping[str, object]],
    ) -> Optional[list[Dict[str, object]]]:
        if maximum_cpus_per_node is None:
            return []
        assert maximum_memory_gib_per_node is not None
        assert maximum_nodes is not None
        ordered_lanes = sorted(
            candidate_lanes,
            key=lambda row: (
                -max(
                    int(row["cpu_slots"]) / maximum_cpus_per_node,
                    float(row["memory_gib"])
                    / float(maximum_memory_gib_per_node),
                ),
                -float(row["memory_gib"]),
                -int(row["cpu_slots"]),
                int(row["lane_index"]),
            ),
        )
        nodes: list[Dict[str, object]] = []
        for lane in ordered_lanes:
            selected_node: Optional[Dict[str, object]] = None
            for node in nodes:
                if (
                    int(node["cpu_slots"]) + int(lane["cpu_slots"])
                    <= maximum_cpus_per_node
                    and float(node["memory_gib"])
                    + float(lane["memory_gib"])
                    <= float(maximum_memory_gib_per_node) + 1.0e-12
                ):
                    selected_node = node
                    break
            if selected_node is None:
                if len(nodes) >= maximum_nodes:
                    return None
                selected_node = {
                    "node_index": len(nodes),
                    "cpu_slots": 0,
                    "memory_gib": 0.0,
                    "lane_indices": [],
                }
                nodes.append(selected_node)
            selected_node["cpu_slots"] = (
                int(selected_node["cpu_slots"]) + int(lane["cpu_slots"])
            )
            selected_node["memory_gib"] = (
                float(selected_node["memory_gib"])
                + float(lane["memory_gib"])
            )
            selected_node["lane_indices"].append(  # type: ignore[union-attr]
                int(lane["lane_index"])
            )
        return nodes

    # If the aggregate-optimal lane set fragments badly across physical nodes,
    # serialize the least costly pair of lanes until every simultaneous lane
    # reservation has a valid node bin.  Merging lanes cannot increase their
    # CPU or memory envelope; it changes only the predicted wall time.
    assigned_nodes = node_assignment(lanes)
    while maximum_cpus_per_node is not None and assigned_nodes is None:
        if len(lanes) < 2:
            raise ResourcePlanningError(
                "resource lanes cannot be packed into the configured node count"
            )
        first_index, second_index = min(
            (
                (left, right)
                for left in range(len(lanes))
                for right in range(left + 1, len(lanes))
            ),
            key=lambda pair: (
                float(lanes[pair[0]]["wall_hours"])
                + float(lanes[pair[1]]["wall_hours"]),
                max(
                    float(lanes[pair[0]]["memory_gib"]),
                    float(lanes[pair[1]]["memory_gib"]),
                ),
                pair,
            ),
        )
        first = lanes[first_index]
        second = lanes[second_index]
        first["cpu_slots"] = max(
            int(first["cpu_slots"]), int(second["cpu_slots"])
        )
        first["memory_gib"] = max(
            float(first["memory_gib"]), float(second["memory_gib"])
        )
        first["wall_hours"] = (
            float(first["wall_hours"]) + float(second["wall_hours"])
        )
        first["items"].extend(second["items"])  # type: ignore[union-attr]
        del lanes[second_index]
        for lane_index, lane in enumerate(lanes):
            lane["lane_index"] = lane_index
        assigned_nodes = node_assignment(lanes)

    node_by_lane = {}
    for node in assigned_nodes or []:
        for lane_index in node["lane_indices"]:  # type: ignore[union-attr]
            node_by_lane[int(lane_index)] = int(node["node_index"])

    for lane in lanes:
        lane_items = sorted(
            lane["items"], key=lambda row: int(row["_input_index"])
        )
        for row in lane_items:
            row.pop("_input_index", None)
        lane["items"] = lane_items
        if maximum_cpus_per_node is not None:
            lane["planned_node_index"] = node_by_lane[int(lane["lane_index"])]
            lane["maximum_cpus_per_node"] = maximum_cpus_per_node
            lane["maximum_memory_gib_per_node"] = float(
                maximum_memory_gib_per_node
            )
            lane["maximum_nodes"] = maximum_nodes
    return lanes


def _permissive_minimum_resource_request(
    *,
    request_scope: str,
    minimum_cpu_hours: float,
    minimum_wall_hours: Optional[float],
    minimum_stages: Sequence[Mapping[str, object]],
    maximum_parallel_cpus: int,
    maximum_wall_hours: float,
    maximum_memory_gib: float,
    planning_utilization: float,
    pilot_budget_fraction: float,
    finalization_headroom_fraction: float,
    memory_safety_factor: float,
    memory_overhead_gib: float,
    minimum_scheduler_memory_gib: float,
    minimum_single_task_memory_gib: float,
    maximum_cpus_per_node: Optional[int],
    maximum_memory_gib_per_node: Optional[float],
    maximum_nodes: Optional[int],
) -> Dict[str, object]:
    """Describe a padded scheduler request for one modeled minimum schedule.

    The request preserves the lane packing produced under the caller's CPU and
    aggregate-memory caps. It is intentionally a permissive execution floor,
    not a convergence or scientific-sufficiency claim.
    """

    science_wall_fraction = (
        planning_utilization
        - pilot_budget_fraction
        - finalization_headroom_fraction
    )
    stage_lanes = [
        [lane for lane in (
            stage.get("resource_lanes", stage.get("resource_waves", []))
            if isinstance(
                stage.get("resource_lanes", stage.get("resource_waves", [])),
                list,
            ) else []
        ) if isinstance(lane, Mapping)]
        for stage in minimum_stages
    ]
    lanes = [lane for stage in stage_lanes for lane in stage]
    parallel_cpus = max(
        (sum(int(lane.get("cpu_slots", 0)) for lane in stage) for stage in stage_lanes),
        default=0,
    )
    aggregate_memory = max(
        (
            sum(float(lane.get("memory_gib", 0.0)) for lane in stage)
            for stage in stage_lanes
        ),
        default=0.0,
    )
    planned_nodes = max(
        (
            len({
                int(lane["planned_node_index"])
                for lane in stage
                if lane.get("planned_node_index") is not None
            })
            for stage in stage_lanes
        ),
        default=0,
    )
    exact_requested_wall = (
        float(minimum_wall_hours) / science_wall_fraction
        if minimum_wall_hours is not None and science_wall_fraction > 0.0
        else None
    )
    rounded_requested_wall = (
        int(math.ceil(exact_requested_wall - 1.0e-12))
        if exact_requested_wall is not None else None
    )
    resource_schedule_available = (
        minimum_wall_hours is not None
        and bool(lanes)
        and parallel_cpus <= maximum_parallel_cpus
        and aggregate_memory <= maximum_memory_gib + 1.0e-12
        and minimum_single_task_memory_gib <= maximum_memory_gib + 1.0e-12
    )
    fits_input_wall_cap = bool(
        resource_schedule_available
        and exact_requested_wall is not None
        and exact_requested_wall <= maximum_wall_hours + 1.0e-12
    )
    if not resource_schedule_available:
        request_status = "unavailable_within_cpu_or_memory_caps"
    elif fits_input_wall_cap:
        request_status = "available_within_all_input_caps"
    else:
        request_status = "requires_larger_wall_time"
    return {
        "request_schema": "salsbury-permissive-minimum-resource-request-v1",
        "request_scope": request_scope,
        "status": request_status,
        "recommended_request": {
            "parallel_cpus": parallel_cpus if parallel_cpus > 0 else None,
            "aggregate_memory_gib": (
                float(math.ceil(aggregate_memory)) if aggregate_memory > 0.0
                else None
            ),
            "wall_hours": rounded_requested_wall,
            **({"nodes": planned_nodes} if planned_nodes > 0 else {}),
        },
        "unrounded_request": {
            "aggregate_memory_gib": aggregate_memory or None,
            "wall_hours": exact_requested_wall,
        },
        "modeled_minimum": {
            "cpu_hours": minimum_cpu_hours,
            "science_critical_path_hours": minimum_wall_hours,
            "single_task_memory_floor_gib": minimum_single_task_memory_gib,
        },
        "input_caps": {
            "parallel_cpus": maximum_parallel_cpus,
            "aggregate_memory_gib": maximum_memory_gib,
            "wall_hours": maximum_wall_hours,
            "cpus_per_node": maximum_cpus_per_node,
            "memory_gib_per_node": maximum_memory_gib_per_node,
            "maximum_nodes": maximum_nodes,
        },
        "fits_input_wall_cap": fits_input_wall_cap,
        "additional_wall_hours_required": (
            max(0, int(math.ceil(exact_requested_wall - maximum_wall_hours)))
            if exact_requested_wall is not None else None
        ),
        "padding_factors": {
            "planning_utilization": planning_utilization,
            "pilot_budget_fraction": pilot_budget_fraction,
            "finalization_headroom_fraction": finalization_headroom_fraction,
            "science_wall_fraction": science_wall_fraction,
            "scheduler_memory_safety_factor": memory_safety_factor,
            "scheduler_memory_overhead_gib_per_task": memory_overhead_gib,
            "scheduler_minimum_memory_gib_per_task": (
                minimum_scheduler_memory_gib
            ),
            "modeled_task_runtime_padding": (
                "already included in the supplied calibrated task costs; the "
                "campaign planner does not apply a second time multiplier"
            ),
        },
        "interpretation": (
            "CPU and memory preserve the modeled minimum resource-lane schedule "
            "under the supplied caps. Wall time reverses the configured campaign "
            "utilization and pilot/finalization reserves, then rounds up to a "
            "whole scheduler hour."
        ),
        "warning": {
            "severity": "warning",
            "code": "PERMISSIVE_MINIMUM_NOT_SCIENTIFIC_SUFFICIENCY",
            "message": (
                "This is a permissive minimum for the reported workflow scope "
                "and sampling floors. It does not establish adequate "
                "sampling, convergence, equilibration, or biological validity; "
                "the scientific question may require more time, memory, CPUs, "
                "frames, replicas, or enabled methods."
            ),
        },
    }


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
    maximum_cpus_per_node: Optional[int] = None,
    maximum_memory_gib_per_node: Optional[float] = None,
    maximum_nodes: Optional[int] = None,
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
    node_limits_supplied = (
        maximum_cpus_per_node is not None
        or maximum_memory_gib_per_node is not None
        or maximum_nodes is not None
    )
    if node_limits_supplied and (
        maximum_cpus_per_node is None
        or maximum_memory_gib_per_node is None
    ):
        raise ResourcePlanningError(
            "node-aware planning requires both maximum_cpus_per_node and "
            "maximum_memory_gib_per_node"
        )
    if maximum_cpus_per_node is not None and (
        isinstance(maximum_cpus_per_node, bool)
        or not isinstance(maximum_cpus_per_node, int)
        or maximum_cpus_per_node <= 0
    ):
        raise ResourcePlanningError(
            "maximum_cpus_per_node must be a positive integer"
        )
    node_memory_gib = (
        None
        if maximum_memory_gib_per_node is None
        else _positive_number(
            maximum_memory_gib_per_node, "maximum_memory_gib_per_node"
        )
    )
    if maximum_nodes is not None and (
        isinstance(maximum_nodes, bool)
        or not isinstance(maximum_nodes, int)
        or maximum_nodes <= 0
    ):
        raise ResourcePlanningError("maximum_nodes must be a positive integer")
    if maximum_cpus_per_node is not None and maximum_nodes is None:
        assert node_memory_gib is not None
        maximum_nodes = max(
            math.ceil(maximum_parallel_cpus / maximum_cpus_per_node),
            math.ceil(memory_gib / node_memory_gib),
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
        unconstrained_cap = int(cap)
        if maximum_cpus_per_node is not None:
            cap = min(unconstrained_cap, maximum_cpus_per_node)
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
        parallel_execution_model = raw.get("parallel_execution_model")
        parallel_worker_count: Optional[int] = None
        per_parallel_worker_memory: Optional[float] = None
        reducer_memory: Optional[float] = None
        if parallel_execution_model is not None:
            if (
                not isinstance(parallel_execution_model, str)
                or not parallel_execution_model
            ):
                raise ResourcePlanningError(
                    f"task {task_id} parallel_execution_model must be a "
                    "nonempty string"
                )
            raw_worker_count = raw.get("parallel_worker_count")
            if (
                isinstance(raw_worker_count, bool)
                or not isinstance(raw_worker_count, int)
                or raw_worker_count <= 0
            ):
                raise ResourcePlanningError(
                    f"task {task_id} parallel_worker_count must be a positive "
                    "integer"
                )
            parallel_worker_count = int(raw_worker_count)
            per_parallel_worker_memory = _positive_number(
                raw.get("estimated_peak_memory_gib_per_parallel_worker"),
                f"task {task_id} "
                "estimated_peak_memory_gib_per_parallel_worker",
            )
            reducer_memory = _positive_number(
                raw.get("reducer_memory_gib", per_parallel_worker_memory),
                f"task {task_id} reducer_memory_gib",
            )
            cap = min(int(cap), parallel_worker_count)
            if node_memory_gib is not None:
                unpadded_node_budget = max(
                    0.0,
                    (node_memory_gib - memory_overhead) / memory_factor,
                )
                memory_limited_workers = math.floor(
                    unpadded_node_budget / per_parallel_worker_memory
                )
                if reducer_memory <= unpadded_node_budget:
                    cap = min(int(cap), max(1, memory_limited_workers))
                else:
                    cap = 1
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
            "node_unconstrained_effective_cpu_cap": unconstrained_cap,
            "source_frames_per_replica": list(source_counts),
            "minimum_frames_per_replica": int(minimum),
            "maximum_frames_per_replica": int(maximum),
            "cpu_seconds_per_physical_frame": rate,
            "fixed_cpu_hours": fixed,
            "power_law_cost_model": power_model,
            "measured_memory_cost_model": measured_memory_model,
            "estimated_peak_memory_gib": task_memory,
            "parallel_execution_model": parallel_execution_model,
            "parallel_worker_count": parallel_worker_count,
            "estimated_peak_memory_gib_per_parallel_worker": (
                per_parallel_worker_memory
            ),
            "reducer_memory_gib": reducer_memory,
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
            embedded_contract = row.get("scientific_sampling_requirements")
            profile = (
                profile_from_contract(embedded_contract)
                if isinstance(embedded_contract, Mapping) else None
            )
            if profile is not None and profile.minimum_frames_per_replica > 0:
                system_ids = row.get("system_ids_per_replica")
                intervals = row.get("frame_intervals_ns_per_replica")
                spans = row.get("source_time_spans_ns_per_replica")
                assessment = assess_raw_sampling(
                    profile,
                    selected_frames_per_replica=retained,
                    source_frames_per_replica=source,
                    system_ids_per_replica=(
                        [str(value) for value in system_ids]
                        if isinstance(system_ids, list) else None
                    ),
                    integer_stride=stride,
                    frame_intervals_ns_per_replica=(
                        [float(value) for value in intervals]
                        if isinstance(intervals, list) else None
                    ),
                    source_time_spans_ns_per_replica=(
                        [float(value) for value in spans]
                        if isinstance(spans, list) else None
                    ),
                )
                return bool(assessment["keep_enabled"])
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

        # Refine the common integer stride until every registered scientific
        # method meets both its per-replica and per-system contract. Project-
        # local preprocessing tasks without a registered profile retain their
        # declared pooled or replica-resolved technical minimum.
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
        if row.get("parallel_execution_model") is not None:
            worker_count = min(
                int(row["parallel_worker_count"]),
                int(row["effective_cpu_cap"]),
                maximum_parallel_cpus,
            )
            return max(
                float(row["reducer_memory_gib"]),
                float(row["estimated_peak_memory_gib_per_parallel_worker"])
                * worker_count,
            )
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

    dependency_stages = sorted({
        int(row["dependency_stage"]) for row in normalized
    })

    def schedule_stage(
        stage: int,
        selection: Mapping[str, Sequence[int]],
        costs: Mapping[str, float],
    ) -> Dict[str, object]:
        rows = [
            row for row in normalized
            if int(row["dependency_stage"]) == stage
        ]
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
        resource_lanes = pack_resource_lanes(
            bundle_resources,
            maximum_parallel_cpus=maximum_parallel_cpus,
            maximum_parallel_memory_gib=memory_gib,
            maximum_cpus_per_node=maximum_cpus_per_node,
            maximum_memory_gib_per_node=node_memory_gib,
            maximum_nodes=maximum_nodes,
        )
        longest = max(bundle_walls.values())
        lower_bound = max(
            stage_cpu_hours / maximum_parallel_cpus,
            longest,
        )
        packed_wall = max(
            (float(lane["wall_hours"]) for lane in resource_lanes),
            default=0.0,
        )
        planned_node_count = (
            0
            if maximum_cpus_per_node is None else
            len({
                int(lane["planned_node_index"])
                for lane in resource_lanes
            })
        )
        useful = sum(
            max(int(row["effective_cpu_cap"]) for row in bundle_rows)
            for bundle_rows in bundles.values()
        )
        return {
            "dependency_stage": stage,
            "task_count": len(rows),
            "execution_bundle_count": len(bundles),
            "estimated_cpu_hours": stage_cpu_hours,
            "maximum_useful_parallel_cpus": useful,
            "planned_parallel_cpus": min(maximum_parallel_cpus, useful),
            "planned_node_count": planned_node_count,
            "estimated_wall_hours_lower_bound": lower_bound,
            "estimated_wall_hours_with_resource_lanes": packed_wall,
            # Compatibility fields retain the old name while carrying the new
            # non-convoy lane schedule.  New consumers should use
            # resource_lanes and estimated_wall_hours_with_resource_lanes.
            "estimated_wall_hours_with_resource_waves": packed_wall,
            "resource_lanes": resource_lanes,
            "resource_waves": resource_lanes,
            "task_ids": [str(row["task_id"]) for row in rows],
            "execution_bundle_wall_hours": bundle_walls,
        }

    def schedule_summary(
        selection: Mapping[str, Sequence[int]],
    ) -> tuple[Optional[float], list[Dict[str, object]]]:
        costs = known_costs(selection)
        if len(costs) != len(normalized):
            return None, []
        stages = []
        total_wall = 0.0
        try:
            for stage in dependency_stages:
                stage_report = schedule_stage(stage, selection, costs)
                stages.append(stage_report)
                total_wall += float(
                    stage_report["estimated_wall_hours_with_resource_lanes"]
                )
        except ResourcePlanningError:
            return None, []
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
    single_task_memory_limit_gib = min(
        memory_gib,
        node_memory_gib if node_memory_gib is not None else memory_gib,
    )
    oversized_memory_rows = [
        {
            **row,
            "configured_memory_gib": memory_gib,
            "per_node_memory_gib": node_memory_gib,
            "single_task_memory_limit_gib": single_task_memory_limit_gib,
            "shortfall_gib": (
                float(row["required_memory_gib"])
                - single_task_memory_limit_gib
            ),
        }
        for row in minimum_memory_rows
        if (
            float(row["required_memory_gib"])
            > single_task_memory_limit_gib + 1.0e-12
        )
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

        current_costs = known_costs(selected)
        current_total = sum(current_costs.values())
        current_wall, current_stages = schedule_summary(selected)
        current_stage_reports = {
            int(row["dependency_stage"]): row for row in current_stages
        }
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
            proposed_costs = dict(current_costs)
            for row in rows:
                task_id = str(row["task_id"])
                value = task_cost(row, proposed[task_id])
                if value is None:
                    proposed_costs.pop(task_id, None)
                else:
                    proposed_costs[task_id] = value
            proposed_total = sum(proposed_costs.values())
            delta = proposed_total - current_total
            proposed_wall = current_wall
            if proposed_wall is not None and len(proposed_costs) == len(normalized):
                try:
                    for stage in sorted({
                        int(row["dependency_stage"]) for row in rows
                    }):
                        current_stage = current_stage_reports.get(stage)
                        if current_stage is None:
                            raise ResourcePlanningError(
                                f"dependency stage {stage} is unavailable"
                            )
                        proposed_stage = schedule_stage(
                            stage, proposed, proposed_costs
                        )
                        proposed_wall += (
                            float(proposed_stage[
                                "estimated_wall_hours_with_resource_waves"
                            ])
                            - float(current_stage[
                                "estimated_wall_hours_with_resource_waves"
                            ])
                        )
                except ResourcePlanningError:
                    proposed_wall = None
            proposed_memory_fits = all(
                task_scheduler_memory(row, proposed[str(row["task_id"])])
                <= memory_gib + 1.0e-12
                for row in rows
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
            **({
                "projection_selected_physical_frames_per_replica": counts,
                "projection_selected_physical_frame_count": physical_count,
                "projection_member_observation_count": (
                    physical_count
                    * int(row["member_observation_multiplier"])
                ),
            } if row.get("module_id") == "common_pca" else {}),
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
            "active_parallel_workers_at_selected_observations": (
                min(
                    int(row["parallel_worker_count"]),
                    int(row["effective_cpu_cap"]),
                    maximum_parallel_cpus,
                )
                if row.get("parallel_execution_model") is not None else 1
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

    scientific_below_standard = []
    scientific_postrun_diagnostics = []
    for report in task_reports:
        module_id = str(report.get("module_id", ""))
        try:
            embedded_contract = report.get("scientific_sampling_requirements")
            profile = (
                profile_from_contract(embedded_contract)
                if isinstance(embedded_contract, Mapping)
                else scientific_sampling_profile(module_id)
            )
        except ScientificSamplingError:
            # Preprocessing and project-local extension tasks are not analysis
            # estimators.  Their own explicit task contract remains authoritative.
            continue
        if profile.minimum_frames_per_replica == 0:
            continue
        system_ids = report.get("system_ids_per_replica")
        if not isinstance(system_ids, list):
            system_ids = None
        assessment = assess_raw_sampling(
            profile,
            selected_frames_per_replica=report[
                "selected_physical_frames_per_replica"
            ],
            source_frames_per_replica=report["source_frames_per_replica"],
            system_ids_per_replica=system_ids,
            integer_stride=int(report["integer_stride"]),
            frame_intervals_ns_per_replica=(
                [
                    float(value) for value in report[
                        "frame_intervals_ns_per_replica"
                    ]
                ]
                if isinstance(
                    report.get("frame_intervals_ns_per_replica"), list
                ) else None
            ),
            source_time_spans_ns_per_replica=(
                [
                    float(value) for value in report[
                        "source_time_spans_ns_per_replica"
                    ]
                ]
                if isinstance(
                    report.get("source_time_spans_ns_per_replica"), list
                ) else None
            ),
        )
        if isinstance(embedded_contract, Mapping):
            assessment["requirements"] = dict(embedded_contract)
            assessment["policy_id"] = str(
                embedded_contract.get("policy_id", POLICY_ID)
            )
        report["scientific_sampling_assessment"] = assessment
        if (
            isinstance(embedded_contract, Mapping)
            and not bool(assessment["keep_enabled"])
        ):
            scientific_below_standard.append({
                "task_id": str(report["task_id"]),
                "module_id": module_id,
                "configuration_switch": memory_configuration_switch(report),
                "raw_coverage_status": assessment["raw_coverage_status"],
                "required_frames_per_replica": assessment.get(
                    "required_frames_per_replica"
                ),
                "selected_physical_frames_per_replica": assessment.get(
                    "selected_physical_frames_per_replica",
                    report["selected_physical_frames_per_replica"],
                ),
            })
        if (
            assessment.get("postrun_event_or_transition_diagnostic") is not None
            or assessment.get("temporal_resolution_validation_required")
        ):
            scientific_postrun_diagnostics.append({
                "task_id": str(report["task_id"]),
                "module_id": module_id,
                "minimum_reported_events_or_transitions": assessment.get(
                    "postrun_event_or_transition_diagnostic"
                ),
                "temporal_resolution_validation_required": assessment.get(
                    "temporal_resolution_validation_required"
                ),
                "planner_estimates_autocorrelation_or_event_rates": False,
            })

    if scientific_below_standard:
        infeasibility_reasons.append(
            "enabled tasks fall below their scientific per-replica, per-system, "
            "or ordered-time sampling floors: "
            + ", ".join(
                str(row["task_id"]) for row in scientific_below_standard
            )
        )
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
    useful_parallel_cpu_ceiling = workflow_useful_parallel_cpu_ceiling(
        normalized, maximum_cpus_per_node=maximum_cpus_per_node
    )
    effective_parallel_cpu_cap = min(
        maximum_parallel_cpus, useful_parallel_cpu_ceiling
    )
    resource_warnings = []
    if maximum_parallel_cpus > useful_parallel_cpu_ceiling:
        cpu_label = "CPU" if effective_parallel_cpu_cap == 1 else "CPUs"
        resource_warnings.append({
            "severity": "warning",
            "code": "REQUESTED_CPUS_EXCEED_USEFUL_PARALLELISM",
            "requested_parallel_cpus": maximum_parallel_cpus,
            "useful_parallel_cpu_ceiling": useful_parallel_cpu_ceiling,
            "effective_parallel_cpu_cap": effective_parallel_cpu_cap,
            "excess_parallel_cpus": (
                maximum_parallel_cpus - useful_parallel_cpu_ceiling
            ),
            "message": (
                f"You requested {maximum_parallel_cpus} concurrent CPUs, but "
                f"the resolved workflow can use at most "
                f"{useful_parallel_cpu_ceiling}. The execution cap and generated "
                f"Slurm submission will be changed to "
                f"{effective_parallel_cpu_cap} {cpu_label}."
            ),
        })
    permissive_minimum_request = _permissive_minimum_resource_request(
        request_scope="all_currently_enabled_tasks_at_configured_sampling_floors",
        minimum_cpu_hours=minimum_known_cpu_hours,
        minimum_wall_hours=minimum_wall,
        minimum_stages=minimum_stages,
        maximum_parallel_cpus=maximum_parallel_cpus,
        maximum_wall_hours=wall_hours,
        maximum_memory_gib=memory_gib,
        planning_utilization=utilization,
        pilot_budget_fraction=pilot_fraction,
        finalization_headroom_fraction=finalization_fraction,
        memory_safety_factor=memory_factor,
        memory_overhead_gib=memory_overhead,
        minimum_scheduler_memory_gib=minimum_scheduler_memory,
        minimum_single_task_memory_gib=minimum_required_memory_gib,
        maximum_cpus_per_node=maximum_cpus_per_node,
        maximum_memory_gib_per_node=node_memory_gib,
        maximum_nodes=maximum_nodes,
    )
    return {
        "planning_schema": "salsbury-campaign-resource-plan-v1",
        "technical_status": "complete",
        "scientific_status": "planning only",
        "feasibility_status": feasibility,
        "execution_authorized": feasibility == "feasible",
        "maximum_parallel_cpus_input": maximum_parallel_cpus,
        "effective_parallel_cpu_cap": effective_parallel_cpu_cap,
        "maximum_wall_hours_input": wall_hours,
        "maximum_memory_gib_input": memory_gib,
        "maximum_parallel_memory_gib_input": memory_gib,
        "node_resource_policy": {
            "enabled": maximum_cpus_per_node is not None,
            "maximum_cpus_per_node": maximum_cpus_per_node,
            "maximum_memory_gib_per_node": node_memory_gib,
            "maximum_nodes": maximum_nodes,
            "memory_basis": (
                "safety_adjusted_scheduler_request"
                if maximum_cpus_per_node is not None else None
            ),
            "single_node_per_task": maximum_cpus_per_node is not None,
        },
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
        "warning_count": len(resource_warnings),
        "resource_warnings": resource_warnings,
        "permissive_minimum_resource_request": permissive_minimum_request,
        "workflow_parallel_capacity": {
            "requested_parallel_cpu_cap": maximum_parallel_cpus,
            "useful_parallel_cpu_ceiling": useful_parallel_cpu_ceiling,
            "effective_parallel_cpu_cap": effective_parallel_cpu_cap,
            "coordinate_cache_replica_parallel_cpu_ceiling": max(
                (
                    int(row.get("intrinsic_cpu_cap", row.get("effective_cpu_cap", 1)))
                    for row in normalized
                    if row.get("module_id") == "coordinate_cache"
                ),
                default=0,
            ),
            "interpretation": (
                "The useful ceiling is the largest sum of independent execution "
                "bundle CPU caps in one dependency stage. Aggregate memory, the "
                "configured CPU envelope, scheduler policy, and queue state may "
                "reduce the practical request."
            ),
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
        "scientific_sampling_feasibility": {
            "policy_id": POLICY_ID,
            "raw_coverage_status": (
                "all_enabled_methods_meet_or_exhaust_available_source"
                if not scientific_below_standard else
                "one_or_more_enabled_methods_below_standard"
            ),
            "below_standard_tasks": scientific_below_standard,
            "configuration_switches_to_disable_or_replan": sorted({
                str(row["configuration_switch"])
                for row in scientific_below_standard
            }),
            "postrun_diagnostics": scientific_postrun_diagnostics,
            "source_limited_policy": (
                "A complete short source remains eligible for analysis. Source "
                "duration and selected span are reported as provenance; only "
                "unmet sample-count or applicable ordered-method resolution "
                "requirements place a task below the planning floor."
            ),
        },
        "memory_feasibility": {
            "configured_memory_gib": memory_gib,
            "minimum_required_memory_gib": minimum_required_memory_gib,
            "single_task_memory_limit_gib": single_task_memory_limit_gib,
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
            "receive declared standard scientific raw-frame coverage before additional "
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


def plan_projection_coupled_campaign_resource_budget(
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
    maximum_cpus_per_node: Optional[int] = None,
    maximum_memory_gib_per_node: Optional[float] = None,
    maximum_nodes: Optional[int] = None,
    maximum_coupling_iterations: int = 12,
) -> Dict[str, object]:
    """Replan PCA projections and their clustering fits to one fixed point.

    A conformational-view clustering fit consumes the observations selected by
    its view's ``common_pca`` projection task.  Replanning a saved campaign must
    therefore replace every fit task's source stream whenever that parent
    selection changes.  The ordinary budget allocator is rerun until the parent
    selection and every child source agree exactly; a missing parent or failure
    to converge is an error rather than permission to use stale observations.
    """

    if (
        isinstance(maximum_coupling_iterations, bool)
        or not isinstance(maximum_coupling_iterations, int)
        or maximum_coupling_iterations <= 0
    ):
        raise ResourcePlanningError(
            "maximum_coupling_iterations must be a positive integer"
        )
    working = []
    for raw in tasks:
        row = dict(raw)
        if (
            row.get("task_scope") == "conformational_view"
            and row.get("module_id") == "common_pca"
            and "projection_declared_maximum_frames_per_replica" in row
        ):
            row["maximum_frames_per_replica"] = int(
                row["projection_declared_maximum_frames_per_replica"]
            )
            row.pop("dynamic_coupling_ceiling_per_replica", None)
        working.append(row)
    parent_ids: Dict[str, str] = {}
    child_workflows = set()
    for row in working:
        workflow_id = row.get("workflow_id")
        if row.get("task_scope") == "conformational_view_algorithm_fit":
            if not isinstance(workflow_id, str) or not workflow_id:
                raise ResourcePlanningError(
                    f"clustering task {row.get('task_id')} has no workflow_id"
                )
            child_workflows.add(workflow_id)
        if (
            row.get("task_scope") == "conformational_view"
            and row.get("module_id") == "common_pca"
        ):
            if not isinstance(workflow_id, str) or not workflow_id:
                raise ResourcePlanningError(
                    f"common-PCA task {row.get('task_id')} has no workflow_id"
                )
            if workflow_id in parent_ids:
                raise ResourcePlanningError(
                    f"workflow {workflow_id} has multiple common-PCA projection tasks"
                )
            parent_ids[workflow_id] = str(row.get("task_id"))
    for workflow_id in sorted(child_workflows):
        if workflow_id not in parent_ids:
            raise ResourcePlanningError(
                f"clustering workflow {workflow_id} has no common-PCA projection task"
            )

    history = []
    updated_task_ids = set()
    cycle_resolutions = []
    phase_signatures: Dict[object, int] = {}
    phase_states: list[Dict[str, list[int]]] = []
    for iteration in range(1, maximum_coupling_iterations + 1):
        plan = plan_campaign_resource_budget(
            working,
            maximum_parallel_cpus=maximum_parallel_cpus,
            maximum_wall_hours=maximum_wall_hours,
            maximum_memory_gib=maximum_memory_gib,
            planning_utilization=planning_utilization,
            pilot_budget_fraction=pilot_budget_fraction,
            finalization_headroom_fraction=finalization_headroom_fraction,
            memory_safety_factor=memory_safety_factor,
            memory_overhead_gib=memory_overhead_gib,
            minimum_scheduler_memory_gib=minimum_scheduler_memory_gib,
            maximum_cpus_per_node=maximum_cpus_per_node,
            maximum_memory_gib_per_node=maximum_memory_gib_per_node,
            maximum_nodes=maximum_nodes,
        )
        planned_rows = {
            str(row["task_id"]): row
            for row in plan["tasks"]  # type: ignore[union-attr]
            if isinstance(row, Mapping)
        }
        parent_counts = {
            workflow_id: [
                int(value) for value in planned_rows[parent_id][
                    "selected_physical_frames_per_replica"
                ]
            ]
            for workflow_id, parent_id in parent_ids.items()
        }
        parent_effective_intervals = {}
        parent_source_spans = {}
        for workflow_id, parent_id in parent_ids.items():
            parent_row = planned_rows[parent_id]
            raw_intervals = parent_row.get("frame_intervals_ns_per_replica")
            raw_spans = parent_row.get("source_time_spans_ns_per_replica")
            if isinstance(raw_intervals, list):
                parent_effective_intervals[workflow_id] = [
                    float(value) * int(parent_row["integer_stride"])
                    for value in raw_intervals
                ]
            if isinstance(raw_spans, list):
                parent_source_spans[workflow_id] = [
                    float(value) for value in raw_spans
                ]
        signature = tuple(
            (workflow_id, tuple(values))
            for workflow_id, values in sorted(parent_counts.items())
        )
        input_matches_projection = all(
            [int(value) for value in row["source_frames_per_replica"]]
            == parent_counts[str(row["workflow_id"])]
            and (
                str(row["workflow_id"]) not in parent_effective_intervals
                or [
                    float(value) for value in row.get(
                        "frame_intervals_ns_per_replica", []
                    )
                ] == parent_effective_intervals[str(row["workflow_id"])]
            )
            for row in working
            if row.get("task_scope") == "conformational_view_algorithm_fit"
        )
        stabilized_sources = parent_counts
        stabilized_projection_ceilings: Dict[str, list[int]] = {}
        if not input_matches_projection and signature in phase_signatures:
            cycle_start = phase_signatures[signature]
            cycle_states = phase_states[cycle_start:] + [parent_counts]
            stabilized_sources = {
                workflow_id: [
                    min(state[workflow_id][index] for state in cycle_states)
                    for index in range(len(parent_counts[workflow_id]))
                ]
                for workflow_id in parent_counts
            }
            stabilized_projection_ceilings = {
                workflow_id: list(values)
                for workflow_id, values in stabilized_sources.items()
                if values != parent_counts[workflow_id]
                or any(
                    state[workflow_id] != values for state in cycle_states
                )
            }
            cycle_resolutions.append({
                "detected_at_iteration": iteration,
                "cycle_start_iteration": cycle_start + 1,
                "cycle_length": len(cycle_states) - 1,
                "selected_componentwise_minimum_projection_counts": {
                    workflow_id: list(values)
                    for workflow_id, values in sorted(
                        stabilized_projection_ceilings.items()
                    )
                },
                "resolution": (
                    "dynamically cap each oscillating projection at the "
                    "componentwise minimum affordable cycle state, then replan "
                    "all downstream clustering fits"
                ),
            })
            phase_signatures = {}
            phase_states = []
        elif not input_matches_projection:
            phase_signatures[signature] = len(phase_states)
            phase_states.append({
                workflow_id: list(values)
                for workflow_id, values in parent_counts.items()
            })

        changed = []
        parent_caps_changed = []
        next_working = []
        for raw in working:
            row = dict(raw)
            if (
                row.get("task_scope") == "conformational_view"
                and row.get("module_id") == "common_pca"
                and str(row.get("workflow_id"))
                in stabilized_projection_ceilings
            ):
                workflow_id = str(row["workflow_id"])
                selected_ceiling = max(stabilized_sources[workflow_id])
                declared_maximum = int(row.get(
                    "projection_declared_maximum_frames_per_replica",
                    row["maximum_frames_per_replica"],
                ))
                new_maximum = min(declared_maximum, selected_ceiling)
                if int(row["maximum_frames_per_replica"]) != new_maximum:
                    parent_caps_changed.append(str(row["task_id"]))
                row.update({
                    "maximum_frames_per_replica": new_maximum,
                    "projection_declared_maximum_frames_per_replica": (
                        declared_maximum
                    ),
                    "dynamic_coupling_ceiling_per_replica": selected_ceiling,
                })
            if row.get("task_scope") == "conformational_view_algorithm_fit":
                workflow_id = str(row["workflow_id"])
                selected_counts = stabilized_sources[workflow_id]
                current_counts = [
                    int(value) for value in row["source_frames_per_replica"]
                ]
                if current_counts != selected_counts:
                    changed.append(str(row["task_id"]))
                    updated_task_ids.add(str(row["task_id"]))
                declared_minimum = int(row.get(
                    "projection_declared_minimum_frames_per_replica",
                    row["minimum_frames_per_replica"],
                ))
                maximum = max(selected_counts)
                row.update({
                    "source_frames_per_replica": list(selected_counts),
                    "minimum_frames_per_replica": min(
                        declared_minimum, maximum
                    ),
                    "maximum_frames_per_replica": maximum,
                    "projection_declared_minimum_frames_per_replica": (
                        declared_minimum
                    ),
                    "projection_source_task_id": parent_ids[workflow_id],
                    "projection_source_counts_iteration_input": list(
                        selected_counts
                    ),
                    "projection_source_limited_below_declared_minimum": (
                        maximum < declared_minimum
                    ),
                })
                if workflow_id in parent_effective_intervals:
                    row["frame_intervals_ns_per_replica"] = list(
                        parent_effective_intervals[workflow_id]
                    )
                if workflow_id in parent_source_spans:
                    row["source_time_spans_ns_per_replica"] = list(
                        parent_source_spans[workflow_id]
                    )
            next_working.append(row)
        history.append({
            "iteration": iteration,
            "planned_projection_sources": {
                workflow_id: list(values)
                for workflow_id, values in sorted(parent_counts.items())
            },
            "stabilized_projection_ceilings": {
                workflow_id: list(values)
                for workflow_id, values in sorted(
                    stabilized_projection_ceilings.items()
                )
            },
            "clustering_fit_tasks_rebuilt": sorted(changed),
            "projection_tasks_dynamically_capped": sorted(
                parent_caps_changed
            ),
        })
        if not changed and not parent_caps_changed:
            for row in plan["tasks"]:  # type: ignore[union-attr]
                if (
                    isinstance(row, Mapping)
                    and row.get("task_scope")
                    == "conformational_view_algorithm_fit"
                ):
                    workflow_id = str(row["workflow_id"])
                    if list(row["source_frames_per_replica"]) != parent_counts[
                        workflow_id
                    ]:
                        raise ResourcePlanningError(
                            "projection/clustering coupling verification failed for "
                            f"{row['task_id']}"
                        )
            plan["projection_clustering_coupling"] = {
                "coupling_schema": (
                    "salsbury-projection-clustering-coupling-v1"
                ),
                "converged": True,
                "iterations": iteration,
                "projection_workflow_count": len(parent_ids),
                "clustering_fit_task_count": sum(
                    1
                    for row in plan["tasks"]  # type: ignore[union-attr]
                    if isinstance(row, Mapping)
                    and row.get("task_scope")
                    == "conformational_view_algorithm_fit"
                ),
                "rebuilt_clustering_fit_task_count": len(updated_task_ids),
                "dynamic_cycle_resolution_count": len(cycle_resolutions),
                "cycle_resolutions": cycle_resolutions,
                "iteration_history": history,
                "contract": (
                    "Every clustering-fit source count equals the final selected "
                    "physical-frame count of its workflow's common-PCA projection."
                ),
            }
            return plan
        working = next_working
    raise ResourcePlanningError(
        "projection/clustering campaign replanning did not converge within "
        f"{maximum_coupling_iterations} iterations"
    )


def _overall_trajectory_stride_candidates(maximum_stride: int) -> list[int]:
    """Return the standard interpretable overall-stride pilot grid."""

    if maximum_stride <= 0:
        raise ResourcePlanningError("maximum overall trajectory stride must be positive")
    return [
        stride for stride in (1, 2, 3, 4, 5, 10, 20, 100)
        if stride <= maximum_stride
    ]


def _overall_trajectory_maximum_stride(
    source_counts: Sequence[int], minimum_frames_per_replica: int,
) -> int:
    """Return the coarsest stride retaining the declared minimum everywhere."""

    if minimum_frames_per_replica <= 0:
        raise ResourcePlanningError(
            "minimum retained frames per replica must be positive"
        )
    if any(count <= 0 for count in source_counts):
        raise ResourcePlanningError(
            "overall-stride source frame counts must be positive"
        )
    if min(source_counts) < minimum_frames_per_replica:
        return 1
    # ceil(N / stride) >= minimum.  The closed form avoids a long loop for
    # million-frame local or workstation trajectories.
    if minimum_frames_per_replica == 1:
        return max(source_counts)
    return max(
        1,
        min(
            (count - 1) // (minimum_frames_per_replica - 1)
            for count in source_counts
        ),
    )


def _global_plan_information_metrics(
    plan: Mapping[str, object],
) -> Dict[str, float]:
    """Score balanced normalized coverage across distinct analyses.

    Raw observation totals would let a large pooled view dominate the overall
    stride choice.  Instead, each task contributes the square root of its
    normalized coverage.  The concave utility rewards additional coverage while
    giving more value to preserving breadth across different analyses.  Declared
    priority weights still apply, and technical minima remain hard feasibility
    gates in the underlying campaign planner.
    """

    rows = plan.get("tasks")
    if not isinstance(rows, list):
        raise ResourcePlanningError("campaign plan has no task rows")
    weighted_utility = 0.0
    weight_total = 0.0
    raw_selected_observations = 0.0
    normalized_coverages: list[float] = []
    parent_raw_counts = {
        str(raw["workflow_id"]): list(
            raw["global_stride_raw_source_frames_per_replica"]
        )
        for raw in rows
        if isinstance(raw, Mapping)
        and raw.get("task_scope") == "conformational_view"
        and raw.get("module_id") == "common_pca"
        and isinstance(raw.get("workflow_id"), str)
        and isinstance(
            raw.get("global_stride_raw_source_frames_per_replica"), list
        )
    }
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("module_id") == "coordinate_cache":
            continue
        selected = raw.get("selected_physical_frame_count")
        weight = raw.get("priority_weight", 1.0)
        multiplier = raw.get("member_observation_multiplier", 1)
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or isinstance(multiplier, bool)
            or not isinstance(multiplier, int)
        ):
            raise ResourcePlanningError(
                f"task {raw.get('task_id')} cannot be scored for global-stride planning"
            )
        raw_counts = (
            parent_raw_counts.get(str(raw.get("workflow_id")))
            if raw.get("task_scope") == "conformational_view_algorithm_fit"
            else raw.get("global_stride_raw_source_frames_per_replica")
        )
        if not isinstance(raw_counts, list) or not raw_counts:
            raw_counts = raw.get("source_frames_per_replica")
        if (
            not isinstance(raw_counts, list)
            or not raw_counts
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in raw_counts
            )
        ):
            raise ResourcePlanningError(
                f"task {raw.get('task_id')} lacks a valid information denominator"
            )
        denominator = float(sum(raw_counts) * multiplier)
        selected_observations = float(selected * multiplier)
        coverage = min(1.0, selected_observations / denominator)
        resolved_weight = float(weight)
        weighted_utility += resolved_weight * math.sqrt(coverage)
        weight_total += resolved_weight
        raw_selected_observations += selected_observations
        normalized_coverages.append(coverage)
    if weight_total <= 0.0 or not normalized_coverages:
        raise ResourcePlanningError("campaign has no downstream analyses to score")
    return {
        "balanced_information_utility": weighted_utility / weight_total,
        "minimum_normalized_analysis_coverage": min(normalized_coverages),
        "mean_normalized_analysis_coverage": (
            sum(normalized_coverages) / len(normalized_coverages)
        ),
        "selected_observation_count": raw_selected_observations,
        "scored_analysis_count": float(len(normalized_coverages)),
    }


def _global_stride_information_upper_bound(
    tasks: Sequence[Mapping[str, object]], stride: int
) -> float:
    """Bound the best normalized information possible after an overall stride."""

    parent_counts = {
        str(row.get("workflow_id")): list(
            row.get(
                "global_stride_raw_source_frames_per_replica",
                row.get("source_frames_per_replica", []),
            )
        )
        for row in tasks
        if row.get("task_scope") == "conformational_view"
        and row.get("module_id") == "common_pca"
    }
    weighted = 0.0
    total_weight = 0.0
    for row in tasks:
        if row.get("module_id") == "coordinate_cache":
            continue
        raw = (
            parent_counts.get(str(row.get("workflow_id")))
            if row.get("task_scope") == "conformational_view_algorithm_fit"
            else row.get(
                "global_stride_raw_source_frames_per_replica",
                row.get("source_frames_per_replica"),
            )
        )
        if not isinstance(raw, (list, tuple)) or not raw:
            coverage = 1.0
        else:
            counts = [int(value) for value in raw]
            coverage = min(
                1.0,
                sum(
                    integer_stride_selected_count(value, stride)
                    for value in counts
                ) / sum(counts),
            )
        weight = float(row.get("priority_weight", 1.0))
        weighted += weight * math.sqrt(coverage)
        total_weight += weight
    return 1.0 if total_weight <= 0.0 else weighted / total_weight


def plan_global_stride_projection_coupled_campaign_resource_budget(
    tasks: Sequence[Mapping[str, object]],
    *,
    maximum_parallel_cpus: int,
    maximum_wall_hours: float,
    maximum_memory_gib: float,
    coordinate_cache_minimum_frames_per_replica: int = 50,
    coordinate_cache_full_scan_fraction: float = 1.0,
    overall_stride_candidate_strides: Optional[Sequence[int]] = None,
    coordinate_cache_candidate_strides: Optional[Sequence[int]] = None,
    planning_utilization: float = 0.85,
    pilot_budget_fraction: float = 0.05,
    finalization_headroom_fraction: float = 0.0,
    memory_safety_factor: float = 1.0,
    memory_overhead_gib: float = 0.0,
    minimum_scheduler_memory_gib: float = 0.0,
    maximum_cpus_per_node: Optional[int] = None,
    maximum_memory_gib_per_node: Optional[float] = None,
    maximum_nodes: Optional[int] = None,
    maximum_coupling_iterations: int = 12,
    protected_module_ids: Sequence[str] = (
        "coordinate_cache", "provenance_manifest", "preflight_inventory",
        "common_atom_mapping", "structural_integrity_qc", "replica_rmsd_rg",
        "pooled_rmsf", "individual_pca", "common_pca", "dccm",
        "pca_fes_basins", "representative_frames",
    ),
) -> Dict[str, object]:
    """Jointly choose an overall trajectory stride and downstream strides.

    For each deterministic integer overall stride, this planner rebuilds every
    raw-trajectory source stream, including solvated analyses, the molecular
    coordinate cache, and every conformational view.  It then replans each PCA
    projection and rebuilds the algorithm-specific clustering fits.  The
    selected candidate maximizes balanced normalized coverage across distinct
    analyses inside one CPU, wall-time, and aggregate-memory envelope; exact
    information ties retain the finer overall stride.

    ``coordinate_cache_full_scan_fraction`` separates work that must still scan
    every source frame from work avoided when only retained cache frames are
    materialized.  A value of one models the current stateful continuous-unwrapping
    implementation conservatively; zero is an optimistic direct-skip bound.  The
    parameter is explicit because the existing one-point cache calibration cannot
    identify this split.  ``overall_stride_candidate_strides`` can restrict a
    sensitivity study to a declared set.  The default grid includes stride one
    as the information-preserving reference; callers supplying an explicit grid
    control the complete evaluated set.  ``coordinate_cache_candidate_strides``
    is retained as a compatibility spelling for the first local prototype.
    """

    overall_planning_started = time.monotonic()
    scan_fraction = _fraction(
        coordinate_cache_full_scan_fraction,
        "coordinate_cache_full_scan_fraction",
        allow_zero=True,
    )
    if not tasks:
        raise ResourcePlanningError(
            "global-stride campaign planning requires at least one task"
        )
    cache_rows = [
        dict(row) for row in tasks if row.get("module_id") == "coordinate_cache"
    ]
    if len(cache_rows) != 1:
        raise ResourcePlanningError(
            "global-stride campaign planning currently requires exactly one "
            "coordinate-cache task"
        )
    cache_id = str(cache_rows[0].get("task_id"))
    cache_source = [
        int(value) for value in cache_rows[0]["source_frames_per_replica"]
    ]
    declared_minimum = coordinate_cache_minimum_frames_per_replica
    if (
        isinstance(declared_minimum, bool)
        or not isinstance(declared_minimum, int)
        or declared_minimum <= 0
    ):
        raise ResourcePlanningError(
            "coordinate_cache_minimum_frames_per_replica must be a positive integer"
        )

    protected_modules = {str(value) for value in protected_module_ids}
    protected_modules.add("coordinate_cache")
    base_tasks: list[Dict[str, object]] = []
    protected_replica_minima = [declared_minimum]
    for original in tasks:
        row = dict(original)
        if (
            row.get("task_scope") != "conformational_view_algorithm_fit"
            and str(row.get("task_id")) != cache_id
        ):
            raw_counts = row.get(
                "global_stride_raw_source_frames_per_replica",
                row.get("source_frames_per_replica"),
            )
            if not isinstance(raw_counts, (list, tuple)) or not raw_counts:
                raise ResourcePlanningError(
                    f"task {row.get('task_id')} lacks raw trajectory-source counts"
                )
            row["global_stride_raw_source_frames_per_replica"] = [
                int(value) for value in raw_counts
            ]
            row["global_stride_declared_minimum_frames_per_replica"] = int(
                row.get("minimum_frames_per_replica", 1)
            )
            row["global_stride_declared_maximum_frames_per_replica"] = int(
                row.get(
                    "maximum_frames_per_replica",
                    max(int(value) for value in raw_counts),
                )
            )
            if str(row.get("module_id")) in protected_modules:
                protected_replica_minima.append(int(
                    row["global_stride_declared_minimum_frames_per_replica"]
                ))
        base_tasks.append(row)
    declared_minimum = max(protected_replica_minima)
    maximum_stride = _overall_trajectory_maximum_stride(
        cache_source, declared_minimum
    )
    if (
        overall_stride_candidate_strides is not None
        and coordinate_cache_candidate_strides is not None
    ):
        raise ResourcePlanningError(
            "declare overall_stride_candidate_strides, not both candidate-stride "
            "spellings"
        )
    declared_candidates = (
        overall_stride_candidate_strides
        if overall_stride_candidate_strides is not None
        else coordinate_cache_candidate_strides
    )
    if declared_candidates is None:
        candidates = _overall_trajectory_stride_candidates(maximum_stride)
        requested_candidate_values = list(candidates)
        scientifically_pruned_candidates: list[int] = []
    else:
        requested_candidates: set[int] = set()
        for value in declared_candidates:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ResourcePlanningError(
                    "overall_stride_candidate_strides must contain only "
                    "positive integers"
                )
            requested_candidates.add(value)
        if not requested_candidates:
            raise ResourcePlanningError(
                "overall_stride_candidate_strides must not be empty"
            )
        requested_candidate_values = sorted(requested_candidates)
        candidates = [
            value for value in requested_candidate_values
            if value <= maximum_stride
        ]
        scientifically_pruned_candidates = [
            value for value in requested_candidate_values
            if value > maximum_stride
        ]
        if not candidates:
            raise ResourcePlanningError(
                "every requested overall trajectory candidate stride exceeds "
                "the maximum scientifically allowed stride "
                f"{maximum_stride}"
            )

    evaluated_candidates: list[int] = []
    early_terminated_candidates: list[int] = []

    evaluations: list[Dict[str, object]] = []
    feasible: list[
        tuple[tuple[float, float, float, int, float], Dict[str, object]]
    ] = []
    candidate_plans: list[Dict[str, object]] = []
    best_feasible_information = -math.inf
    for candidate_index, cache_stride in enumerate(candidates):
        evaluated_candidates.append(cache_stride)
        candidate_started = time.monotonic()
        candidate_tasks: list[Dict[str, object]] = []
        for original in base_tasks:
            row = dict(original)
            if str(row.get("task_id")) == cache_id:
                rate = row.get("coordinate_cache_full_rate_seconds_per_frame")
                if rate is None:
                    rate = row.get("cpu_seconds_per_physical_frame")
                if (
                    isinstance(rate, bool)
                    or not isinstance(rate, (int, float))
                    or float(rate) < 0.0
                ):
                    raise ResourcePlanningError(
                        "coordinate-cache task lacks a nonnegative frame-cost rate"
                    )
                original_fixed = float(
                    row.get("coordinate_cache_original_fixed_cpu_hours", row.get(
                        "fixed_cpu_hours", 0.0
                    ))
                )
                selected_counts = [
                    integer_stride_selected_count(count, cache_stride)
                    for count in cache_source
                ]
                selected_ceiling = max(selected_counts)
                row.update({
                    "task_scope": "planner_selected_coordinate_preprocessing",
                    "source_frames_per_replica": list(cache_source),
                    "minimum_frames_per_replica": selected_ceiling,
                    "maximum_frames_per_replica": selected_ceiling,
                    "cpu_seconds_per_physical_frame": (
                        float(rate) * (1.0 - scan_fraction)
                    ),
                    "fixed_cpu_hours": (
                        original_fixed
                        + float(rate) * sum(cache_source) * scan_fraction / 3600.0
                    ),
                    "coordinate_cache_full_rate_seconds_per_frame": float(rate),
                    "coordinate_cache_original_fixed_cpu_hours": original_fixed,
                    "coordinate_cache_full_scan_fraction": scan_fraction,
                    "coordinate_cache_candidate_stride": cache_stride,
                    "coordinate_cache_raw_source_frames_per_replica": list(
                        cache_source
                    ),
                    "coordinate_cache_candidate_selected_frames_per_replica": (
                        selected_counts
                    ),
                    "minimum_frame_role": (
                        "fixed candidate working-cache coverage"
                    ),
                    "maximum_frame_role": (
                        "fixed candidate working-cache coverage"
                    ),
                    "priority_weight": 1.0,
                })
            elif row.get("task_scope") != "conformational_view_algorithm_fit":
                raw_counts = [
                    int(value) for value in row[
                        "global_stride_raw_source_frames_per_replica"
                    ]
                ]
                overall_counts = [
                    integer_stride_selected_count(count, cache_stride)
                    for count in raw_counts
                ]
                declared_task_minimum = int(row.get(
                    "global_stride_declared_minimum_frames_per_replica",
                    row["minimum_frames_per_replica"],
                ))
                declared_task_maximum = int(row.get(
                    "global_stride_declared_maximum_frames_per_replica",
                    row["maximum_frames_per_replica"],
                ))
                candidate_maximum = min(
                    declared_task_maximum, max(overall_counts)
                )
                if max(raw_counts) < declared_task_minimum:
                    # The real supplied trajectory is already shorter than the
                    # standard floor. Preserve that source-limited declaration;
                    # do not confuse it with resource-induced subsampling.
                    candidate_maximum = max(
                        candidate_maximum, declared_task_minimum
                    )
                row.update({
                    "source_frames_per_replica": overall_counts,
                    "minimum_frames_per_replica": declared_task_minimum,
                    "maximum_frames_per_replica": candidate_maximum,
                    "overall_trajectory_integer_stride": cache_stride,
                })
                raw_intervals = row.get("frame_intervals_ns_per_replica")
                if isinstance(raw_intervals, list):
                    row["global_stride_raw_frame_intervals_ns_per_replica"] = [
                        float(value) for value in raw_intervals
                    ]
                    row["frame_intervals_ns_per_replica"] = [
                        float(value) * cache_stride for value in raw_intervals
                    ]
                if (
                    row.get("task_scope") == "conformational_view"
                    and row.get("module_id") == "common_pca"
                ):
                    row["projection_declared_maximum_frames_per_replica"] = (
                        candidate_maximum
                    )
                    row.pop("dynamic_coupling_ceiling_per_replica", None)
            candidate_tasks.append(row)
        invalid_protected_tasks: list[Dict[str, object]] = []
        invalid_optional_tasks: list[Dict[str, object]] = []
        for row in candidate_tasks:
            is_protected = str(row.get("module_id")) in protected_modules
            invalid_rows = (
                invalid_protected_tasks if is_protected
                else invalid_optional_tasks
            )
            task_id = str(row.get("task_id"))
            if task_id == cache_id:
                retained = row.get(
                    "coordinate_cache_candidate_selected_frames_per_replica", []
                )
                if not isinstance(retained, list) or any(
                    int(value) < coordinate_cache_minimum_frames_per_replica
                    for value in retained
                ):
                    invalid_rows.append({
                        "task_id": task_id,
                        "reason": "coordinate_cache_minimum_frames_per_replica",
                    })
                continue
            if row.get("task_scope") == "conformational_view_algorithm_fit":
                # This task consumes its parent projection. The parent protected
                # estimator is checked here; the fit's method stride is checked
                # by the coupled downstream planner.
                continue
            raw_counts = [int(value) for value in row[
                "global_stride_raw_source_frames_per_replica"
            ]]
            retained = [
                integer_stride_selected_count(value, cache_stride)
                for value in raw_counts
            ]
            embedded = row.get("scientific_sampling_requirements")
            profile = (
                profile_from_contract(embedded)
                if isinstance(embedded, Mapping) else None
            )
            profile_failed = False
            if profile is not None and profile.minimum_frames_per_replica > 0:
                system_ids = row.get("system_ids_per_replica")
                raw_intervals = row.get(
                    "global_stride_raw_frame_intervals_ns_per_replica",
                    row.get("frame_intervals_ns_per_replica"),
                )
                spans = row.get("source_time_spans_ns_per_replica")
                assessment = assess_raw_sampling(
                    profile,
                    selected_frames_per_replica=retained,
                    source_frames_per_replica=raw_counts,
                    system_ids_per_replica=(
                        [str(value) for value in system_ids]
                        if isinstance(system_ids, list) else None
                    ),
                    integer_stride=cache_stride,
                    frame_intervals_ns_per_replica=(
                        [float(value) for value in raw_intervals]
                        if isinstance(raw_intervals, list) else None
                    ),
                    source_time_spans_ns_per_replica=(
                        [float(value) for value in spans]
                        if isinstance(spans, list) else None
                    ),
                )
                if not bool(assessment["keep_enabled"]):
                    invalid_rows.append({
                        "task_id": task_id,
                        "reason": "scientific_sampling_contract",
                        "assessment": assessment,
                    })
                    profile_failed = True
            # Some task definitions impose a stronger, data-dependent floor
            # than the packaged generic profile.  Grouped ML is one example:
            # its required projection count also depends on the requested
            # number and size of independent time blocks.  Check that declared
            # floor even when an embedded profile is present.  Otherwise a
            # coarse cache candidate reaches the resource planner with
            # maximum < minimum and raises instead of being rejected (or made
            # available to dependency-closed optional-module reduction).
            if not profile_failed and any(
                chosen < min(
                    int(row["global_stride_declared_minimum_frames_per_replica"]),
                    available,
                )
                for chosen, available in zip(retained, raw_counts)
            ):
                invalid_rows.append({
                    "task_id": task_id,
                    "reason": "declared_minimum_frames_per_replica",
                })
        if invalid_protected_tasks:
            evaluations.append({
                "overall_trajectory_integer_stride": cache_stride,
                "coordinate_cache_integer_stride": cache_stride,
                "feasibility_status": "protected_scientific_minimum_pruned",
                "invalid_minimum_task_ids": [
                    str(row["task_id"]) for row in invalid_protected_tasks
                ],
                "invalid_protected_tasks": invalid_protected_tasks,
                "balanced_information_upper_bound": (
                    _global_stride_information_upper_bound(
                        base_tasks, cache_stride
                    )
                ),
                "planner_evaluation_wall_seconds": (
                    time.monotonic() - candidate_started
                ),
            })
            continue
        planner_candidate_tasks = candidate_tasks
        if invalid_optional_tasks:
            # Optional methods that cannot meet their own scientific floor at
            # this cache stride make the complete candidate infeasible.  Keep
            # the resource model evaluable so the dependency-closed subset
            # recommender can measure the benefit of disabling them.  The
            # temporary clamp is diagnostic only; the returned plan is forced
            # fail-closed and can never be selected as a feasible campaign.
            invalid_ids = {
                str(row["task_id"]) for row in invalid_optional_tasks
            }
            planner_candidate_tasks = []
            for original in candidate_tasks:
                row = dict(original)
                if str(row.get("task_id")) in invalid_ids:
                    row["minimum_frames_per_replica"] = min(
                        int(row["minimum_frames_per_replica"]),
                        int(row["maximum_frames_per_replica"]),
                    )
                planner_candidate_tasks.append(row)
        plan = plan_projection_coupled_campaign_resource_budget(
            planner_candidate_tasks,
            maximum_parallel_cpus=maximum_parallel_cpus,
            maximum_wall_hours=maximum_wall_hours,
            maximum_memory_gib=maximum_memory_gib,
            planning_utilization=planning_utilization,
            pilot_budget_fraction=pilot_budget_fraction,
            finalization_headroom_fraction=finalization_headroom_fraction,
            memory_safety_factor=memory_safety_factor,
            memory_overhead_gib=memory_overhead_gib,
            minimum_scheduler_memory_gib=minimum_scheduler_memory_gib,
            maximum_cpus_per_node=maximum_cpus_per_node,
            maximum_memory_gib_per_node=maximum_memory_gib_per_node,
            maximum_nodes=maximum_nodes,
            maximum_coupling_iterations=maximum_coupling_iterations,
        )
        candidate_plans.append(plan)
        if invalid_optional_tasks:
            plan["feasibility_status"] = "infeasible"
            reasons = list(plan.get("infeasibility_reasons", []))
            reasons.append(
                "optional tasks fail their scientific sampling contracts at "
                f"overall trajectory stride {cache_stride}"
            )
            plan["infeasibility_reasons"] = reasons
            plan["optional_scientific_minimum_failures"] = (
                invalid_optional_tasks
            )
        information = _global_plan_information_metrics(plan)
        evaluation = {
            "overall_trajectory_integer_stride": cache_stride,
            "coordinate_cache_integer_stride": cache_stride,
            "feasibility_status": plan["feasibility_status"],
            **({
                "invalid_optional_task_ids": [
                    str(row["task_id"]) for row in invalid_optional_tasks
                ],
                "invalid_optional_tasks": invalid_optional_tasks,
            } if invalid_optional_tasks else {}),
            **information,
            "estimated_selected_cpu_hours": plan[
                "estimated_selected_cpu_hours"
            ],
            "estimated_selected_wall_hours_lower_bound": plan[
                "estimated_selected_wall_hours_lower_bound"
            ],
            "planner_evaluation_wall_seconds": (
                time.monotonic() - candidate_started
            ),
        }
        evaluations.append(evaluation)
        if plan["feasibility_status"] == "feasible":
            # Balanced breadth and normalized coverage win before raw counts.
            # Exact information ties retain the finer overall stride; remaining
            # ties prefer the shorter modeled critical path.
            feasible.append((
                (
                    information["balanced_information_utility"],
                    information["minimum_normalized_analysis_coverage"],
                    information["selected_observation_count"],
                    -cache_stride,
                    -float(plan["estimated_selected_wall_hours_lower_bound"]),
                ),
                plan,
            ))
            best_feasible_information = max(
                best_feasible_information,
                information["balanced_information_utility"],
            )
        remaining = candidates[candidate_index + 1:]
        if remaining and math.isfinite(best_feasible_information):
            remaining_upper = max(
                _global_stride_information_upper_bound(base_tasks, value)
                for value in remaining
            )
            if best_feasible_information > remaining_upper + 1.0e-12:
                early_terminated_candidates = list(remaining)
                break
    if not feasible:
        # Return the finest candidate's fail-closed diagnostic rather than
        # obscuring the actual technical minimum or memory shortfall.
        if candidate_plans:
            result = min(candidate_plans, key=_resource_shortfall_score)
        else:
            # No resource calculation is valid when every candidate violates
            # a protected scientific contract.  Return a deliberately
            # fail-closed diagnostic shell without pricing such a stride.
            science_wall = (
                float(maximum_wall_hours)
                * float(planning_utilization)
                * (1.0 - float(pilot_budget_fraction))
                * (1.0 - float(finalization_headroom_fraction))
            )
            result = {
                "feasibility_status": "infeasible",
                "infeasibility_reasons": [
                    "every overall trajectory stride candidate violates a "
                    "protected scientific sampling contract"
                ],
                "tasks": [],
                "science_budget_cpu_hours": (
                    float(maximum_parallel_cpus) * science_wall
                ),
                "science_budget_wall_hours": science_wall,
                "minimum_known_cpu_hours": 0.0,
                "minimum_wall_hours_lower_bound": None,
                "memory_feasibility": {
                    "minimum_required_memory_gib": 0.0,
                    "configured_memory_gib": float(maximum_memory_gib),
                },
                "protected_scientific_minimum_failures": [
                    row
                    for evaluation in evaluations
                    for row in evaluation.get("invalid_protected_tasks", [])
                ],
            }
        result["coordinate_cache_coupling"] = {
            "converged": False,
            "selected_overall_trajectory_integer_stride": None,
            "selected_coordinate_cache_integer_stride": None,
            "candidate_evaluations": evaluations,
            "planner_total_wall_seconds": (
                time.monotonic() - overall_planning_started
            ),
            "reason": "no candidate satisfies the complete campaign envelope",
        }
        result["global_stride_coupling"] = result["coordinate_cache_coupling"]
        return result

    _, result = max(feasible, key=lambda item: item[0])
    result_rows = {
        str(row["task_id"]): row
        for row in result["tasks"]
        if isinstance(row, Mapping)
    }
    cache_report = result_rows[cache_id]
    selected_stride = int(cache_report["coordinate_cache_candidate_stride"])
    parent_strides = {
        str(row["workflow_id"]): int(row["integer_stride"])
        for row in result_rows.values()
        if row.get("task_scope") == "conformational_view"
        and row.get("module_id") == "common_pca"
    }
    effective_rows = []
    for row in result["tasks"]:
        if not isinstance(row, dict):
            continue
        if (
            row.get("task_scope") != "conformational_view_algorithm_fit"
            and row.get("module_id") != "coordinate_cache"
        ):
            local_stride = int(row["integer_stride"])
            row["overall_trajectory_integer_stride"] = selected_stride
            row["coordinate_cache_integer_stride"] = selected_stride
            row["effective_raw_integer_stride"] = selected_stride * local_stride
            effective_rows.append({
                "task_id": str(row["task_id"]),
                "cache_stride": selected_stride,
                "downstream_stride": local_stride,
                "effective_raw_stride": selected_stride * local_stride,
            })
        elif row.get("task_scope") == "conformational_view_algorithm_fit":
            projection_stride = parent_strides[str(row["workflow_id"])]
            fit_stride = int(row["integer_stride"])
            row["coordinate_cache_integer_stride"] = selected_stride
            row["overall_trajectory_integer_stride"] = selected_stride
            row["projection_integer_stride"] = projection_stride
            row["effective_raw_integer_stride"] = (
                selected_stride * projection_stride * fit_stride
            )
            effective_rows.append({
                "task_id": str(row["task_id"]),
                "cache_stride": selected_stride,
                "projection_stride": projection_stride,
                "fit_stride": fit_stride,
                "effective_raw_stride": (
                    selected_stride * projection_stride * fit_stride
                ),
            })
    coupling_report = {
        "coupling_schema": "salsbury-global-stride-coupling-v1",
        "converged": True,
        "selected_overall_trajectory_integer_stride": selected_stride,
        "selected_coordinate_cache_integer_stride": selected_stride,
        "minimum_retained_frames_per_replica": declared_minimum,
        "minimum_cached_frames_per_replica": declared_minimum,
        "maximum_candidate_stride": maximum_stride,
        "coordinate_cache_full_scan_fraction": scan_fraction,
        "requested_candidate_strides": requested_candidate_values,
        "scientifically_pruned_candidate_strides": (
            scientifically_pruned_candidates
        ),
        "evaluated_candidate_strides": evaluated_candidates,
        "early_terminated_candidate_strides": early_terminated_candidates,
        "candidate_count": len(evaluated_candidates),
        "candidate_evaluations": evaluations,
        "planner_total_wall_seconds": time.monotonic() - overall_planning_started,
        "candidate_policy": (
            "Reject a candidate before resource optimization when its raw "
            "stride violates a protected analysis contract. Evaluate every "
            "remaining exact integer overall stride on every retained raw "
            "trajectory consumer, including solvated analyses, then replan "
            "method-specific and projection-derived strides. Optional methods "
            "may be disabled only by dependency-closed reduction. The cache scan "
            "fraction charges sequential connectivity reconstruction to all raw "
            "frames and the remaining materialization cost to retained frames."
        ),
        "effective_stride_rows": effective_rows,
        "execution_ready": True,
        "execution_boundary": (
            "Every source frame is decoded for continuous unwrapping. The "
            "selected cache stride controls only materialization; cached timing "
            "and sample axes preserve physical spacing, downstream cache strides "
            "compose multiplicatively, and lagged analyses retain segment-safe "
            "physical-time metadata."
        ),
        "calibration_boundary": (
            "The cache scan/write cost split is a sensitivity parameter because "
            "the current one-point calibration cannot identify it."
        ),
    }
    result["global_stride_coupling"] = coupling_report
    # Compatibility key for local callers of the first prototype.
    result["coordinate_cache_coupling"] = coupling_report
    return result


def plan_cache_projection_coupled_campaign_resource_budget(
    tasks: Sequence[Mapping[str, object]],
    **kwargs: object,
) -> Dict[str, object]:
    """Compatibility alias for global overall-stride campaign planning."""

    return plan_global_stride_projection_coupled_campaign_resource_budget(
        tasks, **kwargs
    )


def _configuration_switch_for_task(task: Mapping[str, object]) -> str:
    module_id = str(task.get("module_id", task.get("task_id", "unknown")))
    if module_id == "coordinate_cache":
        return "execution.coordinate_cache"
    if module_id == "alternative_clustering" and task.get("algorithm_id"):
        names = {
            "mwpam": "minkowski_weighted_pam",
        }
        algorithm = names.get(
            str(task["algorithm_id"]), str(task["algorithm_id"])
        )
        return f"clustering.methods.{algorithm}.enabled"
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


def _resource_shortfall_score(plan: Mapping[str, object]) -> float:
    """Return a normalized deterministic infeasibility score."""

    science_cpu = max(1.0e-12, float(plan["science_budget_cpu_hours"]))
    science_wall = max(1.0e-12, float(plan["science_budget_wall_hours"]))
    minimum_wall = plan.get("minimum_wall_hours_lower_bound")
    memory = plan["memory_feasibility"]
    assert isinstance(memory, Mapping)
    score = max(
        0.0, float(plan["minimum_known_cpu_hours"]) / science_cpu - 1.0
    )
    if isinstance(minimum_wall, (int, float)) and not isinstance(
        minimum_wall, bool
    ):
        score += max(0.0, float(minimum_wall) / science_wall - 1.0)
    score += max(
        0.0,
        float(memory["minimum_required_memory_gib"])
        / max(1.0e-12, float(memory["configured_memory_gib"]))
        - 1.0,
    )
    calibration = plan.get("tasks_requiring_project_pilots", [])
    if isinstance(calibration, list):
        score += float(len(calibration))
    optional_scientific_failures = plan.get(
        "optional_scientific_minimum_failures", []
    )
    if isinstance(optional_scientific_failures, list):
        score += float(len(optional_scientific_failures))
    protected_scientific_failures = plan.get(
        "protected_scientific_minimum_failures", []
    )
    if isinstance(protected_scientific_failures, list):
        score += 1000.0 * float(len(protected_scientific_failures))
    return score


def recommend_scientifically_valid_task_subset(
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
    maximum_cpus_per_node: Optional[int] = None,
    maximum_memory_gib_per_node: Optional[float] = None,
    maximum_nodes: Optional[int] = None,
    protected_module_ids: Sequence[str] = (
        "coordinate_cache", "provenance_manifest", "preflight_inventory",
        "common_atom_mapping", "structural_integrity_qc", "replica_rmsd_rg",
        "pooled_rmsf", "individual_pca", "common_pca", "dccm",
        "pca_fes_basins", "representative_frames",
    ),
    use_global_stride_coupling: bool = False,
    coordinate_cache_minimum_frames_per_replica: int = 1,
    coordinate_cache_full_scan_fraction: float = 1.0,
    overall_stride_candidate_strides: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Propose the broadest scientifically valid task subset for an envelope.

    The recommendation never mutates a requested configuration.  It prices the
    standard scientific minimum first, removes the configuration bundle with
    the greatest normalized bottleneck relief per unit of lost scientific
    priority, closes downstream dependencies, and replans after every decision.
    Extra sampling is never sacrificed below a method's standard floor merely
    to keep the method nominally enabled.
    """

    from .analysis_config import DEPENDENCIES, PROTECTED_MODULES

    started = time.monotonic()
    planner_kwargs = {
        "maximum_parallel_cpus": maximum_parallel_cpus,
        "maximum_wall_hours": maximum_wall_hours,
        "maximum_memory_gib": maximum_memory_gib,
        "planning_utilization": planning_utilization,
        "pilot_budget_fraction": pilot_budget_fraction,
        "finalization_headroom_fraction": finalization_headroom_fraction,
        "memory_safety_factor": memory_safety_factor,
        "memory_overhead_gib": memory_overhead_gib,
        "minimum_scheduler_memory_gib": minimum_scheduler_memory_gib,
        "maximum_cpus_per_node": maximum_cpus_per_node,
        "maximum_memory_gib_per_node": maximum_memory_gib_per_node,
        "maximum_nodes": maximum_nodes,
    }

    def run_planner(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        if use_global_stride_coupling:
            return plan_global_stride_projection_coupled_campaign_resource_budget(
                rows,
                coordinate_cache_minimum_frames_per_replica=(
                    coordinate_cache_minimum_frames_per_replica
                ),
                coordinate_cache_full_scan_fraction=(
                    coordinate_cache_full_scan_fraction
                ),
                overall_stride_candidate_strides=(
                    overall_stride_candidate_strides
                ),
                protected_module_ids=tuple(protected_module_ids),
                **planner_kwargs,
            )
        return plan_campaign_resource_budget(rows, **planner_kwargs)
    working = [deepcopy(dict(task)) for task in tasks]
    if not working:
        raise ResourcePlanningError("task-subset recommendation requires tasks")
    protected = (
        set(str(value) for value in protected_module_ids)
        | set(PROTECTED_MODULES)
        | {"coordinate_cache"}
    )
    protected_task_ids = {
        str(row["task_id"])
        for row in working
        if str(row.get("module_id")) in protected
    }
    disabled_switches: list[str] = []
    decisions: list[Dict[str, object]] = []

    def task_ids_for_switch(
        rows: Sequence[Mapping[str, object]], switch: str
    ) -> set[str]:
        removed = {
            str(row["task_id"])
            for row in rows
            if _configuration_switch_for_task(row) == switch
        }
        changed = True
        while changed:
            changed = False
            modules_present = {
                str(row.get("module_id"))
                for row in rows if str(row["task_id"]) not in removed
            }
            for row in rows:
                task_id = str(row["task_id"])
                module_id = str(row.get("module_id"))
                if task_id in removed:
                    continue
                dependencies = DEPENDENCIES.get(module_id, set())
                if any(dependency not in modules_present for dependency in dependencies):
                    removed.add(task_id)
                    changed = True
        # The coordinate cache is also the validated input for cache-compatible
        # base modules.  It is protected preprocessing, not a disposable
        # conformational-view helper, so optional-module removal never deletes
        # it merely because no PCA view remains.
        return removed

    current_plan = run_planner(working)
    current_score = _resource_shortfall_score(current_plan)
    while current_plan["feasibility_status"] != "feasible":
        switches = sorted({
            _configuration_switch_for_task(row)
            for row in working
            if str(row.get("module_id")) not in protected
        })
        candidates = []
        for switch in switches:
            removed_ids = task_ids_for_switch(working, switch)
            if removed_ids.intersection(protected_task_ids):
                continue
            candidate_tasks = [
                row for row in working if str(row["task_id"]) not in removed_ids
            ]
            if not candidate_tasks:
                continue
            candidate_plan = run_planner(candidate_tasks)
            candidate_score = _resource_shortfall_score(candidate_plan)
            relief = current_score - candidate_score
            if relief <= 1.0e-12:
                continue
            removed_rows = [
                row for row in working if str(row["task_id"]) in removed_ids
            ]
            priority_loss = sum(
                float(row.get("priority_weight", 1.0)) for row in removed_rows
            )
            candidates.append((
                relief / max(1.0e-12, priority_loss),
                relief,
                -priority_loss,
                switch,
                candidate_tasks,
                candidate_plan,
                sorted(removed_ids),
            ))
        if not candidates:
            break
        (
            _, relief, negative_priority, switch, candidate_tasks,
            candidate_plan, removed_ids,
        ) = max(candidates, key=lambda value: value[:4])
        decisions.append({
            "iteration": len(decisions) + 1,
            "disabled_configuration_switch": switch,
            "removed_task_ids": removed_ids,
            "normalized_shortfall_before": current_score,
            "normalized_shortfall_after": _resource_shortfall_score(
                candidate_plan
            ),
            "shortfall_relief": relief,
            "scientific_priority_removed": -negative_priority,
        })
        disabled_switches.append(switch)
        working = candidate_tasks
        current_plan = candidate_plan
        current_score = _resource_shortfall_score(current_plan)
    feasible = current_plan["feasibility_status"] == "feasible"
    protected_minimum_request = deepcopy(
        current_plan.get("permissive_minimum_resource_request", {})
    )
    if isinstance(protected_minimum_request, dict):
        protected_minimum_request["request_scope"] = (
            "best_dependency_closed_subset_that_preserves_all_protected_modules"
        )
    return {
        "recommendation_schema": "salsbury-scientific-method-fit-v1",
        "technical_status": "complete",
        "recommendation_status": (
            "feasible_subset_found" if feasible else "no_feasible_subset_found"
        ),
        "recommendation_message": (
            "A reduced configuration can meet the envelope while retaining every "
            "protected module. Review and apply the proposed switches explicitly."
            if feasible else
            "No acceptable reduced plan: the envelope cannot retain every "
            "protected module at its scientific minimum. Increase the resource "
            "envelope; protected checks will not be disabled."
        ),
        "automatic_changes_applied": False,
        "protected_module_ids": sorted(protected),
        "protected_task_ids": sorted(protected_task_ids),
        "protected_set_preserved": not any(
            task_id not in {str(row["task_id"]) for row in working}
            for task_id in protected_task_ids
        ),
        "disabled_configuration_switches": disabled_switches if feasible else [],
        "attempted_configuration_switches": (
            [] if feasible else disabled_switches
        ),
        "configuration_patch": (
            {
                switch: (
                    "off" if switch == "execution.coordinate_cache" else False
                )
                for switch in disabled_switches
            }
            if feasible else {}
        ),
        "retained_task_ids": sorted(str(row["task_id"]) for row in working),
        "decisions": decisions,
        "best_protected_subset_minimum_resource_request": (
            protected_minimum_request
        ),
        "recommended_plan": current_plan,
        "planner_wall_seconds": time.monotonic() - started,
        "strategy": (
            "deterministic dependency-closed removal by normalized CPU, wall, "
            "and memory shortfall relief per lost scientific-priority weight; "
            + (
                "every subset is repriced with coupled cache and method strides"
                if use_global_stride_coupling else
                "every subset is repriced with direct method strides"
            )
        ),
        "scientific_boundary": (
            "This is a proposed configuration. Apply and rerun planning explicitly; "
            "a disabled method is absent, not a low-sample scientific result."
            if feasible else
            "No configuration change is proposed. Diagnostic removals did not make "
            "the protected workflow feasible, so the resource envelope must change."
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
