"""Pure dependency/CPU/node-memory schedule shared by planning and execution."""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence


class ResourceScheduleError(ValueError):
    """Invalid or unschedulable task/resource contract."""


def schedule_resource_tasks(items: Sequence[Mapping[str, object]], *,
                            maximum_cpus: int, maximum_memory: float,
                            node_policy: Mapping[str, object]) -> List[Dict[str, object]]:
    """Keep smaller-node placements eligible when additional nodes are allowed.

    Limits are ceilings, not requirements to occupy every available node.
    Every placement is independently resource validated by the same scheduler.
    """
    nodes = node_policy.get("maximum_nodes_per_campaign")
    if nodes is None or node_policy.get("cpus_per_node") is None:
        return _schedule_resource_tasks(items, maximum_cpus=maximum_cpus,
                                        maximum_memory=maximum_memory, node_policy=node_policy)
    first = max((int(item.get("node_count", 1)) for item in items), default=1)
    results, last_error = [], None
    for count in range(first, int(nodes) + 1):
        try:
            result = _schedule_resource_tasks(items,
                maximum_cpus=min(maximum_cpus, count * int(node_policy["cpus_per_node"])),
                maximum_memory=min(maximum_memory, count * float(node_policy["memory_gib_per_node"])),
                node_policy={**node_policy, "maximum_nodes_per_campaign": count})
            results.append(result)
        except ResourceScheduleError as exc:
            last_error = exc
    if not results:
        raise last_error or ResourceScheduleError("no allowed node placement")
    return min(results, key=lambda result: (
        float(result[0]["planned_wall_hours"]),
        int(result[0]["resource_token_policy"]["reserved_node_count_upper_bound"])))


def _schedule_resource_tasks(items: Sequence[Mapping[str, object]], *,
                             maximum_cpus: int, maximum_memory: float,
                             node_policy: Mapping[str, object]) -> List[Dict[str, object]]:
    """Return a deterministic resource-token schedule without reading files."""
    items = [dict(item) for item in items]
    if maximum_cpus <= 0 or maximum_memory <= 0:
        raise ResourceScheduleError("resource limits must be positive")
    node_cpus = node_policy.get("cpus_per_node")
    node_memory = node_policy.get("memory_gib_per_node")
    configured = node_policy.get("maximum_nodes_per_campaign")
    maximum_nodes = (int(configured) if configured is not None else
                     max(math.ceil(maximum_cpus / int(node_cpus)),
                         math.ceil(maximum_memory / float(node_memory)))
                     if node_cpus is not None and node_memory is not None else None)
    if (node_cpus is None) != (node_memory is None):
        raise ResourceScheduleError(
            "resource-token scheduling requires both node CPU and memory limits"
        )
    memory_quantum_gib = 0.25
    node_reserve = float(node_policy.get("memory_reserve_gib", 0.0))
    # A permitted node count is not itself an allocation. Bound possible
    # occupancy by the tasks and CPU capacity before reserving node overhead.
    occupied_node_bound = min(maximum_nodes or 1, maximum_cpus,
                              sum(int(item["node_count"]) for item in items) or 1)
    if maximum_nodes is not None:
        maximum_nodes = occupied_node_bound
    campaign_reserve = node_reserve * occupied_node_bound
    if not math.isfinite(node_reserve) or node_reserve < 0:
        raise ResourceScheduleError("node memory reserve must be finite and nonnegative")
    memory_token_count = int(
        math.floor((maximum_memory - campaign_reserve) / memory_quantum_gib + 1.0e-9)
    )
    if memory_token_count <= 0:
        raise ResourceScheduleError(
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
            raise ResourceScheduleError(
                "resource-token scheduling could not determine a node count"
            )
        per_node_memory_tokens = int(math.floor(
            (float(node_memory) - node_reserve) / memory_quantum_gib + 1.0e-9
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

    token_orders = {}

    def selected_token_indices(
        tokens: Sequence[Mapping[str, object]], count: int,
    ) -> List[int]:
        key = id(tokens)
        if key not in token_orders:
            token_orders[key] = sorted(range(len(tokens)), key=lambda index: (
                float(tokens[index]["available"]), index))
        return token_orders[key][:count]

    item_by_id = {str(item["item_id"]): item for item in items}
    if len(item_by_id) != len(items):
        raise ResourceScheduleError("execution plan contains duplicate task IDs")

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
        raise ResourceScheduleError(
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
            raise ResourceScheduleError(
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
    allocation_templates = {}

    def candidate_allocation(item: Mapping[str, object]) -> Dict[str, object]:
        scheduling_id = str(item["item_id"])
        signature = tuple(item.get(key) for key in (
            "cpu_slots", "memory_gib", "node_count", "workers_per_node",
            "requested_memory_gib"))
        declared_dependencies = declared_dependencies_by_id[scheduling_id]
        if signature in allocation_templates:
            template = allocation_templates[signature]
            start = max(template["resource_ready"], *(planned_finishes[d]
                        for d in declared_dependencies), 0.0)
            return {**template, "start": start,
                    "finish": start + float(item["planned_wall_hours"]),
                    "resource_predecessors": sorted(template["owners"].difference(declared_dependencies))}
        cpu_count = int(item["cpu_slots"])
        memory_count = int(math.ceil(
            float(item["memory_gib"]) / memory_quantum_gib - 1.0e-12
        ))
        if cpu_count > len(cpu_tokens) or memory_count > len(memory_tokens):
            raise ResourceScheduleError(
                f"task {item['item_id']} exceeds the aggregate resource envelope"
            )
        cpu_indices = selected_token_indices(cpu_tokens, cpu_count)
        memory_indices = selected_token_indices(memory_tokens, memory_count)
        selected_node_tokens: List[tuple[int, List[int], List[int]]] = []
        assigned_node_indices: List[int] = []
        if node_tokens:
            task_nodes = int(item.get("node_count", 1))
            if task_nodes > len(node_tokens):
                raise ResourceScheduleError(
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
                raise ResourceScheduleError(
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
                raise ResourceScheduleError(
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
        resource_ready = max(
            [float(cpu_tokens[index]["available"]) for index in cpu_indices]
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
        start = max(resource_ready, *(planned_finishes[d]
                    for d in declared_dependencies), 0.0)
        finish = start + float(item["planned_wall_hours"])
        template = {
            "cpu_indices": cpu_indices,
            "memory_indices": memory_indices,
            "selected_node_tokens": selected_node_tokens,
            "assigned_node_indices": assigned_node_indices,
            "resource_predecessors": resource_predecessors,
            "start": start,
            "finish": finish,
            "memory_count": memory_count,
            "resource_ready": resource_ready,
            "owners": global_cpu_owners | global_memory_owners | node_owners,
        }
        allocation_templates[signature] = template
        return template

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
            raise ResourceScheduleError(
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
                str(row[0].get("schedule_order_key", row[0]["item_id"])),
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
        token_orders.clear()
        allocation_templates.clear()
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
        "memory_including_node_reserve_gib": peak_memory + campaign_reserve,
        "planned_wall_hours": planned_wall,
        "wall_hours": reservation_wall,
        "resource_token_policy": {
            "node_memory_reserve_gib": node_reserve,
            "reserved_node_count_upper_bound": occupied_node_bound,
            "aggregate_node_reserve_gib": campaign_reserve,
            "memory_reserve_policy": "once per potentially occupied node, never per task",
            "cpu_token_count": maximum_cpus,
            "memory_token_count": memory_token_count,
            "memory_token_quantum_gib": memory_quantum_gib,
            "contract": (
                "a task acquires fixed CPU and memory token chains; afterany "
                "predecessors release those tokens regardless of success"
            ),
        },
    }]
