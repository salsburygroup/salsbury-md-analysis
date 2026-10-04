"""Pure dependency/CPU/node-memory schedule shared by planning and execution."""
from __future__ import annotations

import math
from bisect import bisect_left
from typing import Dict, List, Mapping, NamedTuple, Sequence

from .planning_reuse import memoized_schedule
from .planning_search import record_schedule_work


class ResourceScheduleError(ValueError):
    """Invalid or unschedulable task/resource contract."""


class _TokenSelection(NamedTuple):
    ranges: tuple[tuple[int, int], ...]
    count: int
    ready: float
    owners: frozenset[str]

    def __len__(self) -> int:
        return self.count


class _TokenPool:
    """Exact run-length representation of the former per-token array.

    Tokens are still selected by (availability, original index). Compressing
    equal adjacent states changes neither the quarter-GiB quantum nor ownership
    edges; work scales with allocation boundaries instead of memory capacity.
    """

    def __init__(self, count: int):
        self.count = count
        self.runs = [(0, count, 0.0, None)] if count else []
        self._ordered = None
        self._selections = {}

    def __len__(self) -> int:
        return self.count

    def select(self, count: int) -> _TokenSelection:
        count = min(count, self.count)
        if count in self._selections:
            return self._selections[count]
        if self._ordered is None:
            self._ordered = sorted(self.runs, key=lambda run: (run[2], run[0]))
            self._ends, self._owners = [], []
            total, owners = 0, frozenset()
            for lo, hi, _available, owner in self._ordered:
                total += hi - lo
                self._ends.append(total)
                if owner is not None:
                    owners = owners | {owner}
                self._owners.append(owners)
            self._ranges = tuple((lo, hi) for lo, hi, *_ in self._ordered)
        if count:
            index = bisect_left(self._ends, count)
            lo, _hi, ready, _owner = self._ordered[index]
            previous = self._ends[index - 1] if index else 0
            ranges = self._ranges[:index] + ((lo, lo + count - previous),)
            selection = _TokenSelection(ranges, count, ready, self._owners[index])
        else:
            selection = _TokenSelection((), 0, 0.0, frozenset())
        self._selections[count] = selection
        return selection

    def assign(self, selection: _TokenSelection, finish: float, owner: str) -> None:
        if not selection.count:
            return
        chosen = sorted(selection.ranges)
        updated = []

        def append(lo, hi, available, previous_owner):
            if lo == hi:
                return
            if updated and updated[-1][1] == lo and updated[-1][2:] == (available, previous_owner):
                updated[-1] = (updated[-1][0], hi, available, previous_owner)
            else:
                updated.append((lo, hi, available, previous_owner))

        index = 0
        for lo, hi, available, previous_owner in self.runs:
            cursor = lo
            while index < len(chosen) and chosen[index][0] < hi:
                start, end = chosen[index]
                append(cursor, start, available, previous_owner)
                append(start, end, finish, owner)
                cursor = end
                index += 1
            append(cursor, hi, available, previous_owner)
        self.runs = updated
        self._ordered = None
        self._selections.clear()


@memoized_schedule
def schedule_resource_tasks(items: Sequence[Mapping[str, object]], *,
                            maximum_cpus: int, maximum_memory: float,
                            node_policy: Mapping[str, object]) -> List[Dict[str, object]]:
    """Keep smaller-node placements eligible when additional nodes are allowed.

    Limits are ceilings, not requirements to occupy every available node.
    Every placement is independently resource validated by the same scheduler.
    """
    record_schedule_work()
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
    cpu_tokens = _TokenPool(maximum_cpus)
    memory_tokens = _TokenPool(memory_token_count)
    if node_cpus is not None and node_memory is not None:
        if maximum_nodes is None:
            raise ResourceScheduleError(
                "resource-token scheduling could not determine a node count"
            )
        per_node_memory_tokens = int(math.floor(
            (float(node_memory) - node_reserve) / memory_quantum_gib + 1.0e-9
        ))
        node_tokens = [{
            "cpu": _TokenPool(int(node_cpus)),
            "memory": _TokenPool(per_node_memory_tokens),
        } for _ in range(maximum_nodes)]
        # With one node, equal aggregate and node capacities are the same
        # token constraint. Share their exact state instead of recomputing it.
        if maximum_nodes == 1:
            if len(node_tokens[0]["cpu"]) == len(cpu_tokens):
                node_tokens[0]["cpu"] = cpu_tokens
            if (len(node_tokens[0]["memory"]) == len(memory_tokens)
                    and all(float(item["memory_gib"]) == float(item["requested_memory_gib"])
                            for item in items)):
                node_tokens[0]["memory"] = memory_tokens
    else:
        node_tokens = []

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
    signatures = {item_id: tuple(item.get(key) for key in (
        "cpu_slots", "memory_gib", "node_count", "workers_per_node",
        "requested_memory_gib")) for item_id, item in item_by_id.items()}
    priorities = {item_id: (
        -critical_rank(item_id), -float(item["memory_gib"]),
        -int(item["cpu_slots"]), str(item.get("schedule_order_key", item_id)), index,
    ) for index, (item_id, item) in enumerate(item_by_id.items())}
    remaining_dependencies = {key: len(deps) for key, deps in declared_dependencies_by_id.items()}
    dependency_ready = {}
    ready_groups = {}

    def mark_ready(item_id):
        dependency_ready[item_id] = max(
            (planned_finishes[d] for d in declared_dependencies_by_id[item_id]), default=0.0)
        ready_groups.setdefault(signatures[item_id], {})[item_id] = None

    for item_id, count in remaining_dependencies.items():
        if not count:
            mark_ready(item_id)

    def candidate_allocation(item: Mapping[str, object]) -> Dict[str, object]:
        scheduling_id = str(item["item_id"])
        signature = signatures[scheduling_id]
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
        cpu_indices = cpu_tokens.select(cpu_count)
        memory_indices = memory_tokens.select(memory_count)
        selected_node_tokens: List[tuple[int, _TokenSelection, _TokenSelection]] = []
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
                node_cpu_indices = node["cpu"].select(max(per_node_cpus))
                node_memory_indices = node["memory"].select(memory_per_node_count)
                if (
                    len(node_cpu_indices) < max(per_node_cpus)
                    or len(node_memory_indices) < memory_per_node_count
                ):
                    continue
                ready = max(node_cpu_indices.ready, node_memory_indices.ready)
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
                    node["cpu"].select(fragment_cpus),
                    node["memory"].select(memory_per_node_count),
                ))
        global_cpu_owners = cpu_indices.owners
        global_memory_owners = memory_indices.owners
        node_owners = {
            owner
            for _node_index, node_cpu_indices, node_memory_indices
            in selected_node_tokens
            for selection in (node_cpu_indices, node_memory_indices)
            for owner in selection.owners
        }
        resource_predecessors = sorted(
            (global_cpu_owners | global_memory_owners | node_owners)
            .difference(declared_dependencies)
        )
        resource_ready = max(
            [cpu_indices.ready, memory_indices.ready]
            + [
                selection.ready
                for _node_index, node_cpu_indices, node_memory_indices
                in selected_node_tokens
                for selection in (node_cpu_indices, node_memory_indices)
            ]
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

    scheduled_items: List[Dict[str, object]] = []
    while len(scheduled_items) < len(item_by_id):
        if not ready_groups:
            raise ResourceScheduleError(
                "execution plan contains a dependency cycle among: "
                + ", ".join(sorted(set(item_by_id) - set(planned_finishes)))
            )
        candidates = []
        # Equal footprints select exactly the same resource tokens. Compare
        # their dependency times and static priorities before constructing an
        # allocation record. This is the same lexicographic minimum, not a new
        # packing heuristic. The final index preserves the original stable tie.
        for ready_ids in ready_groups.values():
            first_id = next(iter(ready_ids))
            first_allocation = candidate_allocation(item_by_id[first_id])
            resource_ready = first_allocation["resource_ready"]
            best_id = min(ready_ids, key=lambda key: (
                max(resource_ready, dependency_ready[key]), priorities[key]))
            item = item_by_id[best_id]
            allocation = (first_allocation if best_id == first_id else candidate_allocation(item))
            candidates.append((item, allocation))
        item, allocation = min(
            candidates,
            key=lambda row: (
                float(row[1]["start"]),
                priorities[str(row[0]["item_id"])],
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
        cpu_tokens.assign(allocation["cpu_indices"], finish, scheduling_id)
        memory_tokens.assign(allocation["memory_indices"], finish, scheduling_id)
        for (
            node_index, node_cpu_indices, node_memory_indices
        ) in allocation["selected_node_tokens"]:
            if node_tokens[node_index]["cpu"] is not cpu_tokens:
                node_tokens[node_index]["cpu"].assign(node_cpu_indices, finish, scheduling_id)
            if node_tokens[node_index]["memory"] is not memory_tokens:
                node_tokens[node_index]["memory"].assign(node_memory_indices, finish, scheduling_id)
        planned_finishes[scheduling_id] = finish
        allocation_templates.clear()
        scheduled_items.append(item)
        group = ready_groups[signatures[scheduling_id]]
        del group[scheduling_id]
        if not group:
            del ready_groups[signatures[scheduling_id]]
        for successor in successors[scheduling_id]:
            remaining_dependencies[successor] -= 1
            if not remaining_dependencies[successor]:
                mark_ready(successor)

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
