"""Submission-only constraints for a frozen scientific execution plan.

This module never changes observations, estimates, worker counts or scientific
dependencies. A smaller resource ceiling changes scheduling, not the analysis.
"""
from __future__ import annotations

from copy import deepcopy
import math


PACKAGING_DEFAULTS = {
    "maximum_job_cpus": None,
    "maximum_job_memory_gib": None,
    "maximum_concurrent_cpus": None,
    "maximum_concurrent_memory_gib": None,
    "maximum_reserved_cpu_hours": None,
}


def validate_packaging(value):
    if not isinstance(value, dict) or set(value) - set(PACKAGING_DEFAULTS):
        raise ValueError("Slurm packaging contains unknown fields or is not an object")
    result = dict(PACKAGING_DEFAULTS, **value)
    for name, number in result.items():
        if number is None:
            continue
        if (isinstance(number, bool) or not isinstance(number, (float, int))
                or not math.isfinite(number) or number <= 0
                or (name.endswith("cpus") and not isinstance(number, int))):
            raise ValueError(f"packaging.{name} must be positive and finite"
                             + (" (integer CPUs)" if name.endswith("cpus") else ""))
    return result


def package_execution_plan(plan, profile):
    """Return a scheduling copy; preserve every original task byte-for-byte."""
    policy = validate_packaging(profile.get("packaging", {}))
    result = deepcopy(plan)
    reserved_cpu_hours = 0.0
    rows = []
    for phase in result.get("phases", []):
        for task in phase.get("tasks", []):
            task_id = task.get("task_id") or f"{phase['phase_id']}:{task['script']}:{task.get('array_task_id', 'single')}"
            cpus = task["cpu_slots"]
            nodes = task.get("node_count", 1)
            memory = task["requested_memory_gib"] * nodes
            minutes = task["requested_wall_minutes"]
            for name, value in (("CPU slots", cpus), ("nodes", nodes),
                                ("memory", memory), ("wall minutes", minutes)):
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{task_id}: invalid {name} for Slurm packaging")
            if not isinstance(cpus, int) or not isinstance(nodes, int):
                raise ValueError(f"{task_id}: CPU slots and nodes must be integers")
            if task.get("distributed_replica_execution"):
                workers = task.get("distributed_worker_count")
                per_node = task.get("workers_per_node")
                if (workers != cpus or isinstance(per_node, bool) or not isinstance(per_node, int)
                        or per_node <= 0 or math.ceil(cpus / per_node) != nodes):
                    raise ValueError(f"{task_id}: distributed Slurm request disagrees with planned worker slots")
            elif nodes != 1:
                raise ValueError(f"{task_id}: non-distributed task cannot request multiple nodes")
            # sbatch --mem uses whole GiB in the generated launcher. Validate
            # that reservation, not the smaller unrounded model estimate.
            memory = math.ceil(task["requested_memory_gib"]) * nodes
            for name, value, limit in (
                ("job CPUs", cpus, policy["maximum_job_cpus"] or plan["maximum_parallel_cpus"]),
                ("job aggregate memory GiB", memory, policy["maximum_job_memory_gib"] or plan["maximum_parallel_memory_gib"]),
                ("explicit concurrent CPUs", cpus, policy["maximum_concurrent_cpus"]),
                ("explicit concurrent memory GiB", memory, policy["maximum_concurrent_memory_gib"]),
            ):
                if limit is not None and value > limit:
                    raise ValueError(f"{task_id}: indivisible task requires {value:g} {name}; "
                                     f"ceiling is {limit:g}. No worker count, sampling or estimator was changed.")
            retries = (int(plan.get("maximum_task_attempts", 2))
                       if plan.get("autorecovery", True) else 1)
            reserved_cpu_hours += cpus * minutes / 60 * retries
            rows.append({"task_id": task_id, "cpus": cpus, "nodes": nodes,
                         "memory_gib_per_node": math.ceil(task["requested_memory_gib"]),
                         "aggregate_memory_gib": memory, "wall_minutes": minutes,
                         "maximum_attempts": retries})
    # The plan envelope bounds individual work, not the sum of independent
    # allocations on different nodes. Only separately supplied concurrent caps
    # add resource predecessors. Finite sums below are unrestrictive bounds.
    result["maximum_parallel_cpus"] = policy["maximum_concurrent_cpus"] or max(1, sum(r["cpus"] for r in rows))
    result["maximum_parallel_memory_gib"] = policy["maximum_concurrent_memory_gib"] or max(1., sum(math.ceil(r["aggregate_memory_gib"] * 4) / 4 for r in rows))
    result["scheduler_reservation_rounding"] = True
    limit = policy["maximum_reserved_cpu_hours"]
    if limit is not None and reserved_cpu_hours > limit + 1e-9:
        raise ValueError(f"Slurm packaging reserves up to {reserved_cpu_hours:g} CPU-hours "
                         f"including configured attempts, exceeding ceiling {limit:g}; nothing was reduced")
    result["packaging_contract"] = {
        "schema": "salsbury-slurm-packaging-v1", "strategy": "independent_task_jobs",
        "limits": policy, "effective_concurrent_cpus": result["maximum_parallel_cpus"],
        "effective_concurrent_memory_gib": result["maximum_parallel_memory_gib"],
        "reserved_cpu_hours_upper_bound": reserved_cpu_hours,
        "tasks": rows, "scientific_plan_changed": False,
        "memory_interpretation": "Final padded requests passed through once; worker memory is not multiplied again.",
        "placement": "Slurm chooses physical nodes; conceptual token placement is not node binding.",
        "plan_envelope_is_not_a_concurrent_cap": True,
        "aggregate_caps_are_explicit_only": True,
    }
    return result


def packaging_schedule(plan, profile):
    """Use the existing DAG machinery without inherited placement/token limits."""
    from .execution_adapters import _slurm_resource_epochs, _slurm_submission_preview
    packaged = package_execution_plan(plan, profile)
    epochs = _slurm_resource_epochs(packaged, profile["partitions"],
        profile["partition_maximum_wall_minutes"], profile["partition_maximum_nodes"],
        profile["resource_policy"], {})
    # Per-node fit remains mandatory, independent of combined campaign usage.
    nodes = {**profile.get("node_policy", {}), **plan.get("node_policy", {})}
    reserve = float(plan.get("node_memory_reserve_gib", nodes.get("memory_reserve_gib", 0)))
    for row in packaged["packaging_contract"]["tasks"]:
        if nodes.get("memory_gib_per_node") is not None and row["memory_gib_per_node"] + reserve > nodes["memory_gib_per_node"] + 1e-9:
            raise ValueError(f"{row['task_id']}: padded memory plus node reserve exceeds one eligible node; no estimator was split")
    preview = _slurm_submission_preview(packaged, epochs, nodes)
    preview["packaging_contract"] = packaged["packaging_contract"]
    preview["planned_node_count"] = None  # Slurm chooses placement at dispatch.
    preview["physical_node_count"] = "scheduler_assigned; not bounded by the plan's conceptual node count"
    preview["resource_contract"] = (
        "Independent jobs retain scientific and completion-input dependencies. "
        "Only explicit packaging concurrent caps add afterany resource edges; "
        "the scientific plan envelope is not an aggregate submission cap.")
    preview["warnings"] = [w for w in preview["warnings"] if w["code"] != "REQUESTED_CPUS_EXCEED_GENERATED_PARALLELISM"]
    preview["warning_count"] = len(preview["warnings"])
    return packaged, epochs, preview


def check_single_allocation(single, profile):
    """Check the actual one-allocation reservation, not summed task estimates."""
    result = deepcopy(single)
    if not result.get("submission_permitted"):
        return result
    limits = validate_packaging(profile.get("packaging", {}))
    reserved = result["cpus"] * result["requested_wall_hours"]
    result["reserved_cpu_hours_upper_bound"] = reserved
    violations = []
    for value, name in (
        (result["cpus"], "maximum_job_cpus"),
        (result["memory_gib"], "maximum_job_memory_gib"),
        (result["cpus"], "maximum_concurrent_cpus"),
        (result["memory_gib"], "maximum_concurrent_memory_gib"),
        (reserved, "maximum_reserved_cpu_hours"),
    ):
        if limits[name] is not None and value > limits[name] + 1e-9:
            violations.append(f"single allocation requests {value:g}; packaging.{name} is {limits[name]:g}")
    if violations:
        result.update(submission_permitted=False, packaging_errors=violations)
    return result
