#!/usr/bin/env python3
"""Build the September 2026 Apollo node-allocation comparison.

This is a planning-only evidence builder. It reads existing campaign plans,
updates stale structural-QC, hydrogen-bond, and ion CPU terms with the accepted
size/length/work models, and evaluates one through sixteen Apollo nodes. It
does not submit, rerun, cancel, or modify any cluster job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from salsbury_md_analysis.ensemble_parallelism import annotate_task_parallelism
from salsbury_md_analysis.node_sweep import plan_node_sweep
from salsbury_md_analysis.planner_calibration_models import (
    predict_size_length_cpu_terms,
    validate_size_length_models,
)
from salsbury_md_analysis.scientific_sampling import (
    apply_scientific_minimums_to_tasks,
    load_scientific_minimums,
)


PARALLEL_MODULES = {
    "coordinate_cache", "structural_integrity_qc", "replica_rmsd_rg",
    "pooled_rmsf", "dccm", "hydrogen_bond_discovery",
    "water_mediated_hydrogen_bond_networks", "secondary_structure",
    "solvent_accessible_surface_area", "ion_coordination_geometry",
    "ion_atmosphere",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_plan(path: Path) -> list[Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path} has no task list")
    return [dict(row) for row in tasks]


def _system_ids(
    task: Mapping[str, object], groups: Sequence[tuple[str, int]], count: int,
) -> list[str]:
    if len(groups) == 1:
        return [groups[0][0]] * count
    total = sum(size for _, size in groups)
    if count == total:
        return [system_id for system_id, size in groups for _ in range(size)]
    task_id = str(task.get("task_id", "")).lower()
    matches = [
        (system_id, size) for system_id, size in groups
        if system_id.lower() in task_id and size == count
    ]
    if len(matches) == 1:
        return [matches[0][0]] * count
    if count == groups[0][1] and len({size for _, size in groups}) == 1:
        # System-view task IDs sometimes use a normalized spelling. Match the
        # distinctive control/lesion or TBA label before failing closed.
        aliases = {
            "trex1_control": "trex_control",
            "trex1_8oxog": "trex_corrected_8oxoG",
            "trex1_corrected_8oxog": "trex_corrected_8oxoG",
            "tbae6_ddc16": "TBAE6_ddC16",
        }
        for token, system_id in aliases.items():
            if token in task_id and any(system_id == value for value, _ in groups):
                return [system_id] * count
    raise ValueError(
        f"cannot assign {count} replicas in {task.get('task_id')} to systems"
    )


def _resize_tasks(
    tasks: Sequence[Mapping[str, object]],
    *,
    target_frames_per_replica: Optional[int],
    target_replica_count: Optional[int],
    groups: Sequence[tuple[str, int]],
    frame_interval_ns: float,
) -> list[Dict[str, object]]:
    rows = []
    for raw in tasks:
        row = deepcopy(dict(raw))
        source = row.get("source_frames_per_replica")
        if not isinstance(source, list) or not source:
            raise ValueError(f"{row.get('task_id')} has no source-frame list")
        old_source = [int(value) for value in source]
        count = len(old_source)
        if target_replica_count is not None:
            if count != 1:
                raise ValueError(
                    "replica-count expansion is allowed only from a one-replica fixture"
                )
            count = target_replica_count
        new_frame_count = (
            int(target_frames_per_replica)
            if target_frames_per_replica is not None else max(old_source)
        )
        new_source = [new_frame_count] * count
        for field in (
            "source_frames_per_replica",
            "global_stride_raw_source_frames_per_replica",
            "coordinate_cache_raw_source_frames_per_replica",
        ):
            if field == "source_frames_per_replica" or field in row:
                row[field] = list(new_source)
        for field in (
            "frame_intervals_ns_per_replica",
            "global_stride_raw_frame_intervals_ns_per_replica",
        ):
            if field in row:
                row[field] = [frame_interval_ns] * count
        if "source_time_spans_ns_per_replica" in row:
            row["source_time_spans_ns_per_replica"] = [
                (new_frame_count - 1) * frame_interval_ns
            ] * count
        old_maximum = int(row.get("maximum_frames_per_replica", max(old_source)))
        row["maximum_frames_per_replica"] = (
            new_frame_count
            if old_maximum >= max(old_source)
            else min(old_maximum, new_frame_count)
        )
        row["system_ids_per_replica"] = _system_ids(row, groups, count)
        rows.append(row)
    return rows


def _add_coordinate_cache(
    tasks: list[Dict[str, object]],
    *,
    topology_atom_count: int,
    groups: Sequence[tuple[str, int]],
    source_frame_count: int,
) -> None:
    if any(row.get("module_id") == "coordinate_cache" for row in tasks):
        return
    replica_count = sum(size for _, size in groups)
    worker_memory = min(
        4.0, max(0.5, 0.5 * (topology_atom_count / 85_206.0) ** 0.5)
    )
    tasks.insert(0, {
        "task_id": "preprocessing:coordinate_cache",
        "workflow_id": "coordinate_cache",
        "module_id": "coordinate_cache",
        "task_scope": "continuous_unwrap_working_cache",
        "dependency_stage": 0,
        "effective_cpu_cap": replica_count,
        "intrinsic_cpu_cap": replica_count,
        "parallel_execution_model": "replica_worker_exact_global_reducer_v1",
        "parallel_worker_count": replica_count,
        "estimated_peak_memory_gib_per_parallel_worker": worker_memory,
        "reducer_memory_gib": worker_memory,
        "parallel_memory_model": "max(reducer, simultaneously_active_replica_workers)",
        "source_frames_per_replica": [source_frame_count] * replica_count,
        "system_ids_per_replica": [
            system_id for system_id, size in groups for _ in range(size)
        ],
        "minimum_frames_per_replica": source_frame_count,
        "maximum_frames_per_replica": source_frame_count,
        "cpu_seconds_per_physical_frame": (
            (232.39 / 700.0) * (topology_atom_count / 85_206.0) * 1.5
        ),
        "fixed_cpu_hours": 0.0,
        "estimated_peak_memory_gib": worker_memory * replica_count,
        "priority_weight": 100.0,
        "member_observation_multiplier": 1,
        "balance_group": "preprocessing:coordinate_cache:all_frames",
        "replica_sampling_mode": "independent_all_available",
        "calibration_status": "completed_trex_cache_pilot_linear",
    })


def _parallelize_replica_work(tasks: Sequence[Mapping[str, object]]) -> list[Dict[str, object]]:
    rows = []
    for raw in tasks:
        row = annotate_task_parallelism(raw)
        if row.get("module_id") not in PARALLEL_MODULES:
            rows.append(row)
            continue
        source = row.get("source_frames_per_replica")
        if not isinstance(source, list) or not source:
            raise ValueError(f"{row.get('task_id')} has no replica counts")
        workers = len(source)
        old_workers = row.get("parallel_worker_count")
        if (
            isinstance(old_workers, int) and not isinstance(old_workers, bool)
            and old_workers > 0
        ):
            per_worker_memory = float(row.get(
                "estimated_peak_memory_gib_per_parallel_worker",
                float(row.get("estimated_peak_memory_gib", 1.0)) / old_workers,
            ))
        else:
            per_worker_memory = float(row.get("estimated_peak_memory_gib", 1.0))
        per_worker_memory = max(1.0, per_worker_memory)
        row.update({
            "effective_cpu_cap": workers,
            "intrinsic_cpu_cap": workers,
            "parallel_execution_model": "replica_worker_exact_global_reducer_v1",
            "parallel_worker_count": workers,
            "estimated_peak_memory_gib_per_parallel_worker": per_worker_memory,
            "reducer_memory_gib": float(row.get("reducer_memory_gib", per_worker_memory)),
            "parallel_memory_model": "max(reducer, simultaneously_active_replica_workers)",
            "estimated_peak_memory_gib": per_worker_memory * workers,
        })
        rows.append(row)
    return rows


def _reprice_measured_work(
    tasks: Sequence[Mapping[str, object]],
    *,
    models: Mapping[str, Mapping[str, object]],
    topology_atom_count: int,
    maximum_hydrogen_bond_endpoint_count: int,
) -> list[Dict[str, object]]:
    rows = []
    for raw in tasks:
        row = dict(raw)
        module_id = str(row.get("module_id", ""))
        model = models.get(module_id)
        if model is None:
            rows.append(row)
            continue
        source = row.get("source_frames_per_replica")
        if not isinstance(source, list) or not source:
            raise ValueError(f"{row.get('task_id')} has no source frames")
        proxy_count = (
            maximum_hydrogen_bond_endpoint_count
            if module_id == "hydrogen_bond_discovery"
            else topology_atom_count
        )
        terms = predict_size_length_cpu_terms(
            module_id,
            model,
            source_topology_atom_frame_count=(
                topology_atom_count * sum(int(value) for value in source)
            ),
            selected_work_proxy_count_per_frame=proxy_count,
            campaign_time_safety_factor=1.5,
        )
        row.update({
            "fixed_cpu_hours": terms["fixed_cpu_hours"],
            "cpu_seconds_per_physical_frame": terms[
                "cpu_seconds_per_physical_frame"
            ],
            "runtime_workload_scaling": terms["workload_basis"],
            "calibration_status": "heldout_validated_size_length_work_model",
            "calibration_source_policy": (
                "heldout_validated_size_length_selected_work_model"
            ),
            "censored_wall_lower_bound_points": [],
        })
        row.pop("power_law_cost_model", None)
        rows.append(row)
    return rows


def _trim_sweep(report: Mapping[str, object]) -> Dict[str, object]:
    result = deepcopy(dict(report))
    minimum_containers = [result.get("sweet_spot")]
    operational = result.get("operational_balance")
    if isinstance(operational, dict):
        minimum_containers.append(operational)
    for container in minimum_containers:
        if isinstance(container, dict):
            minimums = container.get("scientific_minimum_multiples")
        else:
            minimums = None
        if isinstance(minimums, dict) and isinstance(minimums.get("tasks"), list):
            tasks = minimums.pop("tasks")
            minimums["lowest_task_multiples"] = sorted(
                tasks,
                key=lambda row: (row["multiple_of_scientific_minimum"], row["task_id"]),
            )[:10]
    return result


def _scenario(
    name: str,
    plan_path: Path,
    *,
    models: Mapping[str, Mapping[str, object]],
    topology_atom_count: int,
    maximum_hydrogen_bond_endpoint_count: int,
    groups: Sequence[tuple[str, int]],
    target_frames_per_replica: Optional[int] = None,
    target_replica_count: Optional[int] = None,
    frame_interval_ns: float,
    add_coordinate_cache: bool = False,
    required_stride_by_balance_group: Optional[Mapping[str, int]] = None,
) -> Dict[str, object]:
    tasks = _resize_tasks(
        _load_plan(plan_path),
        target_frames_per_replica=target_frames_per_replica,
        target_replica_count=target_replica_count,
        groups=groups,
        frame_interval_ns=frame_interval_ns,
    )
    if add_coordinate_cache:
        _add_coordinate_cache(
            tasks,
            topology_atom_count=topology_atom_count,
            groups=groups,
            source_frame_count=(
                target_frames_per_replica
                if target_frames_per_replica is not None
                else max(tasks[0]["source_frames_per_replica"])
            ),
        )
    tasks = apply_scientific_minimums_to_tasks(
        tasks, load_scientific_minimums(None)
    )
    if required_stride_by_balance_group:
        for task in tasks:
            group = str(task.get("balance_group") or task.get("task_id"))
            if group in required_stride_by_balance_group:
                task["required_integer_stride"] = int(
                    required_stride_by_balance_group[group]
                )
    tasks = _parallelize_replica_work(tasks)
    tasks = _reprice_measured_work(
        tasks,
        models=models,
        topology_atom_count=topology_atom_count,
        maximum_hydrogen_bond_endpoint_count=(
            maximum_hydrogen_bond_endpoint_count
        ),
    )
    report = plan_node_sweep(
        tasks,
        cpus_per_node=44,
        memory_gib_per_node=185.0,
        maximum_nodes=16,
        maximum_wall_hours=168.0,
        information_plateau_fraction=0.95,
        information_plateau_tolerance_fraction=0.0,
        planning_utilization=0.85,
        pilot_budget_fraction=0.05,
        finalization_headroom_fraction=0.05,
        memory_safety_factor=1.5,
        memory_overhead_gib=1.0,
        minimum_scheduler_memory_gib=0.0,
        coordinate_cache_full_scan_fraction=1.0,
        # With a required full cache scan, overall stride one contains every
        # method-local exact integer-stride choice available to a coarser
        # overall stream. Coarser overall candidates cannot improve information
        # and do not reduce scan work, so the sweep fixes this dominated axis.
        overall_stride_candidate_strides=[1],
        planning_processes=8,
    )
    result = {
        "scenario_id": name,
        "source_plan": str(plan_path),
        "source_plan_sha256": _sha256(plan_path),
        "topology_atom_count": topology_atom_count,
        "maximum_hydrogen_bond_endpoint_count": (
            maximum_hydrogen_bond_endpoint_count
        ),
        "system_replica_counts": dict(groups),
        "source_frames_per_replica": (
            target_frames_per_replica
            if target_frames_per_replica is not None else "from source plan"
        ),
        "frame_interval_ns": frame_interval_ns,
        "sweep": _trim_sweep(report),
    }
    operational = result["sweep"]["operational_balance"][
        "recommended_curve_point"
    ]
    print(json.dumps({
        "scenario_complete": name,
        "recommended_nodes": operational["requested_nodes"],
        "planned_makespan_hours": operational["planned_makespan_hours"],
        "fraction_of_best_information": operational[
            "fraction_of_best_information"
        ],
        "balanced_operational_score": operational[
            "balanced_operational_score"
        ],
    }, sort_keys=True), flush=True)
    return result


def _shared_coarser_operational_stride_contract(
    d_scenario: Mapping[str, object],
    t_scenario: Mapping[str, object],
) -> Dict[str, int]:
    """Use the coarser independent stride for each shared TOP1 group."""

    d_point = d_scenario["sweep"]["operational_balance"][
        "recommended_curve_point"
    ]
    t_point = t_scenario["sweep"]["operational_balance"][
        "recommended_curve_point"
    ]
    d_contract = d_point["selected_stride_by_balance_group"]
    t_contract = t_point["selected_stride_by_balance_group"]
    return {
        group: max(int(d_contract[group]), int(t_contract[group]))
        for group in sorted(set(d_contract) & set(t_contract))
    }


def _top1_joint_stride_analysis(
    d_scenario: Mapping[str, object],
    t_scenario: Mapping[str, object],
) -> Dict[str, object]:
    """Find an operationally balanced D/T node pair with identical strides."""

    def floor_summary(
        scenario: Mapping[str, object], node_count: int,
    ) -> Optional[Dict[str, object]]:
        sweep = scenario["sweep"]
        operational = sweep["operational_balance"]
        if int(operational["recommended_node_count"]) == node_count:
            return dict(operational["scientific_minimum_multiples"])
        for row in sweep["threshold_sensitivity"]:
            if int(row["first_qualifying_node_count"]) == node_count:
                return {
                    "task_count": row["scientific_minimum_task_count"],
                    "mean_multiple_of_scientific_minimum": row[
                        "mean_multiple_of_scientific_minimum"
                    ],
                    "median_multiple_of_scientific_minimum": row[
                        "median_multiple_of_scientific_minimum"
                    ],
                    "minimum_multiple_of_scientific_minimum": row[
                        "minimum_multiple_of_scientific_minimum"
                    ],
                    "maximum_multiple_of_scientific_minimum": row[
                        "maximum_multiple_of_scientific_minimum"
                    ],
                    "source_limited_task_count": row[
                        "source_limited_task_count"
                    ],
                }
        return None

    d_curve = d_scenario["sweep"]["curve"]
    t_curve = t_scenario["sweep"]["curve"]
    candidates = []
    for d_point in d_curve:
        for t_point in t_curve:
            if (
                d_point.get("feasibility_status") != "feasible"
                or t_point.get("feasibility_status") != "feasible"
                or d_point.get("above_task_inventory_useful_node_ceiling")
                or t_point.get("above_task_inventory_useful_node_ceiling")
                or d_point.get("balanced_operational_score") is None
                or t_point.get("balanced_operational_score") is None
            ):
                continue
            d_contract = d_point["selected_stride_by_balance_group"]
            t_contract = t_point["selected_stride_by_balance_group"]
            shared_groups = sorted(set(d_contract) & set(t_contract))
            mismatches = [
                group for group in shared_groups
                if int(d_contract[group]) != int(t_contract[group])
            ]
            if mismatches:
                continue
            candidates.append({
                "d_requested_nodes": d_point["requested_nodes"],
                "t_requested_nodes": t_point["requested_nodes"],
                "d_planned_makespan_hours": d_point["planned_makespan_hours"],
                "t_planned_makespan_hours": t_point["planned_makespan_hours"],
                "concurrent_elapsed_hours_after_start": max(
                    float(d_point["planned_makespan_hours"]),
                    float(t_point["planned_makespan_hours"]),
                ),
                "d_fraction_of_best_information": d_point[
                    "fraction_of_best_information"
                ],
                "t_fraction_of_best_information": t_point[
                    "fraction_of_best_information"
                ],
                "d_balanced_operational_score": d_point[
                    "balanced_operational_score"
                ],
                "t_balanced_operational_score": t_point[
                    "balanced_operational_score"
                ],
                "minimum_component_operational_score": min(
                    float(d_point["balanced_operational_score"]),
                    float(t_point["balanced_operational_score"]),
                ),
                "mean_component_operational_score": (
                    float(d_point["balanced_operational_score"])
                    + float(t_point["balanced_operational_score"])
                ) / 2.0,
                "combined_information_fraction": (
                    float(d_point["fraction_of_best_information"])
                    + float(t_point["fraction_of_best_information"])
                ) / 2.0,
                "total_requested_nodes": (
                    int(d_point["requested_nodes"])
                    + int(t_point["requested_nodes"])
                ),
                "shared_balance_group_count": len(shared_groups),
                "shared_stride_by_balance_group": {
                    group: int(d_contract[group]) for group in shared_groups
                },
            })
    if candidates:
        minimum_elapsed = min(
            float(row["concurrent_elapsed_hours_after_start"])
            for row in candidates
        )
        minimum_total_nodes = min(
            int(row["total_requested_nodes"]) for row in candidates
        )
        for row in candidates:
            information_regret = 1.0 - float(
                row["combined_information_fraction"]
            )
            wait_regret = (
                float(row["concurrent_elapsed_hours_after_start"])
                - minimum_elapsed
            ) / max(1.0e-12, 168.0 - minimum_elapsed)
            node_regret = (
                int(row["total_requested_nodes"]) - minimum_total_nodes
            ) / max(1, 32 - minimum_total_nodes)
            row["joint_balanced_operational_score"] = 1.0 - (
                (
                    information_regret ** 2
                    + wait_regret ** 2
                    + node_regret ** 2
                ) / 3.0
            ) ** 0.5
            row["pareto_efficient"] = not any(
                other is not row
                and float(other["combined_information_fraction"])
                >= float(row["combined_information_fraction"])
                and float(other["concurrent_elapsed_hours_after_start"])
                <= float(row["concurrent_elapsed_hours_after_start"])
                and int(other["total_requested_nodes"])
                <= int(row["total_requested_nodes"])
                and (
                    float(other["combined_information_fraction"])
                    > float(row["combined_information_fraction"])
                    or float(other["concurrent_elapsed_hours_after_start"])
                    < float(row["concurrent_elapsed_hours_after_start"])
                    or int(other["total_requested_nodes"])
                    < int(row["total_requested_nodes"])
                )
                for other in candidates
            )
        candidates.sort(key=lambda row: (
            -float(row["joint_balanced_operational_score"]),
            float(row["concurrent_elapsed_hours_after_start"]),
            int(row["total_requested_nodes"]),
        ))
        recommended = candidates[0]
        recommended["d_scientific_minimum_multiples"] = floor_summary(
            d_scenario, int(recommended["d_requested_nodes"])
        )
        recommended["t_scientific_minimum_multiples"] = floor_summary(
            t_scenario, int(recommended["t_requested_nodes"])
        )
    return {
        "technical_status": "complete" if candidates else "no_exact_pair",
        "required_contract": (
            "every balance group present in both TOP1 components uses the same "
            "effective raw integer stride"
        ),
        "selection_rule": (
            "highest equal-weight closeness to full combined information, the "
            "shortest time until both components finish after their allocations "
            "start, and the smallest total request within the two 16-node caps"
        ),
        "queue_wait_included": False,
        "candidate_pair_count": len(candidates),
        "recommended_pair": candidates[0] if candidates else None,
        "fallback_if_no_exact_pair": (
            "freeze each shared group to the coarser independently selected "
            "stride and rerun both planners before accepting a TOP1 allocation"
        ) if not candidates else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top1-d-plan", type=Path, required=True)
    parser.add_argument("--top1-t-plan", type=Path, required=True)
    parser.add_argument("--tba-plan", type=Path, required=True)
    parser.add_argument("--trex-plan", type=Path, required=True)
    parser.add_argument("--thrombin-plan", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-complete-sweeps", action="store_true",
        help=(
            "reuse the output's complete, hash-matched v2 curves when only "
            "deterministic joint post-processing changed"
        ),
    )
    args = parser.parse_args()

    model_artifact = validate_size_length_models(
        json.loads(args.models.read_text(encoding="utf-8"))
    )
    models = model_artifact["models"]
    if args.reuse_complete_sweeps:
        existing_bytes = args.output.read_bytes()
        existing_sha256 = hashlib.sha256(existing_bytes).hexdigest()
        existing = json.loads(existing_bytes)
        if existing.get("evidence_schema") != "salsbury-apollo-node-sweet-spots-v2":
            raise ValueError("only complete v2 sweep evidence can be reused")
        if existing.get("unexpected_error_count") != 0:
            raise ValueError("sweep evidence with unexpected errors cannot be reused")
        if existing.get("comparison_horizon_hours") != 168.0:
            raise ValueError("reused sweep does not use the 168-hour hard cap")
        hardware = existing.get("hardware", {})
        if (
            hardware.get("maximum_nodes_per_campaign") != 16
            or hardware.get("cpus_per_node") != 44
            or hardware.get("memory_gib_per_node") != 185.0
        ):
            raise ValueError("reused sweep does not match the Apollo node limits")
        if existing.get("model_sha256") != _sha256(args.models):
            raise ValueError("reused sweep model hash is stale")
        expected_hashes = {
            "top1_edu_d_component": _sha256(args.top1_d_plan),
            "top1_edu_t_component": _sha256(args.top1_t_plan),
            "tba_current": _sha256(args.tba_plan),
            "trex_current_250ns": _sha256(args.trex_plan),
            "thrombin_current_100ns": _sha256(args.thrombin_plan),
            "trex_future_1us": _sha256(args.trex_plan),
            "thrombin_future_1us": _sha256(args.thrombin_plan),
        }
        scenarios = existing.get("scenarios")
        components = existing.get("top1_harmonization", {}).get("components")
        if not isinstance(scenarios, list) or not isinstance(components, list):
            raise ValueError("reused sweep lacks primary or harmonized curves")
        observed_hashes = {
            row["scenario_id"]: row["source_plan_sha256"] for row in scenarios
        }
        if observed_hashes != expected_hashes:
            raise ValueError("reused sweep source-plan hashes are stale")
        existing["top1_independent_stride_analysis"] = (
            _top1_joint_stride_analysis(scenarios[0], scenarios[1])
        )
        existing["top1_joint_stride_analysis"] = (
            _top1_joint_stride_analysis(components[0], components[1])
        )
        existing["planning_policy"].update({
            "slurm_requested_wall_hours": 168.0,
            "slurm_requested_time": "7-00:00:00",
            "planned_makespan_role": (
                "expected dependency-chain runtime after allocation start; the "
                "full 168-hour request remains unchanged"
            ),
        })
        existing["generated_at"] = datetime.now(timezone.utc).isoformat()
        existing["postprocessing_reuse"] = {
            "source_evidence_sha256": existing_sha256,
            "scope": "joint TOP1 operational ranking only",
            "curve_values_recomputed": False,
            "reuse_gate": (
                "schema, zero errors, hardware, hard caps, model hash, and every "
                "source-plan hash matched"
            ),
        }
        args.output.write_text(
            json.dumps(existing, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "output": str(args.output),
            "sha256": _sha256(args.output),
            "reused_complete_sweeps_from_sha256": existing_sha256,
            "unexpected_error_count": 0,
        }, indent=2, sort_keys=True))
        return 0
    scenarios = [
        _scenario(
            "top1_edu_d_component", args.top1_d_plan,
            models=models, topology_atom_count=95_207,
            maximum_hydrogen_bond_endpoint_count=402,
            groups=(("D0", 3), ("D1", 3), ("T0-DNA_CONTEXT", 3), ("T1-DNA_CONTEXT", 3)),
            frame_interval_ns=0.05,
        ),
        _scenario(
            "top1_edu_t_component", args.top1_t_plan,
            models=models, topology_atom_count=104_300,
            maximum_hydrogen_bond_endpoint_count=2_348,
            groups=(("T0", 3), ("T1", 3)), frame_interval_ns=0.05,
        ),
        _scenario(
            "tba_current", args.tba_plan,
            models=models, topology_atom_count=15_148,
            maximum_hydrogen_bond_endpoint_count=212,
            groups=(("TBA_K", 4), ("TBAF6_K", 4), ("TBAE3F3_K", 4), ("TBAE6_K", 4), ("TBAE6_ddC16", 4)),
            frame_interval_ns=0.1,
        ),
        _scenario(
            "trex_current_250ns", args.trex_plan,
            models=models, topology_atom_count=85_206,
            maximum_hydrogen_bond_endpoint_count=172,
            groups=(("trex_control", 3), ("trex_corrected_8oxoG", 3)),
            target_frames_per_replica=24_700, frame_interval_ns=0.01,
            add_coordinate_cache=True,
        ),
        _scenario(
            "thrombin_current_100ns", args.thrombin_plan,
            models=models, topology_atom_count=47_645,
            maximum_hydrogen_bond_endpoint_count=85,
            groups=(("thrombin", 60),), target_frames_per_replica=10_000,
            target_replica_count=60, frame_interval_ns=0.01,
        ),
        _scenario(
            "trex_future_1us", args.trex_plan,
            models=models, topology_atom_count=85_206,
            maximum_hydrogen_bond_endpoint_count=172,
            groups=(("trex_control", 3), ("trex_corrected_8oxoG", 3)),
            target_frames_per_replica=100_000, frame_interval_ns=0.01,
            add_coordinate_cache=True,
        ),
        _scenario(
            "thrombin_future_1us", args.thrombin_plan,
            models=models, topology_atom_count=47_645,
            maximum_hydrogen_bond_endpoint_count=85,
            groups=(("thrombin", 63),), target_frames_per_replica=100_000,
            target_replica_count=63, frame_interval_ns=0.01,
        ),
    ]
    for scenario in scenarios:
        sweep = scenario["sweep"]
        for point in sweep["curve"]:
            if int(point["requested_nodes"]) > 16:
                raise ValueError("node sweep exceeded the absolute 16-node cap")
            makespan = point.get("planned_makespan_hours")
            if (
                point.get("feasibility_status") == "feasible"
                and makespan is not None
                and float(makespan) > 168.0 + 1.0e-9
            ):
                raise ValueError(
                    f"{scenario['scenario_id']} has a feasible point above the "
                    "absolute 168-hour wall-time cap"
                )
    top1_independent_stride_analysis = _top1_joint_stride_analysis(
        scenarios[0], scenarios[1]
    )
    shared_top1_strides = _shared_coarser_operational_stride_contract(
        scenarios[0], scenarios[1]
    )
    top1_harmonized_scenarios = [
        _scenario(
            "top1_edu_d_component_harmonized", args.top1_d_plan,
            models=models, topology_atom_count=95_207,
            maximum_hydrogen_bond_endpoint_count=402,
            groups=(("D0", 3), ("D1", 3), ("T0-DNA_CONTEXT", 3), ("T1-DNA_CONTEXT", 3)),
            frame_interval_ns=0.05,
            required_stride_by_balance_group=shared_top1_strides,
        ),
        _scenario(
            "top1_edu_t_component_harmonized", args.top1_t_plan,
            models=models, topology_atom_count=104_300,
            maximum_hydrogen_bond_endpoint_count=2_348,
            groups=(("T0", 3), ("T1", 3)), frame_interval_ns=0.05,
            required_stride_by_balance_group=shared_top1_strides,
        ),
    ]
    for scenario in top1_harmonized_scenarios:
        for point in scenario["sweep"]["curve"]:
            if int(point["requested_nodes"]) > 16:
                raise ValueError("node sweep exceeded the absolute 16-node cap")
            makespan = point.get("planned_makespan_hours")
            if (
                point.get("feasibility_status") == "feasible"
                and makespan is not None
                and float(makespan) > 168.0 + 1.0e-9
            ):
                raise ValueError(
                    f"{scenario['scenario_id']} has a feasible point above the "
                    "absolute 168-hour wall-time cap"
                )
    top1_joint_stride_analysis = _top1_joint_stride_analysis(
        top1_harmonized_scenarios[0], top1_harmonized_scenarios[1]
    )
    result = {
        "evidence_schema": "salsbury-apollo-node-sweet-spots-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "technical_status": "complete",
        "scientific_status": "resource planning only; scientific validity not evaluated",
        "execution_started": False,
        "jobs_submitted": False,
        "hardware": {
            "cluster": "DEAC Apollo",
            "cpus_per_node": 44,
            "memory_gib_per_node": 185.0,
            "maximum_nodes_per_campaign": 16,
            "maximum_parallel_cpus": 704,
            "maximum_memory_gib": 2_960.0,
            "absolute_node_cap": 16,
            "absolute_wall_hours_cap": 168.0,
        },
        "comparison_horizon_hours": 168.0,
        "model_path": str(args.models),
        "model_sha256": _sha256(args.models),
        "model_source_evidence_sha256": model_artifact[
            "source_evidence_sha256"
        ],
        "planning_policy": {
            "information_plateau_fraction": 0.95,
            "information_plateau_tolerance_fraction": 0.0,
            "slurm_requested_wall_hours": 168.0,
            "slurm_requested_time": "7-00:00:00",
            "planned_makespan_role": (
                "expected dependency-chain runtime after allocation start; the "
                "full 168-hour request remains unchanged"
            ),
            "sweet_spot_primary_cost": (
                "planned dependency-chain makespan within a 168-hour Slurm limit"
            ),
            "sweet_spot_secondary_cost": "requested node count",
            "reserved_node_hours_role": "reported for accounting; not the selection target",
            "top1_d_t_stride_policy": (
                "identical effective raw integer strides for every balance group "
                "present in both component plans"
            ),
            "structural_qc": "one worker per replica",
            "hydrogen_bonds": "spatial endpoint selected-work proxy",
            "ion_atmosphere": "topology-atom selected-work proxy",
            "coordinate_cache": (
                "full sequential scan within each replica where retained structural "
                "analyses require continuous unwrapping; ion atmosphere does not"
            ),
            "overall_stride_candidate_strides": [1],
            "overall_stride_reason": (
                "with full cache scanning, stride one preserves every downstream "
                "exact integer-stride option and coarser overall streams cannot "
                "increase information or reduce scan work"
            ),
            "campaign_time_safety_factor": 1.5,
            "model_residual_safety_factor": 1.5,
            "scheduler_memory_safety_factor": 1.5,
            "scheduler_memory_overhead_gib_per_task": 1.0,
        },
        "scenarios": scenarios,
        "top1_independent_stride_analysis": top1_independent_stride_analysis,
        "top1_harmonization": {
            "derivation": (
                "for every shared balance group, freeze both components to the "
                "coarser stride from their independent operational choices"
            ),
            "required_stride_by_balance_group": shared_top1_strides,
            "components": top1_harmonized_scenarios,
        },
        "top1_joint_stride_analysis": top1_joint_stride_analysis,
        "unexpected_error_count": 0,
        "interpretation": (
            "The current scenarios use verified available inputs. Future TREX and "
            "thrombin rows are capacity projections for the stated size, replica "
            "count, and length. They are not authorization to submit campaigns and "
            "do not establish convergence, kinetics, mechanism, or biological meaning."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": _sha256(args.output),
        "primary_scenario_count": len(scenarios),
        "harmonized_top1_component_count": len(top1_harmonized_scenarios),
        "unexpected_error_count": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
