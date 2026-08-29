"""Whole-campaign resource planning across base and conformational workflows."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

from .automatic_sampling import _apply_campaign_direct_allocations
from .execution_adapters import load_slurm_profile
from .frame_sampling import (
    integer_stride_for_budget,
    integer_stride_selected_count,
)
from .manifests import load_json, validate_project
from .resource_planning import (
    ResourcePlanningError,
    alternative_clustering_fit_profiles,
    plan_campaign_resource_budget,
    plan_global_stride_projection_coupled_campaign_resource_budget,
    recommend_scientifically_valid_task_subset,
)
from .resource_calibrations import (
    ResourceCalibrationError, load_resource_calibration_catalog,
)
from .scientific_sampling import (
    apply_scientific_minimums_to_tasks,
    load_scientific_minimums,
    profile_contract,
    required_frames_per_replica,
    scientific_sampling_profile,
)


class CampaignPlanningError(ValueError):
    """Raised when a prepared workflow cannot form one bounded campaign DAG."""

    def __init__(
        self, message: str, *, plan: Optional[Mapping[str, object]] = None
    ) -> None:
        super().__init__(message)
        self.plan = deepcopy(dict(plan)) if plan is not None else None


def _scientific_task_contract(
    module_id: str,
    source_counts: Sequence[int],
    *,
    system_ids_per_replica: Optional[Sequence[str]] = None,
    frame_intervals_ns_per_replica: Optional[Sequence[float]] = None,
    source_time_spans_ns_per_replica: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Return an attainable task minimum plus the unabridged standard gate."""

    profile = scientific_sampling_profile(module_id)
    required = required_frames_per_replica(
        profile,
        system_ids_per_replica=system_ids_per_replica,
        replica_count=(
            None if system_ids_per_replica is not None else len(source_counts)
        ),
        source_frames_per_replica=(
            source_counts if frame_intervals_ns_per_replica is not None else None
        ),
        frame_intervals_ns_per_replica=frame_intervals_ns_per_replica,
        source_time_spans_ns_per_replica=source_time_spans_ns_per_replica,
    )
    attainable = min(required, max(int(value) for value in source_counts))
    return {
        "scientific_sampling_requirements": profile_contract(profile),
        "scientific_minimum_frames_per_replica": required,
        "attainable_scientific_minimum_frames_per_replica": attainable,
        "minimum_frame_role": "standard_scientific_raw_coverage",
        "minimum_frame_interpretation": (
            "Method-specific sample-count floor plus any applicable configured "
            "lag-pair or temporal-resolution requirement; trajectory duration "
            "is provenance, and the planner does not estimate autocorrelation "
            "times or event rates."
        ),
        **({
            "frame_intervals_ns_per_replica": [
                float(value) for value in frame_intervals_ns_per_replica
            ],
            "source_time_spans_ns_per_replica": [
                float(value) for value in source_time_spans_ns_per_replica or ()
            ],
        } if frame_intervals_ns_per_replica is not None else {}),
    }


def _campaign_infeasibility_detail(plan: Mapping[str, object]) -> str:
    """Explain a failed envelope with the measured shortfall and next bound."""

    reasons = plan.get("infeasibility_reasons", [])
    parts = [str(reason) for reason in reasons] if isinstance(reasons, list) else []
    memory = plan.get("memory_feasibility")
    if isinstance(memory, Mapping) and not bool(
        memory.get("fits_configured_memory", True)
    ):
        configured = float(memory["configured_memory_gib"])
        required = float(memory["minimum_required_memory_gib"])
        recommended = float(memory["recommended_memory_gib"])
        modules = memory.get(
            "configuration_switches_to_disable_to_fit_configured_memory",
            memory.get("modules_to_disable_to_fit_configured_memory", []),
        )
        module_text = (
            ", ".join(str(value) for value in modules)
            if isinstance(modules, list) else "see memory feasibility report"
        )
        parts.append(
            f"configured memory {configured:.3f} GiB; largest enabled technical "
            f"minimum requires a safety-adjusted {required:.3f} GiB request; "
            f"raise the aggregate campaign limit to at least "
            f"{recommended:.0f} GiB or disable: {module_text}"
        )
    maximum_wall = float(plan["maximum_wall_hours_input"])
    science_wall = float(plan["science_budget_wall_hours"])
    minimum_wall = plan.get("minimum_wall_hours_lower_bound")
    required_candidates: list[float] = []
    if isinstance(minimum_wall, (int, float)) and not isinstance(minimum_wall, bool):
        minimum_wall_value = float(minimum_wall)
        parts.append(
            "minimum calibrated critical path "
            f"{minimum_wall_value:.3f} h; science wall allowance "
            f"{science_wall:.3f} h within the {maximum_wall:.3f} h campaign ceiling"
        )
        if science_wall > 0.0:
            required_candidates.append(
                minimum_wall_value * maximum_wall / science_wall
            )
    minimum_cpu = float(plan["minimum_known_cpu_hours"])
    science_cpu = float(plan["science_budget_cpu_hours"])
    cpus = int(plan["maximum_parallel_cpus_input"])
    parts.append(
        f"minimum calibrated CPU {minimum_cpu:.3f} CPU-h; science CPU allowance "
        f"{science_cpu:.3f} CPU-h across {cpus} CPUs"
    )
    if science_cpu > 0.0:
        required_candidates.append(minimum_cpu * maximum_wall / science_cpu)
    if required_candidates:
        required = max(required_candidates)
        parts.append(
            "estimated minimum campaign ceiling with the current utilization and "
            f"reserves is {required:.3f} h; retry with --target-wall-hours "
            f"{max(1, math.ceil(required))} or revise the explicit resource policy"
        )
    return "; ".join(parts) or str(plan.get("feasibility_status", "infeasible"))


def _apply_measured_resource_calibrations(
    tasks: Sequence[Dict[str, object]],
    measured: Mapping[str, Mapping[str, object]],
    *,
    time_safety_factor: float,
    memory_safety_factor: float,
) -> None:
    """Conservatively overlay hash-bound measurements on task estimates."""

    for task in tasks:
        module_id = str(task.get("module_id", ""))
        calibration = measured.get(module_id)
        if calibration is None or task.get("measured_calibration_eligible", True) is False:
            continue
        rate_multiplier = task.get("measured_cpu_rate_multiplier", 1.0)
        memory_multiplier = task.get("measured_memory_multiplier", 1.0)
        for value, label in (
            (rate_multiplier, "measured_cpu_rate_multiplier"),
            (memory_multiplier, "measured_memory_multiplier"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise CampaignPlanningError(
                    f"task {task.get('task_id')} {label} must be finite and positive"
                )
        measured_rate = (
            float(calibration.get(
                "conservative_affine_cpu_seconds_per_frame",
                calibration["conservative_cpu_seconds_per_frame"],
            ))
            * time_safety_factor
            * float(rate_multiplier)
        )
        measured_fixed_hours = (
            float(calibration.get("conservative_fixed_cpu_seconds", 0.0))
            * time_safety_factor
            * float(rate_multiplier)
            / 3600.0
        )
        current_fixed = task.get("fixed_cpu_hours", 0.0)
        if isinstance(current_fixed, (int, float)) and not isinstance(
            current_fixed, bool
        ):
            task["fixed_cpu_hours"] = max(
                float(current_fixed), measured_fixed_hours
            )
        current_rate = task.get("cpu_seconds_per_physical_frame")
        if (
            "power_law_cost_model" not in task
            and isinstance(current_rate, (int, float))
            and not isinstance(current_rate, bool)
        ):
            task["cpu_seconds_per_physical_frame"] = max(
                float(current_rate), measured_rate
            )
        measured_memory = (
            float(calibration["maximum_resident_memory_mib"])
            * memory_safety_factor * float(memory_multiplier) / 1024.0
        )
        current_memory = task.get("estimated_peak_memory_gib")
        if isinstance(current_memory, (int, float)) and not isinstance(current_memory, bool):
            task["estimated_peak_memory_gib"] = max(
                float(current_memory), measured_memory, 1.0
            )
            if (
                "power_law_cost_model" not in task
                and "measured_memory_cost_model" not in task
                and task.get(
                    "measured_memory_observation_scaling_eligible", True
                ) is not False
                and int(calibration["maximum_measured_observation_count"]) > 0
            ):
                task["measured_memory_cost_model"] = {
                    "calibration_observations": int(
                        calibration["maximum_measured_observation_count"]
                    ),
                    "calibration_memory_gib": max(
                        float(current_memory),
                        float(calibration["maximum_resident_memory_mib"])
                        * memory_safety_factor
                        * float(memory_multiplier) / 1024.0,
                    ),
                    "memory_exponent": 0.5,
                    "minimum_observation_scale": 0.1,
                    "workload_scaling_applied": True,
                }
        task["measured_resource_calibration"] = {
            "catalog_sha256": calibration["catalog_sha256"],
            "measurement_count": calibration["measurement_count"],
            "complete_measurement_count": calibration[
                "complete_measurement_count"
            ],
            "censored_timeout_count": calibration["censored_timeout_count"],
            "calibration_evidence_status": calibration[
                "calibration_evidence_status"
            ],
            "censored_timeout_safety_factor": calibration[
                "censored_timeout_safety_factor"
            ],
            "maximum_measured_selected_frame_count": calibration[
                "maximum_measured_selected_frame_count"
            ],
            "maximum_measured_observation_count": calibration[
                "maximum_measured_observation_count"
            ],
            "maximum_measured_resident_memory_mib": calibration[
                "maximum_resident_memory_mib"
            ],
            "cpu_rate_workload_multiplier": float(rate_multiplier),
            "affine_fixed_cpu_hours_after_workload_scaling": (
                measured_fixed_hours
            ),
            "memory_workload_multiplier": float(memory_multiplier),
            "policy": (
                "conservative maximum after the task's declared workload scaling; "
                "measured coverage is not a scientific ceiling"
            ),
        }


def _apply_system_memory_scaling(
    tasks: Sequence[Dict[str, object]], atom_count: int
) -> None:
    """Scale large-system reference memory without claiming linear behavior.

    Most retained legacy estimates describe an 85,206-atom solvated TREX
    system.  The square-root relationship is deliberately more conservative
    than linear atom scaling and has a 10% floor for fixed Python/library
    allocations.  Tasks that already applied the same scaling are left alone.
    """

    reference_atom_count = 85_206
    scale = min(4.0, max(0.1, math.sqrt(atom_count / reference_atom_count)))
    for task in tasks:
        if task.get("memory_size_scaling_applied") is True:
            continue
        current = task.get("estimated_peak_memory_gib")
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            task["reference_peak_memory_gib"] = float(current)
            task["estimated_peak_memory_gib"] = max(1.0, float(current) * scale)
        power_model = task.get("power_law_cost_model")
        if isinstance(power_model, dict):
            calibration_memory = power_model.get("calibration_memory_gib")
            if (
                isinstance(calibration_memory, (int, float))
                and not isinstance(calibration_memory, bool)
            ):
                power_model["reference_calibration_memory_gib"] = float(
                    calibration_memory
                )
                power_model["calibration_memory_gib"] = max(
                    1.0, float(calibration_memory) * scale
                )
        task.setdefault("measured_memory_multiplier", scale)
        task["memory_atom_scale"] = scale
        task["memory_reference_atom_count"] = reference_atom_count
        task["memory_size_scaling_applied"] = True


# CPU seconds per physical source frame. The primary entries are retained
# 30,000-frame/60,000-member Apollo measurements from the authoritative TREX
# oligomer validation. Rates are normalized below for member multiplicity.
_VIEW_MODELS: Mapping[str, Mapping[str, object]] = {
    "generalized_correlation_and_information": {
        "seconds_per_frame_member2": 1.671336 / 30_000,
        "fixed_cpu_hours": 0.0, "memory_gib": 1.0, "stage": 2,
        "priority": 7.0, "calibration": "apollo-oligomer-v20-30k",
    },
    "information_dynamics": {
        "seconds_per_frame_member2": 2.580651 / 30_000,
        "fixed_cpu_hours": 0.0, "memory_gib": 1.0, "stage": 2,
        "priority": 7.0, "calibration": "apollo-oligomer-v20-30k",
    },
    "time_lagged_independent_component_analysis": {
        "seconds_per_frame_member2": 2.214 / 30_000,
        "fixed_cpu_hours": 0.0, "memory_gib": 1.0, "stage": 2,
        "priority": 7.0, "calibration": "apollo-oligomer-v20-30k",
    },
    "pca_fes_basins": {
        "seconds_per_frame_member2": 25.866511 / 30_000,
        "fixed_cpu_hours": 0.0, "memory_gib": 4.0, "stage": 2,
        "priority": 10.0, "calibration": "apollo-oligomer-v20-30k",
    },
    "clustering_kmeans": {
        "seconds_per_frame_member2": 655.688205 / 30_000,
        "fixed_cpu_hours": 0.0, "memory_gib": 2.0, "stage": 2,
        "priority": 8.0, "calibration": "apollo-oligomer-v20-30k",
    },
    "clustering_hdbscan": {
        "seconds_per_frame_member2": 21.367511 / 30_000,
        "fixed_cpu_hours": 0.0, "memory_gib": 2.0, "stage": 2,
        "priority": 7.0, "calibration": "apollo-oligomer-v20-30k",
    },
    "clustering_imwkmeans": {
        "seconds_per_frame_member2": 0.05,
        "fixed_cpu_hours": 0.0, "memory_gib": 4.0, "stage": 2,
        "priority": 7.0, "calibration": "conservative-kmeans-derived-proxy-v1",
    },
    "alternative_clustering": {
        "seconds_per_frame_member2": 268.201244 / 30_000,
        "fixed_cpu_hours": 0.0, "memory_gib": 32.0, "stage": 2,
        "priority": 7.0, "calibration": "apollo-oligomer-v20-30k-fit3000",
    },
    "pald_community_analysis": {
        "seconds_per_frame_member2": 0.0,
        "fixed_cpu_hours": 0.0, "memory_gib": 1.0, "stage": 2,
        "priority": 4.0,
        "calibration": "apollo-single-cpu-pald500-cubic-2026-08-16-v1",
    },
    "representative_frames": {
        "seconds_per_frame_member2": 0.0001,
        "fixed_cpu_hours": 0.01, "memory_gib": 1.0, "stage": 3,
        "priority": 9.0, "calibration": "provisional-bounded-postprocess-v1",
    },
    "state_coordinate_exports": {
        "seconds_per_frame_member2": 0.0001,
        "fixed_cpu_hours": 0.20, "memory_gib": 4.0, "stage": 3,
        "priority": 9.0, "calibration": "provisional-random-seek-export-v1",
    },
    "markov_state_models": {
        "seconds_per_frame_member2": 0.002,
        "fixed_cpu_hours": 0.01, "memory_gib": 2.0, "stage": 3,
        "priority": 5.0, "calibration": "provisional-transition-count-v1",
    },
    "grouped_ml": {
        "seconds_per_frame_member2": 0.01,
        "fixed_cpu_hours": 0.02, "memory_gib": 4.0, "stage": 3,
        "priority": 5.0, "calibration": "provisional-grouped-ml-v1",
    },
}

_MEASURED_VIEW_MODULES = {
    "generalized_correlation_and_information", "information_dynamics",
    "time_lagged_independent_component_analysis", "pca_fes_basins",
    "clustering_kmeans", "clustering_hdbscan", "alternative_clustering",
}

_BASE_DERIVED_MODELS: Mapping[str, Mapping[str, object]] = {
    "correlation_networks": {
        "upstream": "dccm", "seconds_per_frame": 0.0002,
        "fixed_cpu_hours": 0.01, "memory_gib": 2.0, "stage": 2,
        "priority": 6.0,
    },
    "convergence_uncertainty": {
        "upstream": "replica_rmsd_rg", "seconds_per_frame": 0.00005,
        "fixed_cpu_hours": 0.005, "memory_gib": 1.0, "stage": 2,
        "priority": 8.0,
    },
    "rmsf_permutation_inference": {
        "upstream": "pooled_rmsf", "seconds_per_frame": 0.00002,
        "fixed_cpu_hours": 0.01, "memory_gib": 1.0, "stage": 2,
        "priority": 8.0,
    },
}

# Retained single-CPU TREX measurements where available.  Rates are per
# selected physical frame and receive the same safety factor as other tasks.
_AUTOMATIC_CONTEXT_MODELS: Mapping[str, Mapping[str, object]] = {
    "trajectory_features": {
        "seconds_per_frame": 2.014402 / 2_100, "memory_gib": 2.0,
        "stage": 1, "priority": 7.0,
        "calibration": "apollo-trex-v21-trajectory-features-2100f",
    },
    "optional_observables": {
        "seconds_per_frame": 1.67906 / 2_100, "memory_gib": 2.0,
        "stage": 1, "priority": 6.0,
        "calibration": "apollo-trex-v21-observables-2100f",
    },
    "radial_distribution_functions": {
        "seconds_per_frame": 10_524.29013 / 2_004, "memory_gib": 2.0,
        "stage": 1, "priority": 8.0,
        "calibration": "apollo-trex-v21-ion-water-rdf-2004f",
    },
    "nucleic_acid_geometry": {
        "seconds_per_frame": 26.880904 / 2_100, "memory_gib": 2.0,
        "stage": 1, "priority": 8.0,
        "calibration": "apollo-trex-v21-nucleic-geometry-2100f",
    },
    "ion_coordination_geometry": {
        "seconds_per_frame": 653.597373 / 2_100, "memory_gib": 4.0,
        "stage": 1, "priority": 8.0,
        "calibration": "apollo-trex-v21-ion-geometry-2100f",
    },
    "ion_atmosphere": {
        "seconds_per_frame": 0.35, "memory_gib": 4.0,
        "stage": 1, "priority": 8.0,
        "calibration": "top1-species-atmosphere-provisional-v1",
    },
    "nucleic_acid_structure": {
        "seconds_per_frame": 2.0, "memory_gib": 2.0,
        "stage": 1, "priority": 7.0,
        "calibration": "conservative-external-dssr-proxy-v1",
    },
    "scalar_feature_distributions": {
        "seconds_per_frame": 0.002, "memory_gib": 2.0,
        "stage": 2, "priority": 7.0,
        "calibration": "bounded-scalar-postprocess-proxy-v1",
    },
    "scalar_threshold_states": {
        "seconds_per_frame": 0.002, "memory_gib": 2.0,
        "stage": 2, "priority": 7.0,
        "calibration": "bounded-scalar-postprocess-proxy-v1",
    },
}


def _automatic_context_tasks(
    project_path: Path,
    source_counts: Sequence[int],
    *,
    time_safety_factor: float,
    frame_intervals_ns_per_replica: Optional[Sequence[float]] = None,
    source_time_spans_ns_per_replica: Optional[Sequence[float]] = None,
    context_id: Optional[str] = None,
    task_namespace: Optional[str] = None,
    task_scope: str = "automatic_chemical_context",
) -> List[Dict[str, object]]:
    project = load_json(project_path)
    if not isinstance(project, dict):
        raise CampaignPlanningError(
            f"automatic-context project is not an object: {project_path}"
        )
    resolved_context_id = (
        context_id
        if context_id is not None
        else project_path.name[len("project-") : -len(".json")]
    )
    resolved_task_namespace = (
        task_namespace
        if task_namespace is not None
        else f"context:{resolved_context_id}"
    )
    requested = project.get("requested_modules")
    if not isinstance(requested, list):
        raise CampaignPlanningError(
            f"automatic-context project {resolved_context_id} has no requested_modules"
        )
    tasks: List[Dict[str, object]] = []
    for module_id in requested:
        module = str(module_id)
        model = _AUTOMATIC_CONTEXT_MODELS.get(module)
        if model is None:
            continue
        maximum = max(source_counts)
        if module == "nucleic_acid_structure":
            maximum = min(maximum, max(1, math.ceil(1_000 / len(source_counts))))
        balance_group = (
            "automatic_context:ion_scalar:shared_frames"
            if module in {
                "trajectory_features",
                "scalar_feature_distributions", "scalar_threshold_states",
            }
            else f"automatic_context:{module}:shared_frames"
        )
        scientific = _scientific_task_contract(
            module, source_counts,
            frame_intervals_ns_per_replica=frame_intervals_ns_per_replica,
            source_time_spans_ns_per_replica=source_time_spans_ns_per_replica,
        )
        materialized_working_set_gib = float(
            model.get("minimum_materialized_working_set_gib", 0.0)
        )
        tasks.append({
            "task_id": f"{resolved_task_namespace}:{module}",
            "workflow_id": resolved_context_id,
            "module_id": module,
            "task_scope": task_scope,
            "dependency_stage": int(model["stage"]),
            "effective_cpu_cap": 1,
            "source_frames_per_replica": list(source_counts),
            "minimum_frames_per_replica": int(
                scientific["attainable_scientific_minimum_frames_per_replica"]
            ),
            **scientific,
            "maximum_frames_per_replica": maximum,
            "maximum_frame_role": "all source frames subject to campaign resources",
            "cpu_seconds_per_physical_frame": (
                float(model["seconds_per_frame"]) * time_safety_factor
            ),
            "fixed_cpu_hours": 0.0,
            "estimated_peak_memory_gib": max(
                float(model["memory_gib"]), materialized_working_set_gib
            ),
            **({
                "minimum_materialized_working_set_gib": (
                    materialized_working_set_gib
                ),
                "memory_cost_basis": (
                    "fixed_upstream_artifact_materialization_floor"
                ),
            } if materialized_working_set_gib > 0.0 else {}),
            "priority_weight": float(model["priority"]),
            "member_observation_multiplier": 1,
            "balance_group": balance_group,
            "replica_sampling_mode": "balanced_pooled",
            "calibration_status": (
                "completed_trex_measurement"
                if str(model["calibration"]).startswith("apollo-")
                else "provisional_complexity_model"
            ),
            "calibration_id": str(model["calibration"]),
            "measured_calibration_eligible": module not in {
                "scalar_feature_distributions", "scalar_threshold_states",
            },
            "measured_calibration_exclusion_reason": (
                "historical measurements included trajectory-feature recomputation; "
                "the staged implementation now consumes a validated upstream report"
                if module in {
                    "scalar_feature_distributions", "scalar_threshold_states",
                }
                else None
            ),
        })
    return tasks


def _view_member_multiplier(project: Mapping[str, object]) -> int:
    definitions = project.get("definitions")
    common_pca = definitions.get("common_pca") if isinstance(definitions, dict) else None
    symmetry = common_pca.get("symmetry_expansion") if isinstance(common_pca, dict) else None
    if not isinstance(symmetry, dict):
        return 1
    value = symmetry.get("member_count", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CampaignPlanningError("view symmetry member_count is invalid")
    return value


def _view_pca_task(
    *,
    view_id: str,
    project: Mapping[str, object],
    source_counts: Sequence[int],
    system_ids_per_replica: Optional[Sequence[str]] = None,
    frame_intervals_ns_per_replica: Optional[Sequence[float]] = None,
    source_time_spans_ns_per_replica: Optional[Sequence[float]] = None,
    time_safety_factor: float,
) -> Dict[str, object]:
    definitions = project.get("definitions")
    common_pca = definitions.get("common_pca") if isinstance(definitions, dict) else None
    if not isinstance(common_pca, dict):
        raise CampaignPlanningError(f"view {view_id} has no common_pca definition")
    features = common_pca.get("maximum_features")
    if isinstance(features, bool) or not isinstance(features, int) or features <= 0:
        raise CampaignPlanningError(f"view {view_id} maximum_features is invalid")
    multiplier = _view_member_multiplier(project)
    balance_family = (
        view_id.split("__", 1)[1]
        if view_id.startswith("system_") and "__" in view_id
        else view_id
    )
    balance_group = (
        f"per_system:{balance_family}:shared_observations"
        if balance_family != view_id
        else f"view:{view_id}:shared_observations"
    )
    # 11,640.163299 CPU seconds for 30,000 projected physical frames, a
    # 1,500-frame basis, 5,616 Cartesian features, and two canonical member
    # observations per frame.  The TBA validation showed that treating basis
    # fitting as free underestimated the global job by about threefold.  The
    # retained empirical conversion makes one basis frame equivalent to 25
    # projection frames, while keeping the two counts explicit for future
    # multi-point calibration.
    feature_factor = max(0.1, (features / 5_616.0) ** 0.75)
    basis_equivalent_projection_weight = 25.0
    reference_equivalent_frames = 30_000 + (
        basis_equivalent_projection_weight * 1_500
    )
    base_rate = 11_640.163299 / reference_equivalent_frames
    rate = base_rate * feature_factor * (multiplier / 2.0)
    basis_counts = _basis_physical_counts(project, source_counts)
    basis_cpu_hours = (
        base_rate
        * basis_equivalent_projection_weight
        * sum(basis_counts)
        * feature_factor
        * (multiplier / 2.0)
        * time_safety_factor
        / 3600.0
    )
    technical_pilot = max(5, min(50, math.ceil(50 / feature_factor)))
    scientific = _scientific_task_contract(
        "common_pca", source_counts,
        system_ids_per_replica=system_ids_per_replica,
        frame_intervals_ns_per_replica=frame_intervals_ns_per_replica,
        source_time_spans_ns_per_replica=source_time_spans_ns_per_replica,
    )
    minimum = max(
        technical_pilot,
        int(scientific[
            "attainable_scientific_minimum_frames_per_replica"
        ]),
    )
    return {
        "task_id": f"view:{view_id}:common_pca",
        "workflow_id": view_id,
        "module_id": "common_pca",
        "task_scope": "conformational_view",
        "dependency_stage": 3,
        "effective_cpu_cap": 1,
        "source_frames_per_replica": list(source_counts),
        "minimum_frames_per_replica": minimum,
        "technical_pilot_frames_per_replica": technical_pilot,
        **scientific,
        # Do not impose an arbitrary 100,000-frame campaign ceiling here.  The
        # PCA basis sample has its own feature/memory gate in the project
        # definition; this task models the fitted projection and downstream
        # observation set, which may use every source frame when the declared
        # CPU, wall-time, and memory envelope can afford it.
        "maximum_frames_per_replica": max(minimum, max(source_counts)),
        "maximum_frame_role": "all_source_frames_subject_to_campaign_resources",
        "cpu_seconds_per_physical_frame": rate * time_safety_factor,
        "fixed_cpu_hours": basis_cpu_hours,
        "basis_selected_physical_frames_per_replica": basis_counts,
        "basis_selected_physical_frame_count": sum(basis_counts),
        "basis_member_observation_count": sum(basis_counts) * multiplier,
        "basis_equivalent_projection_weight": (
            basis_equivalent_projection_weight
        ),
        "pca_workload_accounting": "basis_plus_projection_v1",
        "estimated_peak_memory_gib": max(
            2.0, min(32.0, 4.0 * feature_factor)
        ),
        "priority_weight": 10.0,
        "member_observation_multiplier": multiplier,
        "balance_group": balance_group,
        "calibration_status": "completed_30k_feature_scaled",
        "calibration_id": "apollo-oligomer-v20-common-pca-30k-feature-scaled",
        "measured_cpu_rate_multiplier": feature_factor * (multiplier / 2.0),
        "measured_memory_multiplier": feature_factor,
        "measured_workload_scaling": {
            "reference_cartesian_feature_count": 5_616,
            "cartesian_feature_count": features,
            "feature_exponent": 0.75,
            "reference_member_observation_multiplier": 2,
            "member_observation_multiplier": multiplier,
            "reference_basis_selected_physical_frame_count": 1_500,
            "reference_projection_selected_physical_frame_count": 30_000,
            "basis_equivalent_projection_weight": (
                basis_equivalent_projection_weight
            ),
        },
    }


def _basis_physical_counts(
    project: Mapping[str, object], source_counts: Sequence[int]
) -> List[int]:
    """Return the fixed PCA basis-fit count for each physical replica."""

    definitions = project.get("definitions")
    common_pca = definitions.get("common_pca") if isinstance(definitions, dict) else None
    if not isinstance(common_pca, dict):
        raise CampaignPlanningError("view has no common_pca definition")
    basis_stride = int(common_pca.get("frame_stride", 1))
    selection = common_pca.get("frame_selection", {"mode": "fixed_stride_v1"})
    if not isinstance(selection, dict):
        raise CampaignPlanningError("PCA frame_selection is invalid")
    mode = selection.get("mode")
    if mode == "fixed_stride_v1":
        stride = basis_stride
    elif mode == "integer_stride_per_replica_v1":
        if basis_stride != 1:
            raise CampaignPlanningError(
                "integer PCA basis selection requires frame_stride 1"
            )
        stride = int(selection["stride"])
    else:
        raise CampaignPlanningError(
            "campaign planning requires an exact integer PCA basis stride"
        )
    return [
        integer_stride_selected_count(int(value), stride)
        for value in source_counts
    ]


def _projected_physical_counts(
    project: Mapping[str, object], source_counts: Sequence[int]
) -> List[int]:
    """Return the current PCA-projection count for each physical replica."""

    definitions = project.get("definitions")
    common_pca = definitions.get("common_pca") if isinstance(definitions, dict) else None
    if not isinstance(common_pca, dict):
        raise CampaignPlanningError("view has no common_pca definition")
    projection_stride = int(common_pca.get("projection_frame_stride", 1))
    selection = common_pca.get(
        "projection_frame_selection", {"mode": "fixed_stride_v1"}
    )
    if not isinstance(selection, dict):
        raise CampaignPlanningError("PCA projection_frame_selection is invalid")
    mode = selection.get("mode")
    if mode == "fixed_stride_v1":
        stride = projection_stride
    elif mode == "integer_stride_per_replica_v1":
        if projection_stride != 1:
            raise CampaignPlanningError(
                "integer PCA projection selection requires projection stride 1"
            )
        stride = int(selection["stride"])
    else:
        raise CampaignPlanningError(
            "campaign iteration requires an exact integer PCA projection stride"
        )
    return [integer_stride_selected_count(int(value), stride) for value in source_counts]


def _view_tasks(
    project_path: Path,
    source_counts: Sequence[int],
    maximum_atom_count: int,
    *,
    time_safety_factor: float,
    frame_intervals_ns_per_replica: Optional[Sequence[float]] = None,
    source_time_spans_ns_per_replica: Optional[Sequence[float]] = None,
) -> List[Dict[str, object]]:
    project = load_json(project_path)
    if not isinstance(project, dict):
        raise CampaignPlanningError(f"view project is not an object: {project_path}")
    view_id = project_path.name[len("project-") : -len(".json")]
    requested = project.get("requested_modules")
    if not isinstance(requested, list) or "common_pca" not in requested:
        return []
    system_manifest_value = project.get("system_manifest")
    system_ids_per_replica: Optional[list[str]] = None
    if isinstance(system_manifest_value, str):
        manifest_path = (project_path.parent / system_manifest_value).resolve(
            strict=False
        )
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            raw_systems = manifest.get("systems") if isinstance(manifest, dict) else None
            if isinstance(raw_systems, list):
                inferred_ids = [
                    str(system["system_id"])
                    for system in raw_systems if isinstance(system, dict)
                    for _ in system.get("replicas", [])
                ]
                if len(inferred_ids) == len(source_counts):
                    system_ids_per_replica = inferred_ids
    pca_task = _view_pca_task(
        view_id=view_id,
        project=project,
        source_counts=source_counts,
        system_ids_per_replica=system_ids_per_replica,
        frame_intervals_ns_per_replica=frame_intervals_ns_per_replica,
        source_time_spans_ns_per_replica=source_time_spans_ns_per_replica,
        time_safety_factor=time_safety_factor,
    )
    multiplier = int(pca_task["member_observation_multiplier"])
    projected_counts = _projected_physical_counts(project, source_counts)
    tasks = [pca_task]
    for module_id in requested:
        if module_id == "common_pca" or module_id not in _VIEW_MODELS:
            continue
        model = _VIEW_MODELS[module_id]
        measured_rate = float(model["seconds_per_frame_member2"])
        fixed_cpu_hours = float(model["fixed_cpu_hours"])
        estimated_memory_gib = float(model["memory_gib"])
        calibration_status = (
            "completed_30k_member_scaled"
            if module_id in _MEASURED_VIEW_MODULES
            else "provisional_complexity_model"
        )
        method_specific: Dict[str, object] = {}
        scientific = _scientific_task_contract(
            module_id, source_counts,
            system_ids_per_replica=system_ids_per_replica,
            frame_intervals_ns_per_replica=frame_intervals_ns_per_replica,
            source_time_spans_ns_per_replica=source_time_spans_ns_per_replica,
        )
        minimum_frames_per_replica = max(
            int(pca_task["minimum_frames_per_replica"]),
            int(scientific["attainable_scientific_minimum_frames_per_replica"]),
        )
        if module_id == "information_dynamics":
            definitions = project.get("definitions")
            definition = (
                definitions.get(module_id)
                if isinstance(definitions, dict) else None
            )
            if not isinstance(definition, dict):
                raise CampaignPlanningError(
                    f"view {view_id} has no information-dynamics definition"
                )
            analyses = definition.get("analyses")
            if not isinstance(analyses, list):
                raise CampaignPlanningError(
                    f"view {view_id} information-dynamics analyses are invalid"
                )
            pair_analyses = {
                "transfer_entropy", "lagged_cross_correlation"
            }.intersection(str(value) for value in analyses)
            if pair_analyses:
                lag_frames = definition.get("lag_frames")
                minimum_pairs = definition.get("minimum_pairs")
                if (
                    isinstance(lag_frames, bool)
                    or not isinstance(lag_frames, int)
                    or lag_frames <= 0
                    or isinstance(minimum_pairs, bool)
                    or not isinstance(minimum_pairs, int)
                    or minimum_pairs <= 0
                ):
                    raise CampaignPlanningError(
                        f"view {view_id} information-dynamics lag/pair "
                        "settings are invalid"
                    )
                available_pairs = multiplier * sum(
                    max(0, count - lag_frames)
                    for count in projected_counts
                )
                if available_pairs < minimum_pairs:
                    required_per_replica = lag_frames + math.ceil(
                        minimum_pairs / (len(projected_counts) * multiplier)
                    )
                    raise CampaignPlanningError(
                        f"view {view_id} information_dynamics can provide only "
                        f"{available_pairs} segment-safe lag pairs from the "
                        "available projections; minimum_pairs is "
                        f"{minimum_pairs}. At least {required_per_replica} "
                        "frames per physical replica are required for the "
                        "current lag, replica count, and member multiplier"
                    )
                required_per_replica = lag_frames + math.ceil(
                    minimum_pairs / (len(projected_counts) * multiplier)
                )
                minimum_frames_per_replica = max(
                    minimum_frames_per_replica, required_per_replica
                )
                method_specific = {
                    "lag_frames": lag_frames,
                    "minimum_lag_pairs": minimum_pairs,
                    "minimum_frames_per_replica_for_lag_pairs": (
                        required_per_replica
                    ),
                    "maximum_available_lag_pairs": available_pairs,
                    "lag_pair_count_basis": (
                        "replica-and-member-segment-safe projected sequences"
                    ),
                }
        elif module_id == "time_lagged_independent_component_analysis":
            definitions = project.get("definitions")
            definition = (
                definitions.get(module_id)
                if isinstance(definitions, dict) else None
            )
            if not isinstance(definition, dict):
                raise CampaignPlanningError(
                    f"view {view_id} has no tICA definition"
                )
            lag_frames = definition.get("lag_frames")
            minimum_pairs = definition.get("minimum_pairs_per_segment")
            if (
                isinstance(lag_frames, bool)
                or not isinstance(lag_frames, int)
                or lag_frames <= 0
                or isinstance(minimum_pairs, bool)
                or not isinstance(minimum_pairs, int)
                or minimum_pairs <= 0
            ):
                raise CampaignPlanningError(
                    f"view {view_id} tICA lag/pair settings are invalid"
                )
            required_per_replica = lag_frames + minimum_pairs
            insufficient = [
                index for index, count in enumerate(projected_counts)
                if count < required_per_replica
            ]
            if insufficient:
                raise CampaignPlanningError(
                    f"view {view_id} tICA requires at least "
                    f"{required_per_replica} projected frames in every physical "
                    "replica for the configured lag and minimum pairs per "
                    f"segment; insufficient replicas are {insufficient}"
                )
            available_pairs = multiplier * sum(
                max(0, count - lag_frames) for count in projected_counts
            )
            minimum_frames_per_replica = max(
                minimum_frames_per_replica, required_per_replica
            )
            method_specific = {
                "lag_frames": lag_frames,
                "minimum_lag_pairs_per_segment": minimum_pairs,
                "minimum_frames_per_replica_for_lag_pairs": (
                    required_per_replica
                ),
                "maximum_available_lag_pairs": available_pairs,
                "lag_pair_count_basis": (
                    "each replica-and-member segment independently satisfies "
                    "the configured lag-pair minimum"
                ),
            }
        elif module_id == "markov_state_models":
            definitions = project.get("definitions")
            definition = (
                definitions.get(module_id)
                if isinstance(definitions, dict) else None
            )
            if not isinstance(definition, dict):
                raise CampaignPlanningError(
                    f"view {view_id} has no MSM definition"
                )
            lag_values = definition.get("lag_frames")
            minimum_transitions = definition.get("minimum_transition_count")
            if (
                not isinstance(lag_values, list)
                or not lag_values
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in lag_values
                )
                or isinstance(minimum_transitions, bool)
                or not isinstance(minimum_transitions, int)
                or minimum_transitions <= 0
            ):
                raise CampaignPlanningError(
                    f"view {view_id} MSM lag/transition settings are invalid"
                )
            maximum_lag = max(int(value) for value in lag_values)
            available_pairs = multiplier * sum(
                max(0, count - maximum_lag) for count in projected_counts
            )
            required_per_replica = maximum_lag + math.ceil(
                minimum_transitions
                / (len(projected_counts) * multiplier)
            )
            if available_pairs < minimum_transitions:
                raise CampaignPlanningError(
                    f"view {view_id} markov_state_models can provide only "
                    f"{available_pairs} segment-safe transition pairs at its "
                    f"largest configured lag; minimum_transition_count is "
                    f"{minimum_transitions}. At least {required_per_replica} "
                    "frames per physical replica are required for the current "
                    "lag set, replica count, and member multiplier"
                )
            minimum_frames_per_replica = max(
                minimum_frames_per_replica, required_per_replica
            )
            method_specific = {
                "lag_frames": sorted(int(value) for value in lag_values),
                "largest_configured_lag_frames": maximum_lag,
                "minimum_transition_count": minimum_transitions,
                "minimum_frames_per_replica_for_transition_pairs": (
                    required_per_replica
                ),
                "maximum_available_transition_pairs_at_largest_lag": (
                    available_pairs
                ),
                "transition_pair_count_basis": (
                    "replica-and-member-segment-safe projected state sequences"
                ),
            }
        elif module_id == "grouped_ml":
            definitions = project.get("definitions")
            definition = (
                definitions.get(module_id)
                if isinstance(definitions, dict) else None
            )
            if not isinstance(definition, dict):
                raise CampaignPlanningError(
                    f"view {view_id} has no grouped-ML definition"
                )
            block_size = definition.get("group_block_size_frames")
            minimum_groups = definition.get("minimum_groups")
            if (
                isinstance(block_size, bool)
                or not isinstance(block_size, int)
                or block_size <= 0
                or isinstance(minimum_groups, bool)
                or not isinstance(minimum_groups, int)
                or minimum_groups <= 0
            ):
                raise CampaignPlanningError(
                    f"view {view_id} grouped-ML group settings are invalid"
                )
            available_groups = sum(
                math.ceil(count / block_size) for count in projected_counts
            )
            groups_per_replica = math.ceil(
                minimum_groups / len(projected_counts)
            )
            required_per_replica = (
                (groups_per_replica - 1) * block_size + 1
            )
            if available_groups < minimum_groups:
                raise CampaignPlanningError(
                    f"view {view_id} grouped_ml can provide only "
                    f"{available_groups} segment/time-block groups from the "
                    "available projections; minimum_groups is "
                    f"{minimum_groups}. At least {required_per_replica} "
                    "frames per physical replica are required for the "
                    "current block size and replica count"
                )
            minimum_frames_per_replica = max(
                minimum_frames_per_replica, required_per_replica
            )
            method_specific = {
                "group_block_size_frames": block_size,
                "minimum_groups": minimum_groups,
                "maximum_available_groups": available_groups,
                "minimum_frames_per_replica_for_groups": required_per_replica,
                "group_count_basis": (
                    "replica-and-segment time blocks; equivalent members do "
                    "not create independent groups"
                ),
            }
        if module_id == "alternative_clustering":
            definitions = project.get("definitions")
            definition = (
                definitions.get(module_id) if isinstance(definitions, dict) else None
            )
            algorithms = definition.get("algorithms") if isinstance(definition, dict) else None
            if not isinstance(algorithms, list) or not algorithms:
                raise CampaignPlanningError(
                    f"view {view_id} lacks alternative-clustering algorithms"
                )
            profiles = alternative_clustering_fit_profiles()
            full_observations = sum(projected_counts) * multiplier
            bundle_id = f"view:{view_id}:alternative_clustering"
            for raw_algorithm in algorithms:
                algorithm = str(raw_algorithm)
                if algorithm not in profiles:
                    raise CampaignPlanningError(
                        f"view {view_id} has no clustering profile for {algorithm}"
                    )
                profile = profiles[algorithm]
                full_fit_only = algorithm in {"ward", "quality_threshold"}
                if full_fit_only and full_observations > int(
                    profile["reference_fit_observation_ceiling"]
                ):
                    continue
                minimum_observations = (
                    full_observations if full_fit_only
                    else min(
                        full_observations,
                        int(profile["minimum_fit_observations"]),
                    )
                )
                minimum_per_replica = max(
                    int(scientific[
                        "attainable_scientific_minimum_frames_per_replica"
                    ]),
                    math.ceil(
                        minimum_observations
                        / (len(projected_counts) * multiplier)
                    ),
                )
                if full_fit_only:
                    minimum_per_replica = max(projected_counts)
                tasks.append({
                    "task_id": f"{bundle_id}:{algorithm}",
                    "workflow_id": view_id,
                    "module_id": "alternative_clustering",
                    "algorithm_id": algorithm,
                    "task_scope": "conformational_view_algorithm_fit",
                    "dependency_stage": int(model["stage"]) + 2,
                    "effective_cpu_cap": 1,
                    "execution_bundle_id": bundle_id,
                    "source_frames_per_replica": list(projected_counts),
                    "minimum_frames_per_replica": minimum_per_replica,
                    **scientific,
                    "minimum_frame_role": (
                        "full_observation_fit_or_skip" if full_fit_only
                        else "algorithm_specific_technical_fit_minimum"
                    ),
                    "maximum_frames_per_replica": max(projected_counts),
                    "maximum_frame_role": (
                        "full_projection_observation_set"
                    ),
                    "cpu_seconds_per_physical_frame": 0.0,
                    "fixed_cpu_hours": 0.0,
                    "estimated_peak_memory_gib": 4.0,
                    "priority_weight": float(model["priority"]),
                    "member_observation_multiplier": multiplier,
                    "balance_group": (
                        f"{pca_task['balance_group']}:alternative:{algorithm}"
                    ),
                    "replica_sampling_mode": "balanced_pooled",
                    "calibration_status": (
                        "completed_bundle_memory_linear_fit_v3"
                    ),
                    "calibration_id": str(model["calibration"]),
                    "power_law_cost_model": {
                        "calibration_observations": 3_000,
                        "calibration_cpu_hours": (
                            (268.201244 / 6.0) * time_safety_factor / 3600.0
                        ),
                        "time_exponent": float(profile["time_exponent"]),
                        # The algorithms run sequentially in one executable
                        # bundle and reuse its large work arrays.  The completed
                        # TBA bundle peaked at 27.0 GiB for 40,000 fit
                        # observations; applying a quadratic memory exponent to
                        # each algorithm separately produced 98--147 GiB
                        # requests.  A conservative linear observation model is
                        # used for bundle peak memory; runtime keeps the
                        # algorithm-specific complexity exponent.
                        "calibration_memory_gib": 4.0,
                        "memory_exponent": 1.0,
                    },
                    "execution_bundle_memory_policy": (
                        "sequential_shared_buffers_linear_observation_v1"
                    ),
                    "complexity_class": profile["complexity_class"],
                    "full_fit_only": full_fit_only,
                    "projection_source_counts_iteration_input": list(
                        projected_counts
                    ),
                })
            continue
        elif module_id == "pald_community_analysis":
            definitions = project.get("definitions")
            definition = (
                definitions.get(module_id) if isinstance(definitions, dict) else None
            )
            if not isinstance(definition, dict):
                raise CampaignPlanningError(
                    f"view {view_id} has no PaLD community definition"
                )
            fit_limit = definition.get("maximum_observations")
            if (
                isinstance(fit_limit, bool)
                or not isinstance(fit_limit, int)
                or fit_limit < 2
            ):
                raise CampaignPlanningError(
                    f"view {view_id} PaLD maximum_observations is invalid"
                )
            cubic_scale = (fit_limit / 500.0) ** 3
            fixed_cpu_hours = (
                (0.5 + 28.637 * cubic_scale) * time_safety_factor / 3600.0
            )
            estimated_memory_gib = max(
                1.0, 0.0625 * (fit_limit / 500.0) ** 2 * 1.5
            )
            calibration_status = "completed_bounded_cubic_calibration"
            method_specific = {
                "fit_observation_limit": fit_limit,
                "cost_scaling_model": "O(B^3) time and O(B^2) memory",
                "calibration_observations": 500,
                "calibration_cpu_seconds": 28.637,
                "calibration_wall_seconds": 29.21,
                "calibration_maximum_rss_mib": 62.21875,
            }
        task = {
            "task_id": f"view:{view_id}:{module_id}",
            "workflow_id": view_id,
            "module_id": module_id,
            "task_scope": "conformational_view",
            "dependency_stage": int(model["stage"]) + 2,
            "effective_cpu_cap": 1,
            "source_frames_per_replica": list(source_counts),
            "minimum_frames_per_replica": minimum_frames_per_replica,
            **scientific,
            "maximum_frames_per_replica": int(pca_task["maximum_frames_per_replica"]),
            "cpu_seconds_per_physical_frame": (
                measured_rate * (multiplier / 2.0) * time_safety_factor
            ),
            "fixed_cpu_hours": fixed_cpu_hours,
            "estimated_peak_memory_gib": estimated_memory_gib,
            "priority_weight": float(model["priority"]),
            "member_observation_multiplier": multiplier,
            "balance_group": str(pca_task["balance_group"]),
            "calibration_status": calibration_status,
            "calibration_id": str(model["calibration"]),
            **method_specific,
        }
        tasks.append(task)
    return tasks


def _direct_task_inputs(sampling_plan: Mapping[str, object]) -> List[Dict[str, object]]:
    campaign = sampling_plan.get("campaign_resource_plan")
    rows = campaign.get("tasks") if isinstance(campaign, dict) else None
    if not isinstance(rows, list):
        raise CampaignPlanningError("direct-estimator campaign tasks are unavailable")
    result = []
    for row in rows:
        if not isinstance(row, dict) or row.get("task_scope") != "direct_trajectory_estimator":
            continue
        result.append({
            key: deepcopy(value)
            for key, value in row.items()
            if key not in {
                "selected_physical_frames_per_replica",
                "selected_physical_frame_count", "selected_member_observation_count",
                "coverage_fraction", "subsampling_triggered", "frame_selection",
                "integer_stride", "allocated_maximum_frames_per_replica",
                "candidate_frame_ceiling_per_replica",
                "sampling_strategy", "estimated_cpu_hours",
                "estimated_wall_hours_at_effective_cpu_cap",
                "source_limited_below_declared_minimum", "independent_sampling_unit",
                "member_observations_are_independent_replicas",
            }
        })
    return result


def _base_derived_tasks(
    base_project: Mapping[str, object],
    direct_tasks: Sequence[Mapping[str, object]],
    *,
    time_safety_factor: float,
) -> List[Dict[str, object]]:
    """Model every generated base task that inherits an upstream frame set."""

    requested = base_project.get("requested_modules")
    if not isinstance(requested, list):
        raise CampaignPlanningError("base project requested_modules are unavailable")
    direct_by_module = {
        str(row["module_id"]): row
        for row in direct_tasks
        if row.get("task_scope") == "direct_trajectory_estimator"
    }
    tasks = []
    for module_id, model in _BASE_DERIVED_MODELS.items():
        if module_id not in requested:
            continue
        upstream = direct_by_module.get(str(model["upstream"]))
        if upstream is None:
            raise CampaignPlanningError(
                f"base module {module_id} has no planned upstream task"
            )
        source_counts = [
            int(value) for value in upstream["source_frames_per_replica"]
        ]
        raw_ids = upstream.get("system_ids_per_replica")
        system_ids = (
            [str(value) for value in raw_ids]
            if isinstance(raw_ids, list) and len(raw_ids) == len(source_counts)
            else None
        )
        raw_intervals = upstream.get("frame_intervals_ns_per_replica")
        raw_spans = upstream.get("source_time_spans_ns_per_replica")
        intervals = (
            [float(value) for value in raw_intervals]
            if isinstance(raw_intervals, list)
            and len(raw_intervals) == len(source_counts) else None
        )
        spans = (
            [float(value) for value in raw_spans]
            if isinstance(raw_spans, list)
            and len(raw_spans) == len(source_counts) else None
        )
        scientific = _scientific_task_contract(
            module_id, source_counts, system_ids_per_replica=system_ids,
            frame_intervals_ns_per_replica=intervals,
            source_time_spans_ns_per_replica=spans,
        )
        minimum = max(
            int(upstream["minimum_frames_per_replica"]),
            int(scientific["attainable_scientific_minimum_frames_per_replica"]),
        )
        tasks.append({
            "task_id": f"base:{module_id}",
            "workflow_id": "base",
            "module_id": module_id,
            "task_scope": "base_derived_analysis",
            "dependency_stage": int(model["stage"]),
            "effective_cpu_cap": 1,
            "source_frames_per_replica": source_counts,
            "minimum_frames_per_replica": minimum,
            **scientific,
            "maximum_frames_per_replica": int(
                upstream["maximum_frames_per_replica"]
            ),
            "cpu_seconds_per_physical_frame": (
                float(model["seconds_per_frame"]) * time_safety_factor
            ),
            "fixed_cpu_hours": float(model["fixed_cpu_hours"]),
            "estimated_peak_memory_gib": float(model["memory_gib"]),
            "priority_weight": float(model["priority"]),
            "member_observation_multiplier": 1,
            "balance_group": str(upstream["balance_group"]),
            "replica_sampling_mode": str(
                upstream.get("replica_sampling_mode", "balanced_pooled")
            ),
            "calibration_status": "provisional_complexity_model",
            "calibration_id": "bounded-derived-postprocess-proxy-v1",
        })
    return tasks


def _method_rows(sampling_plan: Mapping[str, object]) -> Dict[str, MutableMapping[str, object]]:
    rows = sampling_plan.get("method_plans")
    if not isinstance(rows, list):
        raise CampaignPlanningError("sampling plan method rows are unavailable")
    return {
        str(row["module_id"]): row
        for row in rows if isinstance(row, dict) and "module_id" in row
    }


def _apply_direct_project_sampling(
    project: MutableMapping[str, object], sampling_plan: Mapping[str, object]
) -> None:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise CampaignPlanningError("base project definitions are unavailable")
    rows = _method_rows(sampling_plan)
    stride_definitions = {
        "replica_rmsd_rg": "replica_rmsd_rg",
        "pooled_rmsf": "pooled_rmsf",
        "dihedral_distributions": "dihedral_distributions",
    }
    for module_id, definition_id in stride_definitions.items():
        row = rows.get(module_id)
        definition = definitions.get(definition_id)
        if isinstance(row, dict) and isinstance(definition, dict):
            selection = row.get("frame_selection")
            definition["frame_stride"] = int(
                selection.get("stride", row.get("frame_stride", 1))
                if isinstance(selection, dict) else row.get("frame_stride", 1)
            )
    selection_definitions = {
        "structural_integrity_qc": "structural_qc",
        "dccm": "dccm", "individual_pca": "individual_pca",
        "hydrogen_bond_discovery": "hydrogen_bond_discovery",
        "solvent_accessible_surface_area": "solvent_accessible_surface_area",
        "water_mediated_hydrogen_bond_networks": "water_mediated_hydrogen_bond_networks",
        "secondary_structure": "secondary_structure",
    }
    for module_id, definition_id in selection_definitions.items():
        row = rows.get(module_id)
        definition = definitions.get(definition_id)
        if isinstance(row, dict) and isinstance(definition, dict):
            selection = row.get("frame_selection", {"mode": "fixed_stride_v1"})
            definition["frame_stride"] = 1
            definition["frame_selection"] = deepcopy(selection)
            if module_id == "individual_pca":
                definition["projection_frame_stride"] = 1
                definition["projection_frame_selection"] = deepcopy(selection)
            if module_id == "secondary_structure":
                definition["maximum_frames"] = int(row["selected_frame_count"])


def _apply_automatic_context_allocation(
    project_path: Path,
    task_rows: Mapping[str, Mapping[str, object]],
    source_counts: Sequence[int],
    *,
    context_id: Optional[str] = None,
    task_namespace: Optional[str] = None,
) -> Dict[str, object]:
    project = load_json(project_path)
    if not isinstance(project, dict):
        raise CampaignPlanningError(
            f"automatic-context project is not an object: {project_path}"
        )
    resolved_context_id = (
        context_id
        if context_id is not None
        else project_path.name[len("project-") : -len(".json")]
    )
    resolved_task_namespace = (
        task_namespace
        if task_namespace is not None
        else f"context:{resolved_context_id}"
    )
    definitions = project.get("definitions")
    requested = project.get("requested_modules")
    if not isinstance(definitions, dict) or not isinstance(requested, list):
        raise CampaignPlanningError(
            f"automatic-context project {resolved_context_id} is incomplete"
        )
    applied: list[Dict[str, object]] = []
    feature_family_budget: Optional[int] = None
    for raw_module in requested:
        module_id = str(raw_module)
        if module_id not in _AUTOMATIC_CONTEXT_MODELS:
            continue
        allocation = task_rows.get(f"{resolved_task_namespace}:{module_id}")
        if allocation is None:
            continue
        selected = [
            int(value)
            for value in allocation["selected_physical_frames_per_replica"]
        ]
        selection = deepcopy(allocation["frame_selection"])
        stride = int(allocation.get("integer_stride", 1))
        selected_total = sum(selected)
        all_frames = selected == list(source_counts)
        if module_id in {
            "trajectory_features",
            "scalar_feature_distributions", "scalar_threshold_states",
        }:
            feature_family_budget = max(selected)
        elif module_id == "radial_distribution_functions":
            definition = definitions.get(module_id)
            if isinstance(definition, dict):
                definition["frame_stride"] = 1
                definition["frame_selection"] = deepcopy(selection)
        elif module_id in {
            "nucleic_acid_geometry", "ion_coordination_geometry",
        }:
            definition = definitions.get(module_id)
            if isinstance(definition, dict):
                definition["frame_stride"] = stride
                definition["maximum_frames"] = selected_total
        elif module_id == "nucleic_acid_structure":
            definition = definitions.get(module_id)
            if isinstance(definition, dict):
                definition["frame_stride"] = 1
                definition["frame_selection"] = deepcopy(selection)
                definition["maximum_frames"] = selected_total
        applied.append({
            "module_id": module_id,
            "source_physical_frames": sum(source_counts),
            "selected_physical_frames": selected_total,
            "selected_physical_frames_per_replica": selected,
            "frame_selection": selection,
        })
    if feature_family_budget is not None:
        stride = integer_stride_for_budget(
            list(source_counts),
            feature_family_budget,
            error_type=CampaignPlanningError,
        )
        selected_total = sum(
            integer_stride_selected_count(source, stride)
            for source in source_counts
        )
        trajectory = definitions.get("trajectory_features")
        feature_count = 1
        if isinstance(trajectory, dict):
            features = trajectory.get("features")
            feature_count = max(1, len(features) if isinstance(features, list) else 1)
            trajectory["frame_stride"] = stride
            trajectory["maximum_feature_values"] = max(
                1, selected_total * feature_count * 16
            )
        observables = definitions.get("optional_observables")
        if isinstance(observables, dict):
            features = observables.get("features")
            count = max(1, len(features) if isinstance(features, list) else 1)
            observables["frame_stride"] = stride
            observables["maximum_observations"] = max(1, selected_total * count)
        for definition_id in (
            "scalar_feature_distributions", "scalar_threshold_states"
        ):
            definition = definitions.get(definition_id)
            if isinstance(definition, dict):
                definition["maximum_observations"] = max(
                    1, selected_total * feature_count * 16
                )
    project_path.write_text(
        json.dumps(project, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_project(project, source_path=project_path, check_paths=True)
    return {"context_id": resolved_context_id, "modules": applied}


def _apply_view_allocation(
    project_path: Path,
    task_rows: Mapping[str, Mapping[str, object]],
    source_counts: Sequence[int],
    *,
    target_wall_hours: float,
) -> Dict[str, object]:
    project = load_json(project_path)
    assert isinstance(project, dict)
    view_id = project_path.name[len("project-") : -len(".json")]
    allocation = task_rows.get(f"view:{view_id}:common_pca")
    if allocation is None:
        raise CampaignPlanningError(f"view {view_id} has no campaign PCA allocation")
    selected = [
        int(value) for value in allocation["selected_physical_frames_per_replica"]
    ]
    budget = max(selected)
    all_frames = selected == list(source_counts)
    selection = deepcopy(allocation["frame_selection"])
    projection_stride = int(allocation.get("integer_stride", 1))
    definitions = project["definitions"]
    assert isinstance(definitions, dict)
    common_pca = definitions["common_pca"]
    assert isinstance(common_pca, dict)
    common_pca["projection_frame_stride"] = 1
    common_pca["projection_frame_selection"] = deepcopy(selection)
    existing_basis = common_pca.get("frame_selection", {"mode": "fixed_stride_v1"})
    existing_stride = (
        int(existing_basis.get("stride", 1))
        if isinstance(existing_basis, dict) else 1
    )
    basis_stride = max(projection_stride, existing_stride)
    basis_counts = [
        integer_stride_selected_count(source, basis_stride)
        for source in source_counts
    ]
    basis_budget = max(basis_counts)
    common_pca["frame_stride"] = 1
    common_pca["frame_selection"] = (
        {"mode": "fixed_stride_v1"} if basis_stride == 1 else {
            "mode": "integer_stride_per_replica_v1",
            "stride": basis_stride,
        }
    )
    multiplier = _view_member_multiplier(project)
    effective = sum(selected) * multiplier
    representative = definitions.get("representative_frames")
    if isinstance(representative, dict):
        representative["maximum_candidates"] = max(1, effective)
    grouped = definitions.get("grouped_ml")
    if isinstance(grouped, dict):
        grouped["maximum_observations"] = max(1, effective)
    alternative = definitions.get("alternative_clustering")
    if isinstance(alternative, dict):
        profiles = alternative_clustering_fit_profiles()
        algorithm_plans: Dict[str, object] = {}
        for raw_algorithm in alternative["algorithms"]:
            algorithm = str(raw_algorithm)
            profile = profiles[algorithm]
            algorithm_allocation = task_rows.get(
                f"view:{view_id}:alternative_clustering:{algorithm}"
            )
            full_observations = sum(selected) * multiplier
            if algorithm_allocation is None:
                algorithm_plans[algorithm] = {
                    "execution": "skip",
                    "skip_reason": (
                        "full-observation fit exceeds the full-fit-only "
                        "resource screen"
                    ),
                    "full_observation_count": full_observations,
                    "fit_observation_ceiling": min(
                        full_observations,
                        int(profile["reference_fit_observation_ceiling"]),
                    ),
                    "complexity_class": profile["complexity_class"],
                    "time_exponent": float(profile["time_exponent"]),
                    "calibration_status": (
                        "globally_coupled_full_fit_or_skip_v2"
                    ),
                }
                continue
            algorithm_stride = int(algorithm_allocation["integer_stride"])
            selected_fit_per_physical_replica = [
                multiplier * integer_stride_selected_count(count, algorithm_stride)
                for count in selected
            ]
            selected_fit_count = sum(selected_fit_per_physical_replica)
            algorithm_plans[algorithm] = {
                "execution": "run",
                "mode": "integer_stride_per_replica_member_v1",
                "strides": [algorithm_stride],
                "primary_stride": algorithm_stride,
                "fit_observation_ceiling": selected_fit_count,
                "selected_fit_observations_per_physical_replica": (
                    selected_fit_per_physical_replica
                ),
                "selected_fit_observation_count": selected_fit_count,
                "full_observation_count": full_observations,
                "complexity_class": profile["complexity_class"],
                "time_exponent": float(profile["time_exponent"]),
                "calibration_status": (
                    "globally_coupled_campaign_allocation_v2"
                ),
            }
        fit_plan = {
            "mode": "algorithm_specific_integer_stride_v1",
            "target_wall_hours": target_wall_hours,
            "member_observation_multiplier": multiplier,
            "source_physical_frames_per_replica": list(selected),
            "full_observation_count": sum(selected) * multiplier,
            "algorithm_plans": algorithm_plans,
            "scientific_boundary": (
                "Every runnable family receives its own exact integer stride "
                "from the globally coupled campaign planner. Allocations are "
                "computational, not evidence of clustering stability, "
                "metastability, or convergence."
            ),
        }
        alternative["fit_sampling"] = fit_plan
        running = [
            plan for plan in fit_plan["algorithm_plans"].values()
            if isinstance(plan, dict) and plan.get("execution") == "run"
        ]
        alternative["maximum_observations"] = max(
            (
                int(plan["selected_fit_observation_count"])
                for plan in running
            ),
            default=1,
        )
    pald = definitions.get("pald_community_analysis")
    if isinstance(pald, dict):
        pald["maximum_observations"] = max(
            2, min(int(pald.get("maximum_observations", 500)), effective)
        )
    exports = definitions.get("state_coordinate_exports")
    if isinstance(exports, dict):
        exports["frame_stride_within_state"] = max(1, math.ceil(effective / 200))
    project_path.write_text(
        json.dumps(project, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_project(project, source_path=project_path, check_paths=True)
    return {
        "view_id": view_id,
        "source_physical_frames": sum(source_counts),
        "selected_physical_frames": sum(selected),
        "selected_physical_frames_per_replica": selected,
        "member_observation_multiplier": multiplier,
        "selected_member_observations": effective,
        "basis_maximum_frames_per_replica": basis_budget,
        "basis_integer_stride": basis_stride,
        "projection_frame_selection": selection,
        "alternative_clustering_fit_sampling": (
            deepcopy(alternative.get("fit_sampling"))
            if isinstance(alternative, dict) else None
        ),
    }


def plan_and_apply_complete_campaign(
    *,
    root: Path,
    sampling_plan: MutableMapping[str, object],
    analysis_config: Mapping[str, object],
    view_project_files: Sequence[str],
    base_project_path: Path,
    view_frame_counts_by_id: Mapping[str, Sequence[int]] | None = None,
    context_project_files: Sequence[str] = (),
    context_frame_counts_by_id: Mapping[str, Sequence[int]] | None = None,
    time_safety_factor: float = 1.5,
) -> Dict[str, object]:
    """Plan and apply one hard envelope to every currently generated task."""

    dimensions = sampling_plan.get("dimensions")
    if not isinstance(dimensions, dict):
        raise CampaignPlanningError("sampling dimensions are unavailable")
    replicas = dimensions.get("replicas")
    if not isinstance(replicas, list) or not replicas:
        raise CampaignPlanningError("sampling dimensions contain no replicas")
    source_counts = [int(row["source_frame_count"]) for row in replicas]
    base_timing_available = all(
        isinstance(row, Mapping)
        and row.get("maximum_frame_interval_ns") is not None
        and row.get("source_time_span_ns") is not None
        for row in replicas
    )
    base_frame_intervals_ns = (
        [float(row["maximum_frame_interval_ns"]) for row in replicas]
        if base_timing_available else None
    )
    base_source_time_spans_ns = (
        [float(row["source_time_span_ns"]) for row in replicas]
        if base_timing_available else None
    )

    def timing_for_counts(
        counts: Sequence[int],
    ) -> tuple[Optional[list[float]], Optional[list[float]]]:
        if list(counts) != source_counts:
            return None, None
        return base_frame_intervals_ns, base_source_time_spans_ns
    execution = analysis_config.get("execution")
    if not isinstance(execution, dict):
        raise CampaignPlanningError("analysis execution configuration is unavailable")
    sampling_configuration = analysis_config.get("sampling")
    if not isinstance(sampling_configuration, dict):
        raise CampaignPlanningError("analysis sampling configuration is unavailable")
    minimums_path = sampling_configuration.get("scientific_minimums_file")
    try:
        scientific_minimums = load_scientific_minimums(
            Path(minimums_path) if isinstance(minimums_path, str) else None
        )
    except (OSError, ValueError) as exc:
        raise CampaignPlanningError(str(exc)) from exc
    try:
        measured_calibrations = load_resource_calibration_catalog(
            execution.get("resource_calibration_catalog"),
            censored_timeout_safety_factor=float(
                execution.get("censored_timeout_safety_factor", 1.5)
            ),
        )
    except (ResourceCalibrationError, OSError, ValueError) as exc:
        raise CampaignPlanningError(str(exc)) from exc
    view_paths = [
        root / filename for filename in view_project_files
        if filename.startswith("project-") and filename.endswith(".json")
    ]
    context_paths = [root / filename for filename in context_project_files]
    delegated_context_modules = set()
    for context_path in context_paths:
        context_project = load_json(context_path)
        requested = (
            context_project.get("requested_modules")
            if isinstance(context_project, Mapping) else None
        )
        if isinstance(requested, list):
            delegated_context_modules.update(
                str(module_id) for module_id in requested
                if str(module_id) in _AUTOMATIC_CONTEXT_MODELS
            )
    memory_policy: Dict[str, float] = {
        "memory_safety_factor": 1.5,
        "memory_overhead_gib": 1.0,
        "minimum_memory_gib": 2.0,
    }
    scheduler_time_policy: Dict[str, float] = {
        "walltime_safety_factor": 1.5,
        "walltime_overhead_minutes": 15.0,
        "minimum_wall_minutes": 30.0,
    }
    if str(execution.get("submission_adapter", "local")) == "slurm":
        profile_path = execution.get("slurm_profile")
        if not isinstance(profile_path, str) or not profile_path:
            raise CampaignPlanningError(
                "Slurm campaign planning requires execution.slurm_profile"
            )
        try:
            profile = load_slurm_profile(Path(profile_path))
        except (OSError, ValueError) as exc:
            raise CampaignPlanningError(str(exc)) from exc
        policy = profile.get("resource_policy")
        if isinstance(policy, Mapping):
            for key in memory_policy:
                memory_policy[key] = float(policy[key])
            for key in scheduler_time_policy:
                scheduler_time_policy[key] = float(policy[key])

    def annotate_permissive_minimum_request(
        request: MutableMapping[str, object],
    ) -> None:
        """Add workflow-level model and scheduler padding to one request."""

        padding = request.get("padding_factors")
        if not isinstance(padding, dict):
            padding = {}
            request["padding_factors"] = padding
        padding.update({
            "modeled_task_time_safety_factor": float(time_safety_factor),
            "analysis_memory_model_safety_factor": float(
                execution.get("memory_safety_factor", 1.25)
            ),
            "scheduler_walltime_safety_factor_per_job": scheduler_time_policy[
                "walltime_safety_factor"
            ],
            "scheduler_walltime_overhead_minutes_per_job": (
                scheduler_time_policy["walltime_overhead_minutes"]
            ),
            "scheduler_minimum_wall_minutes_per_job": scheduler_time_policy[
                "minimum_wall_minutes"
            ],
            "scheduler_walltime_interpretation": (
                "per-job timeout allowance; not additional campaign science "
                "time and therefore not added again to the requested campaign "
                "wall time"
            ),
        })

    def annotate_plan_minimum_request(candidate: Mapping[str, object]) -> None:
        request = candidate.get("permissive_minimum_resource_request")
        if isinstance(request, dict):
            annotate_permissive_minimum_request(request)
    cache_mode = str(execution.get("coordinate_cache", "auto"))
    coordinate_cache_enabled = bool(view_paths) and cache_mode in {"auto", "required"}
    coordinate_cache_input = execution.get("coordinate_cache_input")
    coordinate_cache_build_required = (
        coordinate_cache_enabled and coordinate_cache_input is None
    )
    cache_materialization = str(
        execution.get("coordinate_cache_materialization", "planned_strided")
    )
    def build_tasks() -> tuple[List[Dict[str, object]], Dict[str, object]]:
        built = [
            row for row in _direct_task_inputs(sampling_plan)
            if not (
                row.get("task_scope") == "direct_trajectory_estimator"
                and str(row.get("module_id")) in delegated_context_modules
            )
        ]
        if coordinate_cache_build_required:
            cache_workers = min(
                int(execution["maximum_parallel_cpus"]), len(source_counts)
            )
            cache_atom_multiplier = max(
                0.01, int(dimensions["maximum_atom_count"]) / 85_206.0
            )
            built.insert(0, {
                "task_id": "preprocessing:coordinate_cache",
                "workflow_id": "coordinate_cache",
                "module_id": "coordinate_cache",
                "task_scope": "continuous_unwrap_working_cache",
                "dependency_stage": 0,
                "effective_cpu_cap": cache_workers,
                "intrinsic_cpu_cap": len(source_counts),
                "source_frames_per_replica": list(source_counts),
                "minimum_frames_per_replica": max(source_counts),
                "minimum_frame_role": "planner-selected working-cache coverage",
                "maximum_frames_per_replica": max(source_counts),
                "maximum_frame_role": "lossless materialization when affordable",
                "cpu_seconds_per_physical_frame": (
                    (232.39 / 700.0) * cache_atom_multiplier * time_safety_factor
                ),
                "fixed_cpu_hours": 0.0,
                "estimated_peak_memory_gib": max(1.0, 0.25 * cache_workers),
                "memory_size_scaling_applied": True,
                "measured_memory_observation_scaling_eligible": False,
                "priority_weight": 100.0,
                "member_observation_multiplier": 1,
                "balance_group": "preprocessing:coordinate_cache:all_frames",
                "replica_sampling_mode": "independent_all_available",
                "calibration_status": "completed_trex_cache_pilot_linear",
                "calibration_id": (
                    "apollo-trex-coordinate-cache-700f-85206a-2026-08-15"
                ),
                "calibration_cpu_seconds": 232.39,
                "calibration_frames": 700,
                "calibration_source_atom_count": 85_206,
                "calibration_cached_atom_count": 7_560,
                "atom_runtime_multiplier": cache_atom_multiplier,
                "measured_cpu_rate_multiplier": cache_atom_multiplier,
                "measured_memory_multiplier": max(0.1, cache_atom_multiplier),
                "measured_workload_scaling": {
                    "reference_source_atom_count": 85_206,
                    "source_atom_count": int(dimensions["maximum_atom_count"]),
                    "atom_runtime_multiplier_floor": 0.01,
                },
            })
        current_base = load_json(base_project_path)
        if not isinstance(current_base, dict):
            raise CampaignPlanningError("base project is not an object")
        built.extend(_base_derived_tasks(
            current_base, built, time_safety_factor=time_safety_factor
        ))
        already_planned_modules = {
            str(row.get("module_id", "")) for row in built
        }
        base_automatic_tasks = _automatic_context_tasks(
            base_project_path,
            source_counts,
            time_safety_factor=time_safety_factor,
            frame_intervals_ns_per_replica=base_frame_intervals_ns,
            source_time_spans_ns_per_replica=base_source_time_spans_ns,
            context_id="base",
            task_namespace="base",
            task_scope="base_automatic_chemistry",
        )
        built.extend(
            row for row in base_automatic_tasks
            if str(row["module_id"]) not in already_planned_modules
        )
        for path in context_paths:
            context_id = path.name[len("project-") : -len(".json")]
            context_source_counts = (
                list(context_frame_counts_by_id[context_id])
                if context_frame_counts_by_id is not None
                and context_id in context_frame_counts_by_id
                else source_counts
            )
            built.extend(_automatic_context_tasks(
                path,
                context_source_counts,
                time_safety_factor=time_safety_factor,
                frame_intervals_ns_per_replica=timing_for_counts(
                    context_source_counts
                )[0],
                source_time_spans_ns_per_replica=timing_for_counts(
                    context_source_counts
                )[1],
            ))
        for path in view_paths:
            view_id = path.name[len("project-") : -len(".json")]
            view_source_counts = (
                list(view_frame_counts_by_id[view_id])
                if view_frame_counts_by_id is not None
                and view_id in view_frame_counts_by_id
                else source_counts
            )
            built.extend(_view_tasks(
                path,
                view_source_counts,
                int(dimensions["maximum_atom_count"]),
                time_safety_factor=time_safety_factor,
                frame_intervals_ns_per_replica=timing_for_counts(
                    view_source_counts
                )[0],
                source_time_spans_ns_per_replica=timing_for_counts(
                    view_source_counts
                )[1],
            ))
        built = apply_scientific_minimums_to_tasks(
            built, scientific_minimums
        )
        _apply_system_memory_scaling(
            built, int(dimensions["maximum_atom_count"])
        )
        _apply_measured_resource_calibrations(
            built, measured_calibrations,
            time_safety_factor=time_safety_factor,
            memory_safety_factor=float(
                execution.get("memory_safety_factor", 1.25)
            ),
        )
        return built, current_base

    previous_signature: object = None
    iteration_history: List[Dict[str, object]] = []
    plan: Dict[str, object] = {}
    converged = False
    maximum_iterations = 8
    for planning_iteration in range(1, maximum_iterations + 1):
        tasks, base_project = build_tasks()
        if not tasks:
            raise CampaignPlanningError(
                "the prepared campaign contains no executable tasks"
            )
        try:
            planning_kwargs = {
                "maximum_parallel_cpus": int(execution["maximum_parallel_cpus"]),
                "maximum_wall_hours": float(execution["maximum_hours_per_cpu"]),
                "maximum_memory_gib": float(execution["maximum_memory_gib"]),
                "planning_utilization": float(execution["planning_utilization"]),
                "pilot_budget_fraction": float(execution["pilot_budget_fraction"]),
                "finalization_headroom_fraction": float(
                    execution.get("finalization_headroom_fraction", 0.0)
                ),
                "memory_safety_factor": memory_policy["memory_safety_factor"],
                "memory_overhead_gib": memory_policy["memory_overhead_gib"],
                "minimum_scheduler_memory_gib": memory_policy[
                    "minimum_memory_gib"
                ],
            }
            if (
                coordinate_cache_build_required
                and cache_materialization == "planned_strided"
            ):
                plan = plan_global_stride_projection_coupled_campaign_resource_budget(
                    tasks,
                    coordinate_cache_minimum_frames_per_replica=1,
                    coordinate_cache_full_scan_fraction=float(
                        execution.get("coordinate_cache_full_scan_fraction", 1.0)
                    ),
                    overall_stride_candidate_strides=list(
                        execution.get(
                            "overall_stride_candidates",
                            [1, 2, 3, 4, 5, 10, 20, 100],
                        )
                    ),
                    **planning_kwargs,
                )
            else:
                plan = plan_campaign_resource_budget(tasks, **planning_kwargs)
        except ResourcePlanningError as exc:
            raise CampaignPlanningError(str(exc)) from exc
        annotate_plan_minimum_request(plan)
        if (
            plan["feasibility_status"] != "feasible"
            and bool(execution.get("fail_if_minimum_coverage_unaffordable", True))
        ):
            recommendation = recommend_scientifically_valid_task_subset(
                tasks, **planning_kwargs
            )
            recommended_plan = recommendation.get("recommended_plan")
            if isinstance(recommended_plan, Mapping):
                annotate_plan_minimum_request(recommended_plan)
            protected_request = recommendation.get(
                "best_protected_subset_minimum_resource_request"
            )
            if isinstance(protected_request, dict):
                annotate_permissive_minimum_request(protected_request)
            plan["method_reduction_recommendation"] = recommendation
            if recommendation["recommendation_status"] == "no_feasible_subset_found":
                message = (
                    "No acceptable reduced plan: the configured whole-campaign "
                    "envelope cannot retain every protected module at its "
                    "scientific minimum. "
                )
            else:
                message = (
                    "The configured whole-campaign envelope cannot fund every "
                    "enabled technical minimum; a reduced configuration is "
                    "available for explicit review. "
                )
            raise CampaignPlanningError(
                message + _campaign_infeasibility_detail(plan),
                plan=plan,
            )
        _apply_campaign_direct_allocations(
            sampling_plan["method_plans"],  # type: ignore[arg-type]
            dimensions,
            plan,
            time_safety_factor=time_safety_factor,
        )
        sampling_plan["campaign_resource_plan"] = plan
        if coordinate_cache_enabled and coordinate_cache_input is not None:
            plan["coordinate_cache_reuse"] = {
                "status": "external_lossless_cache",
                "cache_directory": str(coordinate_cache_input),
                "cache_stride": 1,
                "cache_build_task_omitted": True,
            }
        _apply_direct_project_sampling(base_project, sampling_plan)
        base_project_path.write_text(
            json.dumps(base_project, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_project(base_project, source_path=base_project_path, check_paths=True)
        task_rows = {
            str(row["task_id"]): row
            for row in plan["tasks"] if isinstance(row, dict)
        }
        base_automatic_allocation = _apply_automatic_context_allocation(
            base_project_path,
            task_rows,
            source_counts,
            context_id="base",
            task_namespace="base",
        )
        plan["applied_base_automatic_chemistry_allocation"] = (
            base_automatic_allocation
        )
        applied_views = []
        for path in view_paths:
            view_id = path.name[len("project-") : -len(".json")]
            view_source_counts = (
                list(view_frame_counts_by_id[view_id])
                if view_frame_counts_by_id is not None
                and view_id in view_frame_counts_by_id
                else source_counts
            )
            applied_views.append(_apply_view_allocation(
                path,
                task_rows,
                view_source_counts,
                target_wall_hours=float(execution["maximum_hours_per_cpu"]),
            ))
        plan["applied_view_allocations"] = applied_views
        applied_contexts = []
        for path in context_paths:
            context_id = path.name[len("project-") : -len(".json")]
            context_source_counts = (
                list(context_frame_counts_by_id[context_id])
                if context_frame_counts_by_id is not None
                and context_id in context_frame_counts_by_id
                else source_counts
            )
            applied_contexts.append(_apply_automatic_context_allocation(
                path, task_rows, context_source_counts
            ))
        plan["applied_automatic_context_allocations"] = applied_contexts
        signature = tuple(
            (
                str(row["task_id"]),
                int(row["integer_stride"]),
                tuple(int(value) for value in row[
                    "selected_physical_frames_per_replica"
                ]),
            )
            for row in plan["tasks"] if isinstance(row, dict)
        )
        alternative_rows = [
            row for row in plan["tasks"]
            if isinstance(row, dict)
            and row.get("task_scope") == "conformational_view_algorithm_fit"
        ]
        iteration_history.append({
            "iteration": planning_iteration,
            "task_count": len(plan["tasks"]),
            "estimated_selected_cpu_hours": plan["estimated_selected_cpu_hours"],
            "estimated_selected_wall_hours_lower_bound": plan[
                "estimated_selected_wall_hours_lower_bound"
            ],
            "alternative_clustering_logical_task_count": len(alternative_rows),
            "alternative_clustering_integer_strides": {
                str(row["task_id"]): int(row["integer_stride"])
                for row in alternative_rows
            },
        })
        if signature == previous_signature:
            converged = True
            break
        previous_signature = signature
    if not converged:
        raise CampaignPlanningError(
            "globally coupled campaign planning did not converge within "
            f"{maximum_iterations} iterations"
        )
    plan.update({
        "planning_scope": (
            "complete generated base including inferred chemistry, per-system "
            "topology-local automatic chemical context, and conformational-view campaign"
            if context_paths
            else "complete generated base including inferred chemistry plus "
            "conformational-view campaign"
        ),
        "planning_algorithm": "globally_coupled_integer_stride_iteration_v2",
        "planning_iterations": len(iteration_history),
        "planning_converged": True,
        "planning_iteration_history": iteration_history,
        "generated_view_count": len(view_paths),
        "generated_automatic_context_project_count": len(context_paths),
        "coordinate_cache_mode": cache_mode,
        "coordinate_cache_materialization": (
            "external_lossless_stride_1"
            if coordinate_cache_input is not None else cache_materialization
        ),
        "coordinate_cache_enabled": coordinate_cache_enabled,
        "coordinate_cache_scope": (
            "a separately prepared lossless stride-1 cache is reused for molecular-"
            "payload views; base water-dependent analyses retain original solvated "
            "trajectories"
            if coordinate_cache_enabled and coordinate_cache_input is not None
            else "all source frames scanned for continuous unwrapping; the planned "
            "integer-stride working set is materialized without bulk solvent; "
            "base water-dependent analyses retain original solvated trajectories"
            if coordinate_cache_enabled else "disabled"
        ),
        "shared_basis_view_count": sum(
            "__" not in path.name[len("project-") : -len(".json")]
            for path in view_paths
        ),
        "per_system_view_count": sum(
            path.name[len("project-") : -len(".json")].startswith("system_")
            and "__" in path.name[len("project-") : -len(".json")]
            for path in view_paths
        ),
        "per_system_balance_contract": (
            "Corresponding per-system view families receive the same maximum "
            "physical-frame budget per original replica; member expansion is "
            "reported separately and never changes replica identity."
        ),
        "retained_calibration_evidence": (
            "Apollo/TREX matched and oligomer resource snapshot, including the "
            "30,000-physical-frame/60,000-member view campaign"
        ),
        "time_safety_factor": time_safety_factor,
        "analysis_memory_model_safety_factor": float(
            execution.get("memory_safety_factor", 1.25)
        ),
        "scheduler_memory_safety_factor": memory_policy[
            "memory_safety_factor"
        ],
        "scheduler_memory_overhead_gib": memory_policy[
            "memory_overhead_gib"
        ],
        "scheduler_minimum_memory_gib": memory_policy[
            "minimum_memory_gib"
        ],
        "resource_safety_margins": {
            "modeled_task_time_factor": time_safety_factor,
            "analysis_memory_model_factor": float(
                execution.get("memory_safety_factor", 1.25)
            ),
            "planning_utilization": float(execution["planning_utilization"]),
            "pilot_budget_fraction": float(execution["pilot_budget_fraction"]),
            "finalization_headroom_fraction": float(
                execution.get("finalization_headroom_fraction", 0.0)
            ),
            "scheduler_memory_safety_factor": memory_policy[
                "memory_safety_factor"
            ],
            "scheduler_memory_overhead_gib": memory_policy[
                "memory_overhead_gib"
            ],
            "scheduler_minimum_memory_gib": memory_policy[
                "minimum_memory_gib"
            ],
            **scheduler_time_policy,
            "scheduler_walltime_interpretation": (
                "per-job timeout allowance; not additional planned science time"
            ),
        },
        "censored_timeout_safety_factor": float(
            execution.get("censored_timeout_safety_factor", 1.5)
        ),
        "provisional_model_modules": sorted({
            str(row["module_id"])
            for row in tasks
            if str(row.get("calibration_status", "")).startswith("provisional")
        }),
        "alternative_clustering_cost_contract": (
            "Each runnable family is a separate nonlinear logical allocation; "
            "families in one view share a serial execution bundle for wall-time "
            "accounting. Ward and quality-threshold run only on every projected "
            "observation or are explicitly skipped."
        ),
    })
    sampling_plan["campaign_resource_plan"] = plan
    return plan
