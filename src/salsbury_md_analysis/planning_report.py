"""Human-readable reporting for prepared campaign decisions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .manifests import load_json


class PlanningReportError(ValueError):
    """Raised when prepared planning artifacts cannot be summarized safely."""


_FamilyMatcher = Callable[[Mapping[str, object]], bool]


def _module_is(*module_ids: str) -> _FamilyMatcher:
    accepted = set(module_ids)
    return lambda task: str(task.get("module_id")) in accepted


def _view_scope(task: Mapping[str, object]) -> str:
    task_id = str(task.get("task_id", ""))
    if task_id.startswith("view:system_"):
        return "per_system"
    if task_id.startswith("view:"):
        return "shared"
    return "other"


def _scoped_modules(scope: str, *module_ids: str) -> _FamilyMatcher:
    accepted = set(module_ids)
    return lambda task: (
        str(task.get("module_id")) in accepted and _view_scope(task) == scope
    )


def _alternative_methods(scope: str, methods: Sequence[str]) -> _FamilyMatcher:
    accepted = set(methods)
    return lambda task: (
        str(task.get("module_id")) == "alternative_clustering"
        and _view_scope(task) == scope
        and str(task.get("task_id", "")).rsplit(":", 1)[-1] in accepted
    )


# Ordered for the user-facing report. Anything not covered here is appended by
# module id, so a new analysis cannot silently disappear from the report.
_ANALYSIS_FAMILIES: Sequence[Tuple[str, str, _FamilyMatcher, Sequence[str]]] = (
    ("continuous_cache", "Continuous unwrapping/cache", _module_is("coordinate_cache"), ("coordinate_cache",)),
    ("ion_atmosphere", "Ion atmosphere, per system", _module_is("ion_atmosphere"), ("ion_atmosphere",)),
    ("dihedrals_sasa", "Dihedrals and SASA", _module_is("dihedral_distributions", "solvent_accessible_surface_area"), ("dihedral_distributions", "solvent_accessible_surface_area")),
    ("rmsd_rg_convergence", "RMSD/Rg and convergence", _module_is("replica_rmsd_rg", "convergence_uncertainty"), ("replica_rmsd_rg", "convergence_uncertainty")),
    ("rmsf_dccm", "RMSF and DCCM", _module_is("pooled_rmsf", "dccm"), ("pooled_rmsf", "dccm")),
    ("water_networks", "Water-mediated networks", _module_is("water_mediated_hydrogen_bond_networks"), ("water_mediated_hydrogen_bond_networks",)),
    ("structural_qc", "Structural-integrity QC", _module_is("structural_integrity_qc"), ("structural_integrity_qc",)),
    ("hydrogen_bond_discovery", "Hydrogen-bond discovery", _module_is("hydrogen_bond_discovery"), ("hydrogen_bond_discovery",)),
    ("hydrogen_bond_comparisons", "Hydrogen-bond patterns and comparisons", _module_is("hydrogen_bond_comparison", "hydrogen_bond_patterns", "grouped_regularized_classification"), ("hydrogen_bond_comparison", "hydrogen_bond_patterns", "grouped_regularized_classification")),
    ("rdf", "RDF", _module_is("radial_distribution_functions"), ("radial_distribution_functions",)),
    ("ion_coordination", "Ion coordination", _module_is("ion_coordination_geometry"), ("ion_coordination_geometry",)),
    ("nucleic_acid", "Nucleic-acid geometry and structure", _module_is("nucleic_acid_geometry", "nucleic_acid_structure"), ("nucleic_acid_geometry", "nucleic_acid_structure")),
    ("individual_pca", "Individual PCA", _module_is("individual_pca"), ("individual_pca",)),
    ("secondary_structure", "Secondary structure", _module_is("secondary_structure"), ("secondary_structure",)),
    ("shared_pca_states", "Shared common PCA/FES/K-means/MSM", _scoped_modules("shared", "common_pca", "pca_fes_basins", "clustering_kmeans", "representative_frames", "state_coordinate_exports", "markov_state_models"), ("common_pca", "pca_fes_basins", "clustering_kmeans", "representative_frames", "state_coordinate_exports", "markov_state_models")),
    ("system_pca_states", "Per-system PCA/FES/K-means/MSM", _scoped_modules("per_system", "common_pca", "pca_fes_basins", "clustering_kmeans", "representative_frames", "state_coordinate_exports", "markov_state_models"), ("common_pca", "pca_fes_basins", "clustering_kmeans", "representative_frames", "state_coordinate_exports", "markov_state_models")),
    ("shared_pam_family", "Shared PAM/mwPAM/AP/mean-shift", _alternative_methods("shared", ("pam", "mwpam", "affinity_propagation", "mean_shift")), ("alternative_clustering",)),
    ("shared_gaussian_mixtures", "Shared Gaussian-mixture methods", _alternative_methods("shared", ("gaussian_mixture", "variational_gaussian_mixture")), ("alternative_clustering",)),
    ("system_pam_family", "Per-system PAM/mwPAM/AP", _alternative_methods("per_system", ("pam", "mwpam", "affinity_propagation")), ("alternative_clustering",)),
    ("system_gaussian_mixtures", "Per-system Gaussian mixtures", _alternative_methods("per_system", ("gaussian_mixture", "variational_gaussian_mixture")), ("alternative_clustering",)),
    ("system_mean_shift", "Per-system mean shift", _alternative_methods("per_system", ("mean_shift",)), ("alternative_clustering",)),
    ("imwkmeans", "Intelligent Minkowski-weighted K-means", _module_is("clustering_imwkmeans"), ("clustering_imwkmeans",)),
    ("hdbscan", "HDBSCAN", _module_is("clustering_hdbscan"), ("clustering_hdbscan",)),
    ("pald", "PaLD community analysis", _module_is("pald_community_analysis"), ("pald_community_analysis",)),
    ("information", "Nonlinear information and information dynamics", _module_is("generalized_correlation_and_information", "information_dynamics"), ("generalized_correlation_and_information", "information_dynamics")),
    ("tica", "TICA", _module_is("time_lagged_independent_component_analysis"), ("time_lagged_independent_component_analysis",)),
    ("correlation_networks", "Correlation networks", _module_is("correlation_networks"), ("correlation_networks",)),
    ("question_features", "Inferred and explicit observables/scalar states", _module_is("trajectory_features", "optional_observables", "scalar_feature_distributions", "scalar_threshold_states"), ("trajectory_features", "optional_observables", "scalar_feature_distributions", "scalar_threshold_states")),
    ("grouped_ml", "Grouped machine learning", _module_is("grouped_ml"), ("grouped_ml",)),
    ("rmsf_inference", "RMSF permutation inference", _module_is("rmsf_permutation_inference"), ("rmsf_permutation_inference",)),
    ("integrated_comparison", "Integrated comparison", _module_is("integrated_comparison"), ("integrated_comparison",)),
)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else []


def _numeric_sequence(value: object) -> List[float]:
    rows = []
    for item in _sequence(value):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            return []
        rows.append(float(item))
    return rows


def _integer_or_none(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _float_or_none(value: object) -> Optional[float]:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _effective_stride(task: Mapping[str, object]) -> int:
    explicit = _integer_or_none(task.get("effective_raw_integer_stride"))
    if explicit is not None:
        return explicit
    overall = _integer_or_none(task.get("overall_trajectory_integer_stride")) or 1
    projection = _integer_or_none(task.get("projection_integer_stride")) or 1
    local = _integer_or_none(task.get("integer_stride")) or 1
    return overall * projection * local


def _range_summary(values: Sequence[float], *, integer: bool = False) -> str:
    if not values:
        return "not reported"
    low = min(values)
    high = max(values)
    formatter = (lambda value: str(int(round(value)))) if integer else (
        lambda value: f"{value:.6g}"
    )
    if math.isclose(low, high):
        return f"{formatter(low)} each ({len(values)} replicas)"
    return f"{formatter(low)}-{formatter(high)} across {len(values)} replicas"


def _sampling_row(task: Mapping[str, object]) -> Dict[str, object]:
    task_id = str(task.get("task_id", task.get("balance_group", "unknown")))
    module_id = str(task.get("module_id", task_id.rsplit(":", 1)[-1]))
    source = _numeric_sequence(task.get("source_frames_per_replica"))
    if not source:
        source = _numeric_sequence(task.get("coordinate_cache_raw_source_frames_per_replica"))
    selected = _numeric_sequence(task.get("selected_physical_frames_per_replica"))
    if not selected:
        selected = _numeric_sequence(
            task.get("coordinate_cache_candidate_selected_frames_per_replica")
        )
    effective_stride = _effective_stride(task)
    raw_intervals = _numeric_sequence(task.get("frame_intervals_ns_per_replica"))
    if not raw_intervals:
        raw_intervals = _numeric_sequence(
            task.get("global_stride_raw_frame_intervals_ns_per_replica")
        )
    retained_intervals = [value * effective_stride for value in raw_intervals]
    minimum_frames = _integer_or_none(task.get("minimum_frames_per_replica"))
    minimum_total = _integer_or_none(task.get("minimum_selected_physical_frame_count"))
    below_floor = bool(task.get("source_limited_below_declared_minimum", False))
    return {
        "task_id": task_id,
        "module_id": module_id,
        "dependency_stage": _integer_or_none(task.get("dependency_stage")),
        "sampling_strategy": task.get("sampling_strategy"),
        "overall_trajectory_integer_stride": (
            _integer_or_none(task.get("overall_trajectory_integer_stride")) or 1
        ),
        "coordinate_cache_integer_stride": (
            _integer_or_none(task.get("coordinate_cache_integer_stride")) or 1
        ),
        "projection_integer_stride": (
            _integer_or_none(task.get("projection_integer_stride")) or 1
        ),
        "method_integer_stride": _integer_or_none(task.get("integer_stride")) or 1,
        "effective_raw_integer_stride": effective_stride,
        "source_frames_per_replica": [int(value) for value in source],
        "selected_frames_per_replica": [int(value) for value in selected],
        "source_frames_summary": _range_summary(source, integer=True),
        "selected_frames_summary": _range_summary(selected, integer=True),
        "selected_physical_frame_count": _integer_or_none(
            task.get("selected_physical_frame_count")
        ),
        "selected_member_observation_count": _integer_or_none(
            task.get("selected_member_observation_count")
        ),
        "coverage_fraction": _float_or_none(task.get("coverage_fraction")),
        "raw_frame_interval_ns_per_replica": raw_intervals,
        "retained_frame_spacing_ns_per_replica": retained_intervals,
        "retained_frame_spacing_summary": _range_summary(retained_intervals),
        "minimum_frames_per_replica": minimum_frames,
        "minimum_selected_physical_frame_count": minimum_total,
        "sampling_floor_status": "below_floor" if below_floor else "met_or_source_exhausted",
        "minimum_frame_scope": task.get("minimum_frame_scope"),
    }


def _status_item(identifier: str, reason: str, category: str) -> Dict[str, str]:
    return {"id": identifier, "category": category, "reason": reason}


def _explicitly_disabled(config: Mapping[str, object]) -> List[Dict[str, str]]:
    disabled: Dict[str, Dict[str, str]] = {}

    def add(identifier: str, reason: str, category: str) -> None:
        disabled[identifier] = _status_item(identifier, reason, category)

    for module_id, settings in _mapping(config.get("modules")).items():
        if _mapping(settings).get("enabled") is False:
            add(f"module:{module_id}", "disabled in modules configuration", "module")
    for view_id, settings in _mapping(config.get("views")).items():
        view = _mapping(settings)
        if view.get("enabled") is False:
            add(f"view:{view_id}", "conformational view disabled", "view")
        if view.get("state_trajectory_exports_enabled") is False:
            add(
                f"view_export:{view_id}",
                "state-trajectory export disabled for this view",
                "export",
            )
    clustering = _mapping(_mapping(config.get("clustering")).get("methods"))
    for method_id, settings in clustering.items():
        if _mapping(settings).get("enabled") is False:
            add(
                f"clustering_method:{method_id}",
                "clustering method disabled",
                "method",
            )
    pald = _mapping(_mapping(config.get("community_analysis")).get("pald"))
    if pald.get("enabled") is False:
        add("community_analysis:pald", "PaLD disabled", "method")
    if pald.get("community_msm_enabled") is False:
        add("community_analysis:pald_msm", "PaLD-derived MSM disabled", "method")

    sampling = _mapping(config.get("sampling"))
    for field, label in (
        ("b_vs_2b_sensitivity", "B-versus-2B sensitivity check"),
        ("optional_replica_diagnostics", "optional replica diagnostics"),
    ):
        if sampling.get(field) is False:
            add(f"sampling:{field}", f"{label} disabled", "diagnostic")
    reporting = _mapping(config.get("reporting"))
    for field, label in (
        ("finding_picker_enabled", "finding picker"),
        ("resource_table_enabled", "resource table"),
    ):
        if reporting.get(field) is False:
            add(f"reporting:{field}", f"{label} disabled", "reporting")
    inference = _mapping(config.get("inference"))
    for field, label in (
        ("automatic_chemical_context", "automatic chemical-context inference"),
        ("ion_site_classification_enabled", "ion-site classification"),
    ):
        if inference.get(field) is False:
            add(f"inference:{field}", f"{label} disabled", "inference")
    comparisons = _mapping(config.get("comparisons"))
    for field, label in (
        ("run_per_system_analysis", "per-system analysis"),
        ("run_shared_basis_comparisons", "shared-basis comparisons"),
    ):
        if comparisons.get(field) is False:
            add(f"comparisons:{field}", f"{label} disabled", "comparison")
    exports = _mapping(config.get("exports"))
    if exports.get("include_bulk_solvent") is False:
        add("exports:bulk_solvent", "bulk-solvent coordinate export disabled", "export")
    nearby = _mapping(exports.get("nearby_waters"))
    for target in ("representatives", "trajectories"):
        if _mapping(nearby.get(target)).get("mode") == "none":
            add(
                f"exports:nearby_waters:{target}",
                f"nearby-water export disabled for {target}",
                "export",
            )
    return [disabled[key] for key in sorted(disabled)]


def _deferred_items(
    coverage: Mapping[str, object], explicitly_disabled: Sequence[Mapping[str, str]]
) -> List[Dict[str, str]]:
    explicit_tokens = {item["id"].split(":", 1)[-1] for item in explicitly_disabled}
    rows = []
    for module_id, status in sorted(_mapping(coverage.get("module_status")).items()):
        entry = _mapping(status)
        if entry.get("status") != "deferred":
            continue
        reason = str(entry.get("reason", "deferred by prepared workflow"))
        if module_id in explicit_tokens or "disabled by" in reason.lower():
            continue
        if reason.lower().startswith("optional "):
            category = "optional_manual_utility"
        elif any(token in reason.lower() for token in (
            "no protein", "not present", "inapplicable", "was not supplied"
        )):
            category = "inapplicable"
        else:
            category = "dependency_or_question_specific"
        rows.append(_status_item(str(module_id), reason, category))
    return rows


def _compact_stride(values: Sequence[int]) -> str:
    unique = sorted(set(values))
    if not unique:
        return "Not scheduled"
    if len(unique) == 1:
        return str(unique[0])
    if len(unique) <= 4:
        return ", ".join(str(value) for value in unique)
    return f"{unique[0]}-{unique[-1]} ({len(unique)} distinct strides)"


def _family_stride_display(tasks: Sequence[Mapping[str, object]]) -> str:
    scoped = {
        scope: [
            int(task["effective_raw_integer_stride"])
            for task in tasks
            if _view_scope(task) == scope
        ]
        for scope in ("shared", "per_system", "other")
    }
    pieces = []
    if scoped["shared"]:
        pieces.append(f"{_compact_stride(scoped['shared'])} shared")
    if scoped["per_system"]:
        pieces.append(f"{_compact_stride(scoped['per_system'])} per system")
    if scoped["other"]:
        pieces.append(_compact_stride(scoped["other"]))
    return " / ".join(pieces)


def _disabled_tokens(disabled: Sequence[Mapping[str, str]]) -> set[str]:
    tokens = set()
    for item in disabled:
        identifier = str(item.get("id", ""))
        tokens.add(identifier)
        tokens.add(identifier.split(":", 1)[-1])
    return tokens


def _family_status_without_tasks(
    modules: Sequence[str],
    disabled: Sequence[Mapping[str, str]],
    deferred: Sequence[Mapping[str, str]],
    automatic: Sequence[str] = (),
) -> Tuple[str, str]:
    tokens = _disabled_tokens(disabled)
    aliases = {
        "clustering_hdbscan": "hdbscan",
        "pald_community_analysis": "pald",
    }
    if any(
        module_id in tokens or aliases.get(module_id) in tokens
        for module_id in modules
    ):
        return "off", "Off"
    module_set = set(modules)
    if module_set.intersection(automatic):
        return "on_no_trajectory", "On (final reporting; no trajectory stride)"
    relevant = [item for item in deferred if str(item.get("id")) in module_set]
    if relevant:
        if module_set == {"hydrogen_bonds"}:
            return (
                "optional_manual_utility",
                "Automatic chemistry discovery active; manual subset optional",
            )
        if module_set == {"representative_structures"}:
            return (
                "optional_manual_utility",
                "State representatives automatic; coordinate-space utility optional",
            )
        categories = {str(item.get("category")) for item in relevant}
        if categories == {"inapplicable"}:
            return "not_applicable", "Not applicable"
        return "deferred", "Deferred"
    return "not_scheduled", "Not scheduled"


def _family_summaries(
    tasks: Sequence[Mapping[str, object]],
    disabled: Sequence[Mapping[str, str]],
    deferred: Sequence[Mapping[str, str]],
    automatic: Sequence[str] = (),
) -> List[Dict[str, object]]:
    assigned: set[str] = set()
    summaries: List[Dict[str, object]] = []
    for family_id, label, matcher, modules in _ANALYSIS_FAMILIES:
        matches = [task for task in tasks if matcher(task)]
        assigned.update(str(task.get("task_id")) for task in matches)
        if matches:
            matched_modules = {str(task.get("module_id")) for task in matches}
            missing_module_states = []
            for module_id in modules:
                if module_id in matched_modules:
                    continue
                module_status, module_display = _family_status_without_tasks(
                    (module_id,), disabled, deferred, automatic
                )
                if module_status in {"off", "deferred", "not_applicable"}:
                    missing_module_states.append(
                        f"{module_id}: {module_display}"
                    )
            stride_display = _family_stride_display(matches)
            if missing_module_states:
                stride_display += "; " + "; ".join(missing_module_states)
            summaries.append({
                "family_id": family_id,
                "analysis_family": label,
                "status": "partly_on" if missing_module_states else "on",
                "effective_raw_stride_display": stride_display,
                "effective_raw_integer_strides": sorted({
                    int(task["effective_raw_integer_stride"]) for task in matches
                }),
                "task_count": len(matches),
                "below_floor_task_count": sum(
                    task.get("sampling_floor_status") == "below_floor"
                    for task in matches
                ),
                "task_ids": [str(task.get("task_id")) for task in matches],
            })
        else:
            status, display = _family_status_without_tasks(
                modules, disabled, deferred, automatic
            )
            summaries.append({
                "family_id": family_id,
                "analysis_family": label,
                "status": status,
                "effective_raw_stride_display": display,
                "effective_raw_integer_strides": [],
                "task_count": 0,
                "below_floor_task_count": 0,
                "task_ids": [],
            })

    unassigned: Dict[str, List[Mapping[str, object]]] = {}
    for task in tasks:
        if str(task.get("task_id")) not in assigned:
            unassigned.setdefault(str(task.get("module_id")), []).append(task)
    for module_id, matches in sorted(unassigned.items()):
        summaries.append({
            "family_id": f"other:{module_id}",
            "analysis_family": module_id.replace("_", " ").capitalize(),
            "status": "on",
            "effective_raw_stride_display": _family_stride_display(matches),
            "effective_raw_integer_strides": sorted({
                int(task["effective_raw_integer_stride"]) for task in matches
            }),
            "task_count": len(matches),
            "below_floor_task_count": sum(
                task.get("sampling_floor_status") == "below_floor"
                for task in matches
            ),
            "task_ids": [str(task.get("task_id")) for task in matches],
        })
    return summaries


def build_planning_report(root: Path) -> Dict[str, object]:
    """Combine sampling, feature, resource, and launcher decisions."""

    required = {
        name: root / name
        for name in (
            "analysis-config.json",
            "campaign-resource-plan.json",
            "sampling-plan.json",
            "module-coverage.json",
            "launcher-contract.json",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise PlanningReportError(
            "cannot build planning report; missing " + ", ".join(sorted(missing))
        )
    config = load_json(required["analysis-config.json"])
    resources = load_json(required["campaign-resource-plan.json"])
    sampling = load_json(required["sampling-plan.json"])
    coverage = load_json(required["module-coverage.json"])
    launcher = load_json(required["launcher-contract.json"])
    if not all(isinstance(value, dict) for value in (
        config, resources, sampling, coverage, launcher
    )):
        raise PlanningReportError("planning report inputs must be JSON objects")

    rows = [_sampling_row(_mapping(task)) for task in _sequence(resources.get("tasks"))]
    rows.sort(key=lambda row: (
        row["dependency_stage"] if row["dependency_stage"] is not None else 10**9,
        row["task_id"],
    ))
    disabled = _explicitly_disabled(config)
    deferred = _deferred_items(coverage, disabled)
    automatic = [
        str(module_id)
        for module_id, status in _mapping(coverage.get("module_status")).items()
        if _mapping(status).get("status") == "automatic"
    ]
    family_summaries = _family_summaries(rows, disabled, deferred, automatic)
    below = [row for row in rows if row["sampling_floor_status"] == "below_floor"]
    return {
        "planning_report_schema": "salsbury-user-planning-report-v1",
        "technical_status": resources.get("technical_status"),
        "scientific_status": resources.get("scientific_status", "planning only"),
        "execution_authorized": resources.get("execution_authorized"),
        "feasibility_status": resources.get("feasibility_status"),
        "resource_envelope": {
            "maximum_parallel_cpus": resources.get("maximum_parallel_cpus_input"),
            "maximum_parallel_memory_gib": resources.get(
                "maximum_parallel_memory_gib_input"
            ),
            "maximum_wall_hours": resources.get("maximum_wall_hours_input"),
            "estimated_selected_cpu_hours": resources.get(
                "estimated_selected_cpu_hours"
            ),
            "estimated_selected_wall_hours_lower_bound": resources.get(
                "estimated_selected_wall_hours_lower_bound"
            ),
        },
        "sampling": {
            "task_count": len(rows),
            "below_floor_task_count": len(below),
            "coordinate_cache": resources.get("coordinate_cache_coupling"),
            "analysis_families": family_summaries,
            "tasks": rows,
        },
        "features": {
            "explicitly_disabled": disabled,
            "deferred_or_inapplicable": deferred,
            "automatic_module_count": sum(
                1
                for status in _mapping(coverage.get("module_status")).values()
                if _mapping(status).get("status") == "automatic"
            ),
        },
        "launcher": {
            "active_adapter": _mapping(load_json(root / "execution-adapter.json")).get(
                "active_adapter"
            ),
            "contract_file": "launcher-contract.json",
            "phase_count": len(_sequence(launcher.get("phases"))),
            "custom_launcher_contract": (
                "Run phases in order; tasks within a phase may run concurrently within "
                "the declared CPU and memory envelope. Stop after any failed task."
            ),
        },
        "source_files": sorted(required),
        "interpretation": (
            "Effective raw stride is measured against the original trajectory. It is "
            "the product of upstream cache/projection strides and the method-local "
            "stride. Retained time spacing is provenance, not an independence claim."
        ),
    }


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        output.append(
            "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        )
    return "\n".join(output)


def render_planning_report_markdown(report: Mapping[str, object]) -> str:
    """Render a concise report while leaving exact per-replica lists in JSON."""

    envelope = _mapping(report.get("resource_envelope"))
    sampling = _mapping(report.get("sampling"))
    features = _mapping(report.get("features"))
    launcher = _mapping(report.get("launcher"))
    tasks = [_mapping(value) for value in _sequence(sampling.get("tasks"))]
    families = [
        _mapping(value) for value in _sequence(sampling.get("analysis_families"))
    ]
    lines = [
        "# Analysis planning report",
        "",
        f"- Feasibility: `{report.get('feasibility_status')}`",
        f"- Execution authorized by planner: `{report.get('execution_authorized')}`",
        f"- CPU cap: `{envelope.get('maximum_parallel_cpus')}`",
        f"- Aggregate memory cap: `{envelope.get('maximum_parallel_memory_gib')} GiB`",
        f"- Campaign wall-time cap: `{envelope.get('maximum_wall_hours')} hours`",
        f"- Planned sampling tasks: `{sampling.get('task_count')}`",
        f"- Tasks below a declared sampling floor: `{sampling.get('below_floor_task_count')}`",
        "",
        "## How to read the strides",
        "",
        "`Effective raw stride` is the spacing in the original trajectory. It includes "
        "any coordinate-cache stride, projection stride, and method-local stride. A value "
        "of 20 means original frames 0, 20, 40, and so on. Retained time spacing is "
        "reported as provenance and does not assert statistical independence.",
        "",
        "## Analysis-family status and effective raw strides",
        "",
        _markdown_table(
            ("Analysis family", "Status / effective raw stride", "Tasks", "Floor"),
            (
                (
                    row.get("analysis_family"),
                    row.get("effective_raw_stride_display"),
                    row.get("task_count"),
                    (
                        (
                            "met"
                            if not row.get("below_floor_task_count")
                            else f"{row.get('below_floor_task_count')} below floor"
                        )
                        if row.get("status") in {"on", "partly_on"}
                        else "not evaluated"
                    ),
                )
                for row in families
            ),
        ),
        "",
        "A number is the effective stride over the original trajectory, not merely "
        "a downstream projection or clustering-fit stride. `Off` means the resolved "
        "configuration disabled the family. `Deferred` means the module is available "
        "but this prepared workflow lacks a required upstream result or scientific "
        "definition. `Not applicable` means the supplied chemistry cannot support it.",
        "",
        "## Exact enabled tasks",
        "",
        _markdown_table(
            (
                "Task", "Method", "Raw stride", "Selected/replica",
                "Selected total", "Retained spacing (ns)", "Floor",
            ),
            (
                (
                    row.get("task_id"),
                    row.get("module_id"),
                    row.get("effective_raw_integer_stride"),
                    row.get("selected_frames_summary"),
                    row.get("selected_physical_frame_count"),
                    row.get("retained_frame_spacing_summary"),
                    row.get("sampling_floor_status"),
                )
                for row in tasks
            ),
        ),
        "",
        "Exact per-replica frame counts and every upstream stride component are in "
        "`planning-report.json`.",
        "",
        "## Explicitly turned off",
        "",
    ]
    disabled = [_mapping(value) for value in _sequence(features.get("explicitly_disabled"))]
    lines.append(
        _markdown_table(
            ("Setting", "Category", "Reason"),
            ((row.get("id"), row.get("category"), row.get("reason")) for row in disabled),
        ) if disabled else "Nothing was explicitly disabled."
    )
    lines.extend(["", "## Deferred, inapplicable, or optional utilities", ""])
    deferred = [
        _mapping(value) for value in _sequence(features.get("deferred_or_inapplicable"))
    ]
    lines.append(
        _markdown_table(
            ("Module", "Category", "Reason"),
            ((row.get("id"), row.get("category"), row.get("reason")) for row in deferred),
        ) if deferred else "No module was deferred and no optional utility was omitted."
    )
    lines.extend([
        "",
        "Deferred does not always mean disabled. Some stages wait for accepted upstream "
        "state definitions; others require a scientific question or software not present. "
        "Optional manual utilities are retained for explicit use but are not missing "
        "production analyses in this campaign.",
        "",
        "## Launcher handoff",
        "",
        f"The active adapter is `{launcher.get('active_adapter')}`. "
        "`launcher-contract.json` is scheduler-neutral. A custom launcher reads its phases "
        "in order, runs independent tasks within the CPU and memory caps, supplies the "
        "listed compatibility environment variables, and stops after any failed task.",
        "",
        "With `execution.submission_adapter` set to `custom`, run:",
        "",
        "```bash",
        "export SALSBURY_MD_ANALYSIS_CUSTOM_LAUNCHER=/absolute/path/to/my-launcher",
        "./run-custom.sh",
        "```",
        "",
        "The package passes the absolute path to `launcher-contract.json` as the launcher's "
        "only argument. It does not submit or execute scientific tasks itself in custom mode.",
        "",
    ])
    return "\n".join(lines)


def write_planning_report(root: Path) -> List[str]:
    """Write the JSON source of truth and its Markdown rendering."""

    report = build_planning_report(root)
    (root / "planning-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "planning-report.md").write_text(
        render_planning_report_markdown(report), encoding="utf-8"
    )
    return ["planning-report.json", "planning-report.md"]


def build_plan_matrix(
    scenarios: Sequence[Tuple[str, Mapping[str, object]]]
) -> Dict[str, object]:
    """Combine prepared reports into the family-by-envelope view used for review."""

    if not scenarios:
        raise PlanningReportError("at least one labeled planning report is required")
    labels = []
    family_order: List[str] = []
    family_labels: Dict[str, str] = {}
    values: Dict[str, Dict[str, str]] = {}
    envelopes: Dict[str, object] = {}
    for label, report in scenarios:
        if not label or label in labels:
            raise PlanningReportError("plan-matrix labels must be nonempty and unique")
        labels.append(label)
        envelopes[label] = report.get("resource_envelope")
        sampling = _mapping(report.get("sampling"))
        families = _sequence(sampling.get("analysis_families"))
        if not families:
            raise PlanningReportError(
                f"planning report {label!r} has no analysis-family summaries"
            )
        for item in families:
            family = _mapping(item)
            family_id = str(family.get("family_id"))
            if family_id not in family_order:
                family_order.append(family_id)
            family_labels[family_id] = str(family.get("analysis_family"))
            values.setdefault(family_id, {})[label] = str(
                family.get("effective_raw_stride_display")
            )
    rows = []
    for family_id in family_order:
        rows.append({
            "family_id": family_id,
            "analysis_family": family_labels[family_id],
            "scenarios": {
                label: values.get(family_id, {}).get(label, "Not scheduled")
                for label in labels
            },
        })
    return {
        "plan_matrix_schema": "salsbury-user-plan-matrix-v1",
        "scenario_labels": labels,
        "resource_envelopes": envelopes,
        "rows": rows,
        "cell_definition": (
            "Each numeric cell is an effective raw integer stride over the original "
            "trajectory. Off, Deferred, Not applicable, and Not scheduled are distinct."
        ),
    }


def render_plan_matrix_markdown(matrix: Mapping[str, object]) -> str:
    labels = [str(value) for value in _sequence(matrix.get("scenario_labels"))]
    rows = [_mapping(value) for value in _sequence(matrix.get("rows"))]
    return "\n".join([
        "# Analysis plan comparison",
        "",
        "Numbers are effective raw strides over the original trajectories.",
        "",
        _markdown_table(
            ("Analysis family", *labels),
            (
                (
                    row.get("analysis_family"),
                    *(
                        _mapping(row.get("scenarios")).get(label, "Not scheduled")
                        for label in labels
                    ),
                )
                for row in rows
            ),
        ),
        "",
    ])


def write_plan_matrix(
    scenarios: Sequence[Tuple[str, Path]], output_path: Path
) -> Path:
    loaded = []
    for label, path in scenarios:
        report = load_json(path)
        if not isinstance(report, dict):
            raise PlanningReportError(f"planning report is not a JSON object: {path}")
        loaded.append((label, report))
    matrix = build_plan_matrix(loaded)
    if output_path.suffix.lower() == ".json":
        rendered = json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    else:
        rendered = render_plan_matrix_markdown(matrix)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path
