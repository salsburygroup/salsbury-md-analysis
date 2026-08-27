"""Versioned, fail-closed configuration for zero-input analysis preparation."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .manifests import load_json


class AnalysisConfigError(ValueError):
    """Raised when a preparation configuration is ambiguous or unsafe."""


COMMAND_MODULES = {
    "structural-qc": "structural_integrity_qc",
    "rmsd-rg": "replica_rmsd_rg",
    "rmsf": "pooled_rmsf",
    "dccm": "dccm",
    "individual-pca": "individual_pca",
    "common-pca": "common_pca",
    "information-correlation": "generalized_correlation_and_information",
    "information-dynamics": "information_dynamics",
    "correlation-networks": "correlation_networks",
    "tica": "time_lagged_independent_component_analysis",
    "pca-fes-basins": "pca_fes_basins",
    "cluster-kmeans": "clustering_kmeans",
    "cluster-hdbscan": "clustering_hdbscan",
    "cluster-imwkmeans": "clustering_imwkmeans",
    "alternative-clustering": "alternative_clustering",
    "pald-community": "pald_community_analysis",
    "representative-frames": "representative_frames",
    "state-coordinate-exports": "state_coordinate_exports",
    "markov-models": "markov_state_models",
    "grouped-ml": "grouped_ml",
    "dihedrals": "dihedral_distributions",
    "hydrogen-bond-discovery": "hydrogen_bond_discovery",
    "water-mediated-hydrogen-bonds": "water_mediated_hydrogen_bond_networks",
    "secondary-structure": "secondary_structure",
    "sasa": "solvent_accessible_surface_area",
    "convergence": "convergence_uncertainty",
    "trajectory-features": "trajectory_features",
    "observables": "optional_observables",
    "rdf": "radial_distribution_functions",
    "scalar-distributions": "scalar_feature_distributions",
    "scalar-threshold-states": "scalar_threshold_states",
    "nucleic-acid-structure": "nucleic_acid_structure",
    "nucleic-acid-geometry": "nucleic_acid_geometry",
    "ion-geometry": "ion_coordination_geometry",
    "ion-atmosphere": "ion_atmosphere",
}

DEFINITION_MODULES = {
    "structural_qc": "structural_integrity_qc",
    **{
        module_id: module_id for module_id in COMMAND_MODULES.values()
        if module_id != "structural_integrity_qc"
    },
}

DEPENDENCIES = {
    "generalized_correlation_and_information": {"common_pca"},
    "information_dynamics": {"common_pca"},
    "time_lagged_independent_component_analysis": {"common_pca"},
    "pca_fes_basins": {"common_pca"},
    "clustering_kmeans": {"common_pca"},
    "clustering_hdbscan": {"common_pca"},
    "clustering_imwkmeans": {"common_pca"},
    "alternative_clustering": {"common_pca"},
    "pald_community_analysis": {"common_pca"},
    "representative_frames": {"pca_fes_basins"},
    "state_coordinate_exports": {"pca_fes_basins"},
    "markov_state_models": {"pca_fes_basins"},
    "grouped_ml": {"clustering_kmeans"},
    "correlation_networks": {"dccm"},
    "convergence_uncertainty": {"replica_rmsd_rg"},
}

PROTECTED_MODULES = {"provenance_manifest", "preflight_inventory", "common_atom_mapping"}


CLUSTERING_METHODS = {
    "kmeans": {"module_id": "clustering_kmeans", "algorithm": None},
    "hdbscan": {"module_id": "clustering_hdbscan", "algorithm": None},
    "intelligent_minkowski_weighted_kmeans": {
        "module_id": "clustering_imwkmeans", "algorithm": None,
    },
    "pam": {"module_id": "alternative_clustering", "algorithm": "pam"},
    "minkowski_weighted_pam": {
        "module_id": "alternative_clustering", "algorithm": "mwpam",
    },
    "ward": {"module_id": "alternative_clustering", "algorithm": "ward"},
    "gaussian_mixture": {
        "module_id": "alternative_clustering", "algorithm": "gaussian_mixture",
    },
    "variational_gaussian_mixture": {
        "module_id": "alternative_clustering",
        "algorithm": "variational_gaussian_mixture",
    },
    "affinity_propagation": {
        "module_id": "alternative_clustering", "algorithm": "affinity_propagation",
    },
    "mean_shift": {
        "module_id": "alternative_clustering", "algorithm": "mean_shift",
    },
    "quality_threshold": {
        "module_id": "alternative_clustering", "algorithm": "quality_threshold",
    },
}

_ALTERNATIVE_METHOD_ORDER = [
    "pam", "minkowski_weighted_pam", "ward", "gaussian_mixture",
    "variational_gaussian_mixture", "affinity_propagation", "mean_shift",
    "quality_threshold",
]


def default_analysis_config(
    module_ids: Sequence[str], view_ids: Sequence[str]
) -> Dict[str, object]:
    return {
        "config_schema": "salsbury-analysis-config-v1",
        "default_module_policy": "all_applicable",
        "modules": {
            module_id: {"enabled": True, "options": {}}
            for module_id in sorted(module_ids)
        },
        "views": {
            view_id: {
                "enabled": view_id != "macromolecular_trace",
                "state_trajectory_exports_enabled": (
                    view_id != "macromolecular_trace"
                ),
                "module_options": {},
            }
            for view_id in sorted(view_ids)
        },
        "reporting": {
            "resource_table_enabled": True,
            "finding_picker_enabled": True,
            "maximum_findings": 50,
        },
        "comparisons": {
            "mode": "all_pairs",
            "reference_system_id": None,
            "multiple_testing": "benjamini_hochberg",
            "alpha": 0.05,
            "run_per_system_analysis": True,
            "run_shared_basis_comparisons": True,
        },
        "sampling": {
            "strategy": "balanced_deterministic_stride",
            "preserve_replica_balance": True,
            "b_vs_2b_sensitivity": False,
            "optional_replica_diagnostics": False,
        },
        "clustering": {
            "feature_space": "tica",
            "methods": {
                method: {"enabled": method != "hdbscan"}
                for method in CLUSTERING_METHODS
            },
        },
        "community_analysis": {
            "pald": {
                "enabled": False,
                "community_msm_enabled": False,
            },
        },
        "inference": {
            "automatic_chemical_context": True,
            "ion_site_classification_enabled": True,
        },
        "oligomers": {
            "detect_automatically": True,
            "pool_equivalent_members": True,
            "members_are_independent_replicas": False,
        },
        "exports": {
            "payload": "complete_solute",
            "include_hydrogens": True,
            "include_ligands": True,
            "include_cofactors": True,
            "include_bound_ions": True,
            "include_bulk_solvent": False,
            "nearby_waters": {
                "representatives": {
                    "mode": "interaction",
                    "distance_cutoff_angstrom": 5.0,
                    "minimum_state_occupancy": 0.0,
                },
                "trajectories": {
                    "mode": "none",
                    "distance_cutoff_angstrom": 5.0,
                    "minimum_state_occupancy": 0.0,
                },
            },
        },
        "execution": {
            "maximum_parallel_cpus": 16,
            "maximum_hours_per_cpu": 24.0,
            "maximum_memory_gib": 128.0,
            "maximum_scratch_gib": 512.0,
            "planning_utilization": 0.85,
            "pilot_budget_fraction": 0.05,
            "finalization_headroom_fraction": 0.05,
            "time_safety_factor": 1.5,
            "memory_safety_factor": 1.25,
            "censored_timeout_safety_factor": 1.5,
            "fail_if_minimum_coverage_unaffordable": True,
            "submission_adapter": "local",
            "slurm_profile": None,
            "maximum_total_cpu_hours": 384.0,
            "coordinate_cache": "auto",
            "coordinate_cache_materialization": "planned_strided",
            "coordinate_cache_full_scan_fraction": 1.0,
            "overall_stride_candidates": [1, 2, 3, 4, 5, 10, 20, 100],
            "resource_calibration_catalog": None,
        },
    }


def load_analysis_config(
    path: Path | None, module_ids: Sequence[str], view_ids: Sequence[str]
) -> Dict[str, object]:
    config = default_analysis_config(module_ids, view_ids)
    if path is None:
        return config
    supplied = load_json(Path(path).expanduser().resolve(strict=True))
    if not isinstance(supplied, dict):
        raise AnalysisConfigError("analysis config must be a JSON object")
    allowed_top = {
        "config_schema", "default_module_policy", "modules", "views",
        "reporting", "comparisons", "sampling", "oligomers", "exports",
        "execution", "inference", "clustering", "community_analysis",
    }
    unknown = sorted(set(supplied).difference(allowed_top))
    if unknown:
        raise AnalysisConfigError("analysis config has unknown fields: " + ", ".join(unknown))
    if supplied.get("config_schema") != "salsbury-analysis-config-v1":
        raise AnalysisConfigError("config_schema must be salsbury-analysis-config-v1")
    if supplied.get("default_module_policy", "all_applicable") != "all_applicable":
        raise AnalysisConfigError("default_module_policy must be all_applicable")
    known_modules = set(module_ids)
    raw_modules = supplied.get("modules", {})
    if not isinstance(raw_modules, dict):
        raise AnalysisConfigError("modules must be an object")
    for module_id, raw in raw_modules.items():
        if module_id not in known_modules:
            raise AnalysisConfigError(f"unknown module in analysis config: {module_id}")
        if not isinstance(raw, dict) or set(raw).difference({"enabled", "options"}):
            raise AnalysisConfigError(f"module {module_id} accepts only enabled and options")
        enabled = raw.get("enabled", True)
        options = raw.get("options", {})
        if not isinstance(enabled, bool) or not isinstance(options, dict):
            raise AnalysisConfigError(f"module {module_id} has invalid enabled/options values")
        if module_id in PROTECTED_MODULES and not enabled:
            raise AnalysisConfigError(f"preparation infrastructure module {module_id} cannot be disabled")
        config["modules"][module_id] = {  # type: ignore[index]
            "enabled": enabled, "options": deepcopy(options)
        }
    raw_views = supplied.get("views", {})
    if not isinstance(raw_views, dict):
        raise AnalysisConfigError("views must be an object")
    known_views = set(view_ids)
    for view_id, raw in raw_views.items():
        if view_id not in known_views:
            raise AnalysisConfigError(f"unknown conformational view in config: {view_id}")
        allowed_view_fields = {
            "enabled", "state_trajectory_exports_enabled",
            "state_coordinate_exports_enabled", "module_options",
        }
        if not isinstance(raw, dict) or set(raw).difference(allowed_view_fields):
            raise AnalysisConfigError(
                f"view {view_id} accepts only enabled, "
                "state_trajectory_exports_enabled, and module_options"
            )
        default_view = config["views"][view_id]  # type: ignore[index]
        assert isinstance(default_view, dict)
        enabled = raw.get("enabled", default_view["enabled"])
        old_export_flag = raw.get("state_coordinate_exports_enabled")
        new_export_flag = raw.get("state_trajectory_exports_enabled")
        if (
            old_export_flag is not None
            and new_export_flag is not None
            and old_export_flag != new_export_flag
        ):
            raise AnalysisConfigError(
                f"view {view_id} has conflicting legacy coordinate-export and "
                "trajectory-export flags"
            )
        exports_enabled = (
            new_export_flag if new_export_flag is not None else
            old_export_flag if old_export_flag is not None else
            default_view["state_trajectory_exports_enabled"]
        )
        module_options = raw.get("module_options", {})
        if (
            not isinstance(enabled, bool)
            or not isinstance(exports_enabled, bool)
            or not isinstance(module_options, dict)
        ):
            raise AnalysisConfigError(f"view {view_id} has invalid configuration")
        if any(module_id not in known_modules or not isinstance(value, dict)
               for module_id, value in module_options.items()):
            raise AnalysisConfigError(f"view {view_id} has invalid module_options")
        config["views"][view_id] = {  # type: ignore[index]
            "enabled": enabled,
            "state_trajectory_exports_enabled": exports_enabled,
            "module_options": deepcopy(module_options),
        }
    for section, allowed in (
        ("reporting", {"resource_table_enabled", "finding_picker_enabled", "maximum_findings"}),
        ("comparisons", {
            "mode", "reference_system_id", "multiple_testing", "alpha",
            "run_per_system_analysis", "run_shared_basis_comparisons",
        }),
    ):
        raw = supplied.get(section, {})
        if not isinstance(raw, dict) or set(raw).difference(allowed):
            raise AnalysisConfigError(f"{section} configuration is invalid")
        config[section].update(deepcopy(raw))  # type: ignore[union-attr]
    reporting = config["reporting"]
    comparisons = config["comparisons"]
    assert isinstance(reporting, dict) and isinstance(comparisons, dict)
    if not all(isinstance(reporting[key], bool) for key in ("resource_table_enabled", "finding_picker_enabled")):
        raise AnalysisConfigError("reporting enable flags must be boolean")
    if isinstance(reporting["maximum_findings"], bool) or not isinstance(reporting["maximum_findings"], int) or reporting["maximum_findings"] <= 0:
        raise AnalysisConfigError("reporting.maximum_findings must be positive integer")
    if comparisons["mode"] not in {"all_pairs", "reference_vs_all"}:
        raise AnalysisConfigError("comparisons.mode must be all_pairs or reference_vs_all")
    if comparisons["multiple_testing"] != "benjamini_hochberg":
        raise AnalysisConfigError("comparisons.multiple_testing must be benjamini_hochberg")
    alpha = comparisons["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0.0 < float(alpha) < 1.0:
        raise AnalysisConfigError("comparisons.alpha must be between zero and one")
    if comparisons["mode"] == "reference_vs_all" and not comparisons["reference_system_id"]:
        raise AnalysisConfigError("reference_vs_all requires reference_system_id")
    for field in ("run_per_system_analysis", "run_shared_basis_comparisons"):
        if not isinstance(comparisons[field], bool):
            raise AnalysisConfigError(f"comparisons.{field} must be boolean")

    raw_sampling = supplied.get("sampling", {})
    allowed_sampling = {
        "strategy", "preserve_replica_balance", "b_vs_2b_sensitivity",
        "optional_replica_diagnostics",
    }
    if not isinstance(raw_sampling, dict) or set(raw_sampling).difference(allowed_sampling):
        raise AnalysisConfigError("sampling configuration is invalid")
    sampling = config["sampling"]
    assert isinstance(sampling, dict)
    sampling.update(deepcopy(raw_sampling))
    if sampling["strategy"] != "balanced_deterministic_stride":
        raise AnalysisConfigError(
            "sampling.strategy must be balanced_deterministic_stride"
        )
    for field in (
        "preserve_replica_balance", "b_vs_2b_sensitivity",
        "optional_replica_diagnostics",
    ):
        if not isinstance(sampling[field], bool):
            raise AnalysisConfigError(f"sampling.{field} must be boolean")

    raw_clustering = supplied.get("clustering", {})
    if (
        not isinstance(raw_clustering, dict)
        or set(raw_clustering).difference({"feature_space", "methods"})
    ):
        raise AnalysisConfigError("clustering configuration is invalid")
    clustering = config["clustering"]
    assert isinstance(clustering, dict)
    if "feature_space" in raw_clustering:
        clustering["feature_space"] = raw_clustering["feature_space"]
    if clustering["feature_space"] not in {"tica", "common_pca"}:
        raise AnalysisConfigError(
            "clustering.feature_space must be tica or common_pca"
        )
    raw_methods = raw_clustering.get("methods", {})
    if not isinstance(raw_methods, dict):
        raise AnalysisConfigError("clustering.methods must be an object")
    unknown_methods = sorted(set(raw_methods).difference(CLUSTERING_METHODS))
    if unknown_methods:
        raise AnalysisConfigError(
            "unknown clustering methods: " + ", ".join(unknown_methods)
        )
    methods = clustering["methods"]
    assert isinstance(methods, dict)
    for method, raw in raw_methods.items():
        if not isinstance(raw, dict) or set(raw) != {"enabled"}:
            raise AnalysisConfigError(
                f"clustering method {method} requires exactly one enabled flag"
            )
        if not isinstance(raw["enabled"], bool):
            raise AnalysisConfigError(
                f"clustering method {method} enabled flag must be boolean"
            )
        methods[method] = {"enabled": raw["enabled"]}

    raw_community = supplied.get("community_analysis", {})
    if (
        not isinstance(raw_community, dict)
        or set(raw_community).difference({"pald"})
    ):
        raise AnalysisConfigError("community_analysis configuration is invalid")
    raw_pald = raw_community.get("pald", {})
    if (
        not isinstance(raw_pald, dict)
        or set(raw_pald).difference({"enabled", "community_msm_enabled"})
    ):
        raise AnalysisConfigError("community_analysis.pald is invalid")
    community = config["community_analysis"]
    assert isinstance(community, dict)
    pald = community["pald"]
    assert isinstance(pald, dict)
    pald.update(deepcopy(raw_pald))
    if not all(
        isinstance(pald[field], bool)
        for field in ("enabled", "community_msm_enabled")
    ):
        raise AnalysisConfigError("community_analysis.pald flags must be boolean")
    if pald["community_msm_enabled"] and not pald["enabled"]:
        raise AnalysisConfigError(
            "PaLD community MSM cannot be enabled when PaLD is disabled"
        )

    raw_inference = supplied.get("inference", {})
    allowed_inference = {
        "automatic_chemical_context", "ion_site_classification_enabled",
    }
    if (
        not isinstance(raw_inference, dict)
        or set(raw_inference).difference(allowed_inference)
    ):
        raise AnalysisConfigError("inference configuration is invalid")
    inference = config["inference"]
    assert isinstance(inference, dict)
    inference.update(deepcopy(raw_inference))
    if not all(isinstance(inference[field], bool) for field in allowed_inference):
        raise AnalysisConfigError("inference flags must be boolean")

    raw_oligomers = supplied.get("oligomers", {})
    allowed_oligomers = {
        "detect_automatically", "pool_equivalent_members",
        "members_are_independent_replicas",
    }
    if not isinstance(raw_oligomers, dict) or set(raw_oligomers).difference(allowed_oligomers):
        raise AnalysisConfigError("oligomers configuration is invalid")
    oligomers = config["oligomers"]
    assert isinstance(oligomers, dict)
    oligomers.update(deepcopy(raw_oligomers))
    if not all(isinstance(oligomers[field], bool) for field in allowed_oligomers):
        raise AnalysisConfigError("oligomers flags must be boolean")
    if oligomers["members_are_independent_replicas"]:
        raise AnalysisConfigError(
            "equivalent oligomer members cannot be treated as independent replicas"
        )

    raw_exports = supplied.get("exports", {})
    allowed_exports = {
        "payload", "include_hydrogens", "include_ligands",
        "include_cofactors", "include_bound_ions", "include_bulk_solvent",
        "nearby_waters",
    }
    if not isinstance(raw_exports, dict) or set(raw_exports).difference(allowed_exports):
        raise AnalysisConfigError("exports configuration is invalid")
    exports = config["exports"]
    assert isinstance(exports, dict)
    for field, value in raw_exports.items():
        if field != "nearby_waters":
            exports[field] = deepcopy(value)
    if exports["payload"] not in {"complete_solute", "feature_atoms"}:
        raise AnalysisConfigError(
            "exports.payload must be complete_solute or feature_atoms"
        )
    for field in (
        "include_hydrogens", "include_ligands", "include_cofactors",
        "include_bound_ions", "include_bulk_solvent",
    ):
        if not isinstance(exports[field], bool):
            raise AnalysisConfigError(f"exports.{field} must be boolean")
    raw_nearby = raw_exports.get("nearby_waters", {})
    if not isinstance(raw_nearby, dict) or set(raw_nearby).difference(
        {"representatives", "trajectories"}
    ):
        raise AnalysisConfigError("exports.nearby_waters configuration is invalid")
    nearby = exports["nearby_waters"]
    assert isinstance(nearby, dict)
    allowed_nearby_fields = {
        "mode", "distance_cutoff_angstrom", "minimum_state_occupancy"
    }
    for target in ("representatives", "trajectories"):
        raw_target = raw_nearby.get(target, {})
        if not isinstance(raw_target, dict) or set(raw_target).difference(
            allowed_nearby_fields
        ):
            raise AnalysisConfigError(
                f"exports.nearby_waters.{target} configuration is invalid"
            )
        target_config = nearby[target]
        assert isinstance(target_config, dict)
        target_config.update(deepcopy(raw_target))
        allowed_modes = (
            {"none", "distance", "interaction", "persistent", "union"}
            if target == "representatives" else {"none", "fixed_identity"}
        )
        if target_config["mode"] not in allowed_modes:
            raise AnalysisConfigError(
                f"exports.nearby_waters.{target}.mode is invalid"
            )
        cutoff = target_config["distance_cutoff_angstrom"]
        if (
            isinstance(cutoff, bool) or not isinstance(cutoff, (int, float))
            or not math.isfinite(float(cutoff)) or float(cutoff) <= 0.0
        ):
            raise AnalysisConfigError(
                f"exports.nearby_waters.{target}.distance_cutoff_angstrom "
                "must be finite and positive"
            )
        occupancy = target_config["minimum_state_occupancy"]
        if (
            isinstance(occupancy, bool) or not isinstance(occupancy, (int, float))
            or not math.isfinite(float(occupancy))
            or not 0.0 <= float(occupancy) <= 1.0
        ):
            raise AnalysisConfigError(
                f"exports.nearby_waters.{target}.minimum_state_occupancy "
                "must be between zero and one"
            )
    raw_execution = supplied.get("execution", {})
    allowed_execution = {
        "maximum_parallel_cpus", "maximum_hours_per_cpu",
        "maximum_memory_gib", "maximum_scratch_gib", "planning_utilization",
        "pilot_budget_fraction", "finalization_headroom_fraction",
        "time_safety_factor", "memory_safety_factor",
        "censored_timeout_safety_factor",
        "fail_if_minimum_coverage_unaffordable",
        "submission_adapter", "slurm_profile", "coordinate_cache",
        "coordinate_cache_materialization",
        "coordinate_cache_full_scan_fraction", "overall_stride_candidates",
        "resource_calibration_catalog", "maximum_total_cpu_hours",
    }
    if not isinstance(raw_execution, dict) or set(raw_execution).difference(allowed_execution):
        raise AnalysisConfigError("execution configuration is invalid")
    execution = config["execution"]
    assert isinstance(execution, dict)
    execution.update(deepcopy(raw_execution))
    maximum_cpus = execution["maximum_parallel_cpus"]
    if isinstance(maximum_cpus, bool) or not isinstance(maximum_cpus, int) or maximum_cpus <= 0:
        raise AnalysisConfigError("execution.maximum_parallel_cpus must be a positive integer")
    for field in (
        "maximum_hours_per_cpu", "maximum_memory_gib", "maximum_scratch_gib"
    ):
        value = execution[field]
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0.0
        ):
            raise AnalysisConfigError(f"execution.{field} must be finite and positive")
    for field in (
        "planning_utilization", "pilot_budget_fraction",
        "finalization_headroom_fraction",
    ):
        value = execution[field]
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        ):
            raise AnalysisConfigError(f"execution.{field} must be between zero and one")
    reserved_fraction = (
        float(execution["pilot_budget_fraction"])
        + float(execution["finalization_headroom_fraction"])
    )
    if reserved_fraction >= float(execution["planning_utilization"]):
        raise AnalysisConfigError(
            "execution pilot plus finalization headroom fractions must be smaller "
            "than planning_utilization"
        )
    for field in (
        "time_safety_factor", "memory_safety_factor",
        "censored_timeout_safety_factor",
    ):
        value = execution[field]
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0.0
        ):
            raise AnalysisConfigError(f"execution.{field} must be finite and positive")
    if not isinstance(execution["fail_if_minimum_coverage_unaffordable"], bool):
        raise AnalysisConfigError(
            "execution.fail_if_minimum_coverage_unaffordable must be boolean"
        )
    if execution["submission_adapter"] not in {"unspecified", "local", "slurm"}:
        raise AnalysisConfigError(
            "execution.submission_adapter must be unspecified, local, or slurm"
        )
    if execution["submission_adapter"] == "unspecified":
        # Normalize the legacy placeholder to the safe scheduler-free default.
        execution["submission_adapter"] = "local"
    slurm_profile = execution["slurm_profile"]
    if slurm_profile is not None and (
        not isinstance(slurm_profile, str) or not slurm_profile.strip()
    ):
        raise AnalysisConfigError(
            "execution.slurm_profile must be null or a nonempty path"
        )
    if execution["submission_adapter"] == "slurm" and slurm_profile is None:
        raise AnalysisConfigError(
            "execution.slurm_profile is required when submission_adapter is slurm"
        )
    if slurm_profile is not None:
        candidate = Path(slurm_profile).expanduser()
        if not candidate.is_absolute():
            candidate = Path(path).expanduser().resolve(strict=True).parent / candidate
        execution["slurm_profile"] = str(candidate.resolve(strict=True))
    if execution["coordinate_cache"] not in {"auto", "off", "required"}:
        raise AnalysisConfigError(
            "execution.coordinate_cache must be auto, off, or required"
        )
    if execution["coordinate_cache_materialization"] not in {
        "planned_strided", "lossless"
    }:
        raise AnalysisConfigError(
            "execution.coordinate_cache_materialization must be "
            "planned_strided or lossless"
        )
    scan_fraction = execution["coordinate_cache_full_scan_fraction"]
    if (
        isinstance(scan_fraction, bool)
        or not isinstance(scan_fraction, (int, float))
        or not math.isfinite(float(scan_fraction))
        or not 0.0 <= float(scan_fraction) <= 1.0
    ):
        raise AnalysisConfigError(
            "execution.coordinate_cache_full_scan_fraction must be between zero and one"
        )
    candidates = execution["overall_stride_candidates"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in candidates
        )
        or len(set(candidates)) != len(candidates)
    ):
        raise AnalysisConfigError(
            "execution.overall_stride_candidates must be a nonempty list of "
            "unique positive integers"
        )
    execution["overall_stride_candidates"] = sorted(candidates)
    calibration_catalog = execution["resource_calibration_catalog"]
    if calibration_catalog is not None and (
        not isinstance(calibration_catalog, str) or not calibration_catalog.strip()
    ):
        raise AnalysisConfigError(
            "execution.resource_calibration_catalog must be null or a nonempty path"
        )
    if calibration_catalog is not None:
        candidate = Path(calibration_catalog).expanduser()
        if not candidate.is_absolute():
            candidate = Path(path).expanduser().resolve(strict=True).parent / candidate
        execution["resource_calibration_catalog"] = str(
            candidate.resolve(strict=True)
        )
    execution["maximum_total_cpu_hours"] = (
        int(maximum_cpus) * float(execution["maximum_hours_per_cpu"])
    )
    return config


def enabled_modules(config: Mapping[str, object]) -> set[str]:
    modules = config["modules"]
    assert isinstance(modules, dict)
    enabled = {
        module_id for module_id, row in modules.items()
        if isinstance(row, dict) and row.get("enabled") is True
    }
    dependencies = {key: set(value) for key, value in DEPENDENCIES.items()}
    clustering = config.get("clustering")
    feature_space = (
        clustering.get("feature_space") if isinstance(clustering, dict) else None
    )
    clustering_dependency = (
        "time_lagged_independent_component_analysis"
        if feature_space == "tica" else "common_pca"
    )
    for module_id in (
        "clustering_kmeans", "clustering_hdbscan", "clustering_imwkmeans",
        "alternative_clustering", "pald_community_analysis",
    ):
        dependencies[module_id] = {clustering_dependency}
    methods = clustering.get("methods") if isinstance(clustering, dict) else None
    if isinstance(methods, dict):
        enabled_methods = {
            method for method, row in methods.items()
            if isinstance(row, dict) and row.get("enabled") is True
        }
        for method, contract in CLUSTERING_METHODS.items():
            if contract["algorithm"] is None and method not in enabled_methods:
                enabled.discard(str(contract["module_id"]))
        if not any(
            CLUSTERING_METHODS[method]["algorithm"] is not None
            for method in enabled_methods
        ):
            enabled.discard("alternative_clustering")
    community = config.get("community_analysis")
    pald = community.get("pald") if isinstance(community, dict) else None
    if not isinstance(pald, dict) or pald.get("enabled") is not True:
        enabled.discard("pald_community_analysis")
    changed = True
    while changed:
        changed = False
        for module_id in tuple(enabled):
            if not dependencies.get(module_id, set()).issubset(enabled):
                enabled.remove(module_id)
                changed = True
    return enabled


def make_memory_fit_config(
    config: Mapping[str, object], module_ids: Sequence[str]
) -> tuple[Dict[str, object], list[str], list[str]]:
    """Return an explicit reduced config for a user-approved memory fallback.

    The caller supplies modules whose technical minima exceed the requested
    per-campaign memory cap.  Dependency pruning is then materialized into the
    config so a user can see every resulting on/off decision; the original
    config is never mutated.
    """

    output = deepcopy(dict(config))
    modules = output.get("modules")
    if not isinstance(modules, dict):
        raise AnalysisConfigError("analysis config has no module mapping")
    requested = sorted(set(str(value) for value in module_ids))
    direct = []
    for module_id in requested:
        if module_id in {"coordinate_cache", "execution.coordinate_cache"}:
            execution = output.get("execution")
            if not isinstance(execution, dict):
                raise AnalysisConfigError("analysis config has no execution mapping")
            execution["coordinate_cache"] = "off"
            direct.append(module_id)
        elif module_id.startswith("clustering.methods.") and module_id.endswith(
            ".enabled"
        ):
            method = module_id[len("clustering.methods.") : -len(".enabled")]
            clustering = output.get("clustering")
            methods = (
                clustering.get("methods") if isinstance(clustering, dict) else None
            )
            row = methods.get(method) if isinstance(methods, dict) else None
            if not isinstance(row, dict):
                raise AnalysisConfigError(
                    f"memory fallback clustering switch is invalid: {module_id}"
                )
            row["enabled"] = False
            direct.append(module_id)
        elif module_id == "community_analysis.pald.enabled":
            community = output.get("community_analysis")
            pald = community.get("pald") if isinstance(community, dict) else None
            if not isinstance(pald, dict):
                raise AnalysisConfigError(
                    "memory fallback PaLD configuration is invalid"
                )
            pald["enabled"] = False
            pald["community_msm_enabled"] = False
            direct.append(module_id)
        elif module_id.startswith("modules.") and module_id.endswith(".enabled"):
            resolved_module_id = module_id[len("modules.") : -len(".enabled")]
            row = modules.get(resolved_module_id)
            if not isinstance(row, dict):
                raise AnalysisConfigError(
                    f"memory fallback module switch is invalid: {module_id}"
                )
            row["enabled"] = False
            direct.append(module_id)
        elif module_id in modules:
            row = modules[module_id]
            if not isinstance(row, dict):
                raise AnalysisConfigError(f"module {module_id} config is invalid")
            row["enabled"] = False
            direct.append(module_id)

    effective = enabled_modules(output)
    originally_enabled = {
        module_id for module_id, row in modules.items()
        if isinstance(row, dict) and row.get("enabled") is True
    }
    for module_id, row in modules.items():
        if isinstance(row, dict):
            row["enabled"] = module_id in effective
    if "common_pca" not in effective:
        views = output.get("views")
        if isinstance(views, dict):
            for row in views.values():
                if isinstance(row, dict):
                    row["enabled"] = False
                    row["state_trajectory_exports_enabled"] = False

    clustering = output.get("clustering")
    methods = clustering.get("methods") if isinstance(clustering, dict) else None
    if isinstance(methods, dict):
        for method, contract in CLUSTERING_METHODS.items():
            row = methods.get(method)
            if isinstance(row, dict) and str(contract["module_id"]) not in effective:
                row["enabled"] = False
    community = output.get("community_analysis")
    pald = community.get("pald") if isinstance(community, dict) else None
    if isinstance(pald, dict) and "pald_community_analysis" not in effective:
        pald["enabled"] = False
        pald["community_msm_enabled"] = False

    directly_disabled_module_ids = {
        value[len("modules.") : -len(".enabled")]
        for value in direct
        if value.startswith("modules.") and value.endswith(".enabled")
    } | {value for value in direct if value in modules}
    transitive = sorted(
        originally_enabled.difference(effective).difference(
            directly_disabled_module_ids
        )
    )
    return output, sorted(direct), transitive


def apply_module_configuration(
    definitions: Mapping[str, object], commands: Sequence[str],
    requested: Sequence[str], config: Mapping[str, object],
) -> tuple[Dict[str, object], list[str], list[str], Dict[str, str]]:
    enabled = enabled_modules(config)
    modules = config["modules"]
    assert isinstance(modules, dict)
    output = deepcopy(dict(definitions))
    disabled_reasons: Dict[str, str] = {}
    for definition_id in list(output):
        module_id = DEFINITION_MODULES.get(definition_id, definition_id)
        if module_id not in enabled:
            output.pop(definition_id)
            explicitly = isinstance(modules.get(module_id), dict) and modules[module_id].get("enabled") is False
            community = config.get("community_analysis")
            pald = community.get("pald") if isinstance(community, dict) else None
            if (
                module_id == "pald_community_analysis"
                and isinstance(pald, dict)
                and pald.get("enabled") is not True
            ):
                disabled_reasons[module_id] = "disabled by community_analysis.pald"
            else:
                disabled_reasons[module_id] = (
                    "disabled by analysis config" if explicitly
                    else "disabled because an upstream dependency is disabled"
                )
            continue
        raw = modules.get(module_id)
        options = raw.get("options", {}) if isinstance(raw, dict) else {}
        if options:
            value = output[definition_id]
            if not isinstance(value, dict):
                raise AnalysisConfigError(f"module {module_id} definition is not configurable")
            value.update(deepcopy(options))
    filtered_commands = [
        command for command in commands if COMMAND_MODULES.get(command, command) in enabled
    ]
    filtered_requested = [module_id for module_id in requested if module_id in enabled]

    clustering = config.get("clustering")
    if not isinstance(clustering, dict) or not isinstance(clustering.get("methods"), dict):
        raise AnalysisConfigError("analysis config has no valid clustering methods")
    method_rows = clustering["methods"]
    enabled_methods = {
        method for method, row in method_rows.items()
        if isinstance(row, dict) and row.get("enabled") is True
    }
    # Correlation-profile clustering uses the same optional HDBSCAN package as
    # conformational HDBSCAN. Keep the package-level default coherent: when
    # HDBSCAN is off, retain correlation networks but omit that one subanalysis.
    if "hdbscan" not in enabled_methods:
        correlation_networks = output.get("correlation_networks")
        if isinstance(correlation_networks, dict):
            correlation_networks.pop("profile_clustering", None)
    dedicated = {
        method: str(contract["module_id"])
        for method, contract in CLUSTERING_METHODS.items()
        if contract["algorithm"] is None
    }
    for method, module_id in dedicated.items():
        if method in enabled_methods:
            continue
        output.pop(module_id, None)
        filtered_commands = [
            command for command in filtered_commands
            if COMMAND_MODULES.get(command, command) != module_id
        ]
        filtered_requested = [
            requested_id for requested_id in filtered_requested
            if requested_id != module_id
        ]
        disabled_reasons[module_id] = (
            f"disabled by clustering.methods.{method}"
        )
    if "kmeans" not in enabled_methods:
        representative = output.get("representative_frames")
        if isinstance(representative, dict):
            representative["source"] = "pca_fes_basins"
        output.pop("grouped_ml", None)
        filtered_commands = [
            command for command in filtered_commands
            if COMMAND_MODULES.get(command, command) != "grouped_ml"
        ]
        filtered_requested = [
            requested_id for requested_id in filtered_requested
            if requested_id != "grouped_ml"
        ]
        disabled_reasons["grouped_ml"] = (
            "disabled because its current target contract requires KMeans"
        )
    alternative_algorithms = [
        str(CLUSTERING_METHODS[method]["algorithm"])
        for method in _ALTERNATIVE_METHOD_ORDER
        if method in enabled_methods
    ]
    alternative = output.get("alternative_clustering")
    if alternative_algorithms and isinstance(alternative, dict):
        alternative["algorithms"] = alternative_algorithms
    elif not alternative_algorithms:
        output.pop("alternative_clustering", None)
        filtered_commands = [
            command for command in filtered_commands
            if COMMAND_MODULES.get(command, command) != "alternative_clustering"
        ]
        filtered_requested = [
            requested_id for requested_id in filtered_requested
            if requested_id != "alternative_clustering"
        ]
        disabled_reasons["alternative_clustering"] = (
            "all alternative clustering methods disabled"
        )

    community = config.get("community_analysis")
    pald_config = community.get("pald") if isinstance(community, dict) else None
    pald = output.get("pald_community_analysis")
    if isinstance(pald, dict) and isinstance(pald_config, dict):
        pald["community_msm_enabled"] = bool(
            pald_config["community_msm_enabled"]
        )

    feature_space = str(clustering["feature_space"])
    for definition_id in (
        "clustering_kmeans", "clustering_hdbscan", "clustering_imwkmeans",
        "alternative_clustering",
        "pald_community_analysis",
    ):
        definition = output.get(definition_id)
        if not isinstance(definition, dict):
            continue
        definition["feature_source"] = feature_space
        if feature_space == "tica":
            definition.pop("trajectory_feature_columns", None)
        elif feature_space == "common_pca":
            definition.pop("trajectory_feature_columns", None)
    return output, filtered_commands, filtered_requested, disabled_reasons
