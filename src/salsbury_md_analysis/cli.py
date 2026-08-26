"""Machine-readable CLI for suite inspection and experimental analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .atom_mapping import (
    ATOM_SELECTIONS,
    MAPPING_POLICIES,
    AtomMappingError,
    map_common_atoms,
)
from .context import compile_project_context_file
from .coordinate_cache import build_coordinate_cache_safe
from .convergence import convergence_uncertainty_project_safe
from .clustering import (
    clustering_hdbscan_project_safe,
    clustering_imwkmeans_project_safe,
    clustering_kmeans_project_safe,
)
from .alternative_clustering import alternative_clustering_project_safe
from .pald_community import pald_community_analysis_project_safe
from .automatic_sampling import (
    AutomaticSamplingError,
    SIMULATION_KINDS,
    automatic_sampling_plan,
)
from .analysis_config import COMMAND_MODULES
from .correlation_networks import correlation_networks_project_safe
from .dccm import dccm_project_safe
from .dihedrals import dihedral_distributions_project_safe
from .hydrogen_bonds import hydrogen_bonds_project_safe
from .hydrogen_bond_discovery import hydrogen_bond_discovery_project_safe
from .hydrogen_bond_comparison import compare_hydrogen_bond_reports_file_safe
from .water_mediated_hydrogen_bonds import (
    water_mediated_hydrogen_bond_networks_project_safe,
)
from .grouped_ml import grouped_ml_project_safe
from .grouped_regularized_classification import (
    grouped_regularized_classification_project_safe,
)
from .integrated import integrated_comparison_project_safe
from .information import generalized_correlation_and_information_project_safe
from .information_dynamics import information_dynamics_project_safe
from .trajectory_features import trajectory_features_project_safe
from .hydrogen_bond_patterns import (
    HydrogenBondPatternError,
    encode_bond_patterns,
    hdbscan_jaccard,
    pam_jaccard,
)
from .representative_structures import (
    RepresentativeStructureError,
    representative_structures,
)
from .rmsf_inference import RMSFInferenceError, rmsf_permutation_test
from .manifests import (
    MANIFEST_KINDS,
    ManifestValidationError,
    inventory_system_inputs,
    load_json,
    validate_manifest,
)
from .msm import markov_state_models_project_safe
from .observables import optional_observables_project_safe
from .preflight import preflight_system
from .quickstart import (
    QuickstartError,
    QuickstartMemoryError,
    prepare_standard_analysis,
    prepare_standard_analysis_memory_fit,
)
from .comparative_quickstart import (
    prepare_comparative_analysis,
    prepare_comparative_analysis_memory_fit,
)
from .pca import common_pca_project_safe, individual_pca_project_safe
from .pca_fes import pca_fes_basins_project_safe
from .presentation import PresentationError, summarize_timeseries_presentations
from .registry import list_modules
from .representative_frames import representative_frames_project_safe
from .state_coordinate_exports import state_coordinate_exports_project_safe
from .rdf import radial_distribution_functions_project_safe
from .scalar_distributions import scalar_feature_distributions_project_safe
from .scalar_threshold_states import scalar_threshold_states_project_safe
from .nucleic_acid_structure import nucleic_acid_structure_project_safe
from .nucleic_acid_geometry import nucleic_acid_geometry_project_safe
from .ion_geometry import ion_coordination_geometry_project_safe
from .ion_atmosphere import ion_atmosphere_project_safe
from .regression import run_regression_case_safe
from .resource_planning import (
    ResourcePlanningError,
    calibrate_from_benchmarks,
    recommend_frame_budget,
)
from .resource_calibrations import (
    ResourceCalibrationError, build_resource_calibration_catalog,
    redact_resource_calibration_catalog,
)
from .rmsd_rg import replica_rmsd_rg_project_safe
from .rmsf import pooled_rmsf_project_safe
from .rmsf_visualization import RMSFVisualizationError, export_rmsf_visualization
from .secondary_structure import secondary_structure_project_safe
from .sasa import solvent_accessible_surface_area_project_safe
from .structural_qc import structural_qc_project_safe
from .tica import time_lagged_independent_component_analysis_project_safe
from .execution_resources import (
    analysis_report_sidecar,
    ExecutionResourceError,
    run_instrumented_coordinate_cache,
    run_instrumented_project_command,
    summarize_execution_resources,
)
from .execution_adapters import ExecutionAdapterError, run_local_workflow
from .finding_picker import FindingPickerError, prioritize_findings
from .slurm_capacity import (
    SlurmCapacityError,
    advise_slurm_capacity,
    render_capacity_markdown,
)


EXTENDED_PROJECT_COMMANDS = {
    "alternative-clustering": alternative_clustering_project_safe,
    "pald-community": pald_community_analysis_project_safe,
    "information-dynamics": information_dynamics_project_safe,
    "correlation-networks": correlation_networks_project_safe,
    "trajectory-features": trajectory_features_project_safe,
    "state-coordinate-exports": state_coordinate_exports_project_safe,
    "rdf": radial_distribution_functions_project_safe,
    "scalar-distributions": scalar_feature_distributions_project_safe,
    "scalar-threshold-states": scalar_threshold_states_project_safe,
    "hydrogen-bond-discovery": hydrogen_bond_discovery_project_safe,
    "water-mediated-hydrogen-bonds": water_mediated_hydrogen_bond_networks_project_safe,
    "grouped-regularized-classification": grouped_regularized_classification_project_safe,
    "nucleic-acid-structure": nucleic_acid_structure_project_safe,
    "nucleic-acid-geometry": nucleic_acid_geometry_project_safe,
    "ion-geometry": ion_coordination_geometry_project_safe,
    "ion-atmosphere": ion_atmosphere_project_safe,
}

INSTRUMENTABLE_PROJECT_COMMANDS = tuple(sorted(
    set(COMMAND_MODULES)
    | set(EXTENDED_PROJECT_COMMANDS)
    | {"hydrogen-bonds", "observables", "integrate"}
))

STANDALONE_METHOD_COMMANDS = {
    "hydrogen-bond-patterns": "hydrogen_bond_patterns",
    "representative-structures": "representative_structures",
    "rmsf-permutation": "rmsf_permutation_inference",
}


def _list_modules(as_json: bool) -> int:
    modules = list_modules()
    if as_json:
        print(json.dumps([module.as_dict() for module in modules], indent=2))
        return 0

    print(f"Salsbury MD Analysis {__version__}")
    print("STATUS       CATEGORY     STAGE  MODULE")
    for module in modules:
        stage = module.stage or "-"
        print(f"{module.status:12} {module.category:12} {stage:6} {module.module_id}")
    return 0


def _validation_payload(kind: str, path: Path, check_paths: bool) -> dict:
    try:
        data = load_json(path)
        validate_manifest(kind, data, source_path=path, check_paths=check_paths)
    except ManifestValidationError as exc:
        return {
            "valid": False,
            "kind": kind,
            "path": str(path.expanduser().resolve(strict=False)),
            "issues": list(exc.issues),
        }
    return {
        "valid": True,
        "kind": kind,
        "path": str(path.expanduser().resolve(strict=False)),
        "issues": [],
    }


def _validate_manifest_command(kind: str, path: Path, check_paths: bool, as_json: bool) -> int:
    result = _validation_payload(kind, path, check_paths)
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["valid"]:
        print(f"VALID {kind} manifest: {result['path']}")
    else:
        print(f"INVALID {kind} manifest: {result['path']}", file=sys.stderr)
        for issue in result["issues"]:
            print(f"- {issue}", file=sys.stderr)
    return 0 if result["valid"] else 2


def _inventory_system_command(path: Path, hash_content: bool) -> int:
    try:
        data = load_json(path)
        inventory = inventory_system_inputs(data, source_path=path, hash_content=hash_content)
    except ManifestValidationError as exc:
        payload = {
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "path": str(path.expanduser().resolve(strict=False)),
            "issues": list(exc.issues),
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


def _preflight_system_command(path: Path, hash_content: bool) -> int:
    try:
        data = load_json(path)
        report = preflight_system(data, source_path=path, hash_content=hash_content)
    except ManifestValidationError as exc:
        report = {
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "path": str(path.expanduser().resolve(strict=False)),
            "error_count": len(exc.issues),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "MANIFEST_INVALID", "message": issue}
                for issue in exc.issues
            ],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _map_common_atoms_command(
    reference: Path,
    targets: Sequence[Path],
    policy: str,
    selection: str,
    minimum_reference_coverage: float,
    hash_content: bool,
) -> int:
    try:
        report = map_common_atoms(
            reference,
            targets,
            policy=policy,
            selection=selection,
            minimum_reference_coverage=minimum_reference_coverage,
            hash_content=hash_content,
        )
    except AtomMappingError as exc:
        report = {
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "error_count": 1,
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "ATOM_MAPPING_FAILED", "message": str(exc)}
            ],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _compile_context_command(path: Path, hash_content: bool) -> int:
    try:
        report = compile_project_context_file(path, hash_content=hash_content)
    except ManifestValidationError as exc:
        report = {
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "path": str(path.expanduser().resolve(strict=False)),
            "error_count": len(exc.issues),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "CONTEXT_INVALID", "message": issue}
                for issue in exc.issues
            ],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _structural_qc_command(path: Path, hash_content: bool) -> int:
    report = structural_qc_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _rmsd_rg_command(path: Path, hash_content: bool) -> int:
    report = replica_rmsd_rg_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _rmsf_command(path: Path, hash_content: bool) -> int:
    report = pooled_rmsf_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _dccm_command(path: Path, hash_content: bool) -> int:
    report = dccm_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _information_command(path: Path, hash_content: bool) -> int:
    report = generalized_correlation_and_information_project_safe(
        path, hash_content=hash_content
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _individual_pca_command(path: Path, hash_content: bool) -> int:
    report = individual_pca_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _common_pca_command(path: Path, hash_content: bool) -> int:
    report = common_pca_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _tica_command(path: Path, hash_content: bool) -> int:
    report = time_lagged_independent_component_analysis_project_safe(
        path, hash_content=hash_content
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _pca_fes_command(path: Path, hash_content: bool) -> int:
    report = pca_fes_basins_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _kmeans_command(path: Path, hash_content: bool) -> int:
    report = clustering_kmeans_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _representative_frames_command(path: Path, hash_content: bool) -> int:
    report = representative_frames_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _imwkmeans_command(path: Path, hash_content: bool) -> int:
    report = clustering_imwkmeans_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _hdbscan_command(path: Path, hash_content: bool) -> int:
    report = clustering_hdbscan_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _msm_command(path: Path, hash_content: bool) -> int:
    report = markov_state_models_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _dihedral_command(path: Path, hash_content: bool) -> int:
    report = dihedral_distributions_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _hydrogen_bond_command(path: Path, hash_content: bool) -> int:
    report = hydrogen_bonds_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _hydrogen_bond_comparison_command(path: Path) -> int:
    report = compare_hydrogen_bond_reports_file_safe(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _observable_command(path: Path, hash_content: bool) -> int:
    report = optional_observables_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _sasa_command(path: Path, hash_content: bool) -> int:
    report = solvent_accessible_surface_area_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _convergence_command(path: Path, hash_content: bool) -> int:
    report = convergence_uncertainty_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _grouped_ml_command(path: Path, hash_content: bool) -> int:
    report = grouped_ml_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _integrated_command(path: Path, hash_content: bool) -> int:
    report = integrated_comparison_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _secondary_structure_command(path: Path, hash_content: bool) -> int:
    report = secondary_structure_project_safe(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _regression_command(path: Path) -> int:
    report = run_regression_case_safe(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _plan_frame_resources_command(
    paths: Sequence[Path],
    evaluated_frame_counts: Optional[Sequence[int]],
    total_source_frames: int,
    replica_count: int,
    target_wall_hours: float,
    target_memory_gib: float,
    minimum_frames_per_replica: int,
    sensitivity_check_policy: str,
    calibration_id: Optional[str],
) -> int:
    try:
        benchmarks = [load_json(path) for path in paths]
        calibration = calibrate_from_benchmarks(
            benchmarks,
            evaluated_frame_counts=evaluated_frame_counts,
            calibration_id=calibration_id,
        )
        plan = recommend_frame_budget(
            calibration,
            total_source_frames=total_source_frames,
            replica_count=replica_count,
            target_wall_seconds=target_wall_hours * 3600.0,
            target_memory_mib=target_memory_gib * 1024.0,
            minimum_frames_per_replica=minimum_frames_per_replica,
            sensitivity_check_policy=sensitivity_check_policy,
        )
        issues = []
        if plan["resolved_mode"] == "integer_stride_per_replica_v1":
            issues.append({
                "severity": "warning",
                "code": "FRAME_SUBSAMPLING_RECOMMENDED",
                "message": str(plan["subsampling_reason"]),
            })
        report = {
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "calibration": calibration,
            "plan": plan,
            "error_count": 0,
            "warning_count": len(issues),
            "issues": issues,
        }
    except (ManifestValidationError, ResourcePlanningError, OSError, ValueError) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        report = {
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "RESOURCE_PLAN_INVALID", "message": message}
                for message in messages
            ],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _plan_automatic_sampling_command(
    path: Path,
    simulation_kind: str,
    modules: Optional[Sequence[str]],
    b_vs_2b: bool,
    replica_diagnostics: bool,
    target_wall_hours: float,
    time_safety_factor: float,
) -> int:
    try:
        report = automatic_sampling_plan(
            path,
            simulation_kind=simulation_kind,
            module_ids=modules,
            b_vs_2b=b_vs_2b,
            replica_diagnostics=replica_diagnostics,
            target_wall_seconds=target_wall_hours * 3600.0,
            time_safety_factor=time_safety_factor,
        )
    except (AutomaticSamplingError, ManifestValidationError, OSError, ValueError) as exc:
        report = {
            "planning_schema": "salsbury-automatic-sampling-plan-v1",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "system_manifest_path": str(path.expanduser().resolve(strict=False)),
            "issues": [{
                "severity": "error",
                "code": "AUTOMATIC_SAMPLING_PLAN_FAILED",
                "message": str(exc),
            }],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _build_resource_calibration_catalog_command(
    sidecars: Sequence[Path], timeout_records: Sequence[Path],
    base_catalogs: Sequence[Path], output: Path, redact_source_paths: bool,
) -> int:
    try:
        report = build_resource_calibration_catalog(
            sidecars,
            timeout_records=timeout_records,
            base_catalogs=base_catalogs,
        )
        if redact_source_paths:
            report = redact_resource_calibration_catalog(report)
        destination = output.expanduser().resolve(strict=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result = {
            "technical_status": "complete", "catalog_path": str(destination),
            "entry_count": report["entry_count"],
        }
    except (ResourceCalibrationError, OSError, ValueError) as exc:
        result = {
            "technical_status": "failed", "catalog_path": str(output),
            "issues": [{"severity": "error", "code": "RESOURCE_CALIBRATION_INVALID", "message": str(exc)}],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["technical_status"] == "complete" else 2


def _prepare_analysis_command(
    pdb: Path,
    psf: Optional[Path],
    trajectories: Sequence[Path],
    output_directory: Path,
    project_id: str,
    frame_interval_ps: float,
    first_frame_time_ps: float,
    temperature_kelvin: float,
    target_wall_hours: Optional[float],
    dssp_executable: Optional[str],
    config_path: Optional[Path],
    generate_connectivity_openmm: bool,
    openmm_bond_definitions: Sequence[Path],
    auto_disable_to_fit_memory: bool,
) -> int:
    try:
        prepare = (
            prepare_standard_analysis_memory_fit
            if auto_disable_to_fit_memory else prepare_standard_analysis
        )
        report = prepare(
            pdb_path=pdb,
            psf_path=psf,
            trajectories=trajectories,
            output_directory=output_directory,
            project_id=project_id,
            frame_interval_ps=frame_interval_ps,
            first_frame_time_ps=first_frame_time_ps,
            temperature_kelvin=temperature_kelvin,
            target_wall_hours=target_wall_hours,
            dssp_executable=dssp_executable,
            config_path=config_path,
            generate_connectivity_openmm=generate_connectivity_openmm,
            openmm_bond_definitions=openmm_bond_definitions,
        )
    except (QuickstartError, OSError, ValueError) as exc:
        report = {
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "issues": [{
                "severity": "error",
                "code": "PREPARE_ANALYSIS_FAILED",
                "message": str(exc),
            }],
        }
        if isinstance(exc, QuickstartMemoryError):
            report["memory_feasibility"] = exc.plan.get(
                "memory_feasibility"
            )
            report["partial_output_directory"] = str(exc.output_directory)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _prepare_comparison_command(
    request: Path,
    output_directory: Path,
    project_id: str,
    temperature_kelvin: float,
    target_wall_hours: Optional[float],
    dssp_executable: Optional[str],
    config_path: Optional[Path],
    auto_disable_to_fit_memory: bool,
) -> int:
    try:
        prepare = (
            prepare_comparative_analysis_memory_fit
            if auto_disable_to_fit_memory else prepare_comparative_analysis
        )
        report = prepare(
            request_path=request,
            output_directory=output_directory,
            project_id=project_id,
            temperature_kelvin=temperature_kelvin,
            target_wall_hours=target_wall_hours,
            dssp_executable=dssp_executable,
            config_path=config_path,
        )
    except (QuickstartError, OSError, ValueError) as exc:
        report = {
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "issues": [{
                "severity": "error",
                "code": "PREPARE_COMPARISON_FAILED",
                "message": str(exc),
            }],
        }
        if isinstance(exc, QuickstartMemoryError):
            report["memory_feasibility"] = exc.plan.get(
                "memory_feasibility"
            )
            report["partial_output_directory"] = str(exc.output_directory)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _summarize_timeseries_command(path: Path) -> int:
    try:
        request = load_json(path)
        result = summarize_timeseries_presentations(
            request["segments"],
            fields=request.get("fields"),
            maximum_observations_per_field=int(
                request.get("maximum_observations_per_field", 1_000_000)
            ),
            padding_fraction=float(request.get("padding_fraction", 0.0)),
            minimum_bins=int(request.get("minimum_bins", 2)),
            maximum_bins=int(request.get("maximum_bins", 100)),
        )
    except (ManifestValidationError, PresentationError, KeyError, TypeError, ValueError, OSError) as exc:
        result = {
            "presentation_schema": "salsbury-timeseries-presentation-v1",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "issues": [{
                "severity": "error",
                "code": "TIMESERIES_PRESENTATION_INVALID",
                "message": str(exc),
            }],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["technical_status"] == "complete" else 2


def _export_rmsf_visualization_command(
    report: Path,
    system_id: str,
    output_prefix: Path,
    reference: Optional[Path],
    aggregation: str,
    overwrite: bool,
) -> int:
    try:
        result = export_rmsf_visualization(
            report, system_id, output_prefix,
            reference_path=reference,
            aggregation=aggregation,
            overwrite=overwrite,
        )
    except (ManifestValidationError, RMSFVisualizationError, OSError, ValueError) as exc:
        result = {
            "module_id": "rmsf_visualization_export",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "issues": [{
                "severity": "error",
                "code": "RMSF_VISUALIZATION_EXPORT_FAILED",
                "message": str(exc),
            }],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["technical_status"] == "complete" else 2


def _project_runner_command(runner, path: Path, hash_content: bool) -> int:
    report = runner(path, hash_content=hash_content)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def _run_instrumented_command(
    command: str, path: Path, hash_content: bool,
    summary_sidecar: Optional[Path] = None,
    installed_report_path: Optional[Path] = None,
) -> int:
    try:
        report = run_instrumented_project_command(
            command, path, hash_content=hash_content
        )
    except (ExecutionResourceError, OSError, ValueError) as exc:
        report = {
            "module_id": command,
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "issues": [{
                "severity": "error", "code": "INSTRUMENTED_EXECUTION_FAILED",
                "message": str(exc),
            }],
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if report.get("technical_status") == "complete" and summary_sidecar is not None:
        if installed_report_path is None:
            raise ExecutionResourceError(
                "--installed-report-path is required with --summary-sidecar"
            )
        sidecar_path = summary_sidecar.expanduser().resolve(strict=False)
        if sidecar_path.exists():
            raise ExecutionResourceError(f"summary sidecar already exists: {sidecar_path}")
        sidecar = analysis_report_sidecar(
            report, installed_report_path,
            report_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            report_size_bytes=len(rendered.encode("utf-8")),
        )
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    sys.stdout.write(rendered)
    return 0 if report.get("technical_status") == "complete" else 2


def _run_instrumented_coordinate_cache_command(
    path: Path,
    output: Path,
    workers: int,
    summary_sidecar: Path,
    installed_report_path: Path,
) -> int:
    try:
        report = run_instrumented_coordinate_cache(
            path, output, maximum_workers=workers
        )
    except (ExecutionResourceError, OSError, ValueError) as exc:
        report = {
            "module_id": "coordinate_cache",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "issues": [{
                "severity": "error",
                "code": "INSTRUMENTED_COORDINATE_CACHE_FAILED",
                "message": str(exc),
            }],
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if report.get("technical_status") == "complete":
        sidecar_path = summary_sidecar.expanduser().resolve(strict=False)
        if sidecar_path.exists():
            raise ExecutionResourceError(
                f"summary sidecar already exists: {sidecar_path}"
            )
        sidecar = analysis_report_sidecar(
            report,
            installed_report_path,
            report_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            report_size_bytes=len(rendered.encode("utf-8")),
        )
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    sys.stdout.write(rendered)
    return 0 if report.get("technical_status") == "complete" else 2


def _summarize_execution_resources_command(root: Path) -> int:
    try:
        report = summarize_execution_resources(root)
    except (ExecutionResourceError, OSError, ValueError) as exc:
        report = {
            "technical_status": "failed",
            "issues": [{
                "severity": "error", "code": "RESOURCE_TABLE_FAILED",
                "message": str(exc),
            }],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("technical_status") == "complete" else 2


def _prioritize_findings_command(root: Path, maximum_findings: Optional[int]) -> int:
    try:
        report = prioritize_findings(root, maximum_findings=maximum_findings)
    except (FindingPickerError, OSError, ValueError) as exc:
        report = {
            "technical_status": "failed",
            "issues": [{
                "severity": "error", "code": "FINDING_PICKER_FAILED",
                "message": str(exc),
            }],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("technical_status") == "complete" else 2


def _standalone_method_command(module_id: str, path: Path) -> int:
    """Run a small JSON-in/JSON-out method contract without touching source data."""

    try:
        request = load_json(path)
        if module_id == "hydrogen_bond_patterns":
            _, patterns = encode_bond_patterns(request["frame_bonds"], request.get("bond_ids"))
            if request["method"] == "pam_jaccard":
                result = pam_jaccard(patterns, int(request["cluster_count"]), int(request.get("maximum_iterations", 100)))
            elif request["method"] == "hdbscan_jaccard":
                result = hdbscan_jaccard(patterns, int(request["minimum_cluster_size"]), request.get("minimum_samples"))
            else:
                raise HydrogenBondPatternError("method must be pam_jaccard or hdbscan_jaccard")
        elif module_id == "representative_structures":
            result = representative_structures(
                request["coordinates"], request.get("frame_weights"),
                float(request.get("within_rmsd_standard_deviations", 1.0)),
            )
        elif module_id == "rmsf_permutation_inference":
            result = rmsf_permutation_test(
                request["group_a_profiles"], request["group_b_profiles"],
                int(request.get("permutations", 9999)), int(request.get("random_seed", 0)),
                int(request.get("exact_partition_limit", 100000)),
            )
        else:
            raise AssertionError(module_id)
        report = {
            "module_id": module_id, "technical_status": "complete",
            "scientific_status": "not evaluated", "request_path": str(path.resolve()),
            "result": result, "error_count": 0, "warning_count": 0, "issues": [],
        }
    except (
        ManifestValidationError, HydrogenBondPatternError, RepresentativeStructureError,
        RMSFInferenceError, KeyError, TypeError, ValueError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        report = {
            "module_id": module_id, "technical_status": "failed",
            "scientific_status": "not evaluated", "request_path": str(path.resolve()),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{"severity": "error", "code": "METHOD_REQUEST_INVALID", "message": message} for message in messages],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_status"] == "complete" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="salsbury-md-analysis")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list-modules", help="List registered analyses and honest implementation status."
    )
    list_parser.add_argument("--json", action="store_true", help="Emit JSON.")

    validate_parser = subparsers.add_parser(
        "validate-manifest", help="Validate a project, system, output, or publication-lock manifest."
    )
    validate_parser.add_argument("kind", choices=MANIFEST_KINDS)
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument(
        "--check-paths", action="store_true",
        help="Require referenced files to exist; output manifests also verify recorded SHA-256 values.",
    )
    validate_parser.add_argument("--json", action="store_true", help="Emit a JSON report.")

    inventory_parser = subparsers.add_parser(
        "inventory-system", help="Inventory files named by a system manifest without modifying them."
    )
    inventory_parser.add_argument("path", type=Path)
    inventory_parser.add_argument(
        "--hash-content", action="store_true",
        help="Read each input and include its SHA-256; this can be slow for trajectories.",
    )

    preflight_parser = subparsers.add_parser(
        "preflight-system",
        help="Inspect supported topology/trajectory metadata and segment consistency read-only.",
    )
    preflight_parser.add_argument("path", type=Path)
    preflight_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every topology, trajectory, and weight file.",
    )

    mapping_parser = subparsers.add_parser(
        "map-common-atoms",
        help="Create a deterministic common PDB/GRO atom map with explicit coverage gates.",
    )
    mapping_parser.add_argument("reference", type=Path)
    mapping_parser.add_argument("targets", type=Path, nargs="+")
    mapping_parser.add_argument("--policy", choices=MAPPING_POLICIES, required=True)
    mapping_parser.add_argument("--selection", choices=ATOM_SELECTIONS, required=True)
    mapping_parser.add_argument(
        "--minimum-reference-coverage", type=float, required=True,
        help="Required common fraction of selected reference atoms, between 0 and 1.",
    )
    mapping_parser.add_argument(
        "--hash-content", action="store_true",
        help="Include SHA-256 values for every mapped topology.",
    )

    context_parser = subparsers.add_parser(
        "compile-context",
        help="Compile explicit units, selections, and system identities read-only.",
    )
    context_parser.add_argument("path", type=Path, help="Project manifest path.")
    context_parser.add_argument(
        "--hash-content", action="store_true",
        help="Stream SHA-256 hashes for every topology, trajectory, and weight file.",
    )

    qc_parser = subparsers.add_parser(
        "structural-qc",
        help="Stream supported coordinates and apply explicit initial integrity gates.",
    )
    qc_parser.add_argument("path", type=Path, help="Project manifest path.")
    qc_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    rmsd_rg_parser = subparsers.add_parser(
        "rmsd-rg",
        help="Fit declared selections and report replica-resolved RMSD and radius of gyration.",
    )
    rmsd_rg_parser.add_argument("path", type=Path, help="Project manifest path.")
    rmsd_rg_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    rmsf_parser = subparsers.add_parser(
        "rmsf",
        help="Report frame-pooled, replica, and time-block atomic RMSF estimates.",
    )
    rmsf_parser.add_argument("path", type=Path, help="Project manifest path.")
    rmsf_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    dccm_parser = subparsers.add_parser(
        "dccm",
        help="Calculate common-basis replica and system dynamic cross-correlation matrices.",
    )
    dccm_parser.add_argument("path", type=Path, help="Project manifest path.")
    dccm_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    information_parser = subparsers.add_parser(
        "information-correlation",
        help="Estimate nonlinear mutual-information dependence between declared features.",
    )
    information_parser.add_argument("path", type=Path, help="Project manifest path.")
    information_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    individual_pca_parser = subparsers.add_parser(
        "individual-pca",
        help="Fit an independent Cartesian PCA basis for each declared replica.",
    )
    individual_pca_parser.add_argument("path", type=Path, help="Project manifest path.")
    individual_pca_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    common_pca_parser = subparsers.add_parser(
        "common-pca",
        help="Fit one global common-atom Cartesian PCA basis across replicas.",
    )
    common_pca_parser.add_argument("path", type=Path, help="Project manifest path.")
    common_pca_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    tica_parser = subparsers.add_parser(
        "tica",
        help="Fit segment-safe reversible TICA to declared common-PCA features.",
    )
    tica_parser.add_argument("path", type=Path, help="Project manifest path.")
    tica_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    pca_fes_parser = subparsers.add_parser(
        "pca-fes-basins",
        help="Build a mode-aware PCA landscape and deterministic occupancy basins.",
    )
    pca_fes_parser.add_argument("path", type=Path, help="Project manifest path.")
    pca_fes_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    kmeans_parser = subparsers.add_parser(
        "cluster-kmeans",
        help="Scan a seeded KMeans grid over declared common-PCA features.",
    )
    kmeans_parser.add_argument("path", type=Path, help="Project manifest path.")
    kmeans_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    representative_parser = subparsers.add_parser(
        "representative-frames",
        help="Select deterministic observed representatives for clusters or PCA basins.",
    )
    representative_parser.add_argument("path", type=Path, help="Project manifest path.")
    representative_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    imwkmeans_parser = subparsers.add_parser(
        "cluster-imwkmeans",
        help="Scan a deterministic intelligent Minkowski weighted KMeans grid.",
    )
    imwkmeans_parser.add_argument("path", type=Path, help="Project manifest path.")
    imwkmeans_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    hdbscan_parser = subparsers.add_parser(
        "cluster-hdbscan",
        help="Run an optional reference-HDBSCAN parameter sensitivity scan.",
    )
    hdbscan_parser.add_argument("path", type=Path, help="Project manifest path.")
    hdbscan_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    msm_parser = subparsers.add_parser(
        "markov-models",
        help="Build segment-safe transition models and lag/CK validation diagnostics.",
    )
    msm_parser.add_argument("path", type=Path, help="Project manifest path.")
    msm_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    dihedral_parser = subparsers.add_parser(
        "dihedrals",
        help="Calculate declared backbone and chi1 circular distributions.",
    )
    dihedral_parser.add_argument("path", type=Path, help="Project manifest path.")
    dihedral_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    hydrogen_bond_parser = subparsers.add_parser(
        "hydrogen-bonds",
        help="Evaluate explicitly indexed hydrogen-bond occupancy features.",
    )
    hydrogen_bond_parser.add_argument("path", type=Path, help="Project manifest path.")
    hydrogen_bond_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    observable_parser = subparsers.add_parser(
        "observables",
        help="Evaluate question-linked explicit distance and contact features.",
    )
    observable_parser.add_argument("path", type=Path, help="Project manifest path.")
    observable_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    sasa_parser = subparsers.add_parser(
        "sasa",
        help="Calculate deterministic Shrake-Rupley solvent-accessible surface area.",
    )
    sasa_parser.add_argument("path", type=Path, help="Project manifest path.")
    sasa_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    convergence_parser = subparsers.add_parser(
        "convergence",
        help="Evaluate block, ESS, split-mean, and optional exploratory replica diagnostics.",
    )
    convergence_parser.add_argument("path", type=Path, help="Project manifest path.")
    convergence_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    grouped_ml_parser = subparsers.add_parser(
        "grouped-ml",
        help="Run leakage-resistant grouped decision-tree validation.",
    )
    grouped_ml_parser.add_argument("path", type=Path, help="Project manifest path.")
    grouped_ml_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    integrated_parser = subparsers.add_parser(
        "integrate",
        help="Assemble prespecified module values without hidden aggregation.",
    )
    integrated_parser.add_argument("path", type=Path, help="Project manifest path.")
    integrated_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    secondary_parser = subparsers.add_parser(
        "secondary-structure",
        help="Run the external mkdssp adapter with executable provenance.",
    )
    secondary_parser.add_argument("path", type=Path, help="Project manifest path.")
    secondary_parser.add_argument(
        "--hash-content", action="store_true",
        help="Also stream SHA-256 hashes for every declared input file.",
    )

    extended_help = {
        "alternative-clustering": "Run distinctly labeled clustering families on common-PCA features.",
        "pald-community": "Calculate sampled PaLD cohesion, local depth, strong ties, and communities.",
        "information-dynamics": "Calculate segment-safe transfer entropy and higher-order feature statistics.",
        "correlation-networks": "Build thresholded signed networks from DCCM outputs.",
        "trajectory-features": "Extract Cartesian, COM, distance, fluctuation, and principal-axis features.",
        "state-coordinate-exports": "Write immutable state trajectories and observed representative structures.",
        "rdf": "Calculate periodic, volume-normalized radial distribution functions.",
        "scalar-distributions": "Build Scott/FD/Rice scalar histograms and segment-safe residence runs.",
        "scalar-threshold-states": "Build threshold-sensitive scalar states, transitions, and residence runs.",
        "hydrogen-bond-discovery": "Discover direct hydrogen bonds from topology-backed chemistry.",
        "water-mediated-hydrogen-bonds": "Discover scalable one-water hydrogen-bond networks.",
        "grouped-regularized-classification": "Run nested grouped classification on hydrogen-bond patterns.",
        "nucleic-acid-structure": "Run the external x3dna-dssr JSON motif adapter.",
        "nucleic-acid-geometry": "Calculate intrinsic ring, fused-fold, and base-stacking geometry.",
        "ion-geometry": "Calculate bound-ion coordination and ion-pair geometry.",
        "ion-atmosphere": "Calculate species-resolved ion atmospheres around solute groups.",
    }
    for command, help_text in extended_help.items():
        extended_parser = subparsers.add_parser(command, help=help_text)
        extended_parser.add_argument("path", type=Path, help="Project manifest path.")
        extended_parser.add_argument(
            "--hash-content", action="store_true",
            help="Also stream SHA-256 hashes for every declared input file.",
        )

    standalone_help = {
        "compare-hydrogen-bonds": "Compare two sparse discovery reports after grouping equivalent donor hydrogens.",
        "hydrogen-bond-patterns": "Cluster explicit frame-level bond patterns using Jaccard distance.",
        "representative-structures": "Select average, closest, medoid, and central structures from aligned coordinates.",
        "rmsf-permutation": "Run unit-level exact or seeded RMSF permutation inference.",
    }
    for command, help_text in standalone_help.items():
        method_parser = subparsers.add_parser(command, help=help_text)
        method_parser.add_argument("path", type=Path, help="JSON method-request path.")

    resource_parser = subparsers.add_parser(
        "plan-frame-resources",
        help="Estimate all-frame feasibility or balanced subsampling from retained pilots.",
    )
    resource_parser.add_argument("benchmarks", type=Path, nargs="+")
    resource_parser.add_argument(
        "--evaluated-frame-count", type=int, action="append",
        help="Pilot estimator frame count; repeat in benchmark order when not embedded.",
    )
    resource_parser.add_argument("--total-source-frames", type=int, required=True)
    resource_parser.add_argument("--replica-count", type=int, required=True)
    resource_parser.add_argument("--target-wall-hours", type=float, default=4.0)
    resource_parser.add_argument("--target-memory-gib", type=float, default=16.0)
    resource_parser.add_argument("--minimum-frames-per-replica", type=int, default=100)
    resource_parser.add_argument(
        "--sensitivity-check-policy",
        choices=("off", "recommend", "require"),
        default="off",
        help=(
            "Optional frame-budget sensitivity policy: off disables it, "
            "recommend records a nonblocking recommendation, and require "
            "makes it an explicit project-owner gate."
        ),
    )
    resource_parser.add_argument("--calibration-id")

    automatic_parser = subparsers.add_parser(
        "plan-automatic-sampling",
        help=(
            "Inspect a system manifest, estimate per-method wall time, and "
            "assign method-, size-, trajectory-, and time-aware balanced sampling."
        ),
    )
    automatic_parser.add_argument("path", type=Path, help="System manifest path.")
    automatic_parser.add_argument(
        "--simulation-kind",
        choices=tuple(sorted(SIMULATION_KINDS)),
        required=True,
        help="Simulation/sampling class; no scientific interpretation is inferred.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-analysis",
        help=(
            "Create a validated, time-budgeted local or Slurm analysis "
            "from one PDB, supplied or explicitly requested OpenMM-derived "
            "connectivity, and one or more replica DCD "
            "trajectories."
        ),
    )
    prepare_parser.add_argument("--pdb", type=Path, required=True)
    prepare_parser.add_argument(
        "--psf", "--connectivity", dest="psf", type=Path,
        help=(
            "Explicit bond topology in PSF, salsbury-bonds-v1 JSON, PRMTOP, "
            "or PARM7 format. --psf is retained as a backward-compatible alias."
        ),
    )
    prepare_parser.add_argument(
        "--generate-connectivity-openmm", action="store_true",
        help=(
            "When no connectivity file exists, use optional OpenMM standard-residue "
            "and explicit PDB connectivity to write a reusable bond JSON. This is "
            "fail-closed and never guesses bonds by distance."
        ),
    )
    prepare_parser.add_argument(
        "--openmm-bond-definitions", type=Path, action="append", default=[],
        help=(
            "Reviewed OpenMM residue bond-definition XML for nonstandard chemistry; "
            "repeat as needed and use only with --generate-connectivity-openmm."
        ),
    )
    prepare_parser.add_argument(
        "--trajectory", type=Path, action="append", required=True,
        help="Replica DCD path; repeat once per replica.",
    )
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--project-id", required=True)
    prepare_parser.add_argument(
        "--frame-interval-ps", type=float, required=True,
        help=(
            "Physical time between saved DCD frames in ps. This is required because "
            "a PSF/PDB/DCD set does not always encode an unambiguous physical timestep."
        ),
    )
    prepare_parser.add_argument("--first-frame-time-ps", type=float, default=0.0)
    prepare_parser.add_argument("--temperature-kelvin", type=float, default=300.0)
    prepare_parser.add_argument(
        "--target-wall-hours", type=float,
        help=(
            "Override execution.maximum_hours_per_cpu for the complete campaign. "
            "The generated-config default is 24 wall hours."
        ),
    )
    prepare_parser.add_argument("--dssp-executable")
    prepare_parser.add_argument(
        "--config", type=Path,
        help=(
            "Optional salsbury-analysis-config-v1 JSON. Unspecified modules and "
            "topology-applicable views remain enabled by default."
        ),
    )
    prepare_parser.add_argument(
        "--auto-disable-to-fit-memory", action="store_true",
        help=(
            "When enabled technical minima exceed execution.maximum_memory_gib, "
            "preserve the requested config, explicitly disable the oversized "
            "modules and their dependents, replan, and write the resolved on/off "
            "config. Without this flag preparation fails closed."
        ),
    )

    comparison_parser = subparsers.add_parser(
        "prepare-comparison",
        help=(
            "Create one shared-basis, common-grid analysis for two or more systems "
            "declared in a salsbury-comparative-analysis-input-v1 request."
        ),
    )
    comparison_parser.add_argument("request", type=Path)
    comparison_parser.add_argument("--output", type=Path, required=True)
    comparison_parser.add_argument("--project-id", required=True)
    comparison_parser.add_argument("--temperature-kelvin", type=float, default=300.0)
    comparison_parser.add_argument(
        "--target-wall-hours", type=float,
        help=(
            "Override execution.maximum_hours_per_cpu for the complete campaign. "
            "The generated-config default is 24 wall hours."
        ),
    )

    local_workflow_parser = subparsers.add_parser(
        "run-local-workflow",
        help=(
            "Execute a prepared workflow without Slurm while enforcing its CPU cap "
            "and dependency order."
        ),
    )
    local_workflow_parser.add_argument(
        "root", type=Path, help="Prepared analysis directory."
    )

    capacity_parser = subparsers.add_parser(
        "advise-slurm-capacity",
        help=(
            "Optionally inspect a prepared campaign and the live Slurm queue "
            "without submitting or changing any job."
        ),
    )
    capacity_parser.add_argument(
        "root", type=Path, help="Prepared analysis directory."
    )
    capacity_parser.add_argument(
        "--wall-hours", type=float, required=True,
        help=(
            "Campaign duration to model at the maximum useful detected CPU count."
        ),
    )
    capacity_parser.add_argument(
        "--maximum-memory-gib", type=float,
        help=(
            "Optional per-task memory ceiling; the prepared campaign value is used "
            "when omitted."
        ),
    )
    capacity_parser.add_argument(
        "--cpu-ceiling", type=int,
        help="Optional user policy cap below the detected useful maximum.",
    )
    capacity_parser.add_argument(
        "--slurm-profile", type=Path,
        help="Override the prepared campaign's slurm-profile.json.",
    )
    capacity_parser.add_argument(
        "--slurm-user",
        help="Override the current account name used for read-only association checks.",
    )
    capacity_parser.add_argument(
        "--job-id", action="append", default=[],
        help=(
            "Pending Slurm job ID for scheduler-projected start reporting; repeat "
            "for multiple jobs."
        ),
    )
    capacity_parser.add_argument(
        "--offline", action="store_true",
        help="Replan from saved evidence without querying Slurm.",
    )
    capacity_parser.add_argument(
        "--format", choices=("json", "markdown"), default="json",
        help="Output format; JSON is convenient for ChatGPTWork automation.",
    )

    cache_parser = subparsers.add_parser(
        "build-coordinate-cache",
        help=(
            "Write an atomic made-whole, unaligned molecular-payload DCD cache "
            "for non-water trajectory analyses."
        ),
    )
    cache_parser.add_argument("path", type=Path, help="System manifest path.")
    cache_parser.add_argument("--output", type=Path, required=True)
    cache_parser.add_argument(
        "--hash-source-content", action="store_true",
        help="Also SHA-256 hash every source trajectory (adds a full I/O pass).",
    )
    cache_parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Maximum replica-parallel worker processes. Each worker streams one "
            "replica; the final cache is assembled and published atomically."
        ),
    )
    comparison_parser.add_argument("--dssp-executable")
    comparison_parser.add_argument(
        "--config", type=Path,
        help=(
            "Optional salsbury-analysis-config-v1 JSON; all applicable modules and "
            "topology-derived views are enabled when omitted."
        ),
    )
    comparison_parser.add_argument(
        "--auto-disable-to-fit-memory", action="store_true",
        help=(
            "Preserve the requested comparison config, explicitly turn off "
            "memory-incompatible switches and dependents, replan, and write "
            "the resolved on/off config. Without this flag preparation fails closed."
        ),
    )

    timeseries_parser = subparsers.add_parser(
        "summarize-timeseries",
        help="Apply Scott-histogram-first reporting to generic non-RMSD scalar series.",
    )
    timeseries_parser.add_argument("path", type=Path, help="JSON time-series request path.")

    instrument_parser = subparsers.add_parser(
        "run-instrumented",
        help="Run one project analysis and attach measured CPU, wall, memory, host, and Slurm evidence.",
    )
    instrument_parser.add_argument(
        "analysis_command", choices=INSTRUMENTABLE_PROJECT_COMMANDS
    )
    instrument_parser.add_argument("path", type=Path, help="Project manifest path.")
    instrument_parser.add_argument("--hash-content", action="store_true")
    instrument_parser.add_argument(
        "--summary-sidecar", type=Path,
        help="Write compact report-hash-bound resource and finding evidence.",
    )
    instrument_parser.add_argument(
        "--installed-report-path", type=Path,
        help="Final report path bound into --summary-sidecar evidence.",
    )

    cache_instrument_parser = subparsers.add_parser(
        "run-coordinate-cache-instrumented",
        help=(
            "Build an atomic coordinate cache and attach measured CPU, wall, "
            "memory, host, Slurm, and exact frame evidence."
        ),
    )
    cache_instrument_parser.add_argument("path", type=Path)
    cache_instrument_parser.add_argument("--output", type=Path, required=True)
    cache_instrument_parser.add_argument("--workers", type=int, default=1)
    cache_instrument_parser.add_argument(
        "--summary-sidecar", type=Path, required=True
    )
    cache_instrument_parser.add_argument(
        "--installed-report-path", type=Path, required=True
    )

    resource_summary_parser = subparsers.add_parser(
        "summarize-execution-resources",
        help="Write consolidated CSV, JSON, and Markdown resource/frame tables.",
    )
    resource_summary_parser.add_argument("root", type=Path, help="Generated analysis root.")

    calibration_catalog_parser = subparsers.add_parser(
        "build-resource-calibration-catalog",
        help="Build hash-bound measured CPU, memory, and frame-coverage planner evidence.",
    )
    calibration_catalog_parser.add_argument(
        "sidecar", type=Path, nargs="*", help="Complete report.json.summary.json paths."
    )
    calibration_catalog_parser.add_argument(
        "--timeout-record", type=Path, action="append", default=[],
        help=(
            "Fail-closed right-censored timeout evidence; repeat for each "
            "timed-out module task."
        ),
    )
    calibration_catalog_parser.add_argument(
        "--base-catalog", type=Path, action="append", default=[],
        help=(
            "Previously validated calibration catalog to extend without "
            "discarding its hash-bound evidence; repeat as needed."
        ),
    )
    calibration_catalog_parser.add_argument("--output", type=Path, required=True)
    calibration_catalog_parser.add_argument(
        "--redact-source-paths", action="store_true",
        help=(
            "Remove private cluster paths, scheduler IDs, and hostnames while "
            "retaining hash-bound planner values and evidence identities."
        ),
    )

    finding_parser = subparsers.add_parser(
        "prioritize-findings",
        help="Rank transparent single- and multi-system findings without an opaque score.",
    )
    finding_parser.add_argument("root", type=Path, help="Generated analysis root.")
    finding_parser.add_argument("--maximum-findings", type=int)

    rmsf_export_parser = subparsers.add_parser(
        "export-rmsf-visualization",
        help="Export RMSF as PDB B factors and a VMD NewCartoon/Beta script.",
    )
    rmsf_export_parser.add_argument("report", type=Path, help="Completed pooled-RMSF JSON report.")
    rmsf_export_parser.add_argument("system_id", help="System ID to export.")
    rmsf_export_parser.add_argument("output_prefix", type=Path)
    rmsf_export_parser.add_argument("--reference", type=Path, help="Override the report reference PDB path.")
    rmsf_export_parser.add_argument(
        "--aggregation", choices=("residue_mean", "atom"), default="residue_mean"
    )
    rmsf_export_parser.add_argument("--overwrite", action="store_true")
    automatic_parser.add_argument(
        "--module", action="append",
        help=(
            "Limit the plan to a module ID; repeat as needed. "
            "The default plans every direct trajectory method."
        ),
    )
    automatic_parser.add_argument(
        "--b-vs-2b", action="store_true",
        help="Explicitly plan an optional base-versus-doubled frame-budget comparison.",
    )
    automatic_parser.add_argument(
        "--replica-diagnostics", action="store_true",
        help=(
            "Enable optional exploratory replica diagnostics. They are not recommended "
            "by default and are not scientific acceptance gates."
        ),
    )
    automatic_parser.add_argument(
        "--target-wall-hours", type=float, default=4.0,
        help=(
            "Maximum estimated wall time for each direct method; default: 4 hours. "
            "The estimate includes the time safety factor."
        ),
    )
    automatic_parser.add_argument(
        "--time-safety-factor", type=float, default=1.5,
        help="Multiplicative timing margin applied to every estimate; default: 1.5.",
    )

    regression_parser = subparsers.add_parser(
        "run-regression",
        help="Run a hash-pinned project regression without changing project data.",
    )
    regression_parser.add_argument("path", type=Path, help="Regression-case path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-modules":
        return _list_modules(args.json)
    if args.command == "validate-manifest":
        return _validate_manifest_command(args.kind, args.path, args.check_paths, args.json)
    if args.command == "inventory-system":
        return _inventory_system_command(args.path, args.hash_content)
    if args.command == "preflight-system":
        return _preflight_system_command(args.path, args.hash_content)
    if args.command == "map-common-atoms":
        return _map_common_atoms_command(
            args.reference,
            args.targets,
            args.policy,
            args.selection,
            args.minimum_reference_coverage,
            args.hash_content,
        )
    if args.command == "compile-context":
        return _compile_context_command(args.path, args.hash_content)
    if args.command == "structural-qc":
        return _structural_qc_command(args.path, args.hash_content)
    if args.command == "rmsd-rg":
        return _rmsd_rg_command(args.path, args.hash_content)
    if args.command == "rmsf":
        return _rmsf_command(args.path, args.hash_content)
    if args.command == "dccm":
        return _dccm_command(args.path, args.hash_content)
    if args.command == "information-correlation":
        return _information_command(args.path, args.hash_content)
    if args.command == "individual-pca":
        return _individual_pca_command(args.path, args.hash_content)
    if args.command == "common-pca":
        return _common_pca_command(args.path, args.hash_content)
    if args.command == "tica":
        return _tica_command(args.path, args.hash_content)
    if args.command == "pca-fes-basins":
        return _pca_fes_command(args.path, args.hash_content)
    if args.command == "cluster-kmeans":
        return _kmeans_command(args.path, args.hash_content)
    if args.command == "representative-frames":
        return _representative_frames_command(args.path, args.hash_content)
    if args.command == "cluster-imwkmeans":
        return _imwkmeans_command(args.path, args.hash_content)
    if args.command == "cluster-hdbscan":
        return _hdbscan_command(args.path, args.hash_content)
    if args.command == "markov-models":
        return _msm_command(args.path, args.hash_content)
    if args.command == "dihedrals":
        return _dihedral_command(args.path, args.hash_content)
    if args.command == "hydrogen-bonds":
        return _hydrogen_bond_command(args.path, args.hash_content)
    if args.command == "compare-hydrogen-bonds":
        return _hydrogen_bond_comparison_command(args.path)
    if args.command == "observables":
        return _observable_command(args.path, args.hash_content)
    if args.command == "sasa":
        return _sasa_command(args.path, args.hash_content)
    if args.command == "convergence":
        return _convergence_command(args.path, args.hash_content)
    if args.command == "grouped-ml":
        return _grouped_ml_command(args.path, args.hash_content)
    if args.command == "integrate":
        return _integrated_command(args.path, args.hash_content)
    if args.command == "secondary-structure":
        return _secondary_structure_command(args.path, args.hash_content)
    if args.command in EXTENDED_PROJECT_COMMANDS:
        return _project_runner_command(
            EXTENDED_PROJECT_COMMANDS[args.command], args.path, args.hash_content
        )
    if args.command in STANDALONE_METHOD_COMMANDS:
        return _standalone_method_command(STANDALONE_METHOD_COMMANDS[args.command], args.path)
    if args.command == "plan-frame-resources":
        return _plan_frame_resources_command(
            args.benchmarks,
            args.evaluated_frame_count,
            args.total_source_frames,
            args.replica_count,
            args.target_wall_hours,
            args.target_memory_gib,
            args.minimum_frames_per_replica,
            args.sensitivity_check_policy,
            args.calibration_id,
        )
    if args.command == "plan-automatic-sampling":
        return _plan_automatic_sampling_command(
            args.path,
            args.simulation_kind,
            args.module,
            args.b_vs_2b,
            args.replica_diagnostics,
            args.target_wall_hours,
            args.time_safety_factor,
        )
    if args.command == "prepare-analysis":
        return _prepare_analysis_command(
            args.pdb,
            args.psf,
            args.trajectory,
            args.output,
            args.project_id,
            args.frame_interval_ps,
            args.first_frame_time_ps,
            args.temperature_kelvin,
            args.target_wall_hours,
            args.dssp_executable,
            args.config,
            args.generate_connectivity_openmm,
            args.openmm_bond_definitions,
            args.auto_disable_to_fit_memory,
        )
    if args.command == "prepare-comparison":
        return _prepare_comparison_command(
            args.request,
            args.output,
            args.project_id,
            args.temperature_kelvin,
            args.target_wall_hours,
            args.dssp_executable,
            args.config,
            args.auto_disable_to_fit_memory,
        )
    if args.command == "run-local-workflow":
        try:
            report = run_local_workflow(args.root)
        except (ExecutionAdapterError, OSError, ValueError) as exc:
            report = {
                "technical_status": "failed",
                "scientific_status": "not evaluated",
                "issues": [{
                    "severity": "error",
                    "code": "LOCAL_WORKFLOW_FAILED",
                    "message": str(exc),
                }],
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("technical_status") == "complete" else 2
    if args.command == "advise-slurm-capacity":
        try:
            report = advise_slurm_capacity(
                args.root,
                wall_hours=args.wall_hours,
                maximum_memory_gib=args.maximum_memory_gib,
                cpu_ceiling=args.cpu_ceiling,
                slurm_profile_path=args.slurm_profile,
                live=not args.offline,
                slurm_user=args.slurm_user,
                job_ids=args.job_id,
            )
        except (
            SlurmCapacityError, ExecutionAdapterError, ResourcePlanningError,
            OSError, ValueError,
        ) as exc:
            report = {
                "technical_status": "failed",
                "scientific_status": "not evaluated",
                "read_only": True,
                "jobs_submitted": False,
                "issues": [{
                    "severity": "error",
                    "code": "SLURM_CAPACITY_ADVICE_FAILED",
                    "message": str(exc),
                }],
            }
        if args.format == "markdown" and report.get("technical_status") == "complete":
            print(render_capacity_markdown(report), end="")
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("technical_status") == "complete" else 2
    if args.command == "build-coordinate-cache":
        report = build_coordinate_cache_safe(
            args.path, args.output, hash_source_content=args.hash_source_content,
            maximum_workers=args.workers,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["technical_status"] == "complete" else 2
    if args.command == "summarize-timeseries":
        return _summarize_timeseries_command(args.path)
    if args.command == "run-instrumented":
        return _run_instrumented_command(
            args.analysis_command, args.path, args.hash_content,
            args.summary_sidecar, args.installed_report_path,
        )
    if args.command == "run-coordinate-cache-instrumented":
        return _run_instrumented_coordinate_cache_command(
            args.path,
            args.output,
            args.workers,
            args.summary_sidecar,
            args.installed_report_path,
        )
    if args.command == "summarize-execution-resources":
        return _summarize_execution_resources_command(args.root)
    if args.command == "build-resource-calibration-catalog":
        return _build_resource_calibration_catalog_command(
            args.sidecar, args.timeout_record, args.base_catalog, args.output,
            args.redact_source_paths,
        )
    if args.command == "prioritize-findings":
        return _prioritize_findings_command(args.root, args.maximum_findings)
    if args.command == "export-rmsf-visualization":
        return _export_rmsf_visualization_command(
            args.report,
            args.system_id,
            args.output_prefix,
            args.reference,
            args.aggregation,
            args.overwrite,
        )
    if args.command == "run-regression":
        return _regression_command(args.path)
    raise AssertionError(f"Unhandled command: {args.command}")
