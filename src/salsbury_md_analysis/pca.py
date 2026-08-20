"""Replica-local and global common-atom Cartesian principal-component analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    FrameSelectionPlan, frame_selected, normalize_frame_selection,
    plan_frame_selection, reader_frame_indices,
)
from .geometry import GeometryError, apply_transform, best_fit_transform
from .manifests import ManifestValidationError, load_json, resolve_manifest_path, sha256_file
from .moments import sample_summary
from .pca_math import (
    CartesianCovariance,
    PCAError,
    PCAResult,
    mixture_covariance,
    principal_components,
    project,
    randomized_truncated_pca,
)
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .reporting import atom_identity_record, issue_record
from .selections import AtomCorrespondence, build_common_correspondences
from .trajectory_contracts import (
    TrajectoryContractError,
    frame_axis_value,
    normalize_segment_axis,
    require_periodic_policy,
)
from .upstream_cache import (
    load_cached_project_report,
    project_module_contract_sha256,
)


_BASE_SETTINGS = {
    "alignment_selection",
    "analysis_selection",
    "minimum_reference_coverage",
    "frame_stride",
    "maximum_features",
    "component_count",
    "minimum_evaluated_frames_per_replica",
}
_OPTIONAL_SETTINGS = {
    "frame_selection", "projection_frame_selection", "projection_frame_stride",
    "solver", "symmetry_expansion",
}
_SOLVER = {
    "covariance_denominator": "population_N",
    "eigenvalue_tolerance_angstrom2": 1.0e-12,
    "solver_tolerance": 1.0e-10,
    "maximum_relative_residual": 1.0e-8,
    "maximum_iterations": 10_000,
    "component_orientation": "largest_absolute_loading_positive_first_tie",
}


class PCAAnalysisError(ValueError):
    """Raised when a PCA analysis contract or trajectory fails closed."""


@dataclass
class _ReplicaPlan:
    system_id: str
    replica_id: str
    replica: Mapping[str, object]
    topology_path: Path
    topology_format: str
    target_atoms: Sequence[AtomRecord]
    alignment: AtomCorrespondence
    analysis: AtomCorrespondence


def _settings(project_data: Mapping[str, object], module_id: str) -> Dict[str, object]:
    definitions = project_data.get("definitions")
    if not isinstance(definitions, dict):
        raise PCAAnalysisError(f"project definitions.{module_id} is required")
    raw = definitions.get(module_id)
    if not isinstance(raw, dict):
        raise PCAAnalysisError(f"project definitions.{module_id} must be an object")
    required = set(_BASE_SETTINGS)
    if module_id == "common_pca":
        required.add("basis_weighting")
    unknown = sorted(set(raw).difference(required | _OPTIONAL_SETTINGS))
    missing = sorted(required.difference(raw))
    if unknown:
        raise PCAAnalysisError(
            f"definitions.{module_id} contains unknown fields: " + ", ".join(unknown)
        )
    if missing:
        raise PCAAnalysisError(
            f"definitions.{module_id} is missing required fields: " + ", ".join(missing)
        )
    result: Dict[str, object] = {}
    for field in ("alignment_selection", "analysis_selection"):
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise PCAAnalysisError(f"{field} must be a nonempty selection name")
        result[field] = value.strip()
    coverage = raw["minimum_reference_coverage"]
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not 0.0 <= float(coverage) <= 1.0
    ):
        raise PCAAnalysisError("minimum_reference_coverage must be between 0 and 1")
    result["minimum_reference_coverage"] = float(coverage)
    for field in (
        "frame_stride",
        "maximum_features",
        "component_count",
        "minimum_evaluated_frames_per_replica",
    ):
        value = raw[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PCAAnalysisError(f"{field} must be a positive integer")
        result[field] = value
    if int(result["component_count"]) > int(result["maximum_features"]):
        raise PCAAnalysisError("component_count cannot exceed maximum_features")
    if module_id == "common_pca":
        weighting = raw["basis_weighting"]
        if weighting not in {"frame", "replica_equal"}:
            raise PCAAnalysisError("basis_weighting must be frame or replica_equal")
        result["basis_weighting"] = str(weighting)
        if "symmetry_expansion" in raw:
            symmetry = raw["symmetry_expansion"]
            if not isinstance(symmetry, dict):
                raise PCAAnalysisError("symmetry_expansion must be an object")
            if symmetry.get("planning_schema") != "salsbury-equivalent-oligomer-plan-v1":
                raise PCAAnalysisError(
                    "symmetry_expansion must use salsbury-equivalent-oligomer-plan-v1"
                )
            if symmetry.get("applicable") is not True:
                raise PCAAnalysisError(
                    "symmetry_expansion must be an applicable equivalent-oligomer plan"
                )
            result["symmetry_expansion"] = dict(symmetry)
    raw_solver = raw.get("solver", {"method": "dense_covariance_v1"})
    if not isinstance(raw_solver, dict) or not isinstance(raw_solver.get("method"), str):
        raise PCAAnalysisError("solver must be an object with a method")
    method = raw_solver["method"]
    if method == "dense_covariance_v1":
        if set(raw_solver) != {"method"}:
            raise PCAAnalysisError(
                "dense_covariance_v1 solver accepts only the method field"
            )
        solver = {"method": method}
    elif method == "randomized_truncated_svd_v1":
        if module_id != "common_pca":
            raise PCAAnalysisError(
                "randomized_truncated_svd_v1 is currently supported only for common_pca"
            )
        required_solver = {
            "method", "oversampling", "power_iterations", "random_seed",
            "maximum_sample_matrix_elements", "maximum_relative_residual",
        }
        allowed_solver = required_solver | {"power_iteration_schedule"}
        if frozenset(raw_solver) not in {
            frozenset(required_solver), frozenset(allowed_solver)
        }:
            raise PCAAnalysisError(
                "randomized_truncated_svd_v1 solver fields do not match the contract"
            )
        solver = {"method": method}
        for field, minimum in (
            ("oversampling", 2),
            ("power_iterations", 0),
            ("random_seed", 0),
            ("maximum_sample_matrix_elements", 1),
        ):
            value = raw_solver[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise PCAAnalysisError(f"solver.{field} must be an integer >= {minimum}")
            solver[field] = value
        residual = raw_solver["maximum_relative_residual"]
        if (
            isinstance(residual, bool)
            or not isinstance(residual, (int, float))
            or not math.isfinite(float(residual))
            or float(residual) <= 0.0
        ):
            raise PCAAnalysisError(
                "solver.maximum_relative_residual must be finite and positive"
            )
        solver["maximum_relative_residual"] = float(residual)
        schedule = raw_solver.get(
            "power_iteration_schedule", [solver["power_iterations"]]
        )
        if (
            not isinstance(schedule, list)
            or not schedule
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in schedule
            )
            or schedule[0] != solver["power_iterations"]
            or any(later <= earlier for earlier, later in zip(schedule, schedule[1:]))
        ):
            raise PCAAnalysisError(
                "solver.power_iteration_schedule must begin with power_iterations "
                "and contain strictly increasing nonnegative integers"
            )
        solver["power_iteration_schedule"] = list(schedule)
    else:
        raise PCAAnalysisError(
            "solver.method must be dense_covariance_v1 or randomized_truncated_svd_v1"
        )
    result["solver"] = solver
    result["frame_selection"] = normalize_frame_selection(
        raw.get("frame_selection"), int(result["frame_stride"]),
        error_type=PCAAnalysisError,
    )
    projection_stride = raw.get("projection_frame_stride", result["frame_stride"])
    if (
        isinstance(projection_stride, bool)
        or not isinstance(projection_stride, int)
        or projection_stride <= 0
    ):
        raise PCAAnalysisError("projection_frame_stride must be a positive integer")
    result["projection_frame_stride"] = projection_stride
    result["projection_frame_selection"] = normalize_frame_selection(
        raw.get("projection_frame_selection"), projection_stride,
        error_type=PCAAnalysisError,
    )
    return result


def _coordinates_at(
    coordinates: Sequence[Tuple[float, float, float]], indices: Sequence[int]
) -> Tuple[Tuple[float, float, float], ...]:
    try:
        return tuple(coordinates[index] for index in indices)
    except IndexError as exc:
        raise PCAAnalysisError(
            "atom correspondence index exceeds coordinate atom count"
        ) from exc


def _flatten(coordinates: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    return tuple(float(value) for coordinate in coordinates for value in coordinate)


def _mapping_sets(
    reference_atoms: Sequence[AtomRecord],
    target_atom_sets: Sequence[Sequence[AtomRecord]],
    selections: Mapping[str, object],
    settings: Mapping[str, object],
    policy: str,
    global_common: bool,
) -> Tuple[Dict[str, AtomCorrespondence], ...]:
    results: List[Dict[str, AtomCorrespondence]] = [
        {} for _ in target_atom_sets
    ]
    for role, field in (
        ("alignment", "alignment_selection"),
        ("analysis", "analysis_selection"),
    ):
        selection_id = str(settings[field])
        definition = selections.get(selection_id)
        if not isinstance(definition, dict):
            raise PCAAnalysisError(f"{field} names undefined selection {selection_id!r}")
        if global_common:
            # In a cross-variant common PCA, the alignment basis is the
            # structural-homology gate.  The analysis basis is, by definition,
            # the exact all-topology intersection and may legitimately omit
            # variant-specific heavy atoms.  Its raw reference coverage remains
            # explicit in every correspondence and is surfaced as a warning
            # below instead of being misclassified as a mapping failure.
            coverage_gate = (
                float(settings["minimum_reference_coverage"])
                if role == "alignment" else 0.0
            )
            mappings = build_common_correspondences(
                reference_atoms,
                target_atom_sets,
                definition,
                selection_id,
                policy,
                coverage_gate,
            )
        else:
            mappings = tuple(
                build_common_correspondences(
                    reference_atoms,
                    (target_atoms,),
                    definition,
                    selection_id,
                    policy,
                    float(settings["minimum_reference_coverage"]),
                )[0]
                for target_atoms in target_atom_sets
            )
        for result, mapping in zip(results, mappings):
            result[role] = mapping
    return tuple(results)


def _prepare(
    project_source: Path,
    module_id: str,
    hash_content: bool,
    global_common: bool,
) -> Dict[str, object]:
    project_data = load_json(project_source)
    settings = _settings(project_data, module_id)
    reference_value = project_data.get("reference_structure")
    if not isinstance(reference_value, str) or not reference_value.strip():
        raise PCAAnalysisError(f"reference_structure is required for {module_id}")
    policy_value = project_data.get("common_atom_policy")
    if not isinstance(policy_value, str) or not policy_value.strip():
        raise PCAAnalysisError(f"common_atom_policy is required for {module_id}")
    policy = policy_value.strip()
    context = compile_project_context_file(project_source, hash_content=hash_content)
    contract = context["contract"]
    assert isinstance(contract, dict)
    selections = contract["selections"]
    units = contract["units"]
    assert isinstance(selections, dict)
    assert isinstance(units, dict)
    coordinate_unit = str(units["coordinates"])
    time_value = units.get("time")
    time_unit = str(time_value) if isinstance(time_value, str) else None
    periodic_policy = require_periodic_policy(contract.get("periodic_coordinate_policy"))
    reference_path = resolve_manifest_path(reference_value, project_source)
    reference_format, reference_atoms = read_topology_atoms(reference_path)
    try:
        raw_reference_frame = next(iter_coordinate_frames(reference_path, coordinate_unit))
    except StopIteration as exc:
        raise PCAAnalysisError("reference_structure contains no coordinate frame") from exc
    reference_processor = PeriodicFrameProcessor.from_reference(
        project_data, project_source, len(reference_atoms)
    )
    reference_frame = reference_processor.process(
        raw_reference_frame, str(reference_path)
    )
    if reference_frame.atom_count != len(reference_atoms):
        raise PCAAnalysisError(
            f"reference coordinate count {reference_frame.atom_count} does not match "
            f"reference topology count {len(reference_atoms)}"
        )
    system_path = Path(str(context["system_manifest_path"]))
    system_manifest = load_json(system_path)
    systems = system_manifest["systems"]
    assert isinstance(systems, list)
    frame_selection_plan, frame_selection_report = plan_frame_selection(
        system_manifest, system_path, coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        error_type=PCAAnalysisError,
    )
    projection_frame_selection_plan, projection_frame_selection_report = (
        plan_frame_selection(
            system_manifest, system_path, coordinate_unit,
            settings["projection_frame_selection"],  # type: ignore[arg-type]
            frame_stride=int(settings["projection_frame_stride"]),
            error_type=PCAAnalysisError,
        )
    )
    topology_records: List[Dict[str, object]] = []
    for system in systems:
        assert isinstance(system, dict)
        system_id = str(system["system_id"])
        replicas = system["replicas"]
        assert isinstance(replicas, list)
        for replica in replicas:
            assert isinstance(replica, dict)
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            topology_format, target_atoms = read_topology_atoms(topology_path)
            topology_records.append({
                "system_id": system_id,
                "replica_id": str(replica["replica_id"]),
                "replica": replica,
                "topology_path": topology_path,
                "topology_format": topology_format,
                "target_atoms": target_atoms,
            })
    if not topology_records:
        raise PCAAnalysisError("system manifest contains no replicas")
    mapping_sets = _mapping_sets(
        reference_atoms,
        [record["target_atoms"] for record in topology_records],
        selections,
        settings,
        policy,
        global_common,
    )
    plans = []
    for record, mappings in zip(topology_records, mapping_sets):
        plans.append(_ReplicaPlan(
            system_id=str(record["system_id"]),
            replica_id=str(record["replica_id"]),
            replica=record["replica"],  # type: ignore[arg-type]
            topology_path=record["topology_path"],  # type: ignore[arg-type]
            topology_format=str(record["topology_format"]),
            target_atoms=record["target_atoms"],  # type: ignore[arg-type]
            alignment=mappings["alignment"],
            analysis=mappings["analysis"],
        ))
    inventory = context["input_inventory"]
    assert isinstance(inventory, dict)
    entries = inventory["entries"]
    assert isinstance(entries, list)
    inventory_by_path = {
        str(entry["resolved_path"]): entry
        for entry in entries
        if isinstance(entry, dict)
    }
    return {
        "project_data": project_data,
        "settings": settings,
        "context": context,
        "contract": contract,
        "systems": systems,
        "system_path": system_path,
        "reference_path": reference_path,
        "reference_format": reference_format,
        "reference_atoms": reference_atoms,
        "reference_frame": reference_frame,
        "plans": plans,
        "policy": policy,
        "periodic_policy": periodic_policy,
        "coordinate_unit": coordinate_unit,
        "time_unit": time_unit,
        "sampling_mode": str(contract["sampling_mode"]),
        "inventory_by_path": inventory_by_path,
        "frame_selection_plan": frame_selection_plan,
        "frame_selection_report": frame_selection_report,
        "projection_frame_selection_plan": projection_frame_selection_plan,
        "projection_frame_selection_report": projection_frame_selection_report,
    }


def _scan_replica(
    plan: _ReplicaPlan,
    project_data: Mapping[str, object],
    system_path: Path,
    reference_coordinates: Sequence[Tuple[float, float, float]],
    coordinate_unit: str,
    time_unit: Optional[str],
    periodic_policy: str,
    frame_stride: int,
    frame_selection_plan: FrameSelectionPlan,
    state: Optional[CartesianCovariance] = None,
    mean: Optional[Sequence[float]] = None,
    solution: Optional[PCAResult] = None,
    vector_sink: Optional[Callable[[Sequence[float]], None]] = None,
    inventory_by_path: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> List[Dict[str, object]]:
    if (mean is None) != (solution is None):
        raise PCAAnalysisError("projection mean and PCA solution must be supplied together")
    segments = plan.replica["segments"]
    assert isinstance(segments, list)
    processor = PeriodicFrameProcessor.from_replica(
        project_data, plan.replica, system_path, len(plan.target_atoms)
    )
    reconstruction_atom_indices = tuple(sorted(
        set(plan.alignment.target_indices) | set(plan.analysis.target_indices)
    ))
    reports: List[Dict[str, object]] = []
    for segment in segments:
        assert isinstance(segment, dict)
        segment_id = str(segment["segment_id"])
        location = f"{plan.system_id}/{plan.replica_id}/{segment_id}"
        trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
        selected_indices = frame_selection_plan[(
            plan.system_id, plan.replica_id, segment_id,
        )]
        before = trajectory_path.stat()
        axis = normalize_segment_axis(segment, time_unit)
        observed = 0
        evaluated = 0
        periodic = 0
        first_axis_value = None
        last_axis_value = None
        projections: List[Dict[str, object]] = []
        processor.begin_segment(
            bool(segment.get("continuous_with_previous", False))
        )
        reader_indices = reader_frame_indices(selected_indices, processor.policy)
        for raw_frame in iter_coordinate_frames(
            trajectory_path, coordinate_unit, reader_indices
        ):
            selected = frame_selected(
                raw_frame.frame_index, selected_indices, frame_stride
            )
            if not selected and processor.policy != "unwrap_continuous":
                continue
            frame = processor.process(
                raw_frame, f"{location}/frame-{raw_frame.frame_index}",
                reconstruction_atom_indices,
            )
            observed += 1
            periodic += int(frame.periodic_cell_present)
            if frame.atom_count != len(plan.target_atoms):
                raise PCAAnalysisError(
                    f"{location} frame {frame.frame_index} has {frame.atom_count} atoms; "
                    f"topology has {len(plan.target_atoms)}"
                )
            if not all(
                math.isfinite(value)
                for coordinate in frame.coordinates_angstrom
                for value in coordinate
            ):
                raise PCAAnalysisError(
                    f"{location} frame {frame.frame_index} contains non-finite coordinates"
                )
            if not selected:
                continue
            transform = best_fit_transform(
                _coordinates_at(frame.coordinates_angstrom, plan.alignment.target_indices),
                _coordinates_at(reference_coordinates, plan.alignment.reference_indices),
            )
            aligned = apply_transform(
                _coordinates_at(frame.coordinates_angstrom, plan.analysis.target_indices),
                transform,
            )
            vector = _flatten(aligned)
            evaluated += 1
            current_axis_value = frame_axis_value(axis, frame.frame_index)
            if first_axis_value is None:
                first_axis_value = current_axis_value
            last_axis_value = current_axis_value
            if state is not None:
                state.update(vector)
            if vector_sink is not None:
                vector_sink(vector)
            if mean is not None and solution is not None:
                scores = project(vector, mean, solution.components)
                projection_row: Dict[str, object] = {
                    "source_frame_index": frame.frame_index,
                    "scores_angstrom": list(scores),
                }
                if axis["kind"] == "physical_time":
                    projection_row.update({
                        "time": current_axis_value,
                        "time_unit": time_unit,
                    })
                else:
                    projection_row["sample_index"] = current_axis_value
                projections.append(projection_row)
        if observed == 0:
            raise PCAAnalysisError(f"{location} trajectory contains no coordinate frames")
        after = trajectory_path.stat()
        fingerprint = {
            "size_bytes": before.st_size,
            "modified_time_ns": before.st_mtime_ns,
        }
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise PCAAnalysisError(f"{location} trajectory changed during read-only analysis")
        inventory = (inventory_by_path or {}).get(str(trajectory_path), {})
        report: Dict[str, object] = {
            "segment_id": segment_id,
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": inventory.get("sha256"),
            "source_fingerprint": fingerprint,
            "observed_frame_count": observed,
            "evaluated_frame_count": evaluated,
            "periodic_cell_frame_count": periodic,
            "periodic_reconstruction_replica_cumulative": processor.report(),
            "frame_axis": axis,
            "evaluated_axis_range": (
                {
                    "start": first_axis_value,
                    "end": last_axis_value,
                    "unit": time_unit if axis["kind"] == "physical_time" else "sample",
                }
                if first_axis_value is not None
                else None
            ),
            "projections": projections if solution is not None else None,
        }
        if axis["kind"] == "physical_time":
            report["timing"] = axis["timing"]
            report["evaluated_time_range"] = report["evaluated_axis_range"]
        else:
            report["sample_axis"] = axis["sample_axis"]
        reports.append(report)
    return reports


def _scan_source_identity(
    reports: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    fields = ("segment_id", "trajectory_path", "source_fingerprint", "frame_axis")
    return [{field: report[field] for field in fields} for report in reports]


def _pca_payload(
    mean: Sequence[float], solution: PCAResult, atoms: Sequence[AtomRecord]
) -> Dict[str, object]:
    atom_count = len(atoms)
    mean_rows = []
    for index, atom in enumerate(atoms):
        offset = index * 3
        mean_rows.append({
            **atom_identity_record(atom, index),
            "mean_x_angstrom": mean[offset],
            "mean_y_angstrom": mean[offset + 1],
            "mean_z_angstrom": mean[offset + 2],
        })
    component_rows = []
    for component in solution.components:
        loadings = []
        for atom_index, atom in enumerate(atoms):
            offset = atom_index * 3
            loadings.append({
                **atom_identity_record(atom, atom_index),
                "loading_x": component.vector[offset],
                "loading_y": component.vector[offset + 1],
                "loading_z": component.vector[offset + 2],
            })
        component_rows.append({
            "component_index": component.component_index,
            "eigenvalue_angstrom2": component.eigenvalue_angstrom2,
            "explained_variance_fraction": component.explained_variance_fraction,
            "cumulative_explained_variance_fraction": component.cumulative_explained_variance_fraction,
            "residual_norm_angstrom2": component.residual_norm_angstrom2,
            "iteration_count": component.iteration_count,
            "converged": component.converged,
            "loadings": loadings,
        })
    return {
        "atom_count": atom_count,
        "feature_count": atom_count * 3,
        "total_variance_angstrom2": solution.total_variance_angstrom2,
        "requested_component_count": solution.requested_component_count,
        "returned_component_count": len(solution.components),
        "numerical_rank_lower_bound": solution.numerical_rank_lower_bound,
        "mean_structure": mean_rows,
        "components": component_rows,
    }


def _projection_values(
    reports: Sequence[Mapping[str, object]], component_count: int
) -> List[List[float]]:
    values = [[] for _ in range(component_count)]
    for report in reports:
        projections = report.get("projections")
        assert isinstance(projections, list)
        for row in projections:
            assert isinstance(row, dict)
            scores = row["scores_angstrom"]
            assert isinstance(scores, list)
            for index, score in enumerate(scores):
                values[index].append(float(score))
    return values


def _base_report(
    module_id: str,
    project_source: Path,
    prepared: Mapping[str, object],
    hash_content: bool,
    issues: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    context = prepared["context"]
    assert isinstance(context, dict)
    reference_path = prepared["reference_path"]
    reference_atoms = prepared["reference_atoms"]
    assert isinstance(reference_path, Path)
    assert isinstance(reference_atoms, list)
    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "module_id": module_id,
        "technical_status": "failed" if error_count else "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(project_source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": context["system_manifest_path"],
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "reference": {
            "path": str(reference_path),
            "format": prepared["reference_format"],
            "sha256": sha256_file(reference_path) if hash_content else None,
            "atom_count": len(reference_atoms),
        },
        "settings": prepared["settings"],
        "frame_selection": prepared["frame_selection_report"],
        "basis_frame_selection": prepared["frame_selection_report"],
        "projection_frame_selection": prepared["projection_frame_selection_report"],
        "solver_contract": (
            dict(_SOLVER)
            if prepared["settings"]["solver"]["method"] == "dense_covariance_v1"  # type: ignore[index]
            else {
                "covariance_denominator": "population_N_with_normalized_sample_weights",
                "component_orientation": _SOLVER["component_orientation"],
                **prepared["settings"]["solver"],  # type: ignore[index]
            }
        ),
        "common_atom_policy": prepared["policy"],
        "periodic_coordinate_policy": prepared["periodic_policy"],
        "sampling_mode": prepared["sampling_mode"],
        "frame_axis_kind": (
            "sample_index"
            if prepared["sampling_mode"] == "AI_ENSEMBLE"
            else "physical_time"
        ),
        "time_unit": prepared["time_unit"],
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": list(issues),
    }


def individual_pca_project(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
    """Fit one independent PCA basis for every declared replica."""

    project_source = Path(project_path).expanduser().resolve(strict=False)
    prepared = _prepare(project_source, "individual_pca", hash_content, False)
    settings = prepared["settings"]
    plans = prepared["plans"]
    reference_frame = prepared["reference_frame"]
    system_path = prepared["system_path"]
    assert isinstance(settings, dict)
    assert isinstance(plans, list)
    assert isinstance(system_path, Path)
    issues: List[Mapping[str, object]] = list(prepared["context"]["issues"])  # type: ignore[index]
    reports_by_system: Dict[str, List[Dict[str, object]]] = {}
    for plan in plans:
        assert isinstance(plan, _ReplicaPlan)
        location = f"{plan.system_id}/{plan.replica_id}"
        feature_count = len(plan.analysis.reference_atoms) * 3
        if feature_count > int(settings["maximum_features"]):
            raise PCAAnalysisError(
                f"{location} analysis selection contains {feature_count} Cartesian features; "
                f"maximum_features is {settings['maximum_features']} because covariance scales quadratically"
            )
        if int(settings["component_count"]) > feature_count:
            raise PCAAnalysisError(
                f"{location} component_count exceeds its {feature_count} Cartesian features"
            )
        for role, correspondence in (("alignment", plan.alignment), ("analysis", plan.analysis)):
            if correspondence.residue_name_mismatch_count:
                issues.append(issue_record(
                    "warning",
                    "RESIDUE_NAME_MISMATCH",
                    f"{location}/{role}",
                    f"{correspondence.residue_name_mismatch_count} mapped atoms differ in residue name because the position policy ignores substitutions",
                ))
        state = CartesianCovariance(feature_count)
        first_pass = _scan_replica(
            plan,
            prepared["project_data"],  # type: ignore[arg-type]
            system_path,
            reference_frame.coordinates_angstrom,  # type: ignore[attr-defined]
            str(prepared["coordinate_unit"]),
            prepared["time_unit"],  # type: ignore[arg-type]
            str(prepared["periodic_policy"]),
            int(settings["frame_stride"]),
            prepared["frame_selection_plan"],  # type: ignore[arg-type]
            state=state,
            inventory_by_path=prepared["inventory_by_path"],  # type: ignore[arg-type]
        )
        minimum_frames = int(settings["minimum_evaluated_frames_per_replica"])
        if state.count < minimum_frames:
            issues.append(issue_record(
                "warning",
                "INSUFFICIENT_REPLICA_FRAMES_FOR_LOCAL_PCA",
                location,
                f"replica has {state.count} evaluated frames; a standalone local PCA "
                f"requires {minimum_frames}. The replica remains available to pooled PCA modules.",
            ))
            topology_inventory = prepared["inventory_by_path"].get(  # type: ignore[union-attr]
                str(plan.topology_path), {}
            )
            reports_by_system.setdefault(plan.system_id, []).append({
                "replica_id": plan.replica_id,
                "technical_status": "insufficient_local_frames",
                "topology_path": str(plan.topology_path),
                "topology_format": plan.topology_format,
                "topology_sha256": topology_inventory.get("sha256"),
                "topology_atom_count": len(plan.target_atoms),
                "evaluated_frame_count": state.count,
                "basis_evaluated_frame_count": state.count,
                "projection_evaluated_frame_count": 0,
                "mappings": {
                    "alignment": plan.alignment.as_dict(),
                    "analysis": plan.analysis.as_dict(),
                },
                "pca": None,
                "segments": first_pass,
            })
            continue
        solution = principal_components(
            state.population_covariance(),
            int(settings["component_count"]),
            eigenvalue_tolerance_angstrom2=float(_SOLVER["eigenvalue_tolerance_angstrom2"]),
            solver_tolerance=float(_SOLVER["solver_tolerance"]),
            maximum_relative_residual=float(_SOLVER["maximum_relative_residual"]),
            maximum_iterations=int(_SOLVER["maximum_iterations"]),
        )
        if len(solution.components) < int(settings["component_count"]):
            issues.append(issue_record(
                "warning",
                "NUMERICAL_RANK_LIMIT",
                location,
                f"requested {settings['component_count']} components but only {len(solution.components)} exceeded the eigenvalue gate",
            ))
        second_pass = _scan_replica(
            plan,
            prepared["project_data"],  # type: ignore[arg-type]
            system_path,
            reference_frame.coordinates_angstrom,  # type: ignore[attr-defined]
            str(prepared["coordinate_unit"]),
            prepared["time_unit"],  # type: ignore[arg-type]
            str(prepared["periodic_policy"]),
            int(settings["projection_frame_stride"]),
            prepared["projection_frame_selection_plan"],  # type: ignore[arg-type]
            mean=state.mean(),
            solution=solution,
            inventory_by_path=prepared["inventory_by_path"],  # type: ignore[arg-type]
        )
        if _scan_source_identity(first_pass) != _scan_source_identity(second_pass):
            raise PCAAnalysisError(f"{location} trajectory identity changed between PCA passes")
        periodic_count = sum(int(segment["periodic_cell_frame_count"]) for segment in second_pass)
        if periodic_count and prepared["periodic_policy"] == "allow_wrapped_diagnostic":
            issues.append(issue_record(
                "warning",
                "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                location,
                f"{periodic_count} frames declare a periodic cell; PCA did not make molecules whole",
            ))
        topology_inventory = prepared["inventory_by_path"].get(str(plan.topology_path), {})  # type: ignore[union-attr]
        reports_by_system.setdefault(plan.system_id, []).append({
            "replica_id": plan.replica_id,
            "technical_status": "complete",
            "topology_path": str(plan.topology_path),
            "topology_format": plan.topology_format,
            "topology_sha256": topology_inventory.get("sha256"),
            "topology_atom_count": len(plan.target_atoms),
            "evaluated_frame_count": sum(
                int(segment["evaluated_frame_count"]) for segment in second_pass
            ),
            "basis_evaluated_frame_count": state.count,
            "projection_evaluated_frame_count": sum(
                int(segment["evaluated_frame_count"]) for segment in second_pass
            ),
            "mappings": {
                "alignment": plan.alignment.as_dict(),
                "analysis": plan.analysis.as_dict(),
            },
            "pca": _pca_payload(state.mean(), solution, plan.analysis.reference_atoms),
            "segments": second_pass,
        })
    basis_selection_report = prepared["frame_selection_report"]
    projection_selection_report = prepared["projection_frame_selection_report"]
    assert isinstance(basis_selection_report, dict)
    assert isinstance(projection_selection_report, dict)
    if int(basis_selection_report["selected_frame_count"]) < int(
        basis_selection_report["source_frame_count"]
    ):
        issues.append(issue_record(
            "warning",
            "FRAME_SUBSAMPLING",
            str(project_source),
            f"PCA fitted its basis on {basis_selection_report['selected_frame_count']} of "
            f"{basis_selection_report['source_frame_count']} source frames under "
            f"{basis_selection_report['mode']}",
        ))
    if int(projection_selection_report["selected_frame_count"]) < int(
        projection_selection_report["source_frame_count"]
    ):
        issues.append(issue_record(
            "warning",
            "PCA_PROJECTION_FRAME_SUBSAMPLING",
            str(project_source),
            f"PCA projected {projection_selection_report['selected_frame_count']} of "
            f"{projection_selection_report['source_frame_count']} source frames under "
            f"{projection_selection_report['mode']}",
        ))
    report = _base_report("individual_pca", project_source, prepared, hash_content, issues)
    report["systems"] = [
        {"system_id": system_id, "replicas": replica_reports}
        for system_id, replica_reports in reports_by_system.items()
    ]
    report["limitations"] = [
        "Each replica has its own mean and PCA basis; scores from different replicas are not directly comparable.",
        "Population covariance uses denominator N over the explicitly selected basis-fit frames; frames are not independent uncertainty units.",
        "Projection may use a different explicit frame-selection contract so a budgeted basis can still label every source frame.",
        "The dense covariance and symmetric LAPACK eigensolver are experimental and guarded by maximum_features.",
        "Component signs are deterministic but physically arbitrary; near-degenerate subspaces can rotate between datasets.",
        "Periodic production analysis requires make_whole or unwrap_continuous with explicit connectivity; allow_wrapped_diagnostic remains diagnostic only.",
        "Technical completion does not establish equilibration, convergence, adequate sampling, states, mechanism, or scientific validity.",
        "No real-trajectory regression fixture has yet been approved; status remains experimental.",
    ]
    return report


def common_pca_project(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
    """Fit one shared PCA basis across every replica on a global atom map."""

    project_source = Path(project_path).expanduser().resolve(strict=False)
    cached = load_cached_project_report(
        "common_pca",
        project_source,
        hash_content=hash_content,
        error_type=PCAAnalysisError,
    )
    if cached is not None:
        return cached
    project_data = load_json(project_source)
    settings = _settings(project_data, "common_pca")
    if "symmetry_expansion" in settings:
        from .oligomer_symmetry import symmetry_expanded_common_pca_project

        return symmetry_expanded_common_pca_project(
            project_source, settings, hash_content=hash_content
        )
    prepared = _prepare(project_source, "common_pca", hash_content, True)
    settings = prepared["settings"]
    plans = prepared["plans"]
    reference_frame = prepared["reference_frame"]
    system_path = prepared["system_path"]
    contract = prepared["contract"]
    assert isinstance(settings, dict)
    assert isinstance(plans, list)
    assert isinstance(system_path, Path)
    assert isinstance(contract, dict)
    first_plan = plans[0]
    assert isinstance(first_plan, _ReplicaPlan)
    analysis_atoms = first_plan.analysis.reference_atoms
    feature_count = len(analysis_atoms) * 3
    solver_settings = settings["solver"]
    assert isinstance(solver_settings, dict)
    solver_method = str(solver_settings["method"])
    if feature_count > int(settings["maximum_features"]):
        raise PCAAnalysisError(
            f"global analysis selection contains {feature_count} Cartesian features; "
            f"maximum_features is {settings['maximum_features']}"
            + (
                " because covariance scales quadratically"
                if solver_method == "dense_covariance_v1"
                else " under the declared truncated-solver resource contract"
            )
        )
    if int(settings["component_count"]) > feature_count:
        raise PCAAnalysisError(
            f"component_count exceeds the {feature_count} global Cartesian features"
        )
    issues: List[Mapping[str, object]] = list(prepared["context"]["issues"])  # type: ignore[index]
    analysis_reference_coverage = plans[0].analysis.reference_coverage
    if analysis_reference_coverage < float(settings["minimum_reference_coverage"]):
        issues.append(issue_record(
            "warning",
            "GLOBAL_COMMON_ANALYSIS_INTERSECTION_EXCLUDES_VARIANT_ATOMS",
            str(project_source),
            (
                f"the exact all-topology analysis intersection retains "
                f"{len(plans[0].analysis.reference_indices)} of "
                f"{plans[0].analysis.reference_selected_count} reference atoms "
                f"({analysis_reference_coverage:.6f}); the declared "
                "minimum_reference_coverage remains enforced on the alignment "
                "selection, while variant-specific analysis atoms are explicitly excluded"
            ),
        ))
    first_passes: List[List[Dict[str, object]]] = []
    weighting = str(settings["basis_weighting"])
    states: List[CartesianCovariance] = []
    basis_counts: List[int] = []
    solver_diagnostics: Dict[str, object]
    if solver_method == "dense_covariance_v1":
        for plan in plans:
            assert isinstance(plan, _ReplicaPlan)
            location = f"{plan.system_id}/{plan.replica_id}"
            for role, correspondence in (("alignment", plan.alignment), ("analysis", plan.analysis)):
                if correspondence.residue_name_mismatch_count:
                    issues.append(issue_record(
                        "warning",
                        "RESIDUE_NAME_MISMATCH",
                        f"{location}/{role}",
                        f"{correspondence.residue_name_mismatch_count} mapped atoms differ in residue name because the position policy ignores substitutions",
                    ))
            state = CartesianCovariance(feature_count)
            first_pass = _scan_replica(
                plan,
                prepared["project_data"],  # type: ignore[arg-type]
                system_path,
                reference_frame.coordinates_angstrom,  # type: ignore[attr-defined]
                str(prepared["coordinate_unit"]),
                prepared["time_unit"],  # type: ignore[arg-type]
                str(prepared["periodic_policy"]),
                int(settings["frame_stride"]),
                prepared["frame_selection_plan"],  # type: ignore[arg-type]
                state=state,
                inventory_by_path=prepared["inventory_by_path"],  # type: ignore[arg-type]
            )
            if state.count == 0:
                raise PCAAnalysisError(
                    f"{location} has no evaluated basis frames"
                )
            if state.count < int(settings["minimum_evaluated_frames_per_replica"]):
                issues.append(issue_record(
                    "warning",
                    "LOW_REPLICA_FRAME_COUNT_POOLED_PCA",
                    location,
                    f"replica contributes {state.count} frame to the shared pooled PCA basis; "
                    "no standalone replica covariance is interpreted",
                ))
            states.append(state)
            basis_counts.append(state.count)
            first_passes.append(first_pass)
        if weighting == "frame":
            pooled = CartesianCovariance(feature_count)
            for state in states:
                pooled.merge(state)
            mean = pooled.mean()
            covariance = pooled.population_covariance()
            total_frames = sum(state.count for state in states)
            weights = [state.count / total_frames for state in states]
        else:
            weights = [1.0 / len(states)] * len(states)
            mean, covariance = mixture_covariance(states, weights)
        solution = principal_components(
            covariance,
            int(settings["component_count"]),
            eigenvalue_tolerance_angstrom2=float(_SOLVER["eigenvalue_tolerance_angstrom2"]),
            solver_tolerance=float(_SOLVER["solver_tolerance"]),
            maximum_relative_residual=float(_SOLVER["maximum_relative_residual"]),
            maximum_iterations=int(_SOLVER["maximum_iterations"]),
        )
        solver_diagnostics = {
            "method": solver_method,
            "sample_count": sum(basis_counts),
            "feature_count": feature_count,
        }
    else:
        basis_selection_report = prepared["frame_selection_report"]
        assert isinstance(basis_selection_report, dict)
        selected_count = int(basis_selection_report["selected_frame_count"])
        sample_elements = selected_count * feature_count
        maximum_elements = int(solver_settings["maximum_sample_matrix_elements"])
        if sample_elements > maximum_elements:
            raise PCAAnalysisError(
                f"randomized PCA sample matrix requires {sample_elements} elements; "
                f"maximum_sample_matrix_elements is {maximum_elements}; reduce only the "
                "basis-fit frame budget while retaining full projection coverage"
            )
        samples = np.empty((selected_count, feature_count), dtype=float)
        cursor = [0]
        replica_slices: List[slice] = []
        for plan in plans:
            assert isinstance(plan, _ReplicaPlan)
            location = f"{plan.system_id}/{plan.replica_id}"
            for role, correspondence in (("alignment", plan.alignment), ("analysis", plan.analysis)):
                if correspondence.residue_name_mismatch_count:
                    issues.append(issue_record(
                        "warning",
                        "RESIDUE_NAME_MISMATCH",
                        f"{location}/{role}",
                        f"{correspondence.residue_name_mismatch_count} mapped atoms differ in residue name because the position policy ignores substitutions",
                    ))
            start = cursor[0]

            def store(vector: Sequence[float]) -> None:
                if cursor[0] >= selected_count:
                    raise PCAAnalysisError(
                        "evaluated basis frames exceed the planned randomized sample matrix"
                    )
                samples[cursor[0], :] = vector
                cursor[0] += 1

            first_pass = _scan_replica(
                plan,
                prepared["project_data"],  # type: ignore[arg-type]
                system_path,
                reference_frame.coordinates_angstrom,  # type: ignore[attr-defined]
                str(prepared["coordinate_unit"]),
                prepared["time_unit"],  # type: ignore[arg-type]
                str(prepared["periodic_policy"]),
                int(settings["frame_stride"]),
                prepared["frame_selection_plan"],  # type: ignore[arg-type]
                vector_sink=store,
                inventory_by_path=prepared["inventory_by_path"],  # type: ignore[arg-type]
            )
            count = cursor[0] - start
            if count == 0:
                raise PCAAnalysisError(
                    f"{location} has no evaluated basis frames"
                )
            if count < int(settings["minimum_evaluated_frames_per_replica"]):
                issues.append(issue_record(
                    "warning",
                    "LOW_REPLICA_FRAME_COUNT_POOLED_PCA",
                    location,
                    f"replica contributes {count} frame to the shared pooled PCA basis; "
                    "no standalone replica covariance is interpreted",
                ))
            basis_counts.append(count)
            replica_slices.append(slice(start, cursor[0]))
            first_passes.append(first_pass)
        if cursor[0] != selected_count:
            raise PCAAnalysisError(
                f"randomized PCA collected {cursor[0]} frames; planner declared {selected_count}"
            )
        sample_weights = np.empty(selected_count, dtype=float)
        if weighting == "frame":
            sample_weights.fill(1.0 / selected_count)
            weights = [count / selected_count for count in basis_counts]
        else:
            weights = [1.0 / len(basis_counts)] * len(basis_counts)
            for replica_slice, count in zip(replica_slices, basis_counts):
                sample_weights[replica_slice] = 1.0 / (len(basis_counts) * count)
        mean, solution, solver_diagnostics = randomized_truncated_pca(
            samples,
            sample_weights,
            int(settings["component_count"]),
            oversampling=int(solver_settings["oversampling"]),
            power_iterations=int(solver_settings["power_iterations"]),
            power_iteration_schedule=solver_settings["power_iteration_schedule"],
            random_seed=int(solver_settings["random_seed"]),
            eigenvalue_tolerance_angstrom2=float(_SOLVER["eigenvalue_tolerance_angstrom2"]),
            maximum_relative_residual=float(solver_settings["maximum_relative_residual"]),
        )
        solver_diagnostics["sample_matrix_elements"] = sample_elements
        solver_diagnostics["sample_matrix_bytes_float64"] = sample_elements * 8
    if len(solution.components) < int(settings["component_count"]):
        issues.append(issue_record(
            "warning",
            "NUMERICAL_RANK_LIMIT",
            str(project_source),
            f"requested {settings['component_count']} components but only {len(solution.components)} exceeded the eigenvalue gate",
        ))
    replica_reports: Dict[str, List[Dict[str, object]]] = {}
    all_reports: List[List[Dict[str, object]]] = []
    replica_summaries = []
    for plan, basis_count, first_pass, basis_weight in zip(
        plans, basis_counts, first_passes, weights
    ):
        assert isinstance(plan, _ReplicaPlan)
        location = f"{plan.system_id}/{plan.replica_id}"
        second_pass = _scan_replica(
            plan,
            prepared["project_data"],  # type: ignore[arg-type]
            system_path,
            reference_frame.coordinates_angstrom,  # type: ignore[attr-defined]
            str(prepared["coordinate_unit"]),
            prepared["time_unit"],  # type: ignore[arg-type]
            str(prepared["periodic_policy"]),
            int(settings["projection_frame_stride"]),
            prepared["projection_frame_selection_plan"],  # type: ignore[arg-type]
            mean=mean,
            solution=solution,
            inventory_by_path=prepared["inventory_by_path"],  # type: ignore[arg-type]
        )
        if _scan_source_identity(first_pass) != _scan_source_identity(second_pass):
            raise PCAAnalysisError(f"{location} trajectory identity changed between PCA passes")
        periodic_count = sum(int(segment["periodic_cell_frame_count"]) for segment in second_pass)
        if periodic_count and prepared["periodic_policy"] == "allow_wrapped_diagnostic":
            issues.append(issue_record(
                "warning",
                "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                location,
                f"{periodic_count} frames declare a periodic cell; PCA did not make molecules whole",
            ))
        scores = _projection_values(second_pass, len(solution.components))
        summaries = [sample_summary(component_scores) for component_scores in scores]
        topology_inventory = prepared["inventory_by_path"].get(str(plan.topology_path), {})  # type: ignore[union-attr]
        row = {
            "replica_id": plan.replica_id,
            "topology_path": str(plan.topology_path),
            "topology_format": plan.topology_format,
            "topology_sha256": topology_inventory.get("sha256"),
            "topology_atom_count": len(plan.target_atoms),
            "evaluated_frame_count": sum(
                int(segment["evaluated_frame_count"]) for segment in second_pass
            ),
            "basis_evaluated_frame_count": basis_count,
            "projection_evaluated_frame_count": sum(
                int(segment["evaluated_frame_count"]) for segment in second_pass
            ),
            "basis_weight": basis_weight,
            "mappings": {
                "alignment": plan.alignment.as_dict(),
                "analysis": plan.analysis.as_dict(),
            },
            "projection_summaries_angstrom": summaries,
            "segments": second_pass,
        }
        replica_reports.setdefault(plan.system_id, []).append(row)
        replica_summaries.append({
            "system_id": plan.system_id,
            "replica_id": plan.replica_id,
            "evaluated_frame_count": basis_count,
            "basis_weight": basis_weight,
            "projection_summaries_angstrom": summaries,
        })
        all_reports.append(second_pass)
    systems_out = []
    reference_system = str(contract["reference_system"])
    reference_means: Optional[List[Optional[float]]] = None
    for system_id, replicas in replica_reports.items():
        system_segment_reports = [
            segment
            for replica in replicas
            for segment in replica["segments"]
        ]
        scores = _projection_values(system_segment_reports, len(solution.components))
        summaries = [sample_summary(component_scores) for component_scores in scores]
        if system_id == reference_system:
            reference_means = [summary["mean"] for summary in summaries]
        systems_out.append({
            "system_id": system_id,
            "frame_pooled_projection_summaries_angstrom": summaries,
            "replicas": replicas,
        })
    if reference_means is None:
        raise PCAAnalysisError("reference system produced no PCA projections")
    for system in systems_out:
        summaries = system["frame_pooled_projection_summaries_angstrom"]
        assert isinstance(summaries, list)
        system["projection_mean_difference_from_reference_angstrom"] = [
            (
                None
                if summary["mean"] is None or reference_mean is None
                else float(summary["mean"]) - float(reference_mean)
            )
            for summary, reference_mean in zip(summaries, reference_means)
        ]
    basis_selection_report = prepared["frame_selection_report"]
    projection_selection_report = prepared["projection_frame_selection_report"]
    assert isinstance(basis_selection_report, dict)
    assert isinstance(projection_selection_report, dict)
    if int(basis_selection_report["selected_frame_count"]) < int(
        basis_selection_report["source_frame_count"]
    ):
        issues.append(issue_record(
            "warning",
            "FRAME_SUBSAMPLING",
            str(project_source),
            f"PCA fitted its shared basis on {basis_selection_report['selected_frame_count']} of "
            f"{basis_selection_report['source_frame_count']} source frames under "
            f"{basis_selection_report['mode']}",
        ))
    if int(projection_selection_report["selected_frame_count"]) < int(
        projection_selection_report["source_frame_count"]
    ):
        issues.append(issue_record(
            "warning",
            "PCA_PROJECTION_FRAME_SUBSAMPLING",
            str(project_source),
            f"PCA projected {projection_selection_report['selected_frame_count']} of "
            f"{projection_selection_report['source_frame_count']} source frames under "
            f"{projection_selection_report['mode']}",
        ))
    report = _base_report("common_pca", project_source, prepared, hash_content, issues)
    report["module_contract_sha256"] = project_module_contract_sha256(
        "common_pca", project_source
    )
    report["reference_system_id"] = reference_system
    report["basis"] = {
        "basis_weighting": weighting,
        "replica_count": len(basis_counts),
        "evaluated_frame_count": sum(basis_counts),
        "replica_contributions": replica_summaries,
        "solver_diagnostics": solver_diagnostics,
        "pca": _pca_payload(mean, solution, analysis_atoms),
        "common_atom_contract": {
            "alignment_reference_coverage_gate": float(
                settings["minimum_reference_coverage"]
            ),
            "alignment_reference_coverage": plans[0].alignment.reference_coverage,
            "analysis_mapping": "exact_all_topology_intersection_v1",
            "analysis_reference_selected_atom_count": (
                plans[0].analysis.reference_selected_count
            ),
            "analysis_common_atom_count": len(plans[0].analysis.reference_indices),
            "analysis_reference_coverage": analysis_reference_coverage,
            "variant_specific_analysis_atoms_excluded": (
                plans[0].analysis.reference_selected_count
                - len(plans[0].analysis.reference_indices)
            ),
        },
    }
    report["systems"] = systems_out
    report["limitations"] = [
        "The alignment basis must pass minimum_reference_coverage; the shared analysis basis is the exact all-topology intersection and explicitly excludes variant-specific atoms without treating their absence as failed structural correspondence.",
        "basis_weighting=frame gives every evaluated frame equal weight; basis_weighting=replica_equal gives each replica population equal total weight.",
        "Projection summaries are descriptive; frames are not independent uncertainty units.",
        "Projection may use a different explicit frame-selection contract so a budgeted shared basis can still label every source frame.",
        "Run both weighting modes and atom-map alternatives when sampling imbalance or mapping choice could affect conclusions.",
        (
            "The dense covariance and symmetric LAPACK eigensolver are guarded by maximum_features."
            if solver_method == "dense_covariance_v1"
            else "The truncated randomized solver is deterministic for its declared seed, reports exact covariance-action residuals, and requires atom-selection, basis-budget, seed, oversampling, and power-iteration sensitivity before publication."
        ),
        "Component signs are deterministic but physically arbitrary; near-degenerate subspaces require subspace-level sensitivity analysis.",
        "Periodic production analysis requires make_whole or unwrap_continuous with explicit connectivity; allow_wrapped_diagnostic remains diagnostic only.",
        "Technical completion does not establish equilibration, convergence, adequate sampling, basins, mechanism, or scientific validity.",
        "No real-trajectory regression fixture has yet been approved; status remains experimental.",
    ]
    return report


def _safe(module_id: str, project_path: Path, hash_content: bool) -> Dict[str, object]:
    runner = individual_pca_project if module_id == "individual_pca" else common_pca_project
    try:
        return runner(project_path, hash_content=hash_content)
    except (
        ManifestValidationError,
        PCAAnalysisError,
        PCAError,
        AtomMappingError,
        CoordinateReadError,
        GeometryError,
        PeriodicReconstructionError,
        TrajectoryContractError,
        ValueError,
        OSError,
        StopIteration,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": module_id,
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {
                    "severity": "error",
                    "code": "PCA_INVALID",
                    "message": message,
                }
                for message in messages
            ],
        }


def individual_pca_project_safe(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
    """Return a machine-readable failure rather than an uncaught PCA exception."""

    return _safe("individual_pca", project_path, hash_content)


def common_pca_project_safe(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
    """Return a machine-readable failure rather than an uncaught PCA exception."""

    return _safe("common_pca", project_path, hash_content)
