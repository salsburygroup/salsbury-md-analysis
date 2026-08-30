"""Deterministic Shrake--Rupley solvent-accessible surface area analysis."""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .moments import sample_summary
from .scalar_distributions import ScalarDistributionError, analyze_scalar_distribution
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .replica_execution import ReplicaPartial
from .replica_module_execution import (
    execute_replica_final_module,
    merge_frame_selection_reports,
    restore_source_provenance,
    unique_issues,
)
from .selections import select_atoms
from .trajectory_contracts import (
    TrajectoryContractError,
    frame_axis_value,
    normalize_segment_axis,
)
from .validation import integer_at_least


class SASAAnalysisError(ValueError):
    """Raised when a SASA definition or numerical execution is unsafe."""


# Bondi-style van der Waals radii in angstrom. Unknown elements fail closed.
VDW_RADII_ANGSTROM = {
    "H": 1.20,
    "B": 1.92,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "SI": 2.10,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "SE": 1.90,
    "BR": 1.85,
    "I": 1.98,
}

# The low-level kernel permits smaller point sets for analytic unit fixtures,
# but project execution must not treat a very coarse sphere as scientific SASA.
MINIMUM_PROJECT_SPHERE_POINT_COUNT = 240
VALIDATED_SPHERE_POINT_COUNT = 960


def sphere_points(point_count: int) -> np.ndarray:
    """Return deterministic approximately uniform unit-sphere points."""

    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 24:
        raise SASAAnalysisError("sphere_point_count must be an integer of at least 24")
    indices = np.arange(point_count, dtype=float)
    y = -1.0 + (2.0 * indices + 1.0) / point_count
    radial = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    angle = indices * (math.pi * (3.0 - math.sqrt(5.0)))
    return np.column_stack((radial * np.cos(angle), y, radial * np.sin(angle)))


def shrake_rupley_sasa(
    coordinates_angstrom: Sequence[Sequence[float]],
    elements: Sequence[str],
    *,
    surface_atom_indices: Sequence[int] | None = None,
    occluder_atom_indices: Sequence[int] | None = None,
    probe_radius_angstrom: float = 1.4,
    sphere_point_count: int = 960,
    element_radii_overrides_angstrom: Mapping[str, float] | None = None,
) -> Tuple[float, ...]:
    """Calculate per-surface-atom SASA in square angstrom.

    Surface atoms are reported in the supplied index order. Occluder atoms
    determine which probe-center points are buried. The calculation is
    intentionally nonperiodic; project execution must reconstruct molecules
    before calling this kernel.
    """

    coordinates = np.asarray(coordinates_angstrom, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,) or coordinates.shape[0] == 0:
        raise SASAAnalysisError("coordinates must be a nonempty N by 3 array")
    if not np.isfinite(coordinates).all():
        raise SASAAnalysisError("coordinates contain a non-finite value")
    if len(elements) != coordinates.shape[0]:
        raise SASAAnalysisError("element count must equal coordinate atom count")
    if (
        isinstance(probe_radius_angstrom, bool)
        or not isinstance(probe_radius_angstrom, (int, float))
        or not math.isfinite(float(probe_radius_angstrom))
        or float(probe_radius_angstrom) <= 0.0
    ):
        raise SASAAnalysisError("probe_radius_angstrom must be finite and positive")
    radii = dict(VDW_RADII_ANGSTROM)
    for raw_element, raw_radius in (element_radii_overrides_angstrom or {}).items():
        element = str(raw_element).strip().upper()
        if not element or isinstance(raw_radius, bool) or not isinstance(raw_radius, (int, float)) or not math.isfinite(float(raw_radius)) or float(raw_radius) <= 0.0:
            raise SASAAnalysisError("element radii overrides must map element names to finite positive values")
        radii[element] = float(raw_radius)
    normalized_elements = tuple(str(value).strip().upper() for value in elements)
    atom_count = coordinates.shape[0]
    surface = tuple(range(atom_count)) if surface_atom_indices is None else tuple(surface_atom_indices)
    occluders = tuple(range(atom_count)) if occluder_atom_indices is None else tuple(occluder_atom_indices)
    for label, indices in (("surface_atom_indices", surface), ("occluder_atom_indices", occluders)):
        if not indices:
            raise SASAAnalysisError(f"{label} must be nonempty")
        if len(set(indices)) != len(indices):
            raise SASAAnalysisError(f"{label} contains duplicate indices")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= atom_count for index in indices):
            raise SASAAnalysisError(f"{label} contains an out-of-range index")
    used_indices = set(surface) | set(occluders)
    unknown = sorted({
        normalized_elements[index]
        for index in used_indices
        if normalized_elements[index] not in radii
    })
    if unknown:
        raise SASAAnalysisError("no van der Waals radius for element(s): " + ", ".join(unknown))

    unit_points = sphere_points(sphere_point_count)
    inflated = np.full(atom_count, np.nan, dtype=float)
    for index in used_indices:
        inflated[index] = radii[normalized_elements[index]] + float(probe_radius_angstrom)
    occluder_coordinates = coordinates[list(occluders)]
    occluder_tree = cKDTree(occluder_coordinates)
    maximum_occluder_radius = max(inflated[index] for index in occluders)
    areas: List[float] = []
    for atom_index in surface:
        radius = inflated[atom_index]
        test_points = coordinates[atom_index] + radius * unit_points
        accessible = np.ones(sphere_point_count, dtype=bool)
        neighbor_positions = occluder_tree.query_ball_point(
            coordinates[atom_index], radius + maximum_occluder_radius
        )
        for neighbor_position in neighbor_positions:
            occluder_index = occluders[neighbor_position]
            if occluder_index == atom_index:
                continue
            center_delta = coordinates[occluder_index] - coordinates[atom_index]
            center_cutoff = radius + inflated[occluder_index]
            if float(np.dot(center_delta, center_delta)) >= center_cutoff * center_cutoff:
                continue
            delta = test_points[accessible] - coordinates[occluder_index]
            visible = np.einsum("ij,ij->i", delta, delta) >= inflated[occluder_index] ** 2
            remaining = np.flatnonzero(accessible)
            accessible[remaining[~visible]] = False
            if not accessible.any():
                break
        areas.append(4.0 * math.pi * radius * radius * float(accessible.mean()))
    return tuple(areas)


_positive_integer = partial(integer_at_least, error_type=SASAAnalysisError)


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("solvent_accessible_surface_area") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise SASAAnalysisError("definitions.solvent_accessible_surface_area must be an object")
    required = {
        "surface_selection",
        "occluder_selection",
        "probe_radius_angstrom",
        "sphere_point_count",
        "frame_stride",
        "maximum_surface_atoms",
        "maximum_observations",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(
        required | {
            "element_radii_overrides_angstrom", "frame_selection", "output_detail",
        }
    ))
    if missing:
        raise SASAAnalysisError("SASA settings missing: " + ", ".join(missing))
    if unknown:
        raise SASAAnalysisError("SASA settings contain unknown fields: " + ", ".join(unknown))
    selections = project.get("selections")
    if not isinstance(selections, dict):
        raise SASAAnalysisError("project selections must be an object")
    for label in ("surface_selection", "occluder_selection"):
        selection_id = raw[label]
        if not isinstance(selection_id, str) or selection_id not in selections:
            raise SASAAnalysisError(f"{label} must reference a declared named selection")
    probe = raw["probe_radius_angstrom"]
    if (
        isinstance(probe, bool)
        or not isinstance(probe, (int, float))
        or not math.isfinite(float(probe))
        or float(probe) <= 0.0
    ):
        raise SASAAnalysisError("probe_radius_angstrom must be finite and positive")
    overrides = raw.get("element_radii_overrides_angstrom", {})
    if not isinstance(overrides, dict):
        raise SASAAnalysisError("element_radii_overrides_angstrom must be an object")
    normalized_overrides = {}
    for element, radius in overrides.items():
        key = str(element).strip().upper()
        if not key or isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(float(radius)) or float(radius) <= 0.0:
            raise SASAAnalysisError("element radii overrides must map element names to finite positive values")
        normalized_overrides[key] = float(radius)
    frame_stride = _positive_integer(raw["frame_stride"], "frame_stride")
    output_detail = raw.get("output_detail", "bounded_summary_v1")
    if output_detail not in {"bounded_summary_v1", "full_atom_timeseries"}:
        raise SASAAnalysisError(
            "output_detail must be bounded_summary_v1 or full_atom_timeseries"
        )
    return {
        "surface_selection": raw["surface_selection"],
        "occluder_selection": raw["occluder_selection"],
        "probe_radius_angstrom": float(probe),
        "sphere_point_count": _positive_integer(
            raw["sphere_point_count"],
            "sphere_point_count",
            MINIMUM_PROJECT_SPHERE_POINT_COUNT,
        ),
        "frame_stride": frame_stride,
        "frame_selection": normalize_frame_selection(
            raw.get("frame_selection"), frame_stride,
            error_type=SASAAnalysisError,
        ),
        "maximum_surface_atoms": _positive_integer(raw["maximum_surface_atoms"], "maximum_surface_atoms"),
        "maximum_observations": _positive_integer(raw["maximum_observations"], "maximum_observations"),
        "element_radii_overrides_angstrom": normalized_overrides,
        "output_detail": output_detail,
    }


def _residue_identity(atom: AtomRecord) -> Tuple[str, int, str, str]:
    return atom.chain_id, atom.residue_number, atom.insertion_code, atom.residue_name


class _VectorMoments:
    """Bounded-memory exact first and second moments for a fixed-width vector."""

    def __init__(self, width: int):
        self.count = 0
        self.mean = np.zeros(width, dtype=float)
        self.m2 = np.zeros(width, dtype=float)

    def update(self, values: Sequence[float]) -> None:
        vector = np.asarray(values, dtype=float)
        if vector.shape != self.mean.shape or not np.isfinite(vector).all():
            raise SASAAnalysisError("SASA streaming summary vector is malformed")
        self.count += 1
        delta = vector - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (vector - self.mean)

    def summary(self, index: int) -> Dict[str, object]:
        if self.count <= 0:
            return {"count": 0, "mean": None, "sample_sd": None, "sem": None}
        if self.count == 1:
            return {
                "count": 1, "mean": float(self.mean[index]),
                "sample_sd": None, "sem": None,
            }
        sample_sd = math.sqrt(float(self.m2[index]) / (self.count - 1))
        return {
            "count": self.count,
            "mean": float(self.mean[index]),
            "sample_sd": sample_sd,
            "sem": sample_sd / math.sqrt(self.count),
        }


def _solvent_accessible_surface_area_project_serial(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(context["system_manifest_path"])
    system = load_json(system_path)
    selections = project["selections"]
    assert isinstance(selections, dict)
    coordinate_unit = str(project["coordinate_unit"])
    output_time_unit = project.get("time_unit")
    periodic_policy = str(project["periodic_coordinate_policy"])
    frame_selection_plan, frame_selection_report = plan_frame_selection(
        system, system_path, coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        error_type=SASAAnalysisError,
    )
    issues = [issue for issue in context.get("warnings", []) if isinstance(issue, dict)]
    if int(settings["sphere_point_count"]) < VALIDATED_SPHERE_POINT_COUNT:
        issues.append({
            "severity": "warning",
            "code": "SASA_RESOLUTION_SENSITIVITY_REQUIRED",
            "location": str(source),
            "message": (
                f"sphere_point_count={settings['sphere_point_count']} is below the "
                f"{VALIDATED_SPHERE_POINT_COUNT}-point cross-validation setting; "
                "repeat at higher resolution and establish result stability before interpretation"
            ),
        })
    replica_reports: List[Dict[str, object]] = []
    total_observations = 0

    for raw_system in system["systems"]:
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            surface_atoms = select_atoms(
                atoms,
                selections[str(settings["surface_selection"])],
                str(settings["surface_selection"]),
            )
            occluder_atoms = select_atoms(
                atoms,
                selections[str(settings["occluder_selection"])],
                str(settings["occluder_selection"]),
            )
            if len(surface_atoms) > int(settings["maximum_surface_atoms"]):
                raise SASAAnalysisError(
                    f"{system_id}/{replica_id} surface selection has {len(surface_atoms)} atoms; "
                    f"maximum_surface_atoms is {settings['maximum_surface_atoms']}"
                )
            surface_indices = tuple(atom.atom_index for atom in surface_atoms)
            occluder_indices = tuple(atom.atom_index for atom in occluder_atoms)
            reconstruction_atom_indices = tuple(sorted(
                set(surface_indices) | set(occluder_indices)
            ))
            processor = PeriodicFrameProcessor.from_replica(project, replica, system_path, len(atoms))
            full_detail = settings["output_detail"] == "full_atom_timeseries"
            frames: List[Dict[str, object]] = []
            total_timeseries: List[Dict[str, object]] = []
            atom_moments = _VectorMoments(len(surface_atoms))
            residue_identities = sorted({_residue_identity(atom) for atom in surface_atoms})
            residue_index = {identity: index for index, identity in enumerate(residue_identities)}
            surface_residue_indices = np.asarray([
                residue_index[_residue_identity(atom)] for atom in surface_atoms
            ], dtype=int)
            residue_moments = _VectorMoments(len(residue_identities))
            segment_reports: List[Dict[str, object]] = []
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                selected_indices = frame_selection_plan[(
                    system_id, replica_id, segment_id,
                )]
                axis = normalize_segment_axis(segment, str(output_time_unit) if output_time_unit else None)
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                evaluated = 0
                periodic_frames = 0
                reader_indices = reader_frame_indices(
                    selected_indices, processor.policy
                )
                for raw_frame in iter_coordinate_frames(
                    trajectory_path, coordinate_unit, reader_indices
                ):
                    selected = frame_selected(
                        raw_frame.frame_index, selected_indices,
                        int(settings["frame_stride"]),
                    )
                    if not selected and processor.policy != "unwrap_continuous":
                        continue
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction_atom_indices,
                    )
                    if frame.atom_count != len(atoms):
                        raise SASAAnalysisError("trajectory/topology atom count mismatch")
                    periodic_frames += int(frame.periodic_cell_present)
                    if frame.periodic_cell_present and periodic_policy == "allow_wrapped_diagnostic":
                        raise SASAAnalysisError(
                            "SASA refuses wrapped periodic coordinates; use make_whole or unwrap_continuous"
                        )
                    if not selected:
                        continue
                    values = shrake_rupley_sasa(
                        frame.coordinates_angstrom,
                        [atom.element for atom in atoms],
                        surface_atom_indices=surface_indices,
                        occluder_atom_indices=occluder_indices,
                        probe_radius_angstrom=float(settings["probe_radius_angstrom"]),
                        sphere_point_count=int(settings["sphere_point_count"]),
                        element_radii_overrides_angstrom=settings["element_radii_overrides_angstrom"],
                    )
                    residue_values: Dict[Tuple[str, int, str, str], float] = {}
                    atom_rows = []
                    for atom, value in zip(surface_atoms, values):
                        residue = _residue_identity(atom)
                        residue_values[residue] = residue_values.get(residue, 0.0) + value
                        atom_rows.append({"atom_index": atom.atom_index, "sasa_angstrom2": value})
                    residue_rows = [
                        {
                            "chain_id": residue[0],
                            "residue_number": residue[1],
                            "insertion_code": residue[2],
                            "residue_name": residue[3],
                            "sasa_angstrom2": value,
                        }
                        for residue, value in sorted(residue_values.items())
                    ]
                    total_row = {
                        "segment_id": segment_id,
                        "source_frame_index": frame.frame_index,
                        "axis_kind": axis["kind"],
                        "axis_value": frame_axis_value(axis, frame.frame_index),
                        "total_sasa_angstrom2": sum(values),
                    }
                    total_timeseries.append(total_row)
                    atom_moments.update(values)
                    residue_vector = np.bincount(
                        surface_residue_indices,
                        weights=np.asarray(values, dtype=float),
                        minlength=len(residue_identities),
                    )
                    residue_moments.update(residue_vector)
                    if full_detail:
                        frames.append({
                            **total_row,
                            "per_atom": atom_rows,
                            "per_residue": residue_rows,
                        })
                    evaluated += 1
                    total_observations += len(values)
                    if total_observations > int(settings["maximum_observations"]):
                        raise SASAAnalysisError("maximum_observations gate exceeded")
                segment_reports.append({
                    "segment_id": segment_id,
                    "evaluated_frame_count": evaluated,
                    "periodic_cell_frame_count": periodic_frames,
                    "periodic_reconstruction_replica_cumulative": processor.report(),
                })
            if not total_timeseries:
                raise SASAAnalysisError(f"{system_id}/{replica_id} produced no evaluated frames")
            distribution_segments = []
            for segment_report in segment_reports:
                segment_id = str(segment_report["segment_id"])
                records = [
                    {
                        "source_frame_index": int(row["source_frame_index"]),
                        "value": float(row["total_sasa_angstrom2"]),
                    }
                    for row in total_timeseries if row["segment_id"] == segment_id
                ]
                if records:
                    distribution_segments.append((
                        {"system_id": system_id, "replica_id": replica_id, "segment_id": segment_id},
                        records,
                    ))
            try:
                total_distribution = analyze_scalar_distribution(
                    distribution_segments,
                    binning_rule="scott",
                    padding_fraction=0.0,
                    minimum_bins=2,
                    maximum_bins=200,
                    retain_assignments=False,
                    retain_residence_runs=False,
                )
            except ScalarDistributionError as exc:
                total_distribution = {
                    "technical_status": "not_estimable",
                    "reason": str(exc),
                    "binning": {"rule": "scott"},
                }
            replica_report = {
                "system_id": system_id,
                "replica_id": replica_id,
                "topology_path": str(topology_path),
                "surface_atom_count": len(surface_atoms),
                "occluder_atom_count": len(occluder_atoms),
                "evaluated_frame_count": len(total_timeseries),
                "total_sasa_summary_angstrom2": sample_summary(
                    [float(frame["total_sasa_angstrom2"]) for frame in total_timeseries]
                ),
                "total_sasa_distribution": total_distribution,
                "total_sasa_timeseries": total_timeseries,
                "per_atom_summaries": [
                    {
                        "atom_index": atom.atom_index,
                        "summary_angstrom2": atom_moments.summary(index),
                    }
                    for index, atom in enumerate(surface_atoms)
                ],
                "per_residue_summaries": [
                    {
                        "chain_id": key[0], "residue_number": key[1],
                        "insertion_code": key[2], "residue_name": key[3],
                        "summary_angstrom2": residue_moments.summary(index),
                    }
                    for index, key in enumerate(residue_identities)
                ],
                "segments": segment_reports,
                "output_detail": settings["output_detail"],
            }
            if full_detail:
                replica_report["frames"] = frames
            replica_reports.append(replica_report)

    if int(frame_selection_report["selected_frame_count"]) < int(
        frame_selection_report["source_frame_count"]
    ):
        issues.append({
            "severity": "warning", "code": "FRAME_SUBSAMPLING", "location": str(source),
            "message": (
                f"SASA evaluated {frame_selection_report['selected_frame_count']} of "
                f"{frame_selection_report['source_frame_count']} source frames under "
                f"{frame_selection_report['mode']}"
            ),
        })
    return {
        "module_id": "solvent_accessible_surface_area",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "algorithm": "Shrake-Rupley probe-center point sampling",
        "radii_contract": {
            "base": "Bondi-style element radii in VDW_RADII_ANGSTROM",
            "declared_overrides_angstrom": settings["element_radii_overrides_angstrom"],
        },
        "settings": settings,
        "frame_selection": frame_selection_report,
        "observation_count": total_observations,
        "replicas": replica_reports,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "SASA depends on the declared atom, occluder, radii, probe-radius, and sphere-point definitions.",
            "Project execution refuses fewer than 240 sphere points and warns below the 960-point cross-validation setting; every project still requires resolution sensitivity appropriate to its comparisons.",
            "Periodic trajectories must use connectivity-aware make_whole or unwrap_continuous reconstruction; wrapped diagnostic execution is refused.",
            "Disconnected components require an independently validated relative-image convention before their mutual occlusion is interpreted.",
            "Surface-area changes do not by themselves establish binding, stability, or functional mechanism.",
            "The bounded production output retains exact totals, per-atom and per-residue first and second moments, and a Scott-rule total-SASA histogram while omitting duplicated per-frame atom and residue dictionaries.",
        ],
    }


def _reduce_sasa_replica_reports(
    partials: Sequence[ReplicaPartial[Dict[str, object]]],
    source_context: Dict[str, object],
) -> Dict[str, object]:
    reports = [partial.value for partial in partials]
    first = dict(reports[0])
    for report in reports[1:]:
        for key in ("module_id", "settings", "algorithm", "radii_contract"):
            if report.get(key) != first.get(key):
                raise SASAAnalysisError(f"replica SASA reports disagree on {key}")
    first["frame_selection"] = merge_frame_selection_reports([
        report["frame_selection"] for report in reports
        if isinstance(report.get("frame_selection"), dict)
    ])
    first["observation_count"] = sum(
        int(report.get("observation_count", 0)) for report in reports
    )
    maximum = int(first["settings"]["maximum_observations"])  # type: ignore[index]
    if int(first["observation_count"]) > maximum:
        raise SASAAnalysisError(
            "parallel SASA observation count exceeds maximum_observations"
        )
    first["replicas"] = [
        row for report in reports for row in report.get("replicas", [])
    ]
    issues = unique_issues(reports)
    first["issues"] = issues
    first["error_count"] = sum(issue.get("severity") == "error" for issue in issues)
    first["warning_count"] = sum(
        issue.get("severity") == "warning" for issue in issues
    )
    restore_source_provenance(first, source_context)
    return first


def solvent_accessible_surface_area_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Run replica workers and reduce replica-final SASA estimates exactly."""

    project = load_json(Path(project_path).expanduser().resolve(strict=False))
    settings = _settings(project)
    selection = settings.get("frame_selection")
    if isinstance(selection, dict) and selection.get("mode") == "auto_resource_budget_v1":
        # This legacy mode divides a wall-time budget by the project replica
        # count.  Resolve it in the campaign planner before replica sharding.
        return _solvent_accessible_surface_area_project_serial(
            project_path, hash_content=hash_content
        )
    return execute_replica_final_module(
        project_path,
        runner_id="sasa",
        hash_content=hash_content,
        reducer=_reduce_sasa_replica_reports,
    )


def solvent_accessible_surface_area_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return solvent_accessible_surface_area_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError,
        SASAAnalysisError,
        AtomMappingError,
        CoordinateReadError,
        PeriodicReconstructionError,
        TrajectoryContractError,
        OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "solvent_accessible_surface_area",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "SASA_INVALID", "message": message}
                for message in messages
            ],
        }
