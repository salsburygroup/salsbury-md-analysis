"""Species-resolved ion atmospheres around chemically declared solute groups.

This module deliberately separates a geometric atmosphere from biological
binding.  Every ion species present in the topology is retained, including
reference-distant mobile ions.  Local distances use exact triclinic minimum
images and therefore do not require whole-solvent reconstruction.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .atom_mapping import AtomMappingError, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import frame_selected, normalize_frame_selection, plan_frame_selection
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .periodic import PeriodicReconstructionError, minimum_image_displacement
from .replica_execution import ReplicaPartial
from .replica_module_execution import (
    execute_replica_final_module,
    merge_frame_selection_reports,
    restore_source_provenance,
    unique_issues,
)
from .trajectory_contracts import TrajectoryContractError
from .validation import positive_integer


class IonAtmosphereError(ValueError):
    """Raised when a species-resolved ion atmosphere is not safely defined."""


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("ion_atmosphere") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise IonAtmosphereError("definitions.ion_atmosphere must be an object")
    allowed = {
        "frame_stride", "frame_selection", "maximum_frames",
        "shell_cutoffs_angstrom", "ion_groups", "target_groups",
    }
    if set(raw).difference(allowed) or not {
        "frame_stride", "maximum_frames", "shell_cutoffs_angstrom",
        "ion_groups", "target_groups",
    }.issubset(raw):
        raise IonAtmosphereError("ion_atmosphere fields do not match the contract")
    stride = positive_integer(raw["frame_stride"], "frame_stride", error_type=IonAtmosphereError)
    maximum = positive_integer(raw["maximum_frames"], "maximum_frames", error_type=IonAtmosphereError)
    cutoffs = raw["shell_cutoffs_angstrom"]
    if (
        not isinstance(cutoffs, list) or not cutoffs
        or any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) or float(value) <= 0.0
               for value in cutoffs)
    ):
        raise IonAtmosphereError("shell_cutoffs_angstrom must contain positive finite numbers")
    normalized_cutoffs = sorted({float(value) for value in cutoffs})

    def groups(field: str) -> list[Dict[str, object]]:
        rows = raw[field]
        if not isinstance(rows, list) or not rows:
            raise IonAtmosphereError(f"{field} must be a nonempty array")
        normalized: list[Dict[str, object]] = []
        identifiers: set[str] = set()
        expected = (
            {"species", "charge_class", "atom_indices"}
            if field == "ion_groups" else {"target_id", "atom_indices"}
        )
        for row in rows:
            if not isinstance(row, dict) or set(row) != expected:
                raise IonAtmosphereError(f"{field} row fields do not match the contract")
            identifier = str(row["species" if field == "ion_groups" else "target_id"]).strip()
            indices = row["atom_indices"]
            if (
                not identifier or identifier in identifiers
                or not isinstance(indices, list) or not indices
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in indices)
                or len(set(indices)) != len(indices)
            ):
                raise IonAtmosphereError(f"{field} identifiers and atom indices must be valid and unique")
            identifiers.add(identifier)
            normalized.append({**row, "atom_indices": list(indices)})
        return normalized

    return {
        "frame_stride": stride,
        "frame_selection": normalize_frame_selection(
            raw.get("frame_selection"), stride, error_type=IonAtmosphereError,
        ),
        "maximum_frames": maximum,
        "shell_cutoffs_angstrom": normalized_cutoffs,
        "ion_groups": groups("ion_groups"),
        "target_groups": groups("target_groups"),
    }


def _distance(
    first: Sequence[float], second: Sequence[float], cell: object,
) -> float:
    displacement = tuple(float(second[axis]) - float(first[axis]) for axis in range(3))
    if cell is None:
        raise IonAtmosphereError("ion-atmosphere analysis requires a periodic cell")
    image = minimum_image_displacement(displacement, cell)  # type: ignore[arg-type]
    return math.sqrt(sum(value * value for value in image))


def _nearest_distances(
    ion_coordinates: Sequence[Sequence[float]],
    target_coordinates: Sequence[Sequence[float]],
    cell: object,
) -> tuple[float, ...]:
    """Return exact nearest-target distances for every ion.

    Orthogonal cells have a unique component-wise nearest lattice image in
    their (possibly rotated) lattice basis, so NumPy can evaluate the complete
    ion-by-target matrix without changing the geometry.  Skewed triclinic
    cells retain the exact finite lattice enumeration in
    :func:`minimum_image_displacement`; fractional-coordinate rounding alone
    is not exact for that general case.
    """

    if cell is None:
        raise IonAtmosphereError("ion-atmosphere analysis requires a periodic cell")
    ions = np.asarray(ion_coordinates, dtype=np.float64)
    targets = np.asarray(target_coordinates, dtype=np.float64)
    cell_matrix = np.asarray(cell, dtype=np.float64)
    if (
        ions.ndim != 2 or ions.shape[1:] != (3,) or not ions.shape[0]
        or targets.ndim != 2 or targets.shape[1:] != (3,) or not targets.shape[0]
        or cell_matrix.shape != (3, 3)
        or not np.all(np.isfinite(ions))
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(cell_matrix))
    ):
        raise IonAtmosphereError(
            "ion and target coordinates and periodic cell must be finite three-dimensional arrays"
        )

    gram = cell_matrix @ cell_matrix.T
    diagonal = np.diag(gram)
    scale = float(np.max(diagonal))
    if scale > 0.0 and np.all(np.abs(gram - np.diag(diagonal)) <= 1.0e-12 * scale):
        try:
            inverse_cell = np.linalg.inv(cell_matrix)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by cell validation
            raise IonAtmosphereError("periodic cell is singular") from exc
        displacements = targets[np.newaxis, :, :] - ions[:, np.newaxis, :]
        fractional = displacements @ inverse_cell
        fractional -= np.floor(fractional + 0.5)
        images = fractional @ cell_matrix
        squared = (
            images[:, :, 0] * images[:, :, 0]
            + images[:, :, 1] * images[:, :, 1]
            + images[:, :, 2] * images[:, :, 2]
        )
        nearest_squared = np.min(squared, axis=1)
        return tuple(math.sqrt(float(value)) for value in nearest_squared)

    return tuple(
        min(_distance(ion, target, cell) for target in target_coordinates)
        for ion in ion_coordinates
    )


def _summary(values: Sequence[float]) -> Dict[str, object]:
    if not values:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1 else None
    )
    return {
        "count": len(values), "mean": mean,
        "sample_standard_deviation": math.sqrt(variance) if variance is not None else None,
        "minimum": min(values), "maximum": max(values),
    }


def _ion_atmosphere_project_serial(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(str(context["system_manifest_path"]))
    system = load_json(system_path)
    coordinate_unit = str(project["coordinate_unit"])
    frame_plan, frame_report = plan_frame_selection(
        system, system_path, coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        error_type=IonAtmosphereError,
    )
    if int(frame_report["selected_frame_count"]) > int(settings["maximum_frames"]):
        raise IonAtmosphereError("maximum_frames gate exceeded by frame selection")

    cutoffs = [float(value) for value in settings["shell_cutoffs_angstrom"]]  # type: ignore[union-attr]
    maximum_cutoff = max(cutoffs)
    aggregate_counts: Dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    per_ion_inner_hits: Counter[tuple[str, str, str, int]] = Counter()
    per_ion_frames: Counter[tuple[str, str, str, int]] = Counter()
    frame_records: list[Dict[str, object]] = []
    segment_reports: list[Dict[str, object]] = []

    for raw_system in system["systems"]:
        assert isinstance(raw_system, dict)
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            ion_groups = {
                str(row["species"]): [int(index) for index in row["atom_indices"]]
                for row in settings["ion_groups"]  # type: ignore[union-attr]
            }
            charge_classes = {
                str(row["species"]): str(row["charge_class"])
                for row in settings["ion_groups"]  # type: ignore[union-attr]
            }
            target_groups = {
                str(row["target_id"]): [int(index) for index in row["atom_indices"]]
                for row in settings["target_groups"]  # type: ignore[union-attr]
            }
            all_indices = sorted({index for values in [*ion_groups.values(), *target_groups.values()] for index in values})
            if not all_indices or max(all_indices) >= len(atoms):
                raise IonAtmosphereError("ion-atmosphere atom indices exceed topology atom count")
            for species, indices in ion_groups.items():
                observed = {atoms[index].element.upper() for index in indices}
                if observed != {species.upper()}:
                    raise IonAtmosphereError(
                        f"declared species {species} does not match topology elements {sorted(observed)}"
                    )
            for segment in replica["segments"]:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                selected_indices = frame_plan[(system_id, replica_id, segment_id)]
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                evaluated = 0
                for raw_frame in iter_coordinate_frames(
                    trajectory_path, coordinate_unit, selected_indices,
                ):
                    selected = frame_selected(raw_frame.frame_index, selected_indices, int(settings["frame_stride"]))
                    if not selected:
                        continue
                    frame = raw_frame
                    evaluated += 1
                    record: Dict[str, object] = {
                        "system_id": system_id, "replica_id": replica_id,
                        "segment_id": segment_id,
                        "source_frame_index": frame.frame_index,
                        "frame_index": frame.frame_index,
                        "species": {},
                    }
                    species_record = record["species"]
                    assert isinstance(species_record, dict)
                    coordinate_array = np.asarray(
                        frame.coordinates_angstrom, dtype=np.float64,
                    )
                    for species, ion_indices in sorted(ion_groups.items()):
                        target_record: Dict[str, object] = {}
                        species_record[species] = {
                            "charge_class": charge_classes[species],
                            "targets": target_record,
                        }
                        nearest_by_target_atoms: Dict[tuple[int, ...], tuple[float, ...]] = {}
                        for target_id, target_indices in sorted(target_groups.items()):
                            counts = {cutoff: 0 for cutoff in cutoffs}
                            target_key = tuple(sorted(target_indices))
                            nearest_values = nearest_by_target_atoms.get(target_key)
                            if nearest_values is None:
                                nearest_values = _nearest_distances(
                                    coordinate_array[ion_indices],
                                    coordinate_array[target_indices],
                                    frame.cell_vectors_angstrom,
                                )
                                nearest_by_target_atoms[target_key] = nearest_values
                            for ion_index, nearest in zip(ion_indices, nearest_values):
                                identity = (system_id, replica_id, species, ion_index)
                                if target_id == "all_solute":
                                    per_ion_frames[identity] += 1
                                    if nearest <= cutoffs[0]:
                                        per_ion_inner_hits[identity] += 1
                                for cutoff in cutoffs:
                                    if nearest <= cutoff:
                                        counts[cutoff] += 1
                            for cutoff, count in counts.items():
                                aggregate_counts[(species, charge_classes[species], target_id, cutoff)].append(float(count))
                            target_record[target_id] = {
                                "ion_count_within_shell": {str(cutoff): counts[cutoff] for cutoff in cutoffs},
                                "nearest_distance_angstrom": min(nearest_values),
                                "ion_count": len(ion_indices),
                                "maximum_evaluated_cutoff_angstrom": maximum_cutoff,
                            }
                    frame_records.append(record)
                segment_reports.append({
                    "system_id": system_id, "replica_id": replica_id,
                    "segment_id": segment_id, "evaluated_frame_count": evaluated,
                })

    summaries = [{
        "species": key[0], "charge_class": key[1], "target_id": key[2],
        "cutoff_angstrom": key[3], "ion_count_summary": _summary(values),
    } for key, values in sorted(aggregate_counts.items())]
    persistence = []
    for identity in sorted(per_ion_frames):
        frames = per_ion_frames[identity]
        occupancy = per_ion_inner_hits[identity] / frames
        persistence.append({
            "system_id": identity[0], "replica_id": identity[1],
            "species": identity[2], "ion_atom_index": identity[3],
            "inner_shell_cutoff_angstrom": cutoffs[0],
            "evaluated_frame_count": frames, "inner_shell_occupancy": occupancy,
            "descriptive_class": (
                "persistent" if occupancy >= 0.50 else
                "intermittent" if occupancy >= 0.05 else "mobile_or_transient"
            ),
        })
    issues = [issue for issue in context.get("issues", []) if isinstance(issue, dict)]
    if int(frame_report["selected_frame_count"]) < int(frame_report["source_frame_count"]):
        issues.append({
            "severity": "warning", "code": "FRAME_SUBSAMPLING",
            "message": (
                f"ion atmosphere evaluated {frame_report['selected_frame_count']} of "
                f"{frame_report['source_frame_count']} source frames under {frame_report['mode']}"
            ),
        })
    return {
        "module_id": "ion_atmosphere", "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings, "frame_selection": frame_report,
        "segment_reports": segment_reports, "frame_records": frame_records,
        "species_target_shell_summaries": summaries,
        "per_ion_inner_shell_persistence": persistence,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Shell counts and persistence classes are geometric descriptions, not proof of biological binding, affinity, oxidation state, or mechanism.",
            "The innermost declared shell defines the descriptive persistence classification and requires chemical sensitivity analysis.",
            "Equivalent-member and cross-system pooling require an explicit independent-unit model; this report preserves system and replica identities.",
            "Exact triclinic minimum-image distances are used; pair distances are invariant to independent integer-lattice translations, so local atmosphere distances read selected frames directly and do not require whole-solvent or whole-solute reconstruction.",
        ],
    }


def _combine_shell_summary_rows(
    reports: Sequence[Mapping[str, object]],
) -> list[Dict[str, object]]:
    states: Dict[tuple[str, str, str, float], Dict[str, float]] = {}
    for report in reports:
        for row in report.get("species_target_shell_summaries", []):
            if not isinstance(row, dict) or not isinstance(row.get("ion_count_summary"), dict):
                raise IonAtmosphereError("ion-atmosphere shell summary is malformed")
            summary = row["ion_count_summary"]
            key = (
                str(row["species"]), str(row["charge_class"]),
                str(row["target_id"]), float(row["cutoff_angstrom"]),
            )
            count = int(summary["count"])
            mean = float(summary["mean"])
            standard_deviation = summary.get("sample_standard_deviation")
            m2 = (
                0.0 if count <= 1 or standard_deviation is None
                else (count - 1) * float(standard_deviation) ** 2
            )
            current = states.get(key)
            if current is None:
                states[key] = {
                    "count": float(count), "mean": mean, "m2": m2,
                    "minimum": float(summary["minimum"]),
                    "maximum": float(summary["maximum"]),
                }
                continue
            left_count = int(current["count"])
            combined = left_count + count
            delta = mean - current["mean"]
            current["mean"] += delta * count / combined
            current["m2"] += m2 + delta * delta * left_count * count / combined
            current["count"] = float(combined)
            current["minimum"] = min(current["minimum"], float(summary["minimum"]))
            current["maximum"] = max(current["maximum"], float(summary["maximum"]))
    result = []
    for key, state in sorted(states.items()):
        count = int(state["count"])
        result.append({
            "species": key[0], "charge_class": key[1], "target_id": key[2],
            "cutoff_angstrom": key[3],
            "ion_count_summary": {
                "count": count, "mean": state["mean"],
                "sample_standard_deviation": (
                    math.sqrt(state["m2"] / (count - 1)) if count > 1 else None
                ),
                "minimum": state["minimum"], "maximum": state["maximum"],
            },
        })
    return result


def _reduce_ion_atmosphere_reports(
    partials: Sequence[ReplicaPartial[Dict[str, object]]],
    source_context: Dict[str, object],
) -> Dict[str, object]:
    reports = [partial.value for partial in partials]
    first = dict(reports[0])
    for report in reports[1:]:
        for key in ("module_id", "settings"):
            if report.get(key) != first.get(key):
                raise IonAtmosphereError(
                    f"replica ion-atmosphere reports disagree on {key}"
                )
    first["frame_selection"] = merge_frame_selection_reports([
        report["frame_selection"] for report in reports
        if isinstance(report.get("frame_selection"), dict)
    ])
    if int(first["frame_selection"]["selected_frame_count"]) > int(  # type: ignore[index]
        first["settings"]["maximum_frames"]  # type: ignore[index]
    ):
        raise IonAtmosphereError(
            "parallel ion-atmosphere frame count exceeds maximum_frames"
        )
    for key in ("segment_reports", "frame_records", "per_ion_inner_shell_persistence"):
        first[key] = [row for report in reports for row in report.get(key, [])]
    first["species_target_shell_summaries"] = _combine_shell_summary_rows(reports)
    issues = unique_issues(reports)
    first["issues"] = issues
    first["error_count"] = sum(issue.get("severity") == "error" for issue in issues)
    first["warning_count"] = sum(
        issue.get("severity") == "warning" for issue in issues
    )
    restore_source_provenance(first, source_context)
    return first


def ion_atmosphere_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Analyze replicas independently and pool exact shell-count moments."""

    project = load_json(Path(project_path).expanduser().resolve(strict=False))
    settings = _settings(project)
    selection = settings.get("frame_selection")
    if isinstance(selection, dict) and selection.get("mode") == "auto_resource_budget_v1":
        return _ion_atmosphere_project_serial(project_path, hash_content=hash_content)
    return execute_replica_final_module(
        project_path,
        runner_id="ion_atmosphere",
        hash_content=hash_content,
        reducer=_reduce_ion_atmosphere_reports,
    )


def ion_atmosphere_project_safe(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
    try:
        return ion_atmosphere_project(project_path, hash_content=hash_content)
    except (
        AtomMappingError, CoordinateReadError, ManifestValidationError,
        PeriodicReconstructionError, IonAtmosphereError,
        TrajectoryContractError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "ion_atmosphere", "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "ION_ATMOSPHERE_INVALID", "message": message}
                for message in messages
            ],
        }
