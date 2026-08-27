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

from .atom_mapping import AtomMappingError, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .periodic import (
    PeriodicFrameProcessor, PeriodicReconstructionError,
    minimum_image_displacement,
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


def ion_atmosphere_project(project_path: Path, hash_content: bool = False) -> Dict[str, object]:
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
            processor = PeriodicFrameProcessor.from_replica(project, replica, system_path, len(atoms))
            for segment in replica["segments"]:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                selected_indices = frame_plan[(system_id, replica_id, segment_id)]
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                evaluated = 0
                for raw_frame in iter_coordinate_frames(
                    trajectory_path, coordinate_unit,
                    reader_frame_indices(selected_indices, processor.policy),
                ):
                    selected = frame_selected(raw_frame.frame_index, selected_indices, int(settings["frame_stride"]))
                    if not selected and processor.policy != "unwrap_continuous":
                        continue
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        all_indices,
                    )
                    if not selected:
                        continue
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
                    for species, ion_indices in sorted(ion_groups.items()):
                        target_record: Dict[str, object] = {}
                        species_record[species] = {
                            "charge_class": charge_classes[species],
                            "targets": target_record,
                        }
                        for target_id, target_indices in sorted(target_groups.items()):
                            counts = {cutoff: 0 for cutoff in cutoffs}
                            nearest_values: list[float] = []
                            for ion_index in ion_indices:
                                nearest = min(
                                    _distance(
                                        frame.coordinates_angstrom[ion_index],
                                        frame.coordinates_angstrom[target_index],
                                        frame.cell_vectors_angstrom,
                                    )
                                    for target_index in target_indices
                                )
                                nearest_values.append(nearest)
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
            "Exact triclinic minimum-image distances are used; local atmosphere distances do not require whole-solvent reconstruction.",
        ],
    }


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
