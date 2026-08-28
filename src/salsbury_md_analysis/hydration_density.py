"""Aligned three-dimensional water/ion density and geometric channel candidates.

This experimental module supplements RDFs and discrete interaction analyses.  It
retains spatially resolved occupancy on one reference-aligned grid and exposes
exact frame identities for downstream interaction fingerprints.  Connected
high-occupancy components are geometric channel candidates only; no diffusion,
flux, free energy, or transport mechanism is inferred.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .chemical_identity import ION_RESIDUES, WATER_RESIDUES
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .geometry import GeometryError, apply_transform, best_fit_transform
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .periodic import (
    PeriodicFrameProcessor, PeriodicReconstructionError,
    minimum_image_displacement,
)
from .selections import build_common_correspondences, select_atoms
from .trajectory_contracts import (
    TrajectoryContractError, frame_axis_value, normalize_segment_axis,
)
from .validation import positive_integer


class HydrationDensityError(ValueError):
    """Raised when aligned solvent density cannot be evaluated safely."""


Voxel = Tuple[int, int, int]
FrameKey = Tuple[str, str, str, int]


def _flat_voxel(voxel: Voxel, shape: Tuple[int, int, int]) -> int:
    """Encode one grid voxel compactly for retained per-frame occupancy."""
    return (voxel[0] * shape[1] + voxel[1]) * shape[2] + voxel[2]


def _finite(
    value: object, name: str, *, minimum: float | None = None,
    maximum: float | None = None, positive: bool = False,
) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise HydrationDensityError(f"{name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise HydrationDensityError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise HydrationDensityError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise HydrationDensityError(f"{name} must be at most {maximum}")
    return result


def _string_names(value: object, name: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise HydrationDensityError(f"{name} must be an array of nonempty strings")
    values = tuple(sorted({item.strip().upper() for item in value}))
    if len(values) != len(value):
        raise HydrationDensityError(f"{name} values must be unique")
    return values


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("hydration_density_channels")
        if isinstance(definitions, dict) else None
    )
    required = {
        "alignment_selection", "reference_extent_selection",
        "minimum_reference_coverage", "frame_stride", "frame_selection",
        "maximum_frames", "include_recognized_waters",
        "include_supported_ions", "additional_residue_names",
        "grid_spacing_angstrom", "grid_padding_angstrom",
        "minimum_voxel_frame_occupancy", "minimum_component_voxels",
        "minimum_channel_depth_angstrom", "maximum_grid_voxels",
        "maximum_particle_observations", "maximum_sparse_frame_voxels",
        "minimum_evaluated_frames_per_system",
    }
    if not isinstance(raw, dict):
        raise HydrationDensityError(
            "definitions.hydration_density_channels must be an object"
        )
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing or unknown:
        raise HydrationDensityError(
            "hydration-density settings mismatch; missing=" + ",".join(missing)
            + "; unknown=" + ",".join(unknown)
        )
    for name in ("alignment_selection", "reference_extent_selection"):
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise HydrationDensityError(f"{name} must be a nonempty selection name")
    for name in ("include_recognized_waters", "include_supported_ions"):
        if not isinstance(raw[name], bool):
            raise HydrationDensityError(f"{name} must be boolean")
    result = dict(raw)
    result["minimum_reference_coverage"] = _finite(
        raw["minimum_reference_coverage"], "minimum_reference_coverage",
        minimum=0.0, maximum=1.0,
    )
    result["frame_stride"] = positive_integer(
        raw["frame_stride"], "frame_stride", error_type=HydrationDensityError
    )
    result["frame_selection"] = normalize_frame_selection(
        raw["frame_selection"], int(result["frame_stride"]),
        error_type=HydrationDensityError,
    )
    for name in (
        "maximum_frames", "minimum_component_voxels", "maximum_grid_voxels",
        "maximum_particle_observations", "maximum_sparse_frame_voxels",
        "minimum_evaluated_frames_per_system",
    ):
        result[name] = positive_integer(
            raw[name], name, error_type=HydrationDensityError
        )
    result["additional_residue_names"] = _string_names(
        raw["additional_residue_names"], "additional_residue_names"
    )
    for name in ("grid_spacing_angstrom", "grid_padding_angstrom"):
        result[name] = _finite(raw[name], name, positive=True)
    result["minimum_voxel_frame_occupancy"] = _finite(
        raw["minimum_voxel_frame_occupancy"],
        "minimum_voxel_frame_occupancy", minimum=0.0, maximum=1.0,
    )
    result["minimum_channel_depth_angstrom"] = _finite(
        raw["minimum_channel_depth_angstrom"],
        "minimum_channel_depth_angstrom", minimum=0.0,
    )
    return result


def _coordinates_at(
    coordinates: Sequence[Sequence[float]], indices: Sequence[int]
) -> Tuple[Tuple[float, float, float], ...]:
    try:
        return tuple(tuple(float(value) for value in coordinates[index]) for index in indices)  # type: ignore[return-value]
    except IndexError as exc:
        raise HydrationDensityError("atom index exceeds coordinate atom count") from exc


def _particle_atoms(
    atoms: Sequence[AtomRecord], settings: Mapping[str, object]
) -> Dict[str, Tuple[int, ...]]:
    additional = set(settings["additional_residue_names"])
    result: Dict[str, List[int]] = defaultdict(list)
    for atom in atoms:
        residue = atom.residue_name.strip().upper()
        element = atom.element.strip().upper()
        if (
            bool(settings["include_recognized_waters"])
            and residue in WATER_RESIDUES and element == "O"
        ):
            result["water"].append(atom.atom_index)
        elif bool(settings["include_supported_ions"]) and residue in ION_RESIDUES:
            result[f"ion:{residue}"].append(atom.atom_index)
        elif residue in additional and element != "H":
            result[f"residue:{residue}"].append(atom.atom_index)
    return {key: tuple(values) for key, values in sorted(result.items()) if values}


def _grid(
    reference_coordinates: Sequence[Sequence[float]], spacing: float,
    padding: float, maximum_voxels: int,
) -> Dict[str, object]:
    values = np.asarray(reference_coordinates, dtype=float)
    origin = np.floor((values.min(axis=0) - padding) / spacing) * spacing
    upper = np.ceil((values.max(axis=0) + padding) / spacing) * spacing
    shape = np.maximum(1, np.ceil((upper - origin) / spacing).astype(int) + 1)
    voxel_count = int(np.prod(shape))
    if voxel_count > maximum_voxels:
        raise HydrationDensityError(
            f"reference grid contains {voxel_count} voxels; maximum_grid_voxels "
            f"is {maximum_voxels}"
        )
    return {
        "origin_angstrom": origin.tolist(), "spacing_angstrom": spacing,
        "shape": shape.tolist(), "voxel_count": voxel_count,
        "axis_order": "x_y_z", "voxel_index_rule": "floor_from_origin_v1",
    }


def _voxel(
    point: Sequence[float], origin: np.ndarray, spacing: float,
    shape: Tuple[int, int, int],
) -> Voxel | None:
    index = np.floor((np.asarray(point, dtype=float) - origin) / spacing).astype(int)
    candidate = tuple(int(value) for value in index)
    if all(0 <= candidate[axis] < shape[axis] for axis in range(3)):
        return candidate  # type: ignore[return-value]
    return None


def _components(active: Iterable[Voxel]) -> List[List[Voxel]]:
    unseen = set(active)
    result: List[List[Voxel]] = []
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        queue = deque((seed,))
        component = []
        while queue:
            voxel = queue.popleft()
            component.append(voxel)
            for offset in offsets:
                neighbor = tuple(voxel[axis] + offset[axis] for axis in range(3))
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)  # type: ignore[arg-type]
        result.append(sorted(component))
    return sorted(result, key=lambda values: (-len(values), values[0]))


def _component_record(
    *, system_id: str, species: str, component_number: int,
    voxels: Sequence[Voxel], counts: Mapping[Voxel, int], frame_count: int,
    origin: np.ndarray, spacing: float, shape: Tuple[int, int, int],
    minimum_channel_depth: float,
) -> Dict[str, object]:
    centers = np.asarray([
        origin + (np.asarray(voxel, dtype=float) + 0.5) * spacing
        for voxel in voxels
    ])
    occupancies = [counts[voxel] / frame_count for voxel in voxels]
    touches_boundary = any(
        any(voxel[axis] in {0, shape[axis] - 1} for axis in range(3))
        for voxel in voxels
    )
    depths = [
        min(
            min(
                voxel[axis] * spacing,
                (shape[axis] - 1 - voxel[axis]) * spacing,
            )
            for axis in range(3)
        )
        for voxel in voxels
    ]
    maximum_depth = max(depths) if depths else 0.0
    feature_id = f"{system_id}|{species}|density-component-{component_number}"
    return {
        "feature_id": feature_id, "system_id": system_id, "species": species,
        "component_index": component_number, "voxel_count": len(voxels),
        "volume_angstrom3": len(voxels) * spacing ** 3,
        "mean_voxel_frame_occupancy": sum(occupancies) / len(occupancies),
        "maximum_voxel_frame_occupancy": max(occupancies),
        "centroid_angstrom": centers.mean(axis=0).tolist(),
        "bounding_box_angstrom": {
            "minimum": centers.min(axis=0).tolist(),
            "maximum": centers.max(axis=0).tolist(),
        },
        "touches_grid_boundary": touches_boundary,
        "maximum_interior_depth_angstrom": maximum_depth,
        "geometric_channel_candidate": (
            touches_boundary and maximum_depth >= minimum_channel_depth
        ),
        "voxel_indices": [list(voxel) for voxel in voxels],
    }


def hydration_density_channels_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    contract = context["contract"]
    assert isinstance(contract, dict)
    selections = contract["selections"]
    units = contract["units"]
    assert isinstance(selections, dict) and isinstance(units, dict)
    coordinate_unit = str(units["coordinates"])
    reference_path = resolve_manifest_path(str(project["reference_structure"]), source)
    _, reference_atoms = read_topology_atoms(reference_path)
    try:
        raw_reference = next(iter_coordinate_frames(reference_path, coordinate_unit))
    except StopIteration as exc:
        raise HydrationDensityError("reference structure contains no coordinates") from exc
    reference_frame = PeriodicFrameProcessor.from_reference(
        project, source, len(reference_atoms)
    ).process(raw_reference, str(reference_path))
    extent_definition = selections.get(str(settings["reference_extent_selection"]))
    alignment_definition = selections.get(str(settings["alignment_selection"]))
    if not isinstance(extent_definition, dict) or not isinstance(alignment_definition, dict):
        raise HydrationDensityError("alignment or reference-extent selection is undefined")
    extent_atoms = select_atoms(
        reference_atoms, extent_definition, str(settings["reference_extent_selection"])
    )
    grid = _grid(
        _coordinates_at(
            reference_frame.coordinates_angstrom,
            [atom.atom_index for atom in extent_atoms],
        ),
        float(settings["grid_spacing_angstrom"]),
        float(settings["grid_padding_angstrom"]),
        int(settings["maximum_grid_voxels"]),
    )
    origin = np.asarray(grid["origin_angstrom"], dtype=float)
    shape = tuple(int(value) for value in grid["shape"])
    spacing = float(grid["spacing_angstrom"])

    system_path = Path(str(context["system_manifest_path"]))
    manifest = load_json(system_path)
    frame_plan, frame_report = plan_frame_selection(
        manifest, system_path, coordinate_unit, settings["frame_selection"],
        frame_stride=int(settings["frame_stride"]), error_type=HydrationDensityError,
    )
    if int(frame_report["selected_frame_count"]) > int(settings["maximum_frames"]):
        raise HydrationDensityError("maximum_frames gate exceeded")

    topology_rows = []
    for system in manifest["systems"]:
        for replica in system["replicas"]:
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            topology_rows.append((str(system["system_id"]), str(replica["replica_id"]), atoms))
    mappings = build_common_correspondences(
        reference_atoms, [row[2] for row in topology_rows], alignment_definition,
        str(settings["alignment_selection"]), str(project["common_atom_policy"]),
        float(settings["minimum_reference_coverage"]),
    )
    mapping_by_key = {
        (system_id, replica_id): mapping
        for (system_id, replica_id, _), mapping in zip(topology_rows, mappings)
    }
    atoms_by_key = {(row[0], row[1]): row[2] for row in topology_rows}
    reference_alignment = _coordinates_at(
        reference_frame.coordinates_angstrom, mappings[0].reference_indices
    )

    voxel_frames: Dict[Tuple[str, str], Counter[Voxel]] = defaultdict(Counter)
    particle_counts: Dict[Tuple[str, str], Counter[Voxel]] = defaultdict(Counter)
    # Per-frame occupancy is needed after the aggregate density components are
    # known.  Keeping Python sets of three-integer tuples for every frame used
    # tens of bytes per occupied voxel and exhausted practical campaign limits.
    # Sorted uint32 flat indices preserve exact occupancy while using four bytes
    # per retained voxel.  maximum_grid_voxels is already well below uint32.
    frame_voxels: Dict[FrameKey, Dict[str, array]] = {}
    frame_metadata: Dict[FrameKey, Dict[str, object]] = {}
    system_frames: Counter[str] = Counter()
    observed_species: Dict[str, set[str]] = defaultdict(set)
    topology_inventory = []
    total_particle_observations = 0
    total_sparse_voxels = 0
    outside_grid = Counter()

    for system in manifest["systems"]:
        system_id = str(system["system_id"])
        for replica in system["replicas"]:
            replica_id = str(replica["replica_id"])
            atoms = atoms_by_key[(system_id, replica_id)]
            mapping = mapping_by_key[(system_id, replica_id)]
            particles = _particle_atoms(atoms, settings)
            if not particles:
                continue
            solute_atoms = select_atoms(
                atoms, extent_definition, str(settings["reference_extent_selection"])
            )
            solute_indices = tuple(atom.atom_index for atom in solute_atoms)
            all_particle_indices = tuple(
                sorted({index for values in particles.values() for index in values})
            )
            required_indices = tuple(sorted(
                set(mapping.target_indices) | set(solute_indices) | set(all_particle_indices)
            ))
            topology_inventory.append({
                "system_id": system_id, "replica_id": replica_id,
                "particle_atom_counts": {key: len(value) for key, value in particles.items()},
                "solute_atom_count": len(solute_indices),
                "alignment_mapping": mapping.as_dict(),
            })
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(atoms)
            )
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                selected_indices = frame_plan[(system_id, replica_id, segment_id)]
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                axis = normalize_segment_axis(segment, project.get("time_unit"))
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                for raw_frame in iter_coordinate_frames(
                    trajectory_path, coordinate_unit,
                    reader_frame_indices(selected_indices, processor.policy),
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
                        required_indices,
                    )
                    if not selected:
                        continue
                    transform = best_fit_transform(
                        _coordinates_at(frame.coordinates_angstrom, mapping.target_indices),
                        reference_alignment,
                    )
                    solute_coordinates = _coordinates_at(
                        frame.coordinates_angstrom, solute_indices
                    )
                    anchor = tuple(
                        sum(point[axis_index] for point in solute_coordinates) / len(solute_coordinates)
                        for axis_index in range(3)
                    )
                    key = (system_id, replica_id, segment_id, int(frame.frame_index))
                    meta = {
                        "system_id": system_id, "replica_id": replica_id,
                        "segment_id": segment_id,
                        "source_frame_index": int(frame.frame_index),
                        "axis_kind": axis["kind"],
                        "axis_unit": project.get("time_unit") if axis["kind"] == "physical_time" else "sample",
                        "axis_value": frame_axis_value(axis, frame.frame_index),
                    }
                    frame_metadata[key] = meta
                    per_species: Dict[str, array] = {}
                    for species, indices in particles.items():
                        imaged = []
                        for index in indices:
                            point = frame.coordinates_angstrom[index]
                            if frame.cell_vectors_angstrom is not None:
                                delta = minimum_image_displacement(
                                    tuple(point[axis_index] - anchor[axis_index] for axis_index in range(3)),
                                    frame.cell_vectors_angstrom,
                                )
                                point = tuple(anchor[axis_index] + delta[axis_index] for axis_index in range(3))
                            imaged.append(point)
                        aligned = apply_transform(imaged, transform)
                        occupied: set[Voxel] = set()
                        for point in aligned:
                            total_particle_observations += 1
                            if total_particle_observations > int(settings["maximum_particle_observations"]):
                                raise HydrationDensityError("maximum_particle_observations gate exceeded")
                            voxel = _voxel(point, origin, spacing, shape)
                            if voxel is None:
                                outside_grid[(system_id, species)] += 1
                                continue
                            occupied.add(voxel)
                            particle_counts[(system_id, species)][voxel] += 1
                        voxel_frames[(system_id, species)].update(occupied)
                        observed_species[system_id].add(species)
                        per_species[species] = array(
                            "I", sorted(_flat_voxel(voxel, shape) for voxel in occupied)
                        )
                        total_sparse_voxels += len(occupied)
                        if total_sparse_voxels > int(settings["maximum_sparse_frame_voxels"]):
                            raise HydrationDensityError(
                                "maximum_sparse_frame_voxels gate exceeded: "
                                f"observed more than {settings['maximum_sparse_frame_voxels']} "
                                "distinct species-frame voxels"
                            )
                    frame_voxels[key] = per_species
                    system_frames[system_id] += 1

    if not system_frames:
        return {
            "module_id": "hydration_density_channels",
            "technical_status": "complete", "scientific_status": "not evaluated",
            "availability_status": "not_available",
            "availability_reason": "no_configured_water_or_ion_atoms_detected",
            "project_manifest_path": str(source), "settings": settings,
            "grid": grid, "density_components": [], "frame_feature_records": [],
            "density_projections_xy": [], "pairwise_system_density_differences": [],
            "finding_candidates": [], "issues": [{
                "severity": "warning", "code": "HYDRATION_DENSITY_NOT_AVAILABLE",
                "message": "No configured water, supported ion, or declared residue atoms were detected.",
            }], "error_count": 0, "warning_count": 1,
        }
    for system_id, count in system_frames.items():
        if count < int(settings["minimum_evaluated_frames_per_system"]):
            raise HydrationDensityError(
                f"system {system_id} produced {count} selected frames; minimum is "
                f"{settings['minimum_evaluated_frames_per_system']}"
            )

    component_rows = []
    component_voxels: Dict[str, set[int]] = {}
    projection_rows = []
    occupancy_arrays: Dict[Tuple[str, str], np.ndarray] = {}
    for system_id in sorted(system_frames):
        for species in sorted(observed_species[system_id]):
            counts = voxel_frames[(system_id, species)]
            threshold = float(settings["minimum_voxel_frame_occupancy"])
            active = [
                voxel for voxel, count in counts.items()
                if count / system_frames[system_id] >= threshold
            ]
            retained_components = [
                values for values in _components(active)
                if len(values) >= int(settings["minimum_component_voxels"])
            ]
            for number, voxels in enumerate(retained_components, start=1):
                row = _component_record(
                    system_id=system_id, species=species,
                    component_number=number, voxels=voxels, counts=counts,
                    frame_count=system_frames[system_id], origin=origin,
                    spacing=spacing, shape=shape,
                    minimum_channel_depth=float(settings["minimum_channel_depth_angstrom"]),
                )
                component_rows.append(row)
                component_voxels[str(row["feature_id"])] = {
                    _flat_voxel(voxel, shape) for voxel in voxels
                }
            occupancy_array = np.zeros(shape, dtype=float)
            for voxel, count in counts.items():
                occupancy_array[voxel] = count / system_frames[system_id]
            occupancy_arrays[(system_id, species)] = occupancy_array
            projection_rows.append({
                "system_id": system_id, "species": species,
                "projection_axis": "z", "normalization": "mean frame occupancy summed over z",
                "matrix": occupancy_array.sum(axis=2).tolist(),
                "maximum_projected_occupancy": float(
                    occupancy_array.sum(axis=2).max()
                ),
            })

    frame_records = []
    for key in sorted(frame_voxels):
        active_ids = []
        for feature_id, voxels in component_voxels.items():
            species = feature_id.split("|", 2)[1]
            occupied = frame_voxels[key].get(species)
            if (
                feature_id.startswith(key[0] + "|")
                and occupied is not None
                and any(flat in voxels for flat in occupied)
            ):
                active_ids.append(feature_id)
        frame_records.append({**frame_metadata[key], "active_feature_ids": sorted(active_ids)})

    pairwise = []
    systems = sorted(system_frames)
    for left_index, left in enumerate(systems):
        for right in systems[left_index + 1:]:
            for species in sorted(observed_species[left].intersection(observed_species[right])):
                difference = occupancy_arrays[(left, species)] - occupancy_arrays[(right, species)]
                maximum_index = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
                pairwise.append({
                    "system_i": left, "system_j": right, "species": species,
                    "mean_absolute_voxel_occupancy_difference": float(np.mean(np.abs(difference))),
                    "maximum_absolute_voxel_occupancy_difference": float(np.max(np.abs(difference))),
                    "maximum_difference_voxel_index": list(maximum_index),
                    "signed_difference_at_maximum": float(difference[maximum_index]),
                })
    findings = []
    for system_id in systems:
        rows = [row for row in component_rows if row["system_id"] == system_id]
        if rows:
            strongest = max(rows, key=lambda row: float(row["maximum_voxel_frame_occupancy"]))
            findings.append({
                "category": "other_physical", "evidence_level": "descriptive",
                "statement": (
                    f"Strongest retained aligned solvent-density component in {system_id} "
                    f"is {strongest['species']} component {strongest['component_index']} "
                    f"with maximum voxel occupancy {float(strongest['maximum_voxel_frame_occupancy']):.1%}."
                ),
                "effect_value": strongest["maximum_voxel_frame_occupancy"],
                "system_ids": [system_id],
                "comparison_family": "hydration_density_channels:within_system_component",
            })
    for row in pairwise:
        findings.append({
            "category": "other_physical", "evidence_level": "descriptive",
            "statement": (
                f"Largest aligned {row['species']} voxel-occupancy difference between "
                f"{row['system_i']} and {row['system_j']} is "
                f"{float(row['maximum_absolute_voxel_occupancy_difference']):.1%}."
            ),
            "effect_value": row["maximum_absolute_voxel_occupancy_difference"],
            "system_ids": [row["system_i"], row["system_j"]],
            "comparison_family": "hydration_density_channels:pairwise_density_difference",
        })
    issues = []
    if int(frame_report["selected_frame_count"]) < int(frame_report["source_frame_count"]):
        issues.append({
            "severity": "warning", "code": "HYDRATION_DENSITY_SUBSAMPLED",
            "message": "Density occupancy was estimated from a deterministic subset of source frames.",
        })
    return {
        "module_id": "hydration_density_channels",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "availability_status": "available", "availability_reason": None,
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "grid": grid, "frame_selection": frame_report,
        "topology_inventory": topology_inventory,
        "system_frame_counts": dict(sorted(system_frames.items())),
        "outside_grid_particle_counts": [
            {"system_id": key[0], "species": key[1], "count": value}
            for key, value in sorted(outside_grid.items())
        ],
        "density_components": component_rows,
        "frame_feature_records": frame_records,
        "density_projections_xy": projection_rows,
        "pairwise_system_density_differences": pairwise,
        "finding_candidates": findings,
        "observation_accounting": {
            "selected_physical_frame_count": sum(system_frames.values()),
            "particle_observation_count": total_particle_observations,
            "sparse_frame_voxel_count": total_sparse_voxels,
            "sparse_frame_voxel_storage": "sorted_uint32_flat_indices_v1",
        },
        "error_count": 0,
        "warning_count": sum(issue["severity"] == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Aligned voxel occupancy supplements RDFs, bridges, and interaction fingerprints; it does not replace their radial or chemical definitions.",
            "A connected boundary-reaching high-occupancy component is a geometric channel candidate, not evidence of diffusion, flux, permeability, free energy, or mechanism.",
            "Grid spacing, padding, alignment, occupancy threshold, species identity, and frame selection require sensitivity analysis.",
            "Frame occupancies are descriptive and are not independent-replica uncertainty.",
        ],
    }


def hydration_density_channels_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return hydration_density_channels_project(project_path, hash_content=hash_content)
    except (
        HydrationDensityError, ManifestValidationError, AtomMappingError,
        CoordinateReadError, PeriodicReconstructionError, GeometryError,
        TrajectoryContractError, OSError, KeyError, TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "hydration_density_channels",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "HYDRATION_DENSITY_INVALID",
                "message": message,
            } for message in messages],
        }
