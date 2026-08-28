"""Reference-aligned ensemble pocket geometry and persistence.

The native backends use the same deterministic geometric grid screen.  The
frequency-map backend accumulates pocket-like voxel occupancy before defining
recurrent spatial regions.  The legacy backend tracks discrete frame pockets
by residue overlap.  Neither backend estimates ligandability or druggability.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .geometry import GeometryError, apply_transform, best_fit_transform
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .moments import sample_summary
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .selections import build_common_correspondences, select_atoms
from .trajectory_contracts import (
    TrajectoryContractError, frame_axis_value, normalize_segment_axis,
)
from .validation import positive_integer


class PocketDynamicsError(ValueError):
    """Raised when ensemble pocket geometry cannot be evaluated safely."""


Voxel = Tuple[int, int, int]
FrameKey = Tuple[str, str, str, int]


def _finite(
    value: object, name: str, *, minimum: float | None = None,
    maximum: float | None = None, positive: bool = False,
) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PocketDynamicsError(f"{name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise PocketDynamicsError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise PocketDynamicsError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise PocketDynamicsError(f"{name} must be at most {maximum}")
    return result


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("ensemble_pocket_dynamics")
        if isinstance(definitions, dict) else None
    )
    required = {
        "backend", "alignment_selection", "solute_selection",
        "minimum_reference_coverage", "frame_stride", "frame_selection",
        "maximum_frames", "grid_spacing_angstrom", "grid_padding_angstrom",
        "minimum_clearance_angstrom", "maximum_surface_distance_angstrom",
        "minimum_seed_clearance_angstrom", "minimum_seed_separation_angstrom",
        "pocket_growth_radius_angstrom",
        "neighborhood_radius_angstrom", "minimum_nearby_atoms",
        "minimum_nearby_residues", "minimum_occupied_directions",
        "maximum_directional_imbalance",
        "minimum_pocket_voxels", "maximum_pockets_per_frame",
        "residue_jaccard_threshold", "maximum_centroid_distance_angstrom",
        "maximum_grid_voxels", "maximum_pocket_instances",
        "maximum_tracking_comparisons", "minimum_evaluated_frames_per_system",
    }
    optional_defaults = {
        "minimum_region_frequency_fraction": 0.05,
        "minimum_region_voxels": 4,
        "maximum_frequency_regions": 128,
        "representative_frames_per_region": 2,
        "maximum_sparse_frame_voxels": 250_000_000,
    }
    if not isinstance(raw, dict):
        raise PocketDynamicsError(
            "definitions.ensemble_pocket_dynamics must be an object"
        )
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required | set(optional_defaults)))
    if missing or unknown:
        raise PocketDynamicsError(
            "pocket-dynamics settings mismatch; missing=" + ",".join(missing)
            + "; unknown=" + ",".join(unknown)
        )
    if raw["backend"] not in {"native_grid_v1", "native_frequency_grid_v2"}:
        raise PocketDynamicsError(
            "backend must be native_grid_v1 or native_frequency_grid_v2"
        )
    for name in ("alignment_selection", "solute_selection"):
        if not isinstance(raw[name], str) or not raw[name].strip():
            raise PocketDynamicsError(f"{name} must be a nonempty selection name")
    result = {**optional_defaults, **raw}
    result["minimum_reference_coverage"] = _finite(
        raw["minimum_reference_coverage"], "minimum_reference_coverage",
        minimum=0.0, maximum=1.0,
    )
    result["frame_stride"] = positive_integer(
        raw["frame_stride"], "frame_stride", error_type=PocketDynamicsError
    )
    result["frame_selection"] = normalize_frame_selection(
        raw["frame_selection"], int(result["frame_stride"]),
        error_type=PocketDynamicsError,
    )
    for name in (
        "maximum_frames", "minimum_nearby_atoms", "minimum_nearby_residues",
        "minimum_occupied_directions",
        "minimum_pocket_voxels", "maximum_pockets_per_frame",
        "maximum_grid_voxels", "maximum_pocket_instances",
        "maximum_tracking_comparisons", "minimum_evaluated_frames_per_system",
        "minimum_region_voxels", "maximum_frequency_regions",
        "representative_frames_per_region", "maximum_sparse_frame_voxels",
    ):
        result[name] = positive_integer(
            result[name], name, error_type=PocketDynamicsError
        )
    for name in (
        "grid_spacing_angstrom", "grid_padding_angstrom",
        "minimum_clearance_angstrom", "maximum_surface_distance_angstrom",
        "minimum_seed_clearance_angstrom", "minimum_seed_separation_angstrom",
        "pocket_growth_radius_angstrom",
        "neighborhood_radius_angstrom", "maximum_centroid_distance_angstrom",
    ):
        result[name] = _finite(raw[name], name, positive=True)
    for name in ("maximum_directional_imbalance", "residue_jaccard_threshold"):
        result[name] = _finite(raw[name], name, minimum=0.0, maximum=1.0)
    result["minimum_region_frequency_fraction"] = _finite(
        result["minimum_region_frequency_fraction"],
        "minimum_region_frequency_fraction", minimum=0.0, maximum=1.0,
    )
    if result["minimum_clearance_angstrom"] >= result["maximum_surface_distance_angstrom"]:
        raise PocketDynamicsError(
            "minimum_clearance_angstrom must be smaller than maximum_surface_distance_angstrom"
        )
    if not (
        result["minimum_clearance_angstrom"]
        <= result["minimum_seed_clearance_angstrom"]
        <= result["maximum_surface_distance_angstrom"]
    ):
        raise PocketDynamicsError(
            "minimum_seed_clearance_angstrom must fall within the clearance interval"
        )
    if result["maximum_surface_distance_angstrom"] > result["neighborhood_radius_angstrom"]:
        raise PocketDynamicsError(
            "maximum_surface_distance_angstrom may not exceed neighborhood_radius_angstrom"
        )
    if int(result["minimum_occupied_directions"]) > 6:
        raise PocketDynamicsError("minimum_occupied_directions may not exceed 6")
    return result


def _coordinates_at(
    coordinates: Sequence[Sequence[float]], indices: Sequence[int]
) -> Tuple[Tuple[float, float, float], ...]:
    try:
        return tuple(tuple(float(value) for value in coordinates[index]) for index in indices)  # type: ignore[return-value]
    except IndexError as exc:
        raise PocketDynamicsError("atom index exceeds coordinate atom count") from exc


def _grid(
    reference: Sequence[Sequence[float]], spacing: float, padding: float,
    maximum_voxels: int,
) -> Tuple[Dict[str, object], np.ndarray]:
    coordinates = np.asarray(reference, dtype=float)
    origin = np.floor((coordinates.min(axis=0) - padding) / spacing) * spacing
    upper = np.ceil((coordinates.max(axis=0) + padding) / spacing) * spacing
    shape = np.maximum(1, np.ceil((upper - origin) / spacing).astype(int) + 1)
    count = int(np.prod(shape))
    if count > maximum_voxels:
        raise PocketDynamicsError(
            f"pocket grid contains {count} voxels; maximum_grid_voxels is {maximum_voxels}"
        )
    indices = np.indices(tuple(shape), dtype=float).reshape(3, -1).T
    centers = origin + (indices + 0.5) * spacing
    return ({
        "origin_angstrom": origin.tolist(), "spacing_angstrom": spacing,
        "shape": shape.tolist(), "voxel_count": count, "axis_order": "x_y_z",
    }, centers)


def _components(active: Iterable[Voxel]) -> List[List[Voxel]]:
    unseen = set(active)
    result = []
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


def _residue_id(atom: AtomRecord) -> str:
    return (
        f"{atom.chain_id or '_'}:{atom.residue_number}{atom.insertion_code}:"
        f"{atom.residue_name}"
    )


def _frame_pockets(
    coordinates: np.ndarray, grid_centers: np.ndarray,
    grid_shape: Tuple[int, int, int], atom_residue_ids: Sequence[str],
    settings: Mapping[str, object],
) -> List[Dict[str, object]]:
    minimum2 = float(settings["minimum_clearance_angstrom"]) ** 2
    maximum2 = float(settings["maximum_surface_distance_angstrom"]) ** 2
    neighborhood2 = float(settings["neighborhood_radius_angstrom"]) ** 2
    active_flat = []
    enclosure_by_flat: Dict[int, float] = {}
    clearance_by_flat: Dict[int, float] = {}
    nearby_by_flat: Dict[int, Tuple[int, ...]] = {}
    chunk_size = 2048
    for start in range(0, len(grid_centers), chunk_size):
        points = grid_centers[start:start + chunk_size]
        vectors = coordinates[None, :, :] - points[:, None, :]
        squared = np.einsum("ijk,ijk->ij", vectors, vectors)
        minimum_distance = squared.min(axis=1)
        near = squared <= neighborhood2
        counts = near.sum(axis=1)
        lengths = np.sqrt(np.maximum(squared, 1.0e-18))
        unit = vectors / lengths[:, :, None]
        directional = np.linalg.norm((unit * near[:, :, None]).sum(axis=1), axis=1) / np.maximum(counts, 1)
        direction_count = sum(
            np.any(near & (sign * unit[:, :, axis] >= 0.5), axis=1).astype(int)
            for axis in range(3) for sign in (-1.0, 1.0)
        )
        base = np.where(
            (minimum_distance >= minimum2)
            & (minimum_distance <= maximum2)
            & (counts >= int(settings["minimum_nearby_atoms"]))
            & (direction_count >= int(settings["minimum_occupied_directions"]))
            & (directional <= float(settings["maximum_directional_imbalance"]))
        )[0]
        for local in base:
            atom_indices = tuple(int(value) for value in np.flatnonzero(near[local]))
            residues = {atom_residue_ids[index] for index in atom_indices}
            if len(residues) < int(settings["minimum_nearby_residues"]):
                continue
            flat = start + int(local)
            active_flat.append(flat)
            enclosure_by_flat[flat] = 1.0 - float(directional[local])
            clearance_by_flat[flat] = math.sqrt(float(minimum_distance[local]))
            nearby_by_flat[flat] = atom_indices
    active_centers = {flat: grid_centers[flat] for flat in active_flat}
    active_voxels = {
        tuple(int(value) for value in np.unravel_index(flat, grid_shape)): flat
        for flat in active_flat
    }
    seeds: List[int] = []
    minimum_seed = float(settings["minimum_seed_clearance_angstrom"])
    separation = float(settings["minimum_seed_separation_angstrom"])
    for flat in sorted(
        active_flat,
        key=lambda value: (
            -clearance_by_flat[value], -enclosure_by_flat[value], value
        ),
    ):
        if clearance_by_flat[flat] < minimum_seed:
            continue
        if all(
            float(np.linalg.norm(active_centers[flat] - active_centers[prior]))
            >= separation for prior in seeds
        ):
            seeds.append(flat)
        if len(seeds) >= int(settings["maximum_pockets_per_frame"]):
            break
    result = []
    growth = float(settings["pocket_growth_radius_angstrom"])
    offsets = (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0),
        (0, -1, 0), (0, 0, 1), (0, 0, -1),
    )
    for seed in seeds:
        eligible = {
            voxel: flat for voxel, flat in active_voxels.items()
            if float(np.linalg.norm(active_centers[flat] - active_centers[seed]))
            <= growth
        }
        seed_voxel = tuple(
            int(value) for value in np.unravel_index(seed, grid_shape)
        )
        queue = deque((seed_voxel,))
        visited = {seed_voxel}
        while queue:
            voxel = queue.popleft()
            for offset in offsets:
                neighbor = tuple(
                    voxel[axis] + offset[axis] for axis in range(3)
                )
                if neighbor in eligible and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        flats = sorted(eligible[voxel] for voxel in visited)
        if len(flats) < int(settings["minimum_pocket_voxels"]):
            continue
        voxels = [
            tuple(int(value) for value in np.unravel_index(flat, grid_shape))
            for flat in flats
        ]
        residue_ids = sorted({
            atom_residue_ids[index]
            for flat in flats for index in nearby_by_flat[flat]
        })
        centers = grid_centers[flats]
        result.append({
            "voxel_indices": [list(voxel) for voxel in voxels],
            "voxel_count": len(voxels),
            "volume_angstrom3": len(voxels) * float(settings["grid_spacing_angstrom"]) ** 3,
            "centroid_angstrom": centers.mean(axis=0).tolist(),
            "mean_enclosure_score": sum(enclosure_by_flat[flat] for flat in flats) / len(flats),
            "seed_clearance_angstrom": clearance_by_flat[seed],
            "seed_voxel_index": [
                int(value) for value in np.unravel_index(seed, grid_shape)
            ],
            "lining_residue_ids": residue_ids,
        })
    return result


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _track_instances(
    instances: List[Dict[str, object]], settings: Mapping[str, object]
) -> Tuple[List[Dict[str, object]], int]:
    clusters: List[Dict[str, object]] = []
    cluster_frame_keys: List[set[FrameKey]] = []
    comparisons = 0
    for instance in instances:
        best = None
        best_score = -1.0
        centroid = np.asarray(instance["centroid_angstrom"], dtype=float)
        frame_key: FrameKey = (
            str(instance.get("system_id", "")),
            str(instance.get("replica_id", "")),
            str(instance.get("segment_id", "")),
            int(instance.get("source_frame_index", instance["pocket_instance_index"])),
        )
        for index, cluster in enumerate(clusters):
            comparisons += 1
            if comparisons > int(settings["maximum_tracking_comparisons"]):
                raise PocketDynamicsError("maximum_tracking_comparisons gate exceeded")
            if frame_key in cluster_frame_keys[index]:
                continue
            distance = float(np.linalg.norm(centroid - np.asarray(cluster["centroid_angstrom"], dtype=float)))
            if distance > float(settings["maximum_centroid_distance_angstrom"]):
                continue
            score = _jaccard(
                instance["lining_residue_ids"],  # type: ignore[arg-type]
                cluster["representative_lining_residue_ids"],  # type: ignore[arg-type]
            )
            if score >= float(settings["residue_jaccard_threshold"]) and score > best_score:
                best, best_score = index, score
        if best is None:
            cluster_id = f"pocket-cluster-{len(clusters) + 1}"
            instance["pocket_cluster_id"] = cluster_id
            clusters.append({
                "pocket_cluster_id": cluster_id,
                "representative_lining_residue_ids": list(instance["lining_residue_ids"]),
                "centroid_angstrom": list(instance["centroid_angstrom"]),
                "instance_indices": [int(instance["pocket_instance_index"])],
            })
            cluster_frame_keys.append({frame_key})
        else:
            cluster = clusters[best]
            instance["pocket_cluster_id"] = cluster["pocket_cluster_id"]
            cluster["instance_indices"].append(int(instance["pocket_instance_index"]))  # type: ignore[union-attr]
            cluster_frame_keys[best].add(frame_key)
    return clusters, comparisons


def _flat_pocket_voxels(
    pockets: Sequence[Mapping[str, object]], grid_shape: Tuple[int, int, int]
) -> Tuple[int, ...]:
    """Return the union of pocket voxels as stable flat grid indices."""
    return tuple(sorted({
        int(np.ravel_multi_index(tuple(int(value) for value in voxel), grid_shape))
        for pocket in pockets
        for voxel in pocket["voxel_indices"]  # type: ignore[union-attr]
    }))


def _discover_frequency_region_flats(
    system_voxel_counts: Mapping[str, Counter[int]],
    system_frame_counts: Mapping[str, int],
    grid_shape: Tuple[int, int, int],
    settings: Mapping[str, object],
) -> List[List[int]]:
    """Define recurrent connected regions from per-system voxel frequencies."""
    threshold = float(settings["minimum_region_frequency_fraction"])
    recurrent = set()
    for system_id, counts in system_voxel_counts.items():
        denominator = int(system_frame_counts[system_id])
        recurrent.update(
            flat for flat, count in counts.items()
            if count / denominator >= threshold
        )
    components = [
        component for component in _components(
            tuple(
                int(value) for value in np.unravel_index(flat, grid_shape)
            )
            for flat in recurrent
        )
        if len(component) >= int(settings["minimum_region_voxels"])
    ]
    ranked: List[Tuple[float, List[int]]] = []
    for component in components:
        flats = sorted(
            int(np.ravel_multi_index(voxel, grid_shape)) for voxel in component
        )
        maximum_frequency = max(
            (
                counts.get(flat, 0) / int(system_frame_counts[system_id])
                for system_id, counts in system_voxel_counts.items()
                for flat in flats
            ),
            default=0.0,
        )
        ranked.append((maximum_frequency, flats))
    ranked.sort(key=lambda row: (-row[0], -len(row[1]), row[1][0]))
    if len(ranked) > int(settings["maximum_frequency_regions"]):
        raise PocketDynamicsError(
            f"frequency map produced {len(ranked)} recurrent regions; "
            "maximum_frequency_regions gate exceeded"
        )
    return [row[1] for row in ranked]


def _representative_frame_rows(
    candidates: Sequence[Tuple[float, Mapping[str, object]]], maximum: int,
) -> List[Dict[str, object]]:
    """Choose observed maximum-volume and mean-volume-nearest frames."""
    if not candidates:
        return []
    mean_volume = sum(row[0] for row in candidates) / len(candidates)
    ranked = [
        ("maximum_observed_volume", max(
            candidates,
            key=lambda row: (
                row[0], -int(row[1]["source_frame_index"]),
                str(row[1]["replica_id"]), str(row[1]["segment_id"]),
            ),
        )),
        ("closest_observed_to_mean_volume", min(
            candidates,
            key=lambda row: (
                abs(row[0] - mean_volume), int(row[1]["source_frame_index"]),
                str(row[1]["replica_id"]), str(row[1]["segment_id"]),
            ),
        )),
    ]
    ranked.extend(
        ("additional_central_observed_volume", candidate)
        for candidate in sorted(
            candidates,
            key=lambda row: (
                abs(row[0] - mean_volume), int(row[1]["source_frame_index"]),
                str(row[1]["replica_id"]), str(row[1]["segment_id"]),
            ),
        )
    )
    result = []
    seen = set()
    for reason, (volume, frame) in ranked:
        identity = (
            str(frame["system_id"]), str(frame["replica_id"]),
            str(frame["segment_id"]), int(frame["source_frame_index"]),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append({
            "selection_reason": reason,
            "system_id": identity[0], "replica_id": identity[1],
            "segment_id": identity[2], "source_frame_index": identity[3],
            "axis_kind": frame["axis_kind"], "axis_unit": frame["axis_unit"],
            "axis_value": frame["axis_value"],
            "region_volume_angstrom3": volume,
        })
        if len(result) >= maximum:
            break
    return result


def _frequency_map_project(
    source: Path, project: Mapping[str, object], settings: Mapping[str, object],
    hash_content: bool,
) -> Dict[str, object]:
    """Accumulate aligned pocket occupancy before defining recurrent regions."""
    context = compile_project_context_file(source, hash_content=hash_content)
    contract = context["contract"]
    assert isinstance(contract, dict)
    selections = contract["selections"]
    units = contract["units"]
    assert isinstance(selections, dict) and isinstance(units, dict)
    coordinate_unit = str(units["coordinates"])
    alignment_definition = selections.get(str(settings["alignment_selection"]))
    solute_definition = selections.get(str(settings["solute_selection"]))
    if not isinstance(alignment_definition, dict) or not isinstance(solute_definition, dict):
        raise PocketDynamicsError("alignment or solute selection is undefined")

    reference_path = resolve_manifest_path(str(project["reference_structure"]), source)
    _, reference_atoms = read_topology_atoms(reference_path)
    try:
        raw_reference = next(iter_coordinate_frames(reference_path, coordinate_unit))
    except StopIteration as exc:
        raise PocketDynamicsError("reference structure contains no coordinates") from exc
    reference_frame = PeriodicFrameProcessor.from_reference(
        project, source, len(reference_atoms)
    ).process(raw_reference, str(reference_path))
    reference_solute = select_atoms(
        reference_atoms, solute_definition, str(settings["solute_selection"])
    )
    reference_solute_coordinates = _coordinates_at(
        reference_frame.coordinates_angstrom,
        [atom.atom_index for atom in reference_solute],
    )
    grid, grid_centers = _grid(
        reference_solute_coordinates, float(settings["grid_spacing_angstrom"]),
        float(settings["grid_padding_angstrom"]), int(settings["maximum_grid_voxels"]),
    )
    grid_shape = tuple(int(value) for value in grid["shape"])

    system_path = Path(str(context["system_manifest_path"]))
    manifest = load_json(system_path)
    frame_plan, frame_report = plan_frame_selection(
        manifest, system_path, coordinate_unit, settings["frame_selection"],
        frame_stride=int(settings["frame_stride"]), error_type=PocketDynamicsError,
    )
    if int(frame_report["selected_frame_count"]) > int(settings["maximum_frames"]):
        raise PocketDynamicsError("maximum_frames gate exceeded")
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
    mappings_by_key = {
        (row[0], row[1]): mapping for row, mapping in zip(topology_rows, mappings)
    }
    atoms_by_key = {(row[0], row[1]): row[2] for row in topology_rows}
    reference_alignment = _coordinates_at(
        reference_frame.coordinates_angstrom, mappings[0].reference_indices
    )

    frame_rows: List[Dict[str, object]] = []
    system_frames: Counter[str] = Counter()
    system_voxel_counts: Dict[str, Counter[int]] = defaultdict(Counter)
    voxel_lining_occurrences: Dict[int, Counter[str]] = defaultdict(Counter)
    total_sparse_voxels = 0
    saturated_frame_count = 0
    for system in manifest["systems"]:
        system_id = str(system["system_id"])
        for replica in system["replicas"]:
            replica_id = str(replica["replica_id"])
            atoms = atoms_by_key[(system_id, replica_id)]
            mapping = mappings_by_key[(system_id, replica_id)]
            solute_atoms = select_atoms(
                atoms, solute_definition, str(settings["solute_selection"])
            )
            solute_indices = tuple(atom.atom_index for atom in solute_atoms)
            residue_ids = tuple(_residue_id(atom) for atom in solute_atoms)
            required_indices = tuple(sorted(set(mapping.target_indices) | set(solute_indices)))
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
                    aligned = np.asarray(apply_transform(
                        _coordinates_at(frame.coordinates_angstrom, solute_indices), transform
                    ), dtype=float)
                    pockets = _frame_pockets(
                        aligned, grid_centers, grid_shape, residue_ids, settings
                    )
                    if len(pockets) == int(settings["maximum_pockets_per_frame"]):
                        saturated_frame_count += 1
                    active_flats = _flat_pocket_voxels(pockets, grid_shape)
                    total_sparse_voxels += len(active_flats)
                    if total_sparse_voxels > int(settings["maximum_sparse_frame_voxels"]):
                        raise PocketDynamicsError(
                            "maximum_sparse_frame_voxels gate exceeded"
                        )
                    system_voxel_counts[system_id].update(active_flats)
                    frame_lining: Dict[int, set[str]] = defaultdict(set)
                    for pocket in pockets:
                        for voxel in pocket["voxel_indices"]:  # type: ignore[union-attr]
                            flat = int(np.ravel_multi_index(
                                tuple(int(value) for value in voxel), grid_shape
                            ))
                            frame_lining[flat].update(
                                str(value) for value in pocket["lining_residue_ids"]  # type: ignore[union-attr]
                            )
                    for flat, residues in frame_lining.items():
                        voxel_lining_occurrences[flat].update(residues)
                    frame_rows.append({
                        "system_id": system_id, "replica_id": replica_id,
                        "segment_id": segment_id,
                        "source_frame_index": int(frame.frame_index),
                        "axis_kind": axis["kind"],
                        "axis_unit": project.get("time_unit")
                        if axis["kind"] == "physical_time" else "sample",
                        "axis_value": frame_axis_value(axis, frame.frame_index),
                        "detected_pocket_voxel_count": len(active_flats),
                        "_active_voxel_flat_indices": array("I", active_flats),
                    })
                    system_frames[system_id] += 1
    for system_id, count in system_frames.items():
        if count < int(settings["minimum_evaluated_frames_per_system"]):
            raise PocketDynamicsError(
                f"system {system_id} produced {count} selected frames; minimum is "
                f"{settings['minimum_evaluated_frames_per_system']}"
            )
    if not system_frames:
        raise PocketDynamicsError("no trajectory frames were evaluated")

    region_flats = _discover_frequency_region_flats(
        system_voxel_counts, system_frames, grid_shape, settings
    )
    flat_to_region = {
        flat: index for index, flats in enumerate(region_flats, start=1)
        for flat in flats
    }
    region_frame_candidates: Dict[int, Dict[str, List[Tuple[float, Mapping[str, object]]]]] = defaultdict(lambda: defaultdict(list))
    region_present_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    region_volumes: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    voxel_volume = float(settings["grid_spacing_angstrom"]) ** 3
    for frame in frame_rows:
        counts = Counter(
            flat_to_region[flat]
            for flat in frame["_active_voxel_flat_indices"]  # type: ignore[union-attr]
            if flat in flat_to_region
        )
        frame["active_pocket_regions"] = [
            {
                "pocket_region_id": f"pocket-region-{region_index}",
                "active_voxel_count": count,
                "volume_angstrom3": count * voxel_volume,
            }
            for region_index, count in sorted(counts.items())
        ]
        del frame["_active_voxel_flat_indices"]
        for region_index, count in counts.items():
            system_id = str(frame["system_id"])
            volume = count * voxel_volume
            region_present_counts[region_index][system_id] += 1
            region_volumes[region_index][system_id].append(volume)
            region_frame_candidates[region_index][system_id].append((volume, frame))

    regions = []
    for region_index, flats in enumerate(region_flats, start=1):
        per_system = []
        for system_id in sorted(system_frames):
            present = region_present_counts[region_index][system_id]
            volumes = region_volumes[region_index][system_id]
            per_system.append({
                "system_id": system_id,
                "present_frame_count": present,
                "evaluated_frame_count": system_frames[system_id],
                "occupancy_fraction": present / system_frames[system_id],
                "present_frame_volume_summary_angstrom3": sample_summary(volumes)
                if volumes else None,
                "representative_frames": _representative_frame_rows(
                    region_frame_candidates[region_index][system_id],
                    int(settings["representative_frames_per_region"]),
                ),
            })
        lining = Counter()
        for flat in flats:
            lining.update(voxel_lining_occurrences.get(flat, Counter()))
        centers = grid_centers[flats]
        regions.append({
            "pocket_region_id": f"pocket-region-{region_index}",
            "voxel_flat_indices": flats,
            "voxel_indices": [
                [int(value) for value in np.unravel_index(flat, grid_shape)]
                for flat in flats
            ],
            "voxel_count": len(flats),
            "maximum_geometric_volume_angstrom3": len(flats) * voxel_volume,
            "centroid_angstrom": centers.mean(axis=0).tolist(),
            "lining_residue_voxel_frame_occurrences": [
                {"residue_id": residue_id, "occurrence_count": count}
                for residue_id, count in sorted(
                    lining.items(), key=lambda row: (-row[1], row[0])
                )
            ],
            "per_system_occupancy": per_system,
        })

    systems = sorted(system_frames)
    pairwise = []
    for region in regions:
        occupancies = {
            str(row["system_id"]): float(row["occupancy_fraction"])
            for row in region["per_system_occupancy"]
        }
        for left_index, left in enumerate(systems):
            for right in systems[left_index + 1:]:
                pairwise.append({
                    "pocket_region_id": region["pocket_region_id"],
                    "system_i": left, "system_j": right,
                    "occupancy_difference": occupancies[left] - occupancies[right],
                })

    total_frames = sum(system_frames.values())
    observed_flats = sorted({
        flat for counts in system_voxel_counts.values() for flat in counts
    })
    frequency_voxels = []
    for flat in observed_flats:
        frequency_voxels.append({
            "voxel_flat_index": flat,
            "voxel_index": [
                int(value) for value in np.unravel_index(flat, grid_shape)
            ],
            "center_angstrom": grid_centers[flat].tolist(),
            "pooled_frame_frequency_fraction": sum(
                counts.get(flat, 0) for counts in system_voxel_counts.values()
            ) / total_frames,
            "per_system_frequency": [
                {
                    "system_id": system_id,
                    "occupied_frame_count": system_voxel_counts[system_id].get(flat, 0),
                    "evaluated_frame_count": system_frames[system_id],
                    "frequency_fraction": system_voxel_counts[system_id].get(flat, 0)
                    / system_frames[system_id],
                }
                for system_id in systems
            ],
        })

    findings = []
    for system_id in systems:
        choices = [
            (
                next(
                    float(row["occupancy_fraction"])
                    for row in region["per_system_occupancy"]
                    if row["system_id"] == system_id
                ),
                region,
            )
            for region in regions
        ]
        if choices:
            occupancy, region = max(
                choices, key=lambda row: (row[0], row[1]["pocket_region_id"])
            )
            findings.append({
                "category": "other_physical", "evidence_level": "descriptive",
                "statement": (
                    f"Most persistent recurrent geometric pocket region in {system_id} "
                    f"is {region['pocket_region_id']} at {occupancy:.1%} of evaluated "
                    "frames; this is not a druggability prediction."
                ),
                "effect_value": occupancy, "system_ids": [system_id],
                "comparison_family": "ensemble_pocket_dynamics:within_system_frequency_region",
            })
    for row in pairwise:
        if float(row["occupancy_difference"]) != 0.0:
            findings.append({
                "category": "other_physical", "evidence_level": "descriptive",
                "statement": (
                    f"Recurrent geometric pocket region {row['pocket_region_id']} "
                    f"occupancy differs by {float(row['occupancy_difference']):+.1%} "
                    f"between {row['system_i']} and {row['system_j']}."
                ),
                "effect_value": row["occupancy_difference"],
                "system_ids": [row["system_i"], row["system_j"]],
                "comparison_family": "ensemble_pocket_dynamics:pairwise_frequency_region",
            })

    issues = []
    if not observed_flats:
        issues.append({
            "severity": "warning", "code": "NO_GEOMETRIC_POCKETS_DETECTED",
            "message": "No grid voxel met the configured geometric pocket criteria.",
        })
    elif not regions:
        issues.append({
            "severity": "warning", "code": "NO_RECURRENT_POCKET_REGIONS",
            "message": (
                "Pocket-like voxels were observed, but no connected region met the "
                "configured frequency and size thresholds."
            ),
        })
    if saturated_frame_count:
        issues.append({
            "severity": "warning", "code": "POCKETS_PER_FRAME_CAP_REACHED",
            "message": (
                f"{saturated_frame_count} evaluated frames reached the configured "
                "maximum_pockets_per_frame; lower-ranked clearance seeds were omitted."
            ),
        })
    return {
        "module_id": "ensemble_pocket_dynamics",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "availability_status": "available", "availability_reason": None,
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "grid": grid, "frame_selection": frame_report,
        "system_frame_counts": dict(sorted(system_frames.items())),
        "voxel_frequency_map": frequency_voxels,
        "frame_pocket_region_records": frame_rows,
        "recurrent_pocket_regions": regions,
        "pairwise_system_pocket_differences": pairwise,
        "frames_reaching_pocket_cap": saturated_frame_count,
        "tracking_comparison_count": 0,
        "tracking_contract": (
            "reference-aligned voxel frequencies define recurrent connected spatial "
            "regions after all selected frames; no online discrete pocket IDs"
        ),
        "finding_candidates": findings,
        "observation_accounting": {
            "selected_physical_frame_count": total_frames,
            "sparse_frame_voxel_count": total_sparse_voxels,
            "observed_frequency_voxel_count": len(observed_flats),
            "recurrent_region_count": len(regions),
        },
        "error_count": 0,
        "warning_count": sum(issue["severity"] == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "The native frequency-grid backend detects recurrent geometric empty regions; it does not predict druggability, ligand affinity, binding free energy, or opening kinetics.",
            "The frequency-map design is inspired by ensemble cavity mapping, but the native geometric detector is not an fpocket or MDpocket alpha-sphere reimplementation.",
            "Frequency and region definitions require grid, clearance, atom-selection, region-threshold, and frame-selection sensitivity analysis.",
            "Lining-residue occurrence counts are voxel-frame occurrences and are not independent observations.",
            "Water and ion occupancy are analyzed separately by hydration_density_channels; solvent is not treated as pocket wall material.",
            "Frame persistence is descriptive and is not independent-replica uncertainty.",
        ],
    }


def ensemble_pocket_dynamics_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    if settings["backend"] == "native_frequency_grid_v2":
        return _frequency_map_project(source, project, settings, hash_content)
    context = compile_project_context_file(source, hash_content=hash_content)
    contract = context["contract"]
    assert isinstance(contract, dict)
    selections = contract["selections"]
    units = contract["units"]
    assert isinstance(selections, dict) and isinstance(units, dict)
    coordinate_unit = str(units["coordinates"])
    alignment_definition = selections.get(str(settings["alignment_selection"]))
    solute_definition = selections.get(str(settings["solute_selection"]))
    if not isinstance(alignment_definition, dict) or not isinstance(solute_definition, dict):
        raise PocketDynamicsError("alignment or solute selection is undefined")
    reference_path = resolve_manifest_path(str(project["reference_structure"]), source)
    _, reference_atoms = read_topology_atoms(reference_path)
    try:
        raw_reference = next(iter_coordinate_frames(reference_path, coordinate_unit))
    except StopIteration as exc:
        raise PocketDynamicsError("reference structure contains no coordinates") from exc
    reference_frame = PeriodicFrameProcessor.from_reference(
        project, source, len(reference_atoms)
    ).process(raw_reference, str(reference_path))
    reference_solute = select_atoms(
        reference_atoms, solute_definition, str(settings["solute_selection"])
    )
    reference_solute_coordinates = _coordinates_at(
        reference_frame.coordinates_angstrom,
        [atom.atom_index for atom in reference_solute],
    )
    grid, grid_centers = _grid(
        reference_solute_coordinates, float(settings["grid_spacing_angstrom"]),
        float(settings["grid_padding_angstrom"]), int(settings["maximum_grid_voxels"]),
    )
    grid_shape = tuple(int(value) for value in grid["shape"])

    system_path = Path(str(context["system_manifest_path"]))
    manifest = load_json(system_path)
    frame_plan, frame_report = plan_frame_selection(
        manifest, system_path, coordinate_unit, settings["frame_selection"],
        frame_stride=int(settings["frame_stride"]), error_type=PocketDynamicsError,
    )
    if int(frame_report["selected_frame_count"]) > int(settings["maximum_frames"]):
        raise PocketDynamicsError("maximum_frames gate exceeded")
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
    mappings_by_key = {
        (row[0], row[1]): mapping for row, mapping in zip(topology_rows, mappings)
    }
    atoms_by_key = {(row[0], row[1]): row[2] for row in topology_rows}
    reference_alignment = _coordinates_at(
        reference_frame.coordinates_angstrom, mappings[0].reference_indices
    )

    instances: List[Dict[str, object]] = []
    frame_rows: List[Dict[str, object]] = []
    system_frames: Counter[str] = Counter()
    saturated_frame_count = 0
    for system in manifest["systems"]:
        system_id = str(system["system_id"])
        for replica in system["replicas"]:
            replica_id = str(replica["replica_id"])
            atoms = atoms_by_key[(system_id, replica_id)]
            mapping = mappings_by_key[(system_id, replica_id)]
            solute_atoms = select_atoms(atoms, solute_definition, str(settings["solute_selection"]))
            solute_indices = tuple(atom.atom_index for atom in solute_atoms)
            residue_ids = tuple(_residue_id(atom) for atom in solute_atoms)
            required_indices = tuple(sorted(set(mapping.target_indices) | set(solute_indices)))
            processor = PeriodicFrameProcessor.from_replica(project, replica, system_path, len(atoms))
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
                    selected = frame_selected(raw_frame.frame_index, selected_indices, int(settings["frame_stride"]))
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
                    aligned = np.asarray(apply_transform(
                        _coordinates_at(frame.coordinates_angstrom, solute_indices), transform
                    ), dtype=float)
                    pockets = _frame_pockets(
                        aligned, grid_centers, grid_shape, residue_ids, settings
                    )
                    if len(pockets) == int(settings["maximum_pockets_per_frame"]):
                        saturated_frame_count += 1
                    key: FrameKey = (system_id, replica_id, segment_id, int(frame.frame_index))
                    local_indices = []
                    for pocket in pockets:
                        index = len(instances)
                        if index >= int(settings["maximum_pocket_instances"]):
                            raise PocketDynamicsError("maximum_pocket_instances gate exceeded")
                        pocket.update({
                            "pocket_instance_index": index,
                            "system_id": system_id, "replica_id": replica_id,
                            "segment_id": segment_id,
                            "source_frame_index": int(frame.frame_index),
                        })
                        instances.append(pocket)
                        local_indices.append(index)
                    frame_rows.append({
                        "system_id": system_id, "replica_id": replica_id,
                        "segment_id": segment_id,
                        "source_frame_index": int(frame.frame_index),
                        "axis_kind": axis["kind"],
                        "axis_unit": project.get("time_unit") if axis["kind"] == "physical_time" else "sample",
                        "axis_value": frame_axis_value(axis, frame.frame_index),
                        "pocket_instance_indices": local_indices,
                    })
                    system_frames[system_id] += 1
    for system_id, count in system_frames.items():
        if count < int(settings["minimum_evaluated_frames_per_system"]):
            raise PocketDynamicsError(
                f"system {system_id} produced {count} selected frames; minimum is "
                f"{settings['minimum_evaluated_frames_per_system']}"
            )
    if not system_frames:
        raise PocketDynamicsError("no trajectory frames were evaluated")

    clusters, tracking_comparisons = _track_instances(instances, settings)
    instance_by_index = {int(row["pocket_instance_index"]): row for row in instances}
    for frame in frame_rows:
        frame["active_pocket_cluster_ids"] = sorted({
            str(instance_by_index[index]["pocket_cluster_id"])
            for index in frame["pocket_instance_indices"]  # type: ignore[union-attr]
        })
    summaries = []
    for cluster in clusters:
        cluster_instances = [instance_by_index[index] for index in cluster["instance_indices"]]
        volumes = [float(row["volume_angstrom3"]) for row in cluster_instances]
        systems = sorted({str(row["system_id"]) for row in cluster_instances})
        per_system = []
        for system_id in sorted(system_frames):
            frames = {
                (str(row["replica_id"]), str(row["segment_id"]), int(row["source_frame_index"]))
                for row in cluster_instances if row["system_id"] == system_id
            }
            per_system.append({
                "system_id": system_id,
                "present_frame_count": len(frames),
                "evaluated_frame_count": system_frames[system_id],
                "occupancy_fraction": len(frames) / system_frames[system_id],
            })
        summaries.append({
            **cluster, "instance_count": len(cluster_instances),
            "observed_system_ids": systems, "volume_summary_angstrom3": sample_summary(volumes),
            "mean_enclosure_score": sum(float(row["mean_enclosure_score"]) for row in cluster_instances) / len(cluster_instances),
            "per_system_occupancy": per_system,
        })
    pairwise = []
    systems = sorted(system_frames)
    for summary in summaries:
        occupancies = {
            str(row["system_id"]): float(row["occupancy_fraction"])
            for row in summary["per_system_occupancy"]
        }
        for left_index, left in enumerate(systems):
            for right in systems[left_index + 1:]:
                pairwise.append({
                    "pocket_cluster_id": summary["pocket_cluster_id"],
                    "system_i": left, "system_j": right,
                    "occupancy_difference": occupancies[left] - occupancies[right],
                })
    findings = []
    for system_id in systems:
        choices = []
        for summary in summaries:
            occupancy = next(
                float(row["occupancy_fraction"])
                for row in summary["per_system_occupancy"]
                if row["system_id"] == system_id
            )
            choices.append((occupancy, summary))
        if choices:
            occupancy, summary = max(choices, key=lambda row: (row[0], row[1]["pocket_cluster_id"]))
            findings.append({
                "category": "other_physical", "evidence_level": "descriptive",
                "statement": (
                    f"Most persistent geometric pocket in {system_id} is "
                    f"{summary['pocket_cluster_id']} at {occupancy:.1%} of evaluated frames; "
                    "this is not a druggability prediction."
                ),
                "effect_value": occupancy, "system_ids": [system_id],
                "comparison_family": "ensemble_pocket_dynamics:within_system_persistence",
            })
    for row in pairwise:
        if float(row["occupancy_difference"]) != 0.0:
            findings.append({
                "category": "other_physical", "evidence_level": "descriptive",
                "statement": (
                    f"Geometric pocket {row['pocket_cluster_id']} occupancy differs by "
                    f"{float(row['occupancy_difference']):+.1%} between {row['system_i']} "
                    f"and {row['system_j']}."
                ),
                "effect_value": row["occupancy_difference"],
                "system_ids": [row["system_i"], row["system_j"]],
                "comparison_family": "ensemble_pocket_dynamics:pairwise_occupancy_difference",
            })
    issues = []
    if not instances:
        issues.append({
            "severity": "warning", "code": "NO_GEOMETRIC_POCKETS_DETECTED",
            "message": "No grid component met the configured geometric pocket criteria.",
        })
    if saturated_frame_count:
        issues.append({
            "severity": "warning", "code": "POCKETS_PER_FRAME_CAP_REACHED",
            "message": (
                f"{saturated_frame_count} evaluated frames reached the configured "
                "maximum_pockets_per_frame; lower-ranked clearance seeds were omitted."
            ),
        })
    return {
        "module_id": "ensemble_pocket_dynamics",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "availability_status": "available", "availability_reason": None,
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "grid": grid, "frame_selection": frame_report,
        "system_frame_counts": dict(sorted(system_frames.items())),
        "pocket_instances": instances, "frame_pocket_records": frame_rows,
        "pocket_clusters": summaries,
        "pairwise_system_pocket_differences": pairwise,
        "frames_reaching_pocket_cap": saturated_frame_count,
        "tracking_comparison_count": tracking_comparisons,
        "tracking_contract": "deterministic online residue-Jaccard and centroid gate in exact frame order",
        "finding_candidates": findings,
        "observation_accounting": {
            "selected_physical_frame_count": sum(system_frames.values()),
            "pocket_instance_count": len(instances),
        },
        "error_count": 0,
        "warning_count": sum(issue["severity"] == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "The native grid backend detects geometric empty regions; it does not predict druggability, ligand affinity, binding free energy, or cryptic-pocket opening kinetics.",
            "Pocket identity uses deterministic online residue overlap and centroid gates and requires threshold, grid, atom-selection, and frame-selection sensitivity analysis.",
            "Water and ion occupancy are analyzed separately by hydration_density_channels and can be compared through exact frame identities; solvent is not treated as pocket wall material.",
            "Frame persistence is descriptive and is not independent-replica uncertainty.",
        ],
    }


def ensemble_pocket_dynamics_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return ensemble_pocket_dynamics_project(project_path, hash_content=hash_content)
    except (
        PocketDynamicsError, ManifestValidationError, AtomMappingError,
        CoordinateReadError, PeriodicReconstructionError, GeometryError,
        TrajectoryContractError, OSError, KeyError, TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "ensemble_pocket_dynamics",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "POCKET_DYNAMICS_INVALID",
                "message": message,
            } for message in messages],
        }
