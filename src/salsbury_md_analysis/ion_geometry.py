"""General bound-ion coordination and ion-pair geometry analysis."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import frame_selected, plan_frame_selection, reader_frame_indices
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .moments import sample_summary
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError, minimum_image_displacement
from .replica_execution import ReplicaPartial
from .replica_module_execution import (
    execute_replica_final_module,
    restore_source_provenance,
    unique_issues,
)
from .scalar_distributions import ScalarDistributionError, analyze_scalar_distribution
from .trajectory_contracts import TrajectoryContractError, frame_axis_value, normalize_segment_axis
from .validation import positive_integer


class IonGeometryError(ValueError):
    """Raised when an ion-geometry request is incomplete or unsafe."""


_IDEAL_PAIR_ANGLES = {
    "linear": (2, [180.0]),
    "trigonal_planar": (3, [120.0] * 3),
    "tetrahedral": (4, [109.47122063449069] * 6),
    "square_planar": (4, [90.0] * 4 + [180.0] * 2),
    "trigonal_bipyramidal": (5, [90.0] * 6 + [120.0] * 3 + [180.0]),
    "square_pyramidal": (5, [90.0] * 8 + [180.0] * 2),
    "octahedral": (6, [90.0] * 12 + [180.0] * 3),
}

_SQRT_THREE = math.sqrt(3.0)
_IDEAL_UNIT_VECTORS = {
    "linear": np.asarray([(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]),
    "trigonal_planar": np.asarray([
        (1.0, 0.0, 0.0),
        (-0.5, _SQRT_THREE / 2.0, 0.0),
        (-0.5, -_SQRT_THREE / 2.0, 0.0),
    ]),
    "tetrahedral": np.asarray([
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    ]) / _SQRT_THREE,
    "square_planar": np.asarray([
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
    ]),
    "trigonal_bipyramidal": np.asarray([
        (1.0, 0.0, 0.0),
        (-0.5, _SQRT_THREE / 2.0, 0.0),
        (-0.5, -_SQRT_THREE / 2.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ]),
    "square_pyramidal": np.asarray([
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]),
    "octahedral": np.asarray([
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ]),
}


def _optimal_shape_match(unit_vectors: np.ndarray, template: str) -> Dict[str, object]:
    """Find the label- and rotation-invariant best ideal-polyhedron match."""

    ideal = _IDEAL_UNIT_VECTORS[template]
    best = None
    for assignment in permutations(range(len(ideal))):
        target = ideal[list(assignment)]
        left, _, right = np.linalg.svd(unit_vectors.T @ target)
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        aligned = unit_vectors @ rotation
        difference = aligned - target
        rms = float(np.sqrt(np.mean(np.sum(difference * difference, axis=1))))
        if best is None or rms < best[0] - 1.0e-15:
            cosines = np.clip(np.sum(aligned * target, axis=1), -1.0, 1.0)
            angular = np.degrees(np.arccos(cosines))
            best = (
                rms,
                float(np.sqrt(np.mean(angular * angular))),
                list(assignment),
                aligned,
            )
    assert best is not None
    return {
        "optimal_shape_rms_unit_vector": best[0],
        "optimal_shape_angular_rms_degrees": best[1],
        "optimal_template_assignment": best[2],
        "ideal_unit_vectors": ideal.tolist(),
        "aligned_observed_unit_vectors": best[3].tolist(),
    }


def coordination_geometry_score(
    vectors: Sequence[Sequence[float]], template: str
) -> Dict[str, object]:
    """Compare sorted ligand-pair angles with a rotation-free ideal template."""

    if template not in _IDEAL_PAIR_ANGLES:
        raise IonGeometryError(f"unknown coordination geometry template: {template}")
    expected_count, ideal = _IDEAL_PAIR_ANGLES[template]
    values = np.asarray(vectors, dtype=float)
    if values.shape != (expected_count, 3) or not np.isfinite(values).all():
        raise IonGeometryError(
            f"{template} requires {expected_count} finite three-dimensional vectors"
        )
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1.0e-15):
        raise IonGeometryError("ion-ligand vector has zero length")
    unit = values / norms[:, None]
    angles = []
    for left in range(expected_count - 1):
        for right in range(left + 1, expected_count):
            cosine = min(1.0, max(-1.0, float(np.dot(unit[left], unit[right]))))
            angles.append(math.degrees(math.acos(cosine)))
    ordered = sorted(angles)
    ideal_ordered = sorted(ideal)
    rms = float(np.sqrt(np.mean(np.square(np.asarray(ordered) - np.asarray(ideal_ordered)))))
    tensor = unit.T @ unit / expected_count
    eigenvalues = sorted(np.linalg.eigvalsh(tensor).tolist(), reverse=True)
    return {
        "template": template,
        "coordination_number": expected_count,
        "observed_pair_angles_degrees": ordered,
        "ideal_pair_angles_degrees": ideal_ordered,
        "rms_pair_angle_deviation_degrees": rms,
        "unit_vector_second_moment_eigenvalues": eigenvalues,
        **_optimal_shape_match(unit, template),
    }


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("ion_coordination_geometry") if isinstance(definitions, dict) else None
    required = {
        "frame_stride", "maximum_frames", "ion_sites", "ion_pairs", "block_count",
        "histogram_rule", "histogram_padding_fraction", "minimum_histogram_bins",
        "maximum_histogram_bins",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise IonGeometryError(
            "definitions.ion_coordination_geometry fields do not match the contract"
        )
    normalized: Dict[str, object] = {
        "frame_stride": positive_integer(
            raw["frame_stride"], "frame_stride", error_type=IonGeometryError
        ),
        "maximum_frames": positive_integer(
            raw["maximum_frames"], "maximum_frames", error_type=IonGeometryError
        ),
        "block_count": positive_integer(
            raw["block_count"], "block_count", error_type=IonGeometryError
        ),
    }
    rule = raw["histogram_rule"]
    if rule not in {"scott", "freedman_diaconis", "rice"}:
        raise IonGeometryError("histogram_rule must be scott, freedman_diaconis, or rice")
    padding = raw["histogram_padding_fraction"]
    if (
        isinstance(padding, bool) or not isinstance(padding, (int, float))
        or not math.isfinite(float(padding)) or float(padding) < 0.0
    ):
        raise IonGeometryError("histogram_padding_fraction must be finite and nonnegative")
    minimum_bins = positive_integer(
        raw["minimum_histogram_bins"], "minimum_histogram_bins", error_type=IonGeometryError
    )
    maximum_bins = positive_integer(
        raw["maximum_histogram_bins"], "maximum_histogram_bins", error_type=IonGeometryError
    )
    if minimum_bins < 2 or maximum_bins < minimum_bins:
        raise IonGeometryError("histogram gates require 2 <= minimum <= maximum")
    normalized.update({
        "histogram_rule": rule,
        "histogram_padding_fraction": float(padding),
        "minimum_histogram_bins": minimum_bins,
        "maximum_histogram_bins": maximum_bins,
    })

    sites = raw["ion_sites"]
    if not isinstance(sites, list) or not sites:
        raise IonGeometryError("ion_sites must be a nonempty array")
    site_ids = set()
    normalized_sites = []
    for index, site in enumerate(sites):
        expected = {
            "site_id", "ion_atom_index", "candidate_ligand_atom_indices",
            "coordination_cutoff_angstrom", "geometry_templates",
        }
        if not isinstance(site, dict) or set(site) != expected:
            raise IonGeometryError(f"ion site {index} fields do not match the contract")
        site_id = str(site["site_id"]).strip()
        ion_index = site["ion_atom_index"]
        ligand_indices = site["candidate_ligand_atom_indices"]
        cutoff = site["coordination_cutoff_angstrom"]
        templates = site["geometry_templates"]
        if (
            not site_id or site_id in site_ids or isinstance(ion_index, bool)
            or not isinstance(ion_index, int) or ion_index < 0
            or not isinstance(ligand_indices, list) or not ligand_indices
            or len(set(ligand_indices)) != len(ligand_indices)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ligand_indices)
            or ion_index in ligand_indices
        ):
            raise IonGeometryError("ion site indices/identifiers are invalid")
        if (
            isinstance(cutoff, bool) or not isinstance(cutoff, (int, float))
            or not math.isfinite(float(cutoff)) or float(cutoff) <= 0.0
        ):
            raise IonGeometryError("coordination_cutoff_angstrom must be finite and positive")
        if (
            not isinstance(templates, list) or len(set(templates)) != len(templates)
            or any(template not in _IDEAL_PAIR_ANGLES for template in templates)
        ):
            raise IonGeometryError("geometry_templates contains an unsupported or duplicate template")
        site_ids.add(site_id)
        normalized_sites.append({
            "site_id": site_id, "ion_atom_index": ion_index,
            "candidate_ligand_atom_indices": list(ligand_indices),
            "coordination_cutoff_angstrom": float(cutoff),
            "geometry_templates": list(templates),
        })
    normalized["ion_sites"] = normalized_sites

    pairs = raw["ion_pairs"]
    if not isinstance(pairs, list):
        raise IonGeometryError("ion_pairs must be an array")
    pair_ids = set()
    normalized_pairs = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or set(pair) != {
            "pair_id", "first_ion_atom_index", "second_ion_atom_index"
        }:
            raise IonGeometryError(f"ion pair {index} fields do not match the contract")
        pair_id = str(pair["pair_id"]).strip()
        first = pair["first_ion_atom_index"]
        second = pair["second_ion_atom_index"]
        if (
            not pair_id or pair_id in pair_ids or isinstance(first, bool)
            or isinstance(second, bool) or not isinstance(first, int)
            or not isinstance(second, int) or min(first, second) < 0 or first == second
        ):
            raise IonGeometryError("ion pair identifiers/indices are invalid")
        pair_ids.add(pair_id)
        normalized_pairs.append({
            "pair_id": pair_id, "first_ion_atom_index": first,
            "second_ion_atom_index": second,
        })
    normalized["ion_pairs"] = normalized_pairs
    return normalized


def _identity(atom: AtomRecord) -> Dict[str, object]:
    return {
        "atom_index": atom.atom_index, "atom_name": atom.atom_name,
        "residue_name": atom.residue_name, "chain_id": atom.chain_id,
        "residue_number": atom.residue_number, "insertion_code": atom.insertion_code,
        "element": atom.element,
    }


def _minimum_image_vector(
    start: Sequence[float], end: Sequence[float], cell
) -> np.ndarray:
    displacement = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    if cell is not None:
        displacement = np.asarray(minimum_image_displacement(displacement.tolist(), cell))
    return displacement


def _validate_topology(
    atoms: Sequence[AtomRecord], settings: Mapping[str, object],
    expected: Dict[str, Tuple[Tuple[object, ...], ...]], location: str,
) -> Dict[str, object]:
    reports = []
    for site in settings["ion_sites"]:  # type: ignore[union-attr]
        site_id = str(site["site_id"])
        indices = [int(site["ion_atom_index"])] + [
            int(value) for value in site["candidate_ligand_atom_indices"]
        ]
        if max(indices) >= len(atoms):
            raise IonGeometryError(f"{location}: ion site {site_id} index exceeds topology")
        keys = tuple(atoms[index].match_key("strict") for index in indices)
        if site_id in expected and expected[site_id] != keys:
            raise IonGeometryError(f"{location}: ion site {site_id} identities differ across replicas")
        expected.setdefault(site_id, keys)
        reports.append({
            "site_id": site_id,
            "ion_identity": _identity(atoms[indices[0]]),
            "candidate_ligand_identities": [_identity(atoms[index]) for index in indices[1:]],
        })
    pair_reports = []
    for pair in settings["ion_pairs"]:  # type: ignore[union-attr]
        indices = [int(pair["first_ion_atom_index"]), int(pair["second_ion_atom_index"])]
        if max(indices) >= len(atoms):
            raise IonGeometryError(f"{location}: ion pair {pair['pair_id']} index exceeds topology")
        pair_reports.append({
            "pair_id": pair["pair_id"],
            "ion_identities": [_identity(atoms[index]) for index in indices],
        })
    return {"sites": reports, "ion_pairs": pair_reports}


def _distributions(
    segment_rows: Mapping[Tuple[str, str, str], List[Mapping[str, object]]],
    metric_ids: Sequence[str], settings: Mapping[str, object],
) -> List[Dict[str, object]]:
    reports = []
    for metric_id in metric_ids:
        segments = []
        for (system_id, replica_id, segment_id), rows in segment_rows.items():
            segments.append(({
                "system_id": system_id, "replica_id": replica_id, "segment_id": segment_id,
            }, [{
                "source_frame_index": row["source_frame_index"],
                "axis_kind": row["axis_kind"], "axis_value": row["axis_value"],
                "value": row["metrics"][metric_id],
            } for row in rows]))
        try:
            analysis = analyze_scalar_distribution(
                segments, binning_rule=str(settings["histogram_rule"]),
                padding_fraction=float(settings["histogram_padding_fraction"]),
                minimum_bins=int(settings["minimum_histogram_bins"]),
                maximum_bins=int(settings["maximum_histogram_bins"]),
            )
            reports.append({"metric_id": metric_id, "status": "complete", **analysis})
        except ScalarDistributionError as exc:
            reports.append({"metric_id": metric_id, "status": "not_estimable", "reason": str(exc)})
    return reports


def _ion_coordination_geometry_project_serial(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(str(context["system_manifest_path"]))
    system = load_json(system_path)
    coordinate_unit = str(project["coordinate_unit"])
    selection_plan, _ = plan_frame_selection(
        system,
        system_path,
        coordinate_unit,
        {
            "mode": "integer_stride_per_replica_v1",
            "stride": int(settings["frame_stride"]),
        },
        frame_stride=1,
        error_type=IonGeometryError,
    )
    output_time_unit = project.get("time_unit")
    periodic_policy = str(project["periodic_coordinate_policy"])
    issues = [issue for issue in context.get("issues", []) if isinstance(issue, dict)]
    expected: Dict[str, Tuple[Tuple[object, ...], ...]] = {}
    topology_reports = []
    frame_reports = []
    segment_rows: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    evaluated = 0

    for raw_system in system["systems"]:
        assert isinstance(raw_system, dict)
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            topology_reports.append({
                "system_id": system_id, "replica_id": replica_id,
                "topology_path": str(topology_path),
                **_validate_topology(atoms, settings, expected, f"{system_id}/{replica_id}"),
            })
            processor = PeriodicFrameProcessor.from_replica(project, replica, system_path, len(atoms))
            reconstruction_atom_indices = tuple(sorted({
                int(index)
                for site in settings["ion_sites"]
                for index in (
                    [site["ion_atom_index"]]
                    + list(site["candidate_ligand_atom_indices"])
                )
            }))
            for segment in replica["segments"]:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                key = (system_id, replica_id, segment_id)
                segment_rows[key] = []
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                selected_indices = selection_plan[key]
                axis = normalize_segment_axis(
                    segment, str(output_time_unit) if output_time_unit else None
                )
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                periodic_frames = 0
                for raw_frame in iter_coordinate_frames(
                    trajectory_path,
                    coordinate_unit,
                    reader_frame_indices(selected_indices, periodic_policy),
                ):
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        reconstruction_atom_indices,
                    )
                    periodic_frames += int(frame.periodic_cell_present)
                    if not frame_selected(frame.frame_index, selected_indices, 1):
                        continue
                    evaluated += 1
                    if evaluated > int(settings["maximum_frames"]):
                        raise IonGeometryError("maximum_frames gate exceeded")
                    site_reports = []
                    metrics: Dict[str, float] = {}
                    for site in settings["ion_sites"]:
                        site_id = str(site["site_id"])
                        ion_index = int(site["ion_atom_index"])
                        ligand_indices = [int(value) for value in site["candidate_ligand_atom_indices"]]
                        vectors = [
                            _minimum_image_vector(
                                frame.coordinates_angstrom[ion_index],
                                frame.coordinates_angstrom[index],
                                frame.cell_vectors_angstrom,
                            ) for index in ligand_indices
                        ]
                        distances = [float(np.linalg.norm(vector)) for vector in vectors]
                        bound_positions = [
                            position for position, distance in enumerate(distances)
                            if distance <= float(site["coordination_cutoff_angstrom"])
                        ]
                        order = sorted(range(len(distances)), key=lambda position: (distances[position], ligand_indices[position]))
                        bound_order = [position for position in order if position in bound_positions]
                        geometry_scores = []
                        for template in site["geometry_templates"]:
                            expected_count = _IDEAL_PAIR_ANGLES[str(template)][0]
                            if len(bound_order) == expected_count:
                                geometry_scores.append(coordination_geometry_score(
                                    [vectors[position] for position in bound_order], str(template)
                                ))
                        bound_details = [{
                            "atom_index": ligand_indices[position],
                            "atom_identity": _identity(atoms[ligand_indices[position]]),
                            "distance_angstrom": distances[position],
                        } for position in bound_order]
                        bound_distance_summary = (
                            sample_summary([distances[position] for position in bound_order])
                            if bound_order else None
                        )
                        site_reports.append({
                            "site_id": site_id,
                            "ion_atom_index": ion_index,
                            "ion_identity": _identity(atoms[ion_index]),
                            "coordination_cutoff_angstrom": site["coordination_cutoff_angstrom"],
                            "coordination_number": len(bound_order),
                            "nearest_candidate_distance_angstrom": distances[order[0]],
                            "bound_ligands": bound_details,
                            "bound_distance_summary_angstrom": bound_distance_summary,
                            "geometry_scores": geometry_scores,
                        })
                        metrics[f"ion_site:{site_id}:coordination_number"] = float(len(bound_order))
                        metrics[f"ion_site:{site_id}:nearest_candidate_distance_angstrom"] = distances[order[0]]
                    ion_pair_reports = []
                    sites_by_ion_index = {
                        int(site["ion_atom_index"]): site for site in site_reports
                    }
                    for pair in settings["ion_pairs"]:
                        first_index = int(pair["first_ion_atom_index"])
                        second_index = int(pair["second_ion_atom_index"])
                        displacement = _minimum_image_vector(
                            frame.coordinates_angstrom[first_index],
                            frame.coordinates_angstrom[second_index],
                            frame.cell_vectors_angstrom,
                        )
                        distance = float(np.linalg.norm(displacement))
                        pair_id = str(pair["pair_id"])
                        first_site = sites_by_ion_index.get(first_index)
                        second_site = sites_by_ion_index.get(second_index)
                        shared_bound_ligands = []
                        if first_site is not None and second_site is not None:
                            first_bound = {
                                int(row["atom_index"]): row
                                for row in first_site["bound_ligands"]
                            }
                            second_bound = {
                                int(row["atom_index"]): row
                                for row in second_site["bound_ligands"]
                            }
                            for ligand_index in sorted(set(first_bound) & set(second_bound)):
                                ligand_to_first = _minimum_image_vector(
                                    frame.coordinates_angstrom[ligand_index],
                                    frame.coordinates_angstrom[first_index],
                                    frame.cell_vectors_angstrom,
                                )
                                ligand_to_second = _minimum_image_vector(
                                    frame.coordinates_angstrom[ligand_index],
                                    frame.coordinates_angstrom[second_index],
                                    frame.cell_vectors_angstrom,
                                )
                                denominator = float(
                                    np.linalg.norm(ligand_to_first)
                                    * np.linalg.norm(ligand_to_second)
                                )
                                if denominator <= 1.0e-15:
                                    raise IonGeometryError(
                                        f"ion pair {pair_id} has a shared ligand "
                                        "coincident with an ion"
                                    )
                                cosine = float(
                                    np.dot(ligand_to_first, ligand_to_second) / denominator
                                )
                                bridge_angle = math.degrees(
                                    math.acos(min(1.0, max(-1.0, cosine)))
                                )
                                shared_bound_ligands.append({
                                    "atom_index": ligand_index,
                                    "atom_identity": first_bound[ligand_index]["atom_identity"],
                                    "first_ion_distance_angstrom": first_bound[ligand_index]["distance_angstrom"],
                                    "second_ion_distance_angstrom": second_bound[ligand_index]["distance_angstrom"],
                                    "ion_ligand_ion_angle_degrees": bridge_angle,
                                })
                            metrics[
                                f"ion_pair:{pair_id}:shared_bound_ligand_count"
                            ] = float(len(shared_bound_ligands))
                        ion_pair_reports.append({
                            **pair,
                            "distance_angstrom": distance,
                            "bound_site_pair_evaluated": (
                                first_site is not None and second_site is not None
                            ),
                            "shared_bound_ligand_count": len(shared_bound_ligands),
                            "shared_bound_ligands": shared_bound_ligands,
                        })
                        metrics[f"ion_pair:{pair_id}:distance_angstrom"] = distance
                    row = {
                        "system_id": system_id, "replica_id": replica_id,
                        "segment_id": segment_id, "source_frame_index": frame.frame_index,
                        "axis_kind": axis["kind"],
                        "axis_value": frame_axis_value(axis, frame.frame_index),
                        "ion_sites": site_reports, "ion_pairs": ion_pair_reports,
                        "metrics": metrics,
                    }
                    frame_reports.append(row)
                    segment_rows[key].append(row)
                if periodic_frames and periodic_policy == "allow_wrapped_diagnostic":
                    issues.append({
                        "severity": "warning", "code": "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                        "location": f"{system_id}/{replica_id}/{segment_id}",
                        "message": (
                            f"{periodic_frames} periodic frames used exact minimum-image ion "
                            "distances, but surrounding molecular components were not reconstructed"
                        ),
                    })
    if not frame_reports:
        raise IonGeometryError("no frames were evaluated")
    metric_ids = sorted(frame_reports[0]["metrics"])
    replica_reports = []
    replica_keys = sorted({(row["system_id"], row["replica_id"]) for row in frame_reports})
    for system_id, replica_id in replica_keys:
        rows = [
            row for row in frame_reports
            if row["system_id"] == system_id and row["replica_id"] == replica_id
        ]
        midpoint = max(1, len(rows) // 2)
        occupancy: Dict[str, Counter] = defaultdict(Counter)
        occupancy_identities: Dict[Tuple[str, int], Mapping[str, object]] = {}
        geometry_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for row in rows:
            for site in row["ion_sites"]:
                site_id = str(site["site_id"])
                for ligand in site["bound_ligands"]:
                    atom_index = int(ligand["atom_index"])
                    occupancy[site_id][atom_index] += 1
                    occupancy_identities[(site_id, atom_index)] = ligand["atom_identity"]
                for score in site["geometry_scores"]:
                    geometry_values[(site_id, str(score["template"]))].append(
                        float(score["rms_pair_angle_deviation_degrees"])
                    )
        block_indices = np.array_split(np.arange(len(rows)), min(int(settings["block_count"]), len(rows)))
        replica_reports.append({
            "system_id": system_id, "replica_id": replica_id,
            "evaluated_frame_count": len(rows),
            "metric_summaries": {
                metric: sample_summary([float(row["metrics"][metric]) for row in rows])
                for metric in metric_ids
            },
            "late_minus_early_metric_means": {
                metric: float(
                    np.mean([row["metrics"][metric] for row in rows[-midpoint:]])
                    - np.mean([row["metrics"][metric] for row in rows[:midpoint]])
                ) for metric in metric_ids
            },
            "ligand_occupancies": [{
                "site_id": site_id, "ligand_atom_index": atom_index,
                "ligand_identity": occupancy_identities[(site_id, atom_index)],
                "bound_frame_count": count, "bound_fraction": count / len(rows),
            } for site_id in sorted(occupancy) for atom_index, count in sorted(occupancy[site_id].items())],
            "geometry_score_summaries": [{
                "site_id": key[0], "template": key[1],
                "evaluated_frame_count": len(values),
                "rms_pair_angle_deviation_degrees": sample_summary(values),
            } for key, values in sorted(geometry_values.items())],
            "blocks": [{
                "block_id": block_id,
                "frame_count": len(indices),
                "first_source_frame_index": rows[int(indices[0])]["source_frame_index"],
                "last_source_frame_index": rows[int(indices[-1])]["source_frame_index"],
                "metric_means": {
                    metric: float(np.mean([rows[int(index)]["metrics"][metric] for index in indices]))
                    for metric in metric_ids
                },
            } for block_id, indices in enumerate(block_indices, start=1)],
        })
    return {
        "module_id": "ion_coordination_geometry",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings, "topology_reports": topology_reports,
        "evaluated_frame_count": evaluated, "metric_ids": metric_ids,
        "frame_reports": frame_reports, "replica_reports": replica_reports,
        "distribution_reports": _distributions(segment_rows, metric_ids, settings),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Ion sites, ion identities, candidate coordinating atoms, cutoffs, and geometry templates are explicit and project-specific.",
            "Coordination geometry is a rotation-free sorted pair-angle comparison; it does not infer electronic structure, protonation, or binding affinity.",
            "Ideal-polyhedron scores also minimize ligand labeling and proper rotation; ligand-distance variation remains separate from angular shape.",
            "Ion pairs include separation and, when both ions are declared sites, shared-ligand identities, both ion-ligand distances, and ion-ligand-ion bridge angles.",
            "Ligand occupancy and ion-pair distances are replica-resolved and use exact triclinic minimum-image distances when cells are present.",
            "Scott, Freedman-Diaconis, or Rice histograms are calculated independently for every always-defined scalar metric; constant metrics are labeled not estimable.",
            "Pass thresholds and biological interpretation belong in project or publication locks.",
        ],
    }


def _reduce_ion_geometry_reports(
    partials: Sequence[ReplicaPartial[Dict[str, object]]],
    source_context: Dict[str, object],
) -> Dict[str, object]:
    reports = [partial.value for partial in partials]
    first = dict(reports[0])
    for report in reports[1:]:
        for key in ("module_id", "settings", "metric_ids"):
            if report.get(key) != first.get(key):
                raise IonGeometryError(f"replica ion-geometry reports disagree on {key}")
    for key in ("topology_reports", "frame_reports", "replica_reports"):
        first[key] = [row for report in reports for row in report.get(key, [])]
    first["evaluated_frame_count"] = sum(
        int(report.get("evaluated_frame_count", 0)) for report in reports
    )
    maximum = int(first["settings"]["maximum_frames"])  # type: ignore[index]
    if int(first["evaluated_frame_count"]) > maximum:
        raise IonGeometryError(
            "parallel ion-geometry frame count exceeds maximum_frames"
        )
    segment_rows: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    for raw_row in first["frame_reports"]:  # type: ignore[assignment]
        row = raw_row
        key = (
            str(row["system_id"]), str(row["replica_id"]), str(row["segment_id"])
        )
        segment_rows.setdefault(key, []).append(row)
    first["distribution_reports"] = _distributions(
        segment_rows,
        first["metric_ids"],  # type: ignore[arg-type]
        first["settings"],  # type: ignore[arg-type]
    )
    issues = unique_issues(reports)
    first["issues"] = issues
    first["error_count"] = sum(issue.get("severity") == "error" for issue in issues)
    first["warning_count"] = sum(
        issue.get("severity") == "warning" for issue in issues
    )
    restore_source_provenance(first, source_context)
    return first


def ion_coordination_geometry_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Analyze ion sites by replica, then rebuild campaign distributions."""

    return execute_replica_final_module(
        project_path,
        runner_id="ion_geometry",
        hash_content=hash_content,
        reducer=_reduce_ion_geometry_reports,
    )


def ion_coordination_geometry_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return ion_coordination_geometry_project(project_path, hash_content=hash_content)
    except (
        AtomMappingError, CoordinateReadError, IonGeometryError,
        ManifestValidationError, PeriodicReconstructionError,
        ScalarDistributionError, TrajectoryContractError,
        OSError, KeyError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "ion_coordination_geometry", "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "ION_GEOMETRY_INVALID", "message": message,
            } for message in messages],
        }
