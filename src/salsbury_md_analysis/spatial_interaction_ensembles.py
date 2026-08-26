"""Reference-aligned spatial ensembles of chemically typed interactions.

This experimental analysis supplements binary interaction fingerprints by
retaining where an interaction partner was observed after fitting the receptor
to a declared common reference.  It deliberately uses only atom identities
already present in the fingerprint dictionary.  Spatial clusters are reported
as gated mode candidates, never as binding states, free-energy basins, or
metastable states.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from .atom_mapping import AtomMappingError, read_topology_atoms
from .clustering import ClusteringAnalysisError, run_kmeans, silhouette_score
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .geometry import GeometryError, apply_transform, best_fit_transform
from .interaction_fingerprints import (
    InteractionFingerprintError, interaction_fingerprints_project,
)
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .selections import build_common_correspondences
from .upstream_cache import load_cached_project_report
from .validation import positive_integer


class SpatialInteractionEnsembleError(ValueError):
    """Raised when spatial interaction ensembles cannot be built safely."""


FrameKey = Tuple[str, str, str, int]


def _finite(
    value: object, name: str, *, minimum: float | None = None,
    maximum: float | None = None, positive: bool = False,
) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SpatialInteractionEnsembleError(f"{name} must be finite")
    result = float(value)
    if positive and result <= 0.0:
        raise SpatialInteractionEnsembleError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise SpatialInteractionEnsembleError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise SpatialInteractionEnsembleError(f"{name} must be at most {maximum}")
    return result


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("spatial_interaction_ensembles")
        if isinstance(definitions, dict) else None
    )
    required = {
        "source_module", "alignment_selection", "minimum_reference_coverage",
        "point_construction_policy", "minimum_point_observations",
        "minimum_distinct_frames", "time_block_count", "mode_k_values",
        "minimum_mode_observations", "minimum_mode_fraction",
        "minimum_mode_silhouette",
        "minimum_mode_centroid_separation_angstrom",
        "minimum_mode_time_blocks", "minimum_mode_replicas",
        "maximum_superfeatures", "maximum_point_observations",
        "maximum_exact_mode_points", "maximum_mode_iterations",
        "mode_center_tolerance_angstrom",
    }
    if not isinstance(raw, dict):
        raise SpatialInteractionEnsembleError(
            "definitions.spatial_interaction_ensembles must be an object"
        )
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing or unknown:
        raise SpatialInteractionEnsembleError(
            "spatial-interaction settings mismatch; missing=" + ",".join(missing)
            + "; unknown=" + ",".join(unknown)
        )
    if raw["source_module"] != "interaction_fingerprints":
        raise SpatialInteractionEnsembleError(
            "source_module must be interaction_fingerprints"
        )
    if raw["point_construction_policy"] != "endpoint_partner_coordinates_v1":
        raise SpatialInteractionEnsembleError(
            "point_construction_policy must be endpoint_partner_coordinates_v1"
        )
    if not isinstance(raw["alignment_selection"], str) or not str(
        raw["alignment_selection"]
    ).strip():
        raise SpatialInteractionEnsembleError(
            "alignment_selection must be a nonempty selection name"
        )
    result = dict(raw)
    result["minimum_reference_coverage"] = _finite(
        raw["minimum_reference_coverage"], "minimum_reference_coverage",
        minimum=0.0, maximum=1.0,
    )
    for name in (
        "minimum_point_observations", "minimum_distinct_frames",
        "time_block_count", "minimum_mode_observations",
        "minimum_mode_time_blocks", "minimum_mode_replicas",
        "maximum_superfeatures", "maximum_point_observations",
        "maximum_exact_mode_points", "maximum_mode_iterations",
    ):
        result[name] = positive_integer(
            raw[name], name, error_type=SpatialInteractionEnsembleError
        )
    for name in ("minimum_mode_fraction", "minimum_mode_silhouette"):
        result[name] = _finite(raw[name], name, minimum=0.0, maximum=1.0)
    result["minimum_mode_centroid_separation_angstrom"] = _finite(
        raw["minimum_mode_centroid_separation_angstrom"],
        "minimum_mode_centroid_separation_angstrom", positive=True,
    )
    result["mode_center_tolerance_angstrom"] = _finite(
        raw["mode_center_tolerance_angstrom"],
        "mode_center_tolerance_angstrom", positive=True,
    )
    k_values = raw["mode_k_values"]
    if (
        not isinstance(k_values, list) or not k_values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in k_values
        )
        or len(set(k_values)) != len(k_values)
    ):
        raise SpatialInteractionEnsembleError(
            "mode_k_values must contain unique integers of at least two"
        )
    result["mode_k_values"] = tuple(sorted(k_values))
    if int(result["minimum_mode_time_blocks"]) > int(result["time_block_count"]):
        raise SpatialInteractionEnsembleError(
            "minimum_mode_time_blocks cannot exceed time_block_count"
        )
    return result


def _atom_index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpatialInteractionEnsembleError(
            f"fingerprint {name} must be a nonnegative integer"
        )
    return value


def compile_superfeatures(
    feature_dictionary: Sequence[Mapping[str, object]],
    maximum_superfeatures: int,
) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]], List[Dict[str, object]]]:
    """Compile extractable partner-coordinate definitions from fingerprints."""

    compiled: MutableMapping[str, Dict[str, object]] = {}
    feature_to_points: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    unsupported: List[Dict[str, object]] = []
    for feature in feature_dictionary:
        feature_id = feature.get("feature_id")
        interaction_type = feature.get("interaction_type")
        definition = feature.get("definition")
        if not isinstance(feature_id, str) or not isinstance(definition, dict):
            raise SpatialInteractionEnsembleError(
                "fingerprint feature dictionary contains a malformed feature"
            )
        specs: List[Dict[str, object]] = []
        if interaction_type == "direct_hydrogen_bond":
            donor = _atom_index(definition.get("donor_atom_index"), "donor_atom_index")
            acceptor = _atom_index(
                definition.get("acceptor_atom_index"), "acceptor_atom_index"
            )
            specs = [
                {
                    "superfeature_id": (
                        f"direct_hydrogen_bond|donor_atom_{donor}|"
                        "acceptor_partner_position"
                    ),
                    "interaction_type": interaction_type,
                    "anchor_role": "donor", "anchor_atom_index": donor,
                    "point_role": "acceptor", "point_atom_index": acceptor,
                },
                {
                    "superfeature_id": (
                        f"direct_hydrogen_bond|acceptor_atom_{acceptor}|"
                        "donor_partner_position"
                    ),
                    "interaction_type": interaction_type,
                    "anchor_role": "acceptor", "anchor_atom_index": acceptor,
                    "point_role": "donor", "point_atom_index": donor,
                },
            ]
        elif interaction_type == "ion_ligand_coordination":
            ion = _atom_index(definition.get("ion_atom_index"), "ion_atom_index")
            ligand = _atom_index(
                definition.get("ligand_atom_index"), "ligand_atom_index"
            )
            site_id = str(definition.get("site_id", f"ion-atom-{ion}"))
            specs = [{
                "superfeature_id": (
                    f"ion_ligand_coordination|{site_id}|ion_atom_{ion}|"
                    "ligand_partner_position"
                ),
                "interaction_type": interaction_type,
                "anchor_role": "ion_site", "anchor_atom_index": ion,
                "point_role": "ligand", "point_atom_index": ligand,
            }]
        else:
            unsupported.append({
                "feature_id": feature_id,
                "interaction_type": str(interaction_type),
                "reason": "no_exact_dynamic_partner_atom_in_fingerprint_definition",
            })
            continue
        for spec in specs:
            superfeature_id = str(spec["superfeature_id"])
            row = compiled.setdefault(superfeature_id, {
                **spec, "source_module": "interaction_fingerprints",
                "source_feature_ids": [],
                "coordinate_frame": "declared_reference_alignment",
            })
            source_ids = row["source_feature_ids"]
            assert isinstance(source_ids, list)
            source_ids.append(feature_id)
            feature_to_points[feature_id].append({
                "superfeature_id": superfeature_id,
                "point_atom_index": spec["point_atom_index"],
            })
            if len(compiled) > maximum_superfeatures:
                raise SpatialInteractionEnsembleError(
                    "compiled superfeature count exceeds maximum_superfeatures"
                )
    rows = []
    for key in sorted(compiled):
        row = dict(compiled[key])
        row["source_feature_ids"] = sorted(row["source_feature_ids"])
        rows.append(row)
    return rows, dict(feature_to_points), unsupported


def _frame_key(row: Mapping[str, object]) -> FrameKey:
    try:
        key = (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"]), int(row["source_frame_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SpatialInteractionEnsembleError(
            "fingerprint frame lacks exact source-frame identity"
        ) from exc
    if key[3] < 0:
        raise SpatialInteractionEnsembleError(
            "fingerprint source_frame_index must be nonnegative"
        )
    return key


def _coordinates_at(
    coordinates: Sequence[Sequence[float]], indices: Sequence[int]
) -> Tuple[Tuple[float, float, float], ...]:
    try:
        return tuple(
            tuple(float(value) for value in coordinates[index])
            for index in indices
        )  # type: ignore[return-value]
    except IndexError as exc:
        raise SpatialInteractionEnsembleError(
            "fingerprint atom index exceeds trajectory atom count"
        ) from exc


def _assign_time_blocks(
    observations: List[Dict[str, object]], block_count: int
) -> None:
    frames: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for row in observations:
        key = (
            str(row["system_id"]), str(row["replica_id"]), str(row["segment_id"])
        )
        frames[key].append(int(row["source_frame_index"]))
    block_lookup: Dict[Tuple[str, str, str, int], int] = {}
    for key, values in frames.items():
        unique = sorted(set(values))
        for rank, frame in enumerate(unique):
            block_lookup[(*key, frame)] = min(
                block_count - 1, rank * block_count // len(unique)
            )
    for row in observations:
        key = (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"]), int(row["source_frame_index"]),
        )
        row["time_block_index"] = block_lookup[key]


def _spatial_summary(points: np.ndarray) -> Dict[str, object]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    radii = np.linalg.norm(centered, axis=1)
    return {
        "centroid_angstrom": centroid.tolist(),
        "covariance_angstrom2": covariance.tolist(),
        "principal_variances_angstrom2": values.tolist(),
        "principal_axes": vectors.T.tolist(),
        "rms_radius_angstrom": float(math.sqrt(float(np.mean(radii ** 2)))),
        "median_radius_angstrom": float(np.quantile(radii, 0.50)),
        "radius_90_percentile_angstrom": float(np.quantile(radii, 0.90)),
        "radius_95_percentile_angstrom": float(np.quantile(radii, 0.95)),
    }


def _minimum_center_distance(centers: Sequence[Sequence[float]]) -> float:
    return min(
        math.dist(tuple(left), tuple(right))
        for left, right in combinations(centers, 2)
    )


def _mode_candidates(
    rows: Sequence[Mapping[str, object]], settings: Mapping[str, object]
) -> Tuple[str, List[Dict[str, object]], Dict[str, object] | None]:
    if len(rows) > int(settings["maximum_exact_mode_points"]):
        return "withheld_by_exact_mode_resource_gate", [], None
    vectors = [tuple(float(value) for value in row["coordinate_angstrom"]) for row in rows]
    distinct_vectors = len(set(vectors))
    candidates = []
    for k in settings["mode_k_values"]:  # type: ignore[index]
        k = int(k)
        if k > distinct_vectors or len(vectors) < k * int(settings["minimum_mode_observations"]):
            candidates.append({
                "k": k, "gate_status": "insufficient_distinct_or_total_points",
            })
            continue
        fitted = run_kmeans(
            vectors, k, None, int(settings["maximum_mode_iterations"]),
            float(settings["mode_center_tolerance_angstrom"]),
            initialization_method="nani_strat_all", nani_percentage=100,
        )
        if not fitted.get("valid"):
            candidates.append({
                "k": k, "gate_status": "clustering_failed",
                "failure": fitted.get("failure"),
            })
            continue
        assignments = [int(value) for value in fitted["assignments"]]
        centers = [list(map(float, values)) for values in fitted["centers"]]
        silhouette = float(silhouette_score(vectors, assignments))
        modes = []
        for label in range(k):
            indices = [index for index, value in enumerate(assignments) if value == label]
            blocks = sorted({int(rows[index]["time_block_index"]) for index in indices})
            replicas = sorted({str(rows[index]["replica_id"]) for index in indices})
            frames = {
                (
                    str(rows[index]["replica_id"]),
                    str(rows[index]["segment_id"]),
                    int(rows[index]["source_frame_index"]),
                ) for index in indices
            }
            source_features = sorted({
                str(rows[index].get("source_feature_id", "unspecified"))
                for index in indices
            })
            modes.append({
                "mode_index": label, "point_observation_count": len(indices),
                "point_fraction": len(indices) / len(rows),
                "distinct_frame_count": len(frames),
                "centroid_angstrom": centers[label],
                "time_block_indices": blocks, "time_block_count": len(blocks),
                "replica_ids": replicas, "replica_count": len(replicas),
                "source_feature_ids": source_features,
                "source_feature_count": len(source_features),
            })
        gates = {
            "minimum_mode_observations": all(
                row["point_observation_count"] >= int(settings["minimum_mode_observations"])
                for row in modes
            ),
            "minimum_mode_fraction": all(
                row["point_fraction"] >= float(settings["minimum_mode_fraction"])
                for row in modes
            ),
            "minimum_mode_silhouette": (
                silhouette >= float(settings["minimum_mode_silhouette"])
            ),
            "minimum_mode_centroid_separation": (
                _minimum_center_distance(centers)
                >= float(settings["minimum_mode_centroid_separation_angstrom"])
            ),
            "minimum_mode_time_blocks": all(
                row["time_block_count"] >= int(settings["minimum_mode_time_blocks"])
                for row in modes
            ),
            "minimum_mode_replicas": all(
                row["replica_count"] >= int(settings["minimum_mode_replicas"])
                for row in modes
            ),
        }
        passed = all(gates.values())
        candidates.append({
            "k": k, "gate_status": "passed" if passed else "failed",
            "silhouette": silhouette,
            "minimum_centroid_separation_angstrom": _minimum_center_distance(centers),
            "gates": gates, "modes": modes,
            "assignments": assignments,
            "initialization": fitted["initialization"],
        })
    passed = [row for row in candidates if row.get("gate_status") == "passed"]
    selected = max(
        passed, key=lambda row: (float(row["silhouette"]), -int(row["k"])),
        default=None,
    )
    return (
        "gated_multimodal_candidate" if selected is not None
        else "no_gated_multimodal_candidate",
        candidates, selected,
    )


def build_spatial_interaction_ensembles(
    point_observations: Sequence[Mapping[str, object]],
    superfeature_dictionary: Sequence[Mapping[str, object]],
    unsupported_features: Sequence[Mapping[str, object]],
    settings: Mapping[str, object],
) -> Dict[str, object]:
    """Summarize exact aligned partner coordinates and gate spatial modes."""

    rows = [dict(row) for row in point_observations]
    if len(rows) > int(settings["maximum_point_observations"]):
        raise SpatialInteractionEnsembleError(
            "point observation count exceeds maximum_point_observations"
        )
    for row in rows:
        coordinate = row.get("coordinate_angstrom")
        if (
            not isinstance(coordinate, (list, tuple)) or len(coordinate) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) for value in coordinate
            )
        ):
            raise SpatialInteractionEnsembleError(
                "point observation coordinates must be finite three-vectors"
            )
        _frame_key(row)
    _assign_time_blocks(rows, int(settings["time_block_count"]))
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["system_id"]), str(row["superfeature_id"]))].append(row)

    summaries = []
    mode_candidates = []
    selected_modes = []
    for (system_id, superfeature_id), values in sorted(grouped.items()):
        frame_count = len({
            (
                str(row["replica_id"]), str(row["segment_id"]),
                int(row["source_frame_index"]),
            ) for row in values
        })
        summary_gate = (
            len(values) >= int(settings["minimum_point_observations"])
            and frame_count >= int(settings["minimum_distinct_frames"])
        )
        points = np.asarray([row["coordinate_angstrom"] for row in values], dtype=float)
        summary = {
            "system_id": system_id, "superfeature_id": superfeature_id,
            "point_observation_count": len(values),
            "distinct_frame_count": frame_count,
            "replica_count": len({str(row["replica_id"]) for row in values}),
            "time_block_count": len({int(row["time_block_index"]) for row in values}),
            **_spatial_summary(points),
            "spatial_summary_gate": "passed" if summary_gate else "insufficient_observations",
        }
        summaries.append(summary)
        if not summary_gate:
            summary["mode_inference_status"] = "not_evaluated_summary_gate_failed"
            continue
        status, candidates, selected = _mode_candidates(values, settings)
        summary["mode_inference_status"] = status
        for candidate in candidates:
            mode_candidates.append({
                "system_id": system_id, "superfeature_id": superfeature_id,
                **candidate,
            })
        if selected is not None:
            selected_modes.append({
                "system_id": system_id, "superfeature_id": superfeature_id,
                **selected,
            })

    by_feature_system = {
        (str(row["system_id"]), str(row["superfeature_id"])): row
        for row in summaries if row["spatial_summary_gate"] == "passed"
    }
    systems_by_feature: Dict[str, List[str]] = defaultdict(list)
    for system_id, feature_id in by_feature_system:
        systems_by_feature[feature_id].append(system_id)
    selected_lookup = {
        (str(row["system_id"]), str(row["superfeature_id"])): row
        for row in selected_modes
    }
    comparisons = []
    for feature_id, systems in sorted(systems_by_feature.items()):
        for left, right in combinations(sorted(systems), 2):
            left_row = by_feature_system[(left, feature_id)]
            right_row = by_feature_system[(right, feature_id)]
            comparisons.append({
                "superfeature_id": feature_id,
                "system_i": left, "system_j": right,
                "centroid_displacement_angstrom": math.dist(
                    left_row["centroid_angstrom"], right_row["centroid_angstrom"]
                ),
                "rms_radius_difference_angstrom": (
                    float(right_row["rms_radius_angstrom"])
                    - float(left_row["rms_radius_angstrom"])
                ),
                "selected_mode_count_i": (
                    int(selected_lookup[(left, feature_id)]["k"])
                    if (left, feature_id) in selected_lookup else 0
                ),
                "selected_mode_count_j": (
                    int(selected_lookup[(right, feature_id)]["k"])
                    if (right, feature_id) in selected_lookup else 0
                ),
                "evidence_level": "descriptive",
            })
    available = bool(summaries)
    return {
        "availability_status": "available" if available else "not_available",
        "availability_reason": None if available else "no_extractable_spatial_interaction_points",
        "point_construction_policy": "endpoint_partner_coordinates_v1",
        "superfeature_dictionary": [dict(row) for row in superfeature_dictionary],
        "unsupported_fingerprint_features": [dict(row) for row in unsupported_features],
        "point_observations": rows,
        "spatial_ensemble_summaries": summaries,
        "mode_candidates": mode_candidates,
        "selected_spatial_mode_candidates": selected_modes,
        "pairwise_system_spatial_differences": comparisons,
        "interpretation_contract": (
            "A superfeature is the receptor-aligned spatial distribution of an exact "
            "interaction partner atom around one fingerprint-defined anchor. Gated "
            "clusters are spatial mode candidates only, not binding states, free-energy "
            "basins, kinetics, or metastability."
        ),
    }


def spatial_interaction_ensembles_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    fingerprints = load_cached_project_report(
        "interaction_fingerprints", source, hash_content=hash_content,
        error_type=SpatialInteractionEnsembleError,
    )
    if fingerprints is None:
        fingerprints = interaction_fingerprints_project(
            source, hash_content=hash_content
        )
    dictionary = fingerprints.get("feature_dictionary")
    frame_rows = fingerprints.get("frame_fingerprints")
    if not isinstance(dictionary, list) or not isinstance(frame_rows, list):
        raise SpatialInteractionEnsembleError(
            "interaction-fingerprint report lacks dictionary or frame records"
        )
    superfeatures, feature_to_points, unsupported = compile_superfeatures(
        [row for row in dictionary if isinstance(row, dict)],
        int(settings["maximum_superfeatures"]),
    )
    frames: Dict[FrameKey, List[str]] = {}
    for row in frame_rows:
        if not isinstance(row, dict) or not isinstance(row.get("present_feature_ids"), list):
            raise SpatialInteractionEnsembleError("fingerprint frame is malformed")
        present = [
            str(feature_id) for feature_id in row["present_feature_ids"]
            if str(feature_id) in feature_to_points
        ]
        if present:
            key = _frame_key(row)
            if key in frames:
                raise SpatialInteractionEnsembleError(
                    "fingerprint report contains duplicate exact frame identities"
                )
            frames[key] = present

    context = compile_project_context_file(source, hash_content=hash_content)
    contract = context["contract"]
    assert isinstance(contract, dict)
    selections = contract["selections"]
    units = contract["units"]
    assert isinstance(selections, dict) and isinstance(units, dict)
    alignment_definition = selections.get(str(settings["alignment_selection"]))
    if not isinstance(alignment_definition, dict):
        raise SpatialInteractionEnsembleError("alignment selection is undefined")
    coordinate_unit = str(units["coordinates"])

    reference_path = resolve_manifest_path(str(project["reference_structure"]), source)
    _, reference_atoms = read_topology_atoms(reference_path)
    try:
        raw_reference = next(iter_coordinate_frames(reference_path, coordinate_unit))
    except StopIteration as exc:
        raise SpatialInteractionEnsembleError(
            "reference structure contains no coordinates"
        ) from exc
    reference_frame = PeriodicFrameProcessor.from_reference(
        project, source, len(reference_atoms)
    ).process(raw_reference, str(reference_path))

    system_path = Path(str(context["system_manifest_path"]))
    manifest = load_json(system_path)
    topology_rows = []
    for system in manifest["systems"]:
        for replica in system["replicas"]:
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            topology_rows.append((
                str(system["system_id"]), str(replica["replica_id"]), atoms
            ))
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

    observations: List[Dict[str, object]] = []
    read_keys = set()
    for system in manifest["systems"]:
        system_id = str(system["system_id"])
        for replica in system["replicas"]:
            replica_id = str(replica["replica_id"])
            atoms = atoms_by_key[(system_id, replica_id)]
            mapping = mapping_by_key[(system_id, replica_id)]
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(atoms)
            )
            for segment in replica["segments"]:
                segment_id = str(segment["segment_id"])
                selected = sorted(
                    key[3] for key in frames
                    if key[:3] == (system_id, replica_id, segment_id)
                )
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                if not selected:
                    continue
                point_indices = sorted({
                    int(spec["point_atom_index"])
                    for frame_index in selected
                    for feature_id in frames[(system_id, replica_id, segment_id, frame_index)]
                    for spec in feature_to_points[feature_id]
                })
                required = tuple(sorted(set(mapping.target_indices) | set(point_indices)))
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                selected_set = set(selected)
                for raw_frame in iter_coordinate_frames(trajectory_path, coordinate_unit):
                    frame_index = int(raw_frame.frame_index)
                    is_selected = frame_index in selected_set
                    if not is_selected and processor.policy != "unwrap_continuous":
                        continue
                    frame = processor.process(
                        raw_frame,
                        f"{system_id}/{replica_id}/{segment_id}/frame-{frame_index}",
                        required,
                    )
                    if not is_selected:
                        continue
                    key = (system_id, replica_id, segment_id, frame_index)
                    read_keys.add(key)
                    transform = best_fit_transform(
                        _coordinates_at(frame.coordinates_angstrom, mapping.target_indices),
                        reference_alignment,
                    )
                    aligned_points = {
                        index: apply_transform(
                            [frame.coordinates_angstrom[index]], transform
                        )[0] for index in point_indices
                    }
                    for feature_id in frames[key]:
                        for spec in feature_to_points[feature_id]:
                            observations.append({
                                "system_id": system_id, "replica_id": replica_id,
                                "segment_id": segment_id,
                                "source_frame_index": frame_index,
                                "source_feature_id": feature_id,
                                "superfeature_id": spec["superfeature_id"],
                                "point_atom_index": spec["point_atom_index"],
                                "coordinate_angstrom": list(
                                    aligned_points[int(spec["point_atom_index"])]
                                ),
                            })
                            if len(observations) > int(settings["maximum_point_observations"]):
                                raise SpatialInteractionEnsembleError(
                                    "maximum_point_observations gate exceeded"
                                )
    missing = sorted(set(frames).difference(read_keys))
    if missing:
        raise SpatialInteractionEnsembleError(
            f"{len(missing)} fingerprint frames could not be joined to coordinates"
        )
    result = build_spatial_interaction_ensembles(
        observations, superfeatures, unsupported, settings
    )
    issues = [issue for issue in context.get("issues", []) if isinstance(issue, dict)]
    if result["availability_status"] == "not_available":
        issues.append({
            "severity": "warning", "code": "SPATIAL_INTERACTIONS_NOT_AVAILABLE",
            "message": str(result["availability_reason"]),
        })
    return {
        "module_id": "spatial_interaction_ensembles",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": context["system_manifest_path"],
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "source_module": "interaction_fingerprints", "settings": settings, **result,
        "evaluated_frame_count": len(read_keys),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Only direct-hydrogen-bond and ion-coordination fingerprints currently expose exact partner atoms; unsupported types remain explicitly listed.",
            "Reference alignment can blur or split spatial clouds and requires selection and coverage sensitivity analysis.",
            "Multiple point observations in one frame are not independent samples.",
            "Spatial mode candidates are descriptive geometry, not binding states, free-energy basins, kinetics, metastability, affinity, causality, or mechanism.",
            "Time-block and replica support gates test recurrence; they do not establish convergence or inferential significance.",
        ],
    }


def spatial_interaction_ensembles_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return spatial_interaction_ensembles_project(
            project_path, hash_content=hash_content
        )
    except (
        SpatialInteractionEnsembleError, InteractionFingerprintError,
        ManifestValidationError, AtomMappingError, CoordinateReadError,
        PeriodicReconstructionError, GeometryError, ClusteringAnalysisError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "spatial_interaction_ensembles",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "SPATIAL_INTERACTIONS_INVALID",
                "message": message,
            } for message in messages],
        }
