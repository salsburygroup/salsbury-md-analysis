"""Replica-resolved intrinsic nucleic-acid ring and stacking geometry."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .dihedrals import DihedralAnalysisError, dihedral_degrees
from .frame_sampling import (
    frame_selected,
    plan_frame_selection,
    reader_frame_indices,
)
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .moments import sample_summary
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError, minimum_image_displacement
from .scalar_distributions import ScalarDistributionError, analyze_scalar_distribution
from .trajectory_contracts import TrajectoryContractError, frame_axis_value, normalize_segment_axis
from .validation import positive_integer


class NucleicAcidGeometryError(ValueError):
    """Raised when a nucleic-acid geometry request is ambiguous or unsafe."""


def fit_plane(points: Sequence[Sequence[float]]) -> Dict[str, object]:
    """Fit a least-squares plane and return intrinsic displacement metrics."""

    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 3:
        raise NucleicAcidGeometryError("plane fitting requires at least three 3D points")
    if not np.isfinite(values).all():
        raise NucleicAcidGeometryError("plane coordinates contain non-finite values")
    centered = values - values.mean(axis=0)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 2 or singular[1] <= 1.0e-12:
        raise NucleicAcidGeometryError("ring atoms do not define a stable plane")
    normal = right[-1]
    normal /= np.linalg.norm(normal)
    signed = centered @ normal
    return {
        "centroid_angstrom": values.mean(axis=0).tolist(),
        "normal": normal.tolist(),
        "signed_displacements_angstrom": signed.tolist(),
        "rms_displacement_angstrom": float(np.sqrt(np.mean(signed * signed))),
        "maximum_absolute_displacement_angstrom": float(np.max(np.abs(signed))),
    }


def planar_departure_degrees(angle: float) -> float:
    """Return signed departure from the nearest 0/180-degree planar torsion."""

    if not math.isfinite(angle):
        raise NucleicAcidGeometryError("ring torsion is non-finite")
    return (float(angle) + 90.0) % 180.0 - 90.0


def ring_geometry(points: Sequence[Sequence[float]]) -> Dict[str, object]:
    """Calculate fitted-plane and cyclic consecutive-torsion ring metrics."""

    if len(points) < 4:
        raise NucleicAcidGeometryError("ring geometry requires at least four atoms")
    plane = fit_plane(points)
    departures = []
    raw = []
    for index in range(len(points)):
        quartet = [points[(index + offset) % len(points)] for offset in range(4)]
        angle = dihedral_degrees(*quartet)
        raw.append(angle)
        departures.append(planar_departure_degrees(angle))
    plane.update({
        "cyclic_dihedral_degrees": raw,
        "signed_planar_departures_degrees": departures,
        "torsion_rms_planar_departure_degrees": float(
            np.sqrt(np.mean(np.square(departures)))
        ),
        "torsion_maximum_absolute_planar_departure_degrees": float(
            np.max(np.abs(departures))
        ),
        "torsion_mean_signed_planar_departure_degrees": float(np.mean(departures)),
    })
    return plane


def unoriented_plane_angle_degrees(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """Return the acute angle between unoriented plane normals."""

    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != (3,) or right.shape != (3,):
        raise NucleicAcidGeometryError("plane normals must be three-dimensional")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-15:
        raise NucleicAcidGeometryError("plane normal has zero length")
    cosine = min(1.0, max(-1.0, abs(float(np.dot(left, right))) / denominator))
    return math.degrees(math.acos(cosine))


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("nucleic_acid_geometry") if isinstance(definitions, dict) else None
    required = {
        "frame_stride", "maximum_frames", "rings", "plane_pairs", "block_count",
        "histogram_rule", "histogram_padding_fraction", "minimum_histogram_bins",
        "maximum_histogram_bins",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise NucleicAcidGeometryError(
            "definitions.nucleic_acid_geometry fields do not match the contract"
        )
    frame_stride = positive_integer(
        raw["frame_stride"], "frame_stride", error_type=NucleicAcidGeometryError
    )
    maximum_frames = positive_integer(
        raw["maximum_frames"], "maximum_frames", error_type=NucleicAcidGeometryError
    )
    block_count = positive_integer(
        raw["block_count"], "block_count", error_type=NucleicAcidGeometryError
    )
    rule = raw["histogram_rule"]
    if rule not in {"scott", "freedman_diaconis", "rice"}:
        raise NucleicAcidGeometryError("histogram_rule must be scott, freedman_diaconis, or rice")
    padding = raw["histogram_padding_fraction"]
    if (
        isinstance(padding, bool) or not isinstance(padding, (int, float))
        or not math.isfinite(float(padding)) or float(padding) < 0.0
    ):
        raise NucleicAcidGeometryError(
            "histogram_padding_fraction must be finite and nonnegative"
        )
    minimum_bins = positive_integer(
        raw["minimum_histogram_bins"], "minimum_histogram_bins",
        error_type=NucleicAcidGeometryError,
    )
    maximum_bins = positive_integer(
        raw["maximum_histogram_bins"], "maximum_histogram_bins",
        error_type=NucleicAcidGeometryError,
    )
    if minimum_bins < 2 or maximum_bins < minimum_bins:
        raise NucleicAcidGeometryError("histogram gates require 2 <= minimum <= maximum")

    rings = raw["rings"]
    if not isinstance(rings, list) or not rings:
        raise NucleicAcidGeometryError("rings must be a nonempty array")
    ring_ids = set()
    normalized_rings = []
    for index, row in enumerate(rings):
        if not isinstance(row, dict) or set(row) != {"ring_id", "atom_indices"}:
            raise NucleicAcidGeometryError(f"ring {index} fields do not match the contract")
        ring_id = str(row["ring_id"]).strip()
        indices = row["atom_indices"]
        if (
            not ring_id or ring_id in ring_ids or not isinstance(indices, list)
            or not 4 <= len(indices) <= 12 or len(set(indices)) != len(indices)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indices)
        ):
            raise NucleicAcidGeometryError(
                "ring IDs must be nonempty/unique and atom_indices must contain 4-12 unique nonnegative integers"
            )
        ring_ids.add(ring_id)
        normalized_rings.append({"ring_id": ring_id, "atom_indices": list(indices)})

    pairs = raw["plane_pairs"]
    if not isinstance(pairs, list):
        raise NucleicAcidGeometryError("plane_pairs must be an array")
    pair_ids = set()
    normalized_pairs = []
    for index, row in enumerate(pairs):
        expected = {"pair_id", "first_ring_id", "second_ring_id", "interpretation"}
        if not isinstance(row, dict) or set(row) != expected:
            raise NucleicAcidGeometryError(f"plane pair {index} fields do not match the contract")
        pair_id = str(row["pair_id"]).strip()
        first = str(row["first_ring_id"]).strip()
        second = str(row["second_ring_id"]).strip()
        interpretation = row["interpretation"]
        if (
            not pair_id or pair_id in pair_ids or first == second
            or first not in ring_ids or second not in ring_ids
            or interpretation not in {"fused_ring_fold", "base_stacking"}
        ):
            raise NucleicAcidGeometryError("plane pair IDs/references/interpretations are invalid")
        pair_ids.add(pair_id)
        normalized_pairs.append({
            "pair_id": pair_id, "first_ring_id": first, "second_ring_id": second,
            "interpretation": interpretation,
        })
    return {
        "frame_stride": frame_stride,
        "maximum_frames": maximum_frames,
        "rings": normalized_rings,
        "plane_pairs": normalized_pairs,
        "block_count": block_count,
        "histogram_rule": rule,
        "histogram_padding_fraction": float(padding),
        "minimum_histogram_bins": minimum_bins,
        "maximum_histogram_bins": maximum_bins,
    }


def _identity(atom: AtomRecord) -> Dict[str, object]:
    return {
        "atom_index": atom.atom_index,
        "atom_name": atom.atom_name,
        "residue_name": atom.residue_name,
        "chain_id": atom.chain_id,
        "residue_number": atom.residue_number,
        "insertion_code": atom.insertion_code,
        "element": atom.element,
    }


def _validate_indices(
    atoms: Sequence[AtomRecord], settings: Mapping[str, object],
    expected: Dict[str, Tuple[Tuple[object, ...], ...]], location: str,
) -> List[Dict[str, object]]:
    identities = []
    for ring in settings["rings"]:  # type: ignore[union-attr]
        ring_id = str(ring["ring_id"])
        indices = [int(value) for value in ring["atom_indices"]]
        if max(indices) >= len(atoms):
            raise NucleicAcidGeometryError(f"{location}: ring {ring_id} atom index exceeds topology")
        keys = tuple(atoms[index].match_key("strict") for index in indices)
        if ring_id in expected and expected[ring_id] != keys:
            raise NucleicAcidGeometryError(
                f"{location}: ring {ring_id} atom identities differ across replicas"
            )
        expected.setdefault(ring_id, keys)
        identities.append({
            "ring_id": ring_id,
            "atom_identities": [_identity(atoms[index]) for index in indices],
        })
    return identities


def _block_reports(rows: Sequence[Mapping[str, object]], block_count: int) -> List[Dict[str, object]]:
    if not rows:
        return []
    blocks = np.array_split(np.arange(len(rows)), min(block_count, len(rows)))
    result = []
    for block_id, block in enumerate(blocks, start=1):
        selected = [rows[int(index)] for index in block]
        metrics = sorted(selected[0]["metrics"])
        result.append({
            "block_id": block_id,
            "frame_count": len(selected),
            "first_source_frame_index": selected[0]["source_frame_index"],
            "last_source_frame_index": selected[-1]["source_frame_index"],
            "metric_means": {
                metric: float(np.mean([row["metrics"][metric] for row in selected]))
                for metric in metrics
            },
        })
    return result


def _distribution_reports(
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
            report = analyze_scalar_distribution(
                segments,
                binning_rule=str(settings["histogram_rule"]),
                padding_fraction=float(settings["histogram_padding_fraction"]),
                minimum_bins=int(settings["minimum_histogram_bins"]),
                maximum_bins=int(settings["maximum_histogram_bins"]),
                retain_assignments=False,
                retain_residence_runs=False,
            )
            reports.append({"metric_id": metric_id, "status": "complete", **report})
        except ScalarDistributionError as exc:
            reports.append({
                "metric_id": metric_id, "status": "not_estimable", "reason": str(exc),
            })
    return reports


def nucleic_acid_geometry_project(
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
        error_type=NucleicAcidGeometryError,
    )
    output_time_unit = project.get("time_unit")
    periodic_policy = str(project["periodic_coordinate_policy"])
    issues = [issue for issue in context.get("issues", []) if isinstance(issue, dict)]
    expected_identities: Dict[str, Tuple[Tuple[object, ...], ...]] = {}
    ring_identity_reports = []
    frame_reports = []
    segment_rows: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    evaluated = 0
    ring_by_id = {str(row["ring_id"]): row for row in settings["rings"]}

    for raw_system in system["systems"]:
        assert isinstance(raw_system, dict)
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            identities = _validate_indices(
                atoms, settings, expected_identities, f"{system_id}/{replica_id}"
            )
            ring_identity_reports.append({
                "system_id": system_id, "replica_id": replica_id,
                "topology_path": str(topology_path), "rings": identities,
            })
            processor = PeriodicFrameProcessor.from_replica(project, replica, system_path, len(atoms))
            reconstruction_atom_indices = tuple(sorted({
                int(index)
                for ring in settings["rings"]
                for index in ring["atom_indices"]
            }))
            for segment in replica["segments"]:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                key = (system_id, replica_id, segment_id)
                segment_rows[key] = []
                trajectory_path = resolve_manifest_path(
                    str(segment["trajectory"]), system_path
                )
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
                    if not frame_selected(
                        frame.frame_index,
                        selected_indices,
                        int(settings["frame_stride"]),
                    ):
                        continue
                    evaluated += 1
                    if evaluated > int(settings["maximum_frames"]):
                        raise NucleicAcidGeometryError("maximum_frames gate exceeded")
                    ring_metrics = {}
                    metrics = {}
                    for ring_id, ring in ring_by_id.items():
                        indices = [int(value) for value in ring["atom_indices"]]
                        geometry = ring_geometry([frame.coordinates_angstrom[index] for index in indices])
                        ring_metrics[ring_id] = geometry
                        metrics[f"ring:{ring_id}:plane_rms_angstrom"] = geometry["rms_displacement_angstrom"]
                        metrics[f"ring:{ring_id}:torsion_rms_degrees"] = geometry["torsion_rms_planar_departure_degrees"]
                        metrics[f"ring:{ring_id}:signed_torsion_mean_degrees"] = geometry["torsion_mean_signed_planar_departure_degrees"]
                    pair_metrics = []
                    for pair in settings["plane_pairs"]:
                        first = ring_metrics[str(pair["first_ring_id"])]
                        second = ring_metrics[str(pair["second_ring_id"])]
                        angle = unoriented_plane_angle_degrees(first["normal"], second["normal"])
                        displacement = np.asarray(second["centroid_angstrom"]) - np.asarray(first["centroid_angstrom"])
                        if frame.cell_vectors_angstrom is not None:
                            displacement = np.asarray(minimum_image_displacement(
                                displacement.tolist(), frame.cell_vectors_angstrom
                            ))
                        distance = float(np.linalg.norm(displacement))
                        pair_id = str(pair["pair_id"])
                        pair_metrics.append({
                            **pair,
                            "plane_angle_degrees": angle,
                            "centroid_distance_angstrom": distance,
                        })
                        metrics[f"plane_pair:{pair_id}:angle_degrees"] = angle
                        metrics[f"plane_pair:{pair_id}:centroid_distance_angstrom"] = distance
                    row = {
                        "system_id": system_id, "replica_id": replica_id,
                        "segment_id": segment_id, "source_frame_index": frame.frame_index,
                        "axis_kind": axis["kind"],
                        "axis_value": frame_axis_value(axis, frame.frame_index),
                        "metrics": metrics,
                    }
                    frame_reports.append(row)
                    segment_rows[key].append(row)
                if periodic_frames and periodic_policy == "allow_wrapped_diagnostic":
                    issues.append({
                        "severity": "warning",
                        "code": "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                        "location": f"{system_id}/{replica_id}/{segment_id}",
                        "message": (
                            f"{periodic_frames} periodic frames were analyzed without "
                            "connectivity-aware reconstruction; intraring values may be invalid"
                        ),
                    })

    if not frame_reports:
        raise NucleicAcidGeometryError("no frames were evaluated")
    metric_ids = sorted(frame_reports[0]["metrics"])
    replica_reports = []
    replica_keys = sorted({(row["system_id"], row["replica_id"]) for row in frame_reports})
    for system_id, replica_id in replica_keys:
        rows = [
            row for row in frame_reports
            if row["system_id"] == system_id and row["replica_id"] == replica_id
        ]
        midpoint = max(1, len(rows) // 2)
        early = rows[:midpoint]
        late = rows[-midpoint:]
        replica_reports.append({
            "system_id": system_id, "replica_id": replica_id,
            "evaluated_frame_count": len(rows),
            "metric_summaries": {
                metric: sample_summary([float(row["metrics"][metric]) for row in rows])
                for metric in metric_ids
            },
            "late_minus_early_metric_means": {
                metric: float(
                    np.mean([row["metrics"][metric] for row in late])
                    - np.mean([row["metrics"][metric] for row in early])
                ) for metric in metric_ids
            },
            "blocks": _block_reports(rows, int(settings["block_count"])),
        })
    distributions = _distribution_reports(segment_rows, metric_ids, settings)
    return {
        "module_id": "nucleic_acid_geometry",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "ring_identity_reports": ring_identity_reports,
        "evaluated_frame_count": evaluated,
        "metric_ids": metric_ids,
        "frame_reports": frame_reports,
        "frame_report_encoding": {
            "schema": "nucleic-acid-geometry-scalars-v2",
            "raw_scalar_metrics_retained_for_every_selected_frame": True,
            "duplicated_ring_fit_vectors_retained": False,
            "duplicated_distribution_assignments_retained": False,
            "duplicated_individual_residence_runs_retained": False,
        },
        "replica_reports": replica_reports,
        "distribution_reports": distributions,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Ring atom order is explicit and cyclic; atom indices and strict identities are retained and checked across replicas.",
            "Fitted-plane displacement, cyclic-torsion departure, fused-ring fold, and base-stacking orientation are distinct metrics and are not substituted for one another.",
            "Automatic histograms are calculated independently per metric using the declared Scott, Freedman-Diaconis, or Rice rule; every selected frame's raw scalar metrics and aggregate boundary-censored residence summaries are retained without duplicating per-metric assignment and individual-run tables.",
            "Pass thresholds, residue identities, lesion definitions, and publication decisions belong in project or publication locks, not this reusable implementation.",
            "Periodic production analysis requires connectivity-aware make_whole or unwrap_continuous reconstruction.",
        ],
    }


def nucleic_acid_geometry_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return nucleic_acid_geometry_project(project_path, hash_content=hash_content)
    except (
        AtomMappingError, CoordinateReadError, DihedralAnalysisError,
        ManifestValidationError, NucleicAcidGeometryError,
        PeriodicReconstructionError, ScalarDistributionError,
        TrajectoryContractError, OSError, KeyError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "nucleic_acid_geometry",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "NUCLEIC_ACID_GEOMETRY_INVALID",
                "message": message,
            } for message in messages],
        }
