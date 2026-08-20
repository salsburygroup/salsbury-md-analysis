"""Replica-resolved radial distribution functions with periodic-volume normalization."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .atom_mapping import AtomMappingError, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CellVectors, CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .periodic import (
    PeriodicFrameProcessor,
    PeriodicReconstructionError,
    minimum_image_displacement,
)
from .trajectory_contracts import TrajectoryContractError
from .validation import positive_integer


class RDFAnalysisError(ValueError):
    """Raised when a radial distribution function is not safely defined."""


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _cross(first: Sequence[float], second: Sequence[float]) -> Tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(first, second))


def periodic_cell_geometry(cell: CellVectors) -> Tuple[float, float]:
    """Return volume and the largest safe spherical radius for a triclinic cell."""

    a, b, c = cell
    volume = abs(_dot(a, _cross(b, c)))
    face_areas = (_norm(_cross(b, c)), _norm(_cross(a, c)), _norm(_cross(a, b)))
    if volume <= 0.0 or min(face_areas) <= 0.0:
        raise RDFAnalysisError("periodic cell is degenerate")
    minimum_face_height = min(volume / area for area in face_areas)
    return volume, 0.5 * minimum_face_height


def _pairs(group_a: Sequence[int], group_b: Sequence[int]) -> List[Tuple[int, int]]:
    pairs = {
        tuple(sorted((left, right)))
        for left in group_a for right in group_b if left != right
    }
    if not pairs:
        raise RDFAnalysisError("RDF selections produce no distinct atom pairs")
    return sorted(pairs)


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("radial_distribution_functions") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise RDFAnalysisError(
            "definitions.radial_distribution_functions must be an object"
        )
    required = {"frame_stride", "maximum_observations", "features"}
    if set(raw).difference({"frame_selection"}) != required:
        raise RDFAnalysisError(
            "radial_distribution_functions must contain frame_stride, maximum_observations, and features"
        )
    frame_stride = positive_integer(raw["frame_stride"], "frame_stride", error_type=RDFAnalysisError)
    maximum = positive_integer(
        raw["maximum_observations"], "maximum_observations", error_type=RDFAnalysisError
    )
    features = raw["features"]
    if not isinstance(features, list) or not features:
        raise RDFAnalysisError("RDF features must be a nonempty array")
    normalized = []
    identifiers = set()
    expected = {
        "feature_id", "question", "group_a_atom_indices", "group_b_atom_indices",
        "minimum_radius_angstrom", "maximum_radius_angstrom", "bin_width_angstrom",
    }
    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or set(feature) != expected:
            raise RDFAnalysisError(f"RDF feature {index} fields do not match the contract")
        feature_id = str(feature["feature_id"]).strip()
        question = str(feature["question"]).strip()
        if not feature_id or feature_id in identifiers or not question:
            raise RDFAnalysisError("RDF feature IDs/questions must be nonempty and IDs unique")
        for field in ("group_a_atom_indices", "group_b_atom_indices"):
            values = feature[field]
            if (
                not isinstance(values, list) or not values
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
                or len(set(values)) != len(values)
            ):
                raise RDFAnalysisError(f"{feature_id} {field} must contain unique nonnegative indices")
        minimum = feature["minimum_radius_angstrom"]
        maximum_radius = feature["maximum_radius_angstrom"]
        width = feature["bin_width_angstrom"]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (minimum, maximum_radius, width)
        ):
            raise RDFAnalysisError("RDF radii and bin width must be finite numbers")
        minimum = float(minimum)
        maximum_radius = float(maximum_radius)
        width = float(width)
        if minimum < 0.0 or maximum_radius <= minimum or width <= 0.0:
            raise RDFAnalysisError("RDF radius interval and bin width are invalid")
        raw_bin_count = (maximum_radius - minimum) / width
        bin_count = round(raw_bin_count)
        if bin_count < 1 or not math.isclose(raw_bin_count, bin_count, rel_tol=1e-10, abs_tol=1e-10):
            raise RDFAnalysisError(
                "RDF radius span must be an integer multiple of bin_width_angstrom"
            )
        identifiers.add(feature_id)
        normalized.append({
            **feature,
            "feature_id": feature_id,
            "question": question,
            "minimum_radius_angstrom": minimum,
            "maximum_radius_angstrom": maximum_radius,
            "bin_width_angstrom": width,
            "bin_count": bin_count,
        })
    return {
        "frame_stride": frame_stride,
        "frame_selection": normalize_frame_selection(
            raw.get("frame_selection"), frame_stride,
            error_type=RDFAnalysisError,
        ),
        "maximum_observations": maximum,
        "features": normalized,
    }


def radial_distribution_functions_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(str(context["system_manifest_path"]))
    system = load_json(system_path)
    coordinate_unit = str(project["coordinate_unit"])
    frame_selection_plan, frame_selection_report = plan_frame_selection(
        system, system_path, coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        error_type=RDFAnalysisError,
    )
    accumulators: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    segment_reports = []
    observation_count = 0
    for raw_system in system["systems"]:
        assert isinstance(raw_system, dict)
        system_id = str(raw_system["system_id"])
        for replica in raw_system["replicas"]:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            feature_pairs: Dict[str, List[Tuple[int, int]]] = {}
            for feature in settings["features"]:
                indices = [
                    *feature["group_a_atom_indices"],
                    *feature["group_b_atom_indices"],
                ]
                if max(indices) >= len(atoms):
                    raise RDFAnalysisError(
                        f"RDF feature {feature['feature_id']} exceeds topology atom count"
                    )
                feature_pairs[str(feature["feature_id"])] = _pairs(
                    feature["group_a_atom_indices"], feature["group_b_atom_indices"]
                )
                accumulators[(system_id, replica_id, str(feature["feature_id"]))] = {
                    "observed": [0] * int(feature["bin_count"]),
                    "expected": [0.0] * int(feature["bin_count"]),
                    "evaluated_frame_count": 0,
                    "cell_volume_sum_angstrom3": 0.0,
                }
            processor = PeriodicFrameProcessor.from_replica(
                project, replica, system_path, len(atoms)
            )
            reconstruction_atom_indices = tuple(sorted({
                int(index)
                for feature in settings["features"]
                for group in (
                    feature["group_a_atom_indices"],
                    feature["group_b_atom_indices"],
                )
                for index in group
            }))
            for segment in replica["segments"]:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                selected_indices = frame_selection_plan[(
                    system_id, replica_id, segment_id,
                )]
                processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                evaluated_frames = 0
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
                    if not selected:
                        continue
                    if frame.cell_vectors_angstrom is None:
                        raise RDFAnalysisError(
                            "RDF normalization requires a periodic cell on every evaluated frame"
                        )
                    volume, safe_radius = periodic_cell_geometry(frame.cell_vectors_angstrom)
                    evaluated_frames += 1
                    for feature in settings["features"]:
                        maximum_radius = float(feature["maximum_radius_angstrom"])
                        if maximum_radius > safe_radius + 1.0e-10:
                            raise RDFAnalysisError(
                                f"RDF feature {feature['feature_id']} maximum radius {maximum_radius} exceeds safe half-cell radius {safe_radius}"
                            )
                        key = (system_id, replica_id, str(feature["feature_id"]))
                        accumulator = accumulators[key]
                        pairs = feature_pairs[str(feature["feature_id"])]
                        minimum = float(feature["minimum_radius_angstrom"])
                        width = float(feature["bin_width_angstrom"])
                        bin_count = int(feature["bin_count"])
                        for left, right in pairs:
                            displacement = tuple(
                                frame.coordinates_angstrom[right][axis]
                                - frame.coordinates_angstrom[left][axis]
                                for axis in range(3)
                            )
                            displacement = minimum_image_displacement(
                                displacement, frame.cell_vectors_angstrom
                            )
                            distance = _norm(displacement)
                            if minimum <= distance < maximum_radius:
                                bin_index = min(
                                    bin_count - 1, int((distance - minimum) / width)
                                )
                                accumulator["observed"][bin_index] += 1  # type: ignore[index]
                        for bin_index in range(bin_count):
                            lower = minimum + bin_index * width
                            upper = lower + width
                            shell_volume = 4.0 * math.pi * (upper ** 3 - lower ** 3) / 3.0
                            accumulator["expected"][bin_index] += (  # type: ignore[index]
                                len(pairs) * shell_volume / volume
                            )
                        accumulator["evaluated_frame_count"] += 1  # type: ignore[operator]
                        accumulator["cell_volume_sum_angstrom3"] += volume  # type: ignore[operator]
                        observation_count += 1
                        if observation_count > int(settings["maximum_observations"]):
                            raise RDFAnalysisError("maximum_observations gate exceeded")
                segment_reports.append({
                    "system_id": system_id,
                    "replica_id": replica_id,
                    "segment_id": segment_id,
                    "evaluated_frame_count": evaluated_frames,
                })
    feature_by_id = {
        str(feature["feature_id"]): feature for feature in settings["features"]
    }
    reports = []
    for key in sorted(accumulators):
        accumulator = accumulators[key]
        feature = feature_by_id[key[2]]
        observed = accumulator["observed"]
        expected = accumulator["expected"]
        assert isinstance(observed, list) and isinstance(expected, list)
        bins = []
        for bin_index, (count, expected_count) in enumerate(zip(observed, expected)):
            lower = float(feature["minimum_radius_angstrom"]) + bin_index * float(
                feature["bin_width_angstrom"]
            )
            upper = lower + float(feature["bin_width_angstrom"])
            bins.append({
                "bin_index": bin_index,
                "lower_radius_angstrom": lower,
                "upper_radius_angstrom": upper,
                "center_radius_angstrom": 0.5 * (lower + upper),
                "observed_pair_count": count,
                "uniform_expected_pair_count": expected_count,
                "g_r": count / expected_count if expected_count > 0.0 else None,
            })
        frame_count = int(accumulator["evaluated_frame_count"])
        reports.append({
            "system_id": key[0],
            "replica_id": key[1],
            "feature_id": key[2],
            "question": feature["question"],
            "pair_count": len(_pairs(
                feature["group_a_atom_indices"], feature["group_b_atom_indices"]
            )),
            "evaluated_frame_count": frame_count,
            "mean_cell_volume_angstrom3": (
                float(accumulator["cell_volume_sum_angstrom3"]) / frame_count
                if frame_count else None
            ),
            "bins": bins,
        })
    issues = [
        issue for issue in context.get("issues", []) if isinstance(issue, dict)
    ]
    if int(frame_selection_report["selected_frame_count"]) < int(
        frame_selection_report["source_frame_count"]
    ):
        issues.append({
            "severity": "warning", "code": "FRAME_SUBSAMPLING",
            "location": str(source),
            "message": (
                f"RDF evaluated {frame_selection_report['selected_frame_count']} of "
                f"{frame_selection_report['source_frame_count']} source frames under "
                f"{frame_selection_report['mode']}"
            ),
        })
    return {
        "module_id": "radial_distribution_functions",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "frame_selection": frame_selection_report,
        "normalization_contract": (
            "observed pair count divided by fixed-pair uniform expectation "
            "sum_frames(pair_count * spherical_shell_volume / triclinic_cell_volume)"
        ),
        "segment_reports": segment_reports,
        "observation_count": observation_count,
        "feature_reports": reports,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "RDF selections, radial range, bin width, stride, and sampling mode require sensitivity and scientific justification.",
            "Every evaluated frame must have a valid periodic cell, and the maximum radius may not exceed half the minimum triclinic face height.",
            "Replica RDFs remain separate; pooling and uncertainty require an explicit independent-unit model.",
            "An RDF peak does not by itself establish a specific interaction, binding mode, or mechanism.",
        ],
    }


def radial_distribution_functions_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return radial_distribution_functions_project(
            project_path, hash_content=hash_content
        )
    except (
        AtomMappingError,
        CoordinateReadError,
        ManifestValidationError,
        PeriodicReconstructionError,
        RDFAnalysisError,
        TrajectoryContractError,
        OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "radial_distribution_functions",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "RDF_INVALID", "message": message}
                for message in messages
            ],
        }
