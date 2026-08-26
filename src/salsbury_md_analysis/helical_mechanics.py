"""DSSR-gated duplex helical mechanics from frame-aligned step descriptors."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .context import compile_project_context_file
from .manifests import ManifestValidationError, load_json
from .upstream_cache import load_cached_project_report
from .validation import positive_integer


class HelicalMechanicsError(ValueError):
    """Raised when duplex mechanics cannot be evaluated safely."""


_COMPONENTS = ("shift", "slide", "rise", "tilt", "roll", "twist")
_KB_KCAL_PER_MOL_K = 0.00198720425864083


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("helical_mechanics") if isinstance(definitions, dict) else None
    required = {
        "source_module", "duplex_collection_field", "descriptor_query_ids",
        "angular_input_unit", "minimum_frames_per_step",
        "minimum_frames_per_state", "maximum_states",
        "minimum_silhouette_for_state_split", "covariance_eigenvalue_floor_fraction",
        "maximum_steps", "preparation_availability",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise HelicalMechanicsError(
            "definitions.helical_mechanics fields do not match the contract"
        )
    if raw["source_module"] != "nucleic_acid_structure":
        raise HelicalMechanicsError("source_module must be nucleic_acid_structure")
    collection = raw["duplex_collection_field"]
    if collection != "stems":
        raise HelicalMechanicsError(
            "duplex_collection_field must be stems; generic DSSR helices can include nonduplex assemblies"
        )
    queries = raw["descriptor_query_ids"]
    if not isinstance(queries, dict) or set(queries) != set(_COMPONENTS) or any(
        not isinstance(value, str) or not value for value in queries.values()
    ):
        raise HelicalMechanicsError(
            "descriptor_query_ids must map all six helical components to query IDs"
        )
    if raw["angular_input_unit"] != "degrees":
        raise HelicalMechanicsError("angular_input_unit must be degrees")
    silhouette = raw["minimum_silhouette_for_state_split"]
    floor = raw["covariance_eigenvalue_floor_fraction"]
    for value, name, lower, upper in (
        (silhouette, "minimum_silhouette_for_state_split", -1.0, 1.0),
        (floor, "covariance_eigenvalue_floor_fraction", 0.0, 1.0),
    ):
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not lower < float(value) < upper
        ):
            raise HelicalMechanicsError(f"{name} is outside its finite open interval")
    availability = raw["preparation_availability"]
    if not isinstance(availability, dict):
        raise HelicalMechanicsError("preparation_availability must be an object")
    result = dict(raw)
    result["descriptor_query_ids"] = dict(queries)
    result["preparation_availability"] = dict(availability)
    for name in (
        "minimum_frames_per_step", "minimum_frames_per_state",
        "maximum_states", "maximum_steps",
    ):
        result[name] = positive_integer(raw[name], name, error_type=HelicalMechanicsError)
    if int(result["maximum_states"]) > 3:
        raise HelicalMechanicsError("maximum_states cannot exceed 3")
    result["minimum_silhouette_for_state_split"] = float(silhouette)
    result["covariance_eigenvalue_floor_fraction"] = float(floor)
    return result


def _not_available(reason: str, details: Mapping[str, object] | None = None) -> Dict[str, object]:
    return {
        "availability_status": "not_available",
        "availability_reason": reason,
        "availability_details": dict(details or {}),
        "analysis_performed": False,
        "step_state_models": [], "neighbor_step_couplings": [],
        "evaluated_frame_count": 0,
    }


def _deterministic_kmeans(values: np.ndarray, k: int) -> np.ndarray:
    """Small deterministic Euclidean K-means used only for state gating."""

    centers = [values[np.lexsort(values.T[::-1])[0]].copy()]
    while len(centers) < k:
        distances = np.min(
            np.stack([np.sum((values - center) ** 2, axis=1) for center in centers]),
            axis=0,
        )
        centers.append(values[int(np.argmax(distances))].copy())
    center_array = np.asarray(centers, dtype=float)
    labels = np.zeros(values.shape[0], dtype=int)
    for _ in range(100):
        distances = np.stack([
            np.sum((values - center) ** 2, axis=1) for center in center_array
        ], axis=1)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels) and _ > 0:
            break
        labels = updated
        for state in range(k):
            members = values[labels == state]
            if members.size == 0:
                return np.zeros(values.shape[0], dtype=int)
            center_array[state] = np.mean(members, axis=0)
    # Canonicalize state IDs by lexicographic center order.
    order = sorted(range(k), key=lambda index: tuple(center_array[index].tolist()))
    remap = {old: new for new, old in enumerate(order)}
    return np.asarray([remap[int(value)] for value in labels], dtype=int)


def _silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    unique = sorted(set(int(value) for value in labels.tolist()))
    if len(unique) < 2:
        return -1.0
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    scores = []
    for index, label in enumerate(labels):
        same = np.where(labels == label)[0]
        if len(same) <= 1:
            scores.append(0.0)
            continue
        a = float(np.sum(distances[index, same]) / (len(same) - 1))
        b = min(
            float(np.mean(distances[index, np.where(labels == other)[0]]))
            for other in unique if other != int(label)
        )
        scores.append((b - a) / max(a, b) if max(a, b) > 0.0 else 0.0)
    return float(np.mean(scores))


def _select_states(values: np.ndarray, settings: Mapping[str, object]) -> Tuple[np.ndarray, float | None]:
    means = np.mean(values, axis=0)
    scales = np.std(values, axis=0)
    scales[scales <= 1.0e-15] = 1.0
    standardized = (values - means) / scales
    selected = np.zeros(values.shape[0], dtype=int)
    selected_score = None
    for k in range(2, int(settings["maximum_states"]) + 1):
        labels = _deterministic_kmeans(standardized, k)
        counts = Counter(int(value) for value in labels.tolist())
        if len(counts) != k or min(counts.values()) < int(settings["minimum_frames_per_state"]):
            continue
        score = _silhouette(standardized, labels)
        if (
            score >= float(settings["minimum_silhouette_for_state_split"])
            and (selected_score is None or score > selected_score)
        ):
            selected = labels
            selected_score = score
    return selected, selected_score


def _covariance_model(
    values: np.ndarray, *, temperature_kelvin: float, floor_fraction: float,
) -> Dict[str, object]:
    mean = np.mean(values, axis=0)
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance).astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(float(np.max(eigenvalues)), 1.0e-15)
    floor = largest * floor_fraction
    regularized = np.maximum(eigenvalues, floor)
    inverse = eigenvectors @ np.diag(1.0 / regularized) @ eigenvectors.T
    stiffness = (_KB_KCAL_PER_MOL_K * temperature_kelvin) * inverse
    positive = eigenvalues[eigenvalues > largest * 1.0e-12]
    return {
        "observation_count": int(values.shape[0]),
        "mean_vector": mean.tolist(),
        "covariance_matrix": covariance.tolist(),
        "raw_covariance_eigenvalues": eigenvalues.tolist(),
        "raw_covariance_rank": int(np.linalg.matrix_rank(covariance)),
        "raw_covariance_condition_number": (
            float(np.max(positive) / np.min(positive)) if positive.size else None
        ),
        "eigenvalue_floor": floor,
        "regularized_eigenvalues": regularized.tolist(),
        "stiffness_matrix_kcal_per_mol_mixed_coordinates": stiffness.tolist(),
        "stiffness_contract": "K = k_B T C_regularized^-1",
    }


def _query_values(frame: Mapping[str, object], query_ids: Mapping[str, str]) -> Dict[str, list[float]]:
    rows = frame.get("numeric_queries")
    if not isinstance(rows, list):
        raise HelicalMechanicsError("DSSR frame lacks numeric_queries")
    by_id = {
        str(row.get("query_id")): row.get("values")
        for row in rows if isinstance(row, dict)
    }
    result = {}
    for component in _COMPONENTS:
        values = by_id.get(query_ids[component])
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) for value in values
        ):
            raise HelicalMechanicsError(
                f"DSSR frame lacks finite query {query_ids[component]!r}"
            )
        result[component] = [float(value) for value in values]
    lengths = {len(value) for value in result.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 1:
        raise HelicalMechanicsError(
            "six DSSR helical descriptor arrays must have one equal positive step count"
        )
    return result


def build_helical_mechanics(
    dssr_report: Mapping[str, object] | None,
    settings: Mapping[str, object],
    *, temperature_kelvin: float,
) -> Dict[str, object]:
    """Build state-gated step stiffness and adjacent-step coupling models."""

    preparation = settings.get("preparation_availability")
    if not isinstance(preparation, dict) or preparation.get("status") != "available":
        reason = str(
            preparation.get("reason", "dssr_or_duplex_unavailable")
            if isinstance(preparation, dict) else "dssr_or_duplex_unavailable"
        )
        return _not_available(reason, preparation if isinstance(preparation, dict) else None)
    if dssr_report is None:
        return _not_available("dssr_report_unavailable")
    if (
        dssr_report.get("module_id") != "nucleic_acid_structure"
        or dssr_report.get("technical_status") != "complete"
    ):
        raise HelicalMechanicsError("DSSR source report is not technically complete")
    implementation = dssr_report.get("implementation")
    if not isinstance(implementation, dict) or not implementation.get("executable_path"):
        return _not_available("dssr_executable_provenance_unavailable")
    frames = dssr_report.get("frame_reports")
    if not isinstance(frames, list) or not frames:
        return _not_available("dssr_report_contains_no_frames")
    duplex_field = str(settings["duplex_collection_field"])
    duplex_frames = [
        frame for frame in frames
        if isinstance(frame, dict)
        and isinstance(frame.get("collection_counts"), dict)
        and int(frame["collection_counts"].get(duplex_field, 0)) > 0
    ]
    if not duplex_frames:
        return _not_available("no_duplex_dna_or_rna_detected_by_dssr")

    grouped: Dict[Tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for frame in duplex_frames:
        grouped[(str(frame["system_id"]), str(frame["replica_id"]))].append(frame)
    models = []
    labels_by_group_step: Dict[Tuple[str, str, int], Dict[Tuple[str, int], int]] = {}
    evaluated_keys = set()
    for (system_id, replica_id), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (
            str(row["segment_id"]), int(row["source_frame_index"])
        ))
        parsed = [_query_values(row, settings["descriptor_query_ids"]) for row in rows]
        step_counts = {len(row["shift"]) for row in parsed}
        if len(step_counts) != 1:
            return _not_available(
                "duplex_step_count_changes_across_frames",
                {"system_id": system_id, "replica_id": replica_id},
            )
        step_count = next(iter(step_counts))
        if step_count > int(settings["maximum_steps"]):
            raise HelicalMechanicsError(
                f"DSSR step count {step_count} exceeds maximum_steps"
            )
        frame_keys = [
            (str(row["segment_id"]), int(row["source_frame_index"])) for row in rows
        ]
        evaluated_keys.update((system_id, replica_id, *key) for key in frame_keys)
        for step_index in range(step_count):
            matrix = np.asarray([
                [parsed_row[component][step_index] for component in _COMPONENTS]
                for parsed_row in parsed
            ], dtype=float)
            matrix[:, 3:] = np.deg2rad(matrix[:, 3:])
            if matrix.shape[0] < int(settings["minimum_frames_per_step"]):
                continue
            labels, silhouette = _select_states(matrix, settings)
            labels_by_group_step[(system_id, replica_id, step_index)] = {
                key: int(label) for key, label in zip(frame_keys, labels.tolist())
            }
            states = []
            for state_id in sorted(set(int(value) for value in labels.tolist())):
                members = matrix[labels == state_id]
                if members.shape[0] < int(settings["minimum_frames_per_state"]):
                    continue
                states.append({
                    "state_id": state_id,
                    "population_fraction": float(members.shape[0] / matrix.shape[0]),
                    **_covariance_model(
                        members, temperature_kelvin=temperature_kelvin,
                        floor_fraction=float(settings["covariance_eigenvalue_floor_fraction"]),
                    ),
                })
            models.append({
                "system_id": system_id, "replica_id": replica_id,
                "dssr_step_order_index": step_index,
                "component_order": list(_COMPONENTS),
                "component_units": ["angstrom", "angstrom", "angstrom", "radian", "radian", "radian"],
                "evaluated_frame_count": int(matrix.shape[0]),
                "selected_state_count": len(states),
                "selected_split_silhouette": silhouette,
                "states": states,
            })

    couplings = []
    group_steps: Dict[Tuple[str, str], list[int]] = defaultdict(list)
    for system_id, replica_id, step in labels_by_group_step:
        group_steps[(system_id, replica_id)].append(step)
    for (system_id, replica_id), steps in sorted(group_steps.items()):
        for left, right in zip(sorted(set(steps)), sorted(set(steps))[1:]):
            if right != left + 1:
                continue
            left_labels = labels_by_group_step[(system_id, replica_id, left)]
            right_labels = labels_by_group_step[(system_id, replica_id, right)]
            common = sorted(set(left_labels).intersection(right_labels))
            if not common:
                continue
            joint = Counter((left_labels[key], right_labels[key]) for key in common)
            left_counts = Counter(left_labels[key] for key in common)
            right_counts = Counter(right_labels[key] for key in common)
            mutual_information = 0.0
            for (left_state, right_state), count in joint.items():
                probability = count / len(common)
                mutual_information += probability * math.log2(
                    count * len(common) / (left_counts[left_state] * right_counts[right_state])
                )
            couplings.append({
                "system_id": system_id, "replica_id": replica_id,
                "step_i": left, "step_j": right,
                "paired_frame_count": len(common),
                "mutual_information_bits": mutual_information,
                "joint_state_counts": [
                    {"state_i": key[0], "state_j": key[1], "count": value}
                    for key, value in sorted(joint.items())
                ],
            })

    if not models:
        return _not_available(
            "insufficient_duplex_frames_for_helical_mechanics",
            {"duplex_frame_count": len(duplex_frames)},
        )
    return {
        "availability_status": "available", "availability_reason": None,
        "availability_details": {
            "dssr_executable_path": implementation["executable_path"],
            "dssr_version_output": implementation.get("version_output"),
            "duplex_detection_collection": duplex_field,
        },
        "analysis_performed": True,
        "evaluated_frame_count": len(evaluated_keys),
        "step_state_models": models,
        "neighbor_step_couplings": couplings,
        "step_identity_contract": (
            "dssr_step_order_index is accepted only when all six descriptor arrays "
            "have one stable equal length within a system/replica; changing step counts "
            "make the module unavailable rather than silently realigning positions"
        ),
    }


def helical_mechanics_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    dssr = load_cached_project_report(
        "nucleic_acid_structure", source, hash_content=hash_content,
        error_type=HelicalMechanicsError,
    )
    temperature = project.get("temperature_kelvin")
    if (
        isinstance(temperature, bool) or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature)) or float(temperature) <= 0.0
    ):
        raise HelicalMechanicsError("temperature_kelvin must be finite and positive")
    context = compile_project_context_file(source, hash_content=hash_content)
    result = build_helical_mechanics(
        dssr, settings, temperature_kelvin=float(temperature)
    )
    issues = [issue for issue in context.get("issues", []) if isinstance(issue, dict)]
    if result["availability_status"] == "not_available":
        issues.append({
            "severity": "warning", "code": "HELICAL_MECHANICS_NOT_AVAILABLE",
            "message": str(result["availability_reason"]),
        })
    return {
        "module_id": "helical_mechanics",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": context["system_manifest_path"],
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings, **result,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "The module is unavailable unless an executable DSSR installation and DSSR-detected duplex stem are present.",
            "Stiffness is a local harmonic covariance model within each selected state; it is not fitted across unresolved multimodal mixtures.",
            "Translation-rotation covariance uses angstrom/radian mixed coordinates, so stiffness matrix elements have corresponding mixed units.",
            "DSSR step-order stability, sampling, regularization, state count, and replica sensitivity must be established before interpretation.",
            "Mechanical covariance and adjacent-step state association do not establish energetic causality or biological mechanism.",
        ],
    }


def helical_mechanics_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return helical_mechanics_project(project_path, hash_content=hash_content)
    except (
        HelicalMechanicsError, ManifestValidationError, OSError,
        KeyError, TypeError, ValueError, np.linalg.LinAlgError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "helical_mechanics",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "HELICAL_MECHANICS_INVALID",
                "message": message,
            } for message in messages],
        }
