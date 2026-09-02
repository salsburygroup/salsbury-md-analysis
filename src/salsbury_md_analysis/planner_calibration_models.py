"""Fit and validate portable size-, length-, and work-aware CPU models."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.optimize import nnls


MODEL_SCHEMA = "salsbury-planner-size-length-cpu-models-v1"
HOLDOUT_EVIDENCE_SCHEMA = "salsbury-planner-runtime-holdouts-v1"
HOLDOUT_ACCEPTANCE_SCHEMA = "salsbury-planner-runtime-holdout-acceptance-v1"
SUPPORTED_MODULES = {
    "structural_integrity_qc": {
        "work_field": "topology_atom_frame_count",
        "work_unit": "topology_atom_selected_frames_v1",
        "planning_proxy": "topology_atoms_per_selected_frame_v1",
    },
    "hydrogen_bond_discovery": {
        "work_field": "spatial_neighbor_pair_count",
        "work_unit": "spatial_neighbor_pairs_v1",
        "planning_proxy": "spatial_pairs_per_endpoint_selected_frame_v1",
    },
    "ion_atmosphere": {
        "work_field": "ion_target_minimum_image_pair_count",
        "work_unit": "ion_target_minimum_image_pairs_v1",
        "planning_proxy": "ion_pairs_per_topology_atom_selected_frame_v1",
    },
}


class PlannerCalibrationModelError(ValueError):
    """Raised when calibration evidence or a fitted model is incomplete."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive(value: object, label: str, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise PlannerCalibrationModelError(
            f"{label} must be a finite {qualifier} number"
        )
    return float(value)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PlannerCalibrationModelError(f"{label} must be a positive integer")
    return value


def validate_size_length_models(payload: Mapping[str, object]) -> Dict[str, object]:
    """Return a validated copy of one planner model artifact."""

    if (
        payload.get("model_schema") != MODEL_SCHEMA
        or payload.get("technical_status") != "complete"
        or payload.get("scientific_status") != "runtime evidence only"
    ):
        raise PlannerCalibrationModelError("invalid planner CPU model artifact")
    source_hash = payload.get("source_evidence_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise PlannerCalibrationModelError("model lacks a source evidence SHA-256")
    raw_models = payload.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise PlannerCalibrationModelError("model artifact contains no module models")
    models: Dict[str, object] = {}
    for module_id, raw in raw_models.items():
        if module_id not in SUPPORTED_MODULES or not isinstance(raw, dict):
            raise PlannerCalibrationModelError(f"unsupported model: {module_id}")
        expected = SUPPORTED_MODULES[module_id]
        if raw.get("selected_work_unit") != expected["work_unit"]:
            raise PlannerCalibrationModelError(
                f"{module_id} has the wrong selected-work unit"
            )
        if raw.get("planning_proxy") != expected["planning_proxy"]:
            raise PlannerCalibrationModelError(
                f"{module_id} has the wrong planning proxy"
            )
        for field in (
            "intercept_cpu_seconds",
            "cpu_seconds_per_topology_atom_source_frame",
            "cpu_seconds_per_selected_work_unit",
        ):
            _positive(raw.get(field), f"{module_id}.{field}", allow_zero=True)
        _positive(raw.get("residual_safety_factor"), "residual_safety_factor")
        _positive(raw.get("selected_work_units_per_proxy_unit"), (
            f"{module_id}.selected_work_units_per_proxy_unit"
        ))
        if raw.get("heldout_validation_passed") is not True:
            raise PlannerCalibrationModelError(
                f"{module_id} did not pass independent held-out validation"
            )
        _positive_integer(raw.get("training_point_count"), "training_point_count")
        _positive_integer(raw.get("heldout_point_count"), "heldout_point_count")
        models[module_id] = dict(raw)
    result = dict(payload)
    result["models"] = models
    return result


def validate_runtime_holdouts(
    model_payload: Mapping[str, object],
    holdout_payload: Mapping[str, object],
) -> Dict[str, object]:
    """Check independent runtime observations against planning upper bounds.

    The prediction uses only manifest-derived source size, selected-frame count,
    and the module's pre-coordinate work proxy. Observed spatial work is kept in
    the acceptance record for later model review but never enters the prediction.
    """

    model_artifact = validate_size_length_models(model_payload)
    if (
        holdout_payload.get("holdout_schema") != HOLDOUT_EVIDENCE_SCHEMA
        or holdout_payload.get("technical_status") != "complete"
        or holdout_payload.get("unexpected_error_count") != 0
    ):
        raise PlannerCalibrationModelError("invalid runtime holdout evidence")
    raw_points = holdout_payload.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise PlannerCalibrationModelError("runtime holdout evidence has no points")

    rows = []
    for point in raw_points:
        if not isinstance(point, Mapping):
            raise PlannerCalibrationModelError("runtime holdout point must be an object")
        module_id = point.get("module_id")
        model = model_artifact["models"].get(module_id)  # type: ignore[union-attr]
        if not isinstance(module_id, str) or not isinstance(model, Mapping):
            raise PlannerCalibrationModelError(
                f"holdout has no fitted model: {module_id}"
            )
        source_work = _positive_integer(
            point.get("source_topology_atom_frame_count"),
            "source_topology_atom_frame_count",
        )
        selected_frames = _positive_integer(
            point.get("selected_source_physical_frames"),
            "selected_source_physical_frames",
        )
        proxy_count = _positive_integer(
            point.get("selected_work_proxy_count_per_frame"),
            "selected_work_proxy_count_per_frame",
        )
        observed_cpu = _positive(point.get("observed_total_cpu_seconds"), (
            "observed_total_cpu_seconds"
        ))
        if point.get("stderr_nonempty") is not False:
            raise PlannerCalibrationModelError(
                f"{point.get('point_id')} wrote unexpected stderr"
            )
        for field in (
            "report_sha256",
            "project_manifest_sha256",
            "input_content_signature_sha256",
            "contract_signature_sha256",
        ):
            value = point.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise PlannerCalibrationModelError(
                    f"{point.get('point_id')}.{field} must be a SHA-256"
                )

        estimated_selected_work = (
            float(model["selected_work_units_per_proxy_unit"])
            * proxy_count
            * selected_frames
        )
        raw_prediction = (
            float(model["intercept_cpu_seconds"])
            + float(model["cpu_seconds_per_topology_atom_source_frame"])
            * source_work
            + float(model["cpu_seconds_per_selected_work_unit"])
            * estimated_selected_work
        )
        upper = float(model["residual_safety_factor"]) * raw_prediction
        passed = observed_cpu <= upper
        rows.append({
            "point_id": point.get("point_id"),
            "module_id": module_id,
            "report_sha256": point["report_sha256"],
            "project_manifest_sha256": point["project_manifest_sha256"],
            "input_content_signature_sha256": point[
                "input_content_signature_sha256"
            ],
            "contract_signature_sha256": point["contract_signature_sha256"],
            "slurm_job_id": point.get("slurm_job_id"),
            "source_topology_atom_frame_count": source_work,
            "selected_source_physical_frames": selected_frames,
            "selected_work_proxy_count_per_frame": proxy_count,
            "estimated_selected_work_units": estimated_selected_work,
            "observed_selected_work_units": point.get(
                "observed_selected_work_units"
            ),
            "observed_total_cpu_seconds": observed_cpu,
            "raw_predicted_cpu_seconds": raw_prediction,
            "planning_upper_cpu_seconds": upper,
            "observed_to_planning_upper_ratio": observed_cpu / upper,
            "observed_wall_seconds": point.get("observed_wall_seconds"),
            "instrumented_maximum_resident_memory_mib": point.get(
                "instrumented_maximum_resident_memory_mib"
            ),
            "passed": passed,
        })
    if not all(bool(row["passed"]) for row in rows):
        failed = [str(row["point_id"]) for row in rows if not row["passed"]]
        raise PlannerCalibrationModelError(
            "runtime holdout exceeded planning upper bound: " + ", ".join(failed)
        )
    return {
        "acceptance_schema": HOLDOUT_ACCEPTANCE_SCHEMA,
        "technical_status": "complete",
        "scientific_status": "runtime evidence only; scientific validity not evaluated",
        "model_source_evidence_sha256": model_artifact[
            "source_evidence_sha256"
        ],
        "holdout_evidence_sha256": holdout_payload.get("content_sha256"),
        "prediction_coordinate_data_used": False,
        "prediction_dense_candidate_universe_materialized": False,
        "model_coefficients_changed": False,
        "model_change_reason": (
            "all independent observations remained below the existing planning "
            "upper bounds"
        ),
        "point_count": len(rows),
        "maximum_observed_to_planning_upper_ratio": max(
            float(row["observed_to_planning_upper_ratio"]) for row in rows
        ),
        "all_holdouts_passed": True,
        "points": rows,
    }


def _fit_one(
    module_id: str,
    points: Sequence[Mapping[str, object]],
    *,
    residual_safety_factor: float,
) -> Dict[str, object]:
    definition = SUPPORTED_MODULES[module_id]
    atom_counts = sorted({_positive_integer(
        row.get("topology_atom_count"), "topology_atom_count"
    ) for row in points})
    if len(atom_counts) != 3:
        raise PlannerCalibrationModelError(
            f"{module_id} requires exactly three distinct system sizes"
        )
    heldout_atom_count = atom_counts[1]
    training = [row for row in points if row["topology_atom_count"] != heldout_atom_count]
    heldout = [row for row in points if row["topology_atom_count"] == heldout_atom_count]
    if len(training) < 4 or len(heldout) < 2:
        raise PlannerCalibrationModelError(
            f"{module_id} lacks enough training or held-out points"
        )

    work_field = str(definition["work_field"])
    for row in points:
        _positive_integer(row.get("topology_atom_source_frame_count"), (
            "topology_atom_source_frame_count"
        ))
        _positive_integer(row.get(work_field), work_field)
        _positive(row.get("total_cpu_seconds"), "total_cpu_seconds")
        if row.get("stderr_nonempty") is not False:
            raise PlannerCalibrationModelError(
                f"{row.get('point_id')} wrote unexpected stderr"
            )

    source_scale = max(int(row["topology_atom_source_frame_count"]) for row in training)
    work_scale = max(int(row[work_field]) for row in training)
    design = np.asarray([
        [
            1.0,
            int(row["topology_atom_source_frame_count"]) / source_scale,
            int(row[work_field]) / work_scale,
        ]
        for row in training
    ], dtype=float)
    observed = np.asarray(
        [float(row["total_cpu_seconds"]) for row in training], dtype=float
    )
    scaled_coefficients, residual_norm = nnls(design, observed)
    intercept = float(scaled_coefficients[0])
    source_rate = float(scaled_coefficients[1]) / source_scale
    selected_rate = float(scaled_coefficients[2]) / work_scale

    if module_id == "hydrogen_bond_discovery":
        proxy_ratios = [
            int(row[work_field])
            / (
                int(row["maximum_spatial_endpoint_count_per_system"])
                * int(row["selected_source_physical_frames"])
            )
            for row in training
        ]
        proxy_basis = "maximum donor-H plus acceptor endpoints"
    elif module_id == "ion_atmosphere":
        proxy_ratios = [
            int(row[work_field]) / int(row["topology_atom_frame_count"])
            for row in training
        ]
        proxy_basis = "maximum topology atom count"
    else:
        proxy_ratios = [1.0]
        proxy_basis = "maximum topology atom count"
    proxy_rate = max(proxy_ratios)

    def prediction(row: Mapping[str, object]) -> tuple[float, float, float]:
        frames = int(row["selected_source_physical_frames"])
        if module_id == "hydrogen_bond_discovery":
            proxy_units = (
                int(row["maximum_spatial_endpoint_count_per_system"])
                * frames
            )
        else:
            proxy_units = int(row["topology_atom_frame_count"])
        estimated_work = proxy_rate * proxy_units
        raw = (
            intercept
            + source_rate * int(row["topology_atom_source_frame_count"])
            + selected_rate * estimated_work
        )
        return raw, raw * residual_safety_factor, estimated_work

    validation_rows = []
    for row in heldout:
        raw, upper, estimated_work = prediction(row)
        observed_cpu = float(row["total_cpu_seconds"])
        validation_rows.append({
            "point_id": row["point_id"],
            "observed_cpu_seconds": observed_cpu,
            "raw_predicted_cpu_seconds": raw,
            "planning_upper_cpu_seconds": upper,
            "observed_to_planning_upper_ratio": observed_cpu / upper,
            "estimated_selected_work_units": estimated_work,
            "observed_selected_work_units": int(row[work_field]),
            "passed": observed_cpu <= upper,
        })
    if not all(bool(row["passed"]) for row in validation_rows):
        raise PlannerCalibrationModelError(
            f"{module_id} underpredicts one or more held-out points"
        )

    training_ratios = []
    for row in training:
        raw, upper, _ = prediction(row)
        training_ratios.append(float(row["total_cpu_seconds"]) / upper)
    return {
        "module_id": module_id,
        "model_form": (
            "residual_safety_factor * (intercept + source_rate * "
            "topology_atom_source_frames + selected_rate * selected_work_units)"
        ),
        "intercept_cpu_seconds": intercept,
        "cpu_seconds_per_topology_atom_source_frame": source_rate,
        "cpu_seconds_per_selected_work_unit": selected_rate,
        "selected_work_unit": definition["work_unit"],
        "planning_proxy": definition["planning_proxy"],
        "planning_proxy_basis": proxy_basis,
        "selected_work_units_per_proxy_unit": proxy_rate,
        "residual_safety_factor": residual_safety_factor,
        "training_size_policy": "smallest_and_largest_topology_atom_counts",
        "heldout_size_policy": "middle_topology_atom_count",
        "training_topology_atom_counts": [atom_counts[0], atom_counts[2]],
        "heldout_topology_atom_count": heldout_atom_count,
        "training_point_count": len(training),
        "heldout_point_count": len(heldout),
        "maximum_training_observed_to_planning_upper_ratio": max(training_ratios),
        "maximum_heldout_observed_to_planning_upper_ratio": max(
            float(row["observed_to_planning_upper_ratio"])
            for row in validation_rows
        ),
        "heldout_validation_passed": True,
        "heldout_validation": validation_rows,
        "fit_residual_norm_cpu_seconds": float(residual_norm),
        "measured_ranges": {
            "topology_atom_count": [min(atom_counts), max(atom_counts)],
            "source_frame_count": [
                min(int(row["source_frame_count"]) for row in points),
                max(int(row["source_frame_count"]) for row in points),
            ],
            "selected_source_physical_frames": [
                min(int(row["selected_source_physical_frames"]) for row in points),
                max(int(row["selected_source_physical_frames"]) for row in points),
            ],
            "selected_work_units": [
                min(int(row[work_field]) for row in points),
                max(int(row[work_field]) for row in points),
            ],
        },
        "extrapolation_policy": (
            "planning estimate only; a project-local pilot is required outside "
            "the measured size, source-length, or selected-work ranges"
        ),
    }


def fit_size_length_models(
    evidence_path: Path,
    *,
    residual_safety_factor: float = 1.5,
) -> Dict[str, object]:
    """Fit edge sizes and reserve the middle system as an independent gate."""

    source = evidence_path.expanduser().resolve(strict=True)
    evidence = json.loads(source.read_text(encoding="utf-8"))
    if (
        evidence.get("evidence_schema")
        != "salsbury-planner-calibration-evidence-matrix-v1"
        or evidence.get("technical_status") != "complete"
        or evidence.get("unexpected_error_count") != 0
    ):
        raise PlannerCalibrationModelError("invalid calibration evidence matrix")
    safety = _positive(residual_safety_factor, "residual_safety_factor")
    raw_points = evidence.get("points")
    if not isinstance(raw_points, list):
        raise PlannerCalibrationModelError("calibration evidence has no points")
    models = {}
    for module_id in SUPPORTED_MODULES:
        rows = [
            row for row in raw_points
            if isinstance(row, dict) and row.get("module_id") == module_id
        ]
        if not rows:
            raise PlannerCalibrationModelError(
                f"calibration evidence lacks {module_id}"
            )
        models[module_id] = _fit_one(
            module_id, rows, residual_safety_factor=safety
        )
    result = {
        "model_schema": MODEL_SCHEMA,
        "technical_status": "complete",
        "scientific_status": "runtime evidence only",
        "source_evidence_path": str(source),
        "source_evidence_sha256": sha256(source),
        "fit_policy": {
            "estimator": "nonnegative_least_squares_v1",
            "training_sizes": "smallest_and_largest",
            "independent_holdout_size": "middle",
            "residual_safety_factor": safety,
            "coordinate_data_used_by_planner": False,
        },
        "models": models,
    }
    return validate_size_length_models(result)
