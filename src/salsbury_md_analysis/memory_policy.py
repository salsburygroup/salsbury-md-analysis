"""Explicit calibration-quality adjustments for task memory models."""

from __future__ import annotations

import math
from typing import Dict, Mapping, MutableMapping, Sequence


class MemoryPolicyError(ValueError):
    """Raised when memory-model uncertainty configuration is invalid."""


def _positive_factor(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise MemoryPolicyError(f"{label} must be finite and positive")
    return float(value)


def resolve_memory_uncertainty_policy(
    execution: Mapping[str, object],
) -> Dict[str, object]:
    """Resolve named model-uncertainty factors before cluster adjustment.

    ``memory_safety_factor`` remains a deprecated input alias for the weak-
    calibration factor so older private configurations fail only when both
    names are supplied.  This layer modifies the modeled working set.  It does
    not include any cluster or scheduler reservation adjustment.
    """

    legacy = execution.get("memory_safety_factor")
    explicit = execution.get("poorly_calibrated_memory_uncertainty_factor")
    if legacy is not None and explicit is not None:
        raise MemoryPolicyError(
            "execution.memory_safety_factor is a deprecated alias for "
            "execution.poorly_calibrated_memory_uncertainty_factor; supply "
            "only one"
        )
    well = _positive_factor(
        execution.get("well_calibrated_memory_uncertainty_factor", 1.0),
        "execution.well_calibrated_memory_uncertainty_factor",
    )
    poor = _positive_factor(
        explicit if explicit is not None else (
            legacy if legacy is not None else 1.25
        ),
        "execution.poorly_calibrated_memory_uncertainty_factor",
    )
    return {
        "policy_schema": "salsbury-memory-calibration-uncertainty-v1",
        "well_calibrated_factor": well,
        "poorly_calibrated_factor": poor,
        "legacy_alias_used": legacy is not None,
        "cluster_adjustment_included": False,
    }


def _scale_numeric(mapping: MutableMapping[str, object], key: str, factor: float) -> None:
    value = mapping.get(key)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemoryPolicyError(f"task {key} must be numeric")
    mapping[key] = float(value) * factor


def apply_memory_calibration_uncertainty(
    tasks: Sequence[MutableMapping[str, object]],
    policy: Mapping[str, object],
) -> None:
    """Apply one named task-model adjustment before cluster padding."""

    well = _positive_factor(
        policy.get("well_calibrated_factor"), "well_calibrated_factor"
    )
    poor = _positive_factor(
        policy.get("poorly_calibrated_factor"), "poorly_calibrated_factor"
    )
    for task in tasks:
        calibration = task.get("measured_resource_calibration")
        qualified = bool(
            task.get("memory_replacement_qualified", False)
            or (
                isinstance(calibration, Mapping)
                and calibration.get("memory_replacement_qualified", False)
            )
        )
        quality = "well_calibrated" if qualified else "poorly_calibrated"
        factor = well if qualified else poor
        previous = task.get("memory_calibration_uncertainty")
        if isinstance(previous, Mapping):
            if (
                str(previous.get("quality")) != quality
                or abs(float(previous.get("factor", -1.0)) - factor) > 1.0e-12
            ):
                raise MemoryPolicyError(
                    "task memory calibration uncertainty was already applied "
                    "with a different policy"
                )
            continue
        for key in (
            "estimated_peak_memory_gib",
            "estimated_peak_memory_gib_per_parallel_worker",
            "reducer_memory_gib",
        ):
            _scale_numeric(task, key, factor)
        for model_key in ("measured_memory_cost_model", "power_law_cost_model"):
            model = task.get(model_key)
            if isinstance(model, MutableMapping):
                _scale_numeric(model, "calibration_memory_gib", factor)
        task["memory_calibration_uncertainty"] = {
            "quality": quality,
            "factor": factor,
            "basis": (
                "qualified repeated completed memory evidence"
                if qualified else
                "weak, single-run, censored-only, or unmeasured memory model"
            ),
            "cluster_adjustment_included": False,
        }
