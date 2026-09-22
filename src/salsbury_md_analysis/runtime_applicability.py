"""Bounded workload transfer for direct-estimator timing evidence.

This adjusts the *work* represented by an affine catalog fit, not measured
process startup. Unknown and out-of-envelope workloads retain legacy costs.
Memory, scientific sampling floors, and estimator implementations are unchanged.
"""
from __future__ import annotations

import math
from typing import Mapping, MutableMapping

MODULES = frozenset({
    "replica_rmsd_rg", "pooled_rmsf", "individual_pca", "dccm",
    "secondary_structure",
})
POLICY = "bounded-direct-workload-transfer-v1"


def apply_runtime_applicability(
    task: MutableMapping[str, object], calibration: Mapping[str, object],
    *, time_safety_factor: float,
) -> bool:
    """Apply only the tested one-replica, small/medium workload envelope.

    Catalog affine intercepts include source reading and setup work from larger
    systems; they are not portable measurements of fixed process startup.
    Retain the original baseline setup separately and transfer catalog work
    using the same dimensionless multiplier as initial automatic sampling.
    Timeout evidence stays visible, scaled as a workload-transfer estimate,
    never mislabeled as a measured lower bound for the new system.
    """
    module = task.get("module_id")
    basis = task.get("runtime_workload_scaling")
    counts = task.get("source_frames_per_replica")
    if module not in MODULES or not isinstance(basis, Mapping):
        return False
    atoms = basis.get("observed_atom_count")
    scale = basis.get("resolved_multiplier")
    if not (
        isinstance(atoms, int) and not isinstance(atoms, bool)
        and 423 <= atoms <= 21_478
        and isinstance(counts, list) and len(counts) == 1
        and isinstance(counts[0], int) and not isinstance(counts[0], bool)
        and 1_000 <= counts[0] <= 20_000
        and isinstance(scale, (int, float)) and not isinstance(scale, bool)
        and math.isfinite(scale) and 0.0 < scale <= 1.0
        and basis.get("reference_atom_count") == 85_199
        and basis.get("dimension") == "maximum topology atom count proxy"
        and int(calibration.get("complete_measurement_count", 0)) >= 1
    ):
        return False
    safety = float(time_safety_factor)
    if not math.isfinite(safety) or safety <= 0:
        raise ValueError("time safety factor must be finite and positive")
    rate = float(calibration.get("conservative_affine_cpu_seconds_per_frame",
                                 calibration["conservative_cpu_seconds_per_frame"]))
    work_intercept = float(calibration.get("conservative_fixed_cpu_seconds", 0.0))
    baseline = task.get("runtime_baseline_terms")
    if not isinstance(baseline, Mapping):
        return False  # Never infer startup from a catalog-overwritten task.
    setup = float(baseline["fixed_cpu_hours"])
    baseline_rate = float(baseline["cpu_seconds_per_physical_frame"])
    transferred_rate = rate * scale * safety
    # mkdssp is launched once per selected frame. Its process/file/dictionary
    # cost cannot be scaled away with the protein's atom count. Preserve the
    # completed per-frame wall-time envelope as an unscaled floor.
    external_floor = 0.0
    if module == "secondary_structure":
        wall_rate = calibration.get("maximum_completed_wall_seconds_per_frame")
        if not isinstance(wall_rate, (int, float)) or wall_rate <= 0:
            return False
        external_floor = float(wall_rate) * safety
    legacy = task.get("runtime_applicability", {}).get("legacy_terms") if isinstance(task.get("runtime_applicability"), Mapping) else None
    if legacy is None:
        legacy = {key: task.get(key) for key in (
            "fixed_cpu_hours", "cpu_seconds_per_physical_frame",
            "censored_wall_lower_bound_points",
        )}
    # Continuous reconstruction may decode the full source even at a sparse
    # analysis stride. Use the baseline full per-frame work as a conservative
    # source-read allowance, not as a measured startup coefficient. This can
    # double-count selected-frame work; it must never disappear at larger stride.
    source_scan_hours = baseline_rate * sum(counts) / 3600.0
    task["fixed_cpu_hours"] = (
        max(setup, work_intercept * scale * safety / 3600.0) + source_scan_hours
    )
    task["cpu_seconds_per_physical_frame"] = max(baseline_rate, transferred_rate, external_floor)
    task["censored_wall_lower_bound_points"] = [
        {**dict(point), "planning_wall_hours_lower_bound":
            float(point["planning_wall_seconds_lower_bound"]) * safety * scale / 3600.0}
        for point in calibration.get("censored_wall_lower_bound_points", [])
    ]
    task["runtime_applicability"] = {
        "policy": POLICY,
        "status": "bounded_workload_transfer",
        "workload_multiplier": scale,
        "baseline_setup_cpu_hours": setup,
        "conservative_source_scan_cpu_hours": source_scan_hours,
        "source_scan_policy": "baseline per-frame work times all source frames, independent of selected stride",
        "transferred_catalog_work_intercept_cpu_hours": work_intercept * scale * safety / 3600.0,
        "external_process_seconds_per_frame_floor": external_floor,
        "censored_evidence_interpretation": "transferred estimate, not a measured target-workload lower bound",
        "legacy_terms": legacy,
        "fallback": "unchanged legacy model outside module, size, source-length, replica or evidence scope",
        "expected_runtime_interpretation": "conservative workload estimate, not scheduler timeout or scientific acceptance",
    }
    return True
