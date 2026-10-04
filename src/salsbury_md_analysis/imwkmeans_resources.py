"""Explicit, conservative work model for the pooled Python iMWKMeans grid.

These are performance estimates, never clustering results. Iteration ceilings
are cost inputs, not predictions of convergence. No replica-local fit is used.
"""
from __future__ import annotations

import math

IMPLEMENTATION = "python-imwk-lp100-v1"
REFERENCE_SETTINGS = dict(component_indices=[1, 2, 3], k_values=list(range(2, 13)),
    minkowski_p_values=[1.5, 2.0, 3.0], initialization_ranks=[0, 1, 2, 3],
    maximum_iterations=500, maximum_silhouette_observations=1000)


def feature_dimension(settings):
    if "component_indices" in settings:
        return len(settings["component_indices"])
    return sum(len(row["value_indices"]) for row in settings.get("trajectory_feature_columns", []))


def work_per_fit_observation(settings):
    """Upper work proxy including initialization, Lp centers and scoring.

    Nonquadratic Lp centers use 100 bisection passes. Farthest-first seeding
    repeats distances to earlier centers. Silhouette uses an N*cap bound for
    min(N, cap)^2 work, so smaller fit samples cannot cost more.
    """
    dimensions = feature_dimension(settings)
    ranks = len(settings["initialization_ranks"])
    ks = settings["k_values"]
    ps = settings["minkowski_p_values"]
    iterations = settings["maximum_iterations"]
    silhouette = settings["maximum_silhouette_observations"]
    for value in [dimensions, ranks, iterations, silhouette, *ks]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("iMWKMeans resource work requires positive integer dimensions, grid and limits")
    if not ks or not ps or any(isinstance(p, bool) or not math.isfinite(p) or p <= 1 for p in ps):
        raise ValueError("iMWKMeans resource work requires a nonempty valid k/p grid")
    work = 0.0
    for k in ks:
        for p in ps:
            center_passes = 1 if p == 2 else 100
            work += dimensions * (ranks * (
                center_passes + k * (k - 1) / 2
                + iterations * (2 * k + center_passes + 2)) + silhouette)
    return work


def runtime_workload(settings, fit_observations, assignment_observations):
    for value in (fit_observations, assignment_observations):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("iMWKMeans workload counts must be positive integers")
    return {"implementation_id": IMPLEMENTATION,
        "settings": {key: settings[key] for key in (
            *REFERENCE_SETTINGS, "trajectory_feature_columns", "feature_source") if key in settings},
        "fit_observations": fit_observations,
        "assignment_observations": assignment_observations,
        "fit_work_units": work_per_fit_observation(settings) * fit_observations}


def runtime_model(settings, measurements=()):
    """Keep unqualified evidence out of workload calibration.

    The 0.5 s/reference-observation fallback is provisional: the normal-run
    incident completed 40,000-observation commands in up to 18,428.5 seconds.
    Their complete grid signatures are absent, so they support rejecting the
    old 0.025-second proxy, not a fitted transferable grid coefficient.
    """
    coefficient = 0.5 / work_per_fit_observation(REFERENCE_SETTINGS)
    overhead = 300.0
    accepted, excluded, censored = [], [], 0
    for row in measurements:
        workload = row.get("imwkmeans_workload", {})
        if workload.get("implementation_id") != IMPLEMENTATION:
            excluded.append("missing_or_different_workload_signature")
            continue
        checked = runtime_workload(workload["settings"], workload["fit_observations"], workload["assignment_observations"])
        if checked != workload:
            raise ValueError("iMWKMeans workload evidence does not match its settings/counts")
        timeout = row.get("evidence_status") == "right_censored_timeout"
        if (row.get("execution_started") is False or row.get("runtime_calibration_eligible") is False
                or row.get("evidence_status", "complete_execution") not in {"complete_execution", "right_censored_timeout"}):
            excluded.append("task_not_executed")
            continue
        seconds = row.get("wall_seconds_lower_bound") if timeout else row.get("wall_seconds")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("iMWKMeans runtime evidence requires positive executed wall time")
        # A timeout is a lower bound, with named model uncertainty, never a
        # completed runtime. Whole-command wall is a conservative one-core
        # costing surrogate, not measured CPU consumption or kernel timing.
        factor = 1.5 if timeout else 1.0
        coefficient = max(coefficient, seconds * factor / workload["fit_work_units"])
        accepted.append(row.get("source_sidecar_sha256", row.get("source_timeout_sha256")))
        censored += int(timeout)
    return {"implementation_id": IMPLEMENTATION,
        "work_units_per_fit_observation": work_per_fit_observation(settings),
        "seconds_per_work_unit": coefficient,
        "fixed_overhead_seconds": overhead,
        "assignment_seconds_per_observation": 0.002 + 3 * feature_dimension(settings) * max(settings["k_values"]) / 1e6,
        "qualified_measurement_count": len(accepted), "censored_measurement_count": censored,
        "censored_uncertainty_factor": 1.5,
        "source_evidence_sha256": accepted, "excluded_evidence_reasons": excluded,
        "status": "workload_envelope_not_held_out_validated" if accepted else "provisional_workload_scaled_fallback",
        "time_safety_factor_included": False,
        "limitations": "Grid/iteration ceiling proxy; hardware, convergence and upstream reconstruction can change runtime. Whole-command overhead is not a measured kernel cost."}


def validate_held_out(model, measurements):
    """Evaluate held-out rows without fitting to them or changing the model."""
    results = []
    for row in measurements:
        evidence_id = row.get("source_sidecar_sha256", row.get("source_timeout_sha256"))
        if evidence_id is not None and evidence_id in model["source_evidence_sha256"]:
            raise ValueError("held-out evidence was used to fit the runtime model")
        if (row.get("execution_started") is False or row.get("runtime_calibration_eligible") is False
                or row.get("evidence_status", "complete_execution") not in {"complete_execution", "right_censored_timeout"}):
            raise ValueError("held-out runtime evidence must describe executed completed or censored work")
        work = row["imwkmeans_workload"]
        checked = runtime_workload(work["settings"], work["fit_observations"], work["assignment_observations"])
        if checked != work:
            raise ValueError("held-out iMWKMeans workload signature mismatch")
        predicted = (model["fixed_overhead_seconds"]
            + model["seconds_per_work_unit"] * work["fit_work_units"]
            + (0.002 + 3 * feature_dimension(work["settings"]) * max(work["settings"]["k_values"]) / 1e6)
            * work["assignment_observations"])
        timeout = row.get("evidence_status") == "right_censored_timeout"
        observed = row["wall_seconds_lower_bound" if timeout else "wall_seconds"]
        if not isinstance(observed, (int, float)) or isinstance(observed, bool) or not math.isfinite(observed) or observed <= 0:
            raise ValueError("held-out runtime evidence requires positive executed wall time")
        results.append({"predicted_seconds": predicted, "observed_seconds": observed,
            "right_censored": timeout, "covers_observation_or_lower_bound": predicted >= observed})
    return {"held_out_count": len(results), "rows": results,
        "status": "complete" if results and all(r["covers_observation_or_lower_bound"] for r in results) else "failed",
        "censored_bounds_do_not_validate_completion_time": True}
