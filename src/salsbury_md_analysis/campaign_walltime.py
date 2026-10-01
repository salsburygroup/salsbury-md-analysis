"""Campaign allocation time from an already uncertainty-adjusted schedule."""

from __future__ import annotations

import math
from typing import Mapping


CAMPAIGN_WALLTIME_DEFAULTS = {
    "campaign_walltime_headroom_fraction": 1.0 / 3.0,
    "campaign_walltime_rounding_minutes": 60.0,
}


def campaign_walltime_request(
    estimated_hours: float,
    maximum_hours: float,
    policy: Mapping[str, object] | None = None,
) -> dict:
    """Add explicit allocation headroom once, without changing scientific work.

    A user ceiling can trim headroom/rounding, never the execution estimate.
    Serialized per-task kill limits do not enter this calculation.
    """
    settings = {**CAMPAIGN_WALLTIME_DEFAULTS, **(policy or {})}
    values = {
        "estimated_hours": estimated_hours,
        "maximum_hours": maximum_hours,
        **{key: settings[key] for key in CAMPAIGN_WALLTIME_DEFAULTS},
    }
    for key, value in values.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{key} must be finite and nonnegative")
    rounding = float(settings["campaign_walltime_rounding_minutes"])
    if rounding <= 0 or maximum_hours <= 0:
        raise ValueError("campaign wall-time ceiling and rounding must be positive")
    fraction = float(settings["campaign_walltime_headroom_fraction"])
    preferred_minutes = max(rounding, math.ceil(
        estimated_hours * 60.0 * (1.0 + fraction) / rounding - 1e-12
    ) * rounding)
    # Slurm's wall-time requests have second precision; never round a cap up.
    cap_minutes = math.floor(maximum_hours * 3600.0) / 60.0
    feasible = estimated_hours * 60.0 <= cap_minutes + 1e-9
    selected_minutes = min(preferred_minutes, cap_minutes) if feasible else None
    return {
        "schema": "salsbury-campaign-walltime-request-v1",
        "basis": "final_dependency_resource_schedule_with_model_uncertainty",
        "estimated_execution_hours": estimated_hours,
        "maximum_campaign_wall_hours": maximum_hours,
        "headroom_fraction": fraction,
        "rounding_minutes": rounding,
        "preferred_wall_hours": preferred_minutes / 60.0,
        "requested_wall_hours": (
            selected_minutes / 60.0 if selected_minutes is not None else None
        ),
        "requested_wall_minutes": selected_minutes,
        "headroom_limited_by_campaign_cap": preferred_minutes > cap_minutes + 1e-9,
        "submission_time_feasible": feasible,
        "status": ("estimate_exceeds_campaign_cap" if not feasible else
                   "headroom_limited_by_campaign_cap" if preferred_minutes > cap_minutes + 1e-9
                   else "complete"),
        "interpretation": (
            "The execution estimate already includes planner uncertainty. Add "
            "the declared campaign headroom once and round up, bounded by the "
            "user ceiling. Per-task timeout sums are diagnostics, not execution "
            "time. Queue waiting and a separate interactive build are excluded."
        ),
    }
