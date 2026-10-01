"""Campaign allocation time from an already uncertainty-adjusted schedule."""

from __future__ import annotations

import math
from typing import Mapping


CAMPAIGN_WALLTIME_DEFAULTS = {
    "campaign_walltime_headroom_fraction": 1.0 / 3.0,
    "campaign_walltime_rounding_minutes": 60.0,
}


def campaign_walltime_budget(
    maximum_hours: float,
    policy: Mapping[str, object] | None = None,
) -> dict:
    """Reserve the full allocation allowance before selecting scientific work."""
    settings = {**CAMPAIGN_WALLTIME_DEFAULTS, **(policy or {})}
    values = {
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
    usable_minutes = math.floor(maximum_hours * 60.0 / rounding + 1e-12) * rounding
    return {
        "maximum_campaign_wall_hours": maximum_hours,
        "maximum_estimated_execution_hours": usable_minutes / (60.0 * (1.0 + fraction)),
        "maximum_rounded_allocation_hours": usable_minutes / 60.0,
        "headroom_fraction": fraction,
        "rounding_minutes": rounding,
    }


def campaign_walltime_request(
    estimated_hours: float,
    maximum_hours: float,
    policy: Mapping[str, object] | None = None,
) -> dict:
    """Add full headroom once; reject rather than shrink it at the user ceiling."""
    budget = campaign_walltime_budget(maximum_hours, policy)
    if (isinstance(estimated_hours, bool) or not isinstance(estimated_hours, (int, float))
            or not math.isfinite(estimated_hours) or estimated_hours < 0):
        raise ValueError("estimated_hours must be finite and nonnegative")
    rounding = budget["rounding_minutes"]
    fraction = budget["headroom_fraction"]
    preferred_minutes = max(rounding, math.ceil(
        estimated_hours * 60.0 * (1.0 + fraction) / rounding - 1e-12
    ) * rounding)
    feasible = preferred_minutes <= maximum_hours * 60.0 + 1e-9
    selected_minutes = preferred_minutes if feasible else None
    return {
        "schema": "salsbury-campaign-walltime-request-v1",
        "basis": "final_dependency_resource_schedule_with_model_uncertainty",
        "estimated_execution_hours": estimated_hours,
        **budget,
        "preferred_wall_hours": preferred_minutes / 60.0,
        "requested_wall_hours": (
            selected_minutes / 60.0 if selected_minutes is not None else None
        ),
        "requested_wall_minutes": selected_minutes,
        "headroom_limited_by_campaign_cap": False,
        "submission_time_feasible": feasible,
        "status": "complete" if feasible else "padded_request_exceeds_campaign_cap",
        "interpretation": (
            "The execution estimate already includes planner uncertainty. Add "
            "the full declared campaign headroom once and round up. Planning "
            "reserves this allowance within the user ceiling; never trim it to "
            "make a plan pass. Per-task timeout sums are diagnostics, not execution "
            "time. Queue waiting and a separate interactive build are excluded."
        ),
    }
