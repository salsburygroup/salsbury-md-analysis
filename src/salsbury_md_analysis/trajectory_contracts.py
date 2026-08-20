"""Shared frame-axis and periodic-coordinate execution contracts."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Union


TIME_IN_PS = {"fs": 0.001, "ps": 1.0, "ns": 1000.0, "us": 1_000_000.0}
PERIODIC_COORDINATE_POLICIES = (
    "reject",
    "allow_wrapped_diagnostic",
    "make_whole",
    "unwrap_continuous",
    "preprocessed_make_whole",
)


class TrajectoryContractError(ValueError):
    """Raised when frame timing or periodic-coordinate handling is unsafe."""


def normalize_segment_timing(
    segment: Mapping[str, object], output_unit: str
) -> Dict[str, object]:
    """Normalize one validated segment timing declaration to ``output_unit``."""

    timing = segment.get("timing")
    if not isinstance(timing, dict):
        raise TrajectoryContractError("segment timing is required")
    source_unit = timing.get("unit")
    if source_unit not in TIME_IN_PS:
        raise TrajectoryContractError("segment timing unit must be fs, ps, ns, or us")
    if output_unit not in TIME_IN_PS:
        raise TrajectoryContractError("output time unit must be fs, ps, ns, or us")
    first = timing.get("first_frame_time")
    interval = timing.get("frame_interval")
    if (
        isinstance(first, bool)
        or not isinstance(first, (int, float))
        or not math.isfinite(float(first))
    ):
        raise TrajectoryContractError("first_frame_time must be a finite number")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or float(interval) <= 0.0
    ):
        raise TrajectoryContractError("frame_interval must be a finite positive number")
    scale = TIME_IN_PS[str(source_unit)] / TIME_IN_PS[output_unit]
    return {
        "first_frame_time": float(first) * scale,
        "frame_interval": float(interval) * scale,
        "unit": output_unit,
        "declared_unit": str(source_unit),
    }


def frame_time(timing: Mapping[str, object], frame_index: int) -> float:
    """Return the physical time of a zero-based source frame."""

    if frame_index < 0:
        raise TrajectoryContractError("frame_index must be nonnegative")
    return float(timing["first_frame_time"]) + frame_index * float(
        timing["frame_interval"]
    )


def normalize_sample_axis(segment: Mapping[str, object]) -> Dict[str, object]:
    """Return a validated sample-index axis for a non-temporal ensemble."""

    sample_axis = segment.get("sample_axis")
    if not isinstance(sample_axis, dict):
        raise TrajectoryContractError("segment sample_axis is required")
    first = sample_axis.get("first_sample_index")
    interval = sample_axis.get("sample_interval")
    if isinstance(first, bool) or not isinstance(first, int) or first < 0:
        raise TrajectoryContractError(
            "first_sample_index must be a nonnegative integer"
        )
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise TrajectoryContractError("sample_interval must be a positive integer")
    return {
        "first_sample_index": first,
        "sample_interval": interval,
        "unit": "sample",
    }


def normalize_segment_axis(
    segment: Mapping[str, object], output_time_unit: Optional[str]
) -> Dict[str, object]:
    """Normalize exactly one physical-time or sample-index frame axis."""

    has_timing = segment.get("timing") is not None
    has_samples = segment.get("sample_axis") is not None
    if has_timing == has_samples:
        raise TrajectoryContractError(
            "segment must declare exactly one of timing or sample_axis"
        )
    if has_timing:
        if output_time_unit is None:
            raise TrajectoryContractError(
                "output time unit is required for a physical-time segment"
            )
        return {
            "kind": "physical_time",
            "timing": normalize_segment_timing(segment, output_time_unit),
        }
    return {"kind": "sample_index", "sample_axis": normalize_sample_axis(segment)}


def frame_axis_value(
    axis: Mapping[str, object], frame_index: int
) -> Union[int, float]:
    """Return the declared coordinate of a zero-based source frame."""

    if frame_index < 0:
        raise TrajectoryContractError("frame_index must be nonnegative")
    kind = axis.get("kind")
    if kind == "physical_time":
        timing = axis.get("timing")
        if not isinstance(timing, dict):
            raise TrajectoryContractError("physical-time axis is missing timing")
        return frame_time(timing, frame_index)
    if kind == "sample_index":
        sample_axis = axis.get("sample_axis")
        if not isinstance(sample_axis, dict):
            raise TrajectoryContractError("sample-index axis is missing sample_axis")
        return int(sample_axis["first_sample_index"]) + frame_index * int(
            sample_axis["sample_interval"]
        )
    raise TrajectoryContractError("frame axis kind must be physical_time or sample_index")


def require_periodic_policy(value: object) -> str:
    """Return a valid explicit periodic-coordinate policy or fail closed."""

    if value not in PERIODIC_COORDINATE_POLICIES:
        raise TrajectoryContractError(
            "periodic_coordinate_policy must be reject, allow_wrapped_diagnostic, "
            "make_whole, unwrap_continuous, or preprocessed_make_whole"
        )
    return str(value)


def enforce_periodic_policy(
    periodic_cell_present: bool, policy: str, location: str
) -> None:
    """Block periodic coordinates unless diagnostic wrapped analysis is explicit."""

    require_periodic_policy(policy)
    if periodic_cell_present and policy == "reject":
        raise TrajectoryContractError(
            f"{location} declares a periodic cell; periodic_coordinate_policy=reject "
            "blocks periodic analysis"
        )
