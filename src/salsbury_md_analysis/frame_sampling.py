"""Deterministic, replica-balanced trajectory frame selection.

Production analyses use one exact integer stride per method.  The planner
counts every declared segment, applies that stride over each replica's
concatenated timeline, and emits the exact retained count.  The older
near-uniform budget mode remains readable only for frozen-project
reproducibility; current planners do not emit it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple, Type

from .coordinates import coordinate_format, iter_coordinate_frames
from .manifests import resolve_manifest_path
from .preflight import FileProbeError, probe_dcd
from .validation import positive_integer


FrameSelectionPlan = Dict[Tuple[str, str, str], Optional[Set[int]]]


def normalize_frame_selection(
    value: object,
    frame_stride: int,
    *,
    error_type: Type[Exception] = ValueError,
) -> Dict[str, object]:
    """Validate a frame-selection definition, defaulting to fixed stride."""

    selection = {"mode": "fixed_stride_v1"} if value is None else value
    if not isinstance(selection, dict):
        raise error_type("frame_selection must be an object")
    mode = selection.get("mode")
    if mode == "fixed_stride_v1" and set(selection) == {"mode"}:
        return {"mode": mode}
    if mode == "integer_stride_per_replica_v1" and set(selection) == {
        "mode", "stride",
    }:
        if frame_stride != 1:
            raise error_type(
                "integer_stride_per_replica_v1 requires frame_stride = 1"
            )
        return {
            "mode": mode,
            "stride": positive_integer(
                selection["stride"],
                "frame_selection.stride",
                error_type=error_type,
            ),
        }
    if mode == "uniform_per_replica_budget_v1" and set(selection) == {
        "mode", "maximum_frames_per_replica",
    }:
        if frame_stride != 1:
            raise error_type(
                "uniform_per_replica_budget_v1 requires frame_stride = 1"
            )
        return {
            "mode": mode,
            "maximum_frames_per_replica": positive_integer(
                selection["maximum_frames_per_replica"],
                "frame_selection.maximum_frames_per_replica",
                error_type=error_type,
            ),
        }
    if mode == "auto_resource_budget_v1":
        allowed = {
            "mode", "target_wall_seconds", "estimated_seconds_per_frame",
            "minimum_frames_per_replica", "fixed_overhead_seconds",
            "safety_factor", "sensitivity_check_policy",
            "estimated_peak_memory_mib", "target_memory_mib",
            "calibration_id",
        }
        required = {
            "mode", "target_wall_seconds", "estimated_seconds_per_frame",
            "minimum_frames_per_replica",
        }
        if not required.issubset(selection) or set(selection).difference(allowed):
            raise error_type(
                "auto_resource_budget_v1 requires target_wall_seconds, "
                "estimated_seconds_per_frame, and minimum_frames_per_replica"
            )
        if frame_stride != 1:
            raise error_type("auto_resource_budget_v1 requires frame_stride = 1")

        def positive_float(name: str, default: object | None = None) -> float:
            value = selection.get(name, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise error_type(f"frame_selection.{name} must be finite and positive")
            return float(value)

        overhead = selection.get("fixed_overhead_seconds", 0.0)
        if (
            isinstance(overhead, bool)
            or not isinstance(overhead, (int, float))
            or not math.isfinite(float(overhead))
            or float(overhead) < 0.0
        ):
            raise error_type(
                "frame_selection.fixed_overhead_seconds must be finite and nonnegative"
            )
        sensitivity = selection.get("sensitivity_check_policy", "off")
        if sensitivity not in {"off", "recommend", "require"}:
            raise error_type(
                "frame_selection.sensitivity_check_policy must be off, recommend, or require"
            )
        memory_keys = {
            "estimated_peak_memory_mib", "target_memory_mib"
        }.intersection(selection)
        if memory_keys and len(memory_keys) != 2:
            raise error_type(
                "estimated_peak_memory_mib and target_memory_mib must be declared together"
            )
        result: Dict[str, object] = {
            "mode": mode,
            "target_wall_seconds": positive_float("target_wall_seconds"),
            "estimated_seconds_per_frame": positive_float(
                "estimated_seconds_per_frame"
            ),
            "minimum_frames_per_replica": positive_integer(
                selection["minimum_frames_per_replica"],
                "frame_selection.minimum_frames_per_replica",
                error_type=error_type,
            ),
            "fixed_overhead_seconds": float(overhead),
            "safety_factor": positive_float("safety_factor", 1.5),
            "sensitivity_check_policy": sensitivity,
        }
        if memory_keys:
            result.update({
                "estimated_peak_memory_mib": positive_float(
                    "estimated_peak_memory_mib"
                ),
                "target_memory_mib": positive_float("target_memory_mib"),
            })
        calibration_id = selection.get("calibration_id")
        if calibration_id is not None:
            if not isinstance(calibration_id, str) or not calibration_id.strip():
                raise error_type(
                    "frame_selection.calibration_id must be a nonempty string"
                )
            result["calibration_id"] = calibration_id.strip()
        return result
    raise error_type(
        "frame_selection must be fixed_stride_v1, integer_stride_per_replica_v1, "
        "a legacy uniform per-replica budget, or an automatic resource budget"
    )


def source_frame_count(
    path: Path,
    coordinate_unit: str,
    *,
    error_type: Type[Exception] = ValueError,
) -> int:
    """Return a source-frame count without decoding DCD coordinate payloads."""

    try:
        if coordinate_format(path) == "dcd":
            count = int(probe_dcd(path)["declared_frame_count"])
        else:
            count = sum(1 for _ in iter_coordinate_frames(path, coordinate_unit))
    except (FileProbeError, OSError, ValueError) as exc:
        raise error_type(str(exc)) from exc
    if count <= 0:
        raise error_type("trajectory contains no coordinate frames")
    return count


def uniform_indices(
    frame_count: int,
    budget: int,
    *,
    error_type: Type[Exception] = ValueError,
) -> Set[int]:
    """Return reproducible full-timespan indices for a positive frame budget."""

    if frame_count <= 0 or budget <= 0:
        raise error_type("frame-selection counts must be positive")
    selected = min(frame_count, budget)
    if selected == frame_count:
        return set(range(frame_count))
    if selected == 1:
        return {frame_count // 2}
    return {
        (index * (frame_count - 1)) // (selected - 1)
        for index in range(selected)
    }


def integer_stride_selected_count(frame_count: int, stride: int) -> int:
    """Return the exact number of ``0, stride, 2*stride, ...`` indices."""

    if frame_count <= 0 or stride <= 0:
        raise ValueError("frame count and integer stride must be positive")
    return (frame_count - 1) // stride + 1


def integer_stride_indices(
    frame_count: int,
    stride: int,
    *,
    error_type: Type[Exception] = ValueError,
) -> Set[int]:
    """Return strict integer-stride indices beginning at replica frame zero."""

    if frame_count <= 0 or stride <= 0:
        raise error_type("frame count and integer stride must be positive")
    return set(range(0, frame_count, stride))


def integer_stride_for_budget(
    frame_counts: Tuple[int, ...] | List[int],
    maximum_frames_per_replica: int,
    *,
    error_type: Type[Exception] = ValueError,
) -> int:
    """Choose the finest common stride whose actual counts fit a hard budget.

    The small adjustment loops are intentional: the budget is discrete, so the
    initially estimated stride is verified against the actual counts rather
    than assumed to produce the requested number of frames.
    """

    if (
        not frame_counts
        or any(count <= 0 for count in frame_counts)
        or maximum_frames_per_replica <= 0
    ):
        raise error_type("frame counts and frame budget must be positive")
    stride = max(1, math.ceil(max(frame_counts) / maximum_frames_per_replica))
    while any(
        integer_stride_selected_count(count, stride) > maximum_frames_per_replica
        for count in frame_counts
    ):
        stride += 1
    while stride > 1 and all(
        integer_stride_selected_count(count, stride - 1)
        <= maximum_frames_per_replica
        for count in frame_counts
    ):
        stride -= 1
    return stride


def plan_frame_selection(
    system_manifest: Mapping[str, object],
    system_path: Path,
    coordinate_unit: str,
    selection: Mapping[str, object],
    *,
    frame_stride: int,
    maximum_selected_frames: Optional[int] = None,
    error_type: Type[Exception] = ValueError,
) -> Tuple[FrameSelectionPlan, Dict[str, object]]:
    """Plan frame selection for every declared replica and segment.

    Integer strides and legacy uniform budgets are applied to each replica's
    concatenated segment order.  Fixed stride remains segment-local only for
    backward compatibility.  ``None`` in the returned plan means all source
    frames, avoiding a large index set when the resolved stride is one.
    """

    mode = str(selection["mode"])
    systems = system_manifest.get("systems")
    if not isinstance(systems, list) or not systems:
        raise error_type("system manifest contains no systems")
    replica_count = sum(
        len(raw_system.get("replicas", []))
        for raw_system in systems
        if isinstance(raw_system, dict) and isinstance(raw_system.get("replicas"), list)
    )
    automatic_budget: Optional[int] = None
    resource_estimate: Optional[Dict[str, object]] = None
    if mode == "auto_resource_budget_v1":
        if replica_count <= 0:
            raise error_type("system manifest contains no replicas")
        estimated_memory = selection.get("estimated_peak_memory_mib")
        target_memory = selection.get("target_memory_mib")
        if (
            estimated_memory is not None
            and target_memory is not None
            and float(estimated_memory) > float(target_memory)
        ):
            raise error_type(
                "estimated peak memory exceeds the target memory; reducing frame count "
                "does not safely resolve a frame-independent memory gate"
            )
        target = float(selection["target_wall_seconds"])
        rate = float(selection["estimated_seconds_per_frame"])
        overhead = float(selection["fixed_overhead_seconds"])
        safety = float(selection["safety_factor"])
        usable = target / safety - overhead
        total_capacity = math.floor(usable / rate) if usable > 0.0 else 0
        minimum = int(selection["minimum_frames_per_replica"])
        if total_capacity < minimum * replica_count:
            raise error_type(
                "automatic resource envelope cannot fit minimum_frames_per_replica "
                "for every declared replica"
            )
        automatic_budget = max(minimum, total_capacity // replica_count)
        resource_estimate = {
            "target_wall_seconds": target,
            "estimated_seconds_per_frame": rate,
            "fixed_overhead_seconds": overhead,
            "safety_factor": safety,
            "estimated_total_frame_capacity": total_capacity,
            "resolved_maximum_frames_per_replica": automatic_budget,
            "minimum_frames_per_replica": minimum,
            "sensitivity_check_policy": selection["sensitivity_check_policy"],
            "calibration_id": selection.get("calibration_id"),
            "estimated_peak_memory_mib": estimated_memory,
            "target_memory_mib": target_memory,
        }
    replica_inputs: List[Dict[str, object]] = []
    for raw_system in systems:
        if not isinstance(raw_system, dict):
            raise error_type("system manifest system entry must be an object")
        system_id = str(raw_system["system_id"])
        replicas = raw_system.get("replicas")
        if not isinstance(replicas, list) or not replicas:
            raise error_type(f"system {system_id} contains no replicas")
        for replica in replicas:
            if not isinstance(replica, dict):
                raise error_type("replica entry must be an object")
            replica_id = str(replica["replica_id"])
            raw_segments = replica.get("segments")
            if not isinstance(raw_segments, list) or not raw_segments:
                raise error_type(f"{system_id}/{replica_id} contains no segments")
            segments: List[Tuple[str, int]] = []
            replica_source = 0
            for segment in raw_segments:
                if not isinstance(segment, dict):
                    raise error_type("segment entry must be an object")
                segment_id = str(segment["segment_id"])
                trajectory_path = resolve_manifest_path(
                    str(segment["trajectory"]), system_path
                )
                count = source_frame_count(
                    trajectory_path, coordinate_unit, error_type=error_type
                )
                segments.append((segment_id, count))
                replica_source += count

            replica_inputs.append({
                "system_id": system_id,
                "replica_id": replica_id,
                "segments": segments,
                "source_frame_count": replica_source,
            })

    common_integer_stride: Optional[int] = None
    if mode == "integer_stride_per_replica_v1":
        common_integer_stride = int(selection["stride"])
    elif mode == "auto_resource_budget_v1":
        assert automatic_budget is not None
        common_integer_stride = integer_stride_for_budget(
            [int(row["source_frame_count"]) for row in replica_inputs],
            automatic_budget,
            error_type=error_type,
        )
        assert resource_estimate is not None
        resource_estimate["resolved_integer_stride"] = common_integer_stride

    plan: FrameSelectionPlan = {}
    replica_reports: List[Dict[str, object]] = []
    source_total = 0
    selected_total = 0
    for replica_input in replica_inputs:
            system_id = str(replica_input["system_id"])
            replica_id = str(replica_input["replica_id"])
            segments = replica_input["segments"]
            assert isinstance(segments, list)
            replica_source = int(replica_input["source_frame_count"])

            if mode in {
                "uniform_per_replica_budget_v1", "auto_resource_budget_v1"
            }:
                requested_budget: Optional[int] = int(
                    selection["maximum_frames_per_replica"]
                    if mode == "uniform_per_replica_budget_v1"
                    else automatic_budget
                )
                selected_global: Optional[Set[int]] = (
                    uniform_indices(
                        replica_source, requested_budget, error_type=error_type
                    )
                    if mode == "uniform_per_replica_budget_v1"
                    else integer_stride_indices(
                        replica_source,
                        int(common_integer_stride),
                        error_type=error_type,
                    )
                )
            elif mode == "integer_stride_per_replica_v1":
                requested_budget = None
                selected_global = integer_stride_indices(
                    replica_source,
                    int(common_integer_stride),
                    error_type=error_type,
                )
            elif mode == "fixed_stride_v1":
                requested_budget = None
                selected_global = None
            else:
                raise error_type(f"unsupported frame-selection mode {mode!r}")

            source_total += replica_source
            offset = 0
            replica_selected = 0
            segment_reports: List[Dict[str, object]] = []
            for segment_id, count in segments:
                if selected_global is None:
                    # Historical stride is explicitly segment-local: every new
                    # segment starts again at source frame zero.
                    indices = set(range(0, count, frame_stride))
                else:
                    indices = {
                        index - offset
                        for index in selected_global
                        if offset <= index < offset + count
                    }
                key = (system_id, replica_id, segment_id)
                replica_uses_all = (
                    mode in {
                        "auto_resource_budget_v1",
                        "integer_stride_per_replica_v1",
                    }
                    and selected_global is not None
                    and len(selected_global) == replica_source
                )
                plan[key] = (
                    None
                    if (
                        (mode == "fixed_stride_v1" and frame_stride == 1)
                        or replica_uses_all
                    )
                    else indices
                )
                replica_selected += len(indices)
                segment_reports.append({
                    "segment_id": segment_id,
                    "source_frame_count": count,
                    "selected_frame_count": len(indices),
                    "coverage_fraction": len(indices) / count,
                    "first_selected_source_frame_index": min(indices) if indices else None,
                    "last_selected_source_frame_index": max(indices) if indices else None,
                })
                offset += count
            selected_total += replica_selected
            if selected_global is None:
                spacing = {
                    "kind": "exact_integer_stride",
                    "random": False,
                    "minimum_source_frame_gap": frame_stride,
                    "maximum_source_frame_gap": frame_stride,
                    "mean_source_frame_gap": float(frame_stride),
                }
            elif mode == "uniform_per_replica_budget_v1":
                ordered = sorted(selected_global)
                gaps = [right - left for left, right in zip(ordered, ordered[1:])]
                spacing = {
                    "kind": "deterministic_full_timespan_near_stride",
                    "random": False,
                    "minimum_source_frame_gap": min(gaps) if gaps else None,
                    "maximum_source_frame_gap": max(gaps) if gaps else None,
                    "mean_source_frame_gap": (
                        sum(gaps) / len(gaps) if gaps else None
                    ),
                    "endpoint_frames_retained": len(ordered) > 1,
                }
            else:
                spacing = {
                    "kind": "exact_integer_stride",
                    "random": False,
                    "minimum_source_frame_gap": common_integer_stride,
                    "maximum_source_frame_gap": common_integer_stride,
                    "mean_source_frame_gap": float(common_integer_stride),
                    "starts_at_replica_frame_zero": True,
                    "last_source_frame_forced": False,
                }
            replica_reports.append({
                "system_id": system_id,
                "replica_id": replica_id,
                "source_frame_count": replica_source,
                "requested_frame_budget": requested_budget,
                "selected_frame_count": replica_selected,
                "coverage_fraction": replica_selected / replica_source,
                "selection_spacing": spacing,
                "segments": segment_reports,
            })

    if maximum_selected_frames is not None and selected_total > maximum_selected_frames:
        raise error_type(
            f"frame selection requests {selected_total} frames, exceeding the "
            f"module resource gate of {maximum_selected_frames}"
        )
    resolved_mode = (
        "fixed_stride_v1"
        if selected_total == source_total and frame_stride == 1
        else "integer_stride_per_replica_v1"
        if mode == "auto_resource_budget_v1"
        else mode
    )
    contract = (
        "Uniform deterministic sampling over each replica's concatenated segment "
        "frame order; endpoint frames are retained when the budget permits."
        if mode == "uniform_per_replica_budget_v1"
        else "One exact integer stride over each replica's concatenated segment frame order; frame zero is retained and the final frame is not forced."
        if mode in {"integer_stride_per_replica_v1", "auto_resource_budget_v1"}
        else "Evaluate every frame whose per-segment source index is divisible by frame_stride."
    )
    report = {
        "mode": mode,
        "resolved_mode": resolved_mode,
        "frame_stride": frame_stride,
        "resolved_integer_stride": common_integer_stride,
        "selection_contract": contract,
        "source_frame_count": source_total,
        "selected_frame_count": selected_total,
        "coverage_fraction": selected_total / source_total,
        "replicas": replica_reports,
    }
    if resource_estimate is not None:
        rate = float(resource_estimate["estimated_seconds_per_frame"])
        overhead = float(resource_estimate["fixed_overhead_seconds"])
        safety = float(resource_estimate["safety_factor"])
        resource_estimate.update({
            "estimated_full_wall_seconds": safety * (overhead + rate * source_total),
            "estimated_selected_wall_seconds": safety * (
                overhead + rate * selected_total
            ),
            "subsampling_triggered": selected_total < source_total,
            "subsampling_reason": (
                "estimated full-frame wall time exceeds target resource envelope"
                if selected_total < source_total
                else None
            ),
        })
        report["resource_estimate"] = resource_estimate
    return plan, report


def reader_frame_indices(
    selected_indices: Optional[Set[int]], periodic_policy: str
) -> Optional[Set[int]]:
    """Return safe reader-level filtering for a periodic reconstruction policy."""

    return None if periodic_policy == "unwrap_continuous" else selected_indices


def frame_selected(
    frame_index: int,
    selected_indices: Optional[Set[int]],
    frame_stride: int,
) -> bool:
    """Test selection while retaining fixed-stride backward compatibility."""

    return (
        frame_index in selected_indices
        if selected_indices is not None
        else frame_index % frame_stride == 0
    )
