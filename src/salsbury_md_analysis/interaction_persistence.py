"""Segment-safe temporal persistence of interaction-fingerprint features.

The module consumes the exact-frame sparse fingerprint report.  It never
reinterprets a frame missing from an upstream source as an interaction-negative
observation, and it reports selected-observation persistence rather than an
unobservable continuous-time bond lifetime between saved trajectory frames.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from .context import compile_project_context_file
from .interaction_fingerprints import (
    InteractionFingerprintError,
    interaction_fingerprints_project,
)
from .manifests import ManifestValidationError, load_json
from .upstream_cache import load_cached_project_report
from .validation import positive_integer


class InteractionPersistenceError(ValueError):
    """Raised when fingerprint residence events cannot be evaluated safely."""


SeriesKey = Tuple[str, str, str]


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("interaction_persistence")
        if isinstance(definitions, dict) else None
    )
    required = {
        "source_module", "gap_tolerance_observations",
        "minimum_observations_per_series", "minimum_complete_events",
        "maximum_features", "maximum_event_records",
        "maximum_interval_relative_deviation",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise InteractionPersistenceError(
            "definitions.interaction_persistence fields do not match the contract"
        )
    if raw["source_module"] != "interaction_fingerprints":
        raise InteractionPersistenceError(
            "source_module must be interaction_fingerprints"
        )
    tolerances = raw["gap_tolerance_observations"]
    if (
        not isinstance(tolerances, list)
        or not tolerances
        or 0 not in tolerances
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            or value < 0 or value > 10
            for value in tolerances
        )
        or len(set(tolerances)) != len(tolerances)
    ):
        raise InteractionPersistenceError(
            "gap_tolerance_observations must contain unique integers from 0 to 10 and include 0"
        )
    deviation = raw["maximum_interval_relative_deviation"]
    if (
        isinstance(deviation, bool)
        or not isinstance(deviation, (int, float))
        or not math.isfinite(float(deviation))
        or not 0.0 <= float(deviation) <= 1.0
    ):
        raise InteractionPersistenceError(
            "maximum_interval_relative_deviation must be finite and within [0, 1]"
        )
    result = dict(raw)
    result["gap_tolerance_observations"] = sorted(int(value) for value in tolerances)
    result["maximum_interval_relative_deviation"] = float(deviation)
    for name in (
        "minimum_observations_per_series", "minimum_complete_events",
        "maximum_features", "maximum_event_records",
    ):
        result[name] = positive_integer(
            raw[name], name, error_type=InteractionPersistenceError
        )
    return result


def _axis_contracts(context: Mapping[str, object]) -> Dict[SeriesKey, Dict[str, object]]:
    contract = context.get("contract")
    systems = contract.get("systems") if isinstance(contract, dict) else None
    if not isinstance(systems, list):
        raise InteractionPersistenceError("compiled context has no system axes")
    result: Dict[SeriesKey, Dict[str, object]] = {}
    for system in systems:
        if not isinstance(system, dict) or not isinstance(system.get("replicas"), list):
            raise InteractionPersistenceError("compiled system axis is malformed")
        for replica in system["replicas"]:
            if not isinstance(replica, dict) or not isinstance(replica.get("segments"), list):
                raise InteractionPersistenceError("compiled replica axis is malformed")
            for segment in replica["segments"]:
                if not isinstance(segment, dict):
                    raise InteractionPersistenceError("compiled segment axis is malformed")
                key = (
                    str(system["system_id"]), str(replica["replica_id"]),
                    str(segment["segment_id"]),
                )
                if segment.get("frame_axis_kind") != "physical_time":
                    result[key] = {"kind": "sample_index"}
                    continue
                timing = segment.get("timing")
                if not isinstance(timing, dict):
                    raise InteractionPersistenceError(
                        f"{'/'.join(key)} lacks physical timing"
                    )
                result[key] = {
                    "kind": "physical_time",
                    "first": float(timing["first_frame_time"]),
                    "interval": float(timing["frame_interval"]),
                    "unit": str(timing["unit"]),
                }
    return result


def _event_runs(states: Sequence[bool], gap_tolerance: int) -> list[Dict[str, int | bool]]:
    events: list[Dict[str, int | bool]] = []
    index = 0
    while index < len(states):
        if not states[index]:
            index += 1
            continue
        start = index
        last_positive = index
        positive_count = 1
        bridged_absent = 0
        pending_gap = 0
        cursor = index + 1
        while cursor < len(states):
            if states[cursor]:
                bridged_absent += pending_gap
                pending_gap = 0
                last_positive = cursor
                positive_count += 1
            else:
                pending_gap += 1
                if pending_gap > gap_tolerance:
                    break
            cursor += 1
        events.append({
            "start_observation_index": start,
            "end_observation_index": last_positive,
            "positive_observation_count": positive_count,
            "bridged_absent_observation_count": bridged_absent,
            "left_boundary_censored": start == 0,
            "right_boundary_censored": last_positive == len(states) - 1,
        })
        index = last_positive + 1
    return events


def _duration_summary(values: Sequence[float]) -> Dict[str, object]:
    if not values:
        return {
            "count": 0, "minimum": None, "q25": None, "median": None,
            "mean": None, "q75": None, "maximum": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "minimum": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(array.max()),
    }


def build_interaction_persistence(
    fingerprint_report: Mapping[str, object],
    settings: Mapping[str, object],
    axis_contracts: Mapping[SeriesKey, Mapping[str, object]],
) -> Dict[str, object]:
    """Build complete and boundary-censored persistence events."""

    if fingerprint_report.get("availability_status") != "available":
        return {
            "availability_status": "not_available",
            "availability_reason": "interaction_fingerprints_not_available",
            "event_records": [], "feature_persistence_summaries": [],
        }
    dictionary = fingerprint_report.get("feature_dictionary")
    frames = fingerprint_report.get("frame_fingerprints")
    if not isinstance(dictionary, list) or not isinstance(frames, list):
        raise InteractionPersistenceError("interaction-fingerprint report is incomplete")
    if len(dictionary) > int(settings["maximum_features"]):
        raise InteractionPersistenceError("fingerprint feature count exceeds maximum_features")
    source_by_feature: Dict[str, str] = {}
    definition_by_feature: Dict[str, Mapping[str, object]] = {}
    for row in dictionary:
        if not isinstance(row, dict) or not isinstance(row.get("feature_id"), str):
            raise InteractionPersistenceError("fingerprint feature dictionary is malformed")
        feature_id = str(row["feature_id"])
        source_by_feature[feature_id] = str(row.get("source_module"))
        definition_by_feature[feature_id] = row

    grouped: MutableMapping[SeriesKey, list[Mapping[str, object]]] = defaultdict(list)
    for row in frames:
        if not isinstance(row, dict):
            raise InteractionPersistenceError("fingerprint frame is not an object")
        try:
            key = (
                str(row["system_id"]), str(row["replica_id"]),
                str(row["segment_id"]),
            )
            int(row["source_frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InteractionPersistenceError(
                "fingerprint frame lacks a segment-safe source identity"
            ) from exc
        grouped[key].append(row)

    unavailable_series = []
    event_records: list[Dict[str, object]] = []
    events_by_summary: MutableMapping[
        Tuple[str, str, int], list[Dict[str, object]]
    ] = defaultdict(list)
    accounting: MutableMapping[
        Tuple[str, str, int], Dict[str, object]
    ] = defaultdict(lambda: {
        "evaluated_observation_count": 0,
        "present_observation_count": 0,
        "series_count": 0,
        "regular_series_count": 0,
    })
    for key, group_rows in sorted(grouped.items()):
        axis = axis_contracts.get(key)
        if axis is None or axis.get("kind") != "physical_time":
            unavailable_series.append({
                "system_id": key[0], "replica_id": key[1], "segment_id": key[2],
                "reason": "physical_time_required",
            })
            continue
        ordered = sorted(group_rows, key=lambda row: int(row["source_frame_index"]))
        indices = [int(row["source_frame_index"]) for row in ordered]
        if len(indices) != len(set(indices)):
            raise InteractionPersistenceError(f"duplicate fingerprint frame in {'/'.join(key)}")
        first = float(axis["first"])
        source_interval = float(axis["interval"])
        unit = str(axis["unit"])
        for feature_id, source_module in sorted(source_by_feature.items()):
            available = [
                row for row in ordered
                if source_module in row.get("available_source_modules", [])
            ]
            if len(available) < int(settings["minimum_observations_per_series"]):
                continue
            source_indices = [int(row["source_frame_index"]) for row in available]
            axis_values = [first + value * source_interval for value in source_indices]
            intervals = [right - left for left, right in zip(axis_values, axis_values[1:])]
            if not intervals or any(value <= 0.0 for value in intervals):
                raise InteractionPersistenceError(
                    f"non-increasing source-observed interval in {'/'.join(key)}"
                )
            evaluated_interval = float(np.median(np.asarray(intervals, dtype=float)))
            deviation = max(abs(value - evaluated_interval) for value in intervals)
            relative_deviation = deviation / evaluated_interval
            if relative_deviation > float(settings["maximum_interval_relative_deviation"]):
                unavailable_series.append({
                    "system_id": key[0], "replica_id": key[1], "segment_id": key[2],
                    "feature_id": feature_id,
                    "reason": "irregular_source_observation_interval",
                    "maximum_interval_relative_deviation": relative_deviation,
                })
                continue
            states = [
                feature_id in row.get("present_feature_ids", []) for row in available
            ]
            for gap_tolerance in settings["gap_tolerance_observations"]:  # type: ignore[index]
                account = accounting[(feature_id, key[0], int(gap_tolerance))]
                account["evaluated_observation_count"] = int(
                    account["evaluated_observation_count"]
                ) + len(states)
                account["present_observation_count"] = int(
                    account["present_observation_count"]
                ) + sum(states)
                account["series_count"] = int(account["series_count"]) + 1
                account["regular_series_count"] = int(account["regular_series_count"]) + 1
                for event_index, event in enumerate(
                    _event_runs(states, int(gap_tolerance)), start=1
                ):
                    start_position = int(event["start_observation_index"])
                    end_position = int(event["end_observation_index"])
                    record = {
                        "feature_id": feature_id,
                        "source_module": source_module,
                        "interaction_type": definition_by_feature[feature_id].get(
                            "interaction_type"
                        ),
                        "system_id": key[0], "replica_id": key[1],
                        "segment_id": key[2], "event_index": event_index,
                        "gap_tolerance_observations": int(gap_tolerance),
                        "start_source_frame_index": source_indices[start_position],
                        "end_source_frame_index": source_indices[end_position],
                        "start_time": axis_values[start_position],
                        "end_time": axis_values[end_position],
                        "duration": (end_position - start_position + 1) * evaluated_interval,
                        "time_unit": unit,
                        "evaluated_observation_interval": evaluated_interval,
                        **event,
                    }
                    record["complete_event"] = not (
                        bool(record["left_boundary_censored"])
                        or bool(record["right_boundary_censored"])
                    )
                    event_records.append(record)
                    events_by_summary[(
                        feature_id, key[0], int(gap_tolerance)
                    )].append(record)
                    if len(event_records) > int(settings["maximum_event_records"]):
                        raise InteractionPersistenceError(
                            "interaction persistence event count exceeds maximum_event_records"
                        )

    summaries = []
    for (feature_id, system_id, gap_tolerance), account in sorted(accounting.items()):
        events = events_by_summary[(feature_id, system_id, gap_tolerance)]
        complete = [float(row["duration"]) for row in events if row["complete_event"]]
        all_durations = [float(row["duration"]) for row in events]
        units = sorted({str(row["time_unit"]) for row in events})
        if len(units) > 1:
            raise InteractionPersistenceError(
                f"feature {feature_id} mixes physical time units"
            )
        evaluated = int(account["evaluated_observation_count"])
        present = int(account["present_observation_count"])
        minimum_complete = int(settings["minimum_complete_events"])
        summaries.append({
            "feature_id": feature_id,
            "source_module": source_by_feature[feature_id],
            "interaction_type": definition_by_feature[feature_id].get("interaction_type"),
            "system_id": system_id,
            "gap_tolerance_observations": gap_tolerance,
            "policy": (
                "continuous_source_observed_presence_v1"
                if gap_tolerance == 0
                else "intermittent_source_observed_presence_v1"
            ),
            "time_unit": units[0] if units else None,
            **account,
            "occupancy_fraction": present / evaluated if evaluated else None,
            "event_count": len(events),
            "complete_event_count": len(complete),
            "boundary_censored_event_count": len(events) - len(complete),
            "all_event_duration_summary": _duration_summary(all_durations),
            "complete_event_duration_summary": _duration_summary(complete),
            "minimum_complete_events": minimum_complete,
            "persistence_summary_gate": (
                "passed" if len(complete) >= minimum_complete
                else "insufficient_complete_events"
            ),
        })
    primary = [row for row in summaries if row["gap_tolerance_observations"] == 0]
    passed = [row for row in primary if row["persistence_summary_gate"] == "passed"]
    return {
        "availability_status": "available" if summaries else "not_available",
        "availability_reason": None if summaries else "no_regular_physical_time_feature_series",
        "source_module": "interaction_fingerprints",
        "event_records": event_records,
        "feature_persistence_summaries": summaries,
        "primary_continuous_summary_count": len(primary),
        "primary_continuous_pass_count": len(passed),
        "persistence_readiness_status": (
            "available_with_complete_events" if passed
            else "insufficient_complete_events"
        ),
        "unavailable_series": unavailable_series,
        "duration_contract": (
            "duration is the span of consecutive source-observed positive snapshots, "
            "including explicitly bridged negative snapshots for intermittent policies; "
            "behavior between saved or evaluated frames is not observed"
        ),
    }


def interaction_persistence_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    fingerprint_report = load_cached_project_report(
        "interaction_fingerprints", source, hash_content=hash_content,
        error_type=InteractionPersistenceError,
    )
    if fingerprint_report is None:
        fingerprint_report = interaction_fingerprints_project(
            source, hash_content=hash_content
        )
    context = compile_project_context_file(source, hash_content=hash_content)
    result = build_interaction_persistence(
        fingerprint_report, settings, _axis_contracts(context)
    )
    issues = [issue for issue in context.get("issues", []) if isinstance(issue, dict)]
    if result["availability_status"] == "not_available":
        issues.append({
            "severity": "warning", "code": "INTERACTION_PERSISTENCE_NOT_AVAILABLE",
            "message": str(result["availability_reason"]),
        })
    elif result["persistence_readiness_status"] == "insufficient_complete_events":
        issues.append({
            "severity": "warning", "code": "INTERACTION_PERSISTENCE_EVENTS_INSUFFICIENT",
            "message": "No primary feature/system summary passed the complete-event gate.",
        })
    return {
        "module_id": "interaction_persistence",
        "technical_status": "complete", "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": context["system_manifest_path"],
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings, **result,
        "evaluated_frame_count": len(fingerprint_report.get("frame_fingerprints", [])),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Missing source observations are never encoded as absent interactions.",
            "Durations describe persistence across saved and evaluated snapshots; interactions may break between observations.",
            "Events never cross system, replica, or segment boundaries, and boundary-censored events remain labeled.",
            "Gap-tolerant persistence is a declared sensitivity analysis and must not replace the zero-gap primary result.",
            "Occupancy and persistence do not establish binding free energy, affinity, causality, or mechanism.",
            "Frames and events are not independent-replica uncertainty units.",
        ],
    }


def interaction_persistence_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return interaction_persistence_project(project_path, hash_content=hash_content)
    except (
        InteractionPersistenceError, InteractionFingerprintError,
        ManifestValidationError, OSError, KeyError, TypeError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "interaction_persistence",
            "technical_status": "failed", "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "INTERACTION_PERSISTENCE_INVALID",
                "message": message,
            } for message in messages],
        }
