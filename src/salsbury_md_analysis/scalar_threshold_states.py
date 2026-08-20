"""Declared scalar threshold states with segment-safe occupancy and residence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .manifests import ManifestValidationError, load_json
from .moments import sample_summary
from .state_populations import summarize_state_populations
from .trajectory_features import TrajectoryFeatureError, trajectory_features_project
from .validation import positive_integer


class ScalarThresholdStateError(ValueError):
    """Raised when a threshold-state contract is incomplete or ambiguous."""


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("scalar_threshold_states") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict) or set(raw) != {
        "source", "maximum_observations", "states"
    }:
        raise ScalarThresholdStateError(
            "definitions.scalar_threshold_states must contain source, maximum_observations, and states"
        )
    if raw["source"] != "trajectory_features":
        raise ScalarThresholdStateError("threshold-state source must be trajectory_features")
    maximum = positive_integer(
        raw["maximum_observations"], "maximum_observations",
        error_type=ScalarThresholdStateError,
    )
    states = raw["states"]
    if not isinstance(states, list) or not states:
        raise ScalarThresholdStateError("states must be a nonempty array")
    identifiers = set()
    normalized = []
    expected = {
        "state_analysis_id", "question", "feature_id", "value_index",
        "operator", "threshold", "sensitivity_thresholds",
        "meets_threshold_label", "does_not_meet_threshold_label",
    }
    for index, row in enumerate(states):
        if not isinstance(row, dict) or set(row) != expected:
            raise ScalarThresholdStateError(
                f"threshold state {index} fields do not match the contract"
            )
        identifier = str(row["state_analysis_id"]).strip()
        question = str(row["question"]).strip()
        feature_id = str(row["feature_id"]).strip()
        labels = (
            str(row["meets_threshold_label"]).strip(),
            str(row["does_not_meet_threshold_label"]).strip(),
        )
        if (
            not identifier or identifier in identifiers or not question
            or not feature_id or not all(labels) or labels[0] == labels[1]
        ):
            raise ScalarThresholdStateError(
                "threshold IDs, questions, feature IDs, and distinct labels are required"
            )
        value_index = row["value_index"]
        if isinstance(value_index, bool) or not isinstance(value_index, int) or value_index < 0:
            raise ScalarThresholdStateError("value_index must be a nonnegative integer")
        if row["operator"] not in {"less_than_or_equal", "greater_than_or_equal"}:
            raise ScalarThresholdStateError(
                "operator must be less_than_or_equal or greater_than_or_equal"
            )
        threshold = row["threshold"]
        sensitivity = row["sensitivity_thresholds"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not isinstance(sensitivity, list)
            or not sensitivity
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in sensitivity
            )
        ):
            raise ScalarThresholdStateError(
                "thresholds must be finite and sensitivity_thresholds nonempty"
            )
        thresholds = [float(value) for value in sensitivity]
        if len(set(thresholds)) != len(thresholds) or float(threshold) not in thresholds:
            raise ScalarThresholdStateError(
                "sensitivity_thresholds must be unique and include threshold"
            )
        identifiers.add(identifier)
        normalized.append({
            **row,
            "state_analysis_id": identifier,
            "question": question,
            "feature_id": feature_id,
            "threshold": float(threshold),
            "sensitivity_thresholds": sorted(thresholds),
            "meets_threshold_label": labels[0],
            "does_not_meet_threshold_label": labels[1],
        })
    return {
        "source": "trajectory_features",
        "maximum_observations": maximum,
        "states": normalized,
    }


def analyze_threshold_state(
    segments: Sequence[Tuple[Mapping[str, object], Sequence[Mapping[str, object]]]],
    *,
    operator: str,
    threshold: float,
    sensitivity_thresholds: Sequence[float],
    meets_threshold_label: str,
    does_not_meet_threshold_label: str,
) -> Dict[str, object]:
    """Return two-state labels and segment-safe descriptive transition evidence."""

    if operator not in {"less_than_or_equal", "greater_than_or_equal"}:
        raise ScalarThresholdStateError("unsupported threshold operator")
    compare = (
        (lambda value, cutoff: value <= cutoff)
        if operator == "less_than_or_equal"
        else (lambda value, cutoff: value >= cutoff)
    )
    assignments: List[Dict[str, object]] = []
    residence_runs: List[Dict[str, object]] = []
    transition_counts = {
        (1, 1): 0, (1, 2): 0, (2, 1): 0, (2, 2): 0,
    }
    values: List[float] = []
    for identity, records in segments:
        segment_states = []
        for row in records:
            value = float(row["value"])
            if not math.isfinite(value):
                raise ScalarThresholdStateError("threshold-state value is non-finite")
            state_id = 2 if compare(value, threshold) else 1
            values.append(value)
            segment_states.append(state_id)
            assignments.append({
                **identity,
                **row,
                "state_id": state_id,
                "state_label": (
                    meets_threshold_label
                    if state_id == 2 else does_not_meet_threshold_label
                ),
                "meets_threshold": state_id == 2,
            })
        for left, right in zip(segment_states, segment_states[1:]):
            transition_counts[(left, right)] += 1
        start = 0
        while start < len(records):
            end = start + 1
            while end < len(records) and segment_states[end] == segment_states[start]:
                end += 1
            residence_runs.append({
                **identity,
                "state_id": segment_states[start],
                "state_label": (
                    meets_threshold_label
                    if segment_states[start] == 2 else does_not_meet_threshold_label
                ),
                "start_source_frame_index": records[start]["source_frame_index"],
                "end_source_frame_index": records[end - 1]["source_frame_index"],
                "length_frames": end - start,
                "left_boundary_censored": start == 0,
                "right_boundary_censored": end == len(records),
            })
            start = end
    if not values:
        raise ScalarThresholdStateError("threshold-state analysis has no observations")
    residence_summary = []
    for state_id, state_label in (
        (1, does_not_meet_threshold_label), (2, meets_threshold_label)
    ):
        runs = [row for row in residence_runs if row["state_id"] == state_id]
        complete = [
            row for row in runs
            if not row["left_boundary_censored"] and not row["right_boundary_censored"]
        ]
        residence_summary.append({
            "state_id": state_id,
            "state_label": state_label,
            "run_count": len(runs),
            "complete_run_count": len(complete),
            "all_run_length_summary_frames": sample_summary(
                [float(row["length_frames"]) for row in runs]
            ),
            "complete_run_length_summary_frames": (
                sample_summary([float(row["length_frames"]) for row in complete])
                if complete else None
            ),
        })
    primary_flags = [compare(value, threshold) for value in values]
    sensitivity = []
    for cutoff in sensitivity_thresholds:
        flags = [compare(value, float(cutoff)) for value in values]
        sensitivity.append({
            "threshold": float(cutoff),
            "meets_threshold_count": sum(flags),
            "meets_threshold_fraction": sum(flags) / len(flags),
            "agreement_with_primary_count": sum(
                left == right for left, right in zip(primary_flags, flags)
            ),
            "agreement_with_primary_fraction": sum(
                left == right for left, right in zip(primary_flags, flags)
            ) / len(flags),
        })
    return {
        "operator": operator,
        "primary_threshold": threshold,
        "state_dictionary": [
            {"state_id": 1, "state_label": does_not_meet_threshold_label, "meets_threshold": False},
            {"state_id": 2, "state_label": meets_threshold_label, "meets_threshold": True},
        ],
        "assignments": assignments,
        "state_population_comparison": summarize_state_populations(
            assignments, "state_id"
        ),
        "transition_counts_within_segments": [
            {"from_state_id": left, "to_state_id": right, "count": count}
            for (left, right), count in sorted(transition_counts.items())
        ],
        "residence_runs": residence_runs,
        "residence_by_state": residence_summary,
        "threshold_sensitivity": sensitivity,
    }


def scalar_threshold_states_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    upstream = trajectory_features_project(source, hash_content=hash_content)
    segments = upstream.get("segments")
    if not isinstance(segments, list):
        raise ScalarThresholdStateError("trajectory_features report has no segments")
    reports = []
    total = 0
    for request in settings["states"]:
        source_segments = []
        for segment in segments:
            assert isinstance(segment, dict)
            matches = [
                feature for feature in segment["features"]
                if isinstance(feature, dict)
                and feature.get("feature_id") == request["feature_id"]
            ]
            if len(matches) != 1:
                raise ScalarThresholdStateError(
                    f"feature {request['feature_id']} is absent or duplicated in a segment"
                )
            feature = matches[0]
            if int(request["value_index"]) >= int(feature["dimension"]):
                raise ScalarThresholdStateError("value_index exceeds feature dimension")
            records = [
                {
                    "source_frame_index": row["source_frame_index"],
                    "axis_kind": row["axis_kind"],
                    "axis_value": row["axis_value"],
                    "value": float(row["values"][int(request["value_index"])]),
                }
                for row in feature["records"]
            ]
            total += len(records)
            if total > int(settings["maximum_observations"]):
                raise ScalarThresholdStateError("maximum_observations gate exceeded")
            source_segments.append(({
                "system_id": segment["system_id"],
                "replica_id": segment["replica_id"],
                "segment_id": segment["segment_id"],
            }, records))
        analysis = analyze_threshold_state(
            source_segments,
            operator=str(request["operator"]),
            threshold=float(request["threshold"]),
            sensitivity_thresholds=request["sensitivity_thresholds"],
            meets_threshold_label=str(request["meets_threshold_label"]),
            does_not_meet_threshold_label=str(request["does_not_meet_threshold_label"]),
        )
        reports.append({
            "state_analysis_id": request["state_analysis_id"],
            "question": request["question"],
            "feature_id": request["feature_id"],
            "value_index": request["value_index"],
            **analysis,
        })
    issues = [issue for issue in upstream.get("issues", []) if isinstance(issue, dict)]
    return {
        "module_id": "scalar_threshold_states",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": upstream["project_manifest_sha256"],
        "system_manifest_path": upstream["system_manifest_path"],
        "system_manifest_sha256": upstream["system_manifest_sha256"],
        "input_content_signature_sha256": upstream["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "observation_count": total,
        "state_reports": reports,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Thresholds and feature definitions must be frozen before comparative outcome review and evaluated by declared sensitivity scans.",
            "Residence runs and transitions never cross declared trajectory-segment boundaries; boundary-censored runs are labeled.",
            "Frame counts are descriptive and require physical-time conversion and correlation-aware uncertainty before kinetic interpretation.",
            "A distance threshold defines an operational binding state, not binding free energy, affinity, mechanism, or chemical coordination by itself.",
        ],
    }


def scalar_threshold_states_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return scalar_threshold_states_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, ScalarThresholdStateError,
        TrajectoryFeatureError, OSError, KeyError, ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "scalar_threshold_states",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {"severity": "error", "code": "SCALAR_THRESHOLD_STATE_INVALID", "message": message}
                for message in messages
            ],
        }
