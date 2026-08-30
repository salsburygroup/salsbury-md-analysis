"""Declared scalar threshold states with segment-safe occupancy and residence."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from .manifests import ManifestValidationError, load_json
from .moments import sample_summary
from .trajectory_features import (
    TrajectoryFeatureError,
    iter_feature_records,
    trajectory_features_project,
)
from .upstream_cache import load_cached_project_report
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
    segments: Sequence[Tuple[
        Mapping[str, object],
        Sequence[Mapping[str, object]]
        | Callable[[], Iterable[Mapping[str, object]]],
    ]],
    *,
    operator: str,
    threshold: float,
    sensitivity_thresholds: Sequence[float],
    meets_threshold_label: str,
    does_not_meet_threshold_label: str,
    retain_assignments: bool = True,
    retain_residence_runs: bool = True,
) -> Dict[str, object]:
    """Return two-state labels and segment-safe descriptive transition evidence."""

    if operator not in {"less_than_or_equal", "greater_than_or_equal"}:
        raise ScalarThresholdStateError("unsupported threshold operator")
    if not isinstance(retain_assignments, bool) or not isinstance(
        retain_residence_runs, bool
    ):
        raise ScalarThresholdStateError("retention controls must be boolean")
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
    run_lengths = {1: [], 2: []}
    complete_run_lengths = {1: [], 2: []}
    sensitivity_counts = {
        float(cutoff): {"meets": 0, "agreement": 0}
        for cutoff in sensitivity_thresholds
    }
    population_counts: Dict[Tuple[str, str], Dict[int, int]] = {}
    observation_count = 0

    def records_from(source):
        return iter(source() if callable(source) else source)

    for identity, record_source in segments:
        previous_state = None
        run_state = None
        run_start_frame = None
        run_last_frame = None
        run_length = 0
        run_ordinal = 0

        def finish_run(*, right_censored: bool) -> None:
            nonlocal run_ordinal
            assert run_state is not None
            assert run_start_frame is not None and run_last_frame is not None
            left_censored = run_ordinal == 0
            run_lengths[run_state].append(float(run_length))
            if not left_censored and not right_censored:
                complete_run_lengths[run_state].append(float(run_length))
            if retain_residence_runs:
                residence_runs.append({
                    **identity,
                    "state_id": run_state,
                    "state_label": (
                        meets_threshold_label
                        if run_state == 2 else does_not_meet_threshold_label
                    ),
                    "start_source_frame_index": run_start_frame,
                    "end_source_frame_index": run_last_frame,
                    "length_frames": run_length,
                    "left_boundary_censored": left_censored,
                    "right_boundary_censored": right_censored,
                })
            run_ordinal += 1

        for row in records_from(record_source):
            value = float(row["value"])
            if not math.isfinite(value):
                raise ScalarThresholdStateError("threshold-state value is non-finite")
            state_id = 2 if compare(value, threshold) else 1
            primary_flag = state_id == 2
            observation_count += 1
            key = (str(identity["system_id"]), str(identity["replica_id"]))
            counts = population_counts.setdefault(key, {1: 0, 2: 0})
            counts[state_id] += 1
            for cutoff, accumulator in sensitivity_counts.items():
                flag = compare(value, cutoff)
                accumulator["meets"] += int(flag)
                accumulator["agreement"] += int(flag == primary_flag)
            if retain_assignments:
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
            if previous_state is not None:
                transition_counts[(previous_state, state_id)] += 1
            previous_state = state_id
            frame_index = row["source_frame_index"]
            if run_state is None:
                run_state = state_id
                run_start_frame = frame_index
                run_length = 1
            elif state_id == run_state:
                run_length += 1
            else:
                finish_run(right_censored=False)
                run_state = state_id
                run_start_frame = frame_index
                run_length = 1
            run_last_frame = frame_index
        if run_state is None:
            raise ScalarThresholdStateError(
                "threshold-state analysis contains an empty trajectory segment"
            )
        finish_run(right_censored=True)
    if observation_count == 0:
        raise ScalarThresholdStateError("threshold-state analysis has no observations")
    residence_summary = []
    for state_id, state_label in (
        (1, does_not_meet_threshold_label), (2, meets_threshold_label)
    ):
        runs = run_lengths[state_id]
        complete = complete_run_lengths[state_id]
        residence_summary.append({
            "state_id": state_id,
            "state_label": state_label,
            "run_count": len(runs),
            "complete_run_count": len(complete),
            "all_run_length_summary_frames": sample_summary(
                runs
            ),
            "complete_run_length_summary_frames": (
                sample_summary(complete)
                if complete else None
            ),
        })
    sensitivity = []
    for cutoff in sensitivity_thresholds:
        accumulator = sensitivity_counts[float(cutoff)]
        sensitivity.append({
            "threshold": float(cutoff),
            "meets_threshold_count": accumulator["meets"],
            "meets_threshold_fraction": (
                accumulator["meets"] / observation_count
            ),
            "agreement_with_primary_count": accumulator["agreement"],
            "agreement_with_primary_fraction": (
                accumulator["agreement"] / observation_count
            ),
        })

    def population_row(counts: Mapping[int, int]) -> Dict[str, object]:
        evaluated = sum(counts.values())
        return {
            "evaluated_count": evaluated,
            "assigned_count": evaluated,
            "unassigned_or_noise_count": 0,
            "assigned_coverage_fraction": 1.0,
            "state_populations": [
                {
                    "state_id": state_id,
                    "count": counts[state_id],
                    "fraction_of_all_evaluated": counts[state_id] / evaluated,
                    "fraction_of_assigned": counts[state_id] / evaluated,
                }
                for state_id in (1, 2)
            ],
        }

    systems = sorted({system_id for system_id, _ in population_counts})
    system_populations = []
    for system_id in systems:
        counts = {1: 0, 2: 0}
        for (candidate, _), values in population_counts.items():
            if candidate == system_id:
                counts[1] += values[1]
                counts[2] += values[2]
        system_populations.append({
            "system_id": system_id, **population_row(counts)
        })
    replica_populations = [
        {
            "system_id": system_id,
            "replica_id": replica_id,
            **population_row(counts),
        }
        for (system_id, replica_id), counts in sorted(population_counts.items())
    ]
    by_system = {row["system_id"]: row for row in system_populations}
    pairwise = []
    for left_index, left_id in enumerate(systems[:-1]):
        for right_id in systems[left_index + 1:]:
            left = by_system[left_id]
            right = by_system[right_id]
            left_states = {
                row["state_id"]: row for row in left["state_populations"]
            }
            right_states = {
                row["state_id"]: row for row in right["state_populations"]
            }
            pairwise.append({
                "left_system_id": left_id,
                "right_system_id": right_id,
                "state_fraction_differences": [
                    {
                        "state_id": state_id,
                        "left_fraction_of_all_evaluated": left_states[state_id]["fraction_of_all_evaluated"],
                        "right_fraction_of_all_evaluated": right_states[state_id]["fraction_of_all_evaluated"],
                        "left_minus_right_fraction_of_all_evaluated": (
                            left_states[state_id]["fraction_of_all_evaluated"]
                            - right_states[state_id]["fraction_of_all_evaluated"]
                        ),
                        "left_fraction_of_assigned": left_states[state_id]["fraction_of_assigned"],
                        "right_fraction_of_assigned": right_states[state_id]["fraction_of_assigned"],
                        "left_minus_right_fraction_of_assigned": (
                            left_states[state_id]["fraction_of_assigned"]
                            - right_states[state_id]["fraction_of_assigned"]
                        ),
                    }
                    for state_id in (1, 2)
                ],
                "left_assigned_coverage_fraction": 1.0,
                "right_assigned_coverage_fraction": 1.0,
            })
    population_comparison = {
        "state_field": "state_id",
        "state_ids": [1, 2],
        "system_populations": system_populations,
        "replica_populations": replica_populations,
        "member_populations": [],
        "paired_member_state_coupling": None,
        "pairwise_system_differences": pairwise,
        "observation_independence": (
            "frames remain time-correlated within simulation replicas"
        ),
        "interpretation": (
            "descriptive frame fractions; replica and time-block uncertainty "
            "must be evaluated before inferential system comparisons"
        ),
    }
    return {
        "operator": operator,
        "primary_threshold": threshold,
        "state_dictionary": [
            {"state_id": 1, "state_label": does_not_meet_threshold_label, "meets_threshold": False},
            {"state_id": 2, "state_label": meets_threshold_label, "meets_threshold": True},
        ],
        "assignments": assignments if retain_assignments else None,
        "assignments_retained": retain_assignments,
        "state_population_comparison": population_comparison,
        "transition_counts_within_segments": [
            {"from_state_id": left, "to_state_id": right, "count": count}
            for (left, right), count in sorted(transition_counts.items())
        ],
        "residence_runs": residence_runs if retain_residence_runs else None,
        "residence_runs_retained": retain_residence_runs,
        "residence_by_state": residence_summary,
        "threshold_sensitivity": sensitivity,
        "observation_count": observation_count,
        "reducer_mode": "single_pass_streaming_state_and_sensitivity_reducers",
    }


def scalar_threshold_states_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    artifact_root_text = os.environ.get(
        "SALSBURY_MD_ANALYSIS_COLUMNAR_ARTIFACT_ROOT"
    )
    artifact_bundle = (
        AtomicColumnarBundle(Path(artifact_root_text))
        if artifact_root_text else None
    )
    upstream = load_cached_project_report(
        "trajectory_features",
        source,
        hash_content=hash_content,
        error_type=ScalarThresholdStateError,
    )
    trajectory_feature_source_mode = (
        "validated_upstream_report" if upstream is not None
        else "computed_from_project"
    )
    if upstream is None:
        upstream = trajectory_features_project(source, hash_content=hash_content)
    segments = upstream.get("segments")
    if not isinstance(segments, list):
        raise ScalarThresholdStateError("trajectory_features report has no segments")
    reports = []
    total = 0
    for request_ordinal, request in enumerate(settings["states"]):
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
            inline_records = feature.get("records")
            artifact = feature.get("columnar_artifact")
            if isinstance(inline_records, list):
                feature_count = len(inline_records)
            elif isinstance(artifact, dict) and isinstance(
                artifact.get("row_count"), int
            ):
                feature_count = int(artifact["row_count"])
            else:
                raise ScalarThresholdStateError(
                    "trajectory feature lacks exact record accounting"
                )
            total += feature_count
            if total > int(settings["maximum_observations"]):
                raise ScalarThresholdStateError("maximum_observations gate exceeded")
            value_index = int(request["value_index"])

            def record_source(feature=feature, value_index=value_index):
                for row in iter_feature_records(feature):
                    yield {
                        "source_frame_index": row["source_frame_index"],
                        "axis_kind": row["axis_kind"],
                        "axis_value": row["axis_value"],
                        "value": float(row["values"][value_index]),
                    }

            source_segments.append(({
                "system_id": segment["system_id"],
                "replica_id": segment["replica_id"],
                "segment_id": segment["segment_id"],
            }, record_source))
        analysis = analyze_threshold_state(
            source_segments,
            operator=str(request["operator"]),
            threshold=float(request["threshold"]),
            sensitivity_thresholds=request["sensitivity_thresholds"],
            meets_threshold_label=str(request["meets_threshold_label"]),
            does_not_meet_threshold_label=str(request["does_not_meet_threshold_label"]),
            retain_assignments=artifact_bundle is None,
            retain_residence_runs=artifact_bundle is None,
        )
        if artifact_bundle is not None:
            assignment_artifacts = []
            residence_artifacts = []
            compare = (
                (lambda value, cutoff: value <= cutoff)
                if request["operator"] == "less_than_or_equal"
                else (lambda value, cutoff: value >= cutoff)
            )
            threshold = float(request["threshold"])
            for segment_ordinal, (identity, record_source) in enumerate(
                source_segments
            ):
                frames = []
                axis_values = []
                values = []
                states = []
                meets = []
                run_states = []
                run_starts = []
                run_ends = []
                run_lengths = []
                run_left = []
                run_right = []
                active_state = None
                active_start = None
                active_last = None
                active_length = 0
                run_ordinal = 0
                axis_kind = None
                for row in record_source():
                    value = float(row["value"])
                    state_id = 2 if compare(value, threshold) else 1
                    frame = int(row["source_frame_index"])
                    frames.append(frame)
                    axis_values.append(float(row["axis_value"]))
                    values.append(value)
                    states.append(state_id)
                    meets.append(state_id == 2)
                    if axis_kind is None:
                        axis_kind = str(row["axis_kind"])
                    elif str(row["axis_kind"]) != axis_kind:
                        raise ScalarThresholdStateError(
                            "axis kind changes within one trajectory segment"
                        )
                    if active_state is None:
                        active_state = state_id
                        active_start = frame
                        active_length = 1
                    elif state_id == active_state:
                        active_length += 1
                    else:
                        run_states.append(active_state)
                        run_starts.append(active_start)
                        run_ends.append(active_last)
                        run_lengths.append(active_length)
                        run_left.append(run_ordinal == 0)
                        run_right.append(False)
                        run_ordinal += 1
                        active_state = state_id
                        active_start = frame
                        active_length = 1
                    active_last = frame
                if active_state is None:
                    raise ScalarThresholdStateError(
                        "threshold-state analysis contains an empty trajectory segment"
                    )
                run_states.append(active_state)
                run_starts.append(active_start)
                run_ends.append(active_last)
                run_lengths.append(active_length)
                run_left.append(run_ordinal == 0)
                run_right.append(True)
                prefix = (
                    f"state-{request_ordinal:04d}/"
                    f"segment-{segment_ordinal:05d}"
                )
                provenance = {
                    "module_id": "scalar_threshold_states",
                    "project_manifest_sha256": upstream[
                        "project_manifest_sha256"
                    ],
                    "input_content_signature_sha256": upstream[
                        "input_content_signature_sha256"
                    ],
                    "state_analysis_id": request["state_analysis_id"],
                    "state_dictionary": analysis["state_dictionary"],
                    **identity,
                }
                constants = dict(identity)
                constants["axis_kind"] = axis_kind
                assignment_artifacts.append(artifact_bundle.write_table(
                    f"{prefix}/assignments",
                    {
                        "source_frame_index": np.asarray(frames, dtype=np.int64),
                        "axis_value": np.asarray(axis_values, dtype=np.float64),
                        "value": np.asarray(values, dtype=np.float64),
                        "state_id": np.asarray(states, dtype=np.int8),
                        "meets_threshold": np.asarray(meets, dtype=np.bool_),
                    },
                    constants=constants,
                    provenance=provenance,
                ))
                residence_artifacts.append(artifact_bundle.write_table(
                    f"{prefix}/residence-runs",
                    {
                        "state_id": np.asarray(run_states, dtype=np.int8),
                        "start_source_frame_index": np.asarray(
                            run_starts, dtype=np.int64
                        ),
                        "end_source_frame_index": np.asarray(
                            run_ends, dtype=np.int64
                        ),
                        "length_frames": np.asarray(
                            run_lengths, dtype=np.int64
                        ),
                        "left_boundary_censored": np.asarray(
                            run_left, dtype=np.bool_
                        ),
                        "right_boundary_censored": np.asarray(
                            run_right, dtype=np.bool_
                        ),
                    },
                    constants=identity,
                    provenance=provenance,
                ))
            analysis.update({
                "assignments": None,
                "assignments_retained": True,
                "assignments_inline": False,
                "assignment_artifacts": assignment_artifacts,
                "residence_runs": None,
                "residence_runs_retained": True,
                "residence_runs_inline": False,
                "residence_run_artifacts": residence_artifacts,
            })
        reports.append({
            "state_analysis_id": request["state_analysis_id"],
            "question": request["question"],
            "feature_id": request["feature_id"],
            "value_index": request["value_index"],
            **analysis,
        })
    issues = [issue for issue in upstream.get("issues", []) if isinstance(issue, dict)]
    if artifact_bundle is not None:
        artifact_bundle.publish()
    source_physical_frames = max(
        int(report["observation_count"]) for report in reports
    )
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
        "trajectory_feature_source_mode": trajectory_feature_source_mode,
        "observation_count": total,
        "observation_accounting": {
            "selected_physical_frame_count": source_physical_frames,
            "symmetry_expanded_observation_count": source_physical_frames,
            "feature_observation_count": total,
        },
        "columnar_artifact_root": (
            str(artifact_bundle.output_root)
            if artifact_bundle is not None else None
        ),
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
        ColumnarArtifactError,
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
import numpy as np

from .columnar_artifacts import AtomicColumnarBundle, ColumnarArtifactError
