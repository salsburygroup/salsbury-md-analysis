"""Histogram and segment-safe residence analysis for reusable scalar features."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from .manifests import ManifestValidationError, load_json
from .moments import sample_summary
from .pca_fes import PCAFESAnalysisError
from .trajectory_features import (
    TrajectoryFeatureError,
    iter_feature_records,
    trajectory_features_project,
)
from .upstream_cache import load_cached_project_report
from .validation import positive_integer


class ScalarDistributionError(ValueError):
    """Raised when a scalar distribution contract is incomplete."""


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("scalar_feature_distributions") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict) or set(raw) != {
        "source", "maximum_observations", "distributions"
    }:
        raise ScalarDistributionError(
            "definitions.scalar_feature_distributions must contain source, maximum_observations, and distributions"
        )
    if raw["source"] != "trajectory_features":
        raise ScalarDistributionError("scalar distribution source must be trajectory_features")
    maximum = positive_integer(
        raw["maximum_observations"], "maximum_observations",
        error_type=ScalarDistributionError,
    )
    rows = raw["distributions"]
    if not isinstance(rows, list) or not rows:
        raise ScalarDistributionError("distributions must be a nonempty array")
    identifiers = set()
    normalized = []
    common = {
        "distribution_id", "question", "feature_id", "value_index",
        "binning_rule", "padding_fraction",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ScalarDistributionError(f"distribution {index} must be an object")
        rule = row.get("binning_rule")
        expected = common | (
            {"bin_count"}
            if rule == "explicit"
            else {"minimum_bins", "maximum_bins"}
        )
        if rule not in {"explicit", "scott", "freedman_diaconis", "rice"} or set(row) != expected:
            raise ScalarDistributionError(
                f"distribution {index} fields do not match its binning rule"
            )
        distribution_id = str(row["distribution_id"]).strip()
        question = str(row["question"]).strip()
        feature_id = str(row["feature_id"]).strip()
        if (
            not distribution_id or distribution_id in identifiers
            or not question or not feature_id
        ):
            raise ScalarDistributionError(
                "distribution IDs/questions/feature IDs must be nonempty and IDs unique"
            )
        value_index = row["value_index"]
        if isinstance(value_index, bool) or not isinstance(value_index, int) or value_index < 0:
            raise ScalarDistributionError("value_index must be a nonnegative integer")
        padding = row["padding_fraction"]
        if (
            isinstance(padding, bool) or not isinstance(padding, (int, float))
            or not math.isfinite(float(padding)) or float(padding) < 0.0
        ):
            raise ScalarDistributionError("padding_fraction must be finite and nonnegative")
        normalized_row = {**row, "padding_fraction": float(padding)}
        if rule == "explicit":
            normalized_row["bin_count"] = positive_integer(
                row["bin_count"], "bin_count", error_type=ScalarDistributionError
            )
        else:
            minimum = positive_integer(
                row["minimum_bins"], "minimum_bins", error_type=ScalarDistributionError
            )
            maximum_bins = positive_integer(
                row["maximum_bins"], "maximum_bins", error_type=ScalarDistributionError
            )
            if minimum < 2 or maximum_bins < minimum:
                raise ScalarDistributionError("automatic bin gates require 2 <= minimum <= maximum")
            normalized_row["minimum_bins"] = minimum
            normalized_row["maximum_bins"] = maximum_bins
        identifiers.add(distribution_id)
        normalized.append(normalized_row)
    return {
        "source": "trajectory_features",
        "maximum_observations": maximum,
        "distributions": normalized,
    }


def analyze_scalar_distribution(
    segments: Sequence[Tuple[
        Mapping[str, object],
        Sequence[Mapping[str, object]]
        | Callable[[], Iterable[Mapping[str, object]]],
    ]],
    *,
    binning_rule: str,
    padding_fraction: float,
    bin_count: int | None = None,
    minimum_bins: int = 2,
    maximum_bins: int = 100,
    retain_assignments: bool = True,
    retain_residence_runs: bool = True,
) -> Dict[str, object]:
    """Return histogram and segment-safe residence evidence.

    Assignment and individual-run rows are retained by default.  High-
    dimensional callers that already retain the raw time series may disable
    either duplicated table while preserving exact histograms and aggregated
    boundary-censored residence summaries.
    """

    if not isinstance(retain_assignments, bool) or not isinstance(
        retain_residence_runs, bool
    ):
        raise ScalarDistributionError("retention controls must be boolean")

    def records_from(source):
        return iter(source() if callable(source) else source)

    observation_count = 0
    minimum = math.inf
    maximum = -math.inf
    mean = 0.0
    centered_sum_squares = 0.0
    quantile_values = [] if binning_rule == "freedman_diaconis" else None
    for _, record_source in segments:
        for row in records_from(record_source):
            value = float(row["value"])
            if not math.isfinite(value):
                raise ScalarDistributionError(
                    "scalar distribution contains a non-finite observation"
                )
            observation_count += 1
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            delta = value - mean
            mean += delta / observation_count
            centered_sum_squares += delta * (value - mean)
            if quantile_values is not None:
                quantile_values.append(value)
    if observation_count < 2:
        raise ScalarDistributionError(
            "scalar distribution requires at least two finite observations"
        )
    span = maximum - minimum
    if span <= 0.0:
        raise ScalarDistributionError("scalar distribution is constant")
    if binning_rule == "explicit":
        if bin_count is None or bin_count < 2:
            raise ScalarDistributionError("explicit scalar distribution requires at least two bins")
        selected_bin_count = bin_count
        binning = {
            "rule": "explicit",
            "bins": bin_count,
            "observation_count": observation_count,
        }
    else:
        if binning_rule == "rice":
            raw_bin_count = int(
                math.ceil(2.0 * observation_count ** (1.0 / 3.0))
            )
            rule_width = (
                span * (1.0 + 2.0 * padding_fraction) / raw_bin_count
            )
        elif binning_rule == "scott":
            population_sd = math.sqrt(
                centered_sum_squares / observation_count
            )
            rule_width = (
                3.5 * population_sd * observation_count ** (-1.0 / 3.0)
            )
            if not math.isfinite(rule_width) or rule_width <= 0.0:
                raise ScalarDistributionError(
                    "scott bin width is undefined; use explicit bins"
                )
            raw_bin_count = int(math.ceil(
                span * (1.0 + 2.0 * padding_fraction) / rule_width
            ))
        elif binning_rule == "freedman_diaconis":
            assert quantile_values is not None
            quantile_values.sort()
            ordered = quantile_values
            def percentile(fraction: float) -> float:
                position = (observation_count - 1) * fraction
                lower_index = int(math.floor(position))
                upper_index = int(math.ceil(position))
                if lower_index == upper_index:
                    return ordered[lower_index]
                weight = position - lower_index
                return (
                    ordered[lower_index] * (1.0 - weight)
                    + ordered[upper_index] * weight
                )
            iqr = percentile(0.75) - percentile(0.25)
            rule_width = 2.0 * iqr * observation_count ** (-1.0 / 3.0)
            if not math.isfinite(rule_width) or rule_width <= 0.0:
                raise ScalarDistributionError(
                    "freedman_diaconis bin width is undefined; use explicit bins"
                )
            raw_bin_count = int(math.ceil(
                span * (1.0 + 2.0 * padding_fraction) / rule_width
            ))
        else:
            raise ScalarDistributionError("automatic binning rule is unsupported")
        raw_bin_count = max(1, raw_bin_count)
        selected_bin_count = min(
            maximum_bins, max(minimum_bins, raw_bin_count)
        )
        binning = {
            "rule": binning_rule,
            "observation_count": observation_count,
            "raw_bins": raw_bin_count,
            "bins": selected_bin_count,
            "rule_width": rule_width,
            "minimum_bins": minimum_bins,
            "maximum_bins": maximum_bins,
            "bin_count_clamped": selected_bin_count != raw_bin_count,
            "axis_contract": "the declared scalar feature is binned independently",
        }
    lower = minimum - padding_fraction * span
    upper = maximum + padding_fraction * span
    width = (upper - lower) / selected_bin_count

    def assign(value: float) -> int:
        return max(0, min(selected_bin_count - 1, int((value - lower) / width)))

    counts = [0] * selected_bin_count
    assignments = []
    runs = []
    run_lengths_by_bin = [[] for _ in range(selected_bin_count)]
    complete_run_lengths_by_bin = [[] for _ in range(selected_bin_count)]
    for identity, record_source in segments:
        run_bin = None
        run_start_frame = None
        run_last_frame = None
        run_length = 0
        run_ordinal = 0

        def finish_run(*, right_censored: bool) -> None:
            nonlocal run_ordinal
            assert run_bin is not None
            assert run_start_frame is not None and run_last_frame is not None
            left_censored = run_ordinal == 0
            run_lengths_by_bin[run_bin].append(float(run_length))
            if not left_censored and not right_censored:
                complete_run_lengths_by_bin[run_bin].append(float(run_length))
            if retain_residence_runs:
                runs.append({
                    **identity,
                    "bin_id": run_bin + 1,
                    "start_source_frame_index": run_start_frame,
                    "end_source_frame_index": run_last_frame,
                    "length_frames": run_length,
                    "left_boundary_censored": left_censored,
                    "right_boundary_censored": right_censored,
                })
            run_ordinal += 1

        for row in records_from(record_source):
            value = float(row["value"])
            bin_id = assign(value)
            counts[bin_id] += 1
            if retain_assignments:
                assignments.append({**identity, **row, "bin_id": bin_id + 1})
            frame_index = row["source_frame_index"]
            if run_bin is None:
                run_bin = bin_id
                run_start_frame = frame_index
                run_length = 1
            elif bin_id == run_bin:
                run_length += 1
            else:
                finish_run(right_censored=False)
                run_bin = bin_id
                run_start_frame = frame_index
                run_length = 1
            run_last_frame = frame_index
        if run_bin is None:
            raise ScalarDistributionError(
                "scalar distribution contains an empty trajectory segment"
            )
        finish_run(right_censored=True)
    histogram = [
        {
            "bin_id": index + 1,
            "lower_edge": lower + index * width,
            "upper_edge": lower + (index + 1) * width,
            "center": lower + (index + 0.5) * width,
            "count": count,
            "fraction": count / observation_count,
        }
        for index, count in enumerate(counts)
    ]
    residence = []
    for bin_id in range(1, selected_bin_count + 1):
        lengths = run_lengths_by_bin[bin_id - 1]
        complete_lengths = complete_run_lengths_by_bin[bin_id - 1]
        residence.append({
            "bin_id": bin_id,
            "run_count": len(lengths),
            "complete_run_count": len(complete_lengths),
            "all_run_length_summary_frames": sample_summary(lengths),
            "complete_run_length_summary_frames": (
                sample_summary(complete_lengths) if complete_lengths else None
            ),
        })
    return {
        "binning": binning,
        "bounds": {"lower": lower, "upper": upper},
        "bin_width": width,
        "observation_count": observation_count,
        "reducer_mode": (
            "two_pass_streaming_with_exact_quantile_buffer"
            if binning_rule == "freedman_diaconis" else
            "two_pass_streaming_constant_summary_memory"
        ),
        "histogram": histogram,
        "assignments": assignments if retain_assignments else None,
        "assignments_retained": retain_assignments,
        "residence_runs": runs if retain_residence_runs else None,
        "residence_runs_retained": retain_residence_runs,
        "residence_by_bin": residence,
    }


def scalar_feature_distributions_project(
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
        error_type=ScalarDistributionError,
    )
    trajectory_feature_source_mode = (
        "validated_upstream_report" if upstream is not None
        else "computed_from_project"
    )
    if upstream is None:
        upstream = trajectory_features_project(source, hash_content=hash_content)
    segments = upstream.get("segments")
    if not isinstance(segments, list):
        raise ScalarDistributionError("trajectory_features report has no segments")
    reports = []
    total = 0
    for request_ordinal, request in enumerate(settings["distributions"]):
        source_segments = []
        for segment in segments:
            assert isinstance(segment, dict)
            features = segment["features"]
            assert isinstance(features, list)
            matches = [
                feature for feature in features
                if isinstance(feature, dict)
                and feature.get("feature_id") == request["feature_id"]
            ]
            if len(matches) != 1:
                raise ScalarDistributionError(
                    f"feature {request['feature_id']} is absent or duplicated in a segment"
                )
            feature = matches[0]
            if int(request["value_index"]) >= int(feature["dimension"]):
                raise ScalarDistributionError("value_index exceeds feature dimension")
            inline_records = feature.get("records")
            artifact = feature.get("columnar_artifact")
            if isinstance(inline_records, list):
                feature_count = len(inline_records)
            elif isinstance(artifact, dict) and isinstance(
                artifact.get("row_count"), int
            ):
                feature_count = int(artifact["row_count"])
            else:
                raise ScalarDistributionError(
                    "trajectory feature lacks exact record accounting"
                )
            total += feature_count
            if total > int(settings["maximum_observations"]):
                raise ScalarDistributionError("maximum_observations gate exceeded")
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
        analysis = analyze_scalar_distribution(
            source_segments,
            binning_rule=str(request["binning_rule"]),
            padding_fraction=float(request["padding_fraction"]),
            bin_count=(int(request["bin_count"]) if "bin_count" in request else None),
            minimum_bins=int(request.get("minimum_bins", 2)),
            maximum_bins=int(request.get("maximum_bins", 100)),
            retain_assignments=artifact_bundle is None,
            retain_residence_runs=artifact_bundle is None,
        )
        if artifact_bundle is not None:
            assignment_artifacts = []
            residence_artifacts = []
            lower = float(analysis["bounds"]["lower"])
            width = float(analysis["bin_width"])
            bin_total = int(analysis["binning"]["bins"])
            for segment_ordinal, (identity, record_source) in enumerate(
                source_segments
            ):
                frames = []
                axis_values = []
                values = []
                bins = []
                run_bins = []
                run_starts = []
                run_ends = []
                run_lengths = []
                run_left = []
                run_right = []
                active_bin = None
                active_start = None
                active_last = None
                active_length = 0
                run_ordinal = 0
                axis_kind = None
                for row in record_source():
                    value = float(row["value"])
                    bin_id = max(
                        0,
                        min(bin_total - 1, int((value - lower) / width)),
                    )
                    frame = int(row["source_frame_index"])
                    frames.append(frame)
                    axis_values.append(float(row["axis_value"]))
                    if axis_kind is None:
                        axis_kind = str(row["axis_kind"])
                    elif str(row["axis_kind"]) != axis_kind:
                        raise ScalarDistributionError(
                            "axis kind changes within one trajectory segment"
                        )
                    values.append(value)
                    bins.append(bin_id + 1)
                    if active_bin is None:
                        active_bin = bin_id
                        active_start = frame
                        active_length = 1
                    elif bin_id == active_bin:
                        active_length += 1
                    else:
                        run_bins.append(active_bin + 1)
                        run_starts.append(active_start)
                        run_ends.append(active_last)
                        run_lengths.append(active_length)
                        run_left.append(run_ordinal == 0)
                        run_right.append(False)
                        run_ordinal += 1
                        active_bin = bin_id
                        active_start = frame
                        active_length = 1
                    active_last = frame
                if active_bin is None:
                    raise ScalarDistributionError(
                        "scalar distribution contains an empty trajectory segment"
                    )
                run_bins.append(active_bin + 1)
                run_starts.append(active_start)
                run_ends.append(active_last)
                run_lengths.append(active_length)
                run_left.append(run_ordinal == 0)
                run_right.append(True)
                prefix = (
                    f"distribution-{request_ordinal:04d}/"
                    f"segment-{segment_ordinal:05d}"
                )
                provenance = {
                    "module_id": "scalar_feature_distributions",
                    "project_manifest_sha256": upstream[
                        "project_manifest_sha256"
                    ],
                    "input_content_signature_sha256": upstream[
                        "input_content_signature_sha256"
                    ],
                    "distribution_id": request["distribution_id"],
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
                        "bin_id": np.asarray(bins, dtype=np.int32),
                    },
                    constants=constants,
                    provenance=provenance,
                ))
                residence_artifacts.append(artifact_bundle.write_table(
                    f"{prefix}/residence-runs",
                    {
                        "bin_id": np.asarray(run_bins, dtype=np.int32),
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
            "distribution_id": request["distribution_id"],
            "question": request["question"],
            "feature_id": request["feature_id"],
            "value_index": request["value_index"],
            **analysis,
        })
    issues = [
        issue for issue in upstream.get("issues", []) if isinstance(issue, dict)
    ]
    if artifact_bundle is not None:
        artifact_bundle.publish()
    source_physical_frames = max(
        int(report["observation_count"]) for report in reports
    )
    return {
        "module_id": "scalar_feature_distributions",
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
        "distribution_reports": reports,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Histogram rules, padding, feature definitions, and bin gates require sensitivity analysis.",
            "Residence runs never cross declared trajectory segments; boundary-censored runs are labeled and excluded from complete-run summaries.",
            "Frame-count residence summaries require physical-time conversion before comparison when frame intervals differ.",
            "Histogram occupancy and residence do not establish convergence, metastability, kinetics, or mechanism.",
        ],
    }


def scalar_feature_distributions_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return scalar_feature_distributions_project(
            project_path, hash_content=hash_content
        )
    except (
        ManifestValidationError,
        PCAFESAnalysisError,
        ColumnarArtifactError,
        ScalarDistributionError,
        TrajectoryFeatureError,
        OSError,
        KeyError,
        ValueError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "scalar_feature_distributions",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {
                    "severity": "error",
                    "code": "SCALAR_DISTRIBUTION_INVALID",
                    "message": message,
                }
                for message in messages
            ],
        }
import numpy as np

from .columnar_artifacts import AtomicColumnarBundle, ColumnarArtifactError
