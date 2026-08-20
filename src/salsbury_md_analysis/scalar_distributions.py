"""Histogram and segment-safe residence analysis for reusable scalar features."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .manifests import ManifestValidationError, load_json
from .moments import sample_summary
from .pca_fes import PCAFESAnalysisError, select_bin_counts
from .trajectory_features import (
    TrajectoryFeatureError,
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
    segments: Sequence[Tuple[Mapping[str, object], Sequence[Mapping[str, object]]]],
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

    flattened = [
        (identity, row)
        for identity, records in segments for row in records
    ]
    values = [float(row["value"]) for _, row in flattened]
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ScalarDistributionError(
            "scalar distribution requires at least two finite observations"
        )
    span = max(values) - min(values)
    if span <= 0.0:
        raise ScalarDistributionError("scalar distribution is constant")
    if binning_rule == "explicit":
        if bin_count is None or bin_count < 2:
            raise ScalarDistributionError("explicit scalar distribution requires at least two bins")
        selected_bin_count = bin_count
        binning = {
            "rule": "explicit",
            "bins": bin_count,
            "observation_count": len(values),
        }
    else:
        try:
            selected = select_bin_counts(
                [(value, float(index)) for index, value in enumerate(values)],
                binning_rule,
                padding_fraction,
                minimum_bins,
                maximum_bins,
            )
        except PCAFESAnalysisError as exc:
            raise ScalarDistributionError(str(exc)) from exc
        selected_bin_count = int(selected["bins_x"])
        binning = {
            "rule": selected["rule"],
            "observation_count": selected["observation_count"],
            "raw_bins": selected["raw_bins_x"],
            "bins": selected_bin_count,
            "rule_width": selected["rule_width_x"],
            "minimum_bins": selected["minimum_bins_per_axis"],
            "maximum_bins": selected["maximum_bins_per_axis"],
            "bin_count_clamped": selected["bin_count_clamped"],
            "axis_contract": "the declared scalar feature is binned independently",
        }
    lower = min(values) - padding_fraction * span
    upper = max(values) + padding_fraction * span
    width = (upper - lower) / selected_bin_count

    def assign(value: float) -> int:
        return max(0, min(selected_bin_count - 1, int((value - lower) / width)))

    counts = [0] * selected_bin_count
    assignments = []
    runs = []
    run_lengths_by_bin = [[] for _ in range(selected_bin_count)]
    complete_run_lengths_by_bin = [[] for _ in range(selected_bin_count)]
    for identity, records in segments:
        segment_assignments = [assign(float(row["value"])) for row in records]
        for row, bin_id in zip(records, segment_assignments):
            counts[bin_id] += 1
            if retain_assignments:
                assignments.append({**identity, **row, "bin_id": bin_id + 1})
        run_start = 0
        while run_start < len(records):
            run_end = run_start + 1
            while (
                run_end < len(records)
                and segment_assignments[run_end] == segment_assignments[run_start]
            ):
                run_end += 1
            first = records[run_start]
            last = records[run_end - 1]
            zero_based_bin = segment_assignments[run_start]
            length = run_end - run_start
            left_censored = run_start == 0
            right_censored = run_end == len(records)
            run_lengths_by_bin[zero_based_bin].append(float(length))
            if not left_censored and not right_censored:
                complete_run_lengths_by_bin[zero_based_bin].append(float(length))
            if retain_residence_runs:
                runs.append({
                    **identity,
                    "bin_id": zero_based_bin + 1,
                    "start_source_frame_index": first["source_frame_index"],
                    "end_source_frame_index": last["source_frame_index"],
                    "length_frames": length,
                    "left_boundary_censored": left_censored,
                    "right_boundary_censored": right_censored,
                })
            run_start = run_end
    histogram = [
        {
            "bin_id": index + 1,
            "lower_edge": lower + index * width,
            "upper_edge": lower + (index + 1) * width,
            "center": lower + (index + 0.5) * width,
            "count": count,
            "fraction": count / len(values),
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
    for request in settings["distributions"]:
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
            records = []
            for row in feature["records"]:
                value = float(row["values"][int(request["value_index"])])
                records.append({
                    "source_frame_index": row["source_frame_index"],
                    "axis_kind": row["axis_kind"],
                    "axis_value": row["axis_value"],
                    "value": value,
                })
            total += len(records)
            if total > int(settings["maximum_observations"]):
                raise ScalarDistributionError("maximum_observations gate exceeded")
            source_segments.append(({
                "system_id": segment["system_id"],
                "replica_id": segment["replica_id"],
                "segment_id": segment["segment_id"],
            }, records))
        analysis = analyze_scalar_distribution(
            source_segments,
            binning_rule=str(request["binning_rule"]),
            padding_fraction=float(request["padding_fraction"]),
            bin_count=(int(request["bin_count"]) if "bin_count" in request else None),
            minimum_bins=int(request.get("minimum_bins", 2)),
            maximum_bins=int(request.get("maximum_bins", 100)),
        )
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
