"""Timeseries convergence and uncertainty diagnostics without scientific verdicts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from .geometry import GeometryError
from .manifests import ManifestValidationError, load_json
from .moments import sample_summary
from .rmsd_rg import RMSDRGError, replica_rmsd_rg_project


class ConvergenceAnalysisError(ValueError):
    """Raised when convergence inputs or technical contracts are incomplete."""


def _exact_observation_accounting(
    upstream: Mapping[str, object], metric_count: int
) -> Dict[str, object]:
    """Reconcile the distinct frame and scalar-value workloads exactly."""

    selected = 0
    source = 0
    segment_count = 0
    for system in upstream.get("systems", []):
        if not isinstance(system, dict):
            continue
        for replica in system.get("replicas", []):
            if not isinstance(replica, dict):
                continue
            for segment in replica.get("segments", []):
                if not isinstance(segment, dict):
                    continue
                rows = segment.get("timeseries")
                if not isinstance(rows, list):
                    raise ConvergenceAnalysisError(
                        "upstream RMSD/Rg segment lacks an exact timeseries"
                    )
                evaluated = segment.get("evaluated_frame_count", len(rows))
                if (
                    isinstance(evaluated, bool)
                    or not isinstance(evaluated, int)
                    or evaluated != len(rows)
                ):
                    raise ConvergenceAnalysisError(
                        "upstream RMSD/Rg evaluated-frame count does not match its timeseries"
                    )
                source_frames = segment.get("source_frame_count", evaluated)
                if (
                    isinstance(source_frames, bool)
                    or not isinstance(source_frames, int)
                    or source_frames < evaluated
                ):
                    raise ConvergenceAnalysisError(
                        "upstream RMSD/Rg source-frame count is invalid"
                    )
                selected += evaluated
                source += source_frames
                segment_count += 1
    if selected <= 0 or segment_count <= 0:
        raise ConvergenceAnalysisError(
            "upstream RMSD/Rg report contains no exactly accounted frames"
        )
    return {
        "source_physical_frame_count": source,
        "selected_physical_frame_count": selected,
        "symmetry_expanded_observation_count": selected,
        "metric_value_observation_count": selected * metric_count,
        "metric_count_per_selected_frame": metric_count,
        "segment_count": segment_count,
        "subsampling_triggered": selected < source,
        "observation_contract": (
            "One convergence state observation per selected physical trajectory frame; "
            "the separately reported metric-value workload multiplies frames by metrics."
        ),
    }


def autocorrelation_sequence(
    values: Sequence[float], maximum_lag: int | None = None
) -> Dict[str, object]:
    """Return an overlap-normalized autocorrelation sequence including lag zero."""

    count = len(values)
    if count < 2:
        return {
            "observation_count": count,
            "maximum_lag": 0,
            "autocorrelation": [1.0] if count == 1 else [],
            "constant_series": False,
        }
    if any(not math.isfinite(float(value)) for value in values):
        raise ConvergenceAnalysisError("autocorrelation values must be finite")
    if maximum_lag is None:
        lag_limit = count - 1
    elif isinstance(maximum_lag, bool) or not isinstance(maximum_lag, int) or maximum_lag < 0:
        raise ConvergenceAnalysisError("maximum_lag must be a nonnegative integer")
    else:
        lag_limit = min(maximum_lag, count - 1)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    if variance <= 0.0:
        return {
            "observation_count": count,
            "maximum_lag": lag_limit,
            "autocorrelation": [1.0] + [None] * lag_limit,
            "constant_series": True,
            "normalization_contract": "undefined beyond lag zero for a zero-variance series",
        }
    if count < 512:
        correlations: List[object] = [1.0]
        for lag in range(1, lag_limit + 1):
            covariance = sum(
                (values[index] - mean) * (values[index + lag] - mean)
                for index in range(count - lag)
            ) / (count - lag)
            correlations.append(covariance / variance)
        algorithm = "direct_overlap_normalized_v1"
    else:
        centered = np.asarray(values, dtype=float) - mean
        transform_length = 1 << (2 * count - 1).bit_length()
        spectrum = np.fft.rfft(centered, n=transform_length)
        unnormalized = np.fft.irfft(
            spectrum * np.conjugate(spectrum), n=transform_length
        )[:lag_limit + 1]
        overlaps = np.arange(count, count - lag_limit - 1, -1, dtype=float)
        normalized = (unnormalized / overlaps) / variance
        normalized[0] = 1.0
        correlations = [float(value) for value in normalized]
        algorithm = "fft_overlap_normalized_v1"
    return {
        "observation_count": count,
        "maximum_lag": lag_limit,
        "autocorrelation": correlations,
        "constant_series": False,
        "algorithm": algorithm,
        "normalization_contract": "mean-centered covariance normalized by the population variance and the number of overlapping pairs at each lag",
    }


def effective_sample_size(values: Sequence[float]) -> Dict[str, object]:
    """Estimate ESS using an initial-positive autocorrelation sequence."""

    count = len(values)
    if count < 2:
        return {"observation_count": count, "integrated_autocorrelation_time_frames": None, "effective_sample_size": None, "positive_lag_count": 0}
    autocorrelation = autocorrelation_sequence(values)
    if autocorrelation["constant_series"]:
        return {"observation_count": count, "integrated_autocorrelation_time_frames": None, "effective_sample_size": None, "positive_lag_count": 0}
    positive = []
    for correlation in autocorrelation["autocorrelation"][1:]:
        if correlation is None or float(correlation) <= 0.0:
            break
        positive.append(float(correlation))
    tau = max(1.0, 1.0 + 2.0 * sum(positive))
    return {
        "observation_count": count,
        "integrated_autocorrelation_time_frames": tau,
        "effective_sample_size": min(float(count), count / tau),
        "positive_lag_count": len(positive),
        "positive_autocorrelation_sequence": positive,
    }


def autocorrelation_adjusted_mean_uncertainty(
    values: Sequence[float], confidence_z: float = 1.96
) -> Dict[str, object]:
    """Return an exploratory ESS-adjusted uncertainty interval for a mean.

    This is a single-timeseries diagnostic.  It does not use replica agreement
    or leave-one-replica-out behavior, and it is not an acceptance gate.
    """

    if len(values) < 2:
        raise ConvergenceAnalysisError(
            "autocorrelation-adjusted uncertainty requires at least two observations"
        )
    if (
        isinstance(confidence_z, bool)
        or not isinstance(confidence_z, (int, float))
        or not math.isfinite(float(confidence_z))
        or float(confidence_z) <= 0.0
    ):
        raise ConvergenceAnalysisError("confidence_z must be finite and positive")
    summary = sample_summary(values)
    ess = effective_sample_size(values)
    effective = ess["effective_sample_size"]
    sample_sd = summary["sample_sd"]
    if effective is None or sample_sd is None or float(effective) <= 1.0:
        standard_error = None
        lower = None
        upper = None
    else:
        standard_error = float(sample_sd) / math.sqrt(float(effective))
        half_width = float(confidence_z) * standard_error
        lower = float(summary["mean"]) - half_width
        upper = float(summary["mean"]) + half_width
    return {
        "mean": summary["mean"],
        "sample_sd": sample_sd,
        "effective_sample_size": effective,
        "standard_error": standard_error,
        "confidence_z": float(confidence_z),
        "interval_lower": lower,
        "interval_upper": upper,
        "status": "exploratory_autocorrelation_adjusted",
        "acceptance_gate": False,
        "interpretation": (
            "Approximate within-timeseries uncertainty using the initial-positive-sequence ESS; "
            "not evidence of independent sampling or biological replication."
        ),
    }


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("convergence_uncertainty") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise ConvergenceAnalysisError("definitions.convergence_uncertainty must be an object")
    required = {
        "source_module", "metrics", "block_size_frames", "include_partial_final_block",
        "minimum_blocks",
    }
    missing = sorted(required.difference(raw))
    canonical_references = {
        "effective_sample_size_reference",
        "split_mean_difference_reference_in_sd",
    }
    legacy_references = {
        "minimum_effective_sample_size",
        "maximum_split_mean_difference_in_sd",
    }
    if canonical_references.intersection(raw) and legacy_references.intersection(raw):
        raise ConvergenceAnalysisError(
            "canonical diagnostic reference fields cannot be mixed with legacy threshold aliases"
        )
    has_canonical_references = canonical_references.issubset(raw)
    has_legacy_references = legacy_references.issubset(raw)
    if has_canonical_references == has_legacy_references:
        raise ConvergenceAnalysisError(
            "convergence settings must contain either the canonical diagnostic "
            "reference fields or the complete legacy threshold pair, but not both"
        )
    optional = {
        "replica_diagnostics",
        "minimum_replicas_for_exploratory_diagnostic",
        "minimum_replicas_for_population_validity",
    }
    reference_fields = canonical_references | legacy_references
    unknown = sorted(set(raw).difference(required | optional | reference_fields))
    if missing:
        raise ConvergenceAnalysisError("convergence settings missing: " + ", ".join(missing))
    if unknown:
        raise ConvergenceAnalysisError("convergence settings contain unknown fields: " + ", ".join(unknown))
    if raw["source_module"] != "replica_rmsd_rg":
        raise ConvergenceAnalysisError("source_module currently supports only replica_rmsd_rg")
    metrics = raw["metrics"]
    allowed = {"alignment_rmsd_angstrom", "rmsd_angstrom", "radius_of_gyration_angstrom"}
    if not isinstance(metrics, list) or not metrics or any(value not in allowed for value in metrics) or len(set(metrics)) != len(metrics):
        raise ConvergenceAnalysisError("metrics must contain unique RMSD/Rg metric names")
    for label in ("block_size_frames", "minimum_blocks"):
        value = raw[label]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConvergenceAnalysisError(f"{label} must be a positive integer")
    if not isinstance(raw["include_partial_final_block"], bool):
        raise ConvergenceAnalysisError("include_partial_final_block must be boolean")
    replica_diagnostics = raw.get("replica_diagnostics", False)
    if not isinstance(replica_diagnostics, bool):
        raise ConvergenceAnalysisError("replica_diagnostics must be boolean")
    if (
        "minimum_replicas_for_exploratory_diagnostic" in raw
        and "minimum_replicas_for_population_validity" in raw
    ):
        raise ConvergenceAnalysisError(
            "use only minimum_replicas_for_exploratory_diagnostic; the population-validity "
            "name is retained only as a legacy input alias"
        )
    minimum_replicas = raw.get(
        "minimum_replicas_for_exploratory_diagnostic",
        raw.get("minimum_replicas_for_population_validity", 2),
    )
    if (
        isinstance(minimum_replicas, bool)
        or not isinstance(minimum_replicas, int)
        or minimum_replicas <= 0
    ):
        raise ConvergenceAnalysisError(
            "minimum_replicas_for_exploratory_diagnostic must be a positive integer"
        )
    if has_canonical_references:
        effective_sample_size_reference = raw["effective_sample_size_reference"]
        split_mean_difference_reference = raw["split_mean_difference_reference_in_sd"]
    else:
        effective_sample_size_reference = raw["minimum_effective_sample_size"]
        split_mean_difference_reference = raw["maximum_split_mean_difference_in_sd"]
    for label, value in (
        ("effective_sample_size_reference", effective_sample_size_reference),
        ("split_mean_difference_reference_in_sd", split_mean_difference_reference),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ConvergenceAnalysisError(f"{label} must be finite and positive")
    return {
        "source_module": "replica_rmsd_rg", "metrics": list(metrics),
        "block_size_frames": raw["block_size_frames"],
        "include_partial_final_block": raw["include_partial_final_block"],
        "minimum_blocks": raw["minimum_blocks"],
        "effective_sample_size_reference": float(effective_sample_size_reference),
        "split_mean_difference_reference_in_sd": float(split_mean_difference_reference),
        "replica_diagnostics": replica_diagnostics,
        "minimum_replicas_for_exploratory_diagnostic": minimum_replicas,
    }


def _block_means(values: Sequence[float], size: int, include_partial: bool) -> List[float]:
    blocks = []
    for start in range(0, len(values), size):
        block = values[start:start + size]
        if len(block) < size and not include_partial:
            continue
        if block:
            blocks.append(sum(block) / len(block))
    return blocks


def _minimum_observations_for_blocks(
    block_size: int, minimum_blocks: int, include_partial: bool
) -> int:
    """Return the smallest series that can yield the declared block count."""

    if include_partial:
        return block_size * (minimum_blocks - 1) + 1
    return block_size * minimum_blocks


def _series_diagnostic(values: Sequence[float], settings: Mapping[str, object]) -> Dict[str, object]:
    if len(values) < 2:
        raise ConvergenceAnalysisError("convergence series requires at least two observations")
    block_size = int(settings["block_size_frames"])
    minimum_blocks = int(settings["minimum_blocks"])
    include_partial = bool(settings["include_partial_final_block"])
    minimum_observations = _minimum_observations_for_blocks(
        block_size,
        minimum_blocks,
        include_partial,
    )
    if len(values) < minimum_observations:
        raise ConvergenceAnalysisError(
            "convergence block contract is impossible: "
            f"{len(values)} selected observations cannot yield "
            f"{minimum_blocks} blocks of {block_size} observations "
            f"(minimum required {minimum_observations}); generate the block size "
            "from the upstream selected-frame count or provide a compatible "
            "explicit value"
        )
    summary = sample_summary(values)
    midpoint = len(values) // 2
    first_mean = sum(values[:midpoint]) / midpoint
    second_mean = sum(values[midpoint:]) / (len(values) - midpoint)
    sd = summary["sample_sd"]
    scale = max(float(sd) if sd is not None else 0.0, abs(float(summary["mean"])), 1.0e-12)
    split_difference = abs(first_mean - second_mean) / scale
    ess = effective_sample_size(values)
    adjusted_uncertainty = autocorrelation_adjusted_mean_uncertainty(values)
    blocks = _block_means(values, block_size, include_partial)
    block_contract_satisfied = len(blocks) >= minimum_blocks
    ess_reference = float(settings["effective_sample_size_reference"])
    effective = ess["effective_sample_size"]
    if effective is None:
        ess_relation = "unavailable"
        ess_difference = None
    else:
        ess_difference = float(effective) - ess_reference
        if float(effective) > ess_reference:
            ess_relation = "above_reference"
        elif float(effective) < ess_reference:
            ess_relation = "below_reference"
        else:
            ess_relation = "at_reference"
    split_reference = float(settings["split_mean_difference_reference_in_sd"])
    if split_difference < split_reference:
        split_relation = "below_reference"
    elif split_difference > split_reference:
        split_relation = "above_reference"
    else:
        split_relation = "at_reference"
    return {
        "summary": summary, "first_half_mean": first_mean, "second_half_mean": second_mean,
        "split_mean_difference_in_sd": split_difference,
        "block_means": blocks, "block_mean_summary": sample_summary(blocks),
        "block_contract": {
            "block_size_selected_observations": block_size,
            "minimum_blocks": minimum_blocks,
            "include_partial_final_block": include_partial,
            "minimum_required_observations": minimum_observations,
            "selected_observation_count": len(values),
            "compatible": True,
        },
        "effective_sample_size": ess,
        "effective_sample_size_reference": ess_reference,
        "effective_sample_size_reference_relation": ess_relation,
        "effective_sample_size_minus_reference": ess_difference,
        "autocorrelation_adjusted_mean_uncertainty": adjusted_uncertainty,
        "split_mean_difference_reference_in_sd": split_reference,
        "split_mean_difference_reference_relation": split_relation,
        "block_contract_satisfied": block_contract_satisfied,
        "scientific_conclusion": None,
    }


def _diagnostic_summary(
    diagnostics: Sequence[Mapping[str, object]], settings: Mapping[str, object]
) -> Dict[str, object]:
    """Summarize diagnostic values without assigning a scientific verdict."""

    def summarize(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        ess_values = sorted(
            float(row["effective_sample_size"]["effective_sample_size"])
            for row in rows
            if isinstance(row.get("effective_sample_size"), dict)
            and row["effective_sample_size"].get("effective_sample_size") is not None
        )
        ess_relations = [
            str(row.get("effective_sample_size_reference_relation", "unavailable"))
            for row in rows
        ]
        split_relations = [
            str(row.get("split_mean_difference_reference_relation", "unavailable"))
            for row in rows
        ]
        if ess_values:
            midpoint = len(ess_values) // 2
            median = (
                ess_values[midpoint]
                if len(ess_values) % 2
                else (ess_values[midpoint - 1] + ess_values[midpoint]) / 2.0
            )
        else:
            median = None
        return {
            "series_count": len(rows),
            "effective_sample_size_available_count": len(ess_values),
            "effective_sample_size_above_reference_count": ess_relations.count("above_reference"),
            "effective_sample_size_at_reference_count": ess_relations.count("at_reference"),
            "effective_sample_size_below_reference_count": ess_relations.count("below_reference"),
            "effective_sample_size_unavailable_count": ess_relations.count("unavailable"),
            "effective_sample_size_minimum": ess_values[0] if ess_values else None,
            "effective_sample_size_median": median,
            "effective_sample_size_maximum": ess_values[-1] if ess_values else None,
            "split_mean_difference_below_reference_count": split_relations.count("below_reference"),
            "split_mean_difference_at_reference_count": split_relations.count("at_reference"),
            "split_mean_difference_above_reference_count": split_relations.count("above_reference"),
        }

    by_metric = {}
    for metric in sorted({str(row["metric"]) for row in diagnostics}):
        by_metric[metric] = summarize(
            [row for row in diagnostics if str(row["metric"]) == metric]
        )
    by_system = {}
    for system_id in sorted({str(row["system_id"]) for row in diagnostics}):
        by_system[system_id] = summarize(
            [row for row in diagnostics if str(row["system_id"]) == system_id]
        )
    return {
        **summarize(diagnostics),
        "effective_sample_size_reference": float(settings["effective_sample_size_reference"]),
        "split_mean_difference_reference_in_sd": float(
            settings["split_mean_difference_reference_in_sd"]
        ),
        "by_metric": by_metric,
        "by_system": by_system,
        "interpretation_policy": (
            "Quantitative diagnostics are reported for human scientific review; "
            "the software does not emit a convergence or population-validity verdict."
        ),
    }


def convergence_uncertainty_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    upstream = replica_rmsd_rg_project(source, hash_content=hash_content)
    observation_accounting = _exact_observation_accounting(
        upstream, len(settings["metrics"])
    )
    issues = [issue for issue in upstream.get("issues", []) if isinstance(issue, dict)]
    diagnostics = []
    replica_means: Dict[tuple, List[float]] = {}
    system_replicas: Dict[str, set] = {}
    for system in upstream["systems"]:
        system_id = str(system["system_id"])
        system_replicas.setdefault(system_id, set())
        for replica in system["replicas"]:
            replica_id = str(replica["replica_id"])
            system_replicas[system_id].add(replica_id)
            for segment in replica["segments"]:
                rows = segment["timeseries"]
                for metric in settings["metrics"]:
                    values = [float(row[metric]) for row in rows]
                    diagnostic = _series_diagnostic(values, settings)
                    diagnostics.append({
                        "system_id": system_id, "replica_id": replica_id,
                        "segment_id": str(segment["segment_id"]), "metric": metric,
                        **diagnostic,
                    })
                    replica_means.setdefault((system_id, replica_id, metric), []).extend(values)
    uncertainty = []
    if bool(settings["replica_diagnostics"]):
        for system_id in sorted(system_replicas):
            for metric in settings["metrics"]:
                means = []
                for replica_id in sorted(system_replicas[system_id]):
                    values = replica_means.get((system_id, replica_id, metric), [])
                    if values:
                        means.append({
                            "replica_id": replica_id,
                            "mean": sum(values) / len(values),
                        })
                summary = sample_summary([row["mean"] for row in means])
                uncertainty.append({
                    "system_id": system_id,
                    "metric": metric,
                    "replica_count": len(means),
                    "replica_means": means,
                    "descriptive_between_replica_summary": summary,
                    "status": "optional_exploratory",
                    "recommended": False,
                    "acceptance_gate": False,
                })
    small_replica_ensemble = bool(settings["replica_diagnostics"]) and any(
        len(values) < int(settings["minimum_replicas_for_exploratory_diagnostic"])
        for values in system_replicas.values()
    )
    if small_replica_ensemble:
        issues.append({
            "severity": "warning", "code": "REPLICA_DIAGNOSTIC_SMALL_ENSEMBLE",
            "location": str(source),
            "message": (
                "optional replica diagnostics were requested and one or more systems "
                "have fewer simulations than the exploratory diagnostic target"
            ),
        })
    diagnostic_summary = _diagnostic_summary(diagnostics, settings)
    return {
        "module_id": "convergence_uncertainty", "technical_status": "complete",
        "scientific_status": "not evaluated",
        "scientific_interpretation_status": "human_review_required",
        "scientific_conclusion_emitted": False,
        "project_manifest_path": str(source),
        "project_manifest_sha256": upstream["project_manifest_sha256"],
        "system_manifest_path": upstream["system_manifest_path"],
        "system_manifest_sha256": upstream["system_manifest_sha256"],
        "input_content_signature_sha256": upstream["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "observation_accounting": observation_accounting,
        "series_diagnostics": diagnostics, "replica_diagnostics": uncertainty,
        "diagnostic_summary": diagnostic_summary,
        "replica_diagnostics_enabled": bool(settings["replica_diagnostics"]),
        "replica_count_gate_passed": None,
        "additional_replica_simulations_may_be_useful": (
            True if small_replica_ensemble else None
        ),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "ESS uses an initial-positive autocorrelation estimate and is a diagnostic, not a proof of independent sampling.",
            "Time blocks and split means are convergence diagnostics, not automatically independent replicates.",
            "Replica agreement and leave-one-replica-out behavior are not acceptance measures and are not calculated.",
            "Optional replica summaries are descriptive only and may suggest collecting additional independent simulations.",
            "Within-timeseries uncertainty uses approximate ESS and time blocks; it does not manufacture independent replicas.",
            "The software reports quantitative diagnostics and does not assign a convergence or population-validity verdict.",
        ],
    }


def convergence_uncertainty_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return convergence_uncertainty_project(project_path, hash_content=hash_content)
    except (ManifestValidationError, ConvergenceAnalysisError, RMSDRGError, GeometryError, OSError) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "convergence_uncertainty", "technical_status": "failed",
            "scientific_status": "not evaluated",
            "scientific_interpretation_status": "not_available",
            "scientific_conclusion_emitted": False,
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{"severity": "error", "code": "CONVERGENCE_INVALID", "message": message} for message in messages],
        }
