"""Scientific presentation defaults for frame-indexed scalar observations."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .scalar_distributions import (
    ScalarDistributionError,
    analyze_scalar_distribution,
)


class PresentationError(ValueError):
    """Raised when generic time-series records cannot be summarized safely."""


_IDENTITY_OR_AXIS_FIELDS = {
    "system_id", "replica_id", "segment_id", "source_frame_index",
    "evaluated_frame_index", "frame_index", "time", "time_ps", "time_ns",
    "axis_kind", "axis_value", "time_unit", "frame_id",
}


def _is_rmsd_field(field: str) -> bool:
    normalized = field.strip().lower().replace("-", "_")
    return "rmsd" in normalized.split("_")


def _numeric_fields(
    segments: Sequence[Mapping[str, object]],
) -> List[str]:
    fields = set()
    for segment in segments:
        records = segment.get("records")
        if not isinstance(records, list):
            raise PresentationError("every segment must contain a records array")
        for record in records:
            if not isinstance(record, dict):
                raise PresentationError("time-series records must be objects")
            for field, value in record.items():
                if (
                    field not in _IDENTITY_OR_AXIS_FIELDS
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    fields.add(field)
    return sorted(fields)


def summarize_timeseries_presentations(
    segments: Sequence[Mapping[str, object]],
    *,
    fields: Optional[Sequence[str]] = None,
    maximum_observations_per_field: int = 1_000_000,
    padding_fraction: float = 0.0,
    minimum_bins: int = 2,
    maximum_bins: int = 100,
) -> Dict[str, object]:
    """Apply the suite's histogram-first rule to generic scalar series.

    RMSD is intentionally exempt: it stays replica-resolved and time-series
    first.  Every other finite scalar field is summarized with Scott's rule;
    its time series remains a secondary presentation with source-frame
    identities retained in the histogram assignments.
    """

    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise PresentationError("segments must be an array")
    if not segments:
        raise PresentationError("segments must be nonempty")
    if maximum_observations_per_field <= 0:
        raise PresentationError("maximum_observations_per_field must be positive")
    available = _numeric_fields(segments)
    selected = list(fields) if fields is not None else available
    if not selected or len(set(selected)) != len(selected):
        raise PresentationError("fields must be nonempty and unique")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise PresentationError("numeric fields are absent: " + ", ".join(unknown))

    presentations = []
    for field in selected:
        scalar_segments: List[
            Tuple[Mapping[str, object], Sequence[Mapping[str, object]]]
        ] = []
        time_series_record_count = 0
        for segment_index, segment in enumerate(segments):
            records = segment["records"]
            assert isinstance(records, list)
            identity = {
                key: value for key, value in segment.items() if key != "records"
            }
            identity.setdefault("segment_index", segment_index)
            scalar_records = []
            for record_index, record in enumerate(records):
                assert isinstance(record, dict)
                value = record.get(field)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    continue
                source_frame_index = record.get("source_frame_index", record_index)
                if (
                    isinstance(source_frame_index, bool)
                    or not isinstance(source_frame_index, int)
                    or source_frame_index < 0
                ):
                    raise PresentationError(
                        f"{field} has an invalid source_frame_index"
                    )
                scalar_records.append({
                    "source_frame_index": source_frame_index,
                    "value": float(value),
                    **{
                        key: record[key]
                        for key in ("axis_kind", "axis_value", "time_unit")
                        if key in record
                    },
                })
            if scalar_records:
                scalar_segments.append((identity, scalar_records))
                time_series_record_count += len(scalar_records)
        if time_series_record_count > maximum_observations_per_field:
            raise PresentationError(
                f"{field} has {time_series_record_count} observations, exceeding "
                f"maximum_observations_per_field={maximum_observations_per_field}"
            )
        if _is_rmsd_field(field):
            presentations.append({
                "field": field,
                "primary_presentation": "replica_resolved_time_series",
                "histogram_status": "not_applicable",
                "histogram_exemption": "RMSD is the explicit suite exception",
                "time_series_record_count": time_series_record_count,
            })
            continue
        try:
            distribution = analyze_scalar_distribution(
                scalar_segments,
                binning_rule="scott",
                padding_fraction=padding_fraction,
                minimum_bins=minimum_bins,
                maximum_bins=maximum_bins,
            )
            histogram_status = "complete"
            reason = None
        except ScalarDistributionError as exc:
            distribution = None
            histogram_status = "not_estimable"
            reason = str(exc)
        presentations.append({
            "field": field,
            "primary_presentation": "histogram",
            "secondary_presentation": "time_series",
            "histogram_rule": "scott",
            "histogram_status": histogram_status,
            "not_estimable_reason": reason,
            "time_series_record_count": time_series_record_count,
            "distribution": distribution,
        })
    return {
        "presentation_schema": "salsbury-timeseries-presentation-v1",
        "technical_status": "complete",
        "scientific_status": "presentation summary only",
        "rmsd_exception": True,
        "default_histogram_rule": "scott",
        "presentations": presentations,
        "limitations": [
            "A histogram is a primary display choice, not an independence or convergence claim.",
            "Scott bin widths are computed from pooled finite observations and clamped to declared bin-count bounds.",
            "Segment identities and source-frame indices are retained; residence runs never cross segment boundaries.",
        ],
    }
