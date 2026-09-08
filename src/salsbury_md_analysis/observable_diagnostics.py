"""Observable-specific evidence from identity-preserving, already extracted series.

These descriptive and within-series diagnostics do not certify ensemble adequacy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .convergence import (
    ConvergenceAnalysisError,
    _series_diagnostic,
    _diagnostic_summary,
)
from .manifests import load_json, sha256_file
from .simulation_protocol import validate_simulation_protocol

TIME_TO_NS = {"fs": 1e-6, "ps": 1e-3, "ns": 1.0, "us": 1e3}


def _require(condition, message):
    if not condition:
        raise ConvergenceAnalysisError(message)


def _number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def validate_observable_input(data):
    """Require explicit units, definitions, denominators, and regular segments."""
    _require(
        data.get("schema_version") == "observable_timeseries_v1",
        "expected observable_timeseries_v1",
    )
    _require(
        set(data)
        <= {
            "schema_version",
            "source_provenance",
            "observables",
            "segments",
            "weighting",
            "joint_observables",
        },
        "unknown observable input fields",
    )
    _require(
        data.get("weighting") in ("frame", "replica_equal"),
        "weighting must be frame or replica_equal",
    )
    provenance = data.get("source_provenance")
    _require(
        isinstance(provenance, dict) and bool(provenance),
        "source_provenance is required",
    )
    _require(
        isinstance(provenance.get("description"), str)
        and bool(provenance["description"].strip()),
        "source_provenance.description is required",
    )
    digest = provenance.get("sha256", "")
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest),
        "source_provenance.sha256 is required",
    )
    definitions = data.get("observables")
    _require(
        isinstance(definitions, list) and bool(definitions),
        "observables must be a nonempty list",
    )
    by_id = {}
    for definition in definitions:
        _require(
            isinstance(definition, dict), "observable definition must be an object"
        )
        _require(
            set(definition)
            <= {
                "observable_id",
                "unit",
                "definition",
                "kind",
                "histogram_edges",
                "condition_observable_id",
            },
            "unknown observable definition fields",
        )
        for field in ("observable_id", "unit", "definition"):
            _require(
                isinstance(definition.get(field), str)
                and bool(definition[field].strip()),
                field + " must be explicit",
            )
        identifier = definition["observable_id"]
        _require(identifier not in by_id, "duplicate observable identity")
        _require(
            definition.get("kind") in ("scalar", "indicator"),
            "kind must be scalar or indicator",
        )
        edges = definition.get("histogram_edges")
        _require(
            isinstance(edges, list)
            and len(edges) >= 2
            and all(_number(x) for x in edges)
            and all(a < b for a, b in zip(edges, edges[1:])),
            "histogram_edges must be finite and increasing",
        )
        if definition["kind"] == "indicator":
            _require(
                definition["unit"] == "dimensionless",
                "indicator unit must be dimensionless",
            )
        by_id[identifier] = definition
    for definition in definitions:
        condition = definition.get("condition_observable_id")
        if condition is not None:
            _require(
                isinstance(condition, str)
                and condition in by_id
                and by_id[condition]["kind"] == "indicator"
                and condition != definition["observable_id"]
                and "condition_observable_id" not in by_id[condition],
                "condition must name an unconditional indicator",
            )
    segments = data.get("segments")
    _require(
        isinstance(segments, list) and bool(segments),
        "segments must be a nonempty list",
    )
    identities = set()
    for segment in segments:
        _require(isinstance(segment, dict), "segment must be an object")
        _require(
            set(segment)
            <= {
                "system_id",
                "replica_id",
                "segment_id",
                "rows",
                "frame_interval",
                "time_unit",
                "source_frame_stride",
                "simulation_protocol",
            },
            "unknown segment fields",
        )
        identity = tuple(
            segment.get(key) for key in ("system_id", "replica_id", "segment_id")
        )
        _require(
            all(isinstance(x, str) and bool(x.strip()) for x in identity),
            "complete segment identity is required",
        )
        _require(identity not in identities, "duplicate segment identity")
        identities.add(identity)
        interval, stride = segment.get("frame_interval"), segment.get(
            "source_frame_stride"
        )
        _require(
            _number(interval)
            and interval > 0
            and isinstance(segment.get("time_unit"), str)
            and segment.get("time_unit") in TIME_TO_NS,
            "positive frame_interval and physical time_unit are required",
        )
        _require(
            isinstance(stride, int) and not isinstance(stride, bool) and stride > 0,
            "source_frame_stride must be a positive integer",
        )
        if "simulation_protocol" in segment:
            validate_simulation_protocol(segment["simulation_protocol"])
        rows = segment.get("rows")
        _require(isinstance(rows, list) and bool(rows), "segment rows must be nonempty")
        for i, row in enumerate(rows):
            _require(
                isinstance(row, dict)
                and set(row) == {"source_frame_index", "time", "values"},
                "rows require source_frame_index, time, values",
            )
            frame = row["source_frame_index"]
            _require(
                isinstance(frame, int)
                and not isinstance(frame, bool)
                and frame >= 0
                and _number(row["time"]),
                "invalid source frame or time",
            )
            _require(
                isinstance(row["values"], dict) and set(row["values"]) == set(by_id),
                "every row must contain every declared observable",
            )
            for identifier, value in row["values"].items():
                _require(_number(value), "observable values must be finite numbers")
                if by_id[identifier]["kind"] == "indicator":
                    _require(value in (0, 1), "indicator values must be 0 or 1")
            if i:
                prior = rows[i - 1]
                _require(
                    frame - prior["source_frame_index"] == stride
                    and math.isclose(
                        row["time"] - prior["time"],
                        interval,
                        rel_tol=1e-7,
                        abs_tol=interval * 1e-9,
                    ),
                    "irregular or discontinuous series: split it into explicit continuous segments",
                )
    joints = data.get("joint_observables", [])
    _require(isinstance(joints, list), "joint_observables must be a list")
    seen = set()
    for pair in joints:
        _require(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(x, str) and x in by_id for x in pair)
            and pair[0] != pair[1],
            "joint observables must name two distinct observables",
        )
        _require(tuple(sorted(pair)) not in seen, "duplicate joint observable pair")
        seen.add(tuple(sorted(pair)))
    return by_id


def _runs(mask):
    start = None
    for index, keep in enumerate([*mask, False]):
        if keep and start is None:
            start = index
        elif not keep and start is not None:
            yield start, index
            start = None


def _distribution(values, weights, edges):
    total = float(sum(weights))
    counts = np.histogram(values, bins=edges, weights=weights)[0]
    return {
        "edges": edges,
        "probability": (counts / total).tolist() if total else None,
        "below_range_probability": (
            float(sum(w for v, w in zip(values, weights) if v < edges[0])) / total
            if total
            else None
        ),
        "above_range_probability": (
            float(sum(w for v, w in zip(values, weights) if v > edges[-1])) / total
            if total
            else None
        ),
    }


def analyze_observable_timeseries(data, settings):
    """Summarize each estimand separately, preserving conditional mixture weights."""
    definitions = validate_observable_input(data)
    metrics = settings["metrics"]
    _require(
        all(name in definitions for name in metrics),
        "metrics must name declared observables",
    )
    lags = settings.get("lag_frames", [1])
    _require(
        isinstance(lags, list)
        and bool(lags)
        and all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in lags)
        and len(set(lags)) == len(lags),
        "lag_frames must be unique positive integers",
    )
    replica_counts = defaultdict(int)
    for segment in data["segments"]:
        replica_counts[(segment["system_id"], segment["replica_id"])] += len(
            segment["rows"]
        )
    series, pooled, joints = [], defaultdict(list), defaultdict(list)
    for segment in data["segments"]:
        identity = {
            key: segment[key] for key in ("system_id", "replica_id", "segment_id")
        }
        rows = segment["rows"]
        interval_ns = segment["frame_interval"] * TIME_TO_NS[segment["time_unit"]]
        weight = (
            1.0
            if data["weighting"] == "frame"
            else 1.0 / replica_counts[(segment["system_id"], segment["replica_id"])]
        )
        for metric in metrics:
            definition = definitions[metric]
            condition = definition.get("condition_observable_id")
            values = [row["values"][metric] for row in rows]
            mask = [condition is None or row["values"][condition] == 1 for row in rows]
            for row, value, keep in zip(rows, values, mask):
                pooled[(segment["system_id"], metric)].append(
                    (value, weight, keep, segment["replica_id"])
                )
            temporal = []
            # Conditional observations are never concatenated across excluded intervals.
            for start, stop in _runs(mask):
                run = values[start:stop]
                try:
                    diagnostic = _series_diagnostic(run, settings)
                    status, reason = "available", None
                except ConvergenceAnalysisError as exc:
                    diagnostic, status, reason = None, "not calculable", str(exc)
                constant = len(set(run)) == 1
                temporal.append(
                    {
                        "start_source_frame_index": rows[start]["source_frame_index"],
                        "end_source_frame_index": rows[stop - 1]["source_frame_index"],
                        "observation_count": len(run),
                        "status": status,
                        "reason": reason,
                        "constant_series": constant,
                        "diagnostics": diagnostic,
                    }
                )
            lagged = []
            for lag in lags:
                pairs = [
                    (values[i], values[i + lag])
                    for start, stop in _runs(mask)
                    for i in range(start, stop - lag)
                ]
                if len(pairs) >= 2:
                    a, b = np.asarray(pairs, dtype=float).T
                    correlation = (
                        float(np.corrcoef(a, b)[0, 1])
                        if np.std(a) > 0 and np.std(b) > 0
                        else None
                    )
                else:
                    correlation = None
                lagged.append(
                    {
                        "lag_frames": lag,
                        "lag_time_ns": lag * interval_ns,
                        "pair_count": len(pairs),
                        "correlation": correlation,
                        "indicator_switch_count": (
                            sum(a != b for a, b in pairs)
                            if definition["kind"] == "indicator"
                            else None
                        ),
                    }
                )
            series.append(
                {
                    **identity,
                    "metric": metric,
                    "unit": definition["unit"],
                    "source_frame_count": len(rows),
                    "selected_count": sum(mask),
                    "conditioning_fraction": sum(mask) / len(mask),
                    "first_time_ns": rows[0]["time"] * TIME_TO_NS[segment["time_unit"]],
                    "last_time_ns": rows[-1]["time"] * TIME_TO_NS[segment["time_unit"]],
                    "frame_interval_ns": interval_ns,
                    "continuous_runs": temporal,
                    "lagged_diagnostics": lagged,
                    "kinetic_validity": "not assessed",
                }
            )
        for pair in data.get("joint_observables", []):
            for row in rows:
                keep = all(
                    "condition_observable_id" not in definitions[name]
                    or row["values"][definitions[name]["condition_observable_id"]] == 1
                    for name in pair
                )
                joints[(segment["system_id"], *pair)].append(
                    (row["values"][pair[0]], row["values"][pair[1]], weight, keep)
                )
    summaries = []
    for (system, metric), records in sorted(pooled.items()):
        selected = [row for row in records if row[2]]
        total = sum(row[1] for row in records)
        selected_weight = sum(row[1] for row in selected)
        values, weights = [r[0] for r in selected], [r[1] for r in selected]
        summaries.append(
            {
                "system_id": system,
                "metric": metric,
                **definitions[metric],
                "observation_count": len(records),
                "selected_count": len(selected),
                "replica_count": len({row[3] for row in records}),
                "conditioning_fraction": selected_weight / total,
                "mean": (
                    float(np.average(values, weights=weights)) if selected else None
                ),
                "distribution": _distribution(
                    values, weights, definitions[metric]["histogram_edges"]
                ),
                "population_validity": "not assessed",
                "kinetic_validity": "not assessed",
                "uncertainty_scope": "within-series diagnostics only; no pooled confidence interval is inferred",
            }
        )
    joint_summaries = []
    for (system, first, second), records in sorted(joints.items()):
        selected = [r for r in records if r[3]]
        mass = sum(r[2] for r in selected)
        hist, xedges, yedges = np.histogram2d(
            [r[0] for r in selected],
            [r[1] for r in selected],
            bins=[
                definitions[first]["histogram_edges"],
                definitions[second]["histogram_edges"],
            ],
            weights=[r[2] for r in selected],
        )
        joint_summaries.append(
            {
                "system_id": system,
                "observables": [first, second],
                "conditioning_fraction": mass / sum(r[2] for r in records),
                "selected_count": len(selected),
                "x_edges": xedges.tolist(),
                "y_edges": yedges.tolist(),
                "probability": (hist / mass).tolist() if mass else None,
                "outside_range_probability": (
                    max(0.0, 1.0 - float(hist.sum()) / mass) if mass else None
                ),
            }
        )
    flat_diagnostics = []
    for item in series:
        for run in item["continuous_runs"]:
            if run["diagnostics"] is not None:
                flat_diagnostics.append(
                    {
                        **{
                            key: item[key]
                            for key in (
                                "system_id",
                                "replica_id",
                                "segment_id",
                                "metric",
                                "unit",
                            )
                        },
                        "start_source_frame_index": run["start_source_frame_index"],
                        "end_source_frame_index": run["end_source_frame_index"],
                        **run["diagnostics"],
                    }
                )
    summary = _diagnostic_summary(flat_diagnostics, settings)
    summary["series_kind"] = "observable continuous runs"
    return {
        "module_id": "convergence_uncertainty",
        "source_module": "observable_timeseries",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "scientific_interpretation_status": "human_review_required",
        "scientific_conclusion_emitted": False,
        "diagnostic_summary": summary,
        "diagnostic_series": flat_diagnostics,
        "settings": settings,
        "weighting": data["weighting"],
        "source_provenance": data["source_provenance"],
        "observable_summaries": summaries,
        "series_diagnostics": series,
        "joint_distributions": joint_summaries,
        "numerical_sensitivity_status": "not assessed",
        "error_count": 0,
        "warning_count": 0,
        "issues": [],
        "limitations": [
            "Reported source provenance is declared; the input series file is independently hashed.",
            "Conditional kinetics use only continuous retained runs; no transition spans an excluded observation.",
            "Constant series do not establish adequate sampling or an informative representation.",
            "Population agreement does not establish kinetic agreement.",
        ],
    }


def observable_convergence_project(project_path, project, settings):
    path = Path(settings["timeseries_file"]).expanduser()
    if not path.is_absolute():
        path = Path(project_path).parent / path
    data = load_json(path)
    result = analyze_observable_timeseries(data, settings)
    result.update(
        {
            "project_manifest_path": str(project_path),
            "project_manifest_sha256": sha256_file(project_path),
            "input_timeseries_path": str(path.resolve()),
            "input_timeseries_sha256": sha256_file(path),
        }
    )
    return result
