"""Bounded, observable-specific comparisons of declared integration steps."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .convergence import ConvergenceAnalysisError
from .manifests import ManifestValidationError, load_json, sha256_file
from .observable_diagnostics import (
    TIME_TO_NS,
    _number,
    _require,
    validate_observable_input,
)
from .simulation_protocol import SimulationProtocolError


def _blocks(data, rng=None, block_length=1):
    """Bootstrap within each original segment; each sampled block stays separate."""
    for segment in data["segments"]:
        rows = segment["rows"]
        identity = (segment["system_id"], segment["replica_id"])
        if rng is None:
            yield identity, rows
            continue
        remaining = len(rows)
        while remaining:
            start = int(rng.integers(0, len(rows) - block_length + 1))
            length = min(remaining, block_length)
            yield identity, rows[start : start + length]
            remaining -= length


def _statistic(data, blocks, definition, comparison):
    metric = definition["observable_id"]
    condition = definition.get("condition_observable_id")
    kind = comparison["statistic"]
    counts = defaultdict(int)
    for identity, rows in blocks:
        counts[identity] += len(rows)
    values, weights, left, right, pair_weights = [], [], [], [], []
    all_weight = selected_weight = 0.0
    lag = comparison.get("lag_frames", 1)
    for identity, rows in blocks:
        weight = 1.0 if data["weighting"] == "frame" else 1.0 / counts[identity]
        mask = [condition is None or row["values"][condition] == 1 for row in rows]
        for row, keep in zip(rows, mask):
            all_weight += weight
            if keep:
                selected_weight += weight
                values.append(row["values"][metric])
                weights.append(weight)
        if kind in {"lagged_correlation", "switch_probability"}:
            # All intermediate observations must satisfy the condition.
            excluded_prefix = np.r_[0, np.cumsum(np.logical_not(mask))]
            for i in range(len(rows) - lag):
                if excluded_prefix[i + lag + 1] == excluded_prefix[i]:
                    left.append(rows[i]["values"][metric])
                    right.append(rows[i + lag]["values"][metric])
                    pair_weights.append(weight)
    if kind == "conditioning_fraction":
        return selected_weight / all_weight
    if not values:
        return None
    if kind == "mean":
        return float(np.average(values, weights=weights))
    if kind == "distribution":
        edges = definition["histogram_edges"]
        hist = np.histogram(values, bins=edges, weights=weights)[0]
        tails = [
            sum(w for v, w in zip(values, weights) if v < edges[0]),
            sum(w for v, w in zip(values, weights) if v > edges[-1]),
        ]
        return np.r_[hist, tails] / sum(weights)
    if not left:
        return None
    if kind == "switch_probability":
        return float(
            np.average(np.asarray(left) != np.asarray(right), weights=pair_weights)
        )
    a, b, w = np.asarray(left), np.asarray(right), np.asarray(pair_weights)
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    denominator = np.sqrt(np.average(a * a, weights=w) * np.average(b * b, weights=w))
    return (
        float(np.average(a * b, weights=w) / denominator) if denominator > 0 else None
    )


def _difference(a, b, statistic):
    if a is None or b is None:
        return None
    return (
        float(0.5 * np.abs(b - a).sum())
        if statistic == "distribution"
        else float(b - a)
    )


def _protocol_contract(reference, candidate):
    protocols = []
    for data in (reference, candidate):
        rows = [segment.get("simulation_protocol") for segment in data["segments"]]
        if any(row is None for row in rows):
            return {
                "status": "not assessed",
                "reason": "Missing simulation_protocol; saved-frame timing cannot establish the integration step.",
            }
        first = rows[0]
        if any(row != first for row in rows[1:]):
            return {
                "status": "not assessed",
                "reason": "Each comparison arm must use one declared simulation protocol.",
            }
        protocols.append(first)
    left, right = protocols
    changed = sorted(
        key for key in left if left[key] != right[key] and key != "protocol_id"
    )
    if changed != ["integration_step_fs"]:
        return {
            "status": "not assessed",
            "reason": "A timestep comparison must change only integration_step_fs; identical steps or other protocol changes do not isolate integration sensitivity.",
            "changed_fields": changed,
        }
    return {
        "status": "eligible",
        "reference_protocol": left,
        "candidate_protocol": right,
        "reason": "Declared Hamiltonian, initial ensemble, integrator, thermostat, constraints, temperature and ensemble match.",
    }


def compare_numerical_protocols(config, base_path=Path(".")):
    _require(
        isinstance(config, dict)
        and set(config)
        <= {
            "reference_file",
            "candidate_file",
            "comparisons",
            "bootstrap_repeats",
            "block_length_frames",
            "confidence_level",
            "random_seed",
        },
        "unknown numerical comparison fields",
    )
    inputs, hashes = [], {}
    for key in ("reference_file", "candidate_file"):
        _require(
            isinstance(config.get(key), str) and bool(config[key]), key + " is required"
        )
        path = Path(config[key]).expanduser()
        if not path.is_absolute():
            path = Path(base_path) / path
        data = load_json(path)
        definitions = validate_observable_input(data)
        _require(
            len({s["system_id"] for s in data["segments"]}) == 1,
            "each comparison arm must contain exactly one system",
        )
        inputs.append((data, definitions))
        hashes[key] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    (reference, reference_defs), (candidate, candidate_defs) = inputs
    _require(
        reference["weighting"] == candidate["weighting"],
        "comparison weighting must match",
    )
    comparisons = config.get("comparisons")
    _require(
        isinstance(comparisons, list) and bool(comparisons),
        "comparisons must be nonempty",
    )
    repeats, block_length = config.get("bootstrap_repeats", 200), config.get(
        "block_length_frames", 10
    )
    confidence, seed = config.get("confidence_level", 0.95), config.get(
        "random_seed", 0
    )
    _require(
        isinstance(repeats, int)
        and not isinstance(repeats, bool)
        and 20 <= repeats <= 10000,
        "bootstrap_repeats must be between 20 and 10000",
    )
    _require(
        isinstance(block_length, int)
        and not isinstance(block_length, bool)
        and block_length >= 2,
        "block_length_frames must be at least two",
    )
    _require(
        _number(confidence) and 0 < confidence < 1,
        "confidence_level must be between zero and one",
    )
    _require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
        "random_seed must be nonnegative",
    )
    # The comparison is bounded rather than silently subsampling its input.
    count = sum(
        len(s["rows"]) for data in (reference, candidate) for s in data["segments"]
    )
    _require(
        count * repeats * len(comparisons) <= 50_000_000,
        "bootstrap workload exceeds 50 million observation-statistic evaluations; reduce the declared repeats or input scope",
    )

    def time_contract(data):
        return sorted(
            {
                (
                    round(s["rows"][0]["time"] * TIME_TO_NS[s["time_unit"]], 12),
                    round(s["rows"][-1]["time"] * TIME_TO_NS[s["time_unit"]], 12),
                    round(s["frame_interval"] * TIME_TO_NS[s["time_unit"]], 12),
                )
                for s in data["segments"]
            }
        )

    eligible = _protocol_contract(reference, candidate)
    if time_contract(reference) != time_contract(candidate):
        eligible = {
            "status": "not assessed",
            "reason": "Saved-frame intervals or observation windows differ; match them before attributing differences to the integration step.",
        }
    bootstrap_adequate = all(
        len(s["rows"]) >= 2 * block_length
        for data in (reference, candidate)
        for s in data["segments"]
    )
    rows = []
    seen = set()
    for comparison in comparisons:
        _require(
            isinstance(comparison, dict)
            and set(comparison)
            <= {"observable_id", "statistic", "tolerance", "lag_frames"},
            "unknown comparison fields",
        )
        metric, statistic, tolerance = (
            comparison.get("observable_id"),
            comparison.get("statistic"),
            comparison.get("tolerance"),
        )
        _require(
            isinstance(metric, str)
            and metric in reference_defs
            and metric in candidate_defs,
            "comparison observable must exist in both inputs",
        )
        _require(
            reference_defs[metric] == candidate_defs[metric],
            "observable definitions, units, conditions and histogram edges must match",
        )
        definition = reference_defs[metric]
        condition = definition.get("condition_observable_id")
        if condition is not None:
            _require(
                reference_defs[condition] == candidate_defs[condition],
                "conditioning definitions must match",
            )
        _require(
            isinstance(statistic, str)
            and statistic
            in {
                "mean",
                "distribution",
                "conditioning_fraction",
                "lagged_correlation",
                "switch_probability",
            },
            "unsupported comparison statistic",
        )
        _require(
            _number(tolerance) and tolerance >= 0,
            "tolerance must be finite and nonnegative",
        )
        lag = comparison.get("lag_frames", 1)
        _require(
            isinstance(lag, int) and not isinstance(lag, bool) and lag > 0,
            "lag_frames must be a positive integer",
        )
        _require((metric, statistic, lag) not in seen, "duplicate comparison")
        seen.add((metric, statistic, lag))
        if statistic == "switch_probability":
            _require(
                definition["kind"] == "indicator",
                "switch_probability requires an indicator",
            )
        temporal = statistic in {"lagged_correlation", "switch_probability"}
        if temporal:
            _require(
                block_length > lag,
                "bootstrap blocks must be longer than the requested lag",
            )
        if not temporal:
            _require(
                "lag_frames" not in comparison,
                "lag_frames applies only to temporal statistics",
            )
        a = _statistic(reference, list(_blocks(reference)), definition, comparison)
        b = _statistic(candidate, list(_blocks(candidate)), definition, comparison)
        delta = _difference(a, b, statistic)
        interval = None
        status, reason = "not assessed", eligible["reason"]
        if (
            eligible["status"] == "eligible"
            and delta is not None
            and bootstrap_adequate
        ):
            rng = np.random.default_rng(seed)
            samples = []
            for _ in range(repeats):
                aa = _statistic(
                    reference,
                    list(_blocks(reference, rng, block_length)),
                    definition,
                    comparison,
                )
                bb = _statistic(
                    candidate,
                    list(_blocks(candidate, rng, block_length)),
                    definition,
                    comparison,
                )
                value = _difference(aa, bb, statistic)
                if value is not None:
                    samples.append(value)
            if len(samples) == repeats:
                alpha = (1 - confidence) / 2
                interval = np.quantile(samples, [alpha, 1 - alpha]).tolist()
                if (
                    -tolerance <= interval[0]
                    and interval[1] <= tolerance
                    and abs(delta) <= tolerance
                ):
                    status = "within declared tolerance"
                elif abs(delta) > tolerance and (
                    interval[0] > tolerance or interval[1] < -tolerance
                ):
                    status = "exceeds declared tolerance"
                else:
                    status = "inconclusive"
                reason = "Exploratory within-segment moving-block bootstrap at the declared observation window."
            else:
                reason = "Some bootstrap statistics were undefined; no interval or tolerance verdict is reported."
        elif delta is None:
            reason = "Statistic is undefined (for example, no retained samples, no lag pairs, or constant-series correlation)."
        elif not bootstrap_adequate:
            reason = (
                "Every segment must contain at least two complete bootstrap blocks."
            )
        rows.append(
            {
                **comparison,
                "difference": delta,
                "difference_unit": (
                    definition["unit"] if statistic == "mean" else "dimensionless"
                ),
                "interval": interval,
                "numerical_sensitivity_status": status,
                "reason": reason,
                "conditioning_rule": condition,
                "population_validity": "not assessed",
                "kinetic_validity": "not assessed",
            }
        )
    return {
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "input_files": hashes,
        "protocol_comparison": eligible,
        "comparisons": rows,
        "bootstrap": {
            "repeats": repeats,
            "block_length_frames": block_length,
            "confidence_level": confidence,
            "random_seed": seed,
            "scope": "within each observed segment; replicas are held fixed; resampled blocks are never joined for lag pairs",
        },
        "limitations": [
            "These are observable-specific finite-window sensitivity diagnostics, not a shadowing proof or physical validation.",
            "Distribution distance is total variation on the declared histogram including tails; it cannot resolve changes within a bin.",
            "Bootstrap intervals are exploratory and conditional on observed sampling and block length; unvisited states are not represented.",
            "Simultaneous coverage across multiple comparisons is not asserted.",
            "Changing analysis stride does not constitute integration-step validation.",
        ],
        "error_count": 0,
        "issues": [],
    }


def compare_numerical_protocols_file_safe(path):
    try:
        return compare_numerical_protocols(load_json(path), Path(path).parent)
    except (
        ConvergenceAnalysisError,
        ManifestValidationError,
        SimulationProtocolError,
        OSError,
    ) as exc:
        return {
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "error_count": 1,
            "issues": [
                {
                    "severity": "error",
                    "code": "NUMERICAL_COMPARISON_INVALID",
                    "message": str(exc),
                }
            ],
        }
