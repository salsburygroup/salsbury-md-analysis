"""Segment-safe reactive-path ensembles from ordinary MD trajectories.

The module consumes the complete KMeans assignment table so that state labels,
frame identities, time, and trajectory features are reused exactly.  It can
discover a recurrent endpoint pair without a biological annotation or accept
disjoint multi-state source and sink sets.  Extracted paths use a last-exit /
first-arrival convention and are never joined across trajectory segments.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .clustering import ClusteringAnalysisError, clustering_kmeans_project
from .manifests import ManifestValidationError, load_json
from .upstream_cache import load_cached_project_report
from .validation import positive_integer


class ReactivePathAnalysisError(ValueError):
    """Raised when a reactive-path contract cannot be evaluated safely."""


TrajectoryKey = Tuple[str, ...]
PathSpan = Tuple[int, int]


def _positive(value: object, name: str) -> int:
    return positive_integer(value, name, error_type=ReactivePathAnalysisError)


def _finite_fraction(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ReactivePathAnalysisError(
            f"{name} must be finite and between zero and one"
        )
    return float(value)


def _state_ids(value: object, name: str, *, allow_empty: bool) -> List[int]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
        or len(set(value)) != len(value)
    ):
        qualifier = "possibly empty" if allow_empty else "nonempty"
        raise ReactivePathAnalysisError(
            f"{name} must contain unique positive integer state IDs ({qualifier})"
        )
    return sorted(value)


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = (
        definitions.get("reactive_path_ensembles")
        if isinstance(definitions, dict) else None
    )
    if not isinstance(raw, dict):
        raise ReactivePathAnalysisError(
            "definitions.reactive_path_ensembles must be an object"
        )
    required = {
        "assignment_source", "endpoint_mode", "source_state_ids",
        "sink_state_ids", "feature_indices", "feature_scaling",
        "minimum_pair_events_for_automatic_selection",
        "sakoe_chiba_fraction", "maximum_paths_per_direction",
        "maximum_path_frames", "maximum_pairwise_dtw_cells",
        "maximum_path_clusters", "minimum_path_cluster_size",
        "minimum_complete_paths_for_comparison",
        "minimum_complete_paths_per_direction",
        "minimum_replicas_with_complete_paths",
        "minimum_complete_paths_for_kinetics",
        "minimum_complete_paths_per_direction_for_kinetics",
        "minimum_replicas_with_complete_paths_for_kinetics",
        "require_validated_msm_for_kinetics",
    }
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required))
    if missing or unknown:
        raise ReactivePathAnalysisError(
            "reactive-path settings mismatch; missing=" + ",".join(missing)
            + "; unknown=" + ",".join(unknown)
        )
    if raw["assignment_source"] != "clustering_kmeans":
        raise ReactivePathAnalysisError(
            "assignment_source must be clustering_kmeans in this implementation"
        )
    if raw["endpoint_mode"] not in {
        "automatic_recurrent_pair", "explicit_state_sets"
    }:
        raise ReactivePathAnalysisError(
            "endpoint_mode must be automatic_recurrent_pair or explicit_state_sets"
        )
    source_ids = _state_ids(
        raw["source_state_ids"], "source_state_ids",
        allow_empty=raw["endpoint_mode"] == "automatic_recurrent_pair",
    )
    sink_ids = _state_ids(
        raw["sink_state_ids"], "sink_state_ids",
        allow_empty=raw["endpoint_mode"] == "automatic_recurrent_pair",
    )
    if raw["endpoint_mode"] == "automatic_recurrent_pair" and (
        source_ids or sink_ids
    ):
        raise ReactivePathAnalysisError(
            "automatic_recurrent_pair requires empty source_state_ids and sink_state_ids"
        )
    overlap = sorted(set(source_ids).intersection(sink_ids))
    if overlap:
        raise ReactivePathAnalysisError(
            "source and sink state sets must be disjoint; overlap="
            + ",".join(str(value) for value in overlap)
        )
    feature_indices = raw["feature_indices"]
    if (
        not isinstance(feature_indices, list)
        or not feature_indices
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in feature_indices
        )
        or len(set(feature_indices)) != len(feature_indices)
    ):
        raise ReactivePathAnalysisError(
            "feature_indices must contain unique positive one-based indices"
        )
    if raw["feature_scaling"] not in {"none", "zscore"}:
        raise ReactivePathAnalysisError("feature_scaling must be none or zscore")
    if not isinstance(raw["require_validated_msm_for_kinetics"], bool):
        raise ReactivePathAnalysisError(
            "require_validated_msm_for_kinetics must be boolean"
        )
    settings = dict(raw)
    settings.update({
        "source_state_ids": source_ids,
        "sink_state_ids": sink_ids,
        "feature_indices": list(feature_indices),
        "minimum_pair_events_for_automatic_selection": _positive(
            raw["minimum_pair_events_for_automatic_selection"],
            "minimum_pair_events_for_automatic_selection",
        ),
        "sakoe_chiba_fraction": _finite_fraction(
            raw["sakoe_chiba_fraction"], "sakoe_chiba_fraction"
        ),
    })
    for name in (
        "maximum_paths_per_direction", "maximum_path_frames",
        "maximum_pairwise_dtw_cells", "maximum_path_clusters",
        "minimum_path_cluster_size",
        "minimum_complete_paths_for_comparison",
        "minimum_complete_paths_per_direction",
        "minimum_replicas_with_complete_paths",
        "minimum_complete_paths_for_kinetics",
        "minimum_complete_paths_per_direction_for_kinetics",
        "minimum_replicas_with_complete_paths_for_kinetics",
    ):
        settings[name] = _positive(raw[name], name)
    if (
        int(settings["minimum_complete_paths_for_kinetics"])
        < int(settings["minimum_complete_paths_for_comparison"])
        or int(settings["minimum_complete_paths_per_direction_for_kinetics"])
        < int(settings["minimum_complete_paths_per_direction"])
        or int(settings["minimum_replicas_with_complete_paths_for_kinetics"])
        < int(settings["minimum_replicas_with_complete_paths"])
    ):
        raise ReactivePathAnalysisError(
            "kinetics sufficiency thresholds cannot be weaker than pathway-comparison thresholds"
        )
    return settings


def _trajectory_key(row: Mapping[str, object]) -> TrajectoryKey:
    try:
        values = (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"]),
        )
    except KeyError as exc:
        raise ReactivePathAnalysisError(
            "assignments lack system, replica, or segment identity"
        ) from exc
    return values + ((str(row["member_id"]),) if "member_id" in row else ())


def _group_assignments(
    rows: object, feature_indices: Sequence[int]
) -> Tuple[Dict[TrajectoryKey, List[Dict[str, object]]], List[int], int]:
    if not isinstance(rows, list) or not rows:
        raise ReactivePathAnalysisError(
            "clustering_kmeans returned no complete assignment rows"
        )
    groups: Dict[TrajectoryKey, List[Dict[str, object]]] = defaultdict(list)
    states = set()
    feature_width: Optional[int] = None
    for raw in rows:
        if not isinstance(raw, dict):
            raise ReactivePathAnalysisError("every clustering assignment must be an object")
        try:
            frame = int(raw["source_frame_index"])
            state = int(raw["cluster_id"])
            time = float(raw["time"])
            time_unit = str(raw["time_unit"])
            values = raw["feature_values"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ReactivePathAnalysisError(
                "assignments require frame, state, physical time, and feature values"
            ) from exc
        if state <= 0 or frame < 0 or not math.isfinite(time) or not time_unit:
            raise ReactivePathAnalysisError(
                "assignment states, frame indices, and physical times are invalid"
            )
        if (
            not isinstance(values, list) or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in values
            )
        ):
            raise ReactivePathAnalysisError(
                "assignment feature_values must be a nonempty finite numeric array"
            )
        if feature_width is None:
            feature_width = len(values)
        elif len(values) != feature_width:
            raise ReactivePathAnalysisError(
                "assignment feature_values do not have a consistent width"
            )
        groups[_trajectory_key(raw)].append({
            **raw,
            "source_frame_index": frame,
            "cluster_id": state,
            "time": time,
            "time_unit": time_unit,
            "feature_values": [float(value) for value in values],
        })
        states.add(state)
    assert feature_width is not None
    if max(feature_indices) > feature_width:
        raise ReactivePathAnalysisError(
            f"feature_indices request component {max(feature_indices)} but assignments contain only {feature_width} values"
        )
    ordered_groups = {}
    global_units = set()
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: int(row["source_frame_index"]))
        frame_ids = [int(row["source_frame_index"]) for row in ordered]
        if len(frame_ids) != len(set(frame_ids)):
            raise ReactivePathAnalysisError(
                "duplicate source frame within trajectory segment " + "/".join(key)
            )
        units = {str(row["time_unit"]) for row in ordered}
        if len(units) != 1:
            raise ReactivePathAnalysisError(
                "a trajectory segment contains mixed physical time units"
            )
        global_units.update(units)
        times = [float(row["time"]) for row in ordered]
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ReactivePathAnalysisError(
                "assignment times must increase strictly within every segment"
            )
        ordered_groups[key] = ordered
    if len(global_units) != 1:
        raise ReactivePathAnalysisError(
            "all reactive-path assignments must use one physical time unit"
        )
    return ordered_groups, sorted(states), feature_width


def extract_reactive_path_spans(
    states: Sequence[int], source_states: Sequence[int], sink_states: Sequence[int]
) -> List[PathSpan]:
    """Return last-source-exit to first-sink-arrival spans in one segment."""

    sources = set(source_states)
    sinks = set(sink_states)
    if not sources or not sinks or sources.intersection(sinks):
        raise ReactivePathAnalysisError(
            "reactive path endpoints must be nonempty disjoint state sets"
        )
    spans: List[PathSpan] = []
    last_source: Optional[int] = None
    for index, state in enumerate(states):
        if state in sources:
            last_source = index
        elif state in sinks and last_source is not None:
            spans.append((last_source, index))
            last_source = None
    return spans


def _all_spans(
    groups: Mapping[TrajectoryKey, Sequence[Mapping[str, object]]],
    source_states: Sequence[int], sink_states: Sequence[int],
) -> List[Tuple[TrajectoryKey, int, int]]:
    result = []
    for key in sorted(groups):
        rows = groups[key]
        states = [int(row["cluster_id"]) for row in rows]
        result.extend(
            (key, start, stop)
            for start, stop in extract_reactive_path_spans(
                states, source_states, sink_states
            )
        )
    return result


def _state_centroids(
    groups: Mapping[TrajectoryKey, Sequence[Mapping[str, object]]],
    feature_indices: Sequence[int],
) -> Dict[int, List[float]]:
    selected = [index - 1 for index in feature_indices]
    values: Dict[int, List[List[float]]] = defaultdict(list)
    for rows in groups.values():
        for row in rows:
            vector = row["feature_values"]
            assert isinstance(vector, list)
            values[int(row["cluster_id"])].append(
                [float(vector[index]) for index in selected]
            )
    return {
        state: np.mean(np.asarray(vectors, dtype=float), axis=0).tolist()
        for state, vectors in values.items()
    }


def _automatic_endpoints(
    groups: Mapping[TrajectoryKey, Sequence[Mapping[str, object]]],
    states: Sequence[int], feature_indices: Sequence[int], minimum_events: int,
) -> Tuple[List[int], List[int], str, List[Dict[str, object]]]:
    if len(states) < 2:
        raise ReactivePathAnalysisError(
            "at least two assigned states are required for reactive paths"
        )
    centroids = _state_centroids(groups, feature_indices)
    system_ids = sorted({key[0] for key in groups})
    candidates = []
    for offset, left in enumerate(states):
        for right in states[offset + 1:]:
            system_counts = []
            for system_id in system_ids:
                system_groups = {
                    key: rows for key, rows in groups.items()
                    if key[0] == system_id
                }
                system_counts.append({
                    "system_id": system_id,
                    "forward_complete_path_count": len(
                        _all_spans(system_groups, [left], [right])
                    ),
                    "reverse_complete_path_count": len(
                        _all_spans(system_groups, [right], [left])
                    ),
                })
            forward = sum(
                int(row["forward_complete_path_count"])
                for row in system_counts
            )
            reverse = sum(
                int(row["reverse_complete_path_count"])
                for row in system_counts
            )
            minimum_system_direction = min(
                min(
                    int(row["forward_complete_path_count"]),
                    int(row["reverse_complete_path_count"]),
                )
                for row in system_counts
            )
            distance = math.sqrt(sum(
                (a - b) ** 2 for a, b in zip(centroids[left], centroids[right])
            ))
            candidates.append({
                "source_state_ids": [left],
                "sink_state_ids": [right],
                "forward_complete_path_count": forward,
                "reverse_complete_path_count": reverse,
                "minimum_direction_count": min(forward, reverse),
                "minimum_system_direction_count": minimum_system_direction,
                "total_complete_path_count": forward + reverse,
                "system_counts": system_counts,
                "centroid_distance_in_selected_input_features": distance,
                "passes_recurrence_gate": (
                    minimum_system_direction >= minimum_events
                ),
            })
    selected = max(candidates, key=lambda row: (
        bool(row["passes_recurrence_gate"]),
        int(row["minimum_system_direction_count"]),
        int(row["minimum_direction_count"]),
        int(row["total_complete_path_count"]),
        float(row["centroid_distance_in_selected_input_features"]),
        -int(row["source_state_ids"][0]),
        -int(row["sink_state_ids"][0]),
    ))
    status = (
        "selected_recurrent_pair"
        if selected["passes_recurrence_gate"]
        else "selected_below_recurrence_gate"
    )
    return (
        list(selected["source_state_ids"]),
        list(selected["sink_state_ids"]),
        status,
        sorted(candidates, key=lambda row: (
            -int(row["minimum_system_direction_count"]),
            -int(row["minimum_direction_count"]),
            -int(row["total_complete_path_count"]),
            -float(row["centroid_distance_in_selected_input_features"]),
            int(row["source_state_ids"][0]),
            int(row["sink_state_ids"][0]),
        )),
    )


def _scaling(
    groups: Mapping[TrajectoryKey, Sequence[Mapping[str, object]]],
    feature_indices: Sequence[int], mode: str,
) -> Tuple[List[float], List[float]]:
    selected = [index - 1 for index in feature_indices]
    matrix = np.asarray([
        [float(row["feature_values"][index]) for index in selected]
        for rows in groups.values() for row in rows
    ], dtype=float)
    means = np.mean(matrix, axis=0) if mode == "zscore" else np.zeros(matrix.shape[1])
    scales = np.std(matrix, axis=0) if mode == "zscore" else np.ones(matrix.shape[1])
    scales = np.where(scales > 0.0, scales, 1.0)
    return means.tolist(), scales.tolist()


def _path_records(
    groups: Mapping[TrajectoryKey, Sequence[Mapping[str, object]]],
    source_states: Sequence[int], sink_states: Sequence[int], direction: str,
    feature_indices: Sequence[int], means: Sequence[float], scales: Sequence[float],
) -> List[Dict[str, object]]:
    records = []
    selected = [index - 1 for index in feature_indices]
    for key, start, stop in _all_spans(groups, source_states, sink_states):
        rows = groups[key]
        path_rows = rows[start:stop + 1]
        first, last = path_rows[0], path_rows[-1]
        points = []
        for row in path_rows:
            raw = [float(row["feature_values"][index]) for index in selected]
            points.append({
                "source_frame_index": int(row["source_frame_index"]),
                "time": float(row["time"]),
                "time_unit": str(row["time_unit"]),
                "state_id": int(row["cluster_id"]),
                "feature_values": raw,
                "scaled_feature_values": [
                    (value - means[index]) / scales[index]
                    for index, value in enumerate(raw)
                ],
            })
        identity = {
            "system_id": key[0], "replica_id": key[1], "segment_id": key[2],
            **({"member_id": key[3]} if len(key) == 4 else {}),
        }
        records.append({
            "path_id": "",
            "direction": direction,
            **identity,
            "source_state_id": int(first["cluster_id"]),
            "sink_state_id": int(last["cluster_id"]),
            "start_source_frame_index": int(first["source_frame_index"]),
            "end_source_frame_index": int(last["source_frame_index"]),
            "start_time": float(first["time"]),
            "end_time": float(last["time"]),
            "duration": float(last["time"]) - float(first["time"]),
            "time_unit": str(first["time_unit"]),
            "observed_frame_count": len(points),
            "points": points,
        })
    for index, record in enumerate(records, start=1):
        record["path_id"] = f"{direction}-{index:05d}"
    return records


def multidimensional_dtw_distance(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]],
    sakoe_chiba_fraction: float,
) -> float:
    """Return path-length-normalized multidimensional DTW distance."""

    if not left or not right:
        raise ReactivePathAnalysisError("DTW paths must be nonempty")
    width = len(left[0])
    if width == 0 or any(len(row) != width for row in [*left, *right]):
        raise ReactivePathAnalysisError("DTW paths must share one nonzero width")
    n, m = len(left), len(right)
    window = max(abs(n - m), int(math.ceil(sakoe_chiba_fraction * max(n, m))))
    previous: Dict[int, Tuple[float, int]] = {0: (0.0, 0)}
    for i in range(1, n + 1):
        current: Dict[int, Tuple[float, int]] = {}
        for j in range(max(1, i - window), min(m, i + window) + 1):
            choices = [
                value for value in (
                    previous.get(j), current.get(j - 1), previous.get(j - 1)
                ) if value is not None
            ]
            if not choices:
                continue
            prior_cost, prior_length = min(choices, key=lambda value: (value[0], value[1]))
            local = math.sqrt(sum(
                (float(left[i - 1][axis]) - float(right[j - 1][axis])) ** 2
                for axis in range(width)
            ))
            current[j] = (prior_cost + local, prior_length + 1)
        previous = current
    if m not in previous:
        raise ReactivePathAnalysisError("Sakoe-Chiba window admitted no complete DTW path")
    cost, length = previous[m]
    return cost / length


def _average_linkage_partitions(
    distances: Sequence[Sequence[float]], maximum_clusters: int,
) -> Dict[int, List[int]]:
    n = len(distances)
    clusters: List[Tuple[int, ...]] = [(index,) for index in range(n)]
    partitions: Dict[int, List[int]] = {}
    while len(clusters) > 1:
        if len(clusters) <= maximum_clusters:
            labels = [0] * n
            for label, cluster in enumerate(sorted(clusters), start=1):
                for index in cluster:
                    labels[index] = label
            partitions[len(clusters)] = labels
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                values = [distances[a][b] for a in clusters[i] for b in clusters[j]]
                key = (sum(values) / len(values), clusters[i], clusters[j], i, j)
                if best is None or key < best:
                    best = key
        assert best is not None
        i, j = int(best[-2]), int(best[-1])
        merged = tuple(sorted((*clusters[i], *clusters[j])))
        clusters = [
            cluster for index, cluster in enumerate(clusters)
            if index not in {i, j}
        ] + [merged]
        clusters.sort()
    return partitions


def _silhouette(distances: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    members: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        members[label].append(index)
    values = []
    for index, label in enumerate(labels):
        own = [other for other in members[label] if other != index]
        if not own:
            values.append(0.0)
            continue
        a = sum(distances[index][other] for other in own) / len(own)
        b = min(
            sum(distances[index][other] for other in group) / len(group)
            for other_label, group in members.items() if other_label != label
        )
        values.append((b - a) / max(a, b) if max(a, b) > 0.0 else 0.0)
    return sum(values) / len(values)


def _route_analysis(
    paths: Sequence[Mapping[str, object]], settings: Mapping[str, object]
) -> Dict[str, object]:
    eligible = [
        path for path in paths
        if int(path["observed_frame_count"]) <= int(settings["maximum_path_frames"])
    ]
    retained = eligible[:int(settings["maximum_paths_per_direction"])]
    lengths = [int(path["observed_frame_count"]) for path in retained]
    cells = sum(
        lengths[i] * lengths[j]
        for i in range(len(lengths)) for j in range(i + 1, len(lengths))
    )
    base = {
        "complete_path_count_observed": len(paths),
        "path_count_excluded_over_maximum_path_frames": len(paths) - len(eligible),
        "path_count_excluded_over_maximum_paths_per_direction": (
            len(eligible) - len(retained)
        ),
        "retained_path_count": len(retained),
        "estimated_pairwise_dtw_cells": cells,
        "maximum_pairwise_dtw_cells": int(settings["maximum_pairwise_dtw_cells"]),
    }
    if not retained:
        return {**base, "route_clustering_status": "no_eligible_paths",
                "cluster_selection_status": "not_run_no_eligible_paths",
                "selected_cluster_count": 0, "paths": [],
                "pairwise_dtw_distances": [], "cluster_selection": [],
                "route_clusters": []}
    if cells > int(settings["maximum_pairwise_dtw_cells"]):
        return {**base, "route_clustering_status": "resource_gate_blocked",
                "cluster_selection_status": "not_run_resource_gate_blocked",
                "selected_cluster_count": 0,
                "paths": retained, "pairwise_dtw_distances": [],
                "cluster_selection": [], "route_clusters": []}
    n = len(retained)
    distances = [[0.0] * n for _ in range(n)]
    rows = []
    for i in range(n):
        left = [point["scaled_feature_values"] for point in retained[i]["points"]]
        for j in range(i + 1, n):
            right = [point["scaled_feature_values"] for point in retained[j]["points"]]
            value = multidimensional_dtw_distance(
                left, right, float(settings["sakoe_chiba_fraction"])
            )
            distances[i][j] = distances[j][i] = value
            rows.append({
                "path_id_i": retained[i]["path_id"],
                "path_id_j": retained[j]["path_id"],
                "normalized_dtw_distance": value,
            })
    if n == 1:
        labels = [1]
        candidates = []
        selection_status = "single_path_single_route"
    else:
        partitions = _average_linkage_partitions(
            distances, min(int(settings["maximum_path_clusters"]), n)
        )
        candidates = []
        for k, candidate_labels in sorted(partitions.items()):
            sizes = sorted(Counter(candidate_labels).values())
            eligible_partition = min(sizes) >= int(settings["minimum_path_cluster_size"])
            candidates.append({
                "cluster_count": k,
                "cluster_sizes": sizes,
                "silhouette": _silhouette(distances, candidate_labels),
                "eligible": eligible_partition,
            })
        selectable = [row for row in candidates if row["eligible"]]
        if selectable:
            selected = max(selectable, key=lambda row: (
                float(row["silhouette"]), -int(row["cluster_count"])
            ))
            labels = partitions[int(selected["cluster_count"])]
            selection_status = "selected_by_silhouette"
        else:
            labels = [1] * n
            selection_status = "single_route_fallback_no_partition_passed_minimum_size"
    route_clusters = []
    for label in sorted(set(labels)):
        members = [index for index, value in enumerate(labels) if value == label]
        medoid = min(members, key=lambda index: (
            sum(distances[index][other] for other in members) / len(members),
            str(retained[index]["path_id"]),
        ))
        route_clusters.append({
            "route_cluster_id": label,
            "path_count": len(members),
            "path_ids": [retained[index]["path_id"] for index in members],
            "system_path_counts": [
                {"system_id": system_id, "path_count": count}
                for system_id, count in sorted(Counter(
                    str(retained[index]["system_id"]) for index in members
                ).items())
            ],
            "representative_medoid_path_id": retained[medoid]["path_id"],
            "mean_duration": sum(float(retained[index]["duration"]) for index in members) / len(members),
            "time_unit": retained[members[0]]["time_unit"],
        })
    assigned_paths = [dict(path, route_cluster_id=labels[index])
                      for index, path in enumerate(retained)]
    return {
        **base,
        "route_clustering_status": "complete",
        "cluster_selection_status": selection_status,
        "selected_cluster_count": len(route_clusters),
        "paths": assigned_paths,
        "pairwise_dtw_distances": rows,
        "cluster_selection": candidates,
        "route_clusters": route_clusters,
    }


def _validated_kmeans_msm(msm: Optional[Mapping[str, object]]) -> Dict[str, object]:
    if msm is None:
        return {
            "status": "not_available", "passes_kinetics_gate": False,
            "reason": "no validated cached Markov-state report was available",
        }
    comparisons = msm.get("clustering_state_model_comparison")
    if isinstance(comparisons, list):
        matches = [row for row in comparisons if isinstance(row, dict)
                   and row.get("candidate_id") == "kmeans"]
        if len(matches) != 1:
            return {
                "status": "not_available", "passes_kinetics_gate": False,
                "reason": "Markov-state report has no unique KMeans candidate",
            }
        candidate = matches[0]
        passed = candidate.get("kinetic_validation_status") == "passed"
        return {
            "status": candidate.get("kinetic_validation_status"),
            "passes_kinetics_gate": passed,
            "candidate_id": "kmeans",
            "state_count": candidate.get("state_count"),
            "validation_gates": candidate.get("validation_gates"),
            "reason": None if passed else "KMeans state model did not pass its kinetic gates",
        }
    passed = msm.get("kinetic_validation_status") == "passed"
    return {
        "status": msm.get("kinetic_validation_status", "not_available"),
        "passes_kinetics_gate": passed,
        "candidate_id": "legacy_assignment_source",
        "reason": None if passed else "Markov-state report did not pass kinetic validation",
    }


def reactive_path_ensembles_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    clustering = clustering_kmeans_project(source, hash_content=hash_content)
    groups, state_ids, feature_width = _group_assignments(
        clustering.get("assignments"), settings["feature_indices"]
    )
    declared_source = list(settings["source_state_ids"])
    declared_sink = list(settings["sink_state_ids"])
    candidates: List[Dict[str, object]] = []
    if settings["endpoint_mode"] == "automatic_recurrent_pair":
        source_states, sink_states, selection_status, candidates = _automatic_endpoints(
            groups, state_ids, settings["feature_indices"],
            int(settings["minimum_pair_events_for_automatic_selection"]),
        )
    else:
        source_states, sink_states = declared_source, declared_sink
        unknown = sorted(set(source_states + sink_states).difference(state_ids))
        if unknown:
            raise ReactivePathAnalysisError(
                "explicit endpoint state IDs are absent from assignments: "
                + ",".join(str(value) for value in unknown)
            )
        selection_status = "explicit_state_sets"
    means, scales = _scaling(
        groups, settings["feature_indices"], str(settings["feature_scaling"])
    )
    forward = _path_records(
        groups, source_states, sink_states, "source_to_sink",
        settings["feature_indices"], means, scales,
    )
    reverse = _path_records(
        groups, sink_states, source_states, "sink_to_source",
        settings["feature_indices"], means, scales,
    )
    route_analyses = {
        "source_to_sink": _route_analysis(forward, settings),
        "sink_to_source": _route_analysis(reverse, settings),
    }
    endpoint_counts = Counter(
        (
            str(path["system_id"]), int(path["source_state_id"]),
            int(path["sink_state_id"]), str(path["direction"]),
        )
        for path in [*forward, *reverse]
    )
    endpoint_matrix = [{
        "system_id": key[0], "source_state_id": key[1],
        "sink_state_id": key[2], "direction": key[3],
        "complete_path_count": count,
    } for key, count in sorted(endpoint_counts.items())]
    physical_replicas = {
        (str(path["system_id"]), str(path["replica_id"]))
        for path in [*forward, *reverse]
    }
    physical_frame_count = len({
        (key[0], key[1], key[2], int(row["source_frame_index"]))
        for key, rows in groups.items() for row in rows
    })
    assignment_observation_count = sum(len(rows) for rows in groups.values())
    system_ids = sorted({key[0] for key in groups})
    system_transition_summaries = []
    for system_id in system_ids:
        system_paths = [
            path for path in [*forward, *reverse]
            if str(path["system_id"]) == system_id
        ]
        forward_count = sum(
            path["direction"] == "source_to_sink" for path in system_paths
        )
        reverse_count = sum(
            path["direction"] == "sink_to_source" for path in system_paths
        )
        replicas = {
            str(path["replica_id"]) for path in system_paths
        }
        system_transition_summaries.append({
            "system_id": system_id,
            "source_to_sink_complete_path_count": forward_count,
            "sink_to_source_complete_path_count": reverse_count,
            "total_complete_path_count": forward_count + reverse_count,
            "physical_replica_count_with_complete_paths": len(replicas),
            "physical_replica_ids_with_complete_paths": sorted(replicas),
            "bidirectional_complete_paths_observed": (
                forward_count > 0 and reverse_count > 0
            ),
        })
    total = len(forward) + len(reverse)
    comparison_gates = {
        "complete_paths_per_system": {
            "observed": min(
                int(row["total_complete_path_count"])
                for row in system_transition_summaries
            ),
            "required": int(settings["minimum_complete_paths_for_comparison"]),
            "passed": all(
                int(row["total_complete_path_count"])
                >= int(settings["minimum_complete_paths_for_comparison"])
                for row in system_transition_summaries
            ),
        },
        "complete_paths_each_direction_per_system": {
            "observed": min(
                min(
                    int(row["source_to_sink_complete_path_count"]),
                    int(row["sink_to_source_complete_path_count"]),
                )
                for row in system_transition_summaries
            ),
            "required": int(settings["minimum_complete_paths_per_direction"]),
            "passed": all(
                min(
                    int(row["source_to_sink_complete_path_count"]),
                    int(row["sink_to_source_complete_path_count"]),
                ) >= int(settings["minimum_complete_paths_per_direction"])
                for row in system_transition_summaries
            ),
        },
        "physical_replicas_with_paths_per_system": {
            "observed": min(
                int(row["physical_replica_count_with_complete_paths"])
                for row in system_transition_summaries
            ),
            "required": int(settings["minimum_replicas_with_complete_paths"]),
            "passed": all(
                int(row["physical_replica_count_with_complete_paths"])
                >= int(settings["minimum_replicas_with_complete_paths"])
                for row in system_transition_summaries
            ),
        },
        "systems_with_bidirectional_complete_paths": {
            "observed": sum(
                bool(row["bidirectional_complete_paths_observed"])
                for row in system_transition_summaries
            ),
            "required": len(system_transition_summaries),
            "passed": all(
                bool(row["bidirectional_complete_paths_observed"])
                for row in system_transition_summaries
            ),
        },
        "retained_paths_each_direction": {
            "observed": min(
                int(route_analyses["source_to_sink"]["retained_path_count"]),
                int(route_analyses["sink_to_source"]["retained_path_count"]),
            ),
            "required": int(settings["minimum_complete_paths_per_direction"]),
            "passed": min(
                int(route_analyses["source_to_sink"]["retained_path_count"]),
                int(route_analyses["sink_to_source"]["retained_path_count"]),
            ) >= int(settings["minimum_complete_paths_per_direction"]),
        },
        "dtw_route_analysis_completed": {
            "observed": {
                direction: analysis["route_clustering_status"]
                for direction, analysis in route_analyses.items()
            },
            "required": "complete in both directions",
            "passed": all(
                analysis["route_clustering_status"] == "complete"
                for analysis in route_analyses.values()
            ),
        },
    }
    msm = load_cached_project_report(
        "markov_state_models", source, hash_content=hash_content,
        error_type=ReactivePathAnalysisError,
    )
    msm_gate = _validated_kmeans_msm(msm)
    kinetics_gates = {
        "complete_paths_per_system": {
            "observed": min(
                int(row["total_complete_path_count"])
                for row in system_transition_summaries
            ),
            "required": int(settings["minimum_complete_paths_for_kinetics"]),
            "passed": all(
                int(row["total_complete_path_count"])
                >= int(settings["minimum_complete_paths_for_kinetics"])
                for row in system_transition_summaries
            ),
        },
        "complete_paths_each_direction_per_system": {
            "observed": min(
                min(
                    int(row["source_to_sink_complete_path_count"]),
                    int(row["sink_to_source_complete_path_count"]),
                )
                for row in system_transition_summaries
            ),
            "required": int(settings["minimum_complete_paths_per_direction_for_kinetics"]),
            "passed": all(
                min(
                    int(row["source_to_sink_complete_path_count"]),
                    int(row["sink_to_source_complete_path_count"]),
                ) >= int(settings["minimum_complete_paths_per_direction_for_kinetics"])
                for row in system_transition_summaries
            ),
        },
        "physical_replicas_with_paths_per_system": {
            "observed": min(
                int(row["physical_replica_count_with_complete_paths"])
                for row in system_transition_summaries
            ),
            "required": int(settings["minimum_replicas_with_complete_paths_for_kinetics"]),
            "passed": all(
                int(row["physical_replica_count_with_complete_paths"])
                >= int(settings["minimum_replicas_with_complete_paths_for_kinetics"])
                for row in system_transition_summaries
            ),
        },
        "validated_kmeans_markov_model": {
            "required": bool(settings["require_validated_msm_for_kinetics"]),
            "observed": msm_gate["status"],
            "passed": (
                bool(msm_gate["passes_kinetics_gate"])
                or not bool(settings["require_validated_msm_for_kinetics"])
            ),
        },
    }
    comparison_ready = all(bool(row["passed"]) for row in comparison_gates.values())
    kinetics_ready = comparison_ready and all(
        bool(row["passed"]) for row in kinetics_gates.values()
    )
    if total == 0:
        status = "not_observed"
    elif kinetics_ready:
        status = "kinetics_estimation_ready"
    elif comparison_ready:
        status = "pathway_comparison_ready"
    else:
        status = "observed_but_insufficient"
    issues = [
        issue for issue in clustering.get("issues", []) if isinstance(issue, dict)
    ]
    if selection_status == "selected_below_recurrence_gate":
        issues.append({
            "severity": "warning", "code": "NO_RECURRENT_ENDPOINT_PAIR",
            "message": (
                "no automatically considered state pair passed the declared bidirectional recurrence gate"
            ),
        })
    if status in {"not_observed", "observed_but_insufficient"}:
        failed = [name for name, row in comparison_gates.items() if not row["passed"]]
        issues.append({
            "severity": "warning", "code": "INSUFFICIENT_REACTIVE_TRANSITIONS",
            "message": (
                "no complete source/sink transitions were observed"
                if status == "not_observed" else
                "reactive paths were observed but pathway-comparison gates failed: "
                + ", ".join(failed)
            ),
        })
    for direction, analysis in route_analyses.items():
        if analysis["route_clustering_status"] == "resource_gate_blocked":
            issues.append({
                "severity": "warning", "code": "DTW_RESOURCE_GATE_BLOCKED",
                "location": direction,
                "message": "pairwise DTW work exceeded maximum_pairwise_dtw_cells",
            })
    return {
        "module_id": "reactive_path_ensembles",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "transition_sufficiency_status": status,
        "project_manifest_path": str(source),
        "project_manifest_sha256": clustering["project_manifest_sha256"],
        "system_manifest_path": clustering["system_manifest_path"],
        "system_manifest_sha256": clustering["system_manifest_sha256"],
        "input_content_signature_sha256": clustering["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "upstream_inputs": {
            "assignment_module": "clustering_kmeans",
            "assignment_observation_count": assignment_observation_count,
            "trajectory_segment_count": len(groups),
            "assigned_state_ids": state_ids,
            "selected_feature_width": len(settings["feature_indices"]),
            "available_feature_width": feature_width,
            "markov_state_report_status": msm_gate,
        },
        "observation_accounting": {
            "source_physical_frame_count": physical_frame_count,
            "symmetry_expanded_observation_count": assignment_observation_count,
            "member_observations_are_independent_replicas": False
            if any(len(key) == 4 for key in groups) else None,
            "reactive_path_count": total,
        },
        "endpoint_selection": {
            "mode": settings["endpoint_mode"],
            "status": selection_status,
            "source_state_ids": source_states,
            "sink_state_ids": sink_states,
            "automatic_candidate_inventory": candidates,
            "automatic_selection_rule": (
                "pass bidirectional recurrence gate, then maximize the smaller direction count, total path count, and centroid distance; deterministic state-ID tie break"
            ) if settings["endpoint_mode"] == "automatic_recurrent_pair" else None,
            "biological_annotation_inferred": False,
        },
        "feature_transform": {
            "feature_indices_one_based": list(settings["feature_indices"]),
            "scaling": settings["feature_scaling"],
            "global_means": means,
            "global_scales": scales,
        },
        "path_extraction_contract": (
            "within each declared system/replica/segment/member trajectory, begin at the last source-set observation before leaving the source set and end at the first subsequent sink-set observation; never join segments"
        ),
        "complete_path_count": total,
        "complete_path_count_by_direction": {
            "source_to_sink": len(forward), "sink_to_source": len(reverse),
        },
        "physical_replicas_with_complete_paths": [
            {"system_id": key[0], "replica_id": key[1]}
            for key in sorted(physical_replicas)
        ],
        "system_transition_summaries": system_transition_summaries,
        "endpoint_transition_counts": endpoint_matrix,
        "route_analyses": route_analyses,
        "comparison_sufficiency_gates": comparison_gates,
        "kinetics_readiness_gates": kinetics_gates,
        "kinetics_readiness_contract": (
            "ready means only that declared event-count, physical-replica, and KMeans MSM validation gates passed; this module does not estimate a rate"
        ),
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "Automatic endpoints are recurrent, geometrically separated KMeans labels without inferred biological meaning; they are not automatically metastable macrostates.",
            "Explicit endpoint sets may contain multiple source and multiple sink states, but must be disjoint and justified outside this module.",
            "KMeans state definitions and selected feature coordinates are inherited exactly; upstream feature and clustering sensitivity remains necessary.",
            "Paths contain selected clustering observations. If the upstream view was subsampled, unsaved intermediate observations are not reconstructed.",
            "DTW route clusters are descriptive trajectory-shape families and do not by themselves establish mechanism, committor probability, flux, or causality.",
            "Equivalent oligomer members are separate path sequences but never count as independent physical replicas.",
            "The kinetics-ready label is a prerequisite gate, not a rate estimate; transition-path theory, uncertainty, and state-definition validation remain separate work.",
            "Ordinary MD without complete source-to-sink events cannot reveal an unobserved transition route; the module reports that insufficiency explicitly.",
        ],
    }


def reactive_path_ensembles_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return reactive_path_ensembles_project(
            project_path, hash_content=hash_content
        )
    except (
        ClusteringAnalysisError, ManifestValidationError,
        ReactivePathAnalysisError, OSError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "reactive_path_ensembles",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "transition_sufficiency_status": "not_evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [{
                "severity": "error", "code": "REACTIVE_PATH_ENSEMBLES_INVALID",
                "message": message,
            } for message in messages],
        }
