"""Shared, identity-preserving feature matrices for state-learning modules."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple, Type

from .pca import common_pca_project
from .tica import (
    TICAAnalysisError,
    time_lagged_independent_component_analysis_project,
)
from .trajectory_features import trajectory_features_project
from .replica_execution import ReplicaShard, execute_replica_workers


Vector = Tuple[float, ...]


def parse_feature_selection(
    raw: Mapping[str, object], error_type: Type[ValueError]
) -> Dict[str, object]:
    """Validate either common-PCA columns or declared trajectory-feature columns."""

    source = raw.get("feature_source")
    if source in {"common_pca", "tica"}:
        if "trajectory_feature_columns" in raw:
            raise error_type(
                f"{source} cannot declare trajectory_feature_columns"
            )
        components = raw.get("component_indices")
        if (
            not isinstance(components, list)
            or len(components) < 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in components
            )
            or len(set(components)) != len(components)
        ):
            raise error_type(
                "component_indices must contain at least two unique positive integers"
            )
        return {
            "feature_source": source,
            "component_indices": list(components),
        }
    if source != "trajectory_features":
        raise error_type(
            "feature_source must be common_pca, tica, or trajectory_features"
        )
    if "component_indices" in raw:
        raise error_type(
            "trajectory_features cannot declare component_indices"
        )
    columns = raw.get("trajectory_feature_columns")
    if not isinstance(columns, list) or not columns:
        raise error_type(
            "trajectory_feature_columns must be a nonempty array"
        )
    normalized = []
    identities = set()
    for column in columns:
        if not isinstance(column, dict) or set(column) != {"feature_id", "value_indices"}:
            raise error_type(
                "each trajectory feature selection requires feature_id and value_indices"
            )
        feature_id = str(column["feature_id"]).strip()
        indices = column["value_indices"]
        if (
            not feature_id
            or not isinstance(indices, list)
            or not indices
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in indices
            )
            or len(set(indices)) != len(indices)
        ):
            raise error_type(
                "trajectory feature IDs must be nonempty and value_indices unique positive integers"
            )
        for value in indices:
            identity = (feature_id, value)
            if identity in identities:
                raise error_type("trajectory feature columns must be unique")
            identities.add(identity)
        normalized.append({
            "feature_id": feature_id,
            "value_indices": list(indices),
        })
    return {
        "feature_source": "trajectory_features",
        "trajectory_feature_columns": normalized,
    }


def _pca_replica_records(
    shard: ReplicaShard,
) -> Tuple[List[Dict[str, object]], List[Vector]]:
    system_id, replica, zero_based = shard.payload  # type: ignore[misc]
    metadata: List[Dict[str, object]] = []
    vectors: List[Vector] = []
    assert isinstance(replica, dict)
    for segment in replica["segments"]:
        assert isinstance(segment, dict)
        for projection in segment["projections"]:
            assert isinstance(projection, dict)
            scores = projection["scores_angstrom"]
            assert isinstance(scores, list)
            if max(zero_based) >= len(scores):
                raise ValueError(
                    "component_indices exceed components returned by common_pca"
                )
            vector = tuple(float(scores[index]) for index in zero_based)
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("PCA feature vector is non-finite")
            metadata.append({
                "system_id": str(system_id),
                "replica_id": str(replica["replica_id"]),
                "segment_id": str(segment["segment_id"]),
                "source_frame_index": projection["source_frame_index"],
                **(
                    {"member_id": str(projection["member_id"])}
                    if "member_id" in projection else {}
                ),
                **(
                    {"sample_index": projection["sample_index"]}
                    if "sample_index" in projection else {
                        "time": projection["time"],
                        "time_unit": projection["time_unit"],
                    }
                ),
            })
            vectors.append(vector)
    return metadata, vectors


def _pca_records(
    report: Mapping[str, object], component_indices: Sequence[int], error_type: Type[ValueError]
) -> Tuple[
    List[Dict[str, object]], List[Vector], List[Dict[str, object]],
    Dict[str, object],
]:
    metadata: List[Dict[str, object]] = []
    vectors: List[Vector] = []
    zero_based = [value - 1 for value in component_indices]
    systems = report.get("systems")
    if not isinstance(systems, list):
        raise error_type("common_pca produced no systems")
    shards = []
    for system in systems:
        assert isinstance(system, dict)
        for replica in system["replicas"]:
            assert isinstance(replica, dict)
            segments = replica.get("segments")
            assert isinstance(segments, list)
            shards.append(ReplicaShard(
                ordinal=len(shards),
                system_id=str(system["system_id"]),
                replica_id=str(replica["replica_id"]),
                segment_ids=tuple(str(segment["segment_id"]) for segment in segments),
                payload=(str(system["system_id"]), replica, zero_based),
            ))
    try:
        partials, evidence = execute_replica_workers(
            shards, _pca_replica_records,
            maximum_workers=len(shards),
            worker_backend="thread",
        )
    except (ValueError, TypeError) as exc:
        raise error_type(str(exc)) from exc
    for partial in partials:
        replica_metadata, replica_vectors = partial.value
        metadata.extend(replica_metadata)
        vectors.extend(replica_vectors)
    columns = [
        {
            "column_index": index + 1,
            "source": "common_pca",
            "component_index": component,
            "label": f"PC{component}",
            "unit": "angstrom",
        }
        for index, component in enumerate(component_indices)
    ]
    return metadata, vectors, columns, evidence.as_dict()


def _trajectory_records(
    report: Mapping[str, object], selections: Sequence[Mapping[str, object]],
    error_type: Type[ValueError],
) -> Tuple[List[Dict[str, object]], List[Vector], List[Dict[str, object]]]:
    metadata: List[Dict[str, object]] = []
    vectors: List[Vector] = []
    output_columns: List[Dict[str, object]] = []
    segments = report.get("segments")
    if not isinstance(segments, list):
        raise error_type("trajectory_features produced no segments")
    for segment_index, segment in enumerate(segments):
        assert isinstance(segment, dict)
        features = segment.get("features")
        if not isinstance(features, list):
            raise error_type("trajectory_features segment has no features")
        by_id = {
            str(feature["feature_id"]): feature
            for feature in features if isinstance(feature, dict)
        }
        chosen = []
        for selection in selections:
            feature_id = str(selection["feature_id"])
            if feature_id not in by_id:
                raise error_type(
                    f"trajectory feature {feature_id} is absent from a segment"
                )
            chosen.append((selection, by_id[feature_id]))
        first_records = chosen[0][1].get("records")
        if not isinstance(first_records, list):
            raise error_type("trajectory feature records must be an array")
        for feature_selection, feature in chosen:
            records = feature.get("records")
            if not isinstance(records, list) or len(records) != len(first_records):
                raise error_type(
                    "selected trajectory features do not have identical record counts"
                )
            dimension = feature.get("dimension")
            if not isinstance(dimension, int):
                raise error_type("trajectory feature dimension is invalid")
            if max(feature_selection["value_indices"]) > dimension:  # type: ignore[arg-type]
                raise error_type(
                    f"value_indices exceed dimension of trajectory feature {feature['feature_id']}"
                )
        for row_index, first in enumerate(first_records):
            assert isinstance(first, dict)
            identity = (first.get("source_frame_index"), first.get("axis_kind"), first.get("axis_value"))
            vector_values: List[float] = []
            for feature_selection, feature in chosen:
                record = feature["records"][row_index]  # type: ignore[index]
                assert isinstance(record, dict)
                other_identity = (
                    record.get("source_frame_index"), record.get("axis_kind"),
                    record.get("axis_value"),
                )
                if identity != other_identity:
                    raise error_type(
                        "selected trajectory features are not frame-aligned"
                    )
                values = record.get("values")
                if not isinstance(values, list):
                    raise error_type("trajectory feature values must be an array")
                vector_values.extend(
                    float(values[index - 1])
                    for index in feature_selection["value_indices"]  # type: ignore[union-attr]
                )
            if not all(math.isfinite(value) for value in vector_values):
                raise error_type("trajectory feature vector is non-finite")
            metadata.append({
                "system_id": str(segment["system_id"]),
                "replica_id": str(segment["replica_id"]),
                "segment_id": str(segment["segment_id"]),
                "source_frame_index": first["source_frame_index"],
                "axis_kind": first["axis_kind"],
                "axis_value": first["axis_value"],
            })
            vectors.append(tuple(vector_values))
        if segment_index == 0:
            column_index = 1
            for selection, feature in chosen:
                labels = feature.get("value_labels", [])
                for value_index in selection["value_indices"]:  # type: ignore[union-attr]
                    label = (
                        str(labels[value_index - 1])
                        if isinstance(labels, list) and value_index <= len(labels)
                        else f"value_{value_index}"
                    )
                    output_columns.append({
                        "column_index": column_index,
                        "source": "trajectory_features",
                        "feature_id": str(selection["feature_id"]),
                        "value_index": value_index,
                        "label": label,
                    })
                    column_index += 1
    return metadata, vectors, output_columns


def _tica_records(
    report: Mapping[str, object], component_indices: Sequence[int],
    error_type: Type[ValueError],
) -> Tuple[List[Dict[str, object]], List[Vector], List[Dict[str, object]]]:
    metadata: List[Dict[str, object]] = []
    vectors: List[Vector] = []
    zero_based = [value - 1 for value in component_indices]
    segments = report.get("segments")
    if not isinstance(segments, list):
        raise error_type("tica produced no segments")
    for segment in segments:
        if not isinstance(segment, dict):
            raise error_type("tica segment must be an object")
        projections = segment.get("projections")
        if not isinstance(projections, list):
            raise error_type("tica segment has no projections")
        for projection in projections:
            if not isinstance(projection, dict):
                raise error_type("tica projection must be an object")
            scores = projection.get("scores")
            if not isinstance(scores, list) or not scores:
                raise error_type("tica projection has no component scores")
            if max(zero_based) >= len(scores):
                raise error_type(
                    "component_indices exceed components returned by tica"
                )
            vector = tuple(float(scores[index]) for index in zero_based)
            if not all(math.isfinite(value) for value in vector):
                raise error_type("tica feature vector is non-finite")
            metadata.append({
                "system_id": str(segment["system_id"]),
                "replica_id": str(segment["replica_id"]),
                "segment_id": str(segment["segment_id"]),
                "source_frame_index": projection["source_frame_index"],
                **(
                    {"member_id": str(projection["member_id"])}
                    if "member_id" in projection else {}
                ),
                "time": projection["time"],
                "time_unit": projection["time_unit"],
            })
            vectors.append(vector)
    columns = [
        {
            "column_index": index + 1,
            "source": "tica",
            "component_index": component,
            "label": f"tIC{component}",
            "unit": "dimensionless_projection",
        }
        for index, component in enumerate(component_indices)
    ]
    return metadata, vectors, columns


def load_feature_matrix(
    project_path: Path,
    selection: Mapping[str, object],
    *,
    hash_content: bool,
    error_type: Type[ValueError],
) -> Tuple[Mapping[str, object], List[Dict[str, object]], List[Vector], Dict[str, object]]:
    """Run the selected upstream module and return aligned metadata and vectors."""

    if selection["feature_source"] == "common_pca":
        report = common_pca_project(project_path, hash_content=hash_content)
        metadata, vectors, columns, extraction_parallelism = _pca_records(
            report,
            selection["component_indices"],  # type: ignore[arg-type]
            error_type,
        )
    elif selection["feature_source"] == "tica":
        try:
            report = time_lagged_independent_component_analysis_project(
                project_path, hash_content=hash_content
            )
        except TICAAnalysisError as exc:
            raise error_type(str(exc)) from exc
        metadata, vectors, columns = _tica_records(
            report,
            selection["component_indices"],  # type: ignore[arg-type]
            error_type,
        )
        extraction_parallelism = {
            "execution_model": "serial_identity_preserving_feature_extraction",
            "workers_used": 1,
        }
    else:
        report = trajectory_features_project(project_path, hash_content=hash_content)
        metadata, vectors, columns = _trajectory_records(
            report,
            selection["trajectory_feature_columns"],  # type: ignore[arg-type]
            error_type,
        )
        extraction_parallelism = {
            "execution_model": "serial_identity_preserving_feature_extraction",
            "workers_used": 1,
        }
    if not vectors:
        raise error_type(f"{selection['feature_source']} produced no feature records")
    return report, metadata, vectors, {
        "source": selection["feature_source"],
        "observation_count": len(vectors),
        "feature_count": len(vectors[0]),
        "columns": columns,
        "replica_feature_extraction": extraction_parallelism,
        **(
            {
                "symmetry_expansion": report["settings"].get("symmetry_expansion"),
                "observation_accounting": report.get("observation_accounting"),
            }
            if selection["feature_source"] in {"common_pca", "tica"}
            and isinstance(report.get("settings"), dict)
            and (
                isinstance(report["settings"].get("symmetry_expansion"), dict)
                or isinstance(report.get("observation_accounting"), dict)
            )
            else {}
        ),
        **{
            key: selection[key]
            for key in ("component_indices", "trajectory_feature_columns")
            if key in selection
        },
    }
