"""Temporary one-replica project shards for exact parallel module execution.

The files are execution adapters only.  They never alter the scientific
project or source manifests, and every trajectory, topology, connectivity, and
reference path is resolved back to the original immutable input.
"""

from __future__ import annotations

import copy
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Tuple

from .manifests import load_json, resolve_manifest_path, validate_project, validate_system
from .replica_execution import ReplicaShard


class ReplicaProjectError(ValueError):
    """Raised when a project cannot be losslessly split by replica."""


def _absolute_project_paths(project: Dict[str, object], source: Path) -> None:
    for key in ("reference_structure", "reference_connectivity"):
        value = project.get(key)
        if isinstance(value, str):
            project[key] = str(resolve_manifest_path(value, source))
    preprocessed = project.get("preprocessed_coordinate_source")
    if isinstance(preprocessed, dict):
        report = preprocessed.get("cache_report")
        if isinstance(report, str):
            preprocessed["cache_report"] = str(resolve_manifest_path(report, source))


def _absolute_replica_paths(replica: Dict[str, object], system_source: Path) -> None:
    for key in ("topology", "connectivity"):
        value = replica.get(key)
        if isinstance(value, str):
            replica[key] = str(resolve_manifest_path(value, system_source))
    segments = replica.get("segments")
    if not isinstance(segments, list):
        raise ReplicaProjectError("replica segments are malformed")
    for segment in segments:
        if not isinstance(segment, dict):
            raise ReplicaProjectError("replica segment is malformed")
        for key in ("trajectory", "weights"):
            value = segment.get(key)
            if isinstance(value, str):
                segment[key] = str(resolve_manifest_path(value, system_source))


@contextmanager
def materialized_replica_project_shards(
    project_path: Path,
) -> Iterator[Tuple[List[ReplicaShard], Dict[str, object]]]:
    """Yield stable one-replica project paths and the untouched source project."""

    source = Path(project_path).expanduser().resolve(strict=False)
    source_project = load_json(source)
    system_value = source_project.get("system_manifest")
    if not isinstance(system_value, str):
        raise ReplicaProjectError("project system_manifest is missing")
    system_source = resolve_manifest_path(system_value, source)
    source_system = load_json(system_source)
    systems = source_system.get("systems")
    if not isinstance(systems, list) or not systems:
        raise ReplicaProjectError("system manifest contains no systems")

    with tempfile.TemporaryDirectory(prefix="sma-replica-shards-") as directory:
        root = Path(directory)
        shards: List[ReplicaShard] = []
        ordinal = 0
        for raw_system in systems:
            if not isinstance(raw_system, dict):
                raise ReplicaProjectError("system entry is malformed")
            system_id = str(raw_system.get("system_id", ""))
            replicas = raw_system.get("replicas")
            if not system_id or not isinstance(replicas, list):
                raise ReplicaProjectError("system identity or replicas are malformed")
            for raw_replica in replicas:
                if not isinstance(raw_replica, dict):
                    raise ReplicaProjectError("replica entry is malformed")
                replica = copy.deepcopy(raw_replica)
                replica_id = str(replica.get("replica_id", ""))
                if not replica_id:
                    raise ReplicaProjectError("replica identity is missing")
                _absolute_replica_paths(replica, system_source)
                segment_ids = tuple(
                    str(segment.get("segment_id", ""))
                    for segment in replica["segments"]  # type: ignore[index]
                    if isinstance(segment, dict)
                )
                if len(segment_ids) != len(replica["segments"]):  # type: ignore[arg-type]
                    raise ReplicaProjectError("segment identities are malformed")
                shard_system = {
                    "systems": [{
                        "system_id": system_id,
                        "metadata": copy.deepcopy(raw_system.get("metadata", {})),
                        "replicas": [replica],
                    }]
                }
                system_path = root / f"system-{ordinal:05d}.json"
                system_path.write_text(
                    json.dumps(shard_system, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                shard_project = copy.deepcopy(source_project)
                _absolute_project_paths(shard_project, source)
                shard_project["project_id"] = (
                    f"{source_project.get('project_id', 'project')}--"
                    f"{system_id}--{replica_id}"
                )
                shard_project["system_manifest"] = str(system_path)
                shard_project["reference_system"] = system_id
                project_output = root / f"project-{ordinal:05d}.json"
                project_output.write_text(
                    json.dumps(shard_project, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                validate_system(shard_system, source_path=system_path, check_paths=True)
                validate_project(
                    shard_project, source_path=project_output, check_paths=True
                )
                shards.append(ReplicaShard(
                    ordinal=ordinal,
                    system_id=system_id,
                    replica_id=replica_id,
                    segment_ids=segment_ids,
                    payload={"project_path": str(project_output)},
                ))
                ordinal += 1
        if not shards:
            raise ReplicaProjectError("project contains no replica shards")
        yield shards, source_project


def replica_project_path(shard: ReplicaShard) -> Path:
    payload = shard.payload
    if not isinstance(payload, Mapping):
        raise ReplicaProjectError("replica shard payload is malformed")
    value = payload.get("project_path")
    if not isinstance(value, str):
        raise ReplicaProjectError("replica shard has no project path")
    return Path(value)
