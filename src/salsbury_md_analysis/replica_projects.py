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

from .manifests import (
    load_json,
    resolve_manifest_path,
    sha256_file,
    validate_project,
    validate_system,
)
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


def _validated_preprocessed_cache_report(
    project: Mapping[str, object], source: Path, system_source: Path
) -> tuple[Path, str, Dict[str, object]] | None:
    declared = project.get("preprocessed_coordinate_source")
    if not isinstance(declared, Mapping):
        return None
    report_value = declared.get("cache_report")
    expected_digest = declared.get("cache_report_sha256")
    if not isinstance(report_value, str) or not isinstance(expected_digest, str):
        raise ReplicaProjectError(
            "preprocessed_coordinate_source requires cache_report and "
            "cache_report_sha256"
        )
    report_path = resolve_manifest_path(report_value, source)
    actual_digest = sha256_file(report_path)
    if actual_digest.lower() != expected_digest.lower():
        raise ReplicaProjectError(
            "preprocessed coordinate cache report hash does not match"
        )
    report = load_json(report_path)
    if (
        not isinstance(report, dict)
        or report.get("technical_status") != "complete"
        or report.get("coordinate_representation")
        != "continuous_unwrap_unaligned_strided"
        or report.get("selection") != "molecular_payload"
    ):
        raise ReplicaProjectError(
            "preprocessed coordinate cache report is not a complete molecular-payload cache"
        )
    cached_manifest = report.get("cached_system_manifest")
    cached_manifest_digest = report.get("cached_system_manifest_sha256")
    if not isinstance(cached_manifest, str) or not isinstance(
        cached_manifest_digest, str
    ):
        raise ReplicaProjectError(
            "preprocessed coordinate cache report lacks manifest identity"
        )
    reported_manifest = resolve_manifest_path(cached_manifest, report_path)
    if reported_manifest != system_source:
        raise ReplicaProjectError(
            "preprocessed cache report names a different system manifest"
        )
    if sha256_file(system_source).lower() != cached_manifest_digest.lower():
        raise ReplicaProjectError(
            "preprocessed cache system manifest hash does not match"
        )
    return report_path, actual_digest, report


def _write_replica_cache_report(
    *,
    source_report_path: Path,
    source_report_sha256: str,
    source_report: Mapping[str, object],
    system_path: Path,
    system_id: str,
    replica_id: str,
    destination: Path,
) -> str:
    """Bind one temporary shard manifest to its validated source cache report."""

    report = copy.deepcopy(dict(source_report))
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ReplicaProjectError("preprocessed coordinate cache report has no rows")
    shard_rows = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("system_id", "")) == system_id
        and str(row.get("replica_id", "")) == replica_id
    ]
    if len(shard_rows) != 1:
        raise ReplicaProjectError(
            f"preprocessed coordinate cache report does not name exactly one "
            f"row for {system_id}/{replica_id}"
        )
    report.update({
        "cached_system_manifest": str(system_path),
        "cached_system_manifest_sha256": sha256_file(system_path),
        "rows": shard_rows,
        "replica_shard_source_cache_report": {
            "path": str(source_report_path),
            "sha256": source_report_sha256,
        },
    })
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sha256_file(destination)


@contextmanager
def materialized_replica_project_shards(
    project_path: Path,
    *,
    temporary_root: Path | None = None,
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
    preprocessed_cache = _validated_preprocessed_cache_report(
        source_project, source, system_source
    )

    with tempfile.TemporaryDirectory(
        prefix="sma-replica-shards-",
        dir=None if temporary_root is None else str(temporary_root),
    ) as directory:
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
                if preprocessed_cache is not None:
                    report_path, report_sha256, report = preprocessed_cache
                    shard_report_path = root / f"cache-report-{ordinal:05d}.json"
                    shard_report_sha256 = _write_replica_cache_report(
                        source_report_path=report_path,
                        source_report_sha256=report_sha256,
                        source_report=report,
                        system_path=system_path,
                        system_id=system_id,
                        replica_id=replica_id,
                        destination=shard_report_path,
                    )
                    shard_project["preprocessed_coordinate_source"] = {
                        "cache_report": str(shard_report_path),
                        "cache_report_sha256": shard_report_sha256,
                    }
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
