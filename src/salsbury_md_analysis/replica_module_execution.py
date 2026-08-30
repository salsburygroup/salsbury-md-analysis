"""Shared execution glue for replica-final analysis modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence

from .context import compile_project_context_file
from .replica_execution import ReplicaPartial, ReplicaShard, execute_replica_workers
from .replica_projects import materialized_replica_project_shards, replica_project_path


class ReplicaModuleExecutionError(ValueError):
    """Raised when shard reports cannot be reduced without changing meaning."""


def _module_worker(shard: ReplicaShard) -> Dict[str, object]:
    payload = shard.payload
    if not isinstance(payload, Mapping):
        raise ReplicaModuleExecutionError("replica module payload is malformed")
    runner_id = payload.get("runner_id")
    hash_content = bool(payload.get("hash_content", False))
    project_path = replica_project_path(shard)
    if runner_id == "sasa":
        from .sasa import _solvent_accessible_surface_area_project_serial
        return _solvent_accessible_surface_area_project_serial(
            project_path, hash_content=hash_content
        )
    if runner_id == "rmsd_rg":
        from .rmsd_rg import _replica_rmsd_rg_project_serial
        return _replica_rmsd_rg_project_serial(project_path, hash_content=hash_content)
    if runner_id == "secondary_structure":
        from .secondary_structure import _secondary_structure_project_serial
        return _secondary_structure_project_serial(
            project_path, hash_content=hash_content
        )
    if runner_id == "ion_geometry":
        from .ion_geometry import _ion_coordination_geometry_project_serial
        return _ion_coordination_geometry_project_serial(
            project_path, hash_content=hash_content
        )
    if runner_id == "ion_atmosphere":
        from .ion_atmosphere import _ion_atmosphere_project_serial
        return _ion_atmosphere_project_serial(project_path, hash_content=hash_content)
    if runner_id == "rmsf":
        from .rmsf import _pooled_rmsf_project_serial
        return _pooled_rmsf_project_serial(project_path, hash_content=hash_content)
    if runner_id == "dccm":
        from .dccm import _dccm_project_serial
        return _dccm_project_serial(
            project_path,
            hash_content=hash_content,
            allow_incomplete_pooled_reference=True,
        )
    if runner_id == "hydrogen_bond_discovery":
        from .hydrogen_bond_discovery import _hydrogen_bond_discovery_project_serial
        raw_keys = payload.get("harmonized_candidate_keys")
        keys = (
            {tuple(int(value) for value in row) for row in raw_keys}
            if isinstance(raw_keys, list) else None
        )
        report = payload.get("candidate_harmonization_report")
        return _hydrogen_bond_discovery_project_serial(
            project_path,
            hash_content=hash_content,
            harmonized_candidate_keys_override=keys,
            candidate_harmonization_report_override=(
                dict(report) if isinstance(report, Mapping) else None
            ),
        )
    if runner_id == "water_networks":
        from .water_mediated_hydrogen_bonds import (
            _water_mediated_hydrogen_bond_networks_project_serial,
        )
        return _water_mediated_hydrogen_bond_networks_project_serial(
            project_path, hash_content=hash_content
        )
    raise ReplicaModuleExecutionError(f"unsupported replica runner {runner_id!r}")


def configured_replica_workers(shard_count: int) -> int:
    """Return the explicit local or scheduler worker allocation."""

    value = os.environ.get("SMA_REPLICA_WORKERS", os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    try:
        requested = int(value)
    except ValueError as exc:
        raise ReplicaModuleExecutionError(
            "SMA_REPLICA_WORKERS/SLURM_CPUS_PER_TASK must be a positive integer"
        ) from exc
    if requested <= 0:
        raise ReplicaModuleExecutionError("replica worker allocation must be positive")
    return min(requested, shard_count)


def execute_replica_final_module(
    project_path: Path,
    *,
    runner_id: str,
    hash_content: bool,
    reducer: Callable[[Sequence[ReplicaPartial[Dict[str, object]]], Dict[str, object]], Dict[str, object]],
    worker_payload: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Execute every replica once and apply one declared exact reducer."""

    source = Path(project_path).expanduser().resolve(strict=False)
    source_context = compile_project_context_file(source, hash_content=hash_content)
    with materialized_replica_project_shards(source) as (base_shards, _):
        shards = [
            ReplicaShard(
                ordinal=shard.ordinal,
                system_id=shard.system_id,
                replica_id=shard.replica_id,
                segment_ids=shard.segment_ids,
                payload={
                    **dict(shard.payload),
                    "runner_id": runner_id,
                    "hash_content": hash_content,
                    **dict(worker_payload or {}),
                },
            )
            for shard in base_shards
        ]
        maximum_workers = configured_replica_workers(len(shards))
        partials, evidence = execute_replica_workers(
            shards,
            _module_worker,
            maximum_workers=maximum_workers,
            scheduler_cpu_limit=maximum_workers,
            worker_backend="process",
        )
    report = reducer(partials, source_context)
    report["replica_execution"] = evidence.as_dict()
    return report


def merge_frame_selection_reports(
    reports: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Merge per-replica frame-selection reports without changing selection."""

    if not reports:
        raise ReplicaModuleExecutionError("frame-selection reduction is empty")
    stable_fields = (
        "mode", "resolved_mode", "frame_stride", "resolved_integer_stride",
        "selection_contract",
    )
    first = reports[0]
    for report in reports[1:]:
        for field in stable_fields:
            if report.get(field) != first.get(field):
                raise ReplicaModuleExecutionError(
                    f"replica frame-selection field {field} is inconsistent"
                )
    source_count = sum(int(report["source_frame_count"]) for report in reports)
    selected_count = sum(int(report["selected_frame_count"]) for report in reports)
    merged = {field: first.get(field) for field in stable_fields}
    merged.update({
        "source_frame_count": source_count,
        "selected_frame_count": selected_count,
        "coverage_fraction": selected_count / source_count,
        "replicas": [
            row
            for report in reports
            for row in report.get("replicas", [])  # type: ignore[union-attr]
        ],
    })
    return merged


def unique_issues(reports: Sequence[Mapping[str, object]]) -> list[Dict[str, object]]:
    """Deduplicate project-level issues repeated by one-replica workers."""

    result: list[Dict[str, object]] = []
    seen = set()
    for report in reports:
        for issue in report.get("issues", []):  # type: ignore[union-attr]
            if not isinstance(issue, dict):
                continue
            identity = json.dumps(issue, sort_keys=True, separators=(",", ":"))
            if identity not in seen:
                seen.add(identity)
                result.append(dict(issue))
    return result


def restore_source_provenance(
    report: Dict[str, object], source_context: Mapping[str, object]
) -> None:
    """Replace temporary-shard provenance with the immutable source contract."""

    report["project_manifest_path"] = source_context["project_manifest_path"]
    report["project_manifest_sha256"] = source_context["project_manifest_sha256"]
    report["system_manifest_path"] = source_context["system_manifest_path"]
    report["system_manifest_sha256"] = source_context["system_manifest_sha256"]
    report["contract_signature_sha256"] = source_context["contract_signature_sha256"]
    report["input_content_signature_sha256"] = source_context[
        "input_content_signature_sha256"
    ]
