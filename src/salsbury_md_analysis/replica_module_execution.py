"""Shared execution glue for replica-final analysis modules."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence

from .context import compile_project_context_file
from .replica_execution import (
    ReplicaExecutionEvidence,
    ReplicaPartial,
    ReplicaShard,
    execute_replica_workers,
)
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
        report = payload.get("candidate_harmonization_report")
        policy = report.get("policy") if isinstance(report, Mapping) else None
        if isinstance(raw_keys, list) and policy == "intersection_by_atom_identity_v2":
            keys = {
                tuple(tuple(atom) for atom in row)
                for row in raw_keys
            }
        elif isinstance(raw_keys, list):
            keys = {tuple(int(value) for value in row) for row in raw_keys}
        else:
            keys = None
        raw_donor_maps = payload.get("donor_endpoints_by_system")
        raw_donors = (
            raw_donor_maps.get(shard.system_id)
            if isinstance(raw_donor_maps, Mapping) else None
        )
        donor_endpoints = (
            {
                (tuple(row[0]), tuple(row[1]), str(row[2]))
                for row in raw_donors
            }
            if isinstance(raw_donors, list) else None
        )
        raw_acceptor_maps = payload.get("acceptor_endpoints_by_system")
        raw_acceptors = (
            raw_acceptor_maps.get(shard.system_id)
            if isinstance(raw_acceptor_maps, Mapping) else None
        )
        acceptor_endpoints = (
            {(tuple(row[0]), str(row[1])) for row in raw_acceptors}
            if isinstance(raw_acceptors, list) else None
        )
        return _hydrogen_bond_discovery_project_serial(
            project_path,
            hash_content=hash_content,
            harmonized_candidate_keys_override=keys,
            candidate_harmonization_report_override=(
                dict(report) if isinstance(report, Mapping) else None
            ),
            donor_endpoints_by_system_override=(
                {shard.system_id: donor_endpoints}
                if donor_endpoints is not None else None
            ),
            acceptor_endpoints_by_system_override=(
                {shard.system_id: acceptor_endpoints}
                if acceptor_endpoints is not None else None
            ),
        )
    if runner_id == "water_networks":
        from .water_mediated_hydrogen_bonds import (
            _water_mediated_hydrogen_bond_networks_project_serial,
        )
        return _water_mediated_hydrogen_bond_networks_project_serial(
            project_path, hash_content=hash_content
        )
    if runner_id == "structural_qc":
        from .structural_qc import _structural_qc_project_serial
        raw_project = payload.get("structural_qc_project_path")
        if not isinstance(raw_project, str) or not raw_project:
            raise ReplicaModuleExecutionError(
                "structural-QC replica worker lacks its cache-backed project"
            )
        return _structural_qc_project_serial(
            Path(raw_project),
            hash_content=hash_content,
            only_system_id=shard.system_id,
            only_replica_id=shard.replica_id,
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


def _distributed_worker_entry(manifest_path: Path) -> None:
    """Run the stable shard ordinals assigned to one Slurm task rank."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = manifest.get("shards")
    output_directory = manifest.get("output_directory")
    if not isinstance(rows, list) or not isinstance(output_directory, str):
        raise ReplicaModuleExecutionError("distributed replica manifest is malformed")
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    task_count = int(os.environ.get("SLURM_NTASKS", "1"))
    if rank < 0 or task_count <= 0 or rank >= task_count:
        raise ReplicaModuleExecutionError("distributed Slurm rank metadata is invalid")
    output_root = Path(output_directory)
    for ordinal in range(rank, len(rows), task_count):
        raw = rows[ordinal]
        if not isinstance(raw, dict):
            raise ReplicaModuleExecutionError("distributed replica shard is malformed")
        shard = ReplicaShard(
            ordinal=int(raw["ordinal"]),
            system_id=str(raw["system_id"]),
            replica_id=str(raw["replica_id"]),
            segment_ids=tuple(str(value) for value in raw["segment_ids"]),
            payload=raw["payload"],
        )
        partial = ReplicaPartial(
            ordinal=shard.ordinal,
            system_id=shard.system_id,
            replica_id=shard.replica_id,
            segment_ids=shard.segment_ids,
            value=_module_worker(shard),
        )
        destination = output_root / f"partial-{ordinal:05d}.json"
        temporary = destination.with_suffix(".json.partial")
        temporary.write_text(json.dumps({
            "ordinal": partial.ordinal,
            "system_id": partial.system_id,
            "replica_id": partial.replica_id,
            "segment_ids": list(partial.segment_ids),
            "value": partial.value,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)


def _distributed_replica_workers(
    shards: Sequence[ReplicaShard],
    *,
    maximum_workers: int,
) -> tuple[list[ReplicaPartial[Dict[str, object]]], ReplicaExecutionEvidence]:
    """Execute identity-preserving replica workers across a Slurm allocation."""

    node_count = int(os.environ.get("SLURM_NNODES", "1"))
    workers_per_node = int(os.environ.get(
        "SMA_REPLICA_WORKERS_PER_NODE",
        str(max(1, (maximum_workers + node_count - 1) // node_count)),
    ))
    if node_count <= 1 or workers_per_node <= 0:
        raise ReplicaModuleExecutionError(
            "distributed replica execution requires multiple valid Slurm nodes"
        )
    task_count = min(maximum_workers, len(shards))
    root = replica_project_path(shards[0]).parent
    output_root = root / "distributed-partials"
    output_root.mkdir(parents=False, exist_ok=False)
    manifest_path = root / "distributed-worker-manifest.json"
    manifest_path.write_text(json.dumps({
        "output_directory": str(output_root),
        "shards": [{
            "ordinal": shard.ordinal,
            "system_id": shard.system_id,
            "replica_id": shard.replica_id,
            "segment_ids": list(shard.segment_ids),
            "payload": shard.payload,
        } for shard in shards],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [
        os.environ.get("SMA_SRUN_COMMAND", "srun"),
        "--nodes", str(node_count),
        "--ntasks", str(task_count),
        "--ntasks-per-node", str(workers_per_node),
        sys.executable,
        "-m", "salsbury_md_analysis.replica_module_execution",
        "--worker-manifest", str(manifest_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ReplicaModuleExecutionError(
            "distributed replica workers failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    partials: list[ReplicaPartial[Dict[str, object]]] = []
    for ordinal, shard in enumerate(shards):
        path = output_root / f"partial-{ordinal:05d}.json"
        if not path.is_file():
            raise ReplicaModuleExecutionError(
                f"distributed replica worker omitted shard {ordinal}"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            int(raw.get("ordinal", -1)) != shard.ordinal
            or str(raw.get("system_id", "")) != shard.system_id
            or str(raw.get("replica_id", "")) != shard.replica_id
        ):
            raise ReplicaModuleExecutionError(
                f"distributed replica identity changed for shard {ordinal}"
            )
        partials.append(ReplicaPartial(
            ordinal=shard.ordinal,
            system_id=shard.system_id,
            replica_id=shard.replica_id,
            segment_ids=shard.segment_ids,
            value=raw["value"],
        ))
    return partials, ReplicaExecutionEvidence(
        execution_model="identity_preserving_distributed_replica_workers_v1",
        configured_maximum_workers=maximum_workers,
        scheduler_cpu_limit=task_count,
        workers_used=task_count,
        shard_count=len(shards),
        stable_reduction_order=tuple(shard.identity for shard in shards),
        segment_boundaries_preserved=True,
        worker_backend="slurm_srun_process",
    )


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
    distributed = (
        os.environ.get("SMA_DISTRIBUTED_REPLICA_WORKERS") == "1"
        and int(os.environ.get("SLURM_NNODES", "1")) > 1
    )
    temporary_root = (
        Path(os.environ.get("SMA_DISTRIBUTED_WORK_DIR", os.getcwd()))
        if distributed else None
    )
    with materialized_replica_project_shards(
        source, temporary_root=temporary_root
    ) as (base_shards, _):
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
        if distributed:
            partials, evidence = _distributed_replica_workers(
                shards, maximum_workers=maximum_workers
            )
        else:
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


def execute_registered_replica_workers(
    shards: Sequence[ReplicaShard],
    *,
    maximum_workers: int,
) -> tuple[list[ReplicaPartial[Dict[str, object]]], ReplicaExecutionEvidence]:
    """Run registered replica workers locally or across a Slurm allocation."""

    distributed = (
        os.environ.get("SMA_DISTRIBUTED_REPLICA_WORKERS") == "1"
        and int(os.environ.get("SLURM_NNODES", "1")) > 1
    )
    workers = min(maximum_workers, len(shards))
    if distributed:
        return _distributed_replica_workers(shards, maximum_workers=workers)
    return execute_replica_workers(
        shards,
        _module_worker,
        maximum_workers=workers,
        scheduler_cpu_limit=workers,
        worker_backend="process",
    )


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


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker-manifest":
        raise SystemExit(
            "usage: python -m salsbury_md_analysis.replica_module_execution "
            "--worker-manifest PATH"
        )
    _distributed_worker_entry(Path(sys.argv[2]))
