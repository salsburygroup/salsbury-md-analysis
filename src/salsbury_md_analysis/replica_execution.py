"""Deterministic replica-worker and pooled-reducer execution primitives.

Replica workers are an execution optimization.  They do not redefine an
estimator.  Every shard retains system, replica, and continuous-segment
identity; the caller supplies the scientifically appropriate reducer.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, Tuple, TypeVar


class ReplicaExecutionError(ValueError):
    """Raised when replica sharding or reduction loses estimator identity."""


@dataclass(frozen=True)
class ReplicaShard:
    """One stable worker input with explicit continuous-segment identity."""

    ordinal: int
    system_id: str
    replica_id: str
    segment_ids: Tuple[str, ...]
    payload: object

    @property
    def identity(self) -> Tuple[str, str]:
        return self.system_id, self.replica_id


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class ReplicaPartial(Generic[T]):
    """One identity-preserving worker output."""

    ordinal: int
    system_id: str
    replica_id: str
    segment_ids: Tuple[str, ...]
    value: T

    @property
    def identity(self) -> Tuple[str, str]:
        return self.system_id, self.replica_id


@dataclass(frozen=True)
class ReplicaExecutionEvidence:
    execution_model: str
    configured_maximum_workers: int
    scheduler_cpu_limit: int
    workers_used: int
    shard_count: int
    stable_reduction_order: Tuple[Tuple[str, str], ...]
    segment_boundaries_preserved: bool
    worker_backend: str

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["stable_reduction_order"] = [
            {"system_id": system_id, "replica_id": replica_id}
            for system_id, replica_id in self.stable_reduction_order
        ]
        return payload


def _validated_shards(shards: Sequence[ReplicaShard]) -> list[ReplicaShard]:
    if not shards:
        raise ReplicaExecutionError("replica execution requires at least one shard")
    rows = list(shards)
    identities = set()
    ordinals = set()
    for shard in rows:
        if not shard.system_id or not shard.replica_id:
            raise ReplicaExecutionError("replica shard identities must be nonempty")
        if not shard.segment_ids or any(not value for value in shard.segment_ids):
            raise ReplicaExecutionError(
                "replica shards must declare every continuous segment identity"
            )
        if shard.identity in identities:
            raise ReplicaExecutionError(
                f"duplicate replica shard {shard.system_id}/{shard.replica_id}"
            )
        if shard.ordinal in ordinals or shard.ordinal < 0:
            raise ReplicaExecutionError("replica shard ordinals must be unique")
        identities.add(shard.identity)
        ordinals.add(shard.ordinal)
    expected = list(range(len(rows)))
    if sorted(ordinals) != expected:
        raise ReplicaExecutionError(
            "replica shard ordinals must be contiguous from zero"
        )
    return sorted(rows, key=lambda row: row.ordinal)


def _worker_partial(
    arguments: tuple[Callable[[ReplicaShard], T], ReplicaShard]
) -> ReplicaPartial[T]:
    worker, shard = arguments
    return ReplicaPartial(
        ordinal=shard.ordinal,
        system_id=shard.system_id,
        replica_id=shard.replica_id,
        segment_ids=shard.segment_ids,
        value=worker(shard),
    )


def execute_replica_workers(
    shards: Sequence[ReplicaShard],
    worker: Callable[[ReplicaShard], T],
    *,
    maximum_workers: int,
    scheduler_cpu_limit: Optional[int] = None,
    worker_backend: str = "process",
) -> tuple[list[ReplicaPartial[T]], ReplicaExecutionEvidence]:
    """Execute stable replica shards without applying a scientific reducer."""

    rows = _validated_shards(shards)
    if (
        isinstance(maximum_workers, bool)
        or not isinstance(maximum_workers, int)
        or maximum_workers <= 0
    ):
        raise ReplicaExecutionError("maximum_workers must be a positive integer")
    scheduler_limit = scheduler_cpu_limit
    if scheduler_limit is None:
        scheduler_limit = int(os.environ.get("SLURM_CPUS_PER_TASK", maximum_workers))
    if (
        isinstance(scheduler_limit, bool)
        or not isinstance(scheduler_limit, int)
        or scheduler_limit <= 0
    ):
        raise ReplicaExecutionError("scheduler CPU limit must be a positive integer")
    workers = min(maximum_workers, scheduler_limit, len(rows))
    if worker_backend not in {"process", "thread"}:
        raise ReplicaExecutionError("worker_backend must be process or thread")
    if workers == 1:
        partials = [_worker_partial((worker, row)) for row in rows]
    else:
        executor_type = (
            ProcessPoolExecutor if worker_backend == "process" else ThreadPoolExecutor
        )
        with executor_type(max_workers=workers) as executor:
            partials = list(executor.map(
                _worker_partial, [(worker, row) for row in rows]
            ))
    partials.sort(key=lambda row: row.ordinal)
    expected = [row.identity for row in rows]
    observed = [row.identity for row in partials]
    if observed != expected:
        raise ReplicaExecutionError(
            "replica worker outputs do not exactly match the declared shard order"
        )
    evidence = ReplicaExecutionEvidence(
        execution_model="identity_preserving_replica_workers_v1",
        configured_maximum_workers=maximum_workers,
        scheduler_cpu_limit=scheduler_limit,
        workers_used=workers,
        shard_count=len(rows),
        stable_reduction_order=tuple(expected),
        segment_boundaries_preserved=True,
        worker_backend=worker_backend,
    )
    return partials, evidence


def reduce_replica_partials(
    partials: Sequence[ReplicaPartial[T]],
    reducer: Callable[[Sequence[ReplicaPartial[T]]], R],
) -> R:
    """Apply one caller-declared reducer to a complete stable partial set."""

    if not partials:
        raise ReplicaExecutionError("replica reduction requires partial results")
    ordered = sorted(partials, key=lambda row: row.ordinal)
    if [row.ordinal for row in ordered] != list(range(len(ordered))):
        raise ReplicaExecutionError(
            "replica partials are missing or repeat a shard ordinal"
        )
    if len({row.identity for row in ordered}) != len(ordered):
        raise ReplicaExecutionError("replica partial identities are not unique")
    return reducer(ordered)


def execute_replica_map_reduce(
    shards: Sequence[ReplicaShard],
    worker: Callable[[ReplicaShard], T],
    reducer: Callable[[Sequence[ReplicaPartial[T]]], R],
    *,
    maximum_workers: int,
    scheduler_cpu_limit: Optional[int] = None,
    worker_backend: str = "process",
) -> tuple[R, ReplicaExecutionEvidence]:
    """Run replica workers, validate exact coverage, then reduce once globally."""

    partials, evidence = execute_replica_workers(
        shards,
        worker,
        maximum_workers=maximum_workers,
        scheduler_cpu_limit=scheduler_cpu_limit,
        worker_backend=worker_backend,
    )
    return reduce_replica_partials(partials, reducer), evidence
