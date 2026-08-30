"""Reusable connectivity-aware solute coordinate caches for large campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import BinaryIO, Dict, Mapping, Sequence

import numpy as np

from .atom_mapping import AtomMappingError, AtomRecord, read_pdb_atoms
from .coordinates import (
    DCD_MADE_WHOLE_CACHE_TITLE,
    CellVectors,
    CoordinateFrame,
    iter_coordinate_frames,
)
from .frame_sampling import integer_stride_selected_count
from .manifests import (
    ManifestValidationError,
    load_json,
    resolve_manifest_path,
    sha256_file,
    validate_system,
)
from .periodic import (
    PeriodicFrameProcessor,
    PeriodicReconstructionError,
    load_connectivity,
)
from .preflight import FileProbeError, probe_trajectory
from .selections import select_atoms


class CoordinateCacheError(ValueError):
    """Raised when a reusable cache cannot be built without ambiguity."""


def coordinate_cache_prefix(system_id: str, replica_id: str) -> str:
    """Return the deterministic payload prefix for one cached replica."""

    safe_system = hashlib.sha256(system_id.encode("utf-8")).hexdigest()[:10]
    safe_replica = hashlib.sha256(replica_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe_system}-{safe_replica}"


def coordinate_cache_system_manifest_filename(system_id: str) -> str:
    """Return the deterministic single-system cache manifest filename."""

    safe_system = hashlib.sha256(system_id.encode("utf-8")).hexdigest()[:10]
    return f"system-cache-{safe_system}.json"


def _record(handle: BinaryIO, payload: bytes) -> None:
    marker = struct.pack("<i", len(payload))
    handle.write(marker)
    handle.write(payload)
    handle.write(marker)


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _angle(first: Sequence[float], second: Sequence[float]) -> float:
    denominator = _norm(first) * _norm(second)
    if denominator <= 0.0:
        raise CoordinateCacheError("periodic cell contains a zero-length vector")
    cosine = sum(float(a) * float(b) for a, b in zip(first, second)) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _cell_record(cell: CellVectors) -> bytes:
    a, b, c = cell
    values = (
        _norm(a), _angle(a, b), _norm(b), _angle(a, c), _angle(b, c), _norm(c)
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise CoordinateCacheError("periodic cell cannot be represented in DCD form")
    return struct.pack("<6d", *values)


class _DCDWriter:
    def __init__(
        self,
        path: Path,
        *,
        atom_count: int,
        frame_count: int,
        starting_step: int,
        save_interval: int,
        unit_cell_present: bool,
    ) -> None:
        if min(atom_count, frame_count, save_interval) <= 0:
            raise CoordinateCacheError("DCD output counts and interval must be positive")
        self.path = path
        self.atom_count = atom_count
        self.frame_count = frame_count
        self.unit_cell_present = unit_cell_present
        self.written = 0
        self.handle = path.open("xb")
        header = bytearray(84)
        header[:4] = b"CORD"
        struct.pack_into("<3i", header, 4, frame_count, starting_step, save_interval)
        if unit_cell_present:
            struct.pack_into("<i", header, 44, 1)
            struct.pack_into("<i", header, 80, 24)
        _record(self.handle, bytes(header))
        title = struct.pack("<i", 1) + DCD_MADE_WHOLE_CACHE_TITLE.ljust(80)
        _record(self.handle, title)
        _record(self.handle, struct.pack("<i", atom_count))

    def write(self, frame: CoordinateFrame, atom_indices: Sequence[int]) -> None:
        if frame.periodic_cell_present != self.unit_cell_present:
            raise CoordinateCacheError(
                "periodic-cell presence changes within one DCD segment"
            )
        if self.unit_cell_present:
            if frame.cell_vectors_angstrom is None:
                raise CoordinateCacheError("periodic DCD frame has no usable cell")
            _record(self.handle, _cell_record(frame.cell_vectors_angstrom))
        coordinates = np.asarray(frame.coordinates_angstrom, dtype=np.float64)
        try:
            selected = coordinates[np.asarray(atom_indices, dtype=np.int64)]
        except (IndexError, ValueError) as exc:
            raise CoordinateCacheError("cache atom selection exceeds a frame") from exc
        if selected.shape != (self.atom_count, 3) or not np.isfinite(selected).all():
            raise CoordinateCacheError("cache frame has invalid selected coordinates")
        for axis in range(3):
            _record(self.handle, np.asarray(selected[:, axis], dtype="<f4").tobytes())
        self.written += 1

    def close(self) -> None:
        self.handle.close()
        if self.written != self.frame_count:
            raise CoordinateCacheError(
                f"DCD cache wrote {self.written} frames; expected {self.frame_count}"
            )


def _topology_lines(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines(True)
    except (OSError, UnicodeError) as exc:
        raise CoordinateCacheError(str(exc)) from exc
    records = [line for line in lines if line[:6].strip().upper() in {"ATOM", "HETATM"}]
    if not records:
        raise CoordinateCacheError(f"topology contains no PDB atom records: {path}")
    return records


def _write_subset_pdb(
    path: Path,
    source_topology: Path,
    atom_indices: Sequence[int],
    reference_frame: CoordinateFrame,
) -> None:
    records = _topology_lines(source_topology)
    coordinates = reference_frame.coordinates_angstrom
    lines = []
    for index in atom_indices:
        try:
            original = records[index].rstrip("\n")
            x, y, z = coordinates[index]
        except IndexError as exc:
            raise CoordinateCacheError("topology/reference atom count mismatch") from exc
        if max(abs(float(x)), abs(float(y)), abs(float(z))) >= 10_000.0:
            raise CoordinateCacheError("reference coordinate exceeds PDB field range")
        padded = original.ljust(80)
        lines.append(
            padded[:30] + f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
            + padded[54:] + "\n"
        )
    path.write_text("".join(lines) + "END\n", encoding="ascii")


def _write_subset_connectivity(
    path: Path,
    source_connectivity: Path,
    source_atom_count: int,
    atom_indices: Sequence[int],
) -> Dict[str, object]:
    bonds, identity = load_connectivity(source_connectivity, source_atom_count)
    new_index = {source: target for target, source in enumerate(atom_indices)}
    subset = [
        [new_index[first], new_index[second]]
        for first, second in bonds
        if first in new_index and second in new_index
    ]
    payload = {
        "format": "salsbury-bonds-v1",
        "atom_count": len(atom_indices),
        "index_base": 0,
        "bonds": subset,
        "provenance": {
            "source_connectivity": str(source_connectivity),
            "source_connectivity_identity": identity,
            "selection": "molecular_payload",
            "source_atom_count": source_atom_count,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _source_identity(path: Path, hash_content: bool) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path) if hash_content else None,
    }


def _validate_source_identity(
    recorded: object, current_path: Path, label: str
) -> None:
    if not isinstance(recorded, dict):
        raise CoordinateCacheError(
            f"reusable cache lacks recorded {label} identity"
        )
    current = _source_identity(
        current_path, bool(recorded.get("sha256"))
    )
    for field in ("path", "size_bytes", "modified_time_ns", "sha256"):
        if recorded.get(field) != current.get(field):
            raise CoordinateCacheError(
                f"reusable cache {label} changed for {current_path}: {field}"
            )


def _write_cache_metadata(
    temporary: Path,
    output: Path,
    *,
    source: Path,
    cached_systems: Sequence[Mapping[str, object]],
    report_rows: Sequence[Mapping[str, object]],
    maximum_workers: int,
    cache_stride: int,
) -> Dict[str, object]:
    cached_manifest = {"systems": list(cached_systems)}
    manifest_path = temporary / "system-cache.json"
    manifest_path.write_text(
        json.dumps(cached_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_system(cached_manifest, source_path=manifest_path, check_paths=True)
    per_system_manifests: Dict[str, Dict[str, str]] = {}
    for cached_system in cached_systems:
        system_id = str(cached_system["system_id"])
        filename = coordinate_cache_system_manifest_filename(system_id)
        path = temporary / filename
        payload = {"systems": [cached_system]}
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_system(payload, source_path=path, check_paths=True)
        per_system_manifests[system_id] = {
            "path": filename,
            "sha256": sha256_file(path),
        }
    report = {
        "cache_schema": "salsbury-coordinate-cache-v2",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "source_system_manifest": str(source),
        "source_system_manifest_sha256": sha256_file(source),
        "cached_system_manifest": "system-cache.json",
        "cached_system_manifest_sha256": sha256_file(manifest_path),
        "cached_per_system_manifests": per_system_manifests,
        "coordinate_representation": "continuous_unwrap_unaligned_strided",
        "selection": "molecular_payload",
        "bulk_solvent_included": False,
        "maximum_workers_used": maximum_workers,
        "cache_stride": cache_stride,
        "source_frame_scan": "all source frames decoded in order",
        "materialization_policy": (
            "retain global replica indices 0, stride, 2*stride, ...; do not force "
            "the final frame"
        ),
        "frame_identity_policy": (
            "original system/replica/segment identity plus per-segment first "
            "retained source index and exact integer cache stride preserved"
        ),
        "rows": list(report_rows),
        "limitations": [
            "The cache is a computational representation, not scientific validation.",
            "Water-dependent analyses must use the original solvated trajectories.",
            "Alignment is intentionally deferred to each downstream analysis view.",
            "Every source frame updates the continuous-unwrapping state even when "
            "only a strided subset is materialized.",
        ],
    }
    report_path = temporary / "coordinate-cache-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["coordinate_cache_report_sha256"] = sha256_file(report_path)
    os.replace(temporary, output)
    return report


def _absolute_replica(
    raw_replica: Mapping[str, object], source: Path
) -> Dict[str, object]:
    replica = deepcopy(dict(raw_replica))
    replica["topology"] = str(
        resolve_manifest_path(str(raw_replica["topology"]), source)
    )
    connectivity = raw_replica.get("connectivity")
    if isinstance(connectivity, str):
        replica["connectivity"] = str(resolve_manifest_path(connectivity, source))
    raw_segments = raw_replica.get("segments")
    assert isinstance(raw_segments, list)
    segments = []
    for raw_segment in raw_segments:
        assert isinstance(raw_segment, dict)
        segment = deepcopy(raw_segment)
        segment["trajectory"] = str(
            resolve_manifest_path(str(raw_segment["trajectory"]), source)
        )
        segments.append(segment)
    replica["segments"] = segments
    return replica


def _coordinate_cache_worker(
    arguments: tuple[str, str, bool, float, float, int]
) -> None:
    (
        manifest, output, hash_content, maximum_bond, cycle_tolerance,
        cache_stride,
    ) = arguments
    build_coordinate_cache(
        Path(manifest),
        Path(output),
        hash_source_content=hash_content,
        maximum_bond_length_angstrom=maximum_bond,
        cycle_closure_tolerance_angstrom=cycle_tolerance,
        maximum_workers=1,
        cache_stride=cache_stride,
    )


def _build_coordinate_cache_parallel(
    *,
    source: Path,
    system: Mapping[str, object],
    output: Path,
    hash_source_content: bool,
    maximum_bond_length_angstrom: float,
    cycle_closure_tolerance_angstrom: float,
    maximum_workers: int,
    cache_stride: int,
) -> Dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    worker_root = temporary / ".replica-workers"
    worker_root.mkdir()
    tasks: list[tuple[str, str, bool, float, float, int]] = []
    replica_order: list[tuple[str, str, Path]] = []
    try:
        raw_systems = system["systems"]
        assert isinstance(raw_systems, list)
        worker_index = 0
        for raw_system in raw_systems:
            assert isinstance(raw_system, dict)
            system_id = str(raw_system["system_id"])
            raw_replicas = raw_system["replicas"]
            assert isinstance(raw_replicas, list)
            for raw_replica in raw_replicas:
                assert isinstance(raw_replica, dict)
                replica_id = str(raw_replica["replica_id"])
                manifest_path = worker_root / f"replica-{worker_index:04d}.json"
                worker_output = worker_root / f"replica-{worker_index:04d}-cache"
                manifest = {
                    "systems": [{
                        "system_id": system_id,
                        "metadata": deepcopy(raw_system.get("metadata", {})),
                        "replicas": [_absolute_replica(raw_replica, source)],
                    }]
                }
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                tasks.append((
                    str(manifest_path), str(worker_output), hash_source_content,
                    maximum_bond_length_angstrom,
                    cycle_closure_tolerance_angstrom,
                    cache_stride,
                ))
                replica_order.append((system_id, replica_id, worker_output))
                worker_index += 1
        if len(tasks) < 2:
            raise CoordinateCacheError(
                "parallel cache construction requires at least two replicas"
            )
        with ProcessPoolExecutor(max_workers=min(maximum_workers, len(tasks))) as pool:
            list(pool.map(_coordinate_cache_worker, tasks))

        replicas_by_system: Dict[str, list[Mapping[str, object]]] = {}
        rows_by_system: Dict[str, list[Mapping[str, object]]] = {}
        for system_id, replica_id, worker_output in replica_order:
            worker_manifest = load_json(worker_output / "system-cache.json")
            worker_report = load_json(worker_output / "coordinate-cache-report.json")
            worker_systems = worker_manifest.get("systems")
            worker_rows = worker_report.get("rows")
            if (
                not isinstance(worker_systems, list) or len(worker_systems) != 1
                or not isinstance(worker_systems[0], dict)
                or not isinstance(worker_rows, list) or len(worker_rows) != 1
                or not isinstance(worker_rows[0], dict)
            ):
                raise CoordinateCacheError(
                    f"parallel worker output is incomplete for {system_id}/{replica_id}"
                )
            worker_replicas = worker_systems[0].get("replicas")
            if not isinstance(worker_replicas, list) or len(worker_replicas) != 1:
                raise CoordinateCacheError(
                    f"parallel worker replica is incomplete for {system_id}/{replica_id}"
                )
            replicas_by_system.setdefault(system_id, []).append(worker_replicas[0])
            rows_by_system.setdefault(system_id, []).append(worker_rows[0])
            ignored = {
                "system-cache.json", "coordinate-cache-report.json",
                coordinate_cache_system_manifest_filename(system_id),
            }
            for path in worker_output.iterdir():
                if path.name in ignored:
                    continue
                destination = temporary / path.name
                if destination.exists():
                    raise CoordinateCacheError(
                        f"parallel cache payload name collision: {path.name}"
                    )
                os.replace(path, destination)

        cached_systems = []
        report_rows = []
        for raw_system in raw_systems:
            assert isinstance(raw_system, dict)
            system_id = str(raw_system["system_id"])
            cached_systems.append({
                "system_id": system_id,
                "metadata": {
                    **deepcopy(raw_system.get("metadata", {})),
                    "coordinate_cache": "continuous_unwrap_strided_molecular_payload_v2",
                    "source_system_manifest": str(source),
                    "source_frame_scan": "all_frames_continuous_unwrap",
                    "cache_stride": cache_stride,
                },
                "replicas": replicas_by_system[system_id],
            })
            report_rows.extend(rows_by_system[system_id])
        shutil.rmtree(worker_root)
        return _write_cache_metadata(
            temporary,
            output,
            source=source,
            cached_systems=cached_systems,
            report_rows=report_rows,
            maximum_workers=min(maximum_workers, len(tasks)),
            cache_stride=cache_stride,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_coordinate_cache(
    system_manifest_path: Path,
    output_directory: Path,
    *,
    hash_source_content: bool = False,
    maximum_bond_length_angstrom: float = 4.0,
    cycle_closure_tolerance_angstrom: float = 0.05,
    maximum_workers: int = 1,
    cache_stride: int = 1,
) -> Dict[str, object]:
    """Write an atomic continuously unwrapped, strided working cache.

    Every source frame is decoded and passed through the stateful periodic
    reconstruction.  ``cache_stride`` controls only which reconstructed frames
    are materialized, so a coarse working cache never breaks continuity across
    a periodic boundary.
    """

    source = Path(system_manifest_path).expanduser().resolve(strict=True)
    system = load_json(source)
    if not isinstance(system, dict):
        raise CoordinateCacheError("system manifest must be a JSON object")
    try:
        validate_system(system, source_path=source, check_paths=True)
    except ManifestValidationError as exc:
        raise CoordinateCacheError(str(exc)) from exc
    if (
        isinstance(maximum_workers, bool)
        or not isinstance(maximum_workers, int)
        or maximum_workers <= 0
    ):
        raise CoordinateCacheError("maximum_workers must be a positive integer")
    if (
        isinstance(cache_stride, bool)
        or not isinstance(cache_stride, int)
        or cache_stride <= 0
    ):
        raise CoordinateCacheError("cache_stride must be a positive integer")
    output = Path(output_directory).expanduser().resolve(strict=False)
    if output.exists():
        raise CoordinateCacheError(f"cache output already exists: {output}")
    replica_count = sum(
        len(raw_system["replicas"])
        for raw_system in system["systems"]
        if isinstance(raw_system, dict)
    )
    if maximum_workers > 1 and replica_count > 1:
        return _build_coordinate_cache_parallel(
            source=source,
            system=system,
            output=output,
            hash_source_content=hash_source_content,
            maximum_bond_length_angstrom=maximum_bond_length_angstrom,
            cycle_closure_tolerance_angstrom=cycle_closure_tolerance_angstrom,
            maximum_workers=maximum_workers,
            cache_stride=cache_stride,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    project = {
        "periodic_coordinate_policy": "unwrap_continuous",
        "periodic_reconstruction": {
            "maximum_bond_length_angstrom": maximum_bond_length_angstrom,
            "cycle_closure_tolerance_angstrom": cycle_closure_tolerance_angstrom,
            "maximum_anchor_displacement_angstrom": 100.0,
        },
    }
    report_rows = []
    cached_systems = []
    try:
        for raw_system in system["systems"]:
            assert isinstance(raw_system, dict)
            system_id = str(raw_system["system_id"])
            cached_replicas = []
            for raw_replica in raw_system["replicas"]:
                assert isinstance(raw_replica, dict)
                replica_id = str(raw_replica["replica_id"])
                topology = resolve_manifest_path(str(raw_replica["topology"]), source)
                if topology.suffix.lower() not in {".pdb", ".ent"}:
                    raise CoordinateCacheError(
                        "coordinate caching currently requires PDB replica topologies"
                    )
                connectivity_value = raw_replica.get("connectivity")
                if not isinstance(connectivity_value, str):
                    raise CoordinateCacheError(
                        f"{system_id}/{replica_id} requires explicit connectivity"
                    )
                connectivity = resolve_manifest_path(connectivity_value, source)
                try:
                    atoms = read_pdb_atoms(topology)
                    selected_atoms = select_atoms(
                        atoms, {"preset": "molecular_payload"}, "molecular_payload"
                    )
                except AtomMappingError as exc:
                    raise CoordinateCacheError(str(exc)) from exc
                atom_indices = tuple(atom.atom_index for atom in selected_atoms)
                processor = PeriodicFrameProcessor.from_replica(
                    project, raw_replica, source, len(atoms)
                )
                try:
                    raw_reference = next(iter_coordinate_frames(topology, "angstrom"))
                except StopIteration as exc:
                    raise CoordinateCacheError("topology contains no reference frame") from exc
                reference = processor.process(
                    raw_reference, f"{system_id}/{replica_id}/reference", atom_indices
                )
                prefix = coordinate_cache_prefix(system_id, replica_id)
                cached_topology = temporary / f"{prefix}.pdb"
                cached_connectivity = temporary / f"{prefix}.bonds.json"
                _write_subset_pdb(cached_topology, topology, atom_indices, reference)
                _write_subset_connectivity(
                    cached_connectivity, connectivity, len(atoms), atom_indices
                )
                cached_segments = []
                segment_reports = []
                replica_source_offset = 0
                for segment_index, segment in enumerate(raw_replica["segments"]):
                    assert isinstance(segment, dict)
                    segment_id = str(segment["segment_id"])
                    trajectory = resolve_manifest_path(str(segment["trajectory"]), source)
                    try:
                        probe = probe_trajectory(trajectory)
                    except (FileProbeError, OSError) as exc:
                        raise CoordinateCacheError(str(exc)) from exc
                    declared = int(probe.get("declared_frame_count", 0))
                    if declared <= 0:
                        raise CoordinateCacheError(
                            f"{system_id}/{replica_id}/{segment_id} has no declared frames"
                        )
                    processor.begin_segment(
                        bool(segment.get("continuous_with_previous", False))
                    )
                    first_selected_local = (-replica_source_offset) % cache_stride
                    retained = (
                        0 if first_selected_local >= declared else
                        (declared - 1 - first_selected_local) // cache_stride + 1
                    )
                    cached_trajectory = (
                        temporary / f"{prefix}-segment-{segment_index:02d}.dcd"
                    )
                    writer = None
                    decoded = 0
                    try:
                        for local_index, raw_frame in enumerate(
                            iter_coordinate_frames(trajectory, "angstrom")
                        ):
                            frame = processor.process(
                                raw_frame,
                                f"{system_id}/{replica_id}/{segment_id}/"
                                f"frame-{raw_frame.frame_index}",
                                atom_indices,
                            )
                            decoded += 1
                            if (
                                (replica_source_offset + local_index)
                                % cache_stride != 0
                            ):
                                continue
                            if writer is None:
                                source_interval = int(
                                    probe.get("save_interval_steps", 1)
                                )
                                writer = _DCDWriter(
                                    cached_trajectory,
                                    atom_count=len(atom_indices),
                                    frame_count=retained,
                                    starting_step=(
                                        int(probe.get("starting_step", 0))
                                        + local_index * source_interval
                                    ),
                                    save_interval=source_interval * cache_stride,
                                    unit_cell_present=frame.periodic_cell_present,
                                )
                            writer.write(frame, atom_indices)
                        if decoded != declared:
                            raise CoordinateCacheError(
                                f"{system_id}/{replica_id}/{segment_id} decoded "
                                f"{decoded} frames; header declared {declared}"
                            )
                        if retained > 0:
                            if writer is None:
                                raise CoordinateCacheError(
                                    "cache retained-frame accounting is inconsistent"
                                )
                            writer.close()
                    except Exception:
                        if writer is not None and not writer.handle.closed:
                            writer.handle.close()
                        raise
                    replica_source_offset += declared
                    if retained == 0:
                        segment_reports.append({
                            "segment_id": segment_id,
                            "source_frame_count": declared,
                            "decoded_frame_count": decoded,
                            "retained_frame_count": 0,
                            "first_retained_source_frame_index": None,
                            "cache_stride": cache_stride,
                            "source": _source_identity(
                                trajectory, hash_source_content
                            ),
                            "cache": None,
                        })
                        continue
                    cached_segment = {
                        "segment_id": segment_id,
                        "trajectory": cached_trajectory.name,
                    }
                    if "timing" in segment:
                        timing = deepcopy(segment["timing"])
                        timing["first_frame_time"] = (
                            float(timing["first_frame_time"])
                            + first_selected_local * float(timing["frame_interval"])
                        )
                        timing["frame_interval"] = (
                            float(timing["frame_interval"]) * cache_stride
                        )
                        cached_segment["timing"] = timing
                    elif "sample_axis" in segment:
                        sample_axis = deepcopy(segment["sample_axis"])
                        sample_axis["first_sample_index"] = (
                            int(sample_axis["first_sample_index"])
                            + first_selected_local
                            * int(sample_axis["sample_interval"])
                        )
                        sample_axis["sample_interval"] = (
                            int(sample_axis["sample_interval"]) * cache_stride
                        )
                        cached_segment["sample_axis"] = sample_axis
                    if "continuous_with_previous" in segment:
                        cached_segment["continuous_with_previous"] = segment[
                            "continuous_with_previous"
                        ]
                    cached_segments.append(cached_segment)
                    segment_reports.append({
                        "segment_id": segment_id,
                        "source_frame_count": declared,
                        "decoded_frame_count": decoded,
                        "retained_frame_count": retained,
                        "frame_count": retained,
                        "first_retained_source_frame_index": (
                            first_selected_local
                        ),
                        "cache_stride": cache_stride,
                        "source": _source_identity(trajectory, hash_source_content),
                        "cache": {
                            "path": cached_trajectory.name,
                            "size_bytes": cached_trajectory.stat().st_size,
                            "sha256": sha256_file(cached_trajectory),
                        },
                    })
                cached_replicas.append({
                    "replica_id": replica_id,
                    "topology": cached_topology.name,
                    "connectivity": cached_connectivity.name,
                    "segments": cached_segments,
                })
                report_rows.append({
                    "system_id": system_id,
                    "replica_id": replica_id,
                    "source_atom_count": len(atoms),
                    "cached_atom_count": len(atom_indices),
                    "source_atom_indices_in_cache_order": list(atom_indices),
                    "selection": "molecular_payload",
                    "cache_stride": cache_stride,
                    "source_frame_count": sum(
                        int(row["source_frame_count"])
                        for row in segment_reports
                    ),
                    "decoded_frame_count": sum(
                        int(row["decoded_frame_count"])
                        for row in segment_reports
                    ),
                    "retained_frame_count": sum(
                        int(row["retained_frame_count"])
                        for row in segment_reports
                    ),
                    "source_topology": _source_identity(
                        topology, hash_source_content
                    ),
                    "source_connectivity": _source_identity(
                        connectivity, hash_source_content
                    ),
                    "topology_sha256": sha256_file(cached_topology),
                    "connectivity_sha256": sha256_file(cached_connectivity),
                    "periodic_reconstruction": processor.report(),
                    "segments": segment_reports,
                })
            cached_systems.append({
                "system_id": system_id,
                "metadata": {
                    **deepcopy(raw_system.get("metadata", {})),
                    "coordinate_cache": "continuous_unwrap_strided_molecular_payload_v2",
                    "source_system_manifest": str(source),
                    "source_frame_scan": "all_frames_continuous_unwrap",
                    "cache_stride": cache_stride,
                },
                "replicas": cached_replicas,
            })
        return _write_cache_metadata(
            temporary,
            output,
            source=source,
            cached_systems=cached_systems,
            report_rows=report_rows,
            maximum_workers=1,
            cache_stride=cache_stride,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_coordinate_cache_safe(
    system_manifest_path: Path,
    output_directory: Path,
    *,
    hash_source_content: bool = False,
    maximum_workers: int = 1,
    cache_stride: int = 1,
) -> Dict[str, object]:
    try:
        return build_coordinate_cache(
            system_manifest_path,
            output_directory,
            hash_source_content=hash_source_content,
            maximum_workers=maximum_workers,
            cache_stride=cache_stride,
        )
    except (
        CoordinateCacheError, ManifestValidationError, PeriodicReconstructionError,
        AtomMappingError, FileProbeError, OSError, ValueError,
    ) as exc:
        return {
            "cache_schema": "salsbury-coordinate-cache-v2",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "issues": [{
                "severity": "error",
                "code": "COORDINATE_CACHE_FAILED",
                "message": str(exc),
            }],
        }


def validate_reusable_coordinate_cache(
    cache_directory: Path, source_system_manifest: Path
) -> Dict[str, object]:
    """Validate a reusable all-frame-scanned cache against source inputs.

    Cache reuse does not require materializing every decoded frame.  It does
    require a positive declared integer stride, an in-order scan of every raw
    frame for continuous reconstruction, exact source identities, and the
    deterministic retained-frame count implied by that stride.
    """

    root = Path(cache_directory).expanduser().resolve(strict=True)
    source = Path(source_system_manifest).expanduser().resolve(strict=True)
    report_path = root / "coordinate-cache-report.json"
    manifest_path = root / "system-cache.json"
    report = load_json(report_path)
    cached = load_json(manifest_path)
    original = load_json(source)
    if not isinstance(report, dict) or report.get("technical_status") != "complete":
        raise CoordinateCacheError("reusable coordinate cache report is incomplete")
    cache_stride = report.get("cache_stride")
    if (
        isinstance(cache_stride, bool)
        or not isinstance(cache_stride, int)
        or cache_stride <= 0
    ):
        raise CoordinateCacheError(
            "a reusable coordinate cache must declare a positive integer stride"
        )
    if report.get("source_frame_scan") != "all source frames decoded in order":
        raise CoordinateCacheError(
            "reusable coordinate cache lacks an all-frame continuity scan"
        )
    if not isinstance(cached, dict) or not isinstance(original, dict):
        raise CoordinateCacheError("source or cached system manifest is invalid")
    validate_system(cached, source_path=manifest_path, check_paths=True)
    validate_system(original, source_path=source, check_paths=True)
    report_rows = report.get("rows")
    if not isinstance(report_rows, list):
        raise CoordinateCacheError("reusable coordinate cache has no replica rows")
    rows = {
        (str(row.get("system_id")), str(row.get("replica_id"))): row
        for row in report_rows if isinstance(row, dict)
    }
    expected_keys = set()
    for system in original["systems"]:
        assert isinstance(system, dict)
        for replica in system["replicas"]:
            assert isinstance(replica, dict)
            key = (str(system["system_id"]), str(replica["replica_id"]))
            expected_keys.add(key)
            row = rows.get(key)
            if not isinstance(row, dict):
                raise CoordinateCacheError(
                    f"reusable cache is missing {key[0]}/{key[1]}"
                )
            topology = resolve_manifest_path(str(replica["topology"]), source)
            _validate_source_identity(
                row.get("source_topology"), topology, "topology"
            )
            connectivity_value = replica.get("connectivity")
            if not isinstance(connectivity_value, str):
                raise CoordinateCacheError(
                    f"reusable cache source {key[0]}/{key[1]} lacks connectivity"
                )
            connectivity = resolve_manifest_path(connectivity_value, source)
            _validate_source_identity(
                row.get("source_connectivity"), connectivity, "connectivity"
            )
            cached_segments = row.get("segments")
            source_segments = replica.get("segments")
            if (
                not isinstance(cached_segments, list)
                or not isinstance(source_segments, list)
                or len(cached_segments) != len(source_segments)
            ):
                raise CoordinateCacheError(
                    f"reusable cache segment count changed for {key[0]}/{key[1]}"
                )
            for cached_segment, source_segment in zip(
                cached_segments, source_segments
            ):
                assert isinstance(cached_segment, dict)
                assert isinstance(source_segment, dict)
                path = resolve_manifest_path(
                    str(source_segment["trajectory"]), source
                )
                _validate_source_identity(
                    cached_segment.get("source"), path, "trajectory"
                )
                decoded = int(cached_segment.get("decoded_frame_count", -1))
                retained = int(cached_segment.get("retained_frame_count", -1))
                if decoded <= 0 or retained != integer_stride_selected_count(
                    decoded, cache_stride
                ):
                    raise CoordinateCacheError(
                        "reusable cache retained-frame count is inconsistent with "
                        "its declared integer stride"
                    )
    if set(rows) != expected_keys:
        raise CoordinateCacheError(
            "reusable coordinate cache system/replica identities do not match"
        )
    return {
        "technical_status": "complete",
        "cache_directory": str(root),
        "cache_report": str(report_path),
        "cache_report_sha256": sha256_file(report_path),
        "cached_system_manifest": str(manifest_path),
        "cached_system_manifest_sha256": sha256_file(manifest_path),
        "source_system_manifest": str(source),
        "source_system_manifest_sha256": sha256_file(source),
        "cache_stride": cache_stride,
        "replica_count": len(rows),
        "reuse_boundary": (
            "Every raw frame was scanned in order for continuous reconstruction. "
            f"Every {cache_stride:g} frame was materialized for conformational "
            "and other non-bulk-solvent analyses; "
            "water- and solvent-dependent modules continue to read the original "
            "solvated trajectories."
        ),
    }
