"""Initial coordinate-level structural-integrity gates for declared trajectories."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .context import compile_project_context_file
from .atom_mapping import AtomMappingError, read_topology_atoms
from .coordinates import (
    CoordinateFrame,
    CoordinateReadError,
    coordinate_format,
    finite_coordinate_count,
    iter_coordinate_frames,
)
from .coordinate_cache import (
    CoordinateCacheError,
    validate_reusable_coordinate_cache,
)
from .frame_sampling import (
    frame_selected,
    normalize_frame_selection,
    plan_frame_selection,
    reader_frame_indices,
    source_frame_count,
)
from .manifests import (
    ManifestValidationError,
    load_json,
    resolve_manifest_path,
)
from .preflight import FileProbeError, probe_topology
from .periodic import (
    PeriodicFrameProcessor,
    PeriodicReconstructionError,
    load_connectivity,
)
from .reporting import issue_record
from .replica_execution import ReplicaShard, execute_replica_workers
from .selections import select_atoms
from .structural_chemistry import (
    StructuralChemistryError,
    chemical_integrity_snapshot,
    reference_chirality_signs,
)
from .trajectory_contracts import frame_time, normalize_segment_timing


_REQUIRED_THRESHOLDS = {
    "near_coincident_distance_angstrom",
    "maximum_near_coincident_pairs_per_frame",
    "maximum_absolute_coordinate_angstrom",
    "frame_stride",
}
_ALLOWED_THRESHOLDS = _REQUIRED_THRESHOLDS | {
    "maximum_frame_atom_displacement_angstrom",
    "frame_displacement_selection",
    "chemical_integrity", "frame_selection", "checkpointing",
    "parallel_execution",
}
class StructuralQCError(ValueError):
    """Raised when structural-QC configuration cannot be interpreted safely."""


_QC_FINDING_CODES = {
    "COORDINATE_EXTENT_EXCEEDED",
    "FRAME_DISPLACEMENT_EXCEEDED",
    "NEAR_COINCIDENT_PAIR_THRESHOLD_EXCEEDED",
    "PEPTIDE_CONTINUITY_OUTLIER",
    "OMEGA_GEOMETRY_OUTLIER",
    "CA_CHIRALITY_OUTLIER",
    "DECLARED_COVALENT_LINK_OUTLIER",
    "STERIC_CLASH_THRESHOLD_EXCEEDED",
    "CHEMICAL_CONNECTIVITY_UNAVAILABLE",
}


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise StructuralQCError(f"{label} must be a positive number")
    return float(value)


def _thresholds(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise StructuralQCError(
            "project definitions.structural_qc is required for explicit QC gates"
        )
    raw = definitions.get("structural_qc")
    if not isinstance(raw, dict):
        raise StructuralQCError(
            "project definitions.structural_qc must be an object"
        )
    unknown = sorted(set(raw).difference(_ALLOWED_THRESHOLDS))
    if unknown:
        raise StructuralQCError(
            "definitions.structural_qc contains unknown fields: " + ", ".join(unknown)
        )
    missing = sorted(_REQUIRED_THRESHOLDS.difference(raw))
    if missing:
        raise StructuralQCError(
            "definitions.structural_qc is missing required fields: " + ", ".join(missing)
        )
    maximum_pairs = raw["maximum_near_coincident_pairs_per_frame"]
    if (
        isinstance(maximum_pairs, bool)
        or not isinstance(maximum_pairs, int)
        or maximum_pairs < 0
    ):
        raise StructuralQCError(
            "maximum_near_coincident_pairs_per_frame must be a nonnegative integer"
        )
    frame_stride = raw["frame_stride"]
    if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride <= 0:
        raise StructuralQCError("frame_stride must be a positive integer")
    displacement = raw.get("maximum_frame_atom_displacement_angstrom")
    if displacement is not None:
        displacement = _positive_number(
            displacement, "maximum_frame_atom_displacement_angstrom"
        )
    displacement_selection = raw.get("frame_displacement_selection")
    if displacement_selection is not None and (
        not isinstance(displacement_selection, str)
        or not displacement_selection.strip()
    ):
        raise StructuralQCError(
            "frame_displacement_selection must be a nonempty named selection"
        )
    chemical = raw.get("chemical_integrity")
    if chemical is not None:
        required_chemical = {
            "maximum_peptide_bond_angstrom",
            "maximum_trans_omega_deviation_degrees",
            "minimum_ca_chirality_volume_angstrom3",
            "steric_clash_scale",
            "maximum_steric_clashes_per_frame",
            "allow_cis_proline",
            "declared_covalent_links",
        }
        if not isinstance(chemical, dict) or set(chemical) != required_chemical:
            raise StructuralQCError(
                "chemical_integrity must contain exactly: " + ", ".join(sorted(required_chemical))
            )
        maximum_clashes = chemical["maximum_steric_clashes_per_frame"]
        if isinstance(maximum_clashes, bool) or not isinstance(maximum_clashes, int) or maximum_clashes < 0:
            raise StructuralQCError("maximum_steric_clashes_per_frame must be a nonnegative integer")
        if not isinstance(chemical["allow_cis_proline"], bool):
            raise StructuralQCError("allow_cis_proline must be boolean")
        links = chemical["declared_covalent_links"]
        if not isinstance(links, list):
            raise StructuralQCError("declared_covalent_links must be an array")
        normalized_links = []
        for link in links:
            fields = {"link_id", "atom_indices", "minimum_distance_angstrom", "maximum_distance_angstrom"}
            if not isinstance(link, dict) or set(link) != fields:
                raise StructuralQCError("each declared covalent link must contain link_id, atom_indices, and distance bounds")
            indices = link["atom_indices"]
            if not isinstance(indices, list) or len(indices) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in indices):
                raise StructuralQCError("covalent-link atom_indices must be two nonnegative integers")
            minimum = _positive_number(link["minimum_distance_angstrom"], "minimum_distance_angstrom")
            maximum = _positive_number(link["maximum_distance_angstrom"], "maximum_distance_angstrom")
            if minimum >= maximum:
                raise StructuralQCError("covalent-link minimum distance must be below maximum distance")
            normalized_links.append({**link, "minimum_distance_angstrom": minimum, "maximum_distance_angstrom": maximum})
        chemical = {
            "maximum_peptide_bond_angstrom": _positive_number(chemical["maximum_peptide_bond_angstrom"], "maximum_peptide_bond_angstrom"),
            "maximum_trans_omega_deviation_degrees": _positive_number(chemical["maximum_trans_omega_deviation_degrees"], "maximum_trans_omega_deviation_degrees"),
            "minimum_ca_chirality_volume_angstrom3": _positive_number(chemical["minimum_ca_chirality_volume_angstrom3"], "minimum_ca_chirality_volume_angstrom3"),
            "steric_clash_scale": _positive_number(chemical["steric_clash_scale"], "steric_clash_scale"),
            "maximum_steric_clashes_per_frame": maximum_clashes,
            "allow_cis_proline": chemical["allow_cis_proline"],
            "declared_covalent_links": normalized_links,
        }
    checkpointing = raw.get("checkpointing", {
        "enabled": True,
        "within_segment_interval_seconds": 7200.0,
    })
    if not isinstance(checkpointing, dict) or set(checkpointing) != {
        "enabled", "within_segment_interval_seconds"
    }:
        raise StructuralQCError(
            "checkpointing must contain exactly enabled and "
            "within_segment_interval_seconds"
        )
    if not isinstance(checkpointing["enabled"], bool):
        raise StructuralQCError("checkpointing.enabled must be boolean")
    checkpoint_interval = _positive_number(
        checkpointing["within_segment_interval_seconds"],
        "checkpointing.within_segment_interval_seconds",
    )
    parallel = raw.get("parallel_execution")
    if parallel is not None:
        required_parallel = {
            "enabled", "maximum_workers", "coordinate_cache_system_manifest",
            "coordinate_cache_report",
        }
        if not isinstance(parallel, dict) or set(parallel) != required_parallel:
            raise StructuralQCError(
                "parallel_execution must contain exactly enabled, maximum_workers, "
                "coordinate_cache_system_manifest, and coordinate_cache_report"
            )
        if not isinstance(parallel["enabled"], bool):
            raise StructuralQCError("parallel_execution.enabled must be boolean")
        workers = parallel["maximum_workers"]
        if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
            raise StructuralQCError(
                "parallel_execution.maximum_workers must be a positive integer"
            )
        for field in (
            "coordinate_cache_system_manifest", "coordinate_cache_report",
        ):
            value = parallel[field]
            if not isinstance(value, str) or not value.strip():
                raise StructuralQCError(
                    f"parallel_execution.{field} must be a nonempty path"
                )
        parallel = {
            "enabled": parallel["enabled"],
            "maximum_workers": workers,
            "coordinate_cache_system_manifest": parallel[
                "coordinate_cache_system_manifest"
            ],
            "coordinate_cache_report": parallel["coordinate_cache_report"],
        }
    return {
        "near_coincident_distance_angstrom": _positive_number(
            raw["near_coincident_distance_angstrom"],
            "near_coincident_distance_angstrom",
        ),
        "maximum_near_coincident_pairs_per_frame": maximum_pairs,
        "maximum_absolute_coordinate_angstrom": _positive_number(
            raw["maximum_absolute_coordinate_angstrom"],
            "maximum_absolute_coordinate_angstrom",
        ),
        "maximum_frame_atom_displacement_angstrom": displacement,
        "frame_displacement_selection": displacement_selection,
        "frame_stride": frame_stride,
        "frame_selection": normalize_frame_selection(
            raw.get("frame_selection"),
            frame_stride,
            error_type=StructuralQCError,
        ),
        "chemical_integrity": chemical,
        "checkpointing": {
            "enabled": checkpointing["enabled"],
            "within_segment_interval_seconds": checkpoint_interval,
            "segment_completion_checkpoints": True,
        },
        "parallel_execution": parallel,
    }


def _checkpoint_payload_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _package_source_signature_sha256() -> str:
    """Conservatively invalidate checkpoints after any package-code change."""

    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    try:
        for source in sorted(package_root.glob("*.py"), key=lambda path: path.name):
            digest.update(source.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.read_bytes())
            digest.update(b"\0")
    except OSError as exc:
        raise StructuralQCError(
            f"structural-QC package source identity could not be read: {exc}"
        ) from exc
    return digest.hexdigest()


def _input_inventory_signature_sha256(inventory: Mapping[str, object]) -> str:
    """Identify the declared files without forcing a second full trajectory read."""

    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise StructuralQCError("input inventory entries are unavailable")
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise StructuralQCError("input inventory contains an invalid entry")
        normalized.append({
            "role": entry.get("role"),
            "system_id": entry.get("system_id"),
            "replica_id": entry.get("replica_id"),
            "segment_id": entry.get("segment_id"),
            "resolved_path": entry.get("resolved_path"),
            "size_bytes": entry.get("size_bytes"),
            "modified_utc": entry.get("modified_utc"),
            "sha256": entry.get("sha256"),
        })
    return hashlib.sha256(
        _checkpoint_payload_bytes({"entries": normalized})
    ).hexdigest()


def _checkpoint_path(
    checkpoint_root: Path, system_id: str, replica_id: str, segment_id: str,
) -> Path:
    identity = hashlib.sha256(
        (system_id + "\0" + replica_id + "\0" + segment_id).encode("utf-8")
    ).hexdigest()[:20]
    return checkpoint_root / f"segment-{identity}.json.gz"


def _write_structural_qc_checkpoint(
    path: Path, identity: str, payload: Mapping[str, object],
) -> None:
    body = dict(payload)
    encoded = _checkpoint_payload_bytes(body)
    wrapper = {
        "schema": "structural_qc_checkpoint_v1",
        "checkpoint_identity": identity,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "payload": body,
    }
    serialized = _checkpoint_payload_bytes(wrapper)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                compressed.write(serialized)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _load_structural_qc_checkpoint(
    path: Path, identity: str,
) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as handle:
            wrapper = json.loads(handle.read().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StructuralQCError(f"structural-QC checkpoint is unreadable: {exc}") from exc
    if (
        not isinstance(wrapper, dict)
        or wrapper.get("schema") != "structural_qc_checkpoint_v1"
        or wrapper.get("checkpoint_identity") != identity
        or not isinstance(wrapper.get("payload"), dict)
    ):
        raise StructuralQCError(
            "structural-QC checkpoint identity or schema does not match"
        )
    payload = wrapper["payload"]
    encoded = _checkpoint_payload_bytes(payload)
    if wrapper.get("payload_sha256") != hashlib.sha256(encoded).hexdigest():
        raise StructuralQCError("structural-QC checkpoint payload hash does not match")
    return payload


def _near_coincident_pairs(
    frame: CoordinateFrame, threshold: float, example_limit: int = 10
) -> Tuple[int, Optional[float], List[Dict[str, object]]]:
    coordinates = np.asarray(frame.coordinates_angstrom, dtype=np.float64)
    finite_mask = np.isfinite(coordinates).all(axis=1)
    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size < 2:
        return 0, None, []
    finite_coordinates = coordinates[finite_indices]
    local_pairs = cKDTree(finite_coordinates).query_pairs(
        r=threshold, output_type="ndarray"
    )
    if local_pairs.size == 0:
        return 0, None, []
    atom_pairs = finite_indices[local_pairs]
    order = np.lexsort((atom_pairs[:, 1], atom_pairs[:, 0]))
    atom_pairs = atom_pairs[order]
    displacements = coordinates[atom_pairs[:, 0]] - coordinates[atom_pairs[:, 1]]
    distances = np.linalg.norm(displacements, axis=1)
    examples = [
        {
            "atom_index_1": int(left),
            "atom_index_2": int(right),
            "distance_angstrom": float(distance),
        }
        for (left, right), distance in zip(
            atom_pairs[:example_limit], distances[:example_limit]
        )
    ]
    return int(atom_pairs.shape[0]), float(distances.min()), examples


def _maximum_displacement(
    previous: Sequence[Tuple[float, float, float]],
    current: Sequence[Tuple[float, float, float]],
) -> Optional[float]:
    if len(previous) != len(current):
        return None
    maximum = 0.0
    saw_finite = False
    for before, after in zip(previous, current):
        values = before + after
        if not all(math.isfinite(value) for value in values):
            continue
        displacement = math.sqrt(sum((after[i] - before[i]) ** 2 for i in range(3)))
        maximum = max(maximum, displacement)
        saw_finite = True
    return maximum if saw_finite else None


def _maximum_rigid_body_aligned_displacement(
    previous: Sequence[Tuple[float, float, float]],
    current: Sequence[Tuple[float, float, float]],
) -> Optional[float]:
    """Return the largest finite-atom residual after a proper Kabsch fit.

    A whole reconstructed component may translate or rotate without any
    internal coordinate discontinuity.  The structural-integrity gate must
    therefore remove one global proper rigid-body transform before testing
    per-atom motion.  Tiny fixtures that cannot define a 3D fit retain the
    historical raw-displacement behavior.
    """

    if len(previous) != len(current):
        return None
    before = np.asarray(previous, dtype=np.float64)
    after = np.asarray(current, dtype=np.float64)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 3:
        return None
    finite = np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
    if int(np.count_nonzero(finite)) < 3:
        return _maximum_displacement(previous, current)
    left = after[finite]
    right = before[finite]
    left_center = left.mean(axis=0)
    right_center = right.mean(axis=0)
    left_centered = left - left_center
    right_centered = right - right_center
    covariance = left_centered.T @ right_centered
    u, _, vt = np.linalg.svd(covariance, full_matrices=False)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    aligned = left_centered @ rotation + right_center
    return float(np.linalg.norm(aligned - right, axis=1).max())


def _structural_qc_project_serial(
    project_path: Path, hash_content: bool = False, *,
    only_system_id: Optional[str] = None,
    only_replica_id: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate one project serially, optionally scoped to exactly one replica."""

    project_source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(project_source)
    thresholds = _thresholds(project)
    context = compile_project_context_file(project_source, hash_content=hash_content)
    contract = context["contract"]
    assert isinstance(contract, dict)
    units = contract["units"]
    assert isinstance(units, dict)
    declared_coordinate_unit = str(units["coordinates"])
    time_unit = str(units["time"])
    system_path = Path(str(context["system_manifest_path"]))
    system_manifest = load_json(system_path)
    if only_system_id is not None or only_replica_id is not None:
        if only_system_id is None or only_replica_id is None:
            raise StructuralQCError(
                "structural-QC replica scoping requires both system and replica IDs"
            )
        selected_systems = []
        for raw_system in system_manifest.get("systems", []):
            if not isinstance(raw_system, dict) or str(
                raw_system.get("system_id")
            ) != only_system_id:
                continue
            selected_replicas = [
                deepcopy(replica)
                for replica in raw_system.get("replicas", [])
                if isinstance(replica, dict)
                and str(replica.get("replica_id")) == only_replica_id
            ]
            if selected_replicas:
                selected_systems.append({
                    **deepcopy(raw_system), "replicas": selected_replicas,
                })
        if len(selected_systems) != 1:
            raise StructuralQCError(
                f"structural-QC cache lacks exactly one replica "
                f"{only_system_id}/{only_replica_id}"
            )
        system_manifest = {"systems": selected_systems}
    checkpointing = thresholds["checkpointing"]
    assert isinstance(checkpointing, dict)
    inventory = context["input_inventory"]
    assert isinstance(inventory, dict)
    checkpoint_identity = hashlib.sha256(
        _checkpoint_payload_bytes({
            "project_manifest_sha256": context["project_manifest_sha256"],
            "system_manifest_sha256": context["system_manifest_sha256"],
            "input_content_signature_sha256": context[
                "input_content_signature_sha256"
            ],
            "input_inventory_signature_sha256": (
                _input_inventory_signature_sha256(inventory)
            ),
            "package_source_signature_sha256": (
                _package_source_signature_sha256()
            ),
        })
    ).hexdigest()
    checkpoint_root: Optional[Path] = None
    if bool(checkpointing["enabled"]):
        output_value = project.get("analysis_output_root")
        if not isinstance(output_value, str) or not output_value.strip():
            raise StructuralQCError(
                "analysis_output_root is required when structural-QC checkpointing is enabled"
            )
        checkpoint_root = (
            resolve_manifest_path(output_value, project_source)
            / "structural-qc" / "checkpoints" / checkpoint_identity
        )
    frame_selection_plan, frame_selection_report = plan_frame_selection(
        system_manifest,
        system_path,
        declared_coordinate_unit,
        thresholds["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(thresholds["frame_stride"]),
        error_type=StructuralQCError,
    )

    inventory_entries = inventory["entries"]
    assert isinstance(inventory_entries, list)
    inventory_by_path = {
        str(entry["resolved_path"]): entry
        for entry in inventory_entries
        if isinstance(entry, dict)
    }

    issues: List[Dict[str, object]] = list(context["issues"])
    system_reports: List[Dict[str, object]] = []
    systems = system_manifest["systems"]
    assert isinstance(systems, list)
    for system in systems:
        assert isinstance(system, dict)
        system_id = str(system["system_id"])
        replica_reports: List[Dict[str, object]] = []
        replicas = system["replicas"]
        assert isinstance(replicas, list)
        for replica in replicas:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            replica_location = f"{system_id}/{replica_id}"
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            try:
                topology = probe_topology(topology_path)
                topology_atom_count = int(topology["atom_count"])
            except FileProbeError as exc:
                topology = None
                topology_atom_count = -1
                issues.append(issue_record(
                    "error", "TOPOLOGY_PROBE_FAILED", replica_location, str(exc)
                ))
            chemical_atoms = None
            chirality_reference = None
            chemical_bonds = None
            chemical_connectivity = None
            displacement_indices = None
            need_topology_atoms = (
                thresholds["chemical_integrity"] is not None
                or thresholds["frame_displacement_selection"] is not None
            )
            if topology_atom_count >= 0 and need_topology_atoms:
                try:
                    _, topology_atoms = read_topology_atoms(topology_path)
                    chemical_atoms = topology_atoms
                    if thresholds["chemical_integrity"] is not None:
                        topology_frame = next(iter_coordinate_frames(topology_path, "angstrom"))
                        chirality_reference = reference_chirality_signs(
                            topology_atoms, topology_frame.coordinates_angstrom
                        )
                        connectivity_value = replica.get("connectivity")
                        connectivity_path = (
                            resolve_manifest_path(connectivity_value, system_path)
                            if isinstance(connectivity_value, str)
                            and connectivity_value.strip()
                            else (
                                topology_path
                                if topology_path.suffix.lower()
                                in {".psf", ".prmtop", ".parm7", ".json"}
                                else None
                            )
                        )
                        if connectivity_path is None:
                            issues.append(issue_record(
                                "warning",
                                "CHEMICAL_CONNECTIVITY_UNAVAILABLE",
                                replica_location,
                                "explicit covalent connectivity is unavailable; peptide-link, omega, and topology-aware steric checks are not evaluated",
                            ))
                        else:
                            try:
                                chemical_bonds, chemical_connectivity = load_connectivity(
                                    connectivity_path, topology_atom_count
                                )
                            except PeriodicReconstructionError as exc:
                                issues.append(issue_record(
                                    "warning",
                                    "CHEMICAL_CONNECTIVITY_UNAVAILABLE",
                                    replica_location,
                                    f"explicit covalent connectivity could not be loaded; peptide-link, omega, and topology-aware steric checks are not evaluated: {exc}",
                                ))
                    selection_id = thresholds["frame_displacement_selection"]
                    if selection_id is not None:
                        definitions = project.get("selections")
                        if not isinstance(definitions, dict):
                            raise AtomMappingError(
                                "project selections are required for frame displacement"
                            )
                        definition = definitions.get(selection_id)
                        if not isinstance(definition, dict):
                            raise AtomMappingError(
                                f"frame displacement selection {selection_id!r} is not defined"
                            )
                        displacement_indices = tuple(
                            atom.atom_index
                            for atom in select_atoms(
                                topology_atoms, definition, selection_id
                            )
                        )
                except (AtomMappingError, CoordinateReadError, OSError, StopIteration) as exc:
                    issues.append(issue_record(
                        "error", "CHEMICAL_REFERENCE_FAILED", replica_location, str(exc)
                    ))
            processor = None
            if topology_atom_count >= 0:
                try:
                    processor = PeriodicFrameProcessor.from_replica(
                        project, replica, system_path, topology_atom_count
                    )
                except PeriodicReconstructionError as exc:
                    issues.append(issue_record(
                        "error", "PERIODIC_RECONSTRUCTION_SETUP_FAILED",
                        replica_location, str(exc),
                    ))

            segment_reports: List[Dict[str, object]] = []
            segments = replica["segments"]
            assert isinstance(segments, list)
            for segment in segments:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                location = f"{replica_location}/{segment_id}"
                trajectory_path = resolve_manifest_path(
                    str(segment["trajectory"]), system_path
                )
                source_frames = source_frame_count(
                    trajectory_path,
                    declared_coordinate_unit,
                    error_type=StructuralQCError,
                )
                selected_indices = frame_selection_plan[
                    (system_id, replica_id, segment_id)
                ]
                checkpoint_file = (
                    _checkpoint_path(
                        checkpoint_root, system_id, replica_id, segment_id
                    )
                    if checkpoint_root is not None else None
                )
                checkpoint = (
                    _load_structural_qc_checkpoint(
                        checkpoint_file, checkpoint_identity
                    )
                    if checkpoint_file is not None else None
                )
                if checkpoint is not None:
                    if (
                        checkpoint.get("location") != location
                        or checkpoint.get("source_frame_count") != source_frames
                    ):
                        raise StructuralQCError(
                            f"structural-QC checkpoint does not match {location}"
                        )
                    checkpoint_status = checkpoint.get("status")
                    if checkpoint_status == "complete":
                        report_value = checkpoint.get("segment_report")
                        issue_value = checkpoint.get("issues")
                        processor_value = checkpoint.get("processor_state")
                        if (
                            not isinstance(report_value, dict)
                            or not isinstance(issue_value, list)
                            or not isinstance(processor_value, dict)
                            or processor is None
                        ):
                            raise StructuralQCError(
                                f"completed structural-QC checkpoint is invalid for {location}"
                            )
                        try:
                            processor.restore_checkpoint_state(processor_value)
                        except PeriodicReconstructionError as exc:
                            raise StructuralQCError(
                                f"completed structural-QC checkpoint has invalid "
                                f"periodic state for {location}: {exc}"
                            ) from exc
                        issues.extend(issue_value)
                        segment_reports.append(report_value)
                        continue
                    if checkpoint_status != "in_progress":
                        raise StructuralQCError(
                            f"structural-QC checkpoint status is invalid for {location}"
                        )
                decoded_frames = 0
                evaluated_frames = 0
                periodic_cell_frames = 0
                maximum_absolute_coordinate = 0.0
                maximum_frame_displacement: Optional[float] = None
                total_near_pairs = 0
                maximum_near_pairs = 0
                minimum_near_distance: Optional[float] = None
                near_pair_examples: List[Dict[str, object]] = []
                chemical_frame_count = 0
                chemical_totals = {
                    "peptide_break_count": 0,
                    "omega_outlier_count": 0,
                    "chirality_outlier_count": 0,
                    "declared_covalent_link_outlier_count": 0,
                    "steric_clash_count": 0,
                }
                chemical_examples: Dict[str, List[Dict[str, object]]] = {
                    "peptide_break_examples": [],
                    "omega_outlier_examples": [],
                    "chirality_outlier_examples": [],
                    "declared_covalent_link_outliers": [],
                    "steric_clash_examples": [],
                }
                source_units = set()
                timing = normalize_segment_timing(segment, time_unit)
                first_evaluated_time = None
                last_evaluated_time = None
                previous_displacement_coordinates: Optional[
                    Sequence[Tuple[float, float, float]]
                ] = None
                last_decoded_frame_index = -1
                segment_issue_start = len(issues)
                if checkpoint is not None:
                    state = checkpoint.get("segment_state")
                    processor_value = checkpoint.get("processor_state")
                    issue_value = checkpoint.get("issues")
                    if (
                        not isinstance(state, dict)
                        or not isinstance(processor_value, dict)
                        or not isinstance(issue_value, list)
                        or processor is None
                    ):
                        raise StructuralQCError(
                            f"in-progress structural-QC checkpoint is invalid for {location}"
                        )
                    try:
                        decoded_frames = int(state["decoded_frames"])
                        evaluated_frames = int(state["evaluated_frames"])
                        periodic_cell_frames = int(state["periodic_cell_frames"])
                        maximum_absolute_coordinate = float(
                            state["maximum_absolute_coordinate"]
                        )
                        maximum_frame_displacement = state[
                            "maximum_frame_displacement"
                        ]
                        if maximum_frame_displacement is not None:
                            maximum_frame_displacement = float(
                                maximum_frame_displacement
                            )
                        total_near_pairs = int(state["total_near_pairs"])
                        maximum_near_pairs = int(state["maximum_near_pairs"])
                        minimum_near_distance = state["minimum_near_distance"]
                        if minimum_near_distance is not None:
                            minimum_near_distance = float(minimum_near_distance)
                        near_pair_examples = list(state["near_pair_examples"])
                        chemical_frame_count = int(state["chemical_frame_count"])
                        chemical_totals = {
                            str(key): int(value)
                            for key, value in dict(state["chemical_totals"]).items()
                        }
                        chemical_examples = {
                            str(key): list(value)
                            for key, value in dict(state["chemical_examples"]).items()
                        }
                        source_units = set(state["source_units"])
                        first_evaluated_time = state["first_evaluated_time"]
                        last_evaluated_time = state["last_evaluated_time"]
                        last_decoded_frame_index = int(
                            state["last_decoded_frame_index"]
                        )
                        previous_value = state[
                            "previous_displacement_coordinates"
                        ]
                        previous_displacement_coordinates = (
                            tuple(
                                tuple(float(coordinate) for coordinate in point)
                                for point in previous_value
                            )
                            if previous_value is not None else None
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise StructuralQCError(
                            f"in-progress structural-QC checkpoint state is invalid for {location}"
                        ) from exc
                    if (
                        decoded_frames < 0 or evaluated_frames < 0
                        or last_decoded_frame_index < 0
                        or last_decoded_frame_index >= source_frames
                    ):
                        raise StructuralQCError(
                            f"in-progress structural-QC checkpoint range is invalid for {location}"
                        )
                    try:
                        processor.restore_checkpoint_state(processor_value)
                    except PeriodicReconstructionError as exc:
                        raise StructuralQCError(
                            f"in-progress structural-QC checkpoint has invalid "
                            f"periodic state for {location}: {exc}"
                        ) from exc
                    issues.extend(issue_value)
                elif processor is not None:
                    processor.begin_segment(
                        bool(segment.get("continuous_with_previous", False))
                    )

                def save_checkpoint(
                    status: str, segment_report: Optional[Dict[str, object]] = None,
                ) -> None:
                    if checkpoint_file is None or processor is None:
                        return
                    previous_value = (
                        [list(point) for point in previous_displacement_coordinates]
                        if previous_displacement_coordinates is not None else None
                    )
                    payload: Dict[str, object] = {
                        "status": status,
                        "location": location,
                        "source_frame_count": source_frames,
                        "processor_state": processor.checkpoint_state(),
                        "issues": issues[segment_issue_start:],
                        "segment_report": segment_report,
                        "segment_state": {
                            "decoded_frames": decoded_frames,
                            "evaluated_frames": evaluated_frames,
                            "periodic_cell_frames": periodic_cell_frames,
                            "maximum_absolute_coordinate": maximum_absolute_coordinate,
                            "maximum_frame_displacement": maximum_frame_displacement,
                            "total_near_pairs": total_near_pairs,
                            "maximum_near_pairs": maximum_near_pairs,
                            "minimum_near_distance": minimum_near_distance,
                            "near_pair_examples": near_pair_examples,
                            "chemical_frame_count": chemical_frame_count,
                            "chemical_totals": chemical_totals,
                            "chemical_examples": chemical_examples,
                            "source_units": sorted(source_units),
                            "first_evaluated_time": first_evaluated_time,
                            "last_evaluated_time": last_evaluated_time,
                            "last_decoded_frame_index": last_decoded_frame_index,
                            "previous_displacement_coordinates": previous_value,
                        },
                    }
                    _write_structural_qc_checkpoint(
                        checkpoint_file, checkpoint_identity, payload
                    )

                checkpoint_clock = time.monotonic()

                def save_timed_checkpoint_if_due() -> None:
                    nonlocal checkpoint_clock
                    if checkpoint_file is None:
                        return
                    now = time.monotonic()
                    if now - checkpoint_clock >= float(
                        checkpointing["within_segment_interval_seconds"]
                    ):
                        save_checkpoint("in_progress")
                        checkpoint_clock = now

                try:
                    if processor is None:
                        raise PeriodicReconstructionError(
                            "periodic frame processor could not be initialized"
                        )
                    reader_indices = reader_frame_indices(
                        selected_indices,
                        str(contract["periodic_coordinate_policy"]),
                    )
                    if last_decoded_frame_index >= 0:
                        remaining = set(
                            range(last_decoded_frame_index + 1, source_frames)
                        )
                        reader_indices = (
                            remaining
                            if reader_indices is None
                            else set(reader_indices).intersection(remaining)
                        )
                    frames = iter_coordinate_frames(
                        trajectory_path,
                        declared_coordinate_unit,
                        reader_indices,
                    )
                    for raw_frame in frames:
                        frame = processor.process(
                            raw_frame, f"{location}/frame-{raw_frame.frame_index}"
                        )
                        decoded_frames += 1
                        last_decoded_frame_index = int(frame.frame_index)
                        if not frame_selected(
                            frame.frame_index,
                            selected_indices,
                            int(thresholds["frame_stride"]),
                        ):
                            save_timed_checkpoint_if_due()
                            continue
                        source_units.add(frame.source_unit)
                        if frame.periodic_cell_present:
                            periodic_cell_frames += 1
                        if topology_atom_count >= 0 and frame.atom_count != topology_atom_count:
                            issues.append(issue_record(
                                "error",
                                "ATOM_COUNT_MISMATCH",
                                f"{location}/frame-{frame.frame_index}",
                                f"trajectory frame has {frame.atom_count} atoms; "
                                f"topology has {topology_atom_count}",
                            ))
                        finite_count = finite_coordinate_count(frame)
                        if finite_count != frame.atom_count:
                            issues.append(issue_record(
                                "error",
                                "NONFINITE_COORDINATES",
                                f"{location}/frame-{frame.frame_index}",
                                f"{frame.atom_count - finite_count} atoms have non-finite coordinates",
                            ))
                        frame_maximum = max(
                            (
                                abs(value)
                                for coordinate in frame.coordinates_angstrom
                                for value in coordinate
                                if math.isfinite(value)
                            ),
                            default=0.0,
                        )
                        maximum_absolute_coordinate = max(
                            maximum_absolute_coordinate, frame_maximum
                        )
                        if frame_maximum > thresholds["maximum_absolute_coordinate_angstrom"]:
                            issues.append(issue_record(
                                "warning",
                                "COORDINATE_EXTENT_EXCEEDED",
                                f"{location}/frame-{frame.frame_index}",
                                f"maximum absolute coordinate is {frame_maximum:.6g} angstrom; "
                                f"gate is {thresholds['maximum_absolute_coordinate_angstrom']:.6g}",
                            ))
                        displacement_coordinates = (
                            tuple(
                                frame.coordinates_angstrom[index]
                                for index in displacement_indices
                            )
                            if displacement_indices is not None
                            else frame.coordinates_angstrom
                        )
                        if previous_displacement_coordinates is not None:
                            displacement = _maximum_rigid_body_aligned_displacement(
                                previous_displacement_coordinates,
                                displacement_coordinates,
                            )
                            if displacement is not None:
                                maximum_frame_displacement = (
                                    displacement
                                    if maximum_frame_displacement is None
                                    else max(maximum_frame_displacement, displacement)
                                )
                                gate = thresholds[
                                    "maximum_frame_atom_displacement_angstrom"
                                ]
                                if gate is not None and displacement > gate:
                                    issues.append(issue_record(
                                        "warning",
                                        "FRAME_DISPLACEMENT_EXCEEDED",
                                        f"{location}/frame-{frame.frame_index}",
                                        f"maximum rigid-body-aligned atom displacement from the preceding decoded frame is "
                                        f"{displacement:.6g} angstrom; gate is {gate:.6g}",
                                    ))
                        previous_displacement_coordinates = displacement_coordinates

                        evaluated_frames += 1
                        current_time = frame_time(timing, frame.frame_index)
                        if first_evaluated_time is None:
                            first_evaluated_time = current_time
                        last_evaluated_time = current_time
                        pair_count, minimum, examples = _near_coincident_pairs(
                            frame,
                            float(thresholds["near_coincident_distance_angstrom"]),
                        )
                        total_near_pairs += pair_count
                        maximum_near_pairs = max(maximum_near_pairs, pair_count)
                        if minimum is not None:
                            minimum_near_distance = (
                                minimum
                                if minimum_near_distance is None
                                else min(minimum_near_distance, minimum)
                            )
                        for example in examples:
                            if len(near_pair_examples) >= 10:
                                break
                            near_pair_examples.append({
                                "frame_index": frame.frame_index,
                                **example,
                            })
                        allowed_pairs = int(
                            thresholds["maximum_near_coincident_pairs_per_frame"]
                        )
                        if pair_count > allowed_pairs:
                            issues.append(issue_record(
                                "warning",
                                "NEAR_COINCIDENT_PAIR_THRESHOLD_EXCEEDED",
                                f"{location}/frame-{frame.frame_index}",
                                f"found {pair_count} atom pairs at or below "
                                f"{thresholds['near_coincident_distance_angstrom']:.6g} angstrom; "
                                f"allowed maximum is {allowed_pairs}",
                            ))
                        chemical_settings = thresholds["chemical_integrity"]
                        if chemical_settings is not None:
                            if chemical_atoms is None or chirality_reference is None:
                                raise StructuralChemistryError(
                                    "chemical-integrity reference was not initialized"
                                )
                            snapshot = chemical_integrity_snapshot(
                                chemical_atoms,
                                frame.coordinates_angstrom,
                                maximum_peptide_bond_angstrom=float(chemical_settings["maximum_peptide_bond_angstrom"]),
                                maximum_trans_omega_deviation_degrees=float(chemical_settings["maximum_trans_omega_deviation_degrees"]),
                                minimum_ca_chirality_volume_angstrom3=float(chemical_settings["minimum_ca_chirality_volume_angstrom3"]),
                                steric_clash_scale=float(chemical_settings["steric_clash_scale"]),
                                reference_chirality=chirality_reference,
                                covalent_bonds=chemical_bonds,
                                allow_cis_proline=bool(chemical_settings["allow_cis_proline"]),
                                declared_covalent_links=chemical_settings["declared_covalent_links"],
                            )
                            chemical_frame_count += 1
                            for key in chemical_totals:
                                chemical_totals[key] += int(snapshot[key])
                            for key in chemical_examples:
                                for example in snapshot[key]:
                                    if len(chemical_examples[key]) < 10:
                                        chemical_examples[key].append({"frame_index": frame.frame_index, **example})
                            gate_codes = {
                                "peptide_break_count": "PEPTIDE_CONTINUITY_OUTLIER",
                                "omega_outlier_count": "OMEGA_GEOMETRY_OUTLIER",
                                "chirality_outlier_count": "CA_CHIRALITY_OUTLIER",
                                "declared_covalent_link_outlier_count": "DECLARED_COVALENT_LINK_OUTLIER",
                            }
                            for key, code in gate_codes.items():
                                if int(snapshot[key]):
                                    issues.append(issue_record(
                                        "warning", code, f"{location}/frame-{frame.frame_index}",
                                        f"chemical-integrity snapshot reported {snapshot[key]} {key.replace('_', ' ')}",
                                    ))
                            clash_count = int(snapshot["steric_clash_count"])
                            maximum_clashes = int(chemical_settings["maximum_steric_clashes_per_frame"])
                            if chemical_bonds is not None and clash_count > maximum_clashes:
                                issues.append(issue_record(
                                    "warning", "STERIC_CLASH_THRESHOLD_EXCEEDED", f"{location}/frame-{frame.frame_index}",
                                    f"found {clash_count} inter-residue element-radius clashes; allowed maximum is {maximum_clashes}",
                                ))
                        save_timed_checkpoint_if_due()
                    if decoded_frames == 0:
                        issues.append(issue_record(
                            "error", "NO_COORDINATE_FRAMES", location,
                            "trajectory contains no readable coordinate frames",
                        ))
                except (
                    CoordinateReadError, PeriodicReconstructionError,
                    StructuralChemistryError, OSError,
                ) as exc:
                    issues.append(issue_record(
                        "error", "COORDINATE_READ_FAILED", location, str(exc)
                    ))

                inventory_record = inventory_by_path.get(str(trajectory_path), {})
                segment_report = {
                    "segment_id": segment_id,
                    "trajectory_path": str(trajectory_path),
                    "trajectory_format": (
                        coordinate_format(trajectory_path)
                        if trajectory_path.suffix.lower() in {".pdb", ".ent", ".gro", ".xyz", ".dcd"}
                        else "unsupported"
                    ),
                    "sha256": inventory_record.get("sha256"),
                    "observed_frame_count": source_frames,
                    "decoded_frame_count": decoded_frames,
                    "evaluated_frame_count": evaluated_frames,
                    "source_units": sorted(source_units),
                    "coordinates_normalized_to": "angstrom",
                    "periodic_cell_frame_count": periodic_cell_frames,
                    "timing": timing,
                    "evaluated_time_range": (
                        {
                            "start": first_evaluated_time,
                            "end": last_evaluated_time,
                            "unit": time_unit,
                        }
                        if first_evaluated_time is not None
                        else None
                    ),
                    "maximum_absolute_coordinate_angstrom": maximum_absolute_coordinate,
                    "maximum_frame_atom_displacement_angstrom": maximum_frame_displacement,
                    "frame_displacement_selection": thresholds[
                        "frame_displacement_selection"
                    ],
                    "frame_displacement_atom_count": (
                        len(displacement_indices)
                        if displacement_indices is not None
                        else topology_atom_count
                    ),
                    "frame_displacement_definition": (
                        "maximum finite-atom residual after proper least-squares "
                        "rigid-body superposition of consecutive decoded frames "
                        "on the declared frame-displacement selection"
                    ),
                    "total_near_coincident_pair_observations": total_near_pairs,
                    "maximum_near_coincident_pairs_in_one_frame": maximum_near_pairs,
                    "minimum_near_coincident_distance_angstrom": minimum_near_distance,
                    "near_coincident_pair_examples": near_pair_examples,
                    "chemical_integrity_evaluated_frame_count": chemical_frame_count,
                    "chemical_connectivity": chemical_connectivity,
                    "chemical_integrity_totals": chemical_totals,
                    "chemical_integrity_examples": chemical_examples,
                }
                save_checkpoint("complete", segment_report)
                segment_reports.append(segment_report)
            replica_reports.append({
                "replica_id": replica_id,
                "topology_path": str(topology_path),
                "topology": topology,
                "periodic_reconstruction": processor.report() if processor is not None else None,
                "segments": segment_reports,
            })
        system_reports.append({"system_id": system_id, "replicas": replica_reports})

    if int(frame_selection_report["selected_frame_count"]) < int(
        frame_selection_report["source_frame_count"]
    ):
        issues.append(issue_record(
            "warning",
            "FRAME_SUBSAMPLING",
            str(project_source),
            "coordinate and chemical gates evaluated on the declared exact "
            "integer-stride frame sample; "
            "unselected DCD record envelopes were validated without decoding coordinate payloads",
        ))
    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    qc_finding_count = sum(
        issue.get("code") in _QC_FINDING_CODES for issue in issues
    )
    return {
        "module_id": "structural_integrity_qc",
        "technical_status": "failed" if error_count else "complete",
        "scientific_status": "not evaluated",
        "qc_status": (
            "review_required" if qc_finding_count else "no_findings_observed"
        ),
        "human_review_status": (
            "pending" if qc_finding_count else "not requested"
        ),
        "project_manifest_path": str(project_source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "contract_signature_sha256": context["contract_signature_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "thresholds": thresholds,
        "checkpointing": {
            "enabled": checkpoint_root is not None,
            "within_segment_interval_seconds": checkpointing[
                "within_segment_interval_seconds"
            ],
            "segment_completion_checkpoints": True,
            "checkpoint_identity": checkpoint_identity,
        },
        "frame_selection": frame_selection_report,
        "periodic_coordinate_policy": contract["periodic_coordinate_policy"],
        "time_unit": time_unit,
        "error_count": error_count,
        "warning_count": warning_count,
        "qc_finding_count": qc_finding_count,
        "issues": issues,
        "systems": system_reports,
        "limitations": [
            "Coordinate gates always run; peptide continuity, omega, C-alpha chirality, element-radius clashes, and declared covalent-link checks run when chemical_integrity is explicitly configured.",
            "Peptide-link and omega checks are evaluated only for explicit topology C-N bonds; residue ordering, numbering, and shared chain identifiers are never treated as proof of a peptide bond.",
            "Same-residue and explicit topology 1-2 and 1-3 pairs are excluded from the conservative steric screen; specialized cofactor, metal, interface, and nonstandard-residue chemistry still requires declared project adapters.",
            "Near-coincident checks do not apply periodic minimum-image distances; their short default-style gates are intended to detect duplicate or corrupted coordinates, not all steric clashes.",
            "Frame displacement removes one proper rigid-body transform after the declared periodic reconstruction and uses the declared selection; bulk solvent and mobile ions should be excluded because their physical diffusion and box-image changes are not macromolecular structural corruption.",
            "When frame subsampling is active, coordinate-level QC conclusions apply to the reported exact integer-stride sample; preflight still validates every DCD record envelope but cannot detect coordinate anomalies inside unselected payloads.",
            "Threshold exceedances are review findings, not execution failures, and do not block other technically executable analyses.",
            "A machine-generated report never declares scientific failure; scientific usability, equilibration, convergence, and adequate sampling require human review.",
        ],
    }


def _write_cache_backed_qc_project(
    project_source: Path,
    project: Mapping[str, object],
    cache_validation: Mapping[str, object],
    parallel: Mapping[str, object],
) -> Path:
    """Write one deterministic runtime project that declares verified cache use."""

    payload = deepcopy(dict(project))
    payload["system_manifest"] = str(cache_validation["cached_system_manifest"])
    payload["periodic_coordinate_policy"] = "preprocessed_make_whole"
    payload["preprocessed_coordinate_source"] = {
        "cache_report": str(cache_validation["cache_report"]),
        "cache_report_sha256": str(cache_validation["cache_report_sha256"]),
    }
    definitions = payload.get("definitions")
    if not isinstance(definitions, dict):
        raise StructuralQCError("cache-backed structural QC lacks definitions")
    structural = definitions.get("structural_qc")
    if not isinstance(structural, dict):
        raise StructuralQCError("cache-backed structural QC lacks its definition")
    structural["parallel_execution"] = {
        **dict(parallel), "enabled": False, "maximum_workers": 1,
    }
    signature = hashlib.sha256(
        _checkpoint_payload_bytes({
            "source_project": str(project_source),
            "source_project_sha256": hashlib.sha256(
                project_source.read_bytes()
            ).hexdigest(),
            "cache_report_sha256": cache_validation["cache_report_sha256"],
            "cached_system_manifest_sha256": cache_validation[
                "cached_system_manifest_sha256"
            ],
        })
    ).hexdigest()[:16]
    destination = project_source.parent / (
        f"project-structural-qc-cache-{signature}.json"
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != rendered:
            raise StructuralQCError(
                f"cache-backed structural-QC project changed: {destination}"
            )
    else:
        destination.write_text(rendered, encoding="utf-8")
    return destination


def _structural_qc_replica_job(shard: ReplicaShard) -> Dict[str, object]:
    project_path = Path(str(shard.payload))
    return _structural_qc_project_serial(
        project_path,
        hash_content=False,
        only_system_id=shard.system_id,
        only_replica_id=shard.replica_id,
    )


def _aggregate_structural_qc_shards(
    project_source: Path,
    thresholds: Mapping[str, object],
    cache_validation: Mapping[str, object],
    reports: Sequence[Mapping[str, object]],
    *,
    configured_workers: int,
    workers_used: int,
) -> Dict[str, object]:
    if not reports:
        raise StructuralQCError("parallel structural QC produced no shard reports")
    systems: Dict[str, Dict[str, object]] = {}
    issues: List[Dict[str, object]] = []
    seen_issues = set()
    replicas = []
    source_frames = 0
    selected_frames = 0
    limitations: List[str] = []
    for report in reports:
        frame_selection = report.get("frame_selection")
        if not isinstance(frame_selection, dict):
            raise StructuralQCError("structural-QC shard lacks frame-selection evidence")
        source_frames += int(frame_selection["source_frame_count"])
        selected_frames += int(frame_selection["selected_frame_count"])
        raw_replicas = frame_selection.get("replicas")
        if not isinstance(raw_replicas, list):
            raise StructuralQCError("structural-QC shard lacks replica sampling evidence")
        replicas.extend(deepcopy(raw_replicas))
        for raw_issue in report.get("issues", []):
            if not isinstance(raw_issue, dict):
                continue
            identity = json.dumps(raw_issue, sort_keys=True, separators=(",", ":"))
            if identity not in seen_issues:
                seen_issues.add(identity)
                issues.append(deepcopy(raw_issue))
        for raw_system in report.get("systems", []):
            if not isinstance(raw_system, dict):
                continue
            system_id = str(raw_system["system_id"])
            target = systems.setdefault(system_id, {
                "system_id": system_id, "replicas": [],
            })
            target_replicas = target["replicas"]
            assert isinstance(target_replicas, list)
            target_replicas.extend(deepcopy(raw_system.get("replicas", [])))
        for limitation in report.get("limitations", []):
            if isinstance(limitation, str) and limitation not in limitations:
                limitations.append(limitation)
    error_count = sum(issue.get("severity") == "error" for issue in issues)
    warning_count = sum(issue.get("severity") == "warning" for issue in issues)
    qc_finding_count = sum(
        issue.get("code") in _QC_FINDING_CODES for issue in issues
    )
    first = reports[0]
    cache_stride = int(cache_validation["cache_stride"])
    raw_method_stride = first["frame_selection"].get("resolved_integer_stride")
    if raw_method_stride is None:
        raw_method_stride = first["frame_selection"].get("frame_stride", 1)
    method_stride = int(raw_method_stride)
    limitations.append(
        "Replica shards consume one validated, all-frame-scanned, made-whole "
        f"molecular-payload cache at raw stride {cache_stride}; bulk-solvent "
        "integrity remains outside this QC report."
    )
    return {
        "module_id": "structural_integrity_qc",
        "technical_status": "failed" if error_count else "complete",
        "scientific_status": "not evaluated",
        "qc_status": (
            "review_required" if qc_finding_count else "no_findings_observed"
        ),
        "human_review_status": (
            "pending" if qc_finding_count else "not requested"
        ),
        "project_manifest_path": str(project_source),
        "project_manifest_sha256": hashlib.sha256(
            project_source.read_bytes()
        ).hexdigest(),
        "system_manifest_path": cache_validation["cached_system_manifest"],
        "system_manifest_sha256": cache_validation[
            "cached_system_manifest_sha256"
        ],
        "contract_signature_sha256": first.get("contract_signature_sha256"),
        "input_content_signature_sha256": first.get(
            "input_content_signature_sha256"
        ),
        "content_hashes_included": True,
        "thresholds": dict(thresholds),
        "checkpointing": first.get("checkpointing"),
        "frame_selection": {
            "mode": first["frame_selection"].get("mode"),
            "resolved_mode": first["frame_selection"].get("resolved_mode"),
            "frame_stride": first["frame_selection"].get("frame_stride"),
            "resolved_integer_stride": first["frame_selection"].get(
                "resolved_integer_stride"
            ),
            "coordinate_cache_integer_stride": cache_stride,
            "effective_raw_integer_stride": cache_stride * method_stride,
            "selection_contract": first["frame_selection"].get(
                "selection_contract"
            ),
            "source_frame_count": source_frames,
            "selected_frame_count": selected_frames,
            "coverage_fraction": selected_frames / source_frames,
            "replicas": replicas,
        },
        "periodic_coordinate_policy": "preprocessed_make_whole",
        "time_unit": first.get("time_unit"),
        "error_count": error_count,
        "warning_count": warning_count,
        "qc_finding_count": qc_finding_count,
        "issues": issues,
        "systems": list(systems.values()),
        "parallel_execution": {
            "execution_model": "one_process_per_replica_v1",
            "configured_maximum_workers": configured_workers,
            "workers_used": workers_used,
            "shard_count": len(reports),
            "cache_validation": dict(cache_validation),
        },
        "limitations": limitations,
    }


def structural_qc_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Run structural QC serially or in validated cache-backed replica shards."""

    project_source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(project_source)
    thresholds = _thresholds(project)
    parallel = thresholds.get("parallel_execution")
    if not isinstance(parallel, dict) or not bool(parallel.get("enabled")):
        return _structural_qc_project_serial(
            project_source, hash_content=hash_content
        )
    context = compile_project_context_file(
        project_source, hash_content=hash_content
    )
    source_system = Path(str(context["system_manifest_path"]))
    cache_manifest = resolve_manifest_path(
        str(parallel["coordinate_cache_system_manifest"]), project_source
    )
    cache_report = resolve_manifest_path(
        str(parallel["coordinate_cache_report"]), project_source
    )
    cache_root = cache_manifest.parent
    cache_validation = validate_reusable_coordinate_cache(
        cache_root, source_system
    )
    if Path(str(cache_validation["cached_system_manifest"])) != cache_manifest:
        raise StructuralQCError(
            "parallel structural-QC cache manifest differs from validated cache"
        )
    if Path(str(cache_validation["cache_report"])) != cache_report:
        raise StructuralQCError(
            "parallel structural-QC cache report differs from validated cache"
        )
    cache_project = _write_cache_backed_qc_project(
        project_source, project, cache_validation, parallel
    )
    cached = load_json(cache_manifest)
    shards = [
        ReplicaShard(
            ordinal=ordinal,
            system_id=str(system["system_id"]),
            replica_id=str(replica["replica_id"]),
            segment_ids=tuple(
                str(segment.get("segment_id", f"segment-{index + 1}"))
                for index, segment in enumerate(replica.get("segments", []))
                if isinstance(segment, dict)
            ),
            payload=str(cache_project),
        )
        for ordinal, (system, replica) in enumerate(
            (system, replica)
            for system in cached.get("systems", [])
            if isinstance(system, dict)
            for replica in system.get("replicas", [])
            if isinstance(replica, dict)
        )
    ]
    if not shards:
        raise StructuralQCError("validated coordinate cache contains no replicas")
    configured_workers = int(parallel["maximum_workers"])
    partials, execution_evidence = execute_replica_workers(
        shards, _structural_qc_replica_job,
        maximum_workers=configured_workers,
    )
    result = _aggregate_structural_qc_shards(
        project_source,
        thresholds,
        cache_validation,
        [partial.value for partial in partials],
        configured_workers=configured_workers,
        workers_used=execution_evidence.workers_used,
    )
    result["parallel_execution"]["replica_execution_evidence"] = (
        execution_evidence.as_dict()
    )
    return result


def structural_qc_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Convert configuration/manifest failures into a machine-readable report."""

    try:
        return structural_qc_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, StructuralQCError, CoordinateCacheError,
        AtomMappingError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "structural_integrity_qc",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "qc_status": "not_evaluated",
            "human_review_status": "pending",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "qc_finding_count": 0,
            "issues": [
                {"severity": "error", "code": "STRUCTURAL_QC_INVALID", "message": message}
                for message in messages
            ],
        }
