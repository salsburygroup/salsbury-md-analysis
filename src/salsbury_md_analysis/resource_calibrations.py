"""Hash-bound measured CPU, memory, and frame-coverage planner evidence."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .manifests import load_json


class ResourceCalibrationError(ValueError):
    """Raised when measured planner evidence is incomplete or untrustworthy."""


LEGACY_SCHEMA = "salsbury-measured-resource-calibration-catalog-v1"
SCHEMA = "salsbury-measured-resource-calibration-catalog-v2"
TIMEOUT_SCHEMA = "salsbury-censored-timeout-resource-evidence-v1"
MEMORY_REPLACEMENT_MIN_COMPLETE_MEASUREMENTS = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_number(value: object, label: str, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
    ):
        raise ResourceCalibrationError(f"{label} must be a finite positive number")
    return float(value)


def _entry_from_sidecar(path: Path) -> Dict[str, object]:
    sidecar = load_json(path)
    if sidecar.get("technical_status") != "complete":
        raise ResourceCalibrationError(f"sidecar is not technically complete: {path}")
    report_path = Path(str(sidecar.get("report_path", ""))).expanduser().resolve(strict=True)
    if sidecar.get("report_sha256") != _sha256(report_path):
        raise ResourceCalibrationError(f"sidecar report hash mismatch: {path}")
    evidence = sidecar.get("resource_evidence")
    resources = evidence.get("execution_resources") if isinstance(evidence, dict) else None
    if not isinstance(evidence, dict) or not isinstance(resources, dict):
        raise ResourceCalibrationError(f"sidecar lacks resource evidence: {path}")
    frames = evidence.get("selected_source_physical_frames")
    observations = evidence.get("symmetry_expanded_observations")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise ResourceCalibrationError(f"sidecar lacks positive frame coverage: {path}")
    if isinstance(observations, bool) or not isinstance(observations, int) or observations <= 0:
        raise ResourceCalibrationError(f"sidecar lacks positive observation coverage: {path}")
    cpu = _positive_number(resources.get("total_cpu_seconds"), "total_cpu_seconds")
    wall = _positive_number(resources.get("wall_seconds"), "wall_seconds")
    memory = _positive_number(
        resources.get("maximum_resident_memory_mib"),
        "maximum_resident_memory_mib",
    )
    module_id = str(sidecar.get("module_id", "")).strip()
    if not module_id:
        raise ResourceCalibrationError(f"sidecar lacks module_id: {path}")
    workload_fields = {
        key: evidence.get(key)
        for key in (
            "basis_selected_physical_frames", "basis_member_observations",
            "model_fit_observations", "model_fit_equivalent_physical_frames",
            "full_assignment_observations", "silhouette_evaluation_observations",
            "conceptual_candidate_frame_count", "spatial_neighbor_pair_count",
            "explicit_geometry_evaluation_count", "present_event_count",
            "maximum_spatial_endpoint_count_per_system",
        )
        if evidence.get(key) is not None
    }
    return {
        "evidence_status": "complete_execution",
        "frame_coverage_status": "technically_complete",
        "module_id": module_id,
        "selected_source_physical_frames": frames,
        "symmetry_expanded_observations": observations,
        "total_cpu_seconds": cpu,
        "wall_seconds": wall,
        "maximum_resident_memory_mib": memory,
        "cpu_seconds_per_selected_physical_frame": cpu / frames,
        "source_sidecar_path": str(path),
        "source_sidecar_sha256": _sha256(path),
        "source_report_path": str(report_path),
        "source_report_sha256": str(sidecar["report_sha256"]),
        "source_report_size_bytes": sidecar.get("report_size_bytes"),
        "computer_hostname": resources.get("computer_hostname"),
        "platform": resources.get("platform"),
        "measurement_scope": resources.get("measurement_scope"),
        "requested_cpu_count": resources.get("requested_cpu_count"),
        **workload_fields,
    }


def _entry_from_timeout(path: Path) -> Dict[str, object]:
    """Load one fail-closed right-censored scheduler timeout record."""

    record = load_json(path)
    if (
        record.get("evidence_schema") != TIMEOUT_SCHEMA
        or record.get("technical_status") != "timeout"
    ):
        raise ResourceCalibrationError(f"invalid censored timeout evidence: {path}")
    module_id = str(record.get("module_id", "")).strip()
    if not module_id:
        raise ResourceCalibrationError(f"timeout evidence lacks module_id: {path}")
    frames = record.get("selected_source_physical_frames")
    observations = record.get("symmetry_expanded_observations")
    cpus = record.get("allocated_cpu_count")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise ResourceCalibrationError(f"timeout evidence lacks planned frame coverage: {path}")
    if (
        isinstance(observations, bool)
        or not isinstance(observations, int)
        or observations <= 0
    ):
        raise ResourceCalibrationError(
            f"timeout evidence lacks planned observation coverage: {path}"
        )
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus <= 0:
        raise ResourceCalibrationError(f"timeout evidence lacks allocated CPUs: {path}")
    elapsed = _positive_number(record.get("elapsed_seconds"), "elapsed_seconds")
    declared_cpu_lower = record.get("total_cpu_seconds_lower_bound")
    cpu_lower = (
        elapsed * cpus
        if declared_cpu_lower is None
        else _positive_number(
            declared_cpu_lower, "total_cpu_seconds_lower_bound"
        )
    )
    if cpu_lower + 1.0e-9 < elapsed:
        raise ResourceCalibrationError(
            f"timeout CPU lower bound is shorter than elapsed wall time: {path}"
        )
    raw_memory = record.get("maximum_resident_memory_mib")
    memory = (
        None
        if raw_memory is None
        else _positive_number(raw_memory, "maximum_resident_memory_mib")
    )
    return {
        "evidence_status": "right_censored_timeout",
        "frame_coverage_status": "planned_not_completed",
        "module_id": module_id,
        "selected_source_physical_frames": frames,
        "symmetry_expanded_observations": observations,
        "wall_seconds_lower_bound": elapsed,
        "total_cpu_seconds_lower_bound": cpu_lower,
        "cpu_seconds_per_selected_physical_frame_lower_bound": cpu_lower / frames,
        "maximum_resident_memory_mib": memory,
        "allocated_cpu_count": cpus,
        "scheduler_job_id": record.get("scheduler_job_id"),
        "scheduler_array_task_id": record.get("scheduler_array_task_id"),
        "source_timeout_path": str(path),
        "source_timeout_sha256": _sha256(path),
        "scientific_status": record.get("scientific_status", "not evaluated"),
    }


def _conservative_affine_cpu_model(
    complete_rows: Sequence[Mapping[str, object]],
    timeout_rows: Sequence[Mapping[str, object]],
    timeout_safety: float,
) -> tuple[float, float]:
    """Return a nonnegative affine envelope over completed/censored evidence."""

    by_frames: Dict[int, float] = {}
    for row in complete_rows:
        frames = int(row["selected_source_physical_frames"])
        by_frames[frames] = max(
            by_frames.get(frames, 0.0), float(row["total_cpu_seconds"])
        )
    for row in timeout_rows:
        frames = int(row["selected_source_physical_frames"])
        by_frames[frames] = max(
            by_frames.get(frames, 0.0),
            float(row["total_cpu_seconds_lower_bound"]) * timeout_safety,
        )
    points = sorted(by_frames.items())
    if not points:
        return 0.0, 0.0
    if len(points) == 1:
        frames, cpu = points[0]
        return 0.0, cpu / frames
    positive_slopes = [
        (right_cpu - left_cpu) / (right_frames - left_frames)
        for (left_frames, left_cpu), (right_frames, right_cpu)
        in zip(points, points[1:])
        if right_cpu > left_cpu
    ]
    slope = max(positive_slopes, default=0.0)
    if slope <= 0.0:
        slope = max(cpu / frames for frames, cpu in points)
    intercept = max(0.0, max(cpu - slope * frames for frames, cpu in points))
    return intercept, slope


def build_resource_calibration_catalog(
    sidecars: Sequence[Path],
    *,
    timeout_records: Sequence[Path] = (),
    base_catalogs: Sequence[Path] = (),
) -> Dict[str, object]:
    """Build a lossless catalog from completed and censored execution evidence."""

    paths = sorted({Path(path).expanduser().resolve(strict=True) for path in sidecars})
    timeout_paths = sorted({
        Path(path).expanduser().resolve(strict=True) for path in timeout_records
    })
    base_paths = sorted({
        Path(path).expanduser().resolve(strict=True) for path in base_catalogs
    })
    if not paths and not timeout_paths and not base_paths:
        raise ResourceCalibrationError(
            "at least one complete sidecar, censored timeout record, or base "
            "catalog is required"
        )
    entries = []
    base_provenance = []
    for base_path in base_paths:
        base = load_json(base_path)
        if (
            base.get("catalog_schema") not in {LEGACY_SCHEMA, SCHEMA}
            or base.get("technical_status") != "complete"
            or not isinstance(base.get("entries"), list)
        ):
            raise ResourceCalibrationError(
                f"invalid base resource calibration catalog: {base_path}"
            )
        entries.extend(base["entries"])
        base_provenance.append({
            "path": str(base_path),
            "sha256": _sha256(base_path),
            "entry_count": len(base["entries"]),
        })
    entries.extend(_entry_from_sidecar(path) for path in paths)
    entries.extend(_entry_from_timeout(path) for path in timeout_paths)
    unique_entries: Dict[str, Dict[str, object]] = {}
    duplicate_count = 0
    nonplanning_fields = {
        "computer_hostname", "scheduler_array_task_id", "scheduler_job_id",
        "source_location_redacted", "source_report_path", "source_sidecar_path",
        "source_timeout_path",
    }
    for row in entries:
        if not isinstance(row, dict):
            raise ResourceCalibrationError("base catalog entry must be an object")
        key = (
            row.get("source_sidecar_sha256")
            or row.get("source_timeout_sha256")
        )
        if not isinstance(key, str) or not key:
            raise ResourceCalibrationError(
                "every calibration entry must retain a source evidence hash"
            )
        existing = unique_entries.get(key)
        if existing is None:
            unique_entries[key] = dict(row)
            continue
        normalized_existing = {
            field: value for field, value in existing.items()
            if field not in nonplanning_fields
        }
        normalized_row = {
            field: value for field, value in row.items()
            if field not in nonplanning_fields
        }
        if normalized_existing != normalized_row:
            raise ResourceCalibrationError(
                "duplicate source evidence has conflicting resource values"
            )
        duplicate_count += 1
    entries = list(unique_entries.values())
    complete_count = sum(
        str(row.get("evidence_status", "complete_execution"))
        == "complete_execution" for row in entries
    )
    timeout_count = sum(
        row.get("evidence_status") == "right_censored_timeout" for row in entries
    )
    return {
        "catalog_schema": SCHEMA,
        "technical_status": "complete",
        "entry_count": len(entries),
        "complete_execution_count": complete_count,
        "censored_timeout_count": timeout_count,
        "duplicate_evidence_entry_count": duplicate_count,
        "base_catalogs": base_provenance,
        "entries": entries,
        "planner_policy": {
            "cpu_rate": (
                "maximum of completed CPU rates and separately labeled "
                "right-censored timeout lower bounds after the configured "
                "timeout safety factor"
            ),
            "memory": (
                "two or more complete measurements may qualify completed RSS "
                "to replace a legacy baseline; a single-CPU timeout may set a "
                "planning lower bound, while multi-CPU timeout MaxRSS remains "
                "aggregate diagnostic evidence and is never replayed as one "
                "worker's memory"
            ),
            "frame_coverage": (
                "maximum technically completed selected physical frames only; "
                "timeout target coverage is retained but never counted complete"
            ),
            "spatial_hydrogen_bond_work": (
                "completed spatial neighbor-pair and exact geometry counts are "
                "retained as the runtime work unit; the implicit Cartesian "
                "candidate-frame count remains a correctness gate only"
            ),
        },
        "scientific_status": "runtime evidence only",
    }


def redact_resource_calibration_catalog(
    catalog: Mapping[str, object],
) -> Dict[str, object]:
    """Remove private execution locations without changing planner values.

    Evidence hashes, measured resources, frame coverage, module identities, and
    censoring status remain intact. The redacted result is therefore usable by
    the planner and auditable against a separately retained full provenance
    catalog, but it does not publish cluster paths, scheduler IDs, or hostnames.
    """

    if (
        catalog.get("catalog_schema") != SCHEMA
        or catalog.get("technical_status") != "complete"
        or not isinstance(catalog.get("entries"), list)
    ):
        raise ResourceCalibrationError(
            "only a complete v2 resource calibration catalog can be redacted"
        )
    redacted = deepcopy(dict(catalog))
    base_catalogs = redacted.get("base_catalogs")
    if isinstance(base_catalogs, list):
        for base in base_catalogs:
            if isinstance(base, dict):
                base.pop("path", None)
                base["source_location_redacted"] = True
    private_fields = {
        "computer_hostname",
        "scheduler_array_task_id",
        "scheduler_job_id",
        "source_report_path",
        "source_sidecar_path",
        "source_timeout_path",
    }
    for row in redacted["entries"]:
        if not isinstance(row, dict):
            raise ResourceCalibrationError(
                "resource calibration entry must be an object"
            )
        for field in private_fields:
            row.pop(field, None)
        row["source_location_redacted"] = True
    redacted["provenance_path_policy"] = (
        "Private source paths, scheduler identifiers, and hostnames are retained "
        "only in the separately hash-pinned internal catalog."
    )
    return redacted


def load_resource_calibration_catalog(
    path: Path | str | None,
    *,
    censored_timeout_safety_factor: float = 1.5,
) -> Dict[str, Dict[str, object]]:
    """Validate and conservatively aggregate a catalog by module."""

    if path is None:
        return {}
    source = Path(path).expanduser().resolve(strict=True)
    catalog = load_json(source)
    if (
        catalog.get("catalog_schema") not in {LEGACY_SCHEMA, SCHEMA}
        or catalog.get("technical_status") != "complete"
    ):
        raise ResourceCalibrationError(f"invalid resource calibration catalog: {source}")
    timeout_safety = _positive_number(
        censored_timeout_safety_factor, "censored_timeout_safety_factor"
    )
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ResourceCalibrationError("resource calibration catalog has no entries")
    grouped: Dict[str, list[Mapping[str, object]]] = {}
    for row in entries:
        if not isinstance(row, dict):
            raise ResourceCalibrationError("resource calibration entry must be an object")
        module_id = str(row.get("module_id", "")).strip()
        status = str(row.get("evidence_status", "complete_execution"))
        if status not in {"complete_execution", "right_censored_timeout"}:
            raise ResourceCalibrationError("resource calibration evidence_status is invalid")
        frames = row.get("selected_source_physical_frames")
        observations = row.get("symmetry_expanded_observations")
        if not module_id or isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            raise ResourceCalibrationError("resource calibration module/frame fields are invalid")
        if isinstance(observations, bool) or not isinstance(observations, int) or observations <= 0:
            raise ResourceCalibrationError("resource calibration observation coverage is invalid")
        for field in (
            "conceptual_candidate_frame_count", "spatial_neighbor_pair_count",
            "explicit_geometry_evaluation_count", "present_event_count",
            "maximum_spatial_endpoint_count_per_system",
        ):
            value = row.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ResourceCalibrationError(
                    f"resource calibration {field} must be a nonnegative integer"
                )
        memory = row.get("maximum_resident_memory_mib")
        if memory is not None:
            _positive_number(memory, "maximum_resident_memory_mib")
        if status == "complete_execution":
            cpu = _positive_number(row.get("total_cpu_seconds"), "total_cpu_seconds")
            _positive_number(row.get("wall_seconds"), "wall_seconds")
            if not math.isclose(
                float(row.get("cpu_seconds_per_selected_physical_frame", -1.0)),
                cpu / frames, rel_tol=1e-9, abs_tol=1e-12,
            ):
                raise ResourceCalibrationError("resource calibration CPU rate is inconsistent")
        else:
            cpu = _positive_number(
                row.get("total_cpu_seconds_lower_bound"),
                "total_cpu_seconds_lower_bound",
            )
            _positive_number(row.get("wall_seconds_lower_bound"), "wall_seconds_lower_bound")
            if not math.isclose(
                float(row.get(
                    "cpu_seconds_per_selected_physical_frame_lower_bound", -1.0
                )),
                cpu / frames, rel_tol=1e-9, abs_tol=1e-12,
            ):
                raise ResourceCalibrationError(
                    "censored resource calibration CPU lower-bound rate is inconsistent"
                )
        grouped.setdefault(module_id, []).append(row)
    result: Dict[str, Dict[str, object]] = {}
    for module_id, rows in grouped.items():
        complete_rows = [
            row for row in rows
            if str(row.get("evidence_status", "complete_execution"))
            == "complete_execution"
        ]
        timeout_rows = [
            row for row in rows if row.get("evidence_status") == "right_censored_timeout"
        ]
        completed_rates = [
            float(row["cpu_seconds_per_selected_physical_frame"])
            for row in complete_rows
        ]
        timeout_rates = [
            float(row["cpu_seconds_per_selected_physical_frame_lower_bound"])
            for row in timeout_rows
        ]
        censored_wall_lower_bound_points = sorted(
            (
                {
                    "selected_source_physical_frames": int(
                        row["selected_source_physical_frames"]
                    ),
                    "symmetry_expanded_observations": int(
                        row["symmetry_expanded_observations"]
                    ),
                    "allocated_cpu_count": int(row["allocated_cpu_count"]),
                    "wall_seconds_lower_bound": float(
                        row["wall_seconds_lower_bound"]
                    ),
                    "planning_wall_seconds_lower_bound": float(
                        row["wall_seconds_lower_bound"]
                    ) * timeout_safety,
                }
                for row in timeout_rows
            ),
            key=lambda row: (
                int(row["selected_source_physical_frames"]),
                int(row["allocated_cpu_count"]),
                float(row["wall_seconds_lower_bound"]),
            ),
        )
        planning_rates = list(completed_rates)
        planning_rates.extend(rate * timeout_safety for rate in timeout_rates)
        affine_fixed, affine_rate = _conservative_affine_cpu_model(
            complete_rows, timeout_rows, timeout_safety
        )
        observed_memories = [
            float(row["maximum_resident_memory_mib"])
            for row in rows if row.get("maximum_resident_memory_mib") is not None
        ]
        completed_memories = [
            float(row["maximum_resident_memory_mib"])
            for row in complete_rows
            if row.get("maximum_resident_memory_mib") is not None
        ]
        qualified_censored_memories = [
            float(row["maximum_resident_memory_mib"])
            for row in timeout_rows
            if row.get("maximum_resident_memory_mib") is not None
            and int(row.get("allocated_cpu_count", 0)) == 1
        ]
        planning_memories = [
            *completed_memories, *qualified_censored_memories,
        ]
        complete_observation_counts = [
            int(row["symmetry_expanded_observations"])
            for row in complete_rows
        ]
        spatial_rows = [
            row for row in complete_rows
            if int(row.get("spatial_neighbor_pair_count", 0)) > 0
        ]
        spatial_cpu_rates = [
            float(row["total_cpu_seconds"])
            / int(row["spatial_neighbor_pair_count"])
            for row in spatial_rows
        ]
        spatial_pairs_per_frame = [
            int(row["spatial_neighbor_pair_count"])
            / int(row["selected_source_physical_frames"])
            for row in spatial_rows
        ]
        geometry_evaluations_per_frame = [
            int(row["explicit_geometry_evaluation_count"])
            / int(row["selected_source_physical_frames"])
            for row in complete_rows
            if int(row.get("explicit_geometry_evaluation_count", 0)) > 0
        ]
        spatial_endpoint_counts = [
            int(row["maximum_spatial_endpoint_count_per_system"])
            for row in complete_rows
            if int(row.get("maximum_spatial_endpoint_count_per_system", 0)) > 0
        ]
        memory_replacement_qualified = (
            len(completed_memories)
            >= MEMORY_REPLACEMENT_MIN_COMPLETE_MEASUREMENTS
        )
        measurement_scopes = sorted({
            str(row.get("measurement_scope", "unspecified"))
            for row in complete_rows
        })
        completed_requested_cpu_counts = [
            int(row["requested_cpu_count"])
            for row in complete_rows
            if isinstance(row.get("requested_cpu_count"), int)
            and not isinstance(row.get("requested_cpu_count"), bool)
            and int(row["requested_cpu_count"]) > 0
        ]
        per_worker_memory_replacement_qualified = (
            memory_replacement_qualified
            and measurement_scopes
            and set(measurement_scopes) == {
                "one fresh child process for one analysis command"
            }
            and completed_requested_cpu_counts
            and max(completed_requested_cpu_counts) == 1
        )
        result[module_id] = {
            "module_id": module_id,
            "conservative_cpu_seconds_per_frame": max(planning_rates),
            "conservative_cpu_seconds_per_spatial_neighbor_pair": (
                max(spatial_cpu_rates) if spatial_cpu_rates else None
            ),
            "conservative_spatial_neighbor_pairs_per_selected_frame": (
                max(spatial_pairs_per_frame) if spatial_pairs_per_frame else None
            ),
            "conservative_explicit_geometry_evaluations_per_selected_frame": (
                max(geometry_evaluations_per_frame)
                if geometry_evaluations_per_frame else None
            ),
            "maximum_measured_spatial_neighbor_pair_count": max(
                (int(row["spatial_neighbor_pair_count"]) for row in spatial_rows),
                default=0,
            ),
            "maximum_measured_explicit_geometry_evaluation_count": max(
                (
                    int(row["explicit_geometry_evaluation_count"])
                    for row in complete_rows
                    if int(row.get("explicit_geometry_evaluation_count", 0)) > 0
                ),
                default=0,
            ),
            "maximum_measured_spatial_endpoint_count_per_system": (
                max(spatial_endpoint_counts) if spatial_endpoint_counts else 0
            ),
            "spatial_work_measurement_count": len(spatial_rows),
            "runtime_work_unit": (
                "spatial_neighbor_pairs_v1" if spatial_rows else
                "selected_physical_frames_v1"
            ),
            "conservative_fixed_cpu_seconds": affine_fixed,
            "conservative_affine_cpu_seconds_per_frame": affine_rate,
            "maximum_completed_cpu_seconds_per_frame": (
                max(completed_rates) if completed_rates else None
            ),
            "maximum_censored_cpu_seconds_per_frame_lower_bound": (
                max(timeout_rates) if timeout_rates else None
            ),
            "maximum_censored_wall_seconds_lower_bound": (
                max(
                    float(row["wall_seconds_lower_bound"])
                    for row in timeout_rows
                )
                if timeout_rows else None
            ),
            "censored_wall_lower_bound_points": (
                censored_wall_lower_bound_points
            ),
            "censored_timeout_safety_factor": timeout_safety,
            "maximum_resident_memory_mib": (
                max(planning_memories) if planning_memories else 0.0
            ),
            "maximum_observed_resident_memory_mib_all_records": (
                max(observed_memories) if observed_memories else 0.0
            ),
            "maximum_qualified_censored_resident_memory_mib": (
                max(qualified_censored_memories)
                if qualified_censored_memories else 0.0
            ),
            "memory_timeout_evidence_policy": (
                "completed executions plus single-CPU censored observations may "
                "set a planning lower bound; multi-CPU censored MaxRSS is retained "
                "as aggregate diagnostic evidence but is not treated as per-worker "
                "memory"
            ),
            "maximum_completed_resident_memory_mib": (
                max(completed_memories) if completed_memories else 0.0
            ),
            "maximum_source_report_size_bytes": max(
                (
                    int(row["source_report_size_bytes"])
                    for row in rows
                    if isinstance(row.get("source_report_size_bytes"), int)
                    and not isinstance(row.get("source_report_size_bytes"), bool)
                ),
                default=0,
            ),
            "maximum_measured_selected_frame_count": (
                max(int(row["selected_source_physical_frames"]) for row in complete_rows)
                if complete_rows else 0
            ),
            "maximum_measured_observation_count": (
                max(complete_observation_counts)
                if complete_observation_counts else 0
            ),
            "minimum_measured_observation_count": (
                min(complete_observation_counts)
                if complete_observation_counts else 0
            ),
            "maximum_timeout_target_frame_count": (
                max(int(row["selected_source_physical_frames"]) for row in timeout_rows)
                if timeout_rows else 0
            ),
            "measurement_count": len(rows),
            "complete_measurement_count": len(complete_rows),
            "censored_timeout_count": len(timeout_rows),
            "memory_replacement_qualified": memory_replacement_qualified,
            "memory_replacement_policy": (
                "replace_legacy_baseline_with_conservative_completed_measurement"
                if memory_replacement_qualified else
                "retain_legacy_baseline_and_use_measurement_as_lower_bound"
            ),
            "memory_replacement_minimum_complete_measurements": (
                MEMORY_REPLACEMENT_MIN_COMPLETE_MEASUREMENTS
            ),
            "completed_measurement_scopes": measurement_scopes,
            "maximum_completed_requested_cpu_count": (
                max(completed_requested_cpu_counts)
                if completed_requested_cpu_counts else None
            ),
            "per_worker_memory_replacement_qualified": (
                per_worker_memory_replacement_qualified
            ),
            "calibration_evidence_status": (
                "completed_plus_censored_lower_bound"
                if complete_rows and timeout_rows else
                "completed_execution" if complete_rows else
                "censored_lower_bound_only"
            ),
            "catalog_path": str(source),
            "catalog_sha256": _sha256(source),
        }
    return result
