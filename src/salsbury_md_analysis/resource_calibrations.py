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
        "computer_hostname": resources.get("computer_hostname"),
        "platform": resources.get("platform"),
        "requested_cpu_count": resources.get("requested_cpu_count"),
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
    evidence_keys = []
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
        evidence_keys.append(key)
    if len(evidence_keys) != len(set(evidence_keys)):
        raise ResourceCalibrationError(
            "duplicate source evidence appears in the combined calibration catalog"
        )
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
        "base_catalogs": base_provenance,
        "entries": entries,
        "planner_policy": {
            "cpu_rate": (
                "maximum of completed CPU rates and separately labeled "
                "right-censored timeout lower bounds after the configured "
                "timeout safety factor"
            ),
            "memory": "maximum observed RSS with a planner safety factor",
            "frame_coverage": (
                "maximum technically completed selected physical frames only; "
                "timeout target coverage is retained but never counted complete"
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
        planning_rates = list(completed_rates)
        planning_rates.extend(rate * timeout_safety for rate in timeout_rates)
        memories = [
            float(row["maximum_resident_memory_mib"])
            for row in rows if row.get("maximum_resident_memory_mib") is not None
        ]
        result[module_id] = {
            "module_id": module_id,
            "conservative_cpu_seconds_per_frame": max(planning_rates),
            "maximum_completed_cpu_seconds_per_frame": (
                max(completed_rates) if completed_rates else None
            ),
            "maximum_censored_cpu_seconds_per_frame_lower_bound": (
                max(timeout_rates) if timeout_rates else None
            ),
            "censored_timeout_safety_factor": timeout_safety,
            "maximum_resident_memory_mib": max(memories) if memories else 0.0,
            "maximum_measured_selected_frame_count": (
                max(int(row["selected_source_physical_frames"]) for row in complete_rows)
                if complete_rows else 0
            ),
            "maximum_measured_observation_count": (
                max(int(row["symmetry_expanded_observations"]) for row in complete_rows)
                if complete_rows else 0
            ),
            "maximum_timeout_target_frame_count": (
                max(int(row["selected_source_physical_frames"]) for row in timeout_rows)
                if timeout_rows else 0
            ),
            "measurement_count": len(rows),
            "complete_measurement_count": len(complete_rows),
            "censored_timeout_count": len(timeout_rows),
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
