#!/usr/bin/env python3
"""Verify completed planner-calibration reports and collect fit-ready evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ion_pair_count(report: dict[str, object], frames: int) -> int | None:
    if report.get("module_id") != "ion_atmosphere":
        return None
    settings = report.get("settings")
    if not isinstance(settings, dict):
        return None
    ion_groups = settings.get("ion_groups")
    target_groups = settings.get("target_groups")
    if not isinstance(ion_groups, list) or not isinstance(target_groups, list):
        return None
    ion_count = sum(
        len(row.get("atom_indices", []))
        for row in ion_groups if isinstance(row, dict)
    )
    unique_targets = {
        tuple(sorted(int(value) for value in row.get("atom_indices", [])))
        for row in target_groups if isinstance(row, dict)
    }
    return frames * ion_count * sum(len(indices) for indices in unique_targets)


def collect_matrix(matrix_path: Path) -> dict[str, object]:
    source = matrix_path.expanduser().resolve(strict=True)
    root = source.parent
    matrix = json.loads(source.read_text(encoding="utf-8"))
    if matrix.get("matrix_schema") != "salsbury-planner-calibration-matrix-v1":
        raise ValueError("invalid calibration matrix schema")
    rows = []
    for point in matrix.get("points", []):
        if not isinstance(point, dict):
            raise ValueError("calibration point must be an object")
        report_path = (root / str(point["report"])).resolve(strict=True)
        sidecar_path = (root / str(point["sidecar"])).resolve(strict=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if report.get("technical_status") != "complete" or report.get("error_count") != 0:
            raise ValueError(f"calibration report failed: {point['point_id']}")
        if sidecar.get("technical_status") != "complete":
            raise ValueError(f"calibration sidecar failed: {point['point_id']}")
        if sidecar.get("report_sha256") != sha256(report_path):
            raise ValueError(f"report hash mismatch: {point['point_id']}")
        evidence = sidecar.get("resource_evidence")
        resources = evidence.get("execution_resources") if isinstance(evidence, dict) else None
        if not isinstance(evidence, dict) or not isinstance(resources, dict):
            raise ValueError(f"resource evidence missing: {point['point_id']}")
        frames = int(evidence["selected_source_physical_frames"])
        if frames != int(point["selected_source_physical_frames"]):
            raise ValueError(f"frame count mismatch: {point['point_id']}")
        accounting = report.get("observation_accounting")
        accounting = accounting if isinstance(accounting, dict) else {}
        row = {
            "point_id": point["point_id"],
            "label": point["label"],
            "module_id": report["module_id"],
            "topology_atom_count": point["topology_atom_count"],
            "selected_source_physical_frames": frames,
            "topology_atom_frame_count": int(point["topology_atom_count"]) * frames,
            "total_cpu_seconds": resources["total_cpu_seconds"],
            "wall_seconds": resources["wall_seconds"],
            "maximum_resident_memory_mib": resources["maximum_resident_memory_mib"],
            "requested_cpu_count": resources["requested_cpu_count"],
            "measurement_scope": resources["measurement_scope"],
            "stderr_nonempty": resources["stderr_nonempty"],
            "report_sha256": sidecar["report_sha256"],
            "sidecar_sha256": sha256(sidecar_path),
            "project_manifest_sha256": point["project_manifest_sha256"],
            "system_manifest_sha256": point["system_manifest_sha256"],
            "scientific_status": report.get("scientific_status", "not evaluated"),
        }
        for field in (
            "conceptual_candidate_frame_count",
            "spatial_neighbor_pair_count",
            "explicit_geometry_evaluation_count",
            "present_event_count",
            "maximum_spatial_endpoint_count_per_system",
        ):
            value = evidence.get(field, accounting.get(field))
            if isinstance(value, int) and not isinstance(value, bool):
                row[field] = value
        ion_pairs = _ion_pair_count(report, frames)
        if ion_pairs is not None:
            row["ion_target_minimum_image_pair_count"] = ion_pairs
        rows.append(row)
    return {
        "evidence_schema": "salsbury-planner-calibration-evidence-matrix-v1",
        "technical_status": "complete",
        "scientific_status": "runtime evidence only; scientific validity not evaluated",
        "matrix_sha256": sha256(source),
        "point_count": len(rows),
        "unexpected_error_count": sum(bool(row["stderr_nonempty"]) for row in rows),
        "points": rows,
    }


def collect_matrices(matrix_paths: list[Path]) -> dict[str, object]:
    if not matrix_paths:
        raise ValueError("at least one calibration matrix is required")
    collected = [collect_matrix(path) for path in matrix_paths]
    rows = [row for result in collected for row in result["points"]]
    point_ids = [str(row["point_id"]) for row in rows]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("calibration matrices contain duplicate point identifiers")
    return {
        "evidence_schema": "salsbury-planner-calibration-evidence-matrix-v1",
        "technical_status": "complete",
        "scientific_status": "runtime evidence only; scientific validity not evaluated",
        "matrix_sha256s": [result["matrix_sha256"] for result in collected],
        "matrix_count": len(collected),
        "point_count": len(rows),
        "unexpected_error_count": sum(
            int(result["unexpected_error_count"]) for result in collected
        ),
        "points": sorted(
            rows,
            key=lambda row: (
                str(row["module_id"]),
                int(row["topology_atom_count"]),
                int(row["selected_source_physical_frames"]),
            ),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect_matrices(args.matrix)
    if result["unexpected_error_count"]:
        raise ValueError("one or more calibration commands wrote unexpected stderr")
    output = args.output.expanduser().resolve(strict=False)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
