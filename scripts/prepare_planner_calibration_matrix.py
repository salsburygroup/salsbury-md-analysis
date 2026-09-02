#!/usr/bin/env python3
"""Prepare immutable one-replica projects for planner performance calibration.

The command reads an existing project and system manifest, selects one declared
replica, and writes exact integer-stride variants for the requested frame
budgets.  It never copies or rewrites trajectory, topology, or connectivity
inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from salsbury_md_analysis.frame_sampling import (
    integer_stride_for_budget,
    integer_stride_selected_count,
    source_frame_count,
)
from salsbury_md_analysis.manifests import load_json, resolve_manifest_path
from salsbury_md_analysis.oligomer_symmetry import read_topology_atoms


MODULE_DEFINITION_KEYS = {
    "structural_integrity_qc": "structural_qc",
    "hydrogen_bond_discovery": "hydrogen_bond_discovery",
    "ion_atmosphere": "ion_atmosphere",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_budgets(raw: str) -> list[int]:
    try:
        values = sorted({int(value.strip()) for value in raw.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frame budgets must be integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("frame budgets must be positive")
    return values


def _source_frames(replica: dict[str, object], system_path: Path, unit: str) -> int:
    segments = replica.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("selected replica has no trajectory segments")
    return sum(
        source_frame_count(
            resolve_manifest_path(str(segment["trajectory"]), system_path), unit
        )
        for segment in segments
        if isinstance(segment, dict)
    )


def prepare_matrix(
    project_path: Path,
    system_path: Path,
    *,
    system_id: str,
    replica_id: str,
    module_id: str,
    frame_budgets: list[int],
    output: Path,
    label: str,
) -> dict[str, object]:
    source_project_path = project_path.expanduser().resolve(strict=True)
    source_system_path = system_path.expanduser().resolve(strict=True)
    output_path = output.expanduser().resolve(strict=False)
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}")
    project = load_json(source_project_path)
    system = load_json(source_system_path)
    systems = [
        row for row in system.get("systems", [])
        if isinstance(row, dict) and row.get("system_id") == system_id
    ]
    if len(systems) != 1:
        raise ValueError(f"system_id must resolve exactly once: {system_id}")
    replicas = [
        row for row in systems[0].get("replicas", [])
        if isinstance(row, dict) and row.get("replica_id") == replica_id
    ]
    if len(replicas) != 1:
        raise ValueError(f"replica_id must resolve exactly once: {replica_id}")
    replica = replicas[0]
    definition_key = MODULE_DEFINITION_KEYS[module_id]
    definitions = project.get("definitions")
    if not isinstance(definitions, dict) or not isinstance(
        definitions.get(definition_key), dict
    ):
        raise ValueError(f"source project lacks definitions.{definition_key}")
    coordinate_unit = str(project.get("coordinate_unit", ""))
    if coordinate_unit not in {"angstrom", "nanometer"}:
        raise ValueError("source project has no supported coordinate_unit")
    source_frames = _source_frames(replica, source_system_path, coordinate_unit)
    topology_path = resolve_manifest_path(str(replica["topology"]), source_system_path)
    _, topology_atoms = read_topology_atoms(topology_path)
    atom_count = len(topology_atoms)
    if atom_count <= 0:
        raise ValueError("selected topology contains no atoms")

    output_path.mkdir(parents=True)
    points: list[dict[str, object]] = []
    for budget in frame_budgets:
        stride = integer_stride_for_budget([source_frames], budget)
        selected_frames = integer_stride_selected_count(source_frames, stride)
        point_id = f"{label}-{module_id}-{selected_frames}-frames"
        point_dir = output_path / point_id
        point_dir.mkdir()

        point_system = deepcopy(system)
        chosen_system = deepcopy(systems[0])
        chosen_system["replicas"] = [deepcopy(replica)]
        point_system["systems"] = [chosen_system]
        system_output = point_dir / "system.json"
        system_output.write_text(
            json.dumps(point_system, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        point_project = deepcopy(project)
        point_project["project_id"] = point_id
        point_project["system_manifest"] = "system.json"
        point_project["analysis_output_root"] = "results"
        point_project["reference_system"] = system_id
        point_project["reference_structure"] = str(topology_path)
        connectivity = replica.get("connectivity")
        if connectivity is not None:
            point_project["reference_connectivity"] = str(
                resolve_manifest_path(str(connectivity), source_system_path)
            )
        point_project["requested_modules"] = [module_id]
        definition = point_project["definitions"][definition_key]
        definition["frame_stride"] = 1
        definition["frame_selection"] = {
            "mode": "integer_stride_per_replica_v1",
            "stride": stride,
        }
        if module_id == "ion_atmosphere":
            definition["maximum_frames"] = selected_frames
        if module_id == "structural_integrity_qc":
            definition.pop("parallel_execution", None)
            checkpointing = definition.get("checkpointing")
            if isinstance(checkpointing, dict):
                checkpointing["enabled"] = False
        project_output = point_dir / "project.json"
        project_output.write_text(
            json.dumps(point_project, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        point = {
            "point_id": point_id,
            "label": label,
            "module_id": module_id,
            "system_id": system_id,
            "replica_id": replica_id,
            "topology_atom_count": atom_count,
            "source_frame_count": source_frames,
            "requested_maximum_frames_per_replica": budget,
            "resolved_integer_stride": stride,
            "selected_source_physical_frames": selected_frames,
            "topology_atom_frame_count": atom_count * selected_frames,
            "project_manifest": str(project_output.relative_to(output_path)),
            "project_manifest_sha256": sha256(project_output),
            "system_manifest": str(system_output.relative_to(output_path)),
            "system_manifest_sha256": sha256(system_output),
            "report": str((point_dir / "report.json").relative_to(output_path)),
            "sidecar": str(
                (point_dir / "report.json.summary.json").relative_to(output_path)
            ),
        }
        (point_dir / "point.json").write_text(
            json.dumps(point, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        points.append(point)

    matrix = {
        "matrix_schema": "salsbury-planner-calibration-matrix-v1",
        "technical_status": "prepared",
        "scientific_status": "not evaluated",
        "source_project_manifest_sha256": sha256(source_project_path),
        "source_system_manifest_sha256": sha256(source_system_path),
        "input_mutation_policy": "read_only_source_inputs",
        "execution_model": "one_process_for_one_replica",
        "point_count": len(points),
        "points": points,
    }
    matrix_path = output_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--replica-id", required=True)
    parser.add_argument("--module-id", choices=sorted(MODULE_DEFINITION_KEYS), required=True)
    parser.add_argument("--frames-per-replica", type=_positive_budgets, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = prepare_matrix(
        args.project,
        args.system,
        system_id=args.system_id,
        replica_id=args.replica_id,
        module_id=args.module_id,
        frame_budgets=args.frames_per_replica,
        output=args.output,
        label=args.label,
    )
    print(json.dumps(matrix, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
