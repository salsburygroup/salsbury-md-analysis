#!/usr/bin/env python3
"""Build a local, strided, connectivity-aware XYZ validation fixture read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from salsbury_md_analysis.atom_mapping import read_topology_atoms
from salsbury_md_analysis.coordinates import iter_coordinate_frames
from salsbury_md_analysis.manifests import load_json, resolve_manifest_path
from salsbury_md_analysis.periodic import PeriodicFrameProcessor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.stride < 1:
        parser.error("stride must be positive")
    project_path = arguments.project.expanduser().resolve(strict=True)
    project = load_json(project_path)
    make_whole_project = dict(project)
    make_whole_project["periodic_coordinate_policy"] = "make_whole"
    settings = dict(project["periodic_reconstruction"])
    settings.pop("maximum_anchor_displacement_angstrom", None)
    make_whole_project["periodic_reconstruction"] = settings
    system_path = resolve_manifest_path(str(project["system_manifest"]), project_path)
    system = load_json(system_path)
    output = arguments.output_directory.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for raw_system in system["systems"]:
        for replica in raw_system["replicas"]:
            topology = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology)
            processor = PeriodicFrameProcessor.from_replica(
                make_whole_project, replica, system_path, len(atoms)
            )
            for segment in replica["segments"]:
                trajectory = resolve_manifest_path(str(segment["trajectory"]), system_path)
                output_path = output / f"{raw_system['system_id']}__{replica['replica_id']}__{segment['segment_id']}.xyz"
                selected = 0
                observed = 0
                with output_path.open("w", encoding="utf-8") as handle:
                    for frame in iter_coordinate_frames(trajectory, str(project["coordinate_unit"])):
                        observed += 1
                        if frame.frame_index % arguments.stride:
                            continue
                        reconstructed = processor.process(
                            frame, f"{raw_system['system_id']}/{replica['replica_id']}/{segment['segment_id']}/{frame.frame_index}"
                        )
                        handle.write(f"{len(atoms)}\n")
                        handle.write(
                            f"source_frame_index={frame.frame_index} representation={reconstructed.coordinate_representation}\n"
                        )
                        for atom, coordinate in zip(atoms, reconstructed.coordinates_angstrom):
                            element = atom.element or atom.atom_name[:1] or "X"
                            handle.write(
                                f"{element} {coordinate[0]:.8f} {coordinate[1]:.8f} {coordinate[2]:.8f}\n"
                            )
                        selected += 1
                records.append({
                    "system_id": raw_system["system_id"], "replica_id": replica["replica_id"],
                    "segment_id": segment["segment_id"], "source_trajectory": str(trajectory),
                    "source_sha256": sha256(trajectory), "observed_frame_count": observed,
                    "selected_frame_count": selected, "source_frame_stride": arguments.stride,
                    "output_path": str(output_path), "output_sha256": sha256(output_path),
                    "periodic_reconstruction": processor.report(),
                })
    report = {
        "technical_status": "complete", "scientific_status": "not evaluated",
        "source_project": str(project_path), "source_project_sha256": sha256(project_path),
        "stride": arguments.stride, "records": records,
        "limitations": [
            "This derived fixture uses independent make-whole reconstruction at selected frames; the full structural scan separately validates continuous unwrapping on every source frame.",
            "The fixture is for technical module validation and does not establish convergence, populations, or scientific validity.",
        ],
    }
    report_path = output / "fixture_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "technical_status": "complete", "record_count": len(records),
        "selected_frame_count": sum(record["selected_frame_count"] for record in records),
        "manifest": str(report_path), "manifest_sha256": sha256(report_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
