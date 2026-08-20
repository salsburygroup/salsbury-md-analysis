#!/usr/bin/env python3
"""Exercise standalone extension methods on a bounded long-trajectory fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from salsbury_md_analysis.atom_mapping import read_topology_atoms
from salsbury_md_analysis.coordinates import iter_coordinate_frames
from salsbury_md_analysis.geometry import apply_transform, best_fit_transform
from salsbury_md_analysis.hydrogen_bond_patterns import (
    encode_bond_patterns,
    hdbscan_jaccard,
    pam_jaccard,
)
from salsbury_md_analysis.hydrogen_bonds import hydrogen_bond_present
from salsbury_md_analysis.manifests import load_json, resolve_manifest_path
from salsbury_md_analysis.representative_structures import representative_structures
from salsbury_md_analysis.rmsf_inference import rmsf_permutation_test


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected(coordinates, indices):
    return tuple(coordinates[index] for index in indices)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("pooled_rmsf_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    project_path = arguments.project.expanduser().resolve(strict=True)
    rmsf_path = arguments.pooled_rmsf_report.expanduser().resolve(strict=True)
    project = load_json(project_path)
    system_path = resolve_manifest_path(str(project["system_manifest"]), project_path)
    system = load_json(system_path)
    settings = project["definitions"]["hydrogen_bonds"]
    feature = settings["features"][0]

    frame_bonds = []
    aligned_coordinates = []
    frame_lineage = []
    reference_alignment = None
    alignment_indices = None
    representative_indices = None
    for raw_system in system["systems"]:
        for replica in raw_system["replicas"]:
            topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
            _, atoms = read_topology_atoms(topology_path)
            if alignment_indices is None:
                alignment_indices = [index for index, atom in enumerate(atoms) if atom.atom_name == "CA"]
                representative_indices = alignment_indices[::20]
                if len(alignment_indices) < 3 or not representative_indices:
                    raise ValueError("the long fixture requires at least three CA atoms")
            for segment in replica["segments"]:
                trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                for frame in iter_coordinate_frames(trajectory_path, str(project["coordinate_unit"])):
                    coordinates = frame.coordinates_angstrom
                    mobile_alignment = selected(coordinates, alignment_indices)
                    if reference_alignment is None:
                        reference_alignment = mobile_alignment
                    transform = best_fit_transform(mobile_alignment, reference_alignment)
                    aligned_coordinates.append(
                        apply_transform(selected(coordinates, representative_indices), transform)
                    )
                    present, _, _ = hydrogen_bond_present(
                        coordinates[int(feature["donor_atom_index"])],
                        coordinates[int(feature["hydrogen_atom_index"])],
                        coordinates[int(feature["acceptor_atom_index"])],
                        float(settings["maximum_donor_acceptor_distance_angstrom"]),
                        float(settings["minimum_donor_hydrogen_acceptor_angle_degrees"]),
                    )
                    frame_bonds.append({str(feature["feature_id"])} if present else set())
                    frame_lineage.append({
                        "replica_id": str(replica["replica_id"]),
                        "segment_id": str(segment["segment_id"]),
                        "frame_index": frame.frame_index,
                    })

    bond_ids, patterns = encode_bond_patterns(frame_bonds, [str(feature["feature_id"])])
    hydrogen_patterns = {
        "bond_ids": bond_ids,
        "frame_count": int(patterns.shape[0]),
        "present_frame_count": int(patterns.sum()),
        "pam": pam_jaccard(patterns, 1),
        "hdbscan": hdbscan_jaccard(patterns, minimum_cluster_size=5, minimum_samples=3),
    }
    representatives = representative_structures(aligned_coordinates)
    representatives.pop("arithmetic_mean_coordinates")
    representatives["representative_atom_count"] = len(representative_indices)
    representatives["closest_to_mean_lineage"] = frame_lineage[
        int(representatives["closest_to_mean_frame_index"])
    ]
    representatives["medoid_lineage"] = frame_lineage[
        int(representatives["medoid_frame_index"])
    ]

    rmsf = load_json(rmsf_path)
    blocks = []
    for replica in rmsf["systems"][0]["replicas"]:
        blocks.extend(replica["segments"][0]["time_blocks"])
    early = [
        block["rmsf_angstrom_by_common_atom_index"][:24]
        for block in blocks if int(block["block_index"]) in {0, 1}
    ]
    late = [
        block["rmsf_angstrom_by_common_atom_index"][:24]
        for block in blocks if int(block["block_index"]) in {3, 4}
    ]
    rmsf_inference = rmsf_permutation_test(early, late, random_seed=17)
    report = {
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "source_project": str(project_path),
        "source_project_sha256": sha256(project_path),
        "pooled_rmsf_report": str(rmsf_path),
        "pooled_rmsf_report_sha256": sha256(rmsf_path),
        "long_trajectory_frame_count": len(frame_lineage),
        "long_trajectory_replica_count": len(system["systems"][0]["replicas"]),
        "modules": {
            "hydrogen_bond_patterns": hydrogen_patterns,
            "representative_structures": representatives,
            "rmsf_permutation_inference": rmsf_inference,
        },
        "limitations": [
            "The early-versus-late RMSF permutation is a technical exercise of the inference engine, not an exchangeability claim or scientific comparison.",
            "The representative-structure calculation uses globally CA-aligned, strided long-trajectory frames and a bounded every-20th-CA representation.",
            "The one explicitly declared hydrogen-bond feature is sufficient to validate pattern encoding and clustering, not to characterize the full hydrogen-bond network.",
        ],
    }
    output = arguments.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "technical_status": report["technical_status"],
        "module_count": len(report["modules"]),
        "frame_count": len(frame_lineage),
        "output": str(output),
        "output_sha256": sha256(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
