#!/usr/bin/env python3
"""Run one project module in isolation and retain reproducible resource evidence.

This harness is intentionally environment-neutral.  Project and system manifests
hold the data lineage; the harness never rewrites an input or silently changes a
frame-selection contract.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from salsbury_md_analysis.dccm import dccm_project_safe
from salsbury_md_analysis.energetic_network_embeddings import (
    energetic_network_embeddings_project_safe,
)
from salsbury_md_analysis.dihedrals import dihedral_distributions_project_safe
from salsbury_md_analysis.hydrogen_bond_discovery import hydrogen_bond_discovery_project_safe
from salsbury_md_analysis.hydrogen_bonds import hydrogen_bonds_project_safe
from salsbury_md_analysis.hydration_density import hydration_density_channels_project_safe
from salsbury_md_analysis.ion_geometry import ion_coordination_geometry_project_safe
from salsbury_md_analysis.ion_atmosphere import ion_atmosphere_project_safe
from salsbury_md_analysis.nucleic_acid_geometry import nucleic_acid_geometry_project_safe
from salsbury_md_analysis.nucleic_acid_structure import nucleic_acid_structure_project_safe
from salsbury_md_analysis.multivalent_bridges import (
    multivalent_molecular_bridges_project_safe,
)
from salsbury_md_analysis.observables import optional_observables_project_safe
from salsbury_md_analysis.pca import common_pca_project_safe, individual_pca_project_safe
from salsbury_md_analysis.pocket_dynamics import ensemble_pocket_dynamics_project_safe
from salsbury_md_analysis.rdf import radial_distribution_functions_project_safe
from salsbury_md_analysis.rmsd_rg import replica_rmsd_rg_project_safe
from salsbury_md_analysis.rmsf import pooled_rmsf_project_safe
from salsbury_md_analysis.sasa import solvent_accessible_surface_area_project_safe
from salsbury_md_analysis.secondary_structure import secondary_structure_project_safe
from salsbury_md_analysis.structural_qc import structural_qc_project_safe
from salsbury_md_analysis.trajectory_features import trajectory_features_project_safe
from salsbury_md_analysis.water_mediated_hydrogen_bonds import (
    water_mediated_hydrogen_bond_networks_project_safe,
)


Runner = Callable[[Path, bool], dict[str, object]]

RUNNERS: Mapping[str, Runner] = {
    "structural_integrity_qc": structural_qc_project_safe,
    "replica_rmsd_rg": replica_rmsd_rg_project_safe,
    "pooled_rmsf": pooled_rmsf_project_safe,
    "dccm": dccm_project_safe,
    "individual_pca": individual_pca_project_safe,
    "common_pca": common_pca_project_safe,
    "trajectory_features": trajectory_features_project_safe,
    "dihedral_distributions": dihedral_distributions_project_safe,
    "hydrogen_bonds": hydrogen_bonds_project_safe,
    "hydrogen_bond_discovery": hydrogen_bond_discovery_project_safe,
    "water_mediated_hydrogen_bond_networks": (
        water_mediated_hydrogen_bond_networks_project_safe
    ),
    "multivalent_molecular_bridges": multivalent_molecular_bridges_project_safe,
    "hydration_density_channels": hydration_density_channels_project_safe,
    "ensemble_pocket_dynamics": ensemble_pocket_dynamics_project_safe,
    "energetic_network_embeddings": energetic_network_embeddings_project_safe,
    "secondary_structure": secondary_structure_project_safe,
    "nucleic_acid_structure": nucleic_acid_structure_project_safe,
    "nucleic_acid_geometry": nucleic_acid_geometry_project_safe,
    "ion_coordination_geometry": ion_coordination_geometry_project_safe,
    "ion_atmosphere": ion_atmosphere_project_safe,
    "optional_observables": optional_observables_project_safe,
    "solvent_accessible_surface_area": solvent_accessible_surface_area_project_safe,
    "radial_distribution_functions": radial_distribution_functions_project_safe,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_counts(value: object, result: dict[str, list[int]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                isinstance(child, int)
                and not isinstance(child, bool)
                and (
                    key.endswith("frame_count")
                    or key.endswith("observation_count")
                    or key.endswith("candidate_count")
                    or key.endswith("atom_count")
                )
            ):
                result.setdefault(key, []).append(child)
            _collect_counts(child, result)
    elif isinstance(value, list):
        for child in value:
            _collect_counts(child, result)


def _environment() -> dict[str, object]:
    try:
        import scipy

        scipy_version: str | None = scipy.__version__
    except ImportError:
        scipy_version = None
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy_version,
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURM_NODELIST",
            )
        },
    }


def _frame_coverage(report: Mapping[str, object]) -> dict[str, object]:
    """Extract the module's canonical estimator and optional projection coverage."""

    coverage: dict[str, object] = {}
    estimator = report.get("basis_frame_selection", report.get("frame_selection"))
    if isinstance(estimator, dict):
        for source_key, target_key in (
            ("source_frame_count", "source_frame_count"),
            ("selected_frame_count", "estimator_selected_frame_count"),
            ("coverage_fraction", "estimator_coverage_fraction"),
            ("mode", "estimator_selection_mode"),
            ("resolved_mode", "estimator_resolved_mode"),
        ):
            if source_key in estimator:
                coverage[target_key] = estimator[source_key]
    projection = report.get("projection_frame_selection")
    if isinstance(projection, dict):
        for source_key, target_key in (
            ("selected_frame_count", "projection_selected_frame_count"),
            ("coverage_fraction", "projection_coverage_fraction"),
            ("mode", "projection_selection_mode"),
            ("resolved_mode", "projection_resolved_mode"),
        ):
            if source_key in projection:
                coverage[target_key] = projection[source_key]
    direct = report.get("evaluated_frame_count")
    if (
        "estimator_selected_frame_count" not in coverage
        and isinstance(direct, int)
        and not isinstance(direct, bool)
    ):
        coverage["estimator_selected_frame_count"] = direct
    if "estimator_selected_frame_count" not in coverage:
        selected = 0
        source = 0

        def visit(value: object) -> None:
            nonlocal selected, source
            if isinstance(value, dict):
                if "segment_id" in value:
                    evaluated = value.get(
                        "physical_evaluated_frame_count",
                        value.get("evaluated_frame_count"),
                    )
                    observed = value.get("observed_frame_count")
                    if isinstance(evaluated, int) and not isinstance(evaluated, bool):
                        selected += evaluated
                    if isinstance(observed, int) and not isinstance(observed, bool):
                        source += observed
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(report)
        if selected:
            coverage["estimator_selected_frame_count"] = selected
            coverage["source_frame_count"] = source or selected
            coverage["estimator_coverage_fraction"] = selected / (source or selected)
            coverage["estimator_selection_mode"] = (
                "all_frames" if not source or selected == source
                else "reported_segment_selection"
            )
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("module_id", choices=sorted(RUNNERS))
    parser.add_argument("project", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--hash-content", action="store_true")
    arguments = parser.parse_args()

    project = arguments.project.expanduser().resolve(strict=True)
    output = arguments.output_directory.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    wall_start = time.perf_counter()
    process_start = time.process_time()
    report = RUNNERS[arguments.module_id](project, arguments.hash_content)
    process_seconds = time.process_time() - process_start
    wall_seconds = time.perf_counter() - wall_start

    report_path = output / f"{arguments.module_id}.report.json.gz"
    with gzip.open(report_path, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    collected: dict[str, list[int]] = {}
    _collect_counts(report, collected)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    maximum_rss_kib = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        maximum_rss_kib //= 1024
    evidence = {
        "benchmark_schema": "salsbury-frame-coverage-benchmark-v1",
        "module_id": arguments.module_id,
        "started_utc": started_utc,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_path": str(project),
        "project_sha256": sha256(project),
        "hash_content": arguments.hash_content,
        "technical_status": report.get("technical_status"),
        "scientific_status": report.get("scientific_status"),
        "error_count": report.get("error_count"),
        "warning_count": report.get("warning_count"),
        "issue_codes": sorted(
            {
                str(row.get("code"))
                for row in report.get("issues", [])
                if isinstance(row, dict) and row.get("code")
            }
        ),
        "observed_counts": {
            key: {
                "occurrence_count": len(values),
                **({"values": values} if len(values) <= 40 else {}),
                **(
                    {
                        "value_frequencies": {
                            str(value): values.count(value)
                            for value in sorted(set(values))
                        }
                    }
                    if len(values) > 40 and len(set(values)) <= 40
                    else {}
                ),
                "sum": sum(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for key, values in sorted(collected.items())
        },
        "frame_coverage": _frame_coverage(report),
        "resources": {
            "wall_seconds": wall_seconds,
            "process_seconds": process_seconds,
            "maximum_rss_kib": maximum_rss_kib,
            "input_blocks": int(usage.ru_inblock),
            "output_blocks": int(usage.ru_oublock),
        },
        "environment": _environment(),
        "report_path": str(report_path),
        "report_size_bytes": report_path.stat().st_size,
        "report_sha256": sha256(report_path),
    }
    evidence_path = output / f"{arguments.module_id}.benchmark.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return 0 if report.get("technical_status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
