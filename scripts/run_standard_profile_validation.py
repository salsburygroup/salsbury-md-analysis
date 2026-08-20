#!/usr/bin/env python3
"""Run every executable standard-MD project module and retain bounded evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from salsbury_md_analysis.alternative_clustering import alternative_clustering_project_safe
from salsbury_md_analysis.clustering import clustering_hdbscan_project_safe, clustering_imwkmeans_project_safe, clustering_kmeans_project_safe
from salsbury_md_analysis.convergence import convergence_uncertainty_project_safe
from salsbury_md_analysis.correlation_networks import correlation_networks_project_safe
from salsbury_md_analysis.dccm import dccm_project_safe
from salsbury_md_analysis.dihedrals import dihedral_distributions_project_safe
from salsbury_md_analysis.grouped_ml import grouped_ml_project_safe
from salsbury_md_analysis.grouped_regularized_classification import grouped_regularized_classification_project_safe
from salsbury_md_analysis.hydrogen_bonds import hydrogen_bonds_project_safe
from salsbury_md_analysis.hydrogen_bond_discovery import hydrogen_bond_discovery_project_safe
from salsbury_md_analysis.information import generalized_correlation_and_information_project_safe
from salsbury_md_analysis.information_dynamics import information_dynamics_project_safe
from salsbury_md_analysis.integrated import integrated_comparison_project_safe
from salsbury_md_analysis.msm import markov_state_models_project_safe
from salsbury_md_analysis.observables import optional_observables_project_safe
from salsbury_md_analysis.pca import common_pca_project_safe, individual_pca_project_safe
from salsbury_md_analysis.pca_fes import pca_fes_basins_project_safe
from salsbury_md_analysis.representative_frames import representative_frames_project_safe
from salsbury_md_analysis.rmsd_rg import replica_rmsd_rg_project_safe
from salsbury_md_analysis.rmsf import pooled_rmsf_project_safe
from salsbury_md_analysis.sasa import solvent_accessible_surface_area_project_safe
from salsbury_md_analysis.secondary_structure import secondary_structure_project_safe
from salsbury_md_analysis.structural_qc import structural_qc_project_safe
from salsbury_md_analysis.tica import time_lagged_independent_component_analysis_project_safe
from salsbury_md_analysis.trajectory_features import trajectory_features_project_safe
from salsbury_md_analysis.scalar_threshold_states import scalar_threshold_states_project_safe


RUNNERS = {
    "structural_integrity_qc": structural_qc_project_safe,
    "replica_rmsd_rg": replica_rmsd_rg_project_safe,
    "pooled_rmsf": pooled_rmsf_project_safe,
    "dccm": dccm_project_safe,
    "generalized_correlation_and_information": generalized_correlation_and_information_project_safe,
    "information_dynamics": information_dynamics_project_safe,
    "correlation_networks": correlation_networks_project_safe,
    "individual_pca": individual_pca_project_safe,
    "common_pca": common_pca_project_safe,
    "trajectory_features": trajectory_features_project_safe,
    "scalar_threshold_states": scalar_threshold_states_project_safe,
    "time_lagged_independent_component_analysis": time_lagged_independent_component_analysis_project_safe,
    "pca_fes_basins": pca_fes_basins_project_safe,
    "clustering_kmeans": clustering_kmeans_project_safe,
    "clustering_hdbscan": clustering_hdbscan_project_safe,
    "clustering_imwkmeans": clustering_imwkmeans_project_safe,
    "alternative_clustering": alternative_clustering_project_safe,
    "representative_frames": representative_frames_project_safe,
    "markov_state_models": markov_state_models_project_safe,
    "dihedral_distributions": dihedral_distributions_project_safe,
    "hydrogen_bonds": hydrogen_bonds_project_safe,
    "hydrogen_bond_discovery": hydrogen_bond_discovery_project_safe,
    "grouped_ml": grouped_ml_project_safe,
    "grouped_regularized_classification": grouped_regularized_classification_project_safe,
    "secondary_structure": secondary_structure_project_safe,
    "solvent_accessible_surface_area": solvent_accessible_surface_area_project_safe,
    "optional_observables": optional_observables_project_safe,
    "convergence_uncertainty": convergence_uncertainty_project_safe,
    "integrated_comparison": integrated_comparison_project_safe,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--hash-content", action="store_true")
    arguments = parser.parse_args()
    project = arguments.project.expanduser().resolve(strict=True)
    output = arguments.output_directory.expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for module_id, runner in RUNNERS.items():
        started = time.monotonic()
        print(f"START {module_id}", flush=True)
        try:
            report = runner(project, hash_content=arguments.hash_content)
        except Exception as exc:  # preserve unexpected crashes as validation evidence
            report = {
                "module_id": module_id, "technical_status": "crashed",
                "scientific_status": "not evaluated", "error_count": 1,
                "warning_count": 0,
                "issues": [{"severity": "error", "code": "UNHANDLED_VALIDATION_EXCEPTION", "message": f"{type(exc).__name__}: {exc}"}],
            }
        elapsed = time.monotonic() - started
        report_path = output / f"{module_id}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        record = {
            "module_id": module_id,
            "technical_status": report.get("technical_status"),
            "scientific_status": report.get("scientific_status"),
            "error_count": report.get("error_count"),
            "warning_count": report.get("warning_count"),
            "elapsed_seconds": elapsed,
            "report_path": str(report_path),
            "report_sha256": sha256(report_path),
        }
        summary.append(record)
        print(f"END {module_id} {record['technical_status']} {elapsed:.3f}s", flush=True)
    payload = {
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_path": str(project), "project_sha256": sha256(project),
        "module_count": len(summary), "modules": summary,
        "complete_count": sum(row["technical_status"] == "complete" for row in summary),
        "failed_count": sum(row["technical_status"] == "failed" for row in summary),
        "crashed_count": sum(row["technical_status"] == "crashed" for row in summary),
    }
    summary_path = output / "validation_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**{key: payload[key] for key in ("module_count", "complete_count", "failed_count", "crashed_count")}, "summary_path": str(summary_path), "summary_sha256": sha256(summary_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
