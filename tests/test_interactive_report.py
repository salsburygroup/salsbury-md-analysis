import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.interactive_report import (
    InteractiveReportError,
    build_interactive_report,
)


class InteractiveReportTests(unittest.TestCase):
    def test_dashboard_indexes_every_default_off_method_and_spatial_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            module_ids = (
                "perturbation_response_dynamics", "trajectory_reweighting",
                "allosteric_pathways", "multivalent_molecular_bridges",
                "energetic_network_embeddings",
                "reactive_path_ensembles", "interaction_fingerprints",
                "spatial_interaction_ensembles", "interaction_persistence",
                "random_feature_koopman",
                "helical_mechanics", "hydration_density_channels",
                "ensemble_pocket_dynamics",
            )
            for module_id in module_ids:
                path = (
                    root / "helical-mechanics-availability.json"
                    if module_id == "helical_mechanics"
                    else root / "results" / module_id / "report.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "module_id": module_id, "technical_status": "complete",
                    "scientific_status": "not evaluated", "evaluated_frame_count": 20,
                    "issues": [],
                }
                if module_id == "helical_mechanics":
                    payload.update({
                        "availability_status": "not_available",
                        "availability_reason": "no_duplex_dna_or_rna",
                    })
                if module_id == "hydration_density_channels":
                    payload["density_projections_xy"] = [{
                        "system_id": "control", "species": "water",
                        "matrix": [[0.0, 0.5], [1.0, 0.0]],
                        "normalization": "mean frame occupancy summed over z",
                    }]
                if module_id == "ensemble_pocket_dynamics":
                    payload["pocket_clusters"] = [{
                        "pocket_cluster_id": "pocket-cluster-1",
                        "per_system_occupancy": [{
                            "system_id": "control", "occupancy_fraction": 0.4,
                        }],
                    }]
                if module_id == "interaction_persistence":
                    payload["feature_persistence_summaries"] = [{
                        "feature_id": "feature-1", "system_id": "control",
                        "gap_tolerance_observations": 0,
                        "persistence_summary_gate": "passed",
                        "time_unit": "ps", "complete_event_count": 3,
                        "complete_event_duration_summary": {"median": 12.0},
                    }]
                if module_id == "energetic_network_embeddings":
                    payload["availability_status"] = "available"
                    payload["pairwise_system_comparisons"] = [{
                        "system_i": "control", "system_j": "variant",
                        "residue_distances": [{
                            "node_id": "A:10:ALA",
                            "summed_wasserstein_distance": 1.25,
                        }],
                    }]
                if module_id == "spatial_interaction_ensembles":
                    payload["spatial_ensemble_summaries"] = [{
                        "system_id": "control", "superfeature_id": "site-1",
                        "spatial_summary_gate": "passed",
                        "point_observation_count": 20,
                        "rms_radius_angstrom": 1.2,
                        "centroid_angstrom": [1.0, 2.0, 3.0],
                        "mode_inference_status": "no_gated_multimodal_candidate",
                    }]
                    payload["point_observations"] = [{
                        "system_id": "control", "superfeature_id": "site-1",
                        "coordinate_angstrom": [1.0, 2.0, 3.0],
                    }]
                if module_id == "random_feature_koopman":
                    payload["hyperparameter_candidates"] = [{
                        "random_feature_count": 32, "bandwidth_scale": 1.0,
                        "mean_seed_heldout_vamp_e": 0.8, "eligible": True,
                    }]
                path.write_text(json.dumps(payload), encoding="utf-8")
            result = build_interactive_report(root)
            html_text = Path(result["index_path"]).read_text(encoding="utf-8")
            embedded = html_text.split(
                '<script id="report-data" type="application/json">', 1
            )[1].split("</script>", 1)[0]
            data = json.loads(embedded)
        by_module = {row["module_id"]: row for row in data["reports"]}
        for module_id in module_ids:
            self.assertIn(
                "experimental_method_summary",
                {visual["kind"] for visual in by_module[module_id]["visuals"]},
            )
        self.assertEqual(
            by_module["helical_mechanics"]["visuals"][0]["availability_status"],
            "not_available",
        )
        self.assertIn(
            "spatial_density",
            {row["kind"] for row in by_module["hydration_density_channels"]["visuals"]},
        )
        self.assertIn(
            "experimental_metric_bars",
            {row["kind"] for row in by_module["ensemble_pocket_dynamics"]["visuals"]},
        )
        for module_id in (
            "energetic_network_embeddings", "interaction_persistence",
            "random_feature_koopman",
        ):
            self.assertIn(
                "experimental_metric_bars",
                {row["kind"] for row in by_module[module_id]["visuals"]},
            )
        self.assertIn(
            "spatial_interaction_cloud",
            {row["kind"] for row in by_module[
                "spatial_interaction_ensembles"
            ]["visuals"]},
        )

    def _root(self, temporary: str) -> Path:
        root = Path(temporary)
        (root / "results" / "pca-fes-basins").mkdir(parents=True)
        (root / "results" / "cluster-kmeans").mkdir(parents=True)
        (root / "results" / "rmsf").mkdir(parents=True)
        (root / "results" / "states").mkdir(parents=True)
        (root / "analysis-config.json").write_text(json.dumps({
            "config_schema": "salsbury-analysis-config-v1",
            "reporting": {
                "resource_table_enabled": True,
                "finding_picker_enabled": True,
                "interactive_report_enabled": True,
                "maximum_findings": 20,
            },
        }), encoding="utf-8")
        (root / "system.json").write_text(json.dumps({
            "project_id": "interactive-fixture",
            "systems": [{"system_id": "control"}],
        }), encoding="utf-8")
        (root / "preflight.report.json").write_text(json.dumps({
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "issues": [{
                "severity": "warning", "code": "FIXTURE_WARNING",
                "message": "Review this synthetic fixture.",
            }],
        }), encoding="utf-8")
        (root / "prioritized_findings.json").write_text(json.dumps({
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "findings": [{
                "finding_id": "finding-000001",
                "category": "free_energy_surface",
                "module_id": "pca_fes_basins",
                "statement": "Basin 1 contains the largest observed frame fraction.",
                "evidence_level": "descriptive",
                "system_ids": ["control"],
                "effect_value": 0.75,
                "statistically_significant": None,
                "report_path": "results/pca-fes-basins/report.json",
            }],
        }), encoding="utf-8")
        (root / "analysis_resource_and_frame_table.json").write_text(json.dumps({
            "technical_status": "complete", "scientific_status": "not evaluated",
            "rows": [{
                "module_id": "pca_fes_basins", "technical_status": "complete",
                "total_cpu_seconds": 2.0, "wall_seconds": 3.0,
                "maximum_resident_memory_mib": 40.0,
                "selected_source_physical_frames": 100,
                "source_physical_frames_available": 400,
            }],
        }), encoding="utf-8")
        (root / "sampling-plan.json").write_text(json.dumps({
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "method_plans": [{
                "module_id": "pca_fes_basins", "frame_stride": 4,
            }],
        }), encoding="utf-8")
        landscape = {
            "bounds": {
                "x_min_angstrom": -1.0, "x_max_angstrom": 1.0,
                "y_min_angstrom": -1.0, "y_max_angstrom": 1.0,
            },
            "grid": [
                {
                    "x_bin": x, "y_bin": y, "count": 25,
                    "relative_free_energy_kcal_per_mol": float(x + y),
                    "basin_id": 1,
                }
                for x in range(2) for y in range(2)
            ],
            "basins": [{
                "basin_id": 1, "root_x_bin": 0, "root_y_bin": 0,
                "assigned_count": 100, "assigned_fraction": 1.0,
            }],
        }
        (root / "results" / "pca-fes-basins" / "report.json").write_text(
            json.dumps({
                "module_id": "pca_fes_basins", "technical_status": "complete",
                "scientific_status": "not evaluated", "pca_basis": {
                    "x_component": 1, "y_component": 2,
                },
                "primary_smoothing_sigma_bins": 0.0,
                "smoothing_landscapes": [{
                    "smoothing_sigma_bins": 0.0, "landscape": landscape,
                    "per_system_landscapes": [],
                }],
                "landscape": landscape, "issues": [],
            }), encoding="utf-8"
        )
        (root / "results" / "cluster-kmeans" / "report.json").write_text(
            json.dumps({
                "module_id": "clustering_kmeans", "technical_status": "complete",
                "scientific_status": "not evaluated", "selected_model": {
                    "k": 2, "silhouette": 0.61, "cluster_sizes": [60, 40],
                    "silhouette_evaluation": {
                        "method": "exact_all_observations", "estimated": False,
                        "evaluated_observation_count": 100,
                        "total_observation_count": 100,
                        "sampling_replicate_count": 0,
                        "configured_random_seeds": [0, 7, 19, 41],
                        "evaluated_random_seeds": [],
                    },
                }, "silhouette_selection_stability": {
                    "gate_applied": False,
                    "status": "not_applicable_exact_silhouette",
                    "configured_random_seeds": [0, 7, 19, 41],
                    "evaluated_random_seeds": [], "selected_k": 2,
                }, "issues": [],
            }), encoding="utf-8"
        )
        (root / "results" / "rmsf" / "report.json").write_text(
            json.dumps({
                "module_id": "pooled_rmsf", "technical_status": "complete",
                "scientific_status": "not evaluated", "systems": [{
                    "system_id": "control", "atom_statistics": [{
                        "chain_id": "A", "residue_name": "CYS",
                        "residue_number": 1, "insertion_code": "",
                        "atom_name": "CA", "frame_pooled_rmsf_angstrom": 1.2,
                    }],
                }], "issues": [],
            }), encoding="utf-8"
        )
        pdb = (
            "ATOM      1  CA  CYS A   1       0.000   0.000   0.000  1.00  1.20           C  \n"
            "HETATM    2 ZN   ZN  A 101       2.000   0.000   0.000  1.00  0.00          ZN  \n"
            "END\n"
        )
        (root / "results" / "states" / "representative-1.pdb").write_text(
            pdb, encoding="utf-8"
        )
        return root

    def test_builds_offline_findings_figures_and_molecule_browser(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            result = build_interactive_report(root)
            self.assertEqual(result["technical_status"], "complete")
            self.assertEqual(result["scientific_status"], "not evaluated")
            index = root / "interactive-report" / "index.html"
            manifest = json.loads(
                (root / "interactive-report" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            text = index.read_text(encoding="utf-8")
            self.assertEqual(manifest["module_report_count"], 3)
            self.assertEqual(manifest["finding_count"], 1)
            self.assertEqual(manifest["inline_structure_count"], 1)
            self.assertEqual(manifest["network_dependency"], "none")
            self.assertEqual(
                manifest["index_sha256"], hashlib.sha256(index.read_bytes()).hexdigest()
            )
            for marker in (
                "Highest-priority findings", "Molecular states & figures",
                "All analyses", "Resources, frames, and sampling",
                "representative-1", "Basin 1 contains",
                '\"analysis_frame_stride\":4',
                '\"status\":\"not_applicable_exact_silhouette\"',
            ):
                self.assertIn(marker, text)
            self.assertNotIn("https://", text)
            reused = build_interactive_report(root)
            self.assertTrue(reused["reused"])

    def test_existing_report_hash_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            build_interactive_report(root)
            with (root / "interactive-report" / "index.html").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write("tamper")
            with self.assertRaises(InteractiveReportError):
                build_interactive_report(root)

    def test_nonfinite_scientific_values_become_strict_json_nulls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            report_path = root / "results" / "pca-fes-basins" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["undefined_score"] = -math.inf
            report_path.write_text(
                json.dumps(report, allow_nan=True), encoding="utf-8"
            )

            result = build_interactive_report(root)
            html_text = Path(result["index_path"]).read_text(encoding="utf-8")
            embedded_json = html_text.split(
                '<script id="report-data" type="application/json">', 1
            )[1].split("</script>", 1)[0]
            self.assertNotIn("-Infinity", embedded_json)
            self.assertNotIn("NaN", embedded_json)
            self.assertIn('\"undefined_score\":null', embedded_json)

    def test_inline_structure_limits_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            result = build_interactive_report(
                root, maximum_inline_structures=0,
            )
            self.assertEqual(result["inline_structure_count"], 0)
            self.assertEqual(result["omitted_structure_count"], 1)


if __name__ == "__main__":
    unittest.main()
