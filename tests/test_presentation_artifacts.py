import tempfile
import unittest
import json
from pathlib import Path

from salsbury_md_analysis.presentation_artifacts import (
    PresentationArtifactError,
    artifact_record,
    finding_target,
    stable_artifact_id,
    validate_manifest,
    write_manifest,
    generate_presentation_artifacts,
)


class PresentationArtifactContractTests(unittest.TestCase):
    def test_many_state_legend_stays_inside_canvas(self):
        import xml.etree.ElementTree as ET
        from salsbury_md_analysis.presentation_artifacts import _state_population_svg
        rows = [{"system_id": "a", "state_id": i, "fraction_of_all_evaluated": 1/40} for i in range(40)]
        width,height,body = _state_population_svg(rows,"Forty states")
        nodes = ET.fromstring('<svg>'+body+'</svg>')
        labels = [node for node in nodes.findall('text') if (node.text or '').startswith('State ')]
        self.assertEqual(len(labels),40)
        self.assertGreater(len({node.get('y') for node in labels}),1)
        self.assertTrue(all(float(node.get('x'))+60 < width and float(node.get('y')) < height for node in labels))

    def test_alternative_populations_and_model_tables_do_not_overwrite_scopes(self):
        from salsbury_md_analysis.presentation_artifacts import _alternative_clustering_artifacts
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = []
            for sid in ("control", "variant"):
                path = root / "results/per-system" / sid / "conformational-views/common/alternative/report.json"
                path.parent.mkdir(parents=True)
                path.write_text('{}')
                methods = []
                for algorithm, fraction in (("pam", .7), ("gaussian_mixture", .4)):
                    methods.append({"algorithm": algorithm, "silhouette": fraction,
                                    "state_population_comparison": {"system_populations": [{
                                        "system_id": sid, "evaluated_count": 100,
                                        "state_populations": [{"state_id": 1, "count": int(100*fraction),
                                                               "fraction_of_all_evaluated": fraction}]}]}})
                _alternative_clustering_artifacts(root / "artifacts", path, {"algorithm_results": methods}, artifacts)
            self.assertEqual(len(artifacts), 12)
            self.assertEqual(len({a["relative_path"] for a in artifacts}), 12)
            populations = [a for a in artifacts if a["purpose"] == "state_populations"]
            self.assertEqual({a["context"]["algorithm"] for a in populations}, {"pam", "gaussian_mixture"})
            self.assertTrue(all((root / "artifacts" / a["relative_path"]).is_file() for a in artifacts))

    def test_explicit_unavailable_result_has_no_numeric_substitute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "results" / "unknown" / "report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"module_id": "unknown_scientific_module",
                "technical_status": "complete", "availability_status": "not_available",
                "availability_reason": "Required scientific input is absent."}))
            manifest = generate_presentation_artifacts(root)
            self.assertEqual(manifest["unadapted_report_count"], 0)
            self.assertEqual(manifest["reviewed_reports"][0]["presentation_adapter"],
                             "unavailable_with_explanation")
            self.assertTrue(all(not item["primary_human_output"] for item in manifest["artifacts"]))

    def test_generic_index_cannot_pass_scientific_figure_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "results" / "unknown" / "report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"module_id": "unknown_scientific_module",
                "technical_status": "complete", "rows": [{"index": 1}, {"index": 2}]}))
            with self.assertRaises(PresentationArtifactError):
                generate_presentation_artifacts(root)
            manifest = json.loads((root / "presentation-artifacts/presentation-manifest.json").read_text())
            self.assertEqual(manifest["technical_status"], "failed")
            self.assertEqual(manifest["unadapted_report_count"], 1)

    def test_every_optional_adapter_plots_only_its_named_result_quantity(self):
        from salsbury_md_analysis.experimental_presentation import SPECIFICATIONS
        for module, fields in SPECIFICATIONS.items():
            with self.subTest(module=module), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                report = root / "results" / module / "report.json"
                report.parent.mkdir(parents=True)
                quantity = next(iter(fields))
                report.write_text(json.dumps({"module_id": module, "technical_status": "complete",
                    "settings": {quantity: 999}, "results": [{"system_id": "fixture", quantity: .25}]}))
                manifest = generate_presentation_artifacts(root)
                table = next(row for row in manifest['artifacts'] if row['artifact_type'] == 'table')
                text = (root / 'presentation-artifacts' / table['relative_path']).read_text()
                self.assertIn('0.25', text)
                self.assertNotIn('999', text)

    def test_stable_ids_include_context_without_exposing_internal_paths(self):
        context = {"left_system_id": "A", "right_system_id": "B", "state_id": 1}
        first = stable_artifact_id(
            "figure", "pca_fes_basins", "state_populations", context=context
        )
        second = stable_artifact_id(
            "figure", "pca_fes_basins", "state_populations", context=context
        )
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("figure-pca-fes-basins-state-populations-"))
        self.assertNotIn("/", first)

    def test_finding_target_names_exact_context(self):
        target = finding_target(
            module_id="dccm",
            purpose="pairwise_difference",
            context={"left_system_id": "A", "right_system_id": "B", "atom_i": 1},
        )
        self.assertEqual(target["purpose"], "pairwise_difference")
        self.assertEqual(target["context"]["atom_i"], 1)
        self.assertEqual(target["preferred_artifact_types"], ["figure", "table"])

    def test_manifest_rejects_duplicate_ids(self):
        row = artifact_record(
            artifact_type="figure",
            module_id="dccm",
            purpose="system_matrix",
            title="System A DCCM",
            relative_path="dccm/system-a.svg",
            source_report_paths=["/tmp/report.json"],
            source_report_sha256=["a" * 64],
            context={"system_id": "A"},
            media_type="image/svg+xml",
        )
        row["artifact_sha256"] = "c" * 64
        row["artifact_size_bytes"] = 1
        manifest = {
            "presentation_manifest_schema": "salsbury-presentation-artifacts-v1",
            "artifacts": [row, row],
        }
        with self.assertRaisesRegex(PresentationArtifactError, "duplicate"):
            validate_manifest(manifest)

    def test_write_manifest_is_deterministic(self):
        row = artifact_record(
            artifact_type="table",
            module_id="clustering_kmeans",
            purpose="state_populations",
            title="K-means populations",
            relative_path="clustering-kmeans/state-populations.csv",
            source_report_paths=["/tmp/report.json"],
            source_report_sha256=["b" * 64],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "presentation-manifest.json"
            artifact_path = Path(temporary) / row["relative_path"]
            artifact_path.parent.mkdir(parents=True)
            artifact_path.write_text("value\n", encoding="utf-8")
            manifest = write_manifest(path, [row], analysis_root=Path(temporary))
            self.assertEqual(manifest["artifact_count"], 1)
            self.assertEqual(manifest["artifacts"][0]["artifact_id"], row["artifact_id"])
            self.assertTrue(path.is_file())
            self.assertEqual(len(manifest["artifacts"][0]["artifact_sha256"]), 64)

    def test_generate_primary_figures_and_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_report(directory, payload):
                path = root / "results" / directory / "report.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload), encoding="utf-8")

            comparison = {
                "system_populations": [{
                    "system_id": system_id,
                    "evaluated_count": 10,
                    "assigned_count": 10,
                    "assigned_coverage_fraction": 1.0,
                    "state_populations": [
                        {
                            "state_id": 1,
                            "count": count,
                            "fraction_of_all_evaluated": count / 10,
                        },
                        {
                            "state_id": 2,
                            "count": 10 - count,
                            "fraction_of_all_evaluated": (10 - count) / 10,
                        },
                    ],
                } for system_id, count in (("control", 7), ("variant", 3))],
            }
            grid = []
            for x in range(2):
                for y in range(2):
                    grid.append({
                        "x_bin": x,
                        "y_bin": y,
                        "relative_free_energy_kcal_per_mol": float(x + y),
                    })
            write_report("conformational-views/global_common_heavy/pca-fes-basins", {
                "module_id": "pca_fes_basins",
                "technical_status": "complete",
                "primary_smoothing_sigma_bins": 1.0,
                "pca_basis": {"x_component": 1, "y_component": 2},
                "landscape": {
                    "grid": grid,
                    "basins": [{
                        "basin_id": 1,
                        "root_x_bin": 0,
                        "root_y_bin": 0,
                        "assigned_fraction": 0.7,
                    }],
                },
                "state_population_comparison": comparison,
                "smoothing_sensitivity": [{
                    "alternate_smoothing_sigma_bins": 2.0,
                    "adjusted_rand_index": 0.9,
                }],
            })
            write_report("conformational-views/global_common_heavy/cluster-kmeans", {
                "module_id": "clustering_kmeans",
                "technical_status": "complete",
                "feature_contract": {"feature_source": "common_pca"},
                "selected_model": {"k": 2, "silhouette": 0.5},
                "grid_diagnostics": [
                    {"k": 2, "silhouette": 0.5},
                    {"k": 3, "silhouette": 0.4},
                ],
                "state_population_comparison": comparison,
            })
            write_report("dccm", {
                "module_id": "dccm",
                "technical_status": "complete",
                "analysis_atoms": [
                    {"chain_id": "A", "residue_name": "GUA", "residue_number": 1, "atom_name": "C1'"},
                    {"chain_id": "A", "residue_name": "GUA", "residue_number": 2, "atom_name": "C1'"},
                ],
                "systems": [
                    {"system_id": "control", "frame_pooled_dccm": {"matrix": [[1.0, 0.4], [0.4, 1.0]]}},
                    {"system_id": "variant", "frame_pooled_dccm": {"matrix": [[1.0, -0.2], [-0.2, 1.0]]}},
                ],
            })
            write_report("rmsd-rg", {
                "module_id": "replica_rmsd_rg",
                "technical_status": "complete",
                "time_unit": "ns",
                "systems": [{
                    "system_id": "control",
                    "replicas": [{
                        "replica_id": "rep1",
                        "segments": [{
                            "segment_id": "seg1",
                            "timeseries": [
                                {
                                    "time": float(index),
                                    "rmsd_angstrom": 1.0 + 0.1 * index,
                                    "radius_of_gyration_angstrom": (
                                        10.0 + 0.15 * index
                                    ),
                                }
                                for index in range(12)
                            ],
                        }],
                    }],
                }],
            })
            manifest = generate_presentation_artifacts(root)
            self.assertGreaterEqual(manifest["artifact_count"], 9)
            paths = [root / "presentation-artifacts" / row["relative_path"] for row in manifest["artifacts"]]
            self.assertTrue(all(path.is_file() for path in paths))
            fes = next(path for path in paths if path.name == "primary-fes.svg")
            text = fes.read_text(encoding="utf-8")
            self.assertIn("Relative free energy (kcal/mol)", text)
            self.assertIn("PC1 (Å)", text)
            dccm = next(
                path for path in paths
                if "comparisons" in path.parts and path.suffix == ".csv"
            )
            self.assertIn("left_minus_right", dccm.read_text(encoding="utf-8"))
            rg_histogram = next(
                row for row in manifest["artifacts"]
                if row["module_id"] == "replica_rmsd_rg"
                and row["purpose"] == "radius_of_gyration_histogram"
                and row["artifact_type"] == "figure"
            )
            self.assertEqual(rg_histogram["context"]["binning_rule"], "scott")
            self.assertEqual(
                rg_histogram["analysis_class"],
                "rmsd_and_radius_of_gyration",
            )
            rg_timeseries = next(
                row for row in manifest["artifacts"]
                if row["module_id"] == "replica_rmsd_rg"
                and row["purpose"] == "radius_of_gyration_angstrom_timeseries"
                and row["artifact_type"] == "figure"
            )
            rmsd_timeseries = next(
                row for row in manifest["artifacts"]
                if row["module_id"] == "replica_rmsd_rg"
                and row["purpose"] == "rmsd_angstrom_timeseries"
                and row["artifact_type"] == "figure"
            )
            self.assertFalse(rg_timeseries["primary_human_output"])
            self.assertTrue(rmsd_timeseries["primary_human_output"])
            rg_table = next(
                root / "presentation-artifacts" / row["relative_path"]
                for row in manifest["artifacts"]
                if row["module_id"] == "replica_rmsd_rg"
                and row["purpose"] == "radius_of_gyration_histogram"
                and row["artifact_type"] == "table"
            )
            self.assertIn("lower_edge_angstrom", rg_table.read_text(encoding="utf-8"))

    def test_experimental_reports_receive_named_analysis_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = {
                "pockets": {
                    "module_id": "ensemble_pocket_dynamics",
                    "systems": [
                        {"system_id": "sample", "occupancy_fraction": 0.2},
                        {"system_id": "variant", "occupancy_fraction": 0.3},
                    ],
                },
                "hydration": {
                    "module_id": "hydration_density_channels",
                    "systems": [
                        {"system_id": "sample", "mean_voxel_frame_occupancy": 0.1},
                        {"system_id": "variant", "mean_voxel_frame_occupancy": 0.2},
                    ],
                },
                "koopman": {
                    "module_id": "random_feature_koopman",
                    "selected_hyperparameters": {
                        "random_feature_count": 128,
                        "selection_score": 0.4,
                    },
                    "components": [{"component_index": 1, "eigenvalue": 0.7,
                                    "implied_timescale": 4.0, "time_unit": "ps"}],
                },
            }
            for directory, payload in reports.items():
                path = root / "results" / directory / "report.json"
                path.parent.mkdir(parents=True)
                payload["technical_status"] = "complete"
                path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = generate_presentation_artifacts(root)
            classes = {
                row["module_id"]: row["analysis_class"]
                for row in manifest["artifacts"]
            }
            self.assertEqual(
                classes["ensemble_pocket_dynamics"], "pocket_dynamics"
            )
            self.assertEqual(
                classes["hydration_density_channels"], "ions_and_solvation"
            )
            self.assertEqual(
                classes["random_feature_koopman"], "kinetic_models"
            )

    def test_integrated_comparison_reuses_module_owned_pairwise_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_report(directory, payload):
                path = root / "results" / directory / "report.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload), encoding="utf-8")

            atom = {
                "common_atom_index": 0,
                "chain_id": "A",
                "residue_name": "GLY",
                "residue_number": 1,
                "atom_name": "CA",
                "frame_pooled_rmsf_angstrom": 1.0,
            }
            write_report("rmsf", {
                "module_id": "pooled_rmsf",
                "technical_status": "complete",
                "systems": [
                    {"system_id": "control", "atom_statistics": [atom]},
                    {
                        "system_id": "variant",
                        "atom_statistics": [
                            {**atom, "frame_pooled_rmsf_angstrom": 1.5}
                        ],
                    },
                ],
            })
            write_report("integrated-comparison", {
                "module_id": "integrated_comparison",
                "technical_status": "complete",
                "comparison_system_ids": ["control", "variant"],
                "comparison_findings": [{
                    "module_id": "pooled_rmsf",
                    "system_ids": ["control", "variant"],
                    "comparison_family": "pooled_rmsf:pairwise_atom_difference",
                    "statement": "Variant RMSF differs from control.",
                    "effect_value": 0.5,
                }],
            })

            manifest = generate_presentation_artifacts(root)
            pairwise = [
                row for row in manifest["artifacts"]
                if row["module_id"] == "pooled_rmsf"
                and row["purpose"] == "pairwise_comparison"
            ]
            self.assertEqual(len(pairwise), 2)
            self.assertEqual(
                {row["source_report_paths"][0] for row in pairwise},
                {str((root / "results" / "rmsf" / "report.json").resolve())},
            )
            self.assertTrue(any(
                row["module_id"] == "integrated_comparison"
                and row["purpose"] == "comparison_coverage"
                for row in manifest["artifacts"]
            ))


if __name__ == "__main__":
    unittest.main()
