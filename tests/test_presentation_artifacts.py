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
            rg_table = next(
                root / "presentation-artifacts" / row["relative_path"]
                for row in manifest["artifacts"]
                if row["module_id"] == "replica_rmsd_rg"
                and row["purpose"] == "radius_of_gyration_histogram"
                and row["artifact_type"] == "table"
            )
            self.assertIn("lower_edge_angstrom", rg_table.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
