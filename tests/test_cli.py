import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from salsbury_md_analysis.cli import main


class CliTests(unittest.TestCase):
    def test_validate_manifest_json_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "system.json"
            path.write_text(
                json.dumps({
                    "systems": [{
                        "system_id": "system",
                        "replicas": [{
                            "replica_id": "r1",
                            "topology": "not-checked.pdb",
                            "segments": [{
                                "segment_id": "s1",
                                "trajectory": "not-checked.dcd",
                                "timing": {
                                    "first_frame_time": 0,
                                    "frame_interval": 2,
                                    "unit": "ps",
                                },
                            }],
                        }],
                    }]
                }),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["validate-manifest", "system", str(path), "--json"])
            report = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertTrue(report["valid"])

    def test_invalid_manifest_uses_exit_status_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "project.json"
            path.write_text("{}", encoding="utf-8")
            output = io.StringIO()
            with redirect_stderr(output):
                status = main(["validate-manifest", "project", str(path)])
            self.assertEqual(status, 2)
            self.assertIn("INVALID project manifest", output.getvalue())

    def test_preflight_command_reports_scientific_status_separately(self):
        root = Path(__file__).resolve().parents[1]
        manifest = root / "examples" / "manifest_fixture" / "system.json"
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["preflight-system", str(manifest), "--hash-content"])
        report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["scientific_status"], "not evaluated")

    def test_resource_planner_reports_automatic_subsampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "benchmark.json"
            path.write_text(json.dumps({
                "technical_status": "complete",
                "module_id": "example",
                "project_sha256": "a" * 64,
                "report_sha256": "b" * 64,
                "finished_utc": "2026-08-12T00:00:00Z",
                "resources": {"wall_seconds": 100.0, "maximum_rss_kib": 102400},
                "report_size_bytes": 2000,
                "frame_coverage": {"estimator_selected_frame_count": 100},
            }), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                status = main([
                    "plan-frame-resources", str(path),
                    "--total-source-frames", "30000",
                    "--replica-count", "3",
                    "--target-wall-hours", "1.25",
                    "--sensitivity-check-policy", "off",
                ])
            report = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual(
            report["plan"]["resolved_mode"], "integer_stride_per_replica_v1"
        )
        self.assertEqual(report["plan"]["sensitivity_check_policy"], "off")
        self.assertEqual(report["issues"][0]["code"], "FRAME_SUBSAMPLING_RECOMMENDED")

    def test_instrumented_runner_accepts_extended_rdf_project_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-project.json"
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["run-instrumented", "rdf", str(missing)])
            report = json.loads(output.getvalue())
        self.assertEqual(status, 2)
        self.assertEqual(report["module_id"], "radial_distribution_functions")
        self.assertEqual(report["technical_status"], "failed")

    def test_multivalent_bridge_command_is_exposed_by_top_level_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-project.json"
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["multivalent-bridges", str(missing)])
            report = json.loads(output.getvalue())
        self.assertEqual(status, 2)
        self.assertEqual(report["module_id"], "multivalent_molecular_bridges")
        self.assertEqual(report["technical_status"], "failed")

    def test_spatial_optional_commands_are_exposed_by_top_level_parser(self):
        for command, module_id in (
            ("hydration-density-channels", "hydration_density_channels"),
            ("ensemble-pocket-dynamics", "ensemble_pocket_dynamics"),
        ):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as temporary:
                missing = Path(temporary) / "missing-project.json"
                output = io.StringIO()
                with redirect_stdout(output):
                    status = main([command, str(missing)])
                report = json.loads(output.getvalue())
            self.assertEqual(status, 2)
            self.assertEqual(report["module_id"], module_id)
            self.assertEqual(report["technical_status"], "failed")

    def test_new_paper_method_commands_are_exposed_by_top_level_parser(self):
        for command, module_id in (
            ("energetic-network-embeddings", "energetic_network_embeddings"),
            ("spatial-interaction-ensembles", "spatial_interaction_ensembles"),
            ("interaction-persistence", "interaction_persistence"),
            ("random-feature-koopman", "random_feature_koopman"),
        ):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as temporary:
                missing = Path(temporary) / "missing-project.json"
                output = io.StringIO()
                with redirect_stdout(output):
                    status = main([command, str(missing)])
                report = json.loads(output.getvalue())
            self.assertEqual(status, 2)
            self.assertEqual(report["module_id"], module_id)
            self.assertEqual(report["technical_status"], "failed")

    def test_reactive_path_command_is_exposed_by_top_level_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-project.json"
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["reactive-path-ensembles", str(missing)])
            report = json.loads(output.getvalue())
        self.assertEqual(status, 2)
        self.assertEqual(report["module_id"], "reactive_path_ensembles")
        self.assertEqual(report["technical_status"], "failed")

if __name__ == "__main__":
    unittest.main()
