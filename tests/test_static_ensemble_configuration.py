import json
import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.analysis_config import AnalysisConfigError, load_analysis_config
from salsbury_md_analysis.execution_adapters import prepare_execution_artifacts
from salsbury_md_analysis.registry import list_modules
from salsbury_md_analysis.cli import main
from salsbury_md_analysis.comparative_quickstart import prepare_comparative_analysis
from tests.test_quickstart import _write_oligomer_inputs


class StaticEnsembleConfigurationTests(unittest.TestCase):
    @patch("salsbury_md_analysis.comparative_quickstart._discover_dssp_executable", return_value=None)
    def test_prepare_static_comparison_omits_kinetics_and_marks_every_worker(self, _dssp):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, psf, trajectories = _write_oligomer_inputs(root)
            request = root / "request.json"
            request.write_text(json.dumps({"request_schema": "salsbury-comparative-analysis-input-v1",
                "systems": [{"system_id": system, "pdb": str(pdb), "psf": str(psf),
                             "trajectories": [str(path) for path in trajectories], "frame_interval_ps": 10.0}
                            for system in ("control", "variant")]}))
            config = root / "config.json"
            config.write_text(json.dumps({"config_schema": "salsbury-analysis-config-v1",
                                         "execution": {"trajectory_mode": "static_ensemble"}}))
            output = root / "prepared"
            prepare_comparative_analysis(request_path=request, output_directory=output,
                                         project_id="static-test", config_path=config)
            for script in output.glob("*.slurm"):
                self.assertIn("export SALSBURY_STATIC_ENSEMBLE=1", script.read_text())
            for project in output.glob("project*.json"):
                modules = json.loads(project.read_text()).get("requested_modules", [])
                self.assertFalse(set(modules).intersection({"markov_state_models", "time_lagged_independent_component_analysis", "information_dynamics", "convergence_uncertainty", "scalar_threshold_states", "grouped_ml"}))
            report = json.loads((output / "planning-report.json").read_text())
            self.assertTrue(all(row["retained_frame_spacing_ns_per_replica"] == [] for row in report["sampling"]["tasks"]))
    def test_direct_and_instrumented_kinetic_commands_fail_before_reading_inputs(self):
        with patch.dict(os.environ, {"SALSBURY_STATIC_ENSEMBLE": "1"}), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["tica", "/absent/project.json"]), 2)
            self.assertEqual(main(["run-instrumented", "markov-models", "/absent/project.json"]), 2)
        self.assertIn("STATIC_ENSEMBLE_REQUIRES_NO_KINETICS", output.getvalue())
    def test_static_configuration_disables_time_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"config_schema": "salsbury-analysis-config-v1",
                                        "execution": {"trajectory_mode": "static_ensemble"}}))
            ids = [module.module_id for module in list_modules()]
            config = load_analysis_config(path, ids, ["global_common_heavy"])
            for module_id in ("convergence_uncertainty", "markov_state_models",
                              "scalar_threshold_states", "information_dynamics", "grouped_ml",
                              "time_lagged_independent_component_analysis"):
                self.assertFalse(config["modules"][module_id]["enabled"])
            self.assertTrue(config["modules"]["common_pca"]["enabled"])
            self.assertTrue(config["modules"]["pooled_rmsf"]["enabled"])
            self.assertEqual(config["clustering"]["feature_space"], "common_pca")
            self.assertEqual(config["modules"]["alternative_clustering"]["depends_on"], ["common_pca"])
            path.write_text(json.dumps(config))
            self.assertEqual(config, load_analysis_config(path, ids, ["global_common_heavy"]))

    def test_invalid_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for value in (True, None, "random", []):
                path.write_text(json.dumps({"config_schema": "salsbury-analysis-config-v1",
                                            "execution": {"trajectory_mode": value}}))
                with self.assertRaises(AnalysisConfigError):
                    load_analysis_config(path, ["common_pca"], [])

    def test_worker_policy_is_explicit_for_both_modes_and_custom_launcher(self):
        for mode, expected in (("continuous", "0"), ("static_ensemble", "1")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                worker = root / "run_analysis.slurm"
                worker.write_text('#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s" "$SALSBURY_STATIC_ENSEMBLE"\n')
                config = {"execution": {"trajectory_mode": mode, "submission_adapter": "custom",
                                        "maximum_parallel_cpus": 1, "maximum_memory_gib": 8,
                                        "maximum_hours_per_cpu": 1}, "reporting": {}}
                plan = {"phases": [{"phase_id": "analysis", "tasks": [{
                    "task_id": "analysis", "script": str(worker), "cpu_slots": 1,
                    "requested_memory_gib": 1, "requested_wall_minutes": 1,
                }]}], "maximum_parallel_cpus": 1, "maximum_parallel_memory_gib": 8,
                    "maximum_campaign_wall_hours": 1}
                with patch("salsbury_md_analysis.execution_adapters.build_local_execution_plan", return_value=plan):
                    prepare_execution_artifacts(root, config)
                    prepare_execution_artifacts(root, config)
                output = subprocess.check_output(["bash", str(worker)], env={**os.environ, "SALSBURY_STATIC_ENSEMBLE": str(1-int(expected))}, text=True)
                self.assertEqual(output, expected)
                self.assertEqual(worker.read_text().count("export SALSBURY_STATIC_ENSEMBLE="), 1)
                contract = json.loads((root / "launcher-contract.json").read_text())
                self.assertEqual(contract["phases"][0]["tasks"][0]["environment"]["SALSBURY_STATIC_ENSEMBLE"], expected)
