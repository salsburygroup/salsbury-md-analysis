import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.analysis_config import load_analysis_config
from salsbury_md_analysis.registry import list_modules
from salsbury_md_analysis.multivalent_bridges import multivalent_molecular_bridges_project
from tests.test_multivalent_bridges import _write_project
from salsbury_md_analysis.experimental_extension import apply_main_report_reuse, ExperimentalExtensionError


class ExperimentalStaticPolicyTests(unittest.TestCase):
    def test_extension_cannot_reuse_continuous_results_as_static(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis-config.json").write_text(json.dumps({"execution": {"trajectory_mode": "static_ensemble"}}))
            with self.assertRaisesRegex(ExperimentalExtensionError, "trajectory_mode differs"):
                apply_main_report_reuse(root, {"upstream_main_campaign": str(root),
                                             "upstream_trajectory_mode": "continuous"}, [])
    def test_experimental_master_switch_does_not_reenable_temporal_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"config_schema": "salsbury-analysis-config-v1",
                "enable_all_experimental_modules": True,
                "execution": {"trajectory_mode": "static_ensemble"}}))
            config = load_analysis_config(path, [row.module_id for row in list_modules()], [])
            for module in ("random_feature_koopman", "reactive_path_ensembles",
                           "interaction_persistence", "spatial_interaction_ensembles"):
                self.assertFalse(config["modules"][module]["enabled"])
            self.assertTrue(config["modules"]["multivalent_molecular_bridges"]["enabled"])
            self.assertTrue(config["modules"]["hydration_density_channels"]["enabled"])

    def test_static_bridges_retain_occupancy_without_residence_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = _write_project(Path(temporary))
            with patch.dict(os.environ, {"SALSBURY_STATIC_ENSEMBLE": "0"}):
                continuous = multivalent_molecular_bridges_project(project)
            with patch.dict(os.environ, {"SALSBURY_STATIC_ENSEMBLE": "1"}):
                static = multivalent_molecular_bridges_project(project)
            self.assertEqual(static["technical_status"], "complete")
            self.assertEqual(len(continuous["mediator_summaries"]), len(static["mediator_summaries"]))
            for left, right in zip(continuous["mediator_summaries"], static["mediator_summaries"]):
                self.assertEqual(left["bridge_occupancy"], right["bridge_occupancy"])
                self.assertEqual(right["bridge_residence"]["status"], "not_applicable")
