import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from salsbury_md_analysis.accepted_artifacts import reports_complete, validate_complete_report
from salsbury_md_analysis.alternative_clustering import _sklearn_partition
from salsbury_md_analysis.derived_report_publication import write_derived_report_sidecar
from salsbury_md_analysis.periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from salsbury_md_analysis.presentation_artifacts import generate_presentation_artifacts


class FreshCampaignRecoveryTests(unittest.TestCase):
    def test_static_preprocessed_cache_identity_is_mode_specific(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            system = root / "system.json"
            system.write_text("{}")
            cache = root / "cache.json"
            cache.write_text(json.dumps({
                "technical_status": "complete",
                "coordinate_representation": "independent_make_whole_unaligned_strided",
                "selection": "molecular_payload", "cached_system_manifest": str(system),
                "cached_system_manifest_sha256": hashlib.sha256(system.read_bytes()).hexdigest(),
            }))
            project = {"periodic_coordinate_policy": "preprocessed_make_whole",
                       "preprocessed_coordinate_source": {"cache_report": str(cache),
                           "cache_report_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}}
            with patch.dict(os.environ, {"SALSBURY_STATIC_ENSEMBLE": "1"}):
                processor = PeriodicFrameProcessor.from_replica(project, {}, system, 2)
                self.assertIsNotNone(processor)
            with patch.dict(os.environ, {"SALSBURY_STATIC_ENSEMBLE": "0"}):
                with self.assertRaisesRegex(PeriodicReconstructionError, "wrong representation"):
                    PeriodicFrameProcessor.from_replica(project, {}, system, 2)

    def test_mixture_assignment_keeps_real_component_centers(self):
        class Model:
            means_ = np.array([[50.0], [-20.0], [10.0]])
            lower_bound_ = -1.0
            def __init__(self, **kwargs): pass
            def fit_predict(self, values): return np.array([0, 2, 2])
            def predict(self, values): return np.array([0, 1, 2, 1])
            def aic(self, values): return 1.0
            def bic(self, values): return 2.0
        mixture = SimpleNamespace(GaussianMixture=Model, BayesianGaussianMixture=Model)
        for algorithm in ("gaussian_mixture", "variational_gaussian_mixture"):
            with self.subTest(algorithm=algorithm), patch(
                "salsbury_md_analysis.alternative_clustering.importlib.import_module",
                side_effect=lambda name: mixture if name == "sklearn.mixture" else SimpleNamespace(),
            ):
                result = _sklearn_partition(algorithm, np.zeros((3, 1)), 3, 0, {}, np.zeros((4, 1)))
                self.assertEqual(result["centers"], [(-20.0,), (10.0,), (50.0,)])
                self.assertEqual(result["assignments"], [2, 1, 1])
                self.assertEqual(result["_full_assignments"], [2, 0, 1, 0])
                self.assertEqual(result["cluster_sizes"], [0, 2, 1])
                self.assertEqual(result["assignment_only_model_component_count"], 1)

    def test_derived_publication_preserves_bytes_and_checks_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "report.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"module_id": "pooled_rmsf", "technical_status": "complete"}))
            target = root / "target" / "report.json"
            target.parent.mkdir()
            target.write_text(json.dumps({"module_id": "rmsf_permutation_inference", "technical_status": "complete", "comparisons": []}))
            before = target.read_bytes()
            self.assertFalse(reports_complete(root, ["target/report.json"]))
            write_derived_report_sidecar(target, [source])
            write_derived_report_sidecar(target, [source])
            self.assertEqual(target.read_bytes(), before)
            self.assertTrue(reports_complete(root, ["target/report.json"]))
            source.write_text(source.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                validate_complete_report(target, require_sidecar=True)

    def test_rmsf_inference_has_numerical_tables_and_labeled_figures(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "results" / "rmsf-permutation-inference"
            folder.mkdir(parents=True)
            (folder / "report.json").write_text(json.dumps({
                "module_id": "rmsf_permutation_inference", "technical_status": "complete",
                "comparisons": [{"system_a": "A", "system_b": "B", "result": {
                    "observed_mean_difference": [0.25, -0.5],
                    "two_sided_pointwise_p_values": [0.5, 0.25],
                    "max_t_familywise_p_values": [1.0, 0.5]}}],
            }))
            result = generate_presentation_artifacts(root)
            self.assertEqual(result["technical_status"], "complete")
            self.assertEqual(result["unadapted_report_count"], 0)
            tables = list((root / "presentation-artifacts").rglob("*.csv"))
            self.assertEqual(len(tables), 2)
            self.assertIn("max_t_familywise_p_value", tables[0].read_text())
            figures = list((root / "presentation-artifacts").rglob("*.svg"))
            self.assertEqual(len(figures), 2)
            self.assertTrue(any("RMSF difference" in path.read_text() for path in figures))


if __name__ == "__main__":
    unittest.main()
