import importlib.util
import unittest
from pathlib import Path

from salsbury_md_analysis.automatic_sampling import DIRECT_SAMPLING_PROFILES


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkHarnessTests(unittest.TestCase):
    def test_every_direct_sampling_profile_has_a_benchmark_runner(self):
        path = ROOT / "scripts" / "benchmark_frame_coverage.py"
        spec = importlib.util.spec_from_file_location("benchmark_frame_coverage", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = {profile.module_id for profile in DIRECT_SAMPLING_PROFILES}
        self.assertEqual(set(module.RUNNERS), expected)

    def test_segment_reports_produce_exact_physical_frame_coverage(self):
        path = ROOT / "scripts" / "benchmark_frame_coverage.py"
        spec = importlib.util.spec_from_file_location("benchmark_frame_coverage", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = {
            "systems": [{
                "replicas": [{
                    "segments": [
                        {"segment_id": "a", "observed_frame_count": 1000,
                         "evaluated_frame_count": 10},
                        {"segment_id": "b", "observed_frame_count": 500,
                         "evaluated_frame_count": 5},
                    ]
                }]
            }]
        }
        coverage = module._frame_coverage(report)
        self.assertEqual(coverage["source_frame_count"], 1500)
        self.assertEqual(coverage["estimator_selected_frame_count"], 15)
        self.assertAlmostEqual(coverage["estimator_coverage_fraction"], 0.01)


if __name__ == "__main__":
    unittest.main()
