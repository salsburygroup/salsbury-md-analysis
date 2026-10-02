import copy
import unittest

from salsbury_md_analysis.resource_calibrations import _aggregate_calibrations, qualify_calibration


class CalibrationApplicabilityTests(unittest.TestCase):
    def measurement(self, name, *, timeout=False, context=None):
        record = {"module_id":"test", "selected_source_physical_frames":100,
                  "symmetry_expanded_observations":100,"maximum_resident_memory_mib":10,
                  "memory_replacement_qualified":True, "resource_context":context or {}}
        if timeout:
            record.update(evidence_status="right_censored_timeout", allocated_cpu_count=1,
                          wall_seconds_lower_bound=900, total_cpu_seconds_lower_bound=900,
                          cpu_seconds_per_selected_physical_frame_lower_bound=9,
                          source_timeout_sha256=name*64)
        else:
            record.update(total_cpu_seconds=10, wall_seconds=10,
                          cpu_seconds_per_selected_physical_frame=.1, source_sidecar_sha256=name*64)
        return record

    def catalog(self, rows):
        return _aggregate_calibrations({"test":rows}, "catalog.json", "a"*64, {}, 1.5)["test"]

    def test_matching_context_qualifies_but_mismatched_timeout_is_preserved_not_applied(self):
        context = {"implementation_id":"streaming-v2", "coordinate_source":"cache", "worker_scope":"one_replica"}
        rows = [self.measurement("a",context=context),self.measurement("b",context=context),
                self.measurement("c",timeout=True,context={**context,"coordinate_source":"raw"})]
        original = copy.deepcopy(rows)
        qualified, audit = qualify_calibration(self.catalog(rows), context)
        self.assertTrue(qualified["memory_replacement_qualified"])
        self.assertEqual(qualified["conservative_cpu_seconds_per_frame"], .1)
        self.assertEqual(qualified["censored_timeout_count"], 0)
        self.assertEqual(audit["excluded_evidence"][0]["evidence_sha256"], "c"*64)
        self.assertEqual(rows, original)

    def test_unknown_evidence_kept_conservatively_and_cannot_replace_memory(self):
        rows = [self.measurement("a"),self.measurement("b"),self.measurement("c",timeout=True)]
        result, audit = qualify_calibration(self.catalog(rows), {"coordinate_source":"cache"})
        self.assertFalse(result["memory_replacement_qualified"])
        self.assertEqual(result["censored_timeout_count"], 1)
        self.assertEqual(result["conservative_cpu_seconds_per_frame"], 13.5)
        self.assertEqual(audit["status"], "matched_with_legacy_fallback")

    def test_no_matching_evidence_is_explicitly_provisional(self):
        result, audit = qualify_calibration(self.catalog([self.measurement("a", context={"implementation_id":"old"})]),
                                            {"implementation_id":"new"})
        self.assertIsNone(result)
        self.assertEqual(audit["status"], "no_applicable_measurement_use_provisional_model")

