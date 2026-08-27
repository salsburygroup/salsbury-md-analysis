import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.registry import list_modules
from salsbury_md_analysis.scientific_sampling import (
    ScientificSamplingError,
    apply_scientific_minimums_to_tasks,
    assess_raw_sampling,
    list_scientific_sampling_profiles,
    load_scientific_minimums,
    required_frames_per_replica,
    scientific_minimums_document,
    scientific_sampling_profile,
)


class ScientificSamplingTests(unittest.TestCase):
    def test_editable_minimums_are_complete_and_stricter_values_apply(self):
        document = scientific_minimums_document()
        self.assertEqual(
            set(document["methods"]),
            {row.module_id for row in list_scientific_sampling_profiles()},
        )
        document["methods"]["common_pca"].update({
            "minimum_frames_per_replica": 600,
            "minimum_frames_overall_per_system": 2_400,
        })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "minimums.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            policy = load_scientific_minimums(path)
        tasks = apply_scientific_minimums_to_tasks([{
            "task_id": "pca",
            "module_id": "common_pca",
            "source_frames_per_replica": [2_000, 2_000],
            "system_ids_per_replica": ["system", "system"],
            "minimum_frames_per_replica": 100,
            "maximum_frames_per_replica": 2_000,
        }], policy)
        self.assertEqual(tasks[0]["minimum_frames_per_replica"], 1_200)
        self.assertTrue(
            str(tasks[0]["scientific_sampling_requirements"]["policy_id"])
            .startswith("scientific-sampling-standard-v3+strict-")
        )

    def test_editable_minimums_cannot_weaken_packaged_policy(self):
        document = scientific_minimums_document()
        document["methods"]["common_pca"][
            "minimum_frames_per_replica"
        ] = 249
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "minimums.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                ScientificSamplingError, "at least 250"
            ):
                load_scientific_minimums(path)
        document = scientific_minimums_document()
        document["methods"]["markov_state_models"][
            "maximum_time_gap_between_retained_frames_ns"
        ] = 0.75
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "minimums.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                ScientificSamplingError, "may not exceed"
            ):
                load_scientific_minimums(path)
    def test_every_registered_module_has_exactly_one_policy(self):
        registered = {row.module_id for row in list_modules()}
        profiled = {
            row.module_id for row in list_scientific_sampling_profiles()
        }
        self.assertEqual(profiled, registered)

    def test_resource_limited_sampling_is_not_kept_enabled(self):
        assessment = assess_raw_sampling(
            scientific_sampling_profile("water_mediated_hydrogen_bond_networks"),
            selected_frames_per_replica=[20, 20],
            source_frames_per_replica=[10_000, 10_000],
            system_ids_per_replica=["system", "system"],
            integer_stride=500,
        )
        self.assertEqual(
            assessment["raw_coverage_status"],
            "resource_limited_below_standard",
        )
        self.assertFalse(assessment["keep_enabled"])

    def test_short_complete_fixture_is_source_limited_not_resource_limited(self):
        assessment = assess_raw_sampling(
            scientific_sampling_profile("common_pca"),
            selected_frames_per_replica=[20],
            source_frames_per_replica=[20],
            system_ids_per_replica=["fixture"],
            integer_stride=1,
        )
        self.assertEqual(
            assessment["raw_coverage_status"],
            "source_limited_below_standard",
        )

    def test_thermodynamic_floor_uses_samples_not_temporal_spacing(self):
        profile = scientific_sampling_profile(
            "water_mediated_hydrogen_bond_networks"
        )
        required = required_frames_per_replica(
            profile,
            system_ids_per_replica=["system"],
            source_frames_per_replica=[10_001],
            frame_intervals_ns_per_replica=[0.01],
            source_time_spans_ns_per_replica=[100.0],
        )
        self.assertEqual(required, 500)
        assessment = assess_raw_sampling(
            profile,
            selected_frames_per_replica=[2_001],
            source_frames_per_replica=[10_001],
            system_ids_per_replica=["system"],
            integer_stride=5,
            frame_intervals_ns_per_replica=[0.01],
            source_time_spans_ns_per_replica=[100.0],
        )
        self.assertTrue(assessment["keep_enabled"])
        self.assertEqual(
            assessment["sampling_floor_basis"],
            "minimum_samples",
        )
        self.assertFalse(
            assessment["planner_estimates_autocorrelation_or_event_rates"]
        )

    def test_short_duration_is_provenance_not_an_acceptance_gate(self):
        profile = scientific_sampling_profile("common_pca")
        assessment = assess_raw_sampling(
            profile,
            selected_frames_per_replica=[1_000],
            source_frames_per_replica=[1_000],
            system_ids_per_replica=["short"],
            integer_stride=1,
            frame_intervals_ns_per_replica=[0.001],
            source_time_spans_ns_per_replica=[0.999],
        )
        self.assertTrue(assessment["keep_enabled"])
        self.assertEqual(
            assessment["raw_coverage_status"], "meets_standard_raw_floor"
        )
        self.assertFalse(assessment["time_span_fraction_is_acceptance_gate"])
        self.assertFalse(assessment["physical_time_span_is_acceptance_gate"])

    def test_generic_profiles_have_no_effective_sample_or_duration_gate(self):
        contract = scientific_sampling_profile("common_pca")
        self.assertFalse(hasattr(contract, "minimum_effective_samples_per_system"))
        self.assertFalse(hasattr(contract, "minimum_time_span_fraction"))
        self.assertFalse(hasattr(contract, "minimum_physical_span_ns"))

    def test_observed_event_count_is_a_diagnostic_not_a_planning_gate(self):
        assessment = assess_raw_sampling(
            scientific_sampling_profile("hydrogen_bonds"),
            selected_frames_per_replica=[1_000],
            source_frames_per_replica=[1_000],
            system_ids_per_replica=["system"],
        )
        self.assertTrue(assessment["keep_enabled"])
        self.assertTrue(assessment["scientific_interpretation_ready"])
        self.assertEqual(
            assessment["postrun_event_or_transition_diagnostic"], 20
        )

    def test_temporal_method_converts_maximum_spacing_to_frame_floor(self):
        profile = scientific_sampling_profile("information_dynamics")
        required = required_frames_per_replica(
            profile,
            system_ids_per_replica=["system"] * 6,
            source_frames_per_replica=[100_001] * 6,
            frame_intervals_ns_per_replica=[0.01] * 6,
            source_time_spans_ns_per_replica=[1_000.0] * 6,
        )
        self.assertEqual(required, 2_001)

    def test_method_specific_temporal_spacing_can_fail_despite_frame_count(self):
        profile = scientific_sampling_profile("information_dynamics")
        assessment = assess_raw_sampling(
            profile,
            selected_frames_per_replica=[2_001],
            source_frames_per_replica=[20_001],
            system_ids_per_replica=["system"],
            integer_stride=100,
            frame_intervals_ns_per_replica=[0.01],
            source_time_spans_ns_per_replica=[200.0],
        )
        self.assertFalse(assessment["keep_enabled"])
        self.assertEqual(assessment["temporal_spacing_failures"], [0])

    def test_temporal_separation_is_only_for_order_dependent_methods(self):
        temporal = {
            profile.module_id
            for profile in list_scientific_sampling_profiles()
            if profile.maximum_uniform_spacing_ns > 0.0
        }
        self.assertEqual(temporal, {
            "ensemble_pocket_dynamics",
            "information_dynamics",
            "interaction_persistence",
            "time_lagged_independent_component_analysis",
            "multivalent_molecular_bridges",
            "random_feature_koopman",
            "reactive_path_ensembles",
            "markov_state_models",
        })
        self.assertEqual(
            scientific_sampling_profile("common_pca")
            .maximum_uniform_spacing_ns,
            0.0,
        )
        self.assertEqual(
            scientific_sampling_profile("hydrogen_bonds")
            .maximum_uniform_spacing_ns,
            0.0,
        )
        self.assertEqual(
            scientific_sampling_profile("markov_state_models")
            .maximum_uniform_spacing_ns,
            0.5,
        )

    def test_default_floors_remain_permissive_feasibility_minima(self):
        framed = [
            profile for profile in list_scientific_sampling_profiles()
            if profile.minimum_frames_per_replica > 0
        ]
        self.assertLessEqual(
            max(profile.minimum_frames_per_replica for profile in framed),
            500,
        )
        self.assertLessEqual(
            max(profile.minimum_frames_per_system for profile in framed),
            2_000,
        )
        temporal = [
            profile for profile in framed
            if profile.maximum_uniform_spacing_ns > 0.0
        ]
        self.assertGreaterEqual(
            min(profile.maximum_uniform_spacing_ns for profile in temporal),
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
