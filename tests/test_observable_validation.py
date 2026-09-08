import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.cli import main
from salsbury_md_analysis.convergence import (
    ConvergenceAnalysisError,
    convergence_uncertainty_project_safe,
)
from salsbury_md_analysis.manifests import ManifestValidationError, validate_system
from salsbury_md_analysis.msm import (
    _vamp_fold_pair_indices,
    _vamp_predictive_assessment,
    _kinetic_validation_status,
    cross_validated_vamp_report,
)
from salsbury_md_analysis.numerical_sensitivity import (
    compare_numerical_protocols,
    _statistic,
)
from salsbury_md_analysis.observable_diagnostics import (
    analyze_observable_timeseries,
    validate_observable_input,
)
from salsbury_md_analysis.replica_holdout import replica_pipeline_holdout
from salsbury_md_analysis.simulation_protocol import (
    SimulationProtocolError,
    validate_simulation_protocol,
)


def protocol(step=2.0):
    return {
        "protocol_id": f"dt-{step}",
        "integration_step_fs": step,
        "integrator": "velocity-verlet",
        "thermostat": {"name": "none", "parameters": {}},
        "constraints": {"name": "none", "parameters": {}},
        "hamiltonian_sha256": "a" * 64,
        "ensemble": "NVE",
        "temperature_kelvin": 300.0,
        "initial_ensemble_id": "canonical-300K",
    }


def data(values, *, condition=None, replica="r1", step=2.0, kind="indicator"):
    definitions = [
        {
            "observable_id": "x",
            "unit": "dimensionless" if kind == "indicator" else "angstrom",
            "definition": "declared x",
            "kind": kind,
            "histogram_edges": [-0.5, 0.5, 1.5, 2.5, 3.5],
        }
    ]
    if condition is not None:
        definitions[0]["condition_observable_id"] = "retained"
        definitions.append(
            {
                "observable_id": "retained",
                "unit": "dimensionless",
                "definition": "ion retained",
                "kind": "indicator",
                "histogram_edges": [-0.5, 0.5, 1.5],
            }
        )
    return {
        "schema_version": "observable_timeseries_v1",
        "source_provenance": {"description": "synthetic control", "sha256": "b" * 64},
        "weighting": "frame",
        "observables": definitions,
        "segments": [
            {
                "system_id": "s",
                "replica_id": replica,
                "segment_id": "seg",
                "frame_interval": 1.0,
                "time_unit": "ps",
                "source_frame_stride": 1,
                "simulation_protocol": protocol(step),
                "rows": [
                    {
                        "source_frame_index": i,
                        "time": float(i),
                        "values": {
                            "x": float(value),
                            **(
                                {"retained": float(condition[i])}
                                if condition is not None
                                else {}
                            ),
                        },
                    }
                    for i, value in enumerate(values)
                ],
            }
        ],
    }


def settings():
    return {
        "source_module": "observable_timeseries",
        "metrics": ["x"],
        "timeseries_file": "series.json",
        "block_size_frames": 4,
        "minimum_blocks": 2,
        "include_partial_final_block": False,
        "effective_sample_size_reference": 20.0,
        "split_mean_difference_reference_in_sd": 1.0,
        "lag_frames": [1, 2],
    }


class PurgedVAMPTests(unittest.TestCase):
    def test_every_lag_window_is_disjoint_and_buffered(self):
        for lag in (1, 2, 5):
            for buffer in (0, 3):
                for fold in range(3):
                    train, test = _vamp_fold_pair_indices(
                        [30, 43], lag, 3, fold, buffer
                    )
                    training = {(t, i) for t, a, b in train for i in range(a, b + 1)}
                    testing = {(t, i) for t, a, b in test for i in range(a, b + 1)}
                    self.assertFalse(training & testing)
                    for t, a, b in train:
                        length = [30, 43][t]
                        start, stop = length * fold // 3, length * (fold + 1) // 3
                        self.assertTrue(b < start - buffer or a >= stop + buffer)

    def test_twelve_frame_regression_has_no_shared_endpoints(self):
        train, test = _vamp_fold_pair_indices([12], 2, 2, 0)
        self.assertEqual(train, [(0, i, i + 2) for i in range(6, 10)])
        self.assertEqual(test, [(0, i, i + 2) for i in range(4)])

    def test_large_buffer_abstains_instead_of_reusing_test_frames(self):
        report = cross_validated_vamp_report([[0, 1] * 10], 2, 2, 2, 1e-8, 100)
        self.assertEqual(report["status"], "not_calculable")
        self.assertNotIn("mean_heldout_vamp_e", report)

    def test_finite_bad_score_is_available_but_not_predictively_accepted(self):
        reports = [{"status": "complete", "mean_heldout_vamp_e": -10.0}]
        assessment = _vamp_predictive_assessment(reports, 0.0)
        self.assertFalse(assessment["passes_declared_criterion"])
        self.assertEqual(_kinetic_validation_status(True, assessment), "not passed")
        missing = _vamp_predictive_assessment(reports, None)
        self.assertEqual(_kinetic_validation_status(True, missing), "not assessed")

    def test_scores_use_only_purged_pair_sets(self):
        import salsbury_md_analysis.msm as msm

        original = msm._vamp_covariances
        calls = []

        def capture(pairs, state_count):
            calls.append(pairs)
            return original(pairs, state_count)

        # Distinct labels let the test recover endpoint frame identity.
        with patch.object(msm, "_vamp_covariances", side_effect=capture):
            report = cross_validated_vamp_report([list(range(20))], 20, 2, 2, 1e-8)
        self.assertEqual(report["status"], "complete")
        for train, test in zip(calls[::2], calls[1::2]):
            a = {i for _, x, y in train for i in (x, y)}
            b = {i for _, x, y in test for i in (x, y)}
            self.assertFalse(a & b)


class ObservableDiagnosticsTests(unittest.TestCase):
    def test_matching_populations_do_not_erase_different_switching_rates(self):
        fast = analyze_observable_timeseries(data(([0] * 5 + [1] * 5) * 8), settings())
        slow = analyze_observable_timeseries(
            data(([0] * 20 + [1] * 20) * 2), settings()
        )
        self.assertEqual(
            fast["observable_summaries"][0]["distribution"],
            slow["observable_summaries"][0]["distribution"],
        )
        a = fast["series_diagnostics"][0]["lagged_diagnostics"][0]
        b = slow["series_diagnostics"][0]["lagged_diagnostics"][0]
        self.assertEqual(a["indicator_switch_count"], 15)
        self.assertEqual(b["indicator_switch_count"], 3)
        self.assertEqual(
            fast["observable_summaries"][0]["kinetic_validity"], "not assessed"
        )

    def test_matching_conditional_distributions_preserve_mixture_weights(self):
        a = data([1, 3] * 10 + [0] * 60, condition=[1] * 20 + [0] * 60, kind="scalar")
        b = data([1, 3] * 30 + [0] * 20, condition=[1] * 60 + [0] * 20, kind="scalar")
        aa = analyze_observable_timeseries(a, settings())["observable_summaries"][0]
        bb = analyze_observable_timeseries(b, settings())["observable_summaries"][0]
        self.assertEqual(aa["distribution"], bb["distribution"])
        self.assertEqual(
            (aa["conditioning_fraction"], bb["conditioning_fraction"]), (0.25, 0.75)
        )

    def test_condition_gaps_never_create_lag_pairs(self):
        report = analyze_observable_timeseries(
            data([0, 0, 1, 1, 0, 0], condition=[1, 1, 0, 0, 1, 1]), settings()
        )
        series = report["series_diagnostics"][0]
        self.assertEqual(len(series["continuous_runs"]), 2)
        self.assertEqual(series["lagged_diagnostics"][0]["pair_count"], 2)
        self.assertEqual(series["lagged_diagnostics"][1]["pair_count"], 0)

    def test_constant_projection_is_explicit_and_has_no_ess(self):
        report = analyze_observable_timeseries(data([0] * 20), settings())
        run = report["series_diagnostics"][0]["continuous_runs"][0]
        self.assertTrue(run["constant_series"])
        self.assertIsNone(
            run["diagnostics"]["effective_sample_size"]["effective_sample_size"]
        )
        self.assertIsNone(
            report["series_diagnostics"][0]["lagged_diagnostics"][0]["correlation"]
        )
        self.assertFalse(report["scientific_conclusion_emitted"])

    def test_replica_equal_weights_are_based_on_full_denominators(self):
        first = data([0] * 8)
        first["segments"] += data([1] * 24, replica="r2")["segments"]
        self.assertEqual(
            analyze_observable_timeseries(first, settings())["observable_summaries"][0][
                "mean"
            ],
            0.75,
        )
        first["weighting"] = "replica_equal"
        self.assertAlmostEqual(
            analyze_observable_timeseries(first, settings())["observable_summaries"][0][
                "mean"
            ],
            0.5,
        )

    def test_joint_distribution_and_tails_retain_probability(self):
        payload = data([0, 1, 2, 3] * 4, condition=[1] * 16, kind="scalar")
        payload["observables"][0]["histogram_edges"] = [0.5, 1.5, 2.5]
        payload["joint_observables"] = [["x", "retained"]]
        result = analyze_observable_timeseries(payload, settings())
        row = result["observable_summaries"][0]["distribution"]
        self.assertEqual(row["below_range_probability"], 0.25)
        self.assertEqual(row["above_range_probability"], 0.25)
        self.assertEqual(
            result["joint_distributions"][0]["outside_range_probability"], 0.5
        )

    def test_irregular_frames_missing_units_and_bad_indicators_fail_closed(self):
        for mutation in [
            lambda d: d["segments"][0]["rows"][3].update(source_frame_index=9),
            lambda d: d["observables"][0].pop("unit"),
            lambda d: d["segments"][0]["rows"][2]["values"].update(x=2),
        ]:
            payload = data([0, 1] * 10)
            mutation(payload)
            with self.assertRaises(ConvergenceAnalysisError):
                validate_observable_input(payload)

    def test_wrong_json_field_types_raise_contract_errors(self):
        for mutate in (
            lambda payload: payload.update(weighting=[]),
            lambda payload: payload["observables"][0].update(kind=[]),
            lambda payload: payload["segments"][0].update(time_unit=[]),
            lambda payload: payload["observables"][0].update(
                condition_observable_id=[]
            ),
        ):
            payload = data([0, 1] * 10)
            mutate(payload)
            with self.assertRaises(ConvergenceAnalysisError):
                validate_observable_input(payload)

    def test_project_and_cli_consume_existing_series_without_trajectory_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "series.json").write_text(json.dumps(data([0, 1] * 20)))
            (path / "project.json").write_text(
                json.dumps({"definitions": {"convergence_uncertainty": settings()}})
            )
            with patch(
                "salsbury_md_analysis.convergence.replica_rmsd_rg_project",
                side_effect=AssertionError("must not reextract coordinates"),
            ):
                report = convergence_uncertainty_project_safe(path / "project.json")
                self.assertEqual(report["technical_status"], "complete")
                self.assertEqual(len(report["input_timeseries_sha256"]), 64)
                output = io.StringIO()
                with redirect_stdout(output):
                    status = main(["convergence", str(path / "project.json")])
                self.assertEqual(status, 0)
                self.assertEqual(
                    json.loads(output.getvalue())["source_module"],
                    "observable_timeseries",
                )


class ProtocolComparisonTests(unittest.TestCase):
    def compare(self, left, right, statistic="mean", tolerance=0.1, **extra):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for name, payload in [("left.json", left), ("right.json", right)]:
                (path / name).write_text(json.dumps(payload))
            cfg = {
                "reference_file": "left.json",
                "candidate_file": "right.json",
                "bootstrap_repeats": 20,
                "block_length_frames": 4,
                "comparisons": [
                    {
                        "observable_id": "x",
                        "statistic": statistic,
                        "tolerance": tolerance,
                        **extra,
                    }
                ],
            }
            return compare_numerical_protocols(cfg, path)

    def test_known_mean_shift_and_equal_mean_use_explicit_tolerance(self):
        report = self.compare(
            data([0] * 40, kind="scalar"), data([2] * 40, step=1, kind="scalar")
        )
        row = report["comparisons"][0]
        self.assertEqual(row["difference"], 2.0)
        self.assertEqual(row["interval"], [2.0, 2.0])
        self.assertEqual(
            row["numerical_sensitivity_status"], "exceeds declared tolerance"
        )
        row = self.compare(data([0] * 40), data([0] * 40, step=1))["comparisons"][0]
        self.assertEqual(
            row["numerical_sensitivity_status"], "within declared tolerance"
        )
        self.assertEqual(row["kinetic_validity"], "not assessed")

    def test_missing_protocol_identical_step_and_thermostat_change_abstain(self):
        for modify in [
            lambda d: d["segments"][0].pop("simulation_protocol"),
            lambda d: d["segments"][0]["simulation_protocol"].update(
                integration_step_fs=2
            ),
            lambda d: d["segments"][0]["simulation_protocol"]["thermostat"].update(
                name="langevin"
            ),
        ]:
            right = data([0, 1] * 20, step=1)
            modify(right)
            report = self.compare(data([0, 1] * 20), right)
            self.assertEqual(
                report["comparisons"][0]["numerical_sensitivity_status"], "not assessed"
            )

    def test_saving_stride_change_is_not_integration_convergence(self):
        right = data([0, 1] * 20, step=1)
        segment = right["segments"][0]
        segment["frame_interval"] = 2.0
        segment["source_frame_stride"] = 2
        for row in segment["rows"]:
            row["time"] *= 2
            row["source_frame_index"] *= 2
        report = self.compare(data([0, 1] * 20), right)
        self.assertIn("Saved-frame", report["protocol_comparison"]["reason"])
        self.assertIsNone(report["comparisons"][0]["interval"])

    def test_bootstrap_blocks_never_add_transitions(self):
        payload = data([0] * 10 + [1] * 10)
        blocks = [
            (("s", "r1"), payload["segments"][0]["rows"][:10]),
            (("s", "r1"), payload["segments"][0]["rows"][10:]),
        ]
        result = _statistic(
            payload,
            blocks,
            payload["observables"][0],
            {"statistic": "switch_probability", "lag_frames": 1},
        )
        self.assertEqual(result, 0.0)

    def test_constant_correlation_is_not_assessed(self):
        result = self.compare(
            data([0] * 40), data([0] * 40, step=1), "lagged_correlation", lag_frames=1
        )
        self.assertIsNone(result["comparisons"][0]["difference"])
        self.assertEqual(
            result["comparisons"][0]["numerical_sensitivity_status"], "not assessed"
        )

    def test_distribution_comparison_retains_a_known_probability_shift(self):
        result = self.compare(data([0] * 40), data([1] * 40, step=1), "distribution")
        row = result["comparisons"][0]
        self.assertEqual(row["difference"], 1.0)
        self.assertEqual(row["interval"], [1.0, 1.0])
        self.assertEqual(
            row["numerical_sensitivity_status"], "exceeds declared tolerance"
        )

    def test_conditioning_fraction_is_a_separate_comparison_estimand(self):
        left = data([1] * 40, condition=[1] * 10 + [0] * 30)
        right = data([1] * 40, condition=[1] * 30 + [0] * 10, step=1)
        result = self.compare(left, right, "conditioning_fraction")
        self.assertEqual(result["comparisons"][0]["difference"], 0.5)
        self.assertEqual(result["comparisons"][0]["conditioning_rule"], "retained")

    def test_bad_protocol_and_manifest_integration(self):
        manifest = {
            "systems": [
                {
                    "system_id": "s",
                    "replicas": [
                        {
                            "replica_id": "r",
                            "topology": "top.pdb",
                            "simulation_protocol": protocol(),
                            "segments": [
                                {
                                    "segment_id": "seg",
                                    "trajectory": "traj.dcd",
                                    "timing": {
                                        "first_frame_time": 0,
                                        "frame_interval": 1,
                                        "unit": "ps",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        validate_system(manifest)
        manifest["systems"][0]["replicas"][0]["simulation_protocol"][
            "integration_step_fs"
        ] = 0
        with self.assertRaises(ManifestValidationError):
            validate_system(manifest)
        with self.assertRaises(SimulationProtocolError):
            validate_simulation_protocol({"integration_step_fs": 2})


class ReplicaHoldoutTests(unittest.TestCase):
    def test_preprocessing_is_refit_using_only_training_replicas(self):
        source = {"s/r1": [0.0, 2.0], "s/r2": [100.0, 102.0], "s/r3": [10.0, 12.0]}
        original = copy.deepcopy(source)
        fits = []

        def fit(training):
            fits.append(set(training))
            # A learned centering transformation and estimator belong inside fit.
            return {
                "training_ids": set(training),
                "mean": sum(v for rows in training.values() for v in rows)
                / sum(map(len, training.values())),
            }

        def score(model, testing):
            self.assertFalse(model["training_ids"] & set(testing))
            return sum(next(iter(testing.values()))) / 2 - model["mean"]

        result = replica_pipeline_holdout(source, fit, score)
        self.assertEqual(len(fits), 3)
        self.assertEqual(source, original)
        self.assertFalse(result["acceptance_gate"])
        self.assertAlmostEqual(result["folds"][1]["score"], 95.0)

    def test_single_replica_cannot_masquerade_as_an_outer_holdout(self):
        with self.assertRaises(ValueError):
            replica_pipeline_holdout({"r1": [1]}, lambda x: x, lambda a, b: 0.0)


class ObservableIntegrationTests(unittest.TestCase):
    def test_finding_picker_recognizes_observable_diagnostics(self):
        from salsbury_md_analysis.finding_picker import _quality_control_records

        report = analyze_observable_timeseries(data([0, 1] * 20), settings())
        records = _quality_control_records(report, Path("report.json"))
        self.assertEqual(records[0]["status"], "quantitative_diagnostics")
        self.assertIn("observable continuous runs", records[0]["statement"])
        self.assertNotIn("RMSD/Rg", records[0]["statement"])

    def test_msm_evaluator_requires_a_declared_predictive_criterion(self):
        import salsbury_md_analysis.msm as msm

        cfg = msm._settings(
            {
                "definitions": {
                    "markov_state_models": {
                        "assignment_source": "clustering_kmeans",
                        "lag_frames": [1, 2, 4],
                        "estimators": ["reversible_symmetrized"],
                        "minimum_transition_count": 1,
                        "maximum_states": 10,
                        "ck_multiples": [2],
                        "maximum_ck_rmse": 1.0,
                        "maximum_implied_timescale_relative_range": 10.0,
                        "vamp_cross_validation_folds": 2,
                    }
                }
            }
        )
        values = ([1] * 5 + [2] * 5) * 8
        rows = [
            {
                "system_id": "s",
                "replica_id": "r",
                "segment_id": "seg",
                "source_frame_index": i,
                "time": float(i),
                "time_unit": "ps",
                "cluster_id": value,
            }
            for i, value in enumerate(values)
        ]
        result = msm._evaluate_state_definition(
            rows,
            cfg,
            candidate_id="fixture",
            family="clustering",
            geometric_score=None,
            geometric_coverage=1.0,
        )
        self.assertTrue(result["validation_gates"]["vamp_scores_available"])
        self.assertNotIn("cross_validated_vamp", result["validation_gates"])
        self.assertEqual(result["kinetic_validation_status"], "not assessed")
        cfg["minimum_heldout_vamp_e"] = 1e6
        result = msm._evaluate_state_definition(
            rows,
            cfg,
            candidate_id="fixture",
            family="clustering",
            geometric_score=None,
            geometric_coverage=1.0,
        )
        self.assertEqual(result["kinetic_validation_status"], "not passed")

    def test_numerical_comparison_cli_and_invalid_input_exit_status(self):
        root = Path(__file__).resolve().parents[1]
        stream = io.StringIO()
        with redirect_stdout(stream):
            status = main(
                [
                    "compare-numerical-protocols",
                    str(root / "examples/observable_validation/comparison.json"),
                ]
            )
        self.assertEqual(status, 0)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["comparisons"][0]["difference"], 0.0)
        self.assertEqual(
            result["comparisons"][1]["numerical_sensitivity_status"],
            "exceeds declared tolerance",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text("{}")
            stream = io.StringIO()
            with redirect_stdout(stream):
                status = main(["compare-numerical-protocols", str(path)])
            self.assertEqual(status, 2)
            self.assertEqual(
                json.loads(stream.getvalue())["technical_status"], "failed"
            )


if __name__ == "__main__":
    unittest.main()
