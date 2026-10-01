import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_tica import _write_project
from salsbury_md_analysis.feature_matrix import load_feature_matrix
from salsbury_md_analysis.tica import (
    TICAAnalysisError, fit_tica, time_lagged_independent_component_analysis_project,
)


def pca_fixture():
    def segment(name, count, offset, member=None):
        return {"segment_id": name, "projections": [
            {"source_frame_index": i, "time": offset + 100 * i, "time_unit": "ps",
             "scores_angstrom": [math.sin(i / 4 + offset / 100), math.cos(i / 3)],
             **({"member_id": member} if member else {})}
            for i in range(count)
        ]}
    return {
        "technical_status": "complete", "settings": {"basis_weighting": "frame"},
        "project_manifest_sha256": "unchanged-source-pca-project",
        "system_manifest_path": "system.json", "system_manifest_sha256": "system-hash",
        "contract_signature_sha256": "contract-hash", "input_content_signature_sha256": "input-hash",
        "systems": [{"system_id": "A", "replicas": [
            {"replica_id": "r1", "segments": [segment("long", 20, 0), segment("short", 19, 10000)]},
            {"replica_id": "r2", "segments": [segment("long", 21, 50000), segment("one", 1, 80000)]},
        ]}],
    }


class TICAEligibilityTests(unittest.TestCase):
    def configure(self, root, policy="omit"):
        path = _write_project(root)
        project = json.loads(path.read_text())
        config = project["definitions"]["time_lagged_independent_component_analysis"]
        config["lag_frames"] = 10
        if policy is not None:
            config["short_segment_policy"] = policy
        path.write_text(json.dumps(project))
        return path

    def test_default_rejects_short_and_opt_in_retains_basis_and_pairs(self):
        original = pca_fixture()
        snapshot = copy.deepcopy(original)
        with tempfile.TemporaryDirectory() as temporary:
            path = self.configure(Path(temporary), None)
            with patch("salsbury_md_analysis.tica.common_pca_project", return_value=original):
                with self.assertRaisesRegex(TICAAnalysisError, "requires|Requires"):
                    time_lagged_independent_component_analysis_project(path)
                self.configure(Path(temporary))
                result = time_lagged_independent_component_analysis_project(path)
                _, metadata, vectors, contract = load_feature_matrix(path,
                    {"feature_source": "tica", "component_indices": [1]}, hash_content=False, error_type=ValueError)
        self.assertEqual(original, snapshot)
        coverage = result["segment_eligibility"]
        self.assertEqual((coverage["input_stream_count"], coverage["included_stream_count"], coverage["excluded_stream_count"]), (4, 2, 2))
        self.assertEqual((coverage["input_observation_count"], coverage["included_observation_count"], coverage["excluded_observation_count"]), (61, 41, 20))
        self.assertEqual(result["pair_count"], 21)
        self.assertEqual(len(vectors), 41)
        self.assertEqual({m["segment_id"] for m in metadata}, {"long"})
        self.assertEqual(contract["segment_eligibility"], coverage)
        self.assertFalse(result["feature_lineage"]["basis_refitted_after_eligibility_filter"])
        arrays = [[p["scores_angstrom"] for p in r["segments"][0]["projections"]]
                  for r in original["systems"][0]["replicas"]]
        expected = fit_tica(arrays, lag_frames=10, component_count=1, covariance_regularization=1e-8)
        np.testing.assert_allclose(result["instantaneous_covariance"], expected["instantaneous_covariance"], atol=1e-14)
        np.testing.assert_allclose(result["symmetrized_lagged_covariance"], expected["symmetrized_lagged_covariance"], atol=1e-14)

    def test_all_short_fails_without_lowering_minimum(self):
        report = pca_fixture()
        for replica in report["systems"][0]["replicas"]:
            replica["segments"] = replica["segments"][1:]
        with tempfile.TemporaryDirectory() as temporary:
            path = self.configure(Path(temporary))
            with patch("salsbury_md_analysis.tica.common_pca_project", return_value=report):
                with self.assertRaisesRegex(TICAAnalysisError, "no eligible continuous"):
                    time_lagged_independent_component_analysis_project(path)

    def test_omission_does_not_hide_corruption_in_short_stream(self):
        for kind in ("gap", "overlap", "nonfinite", "duplicate", "unit"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                path = self.configure(Path(temporary))
                report = pca_fixture()
                rows = report["systems"][0]["replicas"][0]["segments"][1]["projections"]
                if kind == "gap": rows[1]["time"] += 10
                if kind == "overlap": rows[1]["time"] = rows[0]["time"]
                if kind == "nonfinite": rows[0]["scores_angstrom"][0] = float("nan")
                if kind == "duplicate": rows[1]["source_frame_index"] = rows[0]["source_frame_index"]
                if kind == "unit": rows[0]["time_unit"] = "ns"
                with patch("salsbury_md_analysis.tica.common_pca_project", return_value=report):
                    with self.assertRaises(TICAAnalysisError):
                        time_lagged_independent_component_analysis_project(path)

    def test_members_never_join_and_short_member_is_accounted(self):
        report = pca_fixture()
        replica = report["systems"][0]["replicas"][0]
        report["systems"][0]["replicas"] = [replica]
        rows = replica["segments"][0]["projections"]
        second = copy.deepcopy(rows[:19])
        for row in rows: row["member_id"] = "one"
        for row in second: row["member_id"] = "two"
        rows.extend(second)
        replica["segments"] = replica["segments"][:1]
        with tempfile.TemporaryDirectory() as temporary:
            path = self.configure(Path(temporary))
            with patch("salsbury_md_analysis.tica.common_pca_project", return_value=report):
                result = time_lagged_independent_component_analysis_project(path)
        self.assertEqual(result["pair_count"], 10)
        self.assertEqual(result["segment_eligibility"]["excluded_observation_count"], 19)
        self.assertEqual(result["observation_accounting"]["source_physical_frame_count"], 20)
        self.assertEqual(result["segments"][0]["member_id"], "one")

    def test_policy_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.configure(Path(temporary), "guess")
            with self.assertRaisesRegex(TICAAnalysisError, "short_segment_policy"):
                time_lagged_independent_component_analysis_project(path)

    def test_preparation_option_changes_tica_only(self):
        from salsbury_md_analysis.analysis_config import apply_module_configuration, default_analysis_config
        with tempfile.TemporaryDirectory() as temporary:
            path = self.configure(Path(temporary), None)
            definitions = json.loads(path.read_text())["definitions"]
            snapshot = copy.deepcopy(definitions)
            modules = ["common_pca", "time_lagged_independent_component_analysis"]
            config = default_analysis_config(modules, [])
            config["modules"][modules[1]]["options"] = {"short_segment_policy": "omit"}
            output, _, _, _ = apply_module_configuration(definitions, ["common-pca", "tica"], modules, config)
        self.assertEqual(definitions, snapshot)
        self.assertEqual(output["common_pca"], snapshot["common_pca"])
        self.assertEqual(output[modules[1]]["short_segment_policy"], "omit")
