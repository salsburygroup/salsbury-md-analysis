import json
import math
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.oligomer_symmetry import paired_member_score_correlations
from salsbury_md_analysis.pca import common_pca_project
from tests.test_quickstart import _write_oligomer_inputs
from salsbury_md_analysis.quickstart import _composition


class OligomerSymmetryTests(unittest.TestCase):
    def test_common_pca_expands_physical_frames_and_retains_member_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb, _, _ = _write_oligomer_inputs(root)
            coordinates = [
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 3.0, 0.0),
                (2.0, 4.0, 0.0), (2.0, 5.0, 0.0),
                (30.0, 0.0, 0.0), (31.0, 0.0, 0.0), (32.0, 0.0, 0.0),
                (32.0, 1.0, 0.0), (31.0, 1.0, 0.0), (32.0, 3.0, 0.0),
                (32.0, 4.0, 0.0), (32.0, 5.0, 0.0),
            ]
            frames = []
            for frame in range(30):
                values = list(coordinates)
                displacement = 0.25 * math.sin(frame / 4.0)
                values[4] = (values[4][0], values[4][1] + displacement, 0.0)
                values[12] = (values[12][0], values[12][1] + displacement, 0.0)
                frames.extend(["16\n", f"frame-{frame}\n"])
                frames.extend(
                    f"C {x:.8f} {y:.8f} {z:.8f}\n" for x, y, z in values
                )
            (root / "trajectory.xyz").write_text("".join(frames), encoding="utf-8")
            (root / "system.json").write_text(json.dumps({"systems": [{
                "system_id": "dimer", "replicas": [{
                    "replica_id": "r1", "topology": str(pdb), "segments": [{
                        "segment_id": "production", "trajectory": "trajectory.xyz",
                        "timing": {"first_frame_time": 0, "frame_interval": 10, "unit": "ps"},
                    }],
                }],
            }]}), encoding="utf-8")
            plan = _composition(pdb)["conformational_view_plan"]["equivalent_oligomer"]
            project = {
                "project_id": "oligomer-pca", "analysis_profile": "standard_md_v1",
                "system_manifest": "system.json", "analysis_output_root": "results",
                "reference_system": "dimer", "sampling_mode": "UNBIASED_MD",
                "coordinate_unit": "angstrom", "time_unit": "ps",
                "periodic_coordinate_policy": "reject", "reference_structure": str(pdb),
                "common_atom_policy": "strict",
                "selections": {
                    "alignment": {"preset": "macromolecular_backbone"},
                    "analysis": {"preset": "complex_trace"},
                },
                "definitions": {"common_pca": {
                    "alignment_selection": "alignment", "analysis_selection": "analysis",
                    "minimum_reference_coverage": 1.0, "frame_stride": 1,
                    "frame_selection": {"mode": "fixed_stride_v1"},
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                    "maximum_features": 24, "component_count": 2,
                    "minimum_evaluated_frames_per_replica": 2,
                    "basis_weighting": "replica_equal",
                    "solver": {"method": "dense_covariance_v1"},
                    "symmetry_expansion": plan,
                }},
                "requested_modules": ["common_pca"],
                "protected_locations": [str(pdb), str(root / "trajectory.xyz")],
            }
            project_path = root / "project.json"
            project_path.write_text(json.dumps(project), encoding="utf-8")
            report = common_pca_project(project_path)
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["observation_accounting"]["source_physical_frame_count"], 30)
        self.assertEqual(report["observation_accounting"]["symmetry_expanded_observation_count"], 60)
        projections = report["systems"][0]["replicas"][0]["segments"][0]["projections"]
        self.assertEqual(len(projections), 60)
        self.assertEqual({row["member_id"] for row in projections}, {"member-1", "member-2"})
        pair = report["paired_member_correlation"]["pair_reports"][0]
        self.assertEqual(pair["paired_physical_frame_count"], 30)

    def test_paired_correlations_use_matched_physical_frames(self):
        records = []
        for frame in range(30):
            for member, sign in (("member-1", 1.0), ("member-2", -1.0)):
                records.append({
                    "system_id": "system-a",
                    "replica_id": "replica-1",
                    "segment_id": "production",
                    "source_frame_index": frame,
                    "member_id": member,
                    "scores_angstrom": [sign * frame, float(frame % 5)],
                })
        report = paired_member_score_correlations(records)
        pair = report["pair_reports"][0]
        self.assertEqual(pair["paired_physical_frame_count"], 30)
        self.assertAlmostEqual(pair["same_component_correlations"][0], -1.0)
        self.assertAlmostEqual(pair["same_component_correlations"][1], 1.0)

    def test_member_pca_uses_exact_cross_variant_analysis_atom_intersection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference, _, _ = _write_oligomer_inputs(root)
            variant = root / "variant.pdb"
            variant.write_text(
                reference.read_text(encoding="utf-8").replace(" CB ", " CG "),
                encoding="utf-8",
            )
            coordinates = [
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0), (1.0, 1.0, 0.0), (2.0, 3.0, 0.0),
                (2.0, 4.0, 0.0), (2.0, 5.0, 0.0),
                (30.0, 0.0, 0.0), (31.0, 0.0, 0.0), (32.0, 0.0, 0.0),
                (32.0, 1.0, 0.0), (31.0, 1.0, 0.0), (32.0, 3.0, 0.0),
                (32.0, 4.0, 0.0), (32.0, 5.0, 0.0),
            ]
            frames = []
            for frame in range(30):
                values = list(coordinates)
                values[6] = (2.0, 4.0 + 0.1 * math.sin(frame), 0.0)
                values[14] = (32.0, 4.0 + 0.1 * math.sin(frame), 0.0)
                frames.extend(["16\n", f"frame-{frame}\n"])
                frames.extend(
                    f"C {x:.8f} {y:.8f} {z:.8f}\n" for x, y, z in values
                )
            (root / "trajectory.xyz").write_text("".join(frames), encoding="utf-8")
            (root / "system.json").write_text(json.dumps({"systems": [
                {"system_id": "reference", "replicas": [{
                    "replica_id": "r1", "topology": str(reference), "segments": [{
                        "segment_id": "production", "trajectory": "trajectory.xyz",
                        "timing": {"first_frame_time": 0, "frame_interval": 10, "unit": "ps"},
                    }],
                }]},
                {"system_id": "variant", "replicas": [{
                    "replica_id": "r1", "topology": str(variant), "segments": [{
                        "segment_id": "production", "trajectory": "trajectory.xyz",
                        "timing": {"first_frame_time": 0, "frame_interval": 10, "unit": "ps"},
                    }],
                }]},
            ]}), encoding="utf-8")
            plan = _composition(reference)["conformational_view_plan"]["equivalent_oligomer"]
            project = {
                "project_id": "cross-variant-member-pca",
                "analysis_profile": "standard_md_v1", "system_manifest": "system.json",
                "analysis_output_root": "results", "reference_system": "reference",
                "sampling_mode": "UNBIASED_MD", "coordinate_unit": "angstrom",
                "time_unit": "ps", "periodic_coordinate_policy": "reject",
                "reference_structure": str(reference), "common_atom_policy": "strict",
                "selections": {
                    "alignment": {"preset": "macromolecular_backbone"},
                    "analysis": {"preset": "complex_trace"},
                },
                "definitions": {"common_pca": {
                    "alignment_selection": "alignment", "analysis_selection": "analysis",
                    "minimum_reference_coverage": 1.0, "frame_stride": 1,
                    "frame_selection": {"mode": "fixed_stride_v1"},
                    "projection_frame_stride": 1,
                    "projection_frame_selection": {"mode": "fixed_stride_v1"},
                    "maximum_features": 24, "component_count": 2,
                    "minimum_evaluated_frames_per_replica": 2,
                    "basis_weighting": "replica_equal",
                    "solver": {"method": "dense_covariance_v1"},
                    "symmetry_expansion": plan,
                }},
                "requested_modules": ["common_pca"],
                "protected_locations": [str(reference), str(variant), str(root / "trajectory.xyz")],
            }
            project_path = root / "project.json"
            project_path.write_text(json.dumps(project), encoding="utf-8")
            report = common_pca_project(project_path)

        self.assertEqual(report["technical_status"], "complete", report)
        identity = report["common_member_analysis_identity"]
        self.assertEqual(identity["reference_analysis_atom_count_per_member"], 8)
        self.assertEqual(identity["common_analysis_atom_count_per_member"], 7)
        self.assertEqual(identity["excluded_reference_analysis_atom_count_per_member"], 1)
        self.assertEqual(report["basis"]["pca"]["feature_count"], 21)
        self.assertTrue(any(
            issue["code"] == "SYMMETRY_COMMON_ANALYSIS_INTERSECTION_EXCLUDES_VARIANT_ATOMS"
            for issue in report["issues"]
        ))


if __name__ == "__main__":
    unittest.main()
