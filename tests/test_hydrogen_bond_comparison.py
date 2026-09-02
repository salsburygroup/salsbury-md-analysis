import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.hydrogen_bond_comparison import (
    compare_hydrogen_bond_reports_file,
    compare_hydrogen_bond_reports_file_safe,
)
from salsbury_md_analysis.hydrogen_bond_sparse import pack_sparse_present_geometry


def atom(index, residue_name, atom_name, element, residue_number=4):
    return {
        "atom_index": index, "serial": index + 1, "atom_name": atom_name,
        "altloc": "", "residue_name": residue_name, "chain_id": "C",
        "residue_number": residue_number, "insertion_code": "", "element": element,
    }


def report(residue_name, present_by_frame, scope="protein_nucleic_acid"):
    donor = atom(0, "SER", "OG", "O", residue_number=75)
    hydrogen_1 = atom(1, "SER", "HG1", "H", residue_number=75)
    hydrogen_2 = atom(2, "SER", "HG2", "H", residue_number=75)
    acceptor = atom(3, residue_name, "O6", "O")
    return {
        "module_id": "hydrogen_bond_discovery",
        "technical_status": "complete", "error_count": 0,
        "contract_signature_sha256": "a" * 64,
        "input_content_signature_sha256": "b" * 64,
        "settings": {"interaction_scope": scope},
        "cutoff_definitions": [{
            "cutoff_id": "primary", "kind": "primary",
            "maximum_donor_acceptor_distance_angstrom": 3.0,
            "minimum_donor_hydrogen_acceptor_angle_degrees": 150.0,
        }],
        "frame_matrix_representation": "sparse_implicit_zero_v1",
        "sparse_zero_contract": "absent is evaluated zero",
        "candidate_dictionary": [
            {"bond_id": "a", "donor_atom_index": 0, "hydrogen_atom_index": 1, "acceptor_atom_index": 3},
            {"bond_id": "b", "donor_atom_index": 0, "hydrogen_atom_index": 2, "acceptor_atom_index": 3},
        ],
        "atom_dictionary": [
            {"atom_index": 0, "identity": donor},
            {"atom_index": 1, "identity": hydrogen_1},
            {"atom_index": 2, "identity": hydrogen_2},
            {"atom_index": 3, "identity": acceptor},
        ],
        "frame_bond_matrix": [
            {
                "replica_id": replica,
                "cutoff_present_candidate_indices": {"primary": present},
            }
            for replica, present in present_by_frame
        ],
    }


def packed_report(residue_name, present_by_frame):
    value = report(residue_name, present_by_frame)
    value["frame_matrix_representation"] = "sparse_packed_v2"
    packed_frames = []
    for frame in value["frame_bond_matrix"]:
        present = frame["cutoff_present_candidate_indices"]["primary"]
        packed_frames.append({
            "replica_id": frame["replica_id"],
            "candidate_count": len(value["candidate_dictionary"]),
            **pack_sparse_present_geometry([
                {
                    "candidate_index": index,
                    "donor_acceptor_distance_angstrom": 2.8,
                    "donor_hydrogen_acceptor_angle_degrees": 170.0,
                    "present_cutoff_ids": ["primary"],
                }
                for index in present
            ], value["cutoff_definitions"], len(value["candidate_dictionary"])),
        })
    value["frame_bond_matrix"] = packed_frames
    return value


class HydrogenBondComparisonTests(unittest.TestCase):
    def write_request(self, root, first, second, **updates):
        (root / "control.json").write_text(json.dumps(first), encoding="utf-8")
        (root / "lesion.json").write_text(json.dumps(second), encoding="utf-8")
        request = {
            "comparison_id": "synthetic",
            "conditions": [
                {"condition_id": "control", "report": "control.json"},
                {"condition_id": "lesion", "report": "lesion.json"},
            ],
            "expected_interaction_scope": "protein_nucleic_acid",
            "homolog_mappings": [
                {
                    "condition_id": "control",
                    "match": {"chain_id": "C", "residue_number": 4, "residue_name": "DG"},
                    "canonical_updates": {"residue_name": "TARGET_G"},
                },
                {
                    "condition_id": "lesion",
                    "match": {"chain_id": "C", "residue_number": 4, "residue_name": "8OG"},
                    "canonical_updates": {"residue_name": "TARGET_G"},
                },
            ],
        }
        request.update(updates)
        path = root / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        return path

    def test_groups_equivalent_hydrogens_once_per_frame_and_maps_homologs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_request(
                root,
                report("DG", [("r1", [0, 1]), ("r1", []), ("r2", [1]), ("r2", [])]),
                report("8OG", [("r1", [0, 1]), ("r1", [0]), ("r2", []), ("r2", [])]),
            )
            result = compare_hydrogen_bond_reports_file(path)
        self.assertEqual(result["technical_status"], "complete")
        self.assertEqual(result["observed_group_union_count"], 1)
        row = result["group_comparisons"][0]
        self.assertEqual(row["acceptor_identity"]["residue_name"], "TARGET_G")
        self.assertAlmostEqual(row["condition_occupancies"]["control"], 0.5)
        self.assertAlmostEqual(row["condition_occupancies"]["lesion"], 0.5)

    def test_packed_sparse_reports_preserve_grouped_occupancies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_request(
                root,
                packed_report("DG", [("r1", [0, 1]), ("r1", [])]),
                packed_report("8OG", [("r1", [0]), ("r1", [])]),
            )
            result = compare_hydrogen_bond_reports_file(path)
        self.assertEqual(result["technical_status"], "complete")
        row = result["group_comparisons"][0]
        self.assertAlmostEqual(row["condition_occupancies"]["control"], 0.5)
        self.assertAlmostEqual(row["condition_occupancies"]["lesion"], 0.5)

    def test_accepts_lazy_spatial_observed_union_with_packed_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = packed_report("DG", [("r1", [0]), ("r1", [])])
            lesion = packed_report("8OG", [("r1", [0]), ("r1", [0])])
            control["frame_matrix_representation"] = (
                "sparse_spatial_observed_union_v3"
            )
            lesion["frame_matrix_representation"] = (
                "sparse_spatial_observed_union_v3"
            )
            path = self.write_request(root, control, lesion)
            result = compare_hydrogen_bond_reports_file(path)
        self.assertEqual(result["technical_status"], "complete")
        self.assertEqual(result["error_count"], 0)
        row = result["group_comparisons"][0]
        self.assertAlmostEqual(row["condition_occupancies"]["control"], 0.5)
        self.assertAlmostEqual(row["condition_occupancies"]["lesion"], 1.0)

    def test_rejects_mismatched_interaction_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_request(
                root, report("DG", [("r1", [0])]),
                report("8OG", [("r1", [0])], scope="protein_protein"),
            )
            result = compare_hydrogen_bond_reports_file_safe(path)
        self.assertEqual(result["technical_status"], "failed")
        self.assertIn("interaction_scope differs", result["issues"][0]["message"])

    def test_rejects_unmatched_homolog_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_request(
                root, report("DG", [("r1", [0])]), report("8OG", [("r1", [0])]),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["homolog_mappings"][0]["match"]["residue_number"] = 999
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = compare_hydrogen_bond_reports_file_safe(path)
        self.assertEqual(result["technical_status"], "failed")
        self.assertIn("matched no donor or acceptor atom", result["issues"][0]["message"])

    def test_topology_specific_atom_is_null_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = report("DG", [("r1", [0])])
            lesion = report("8OG", [("r1", [0, 2])])
            lesion["atom_dictionary"].append({
                "atom_index": 4, "identity": atom(4, "8OG", "O8", "O")
            })
            lesion["candidate_dictionary"].append({
                "bond_id": "lesion-o8", "donor_atom_index": 0,
                "hydrogen_atom_index": 1, "acceptor_atom_index": 4,
            })
            path = self.write_request(root, control, lesion)
            result = compare_hydrogen_bond_reports_file(path)
        unique = next(
            row for row in result["group_comparisons"]
            if row["acceptor_identity"]["atom_name"] == "O8"
        )
        self.assertIsNone(unique["condition_occupancies"]["control"])
        self.assertEqual(unique["condition_occupancies"]["lesion"], 1.0)
        self.assertFalse(unique["chemically_comparable_between_conditions"])
        self.assertIsNone(unique["occupancy_difference_second_minus_first"])
        self.assertEqual(result["topology_specific_observed_group_count"], 1)
        self.assertNotIn(unique, result["top_absolute_differences"])

    def test_filters_conditions_from_one_multi_system_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            multi = report(
                "DG", [("r1", [0]), ("r1", []), ("r1", []), ("r1", [])]
            )
            for frame, system_id in zip(
                multi["frame_bond_matrix"], ["D0", "D0", "D1", "D1"]
            ):
                frame["system_id"] = system_id
            (root / "multi.json").write_text(json.dumps(multi), encoding="utf-8")
            request = {
                "comparison_id": "multi-system-filter",
                "conditions": [
                    {"condition_id": "control", "system_id": "D0", "report": "multi.json"},
                    {"condition_id": "lesion", "system_id": "D1", "report": "multi.json"},
                ],
                "expected_interaction_scope": "protein_nucleic_acid",
            }
            path = root / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            result = compare_hydrogen_bond_reports_file(path)
        row = result["group_comparisons"][0]
        self.assertEqual(result["condition_summaries"][0]["system_id"], "D0")
        self.assertEqual(result["condition_summaries"][1]["system_id"], "D1")
        self.assertEqual(row["condition_occupancies"], {"control": 0.5, "lesion": 0.0})

    def test_filters_lazy_spatial_multi_system_report_with_implicit_zeros(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            multi = packed_report(
                "DG", [("r1", [0]), ("r1", []), ("r1", []), ("r1", [])]
            )
            multi["frame_matrix_representation"] = (
                "sparse_spatial_observed_union_v3"
            )
            for frame, system_id in zip(
                multi["frame_bond_matrix"], ["D0", "D0", "D1", "D1"]
            ):
                frame["system_id"] = system_id
            (root / "multi.json").write_text(json.dumps(multi), encoding="utf-8")
            request = {
                "comparison_id": "lazy-multi-system-filter",
                "conditions": [
                    {
                        "condition_id": "control",
                        "system_id": "D0",
                        "report": "multi.json",
                    },
                    {
                        "condition_id": "lesion",
                        "system_id": "D1",
                        "report": "multi.json",
                    },
                ],
                "expected_interaction_scope": "protein_nucleic_acid",
            }
            path = root / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            result = compare_hydrogen_bond_reports_file(path)
        self.assertEqual(result["technical_status"], "complete")
        row = result["group_comparisons"][0]
        self.assertEqual(
            row["condition_occupancies"], {"control": 0.5, "lesion": 0.0}
        )

    def test_requires_system_id_for_multi_system_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            multi = report("DG", [("r1", [0]), ("r1", [])])
            multi["frame_bond_matrix"][0]["system_id"] = "D0"
            multi["frame_bond_matrix"][1]["system_id"] = "D1"
            path = self.write_request(root, multi, multi, homolog_mappings=[])
            result = compare_hydrogen_bond_reports_file_safe(path)
        self.assertEqual(result["technical_status"], "failed")
        self.assertIn("must declare system_id", result["issues"][0]["message"])

    def test_selects_full_per_system_feature_spaces_from_one_v2_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = report("DG", [("r1", [0]), ("r1", [])])
            lesion = report("8OG", [("r1", []), ("r1", [])])
            wrapper = dict(control)
            wrapper["system_feature_spaces"] = [
                {
                    "system_id": "D0",
                    **{
                        key: control[key] for key in (
                            "candidate_dictionary", "atom_dictionary",
                            "frame_bond_matrix",
                        )
                    },
                },
                {
                    "system_id": "D1",
                    **{
                        key: lesion[key] for key in (
                            "candidate_dictionary", "atom_dictionary",
                            "frame_bond_matrix",
                        )
                    },
                },
            ]
            for view in wrapper["system_feature_spaces"]:
                for frame in view["frame_bond_matrix"]:
                    frame["system_id"] = view["system_id"]
            (root / "multi-v2.json").write_text(json.dumps(wrapper), encoding="utf-8")
            request = {
                "conditions": [
                    {"condition_id": "control", "system_id": "D0", "report": "multi-v2.json"},
                    {"condition_id": "lesion", "system_id": "D1", "report": "multi-v2.json"},
                ],
            }
            path = root / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            result = compare_hydrogen_bond_reports_file(path)
        self.assertEqual(result["technical_status"], "complete")
        self.assertEqual(result["condition_summaries"][0]["candidate_count"], 2)
        row = result["group_comparisons"][0]
        self.assertEqual(
            row["condition_feature_status"],
            {"control": "observed", "lesion": "chemically_present_never_observed"},
        )

    def test_rejects_absent_requested_system_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            multi = report("DG", [("r1", [0]), ("r1", [])])
            for frame in multi["frame_bond_matrix"]:
                frame["system_id"] = "D0"
            (root / "multi.json").write_text(json.dumps(multi), encoding="utf-8")
            request = {
                "conditions": [
                    {"condition_id": "control", "system_id": "D0", "report": "multi.json"},
                    {"condition_id": "lesion", "system_id": "D1", "report": "multi.json"},
                ]
            }
            path = root / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            result = compare_hydrogen_bond_reports_file_safe(path)
        self.assertEqual(result["technical_status"], "failed")
        self.assertIn("requested system_id 'D1' is absent", result["issues"][0]["message"])


if __name__ == "__main__":
    unittest.main()
