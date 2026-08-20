import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.state_coordinate_exports import (
    _member_payload_map,
    state_coordinate_exports_project,
    state_coordinate_exports_project_safe,
)
from salsbury_md_analysis.atom_mapping import AtomRecord

from tests.test_pca_fes import _write_ai_project


def _source_report() -> dict:
    rows = []
    for frame_index in range(6):
        state = 1 if frame_index < 3 else 2
        rows.append({
            "system_id": "ai",
            "replica_id": "r1",
            "segment_id": "samples",
            "source_frame_index": frame_index,
            "sample_index": frame_index,
            "cluster_id": state,
            "squared_distance_in_clustering_space": float(frame_index % 3),
        })
    return {
        "technical_status": "complete",
        "contract_signature_sha256": "d" * 64,
        "assignments": rows,
        "issues": [],
    }


def _export_project(root: Path) -> Path:
    path = _write_ai_project(root)
    project = json.loads(path.read_text(encoding="utf-8"))
    project["definitions"]["state_coordinate_exports"] = {
        "source": "clustering_kmeans",
        "export_id": "clusters-v1",
        "trajectory_format": "xyz",
        "representatives_per_state": 1,
        "frame_stride_within_state": 1,
        "maximum_states": 4,
        "maximum_frames_per_state": 10,
        "maximum_total_frames": 20,
        "existing_output_policy": "fail",
        "coordinate_selection": "analysis",
    }
    project["requested_modules"].append("state_coordinate_exports")
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class StateCoordinateExportTests(unittest.TestCase):
    def test_member_payload_keeps_hydrogens_and_chain_heteroatoms_but_not_water(self):
        atoms = []
        for chain in ("A", "B"):
            for residue_number, residue_name, names in (
                (1, "ALA", (("N", "N"), ("H", "H"), ("CA", "C"))),
                (2, "LIG", (("C1", "C"),)),
                (3, "WAT", (("O", "O"), ("H1", "H"), ("H2", "H"))),
            ):
                for atom_name, element in names:
                    index = len(atoms)
                    atoms.append(AtomRecord(
                        atom_index=index,
                        serial=index + 1,
                        atom_name=atom_name,
                        altloc="",
                        residue_name=residue_name,
                        chain_id=chain,
                        residue_number=residue_number,
                        insertion_code="",
                        element=element,
                    ))
        plan = {
            "members": [
                {"member_id": "member-1", "protein_chain_id": "A", "nucleic_chain_ids": []},
                {"member_id": "member-2", "protein_chain_id": "B", "nucleic_chain_ids": []},
            ]
        }
        first_identity, first_indices = _member_payload_map(
            atoms, plan, "member-1", policy="strict"
        )
        second_identity, second_indices = _member_payload_map(
            atoms, plan, "member-2", policy="strict"
        )
        self.assertEqual(first_identity, second_identity)
        self.assertEqual(len(first_indices), 4)
        self.assertEqual(len(second_indices), 4)
        self.assertIn("H", {atoms[index].element for index in first_indices})
        self.assertIn("LIG", {atoms[index].residue_name for index in first_indices})
        self.assertNotIn("WAT", {atoms[index].residue_name for index in first_indices})

    def test_exports_state_trajectories_representatives_and_checksums_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _export_project(root)
            with patch(
                "salsbury_md_analysis.state_coordinate_exports.clustering_kmeans_project",
                return_value=_source_report(),
            ):
                report = state_coordinate_exports_project(path, hash_content=True)
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["state_count"], 2)
            self.assertEqual(report["exported_frame_count"], 6)
            self.assertEqual(report["coordinate_files_written"], 4)
            export = Path(report["export_directory"])
            self.assertTrue((export / "export-manifest.json").is_file())
            trajectories = sorted(export.rglob("trajectory.xyz"))
            representatives = sorted(export.rglob("representative-01.pdb"))
            self.assertEqual((len(trajectories), len(representatives)), (2, 2))
            self.assertTrue(all(path.read_text().count("\n1\n") >= 2 for path in trajectories))
            self.assertTrue(all(path.read_text().endswith("END\n") for path in representatives))
            self.assertTrue(all(path.read_text().count("ATOM") == 1 for path in representatives))
            self.assertEqual(report["coordinate_selection"], "analysis")

            with patch(
                "salsbury_md_analysis.state_coordinate_exports.clustering_kmeans_project",
                return_value=_source_report(),
            ):
                second = state_coordinate_exports_project_safe(path)
            self.assertEqual(second["technical_status"], "failed")
            self.assertIn("overwrite is prohibited", second["issues"][0]["message"])

    def test_trajectory_writing_can_be_disabled_without_disabling_representatives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _export_project(root)
            project = json.loads(path.read_text(encoding="utf-8"))
            project["definitions"]["state_coordinate_exports"][
                "write_trajectories"
            ] = False
            path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.state_coordinate_exports.clustering_kmeans_project",
                return_value=_source_report(),
            ):
                report = state_coordinate_exports_project(path, hash_content=True)
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["exported_frame_count"], 0)
            self.assertEqual(report["representative_count"], 2)
            self.assertEqual(report["coordinate_files_written"], 2)
            export = Path(report["export_directory"])
            self.assertEqual(len(list(export.rglob("trajectory.xyz"))), 0)
            self.assertEqual(len(list(export.rglob("representative-01.pdb"))), 2)

    def test_cross_system_export_uses_shared_partial_alignment_basis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _export_project(root)
            project = json.loads(path.read_text(encoding="utf-8"))
            project["definitions"]["common_pca"]["minimum_reference_coverage"] = 2 / 3
            path.write_text(json.dumps(project), encoding="utf-8")

            # The second system lacks reference alignment atom O, but retains
            # the common C/N alignment basis and its own CB molecular payload.
            (root / "variant.pdb").write_text(
                "".join([
                    "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n",
                    "ATOM      2  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N\n",
                    "ATOM      3  CB  ALA A   1       0.000   2.000   0.000  1.00  0.00           C\n",
                    "END\n",
                ]),
                encoding="utf-8",
            )
            (root / "variant.xyz").write_text(
                "3\nvariant-0\nC 0 0 0\nN 1 0 0\nC 0 2 0\n",
                encoding="utf-8",
            )
            system_path = root / "system.json"
            system = json.loads(system_path.read_text(encoding="utf-8"))
            system["systems"].append({
                "system_id": "variant",
                "replicas": [{
                    "replica_id": "r1",
                    "topology": "variant.pdb",
                    "segments": [{
                        "segment_id": "samples",
                        "trajectory": "variant.xyz",
                        "sample_axis": {"first_sample_index": 0, "sample_interval": 1},
                    }],
                }],
            })
            system_path.write_text(json.dumps(system), encoding="utf-8")
            source = _source_report()
            source["assignments"].append({
                "system_id": "variant",
                "replica_id": "r1",
                "segment_id": "samples",
                "source_frame_index": 0,
                "sample_index": 6,
                "cluster_id": 1,
                "squared_distance_in_clustering_space": 0.0,
            })
            with patch(
                "salsbury_md_analysis.state_coordinate_exports.clustering_kmeans_project",
                return_value=source,
            ):
                report = state_coordinate_exports_project(path, hash_content=True)
            self.assertEqual(report["technical_status"], "complete")
            mappings = report["alignment_mapping"]["replicas"]
            self.assertEqual(len(mappings), 2)
            self.assertEqual({row["mapped_atom_count"] for row in mappings}, {2})
            self.assertEqual(
                {round(row["reference_coverage"], 6) for row in mappings},
                {round(2 / 3, 6)},
            )

    def test_member_export_carries_exact_alignment_mapping_into_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _export_project(root)
            project = json.loads(path.read_text(encoding="utf-8"))
            project["definitions"]["common_pca"]["symmetry_expansion"] = {
                "applicable": True,
                "members": [
                    {"member_id": "member-1"},
                    {"member_id": "member-2"},
                ],
            }
            path.write_text(json.dumps(project), encoding="utf-8")
            source = _source_report()
            source["assignments"] = [
                {**row, "member_id": member_id}
                for row in source["assignments"]
                for member_id in ("member-1", "member-2")
            ]
            atom = AtomRecord(
                atom_index=0,
                serial=1,
                atom_name="CB",
                altloc="",
                residue_name="ALA",
                chain_id="A",
                residue_number=1,
                insertion_code="",
                element="C",
            )
            mapping = [{
                "system_id": "ai",
                "replica_id": "r1",
                "member_id": "member-1",
                "canonical_reference_member_id": "member-1",
                "selection_id": "oligomer_member_alignment",
                "mapped_atom_count": 3,
                "mapping_signature_sha256": "a" * 64,
            }]

            def capture(_project, _project_path, _system_path, requested, _plan):
                return (
                    {key: ((0.0, 0.0, 0.0),) for key in requested},
                    {("ai", "r1"): ([atom], root / "reference.pdb")},
                    {("ai", "r1", "samples"): 0},
                    mapping,
                )

            with patch(
                "salsbury_md_analysis.state_coordinate_exports.clustering_kmeans_project",
                return_value=source,
            ), patch(
                "salsbury_md_analysis.state_coordinate_exports._capture_member_coordinates",
                side_effect=capture,
            ):
                report = state_coordinate_exports_project(path, hash_content=True)
            self.assertEqual(report["technical_status"], "complete")
            self.assertTrue(report["observation_accounting"]["symmetry_expanded"])
            self.assertEqual(
                report["alignment_mapping"]["mode"],
                "equivalent_oligomer_member_to_canonical_member",
            )
            self.assertEqual(report["alignment_mapping"]["replicas"], mapping)
            self.assertEqual(
                report["outputs"][0]["pooled_member_ids"],
                ["member-1", "member-2"],
            )


if __name__ == "__main__":
    unittest.main()
