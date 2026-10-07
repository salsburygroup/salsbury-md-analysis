import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.state_coordinate_exports import (
    _member_payload_map,
    StateCoordinateExportError,
    state_coordinate_exports_project,
    state_coordinate_exports_project_safe,
)
from salsbury_md_analysis.atom_mapping import AtomRecord

from tests.test_pca_fes import _write_ai_project
from tests.test_quickstart import _write_oligomer_inputs
from salsbury_md_analysis.quickstart import _composition
from salsbury_md_analysis.atom_mapping import read_topology_atoms
from salsbury_md_analysis.coordinates import iter_coordinate_frames


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
    def test_member_exports_preserve_system_chemistry_and_replica_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, source = _chemical_member_fixture(root)
            originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
            with patch("salsbury_md_analysis.state_coordinate_exports.clustering_kmeans_project", return_value=source):
                report = state_coordinate_exports_project(path, hash_content=True)
            self.assertEqual(report["technical_status"], "complete")
            self.assertEqual(report["exported_frame_count"], 16)
            for system_id, names, count in (("control", {"H8"}, 9), ("lesion", {"H7", "O8"}, 10)):
                outputs = [r for r in report["outputs"] if r["system_id"] == system_id]
                self.assertTrue(outputs)
                for output in outputs:
                    self.assertEqual(output["pooled_member_ids"], ["member-1", "member-2"])
                    self.assertEqual(output["pooled_replica_ids"], ["r1", "r2"])
                    self.assertEqual(
                        {(r["replica_id"], r["member_id"], r["source_frame_index"]) for r in output["trajectory_frame_provenance"]},
                        {(replica, member, output["state_id"] - 1) for replica in ("r1", "r2") for member in ("member-1", "member-2")},
                    )
                files = list(Path(report["export_directory"]).rglob(f"*{system_id}*/trajectory.pdb"))
                self.assertEqual(len(files), 2)
                for filename in files:
                    _, atoms = read_topology_atoms(filename)
                    self.assertEqual(len(atoms), count)
                    self.assertTrue(names.issubset({a.atom_name for a in atoms}))
                    self.assertNotIn("H8" if system_id == "lesion" else "O8", {a.atom_name for a in atoms})
                    frames = list(iter_coordinate_frames(filename, "angstrom"))
                    self.assertEqual(len(frames), 4)
                    for frame in frames:
                        for atom, coordinate in zip(atoms, frame.coordinates_angstrom):
                            if atom.atom_name in names:
                                self.assertAlmostEqual(coordinate[0], 3.0, places=3)
                                self.assertAlmostEqual(coordinate[1], 4.0, places=3)
            for filename, digest in originals.items():
                self.assertEqual(hashlib.sha256(filename.read_bytes()).hexdigest(), digest)

    def test_member_payload_mismatch_within_system_still_fails(self):
        for change in ("missing_atom", "residue_identity", "member_identity"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, source = _chemical_member_fixture(root)
                topology = root / "lesion-r2.pdb"
                text = topology.read_text()
                if change == "missing_atom":
                    text = "\n".join(line for line in text.splitlines() if " O8 " not in line) + "\n"
                elif change == "residue_identity":
                    text = text.replace("8OG", " DG")
                else:
                    text = "\n".join(
                        line.replace("8OG", " DG") if line.startswith("ATOM") and line[21] == "D" else line
                        for line in text.splitlines()
                    ) + "\n"
                topology.write_text(text)
                with patch("salsbury_md_analysis.state_coordinate_exports.clustering_kmeans_project", return_value=source):
                    with self.assertRaisesRegex(ValueError, "canonical member topology|atom count"):
                        state_coordinate_exports_project(path)
                self.assertFalse((root / "outputs" / "08_clustering").exists())

    def test_diagnostic_never_computes_upstream_or_writes_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _export_project(root)
            phases = []
            with patch("salsbury_md_analysis.state_coordinate_exports.load_cached_project_report", return_value=_source_report()), patch("salsbury_md_analysis.state_coordinate_exports.clustering_kmeans_project", side_effect=AssertionError("must not rebuild")):
                report = state_coordinate_exports_project(path, diagnostic_only=True, phase_observer=phases.append)
            self.assertEqual(report["captured_observations"], 6)
            self.assertEqual(report["coordinate_files_written"], 0)
            self.assertEqual(phases[-1], "diagnostic_complete")
            self.assertIn("coordinate_capture_and_alignment", phases)
            self.assertFalse((root / "outputs").exists())
            with patch("salsbury_md_analysis.state_coordinate_exports.load_cached_project_report", return_value=None):
                with self.assertRaisesRegex(StateCoordinateExportError, "upstream recomputation is prohibited"):
                    state_coordinate_exports_project(path, diagnostic_only=True)

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


def _chemical_member_fixture(root):
    path = _export_project(root)
    reference, _, _ = _write_oligomer_inputs(root)
    plan = _composition(reference)["conformational_view_plan"]["equivalent_oligomer"]
    base = reference.read_text().splitlines()[:-1]
    systems, assignments = [], []
    for system_id, payload in (("control", (("H8", "H"),)), ("lesion", (("H7", "H"), ("O8", "O")))):
        replicas = []
        for replica_id in ("r1", "r2"):
            rows = list(base)
            for chain, offset in (("C", 0), ("D", 30)):
                for name, element in payload:
                    serial = len(rows) + 1
                    rows.append(f"ATOM  {serial:5d} {name:^4s} {'DG':>3s} {chain}{10:4d}    {offset+3:8.3f}{4:8.3f}{0:8.3f}  1.00  0.00          {element:>2s}")
            if system_id == "lesion":
                rows = [line.replace(" DG ", "8OG ") for line in rows]
            if replica_id == "r2":
                rows.reverse()  # Coordinate order may vary; chemical identities may not.
            topology = root / f"{system_id}-{replica_id}.pdb"
            topology.write_text("\n".join(rows) + "\nEND\n")
            trajectory = topology.with_suffix(".xyz")
            trajectory.write_text("".join(
                f"{len(rows)}\nframe-{frame}\n" + "".join(
                    f"{line[76:78].strip()} {float(line[30:38])+frame} {line[38:46]} {line[46:54]}\n"
                    for line in rows
                ) for frame in range(2)
            ))
            replicas.append({"replica_id": replica_id, "topology": topology.name, "segments": [{"segment_id": "samples", "trajectory": trajectory.name, "sample_axis": {"first_sample_index": 0, "sample_interval": 1}}]})
            for frame in range(2):
                for member in ("member-1", "member-2"):
                    assignments.append({"system_id": system_id, "replica_id": replica_id, "segment_id": "samples", "source_frame_index": frame, "sample_index": frame, "member_id": member, "cluster_id": frame + 1, "squared_distance_in_clustering_space": 0.0})
        systems.append({"system_id": system_id, "replicas": replicas})
    (root / "system.json").write_text(json.dumps({"systems": systems}))
    project = json.loads(path.read_text())
    project.update(reference_structure="control-r1.pdb", reference_system="control", common_atom_policy="position")
    project["selections"] = {"alignment": {"atom_names": ["N", "CA", "C", "O"]}, "analysis": {"atom_names": ["CB"]}, "molecular_payload": {"preset": "molecular_payload"}}
    project["definitions"]["common_pca"]["symmetry_expansion"] = plan
    project["definitions"]["state_coordinate_exports"].update(coordinate_selection="molecular_payload", trajectory_format="pdb")
    path.write_text(json.dumps(project))
    return path, {**_source_report(), "assignments": assignments}


if __name__ == "__main__":
    unittest.main()
