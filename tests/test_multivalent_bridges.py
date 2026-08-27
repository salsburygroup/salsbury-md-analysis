import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.multivalent_bridges import (
    multivalent_molecular_bridges_project,
    multivalent_molecular_bridges_project_safe,
)


def _atom(record, serial, name, residue, chain, number, x, element):
    return (
        f"{record:<6s}{serial:5d} {name:<4s} {residue:>3s} {chain:1s}{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _frame(k_position, *, mediator_residue="K", mediator_element="K"):
    return "".join([
        _atom("ATOM", 1, "CA", "ALA", "A", 1, 1.0, "C"),
        _atom("ATOM", 2, "CA", "GLY", "A", 2, 4.0, "C"),
        _atom("ATOM", 3, "CA", "VAL", "B", 1, 8.0, "C"),
        _atom(
            "HETATM", 4, mediator_element, mediator_residue, "M", 1,
            k_position, mediator_element,
        ),
    ])


def _write_project(
    root: Path, *, mediator_residue="K", mediator_element="K",
    include_supported_ions=True, mediator_residue_names=None,
):
    topology = _frame(
        8.0, mediator_residue=mediator_residue,
        mediator_element=mediator_element,
    )
    (root / "topology.pdb").write_text(topology + "END\n", encoding="ascii")
    trajectory = ""
    for model, position in enumerate((8.0, 2.5, 2.5, 8.0), start=1):
        trajectory += (
            f"MODEL     {model:4d}\n"
            + _frame(
                position, mediator_residue=mediator_residue,
                mediator_element=mediator_element,
            )
            + "ENDMDL\n"
        )
    (root / "trajectory.pdb").write_text(trajectory + "END\n", encoding="ascii")
    (root / "system.json").write_text(json.dumps({
        "systems": [{
            "system_id": "bridge-system",
            "replicas": [{
                "replica_id": "r1",
                "topology": "topology.pdb",
                "segments": [{
                    "segment_id": "s1",
                    "trajectory": "trajectory.pdb",
                    "timing": {
                        "first_frame_time": 0.0,
                        "frame_interval": 2.0,
                        "unit": "ps",
                    },
                }],
            }],
        }],
    }), encoding="utf-8")
    definition = {
        "frame_stride": 1,
        "frame_selection": {"mode": "fixed_stride_v1"},
        "maximum_frames": 4,
        "include_supported_ions": include_supported_ions,
        "include_recognized_waters": False,
        "mediator_residue_names": mediator_residue_names or [],
        "solute_residue_classes": ["protein"],
        "solute_residue_names": [],
        "mediator_atom_elements": [],
        "solute_atom_elements": [],
        "contact_cutoff_angstrom": 1.8,
        "water_contact_cutoff_angstrom": 3.5,
        "minimum_distinct_residues": 2,
        "maximum_neighbor_pairs_per_frame": 100,
        "maximum_bridge_records": 100,
        "minimum_evaluated_frames_per_system": 4,
    }
    project = {
        "project_id": "multivalent-bridge-test",
        "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json",
        "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD",
        "coordinate_unit": "angstrom",
        "time_unit": "ps",
        "periodic_coordinate_policy": "reject",
        "selections": {
            "alignment": {"preset": "all"},
            "analysis": {"preset": "all"},
        },
        "definitions": {"multivalent_molecular_bridges": definition},
        "requested_modules": ["multivalent_molecular_bridges"],
        "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class MultivalentMolecularBridgeTests(unittest.TestCase):
    def test_supported_ion_hyperedges_edges_and_segment_safe_residence(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = multivalent_molecular_bridges_project(
                _write_project(Path(temporary))
            )
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(len(report["bridge_hyperedges"]), 2)
        self.assertEqual(
            {row["distinct_residue_count"] for row in report["bridge_hyperedges"]},
            {2},
        )
        self.assertEqual(len(report["projected_residue_edges"]), 1)
        edge = report["projected_residue_edges"][0]
        self.assertAlmostEqual(edge["bridge_occupancy"], 0.5)
        self.assertEqual(edge["bridge_residence"]["event_count"], 1)
        self.assertEqual(edge["bridge_residence"]["complete_event_count"], 1)
        mediator = report["mediator_summaries"][0]
        self.assertAlmostEqual(mediator["bridge_occupancy"], 0.5)
        mediator_type = report["mediator_type_summaries"][0]
        self.assertAlmostEqual(mediator_type["frame_bridge_occupancy"], 0.5)
        self.assertAlmostEqual(
            mediator_type["mean_active_bridge_mediators_per_frame"], 0.5
        )
        self.assertEqual(
            mediator["bridge_residence"][
                "complete_selected_observation_count_summary"
            ]["mean"],
            2.0,
        )
        self.assertEqual(
            mediator["bridge_residence"]["complete_axis_span_summary"]["mean"],
            2.0,
        )

    def test_explicit_ligand_residue_can_be_the_mediator(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = multivalent_molecular_bridges_project(
                _write_project(
                    Path(temporary), mediator_residue="NEO",
                    mediator_element="N", include_supported_ions=False,
                    mediator_residue_names=["NEO"],
                )
            )
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(
            report["mediator_summaries"][0]["mediator"]["mediator_kind"],
            "declared_residue",
        )
        self.assertEqual(
            report["mediator_type_summaries"][0]["mediator_type"], "NEO"
        )

    def test_recognized_water_is_available_as_a_solvent_mediator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = json.loads(
                _write_project(root).read_text(encoding="utf-8")
            )
            atoms = "".join([
                _atom("ATOM", 1, "OD1", "ASP", "A", 1, 1.0, "O"),
                _atom("ATOM", 2, "OE1", "GLU", "A", 2, 5.0, "O"),
                _atom("HETATM", 3, "O", "HOH", "W", 1, 3.0, "O"),
            ])
            (root / "topology.pdb").write_text(atoms + "END\n", encoding="ascii")
            (root / "trajectory.pdb").write_text(
                "MODEL        1\n" + atoms + "ENDMDL\nEND\n", encoding="ascii"
            )
            (root / "system.json").write_text(json.dumps({
                "systems": [{"system_id": "water", "replicas": [{
                    "replica_id": "r1", "topology": "topology.pdb",
                    "segments": [{
                        "segment_id": "s1", "trajectory": "trajectory.pdb",
                        "timing": {
                            "first_frame_time": 0, "frame_interval": 1,
                            "unit": "ps",
                        },
                    }],
                }]}],
            }), encoding="utf-8")
            project["system_manifest"] = "system.json"
            definition = project["definitions"]["multivalent_molecular_bridges"]
            definition.update({
                "maximum_frames": 1,
                "minimum_evaluated_frames_per_system": 1,
                "include_supported_ions": False,
                "include_recognized_waters": True,
                "mediator_residue_names": [],
                "solute_atom_elements": ["O"],
                "water_contact_cutoff_angstrom": 2.1,
            })
            path = root / "water-project.json"
            path.write_text(json.dumps(project), encoding="utf-8")
            report = multivalent_molecular_bridges_project(path)
        self.assertEqual(len(report["bridge_hyperedges"]), 1)
        self.assertEqual(
            report["bridge_hyperedges"][0]["mediator"]["mediator_kind"],
            "recognized_water",
        )
        self.assertEqual(
            report["mediator_type_summaries"][0]["mediator_type"], "WATER"
        )

    def test_missing_configured_mediator_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_project(Path(temporary))
            payload = json.loads(path.read_text(encoding="utf-8"))
            definition = payload["definitions"]["multivalent_molecular_bridges"]
            definition["include_supported_ions"] = False
            definition["mediator_residue_names"] = ["NEO"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = multivalent_molecular_bridges_project_safe(path)
        self.assertEqual(report["technical_status"], "failed")
        self.assertIn(
            "no configured mediator",
            report["issues"][0]["message"],
        )

    def test_comparison_retains_a_system_with_no_mediator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _write_project(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            definition = payload["definitions"]["multivalent_molecular_bridges"]
            definition["maximum_frames"] = 8
            payload["reference_system"] = "bridge-system"
            solute = "".join([
                _atom("ATOM", 1, "CA", "ALA", "A", 1, 1.0, "C"),
                _atom("ATOM", 2, "CA", "GLY", "A", 2, 4.0, "C"),
                _atom("ATOM", 3, "CA", "VAL", "B", 1, 8.0, "C"),
            ])
            (root / "absent-topology.pdb").write_text(
                solute + "END\n", encoding="ascii"
            )
            (root / "absent-trajectory.pdb").write_text(
                "".join(
                    f"MODEL     {model:4d}\n{solute}ENDMDL\n"
                    for model in range(1, 5)
                ) + "END\n",
                encoding="ascii",
            )
            system_path = root / "system.json"
            systems = json.loads(system_path.read_text(encoding="utf-8"))
            systems["systems"].append({
                "system_id": "absent-system",
                "replicas": [{
                    "replica_id": "r1",
                    "topology": "absent-topology.pdb",
                    "segments": [{
                        "segment_id": "s1",
                        "trajectory": "absent-trajectory.pdb",
                        "timing": {
                            "first_frame_time": 0.0,
                            "frame_interval": 2.0,
                            "unit": "ps",
                        },
                    }],
                }],
            })
            system_path.write_text(json.dumps(systems), encoding="utf-8")
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = multivalent_molecular_bridges_project(path)
        summaries = {
            row["system_id"]: row for row in report["system_summaries"]
        }
        self.assertEqual(
            summaries["absent-system"]["topology_mediator_type_counts_across_replicas"],
            {},
        )
        self.assertEqual(
            summaries["absent-system"]["bridge_hyperedge_record_count"], 0
        )
        self.assertGreater(
            summaries["bridge-system"]["bridge_hyperedge_record_count"], 0
        )


if __name__ == "__main__":
    unittest.main()
