import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from salsbury_md_analysis.ion_atmosphere import (
    _distance, _nearest_distances, ion_atmosphere_project,
)


def _atom(record, serial, name, residue, chain, number, x, element):
    return (
        f"{record:<6s}{serial:5d} {name:<4s} {residue:>3s} {chain:1s}{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}\n"
    )


def _write_project(root: Path) -> Path:
    cell = "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n"
    atoms = [
        _atom("ATOM", 1, "OD1", "ASP", "A", 1, 1.0, "O"),
        _atom("HETATM", 2, "K", "K", "I", 1, 3.0, "K"),
        _atom("HETATM", 3, "CL", "CL", "I", 2, 18.5, "CL"),
    ]
    (root / "topology.pdb").write_text(cell + "".join(atoms) + "END\n", encoding="ascii")
    trajectory = cell
    for model in (1, 2):
        trajectory += f"MODEL     {model:4d}\n" + "".join(atoms) + "ENDMDL\n"
    (root / "trajectory.pdb").write_text(trajectory + "END\n", encoding="ascii")
    (root / "system.json").write_text(json.dumps({
        "systems": [{"system_id": "ions", "replicas": [{
            "replica_id": "r1", "topology": "topology.pdb",
            "segments": [{"segment_id": "s1", "trajectory": "trajectory.pdb",
                          "timing": {"first_frame_time": 0, "frame_interval": 1, "unit": "ps"}}],
        }]}],
    }), encoding="utf-8")
    project = {
        "project_id": "ion-atmosphere-test", "analysis_profile": "standard_md_v1",
        "system_manifest": "system.json", "analysis_output_root": "outputs",
        "sampling_mode": "UNBIASED_MD", "coordinate_unit": "angstrom",
        "time_unit": "ps", "periodic_coordinate_policy": "allow_wrapped_diagnostic",
        "selections": {"alignment": {"preset": "all"}, "analysis": {"preset": "all"}},
        "definitions": {"ion_atmosphere": {
            "frame_stride": 1, "maximum_frames": 2,
            "shell_cutoffs_angstrom": [3.5, 5.0, 6.0],
            "ion_groups": [
                {"species": "K", "charge_class": "cation", "atom_indices": [1]},
                {"species": "CL", "charge_class": "anion", "atom_indices": [2]},
            ],
            "target_groups": [{"target_id": "all_solute", "atom_indices": [0]}],
        }},
        "requested_modules": ["ion_atmosphere"], "protected_locations": ["/protected/example"],
    }
    path = root / "project.json"
    path.write_text(json.dumps(project), encoding="utf-8")
    return path


class IonAtmosphereTests(unittest.TestCase):
    def test_vectorized_orthogonal_distances_match_exact_scalar_geometry(self):
        generator = np.random.default_rng(90210)
        ions = generator.uniform(-20.0, 20.0, size=(7, 3))
        targets = generator.uniform(-20.0, 20.0, size=(13, 3))
        cell = (
            (10.0, 10.0, 0.0),
            (-12.0, 12.0, 0.0),
            (0.0, 0.0, 18.0),
        )
        observed = _nearest_distances(
            ions, targets, cell, maximum_pairs_per_chunk=11,
        )
        expected = tuple(
            min(_distance(ion, target, cell) for target in targets)
            for ion in ions
        )
        np.testing.assert_allclose(observed, expected, rtol=1.0e-13, atol=1.0e-13)

    def test_skewed_triclinic_distances_retain_exact_scalar_enumeration(self):
        ions = ((9.0, 7.2, 0.0), (1.0, 2.0, 3.0))
        targets = ((0.0, 0.0, 0.0), (8.0, 1.0, 4.0))
        cell = ((10.0, 0.0, 0.0), (4.0, 8.0, 0.0), (1.0, 2.0, 9.0))
        observed = _nearest_distances(ions, targets, cell)
        expected = tuple(
            min(_distance(ion, target, cell) for target in targets)
            for ion in ions
        )
        self.assertEqual(observed, expected)

    def test_species_resolved_cation_and_anion_use_minimum_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = ion_atmosphere_project(_write_project(Path(temporary)))
        self.assertEqual(report["technical_status"], "complete")
        self.assertEqual(report["frame_selection"]["selected_frame_count"], 2)
        species = {row["species"] for row in report["species_target_shell_summaries"]}
        self.assertEqual(species, {"K", "CL"})
        first = report["frame_records"][0]["species"]
        self.assertAlmostEqual(
            first["CL"]["targets"]["all_solute"]["nearest_distance_angstrom"], 2.5
        )
        self.assertEqual(
            first["K"]["targets"]["all_solute"]["ion_count_within_shell"]["3.5"], 1
        )

    def test_identical_target_groups_reuse_exact_nearest_distances(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_path = _write_project(Path(temporary))
            project = json.loads(project_path.read_text(encoding="utf-8"))
            project["definitions"]["ion_atmosphere"]["target_groups"].append({
                "target_id": "same_atoms_different_label", "atom_indices": [0],
            })
            project_path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.ion_atmosphere._nearest_distances",
                wraps=_nearest_distances,
            ) as nearest:
                report = ion_atmosphere_project(project_path)
        self.assertEqual(nearest.call_count, 4)
        for frame in report["frame_records"]:
            for species in ("CL", "K"):
                targets = frame["species"][species]["targets"]
                self.assertEqual(
                    targets["all_solute"], targets["same_atoms_different_label"]
                )

    def test_local_periodic_distances_do_not_require_continuous_unwrapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            project_path = _write_project(Path(temporary))
            root = project_path.parent
            (root / "bonds.json").write_text(json.dumps({
                "format": "salsbury-bonds-v1", "atom_count": 3,
                "index_base": 0, "bonds": [[0, 1], [0, 2]],
            }), encoding="utf-8")
            system_path = root / "system.json"
            system = json.loads(system_path.read_text(encoding="utf-8"))
            system["systems"][0]["replicas"][0]["connectivity"] = "bonds.json"
            system_path.write_text(json.dumps(system), encoding="utf-8")
            project = json.loads(project_path.read_text(encoding="utf-8"))
            project["periodic_coordinate_policy"] = "unwrap_continuous"
            project["periodic_reconstruction"] = {
                "maximum_bond_length_angstrom": 20.0,
                "cycle_closure_tolerance_angstrom": 1.0e-6,
                "maximum_anchor_displacement_angstrom": 20.0,
            }
            project_path.write_text(json.dumps(project), encoding="utf-8")
            with patch(
                "salsbury_md_analysis.periodic.PeriodicFrameProcessor.from_replica"
            ) as reconstruct:
                report = ion_atmosphere_project(project_path)
            reconstruct.assert_not_called()
        self.assertEqual(report["technical_status"], "complete")
        self.assertAlmostEqual(
            report["frame_records"][0]["species"]["CL"]["targets"]
            ["all_solute"]["nearest_distance_angstrom"],
            2.5,
        )


if __name__ == "__main__":
    unittest.main()
