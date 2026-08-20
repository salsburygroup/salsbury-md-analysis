import json
import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.ion_atmosphere import ion_atmosphere_project


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


if __name__ == "__main__":
    unittest.main()
