import tempfile
import unittest
from pathlib import Path

from salsbury_md_analysis.automatic_chemistry import (
    infer_standard_chemistry_definitions,
)


def _atom(serial, name, residue, chain, number, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {name:<4s} {residue:>3s} {chain:1s}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n"
    )


class AutomaticChemistryTests(unittest.TestCase):
    def test_rna_histidine_water_and_iron_aliases_route_to_expected_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "generic-chemistry.pdb"
            path.write_text(
                "".join([
                    _atom(1, "N", "HSD", "A", 1, 0.0, 0.0, 0.0, "N"),
                    _atom(2, "O", "HSD", "A", 1, 1.0, 0.0, 0.0, "O"),
                    _atom(3, "P", "RA", "R", 2, 0.0, 2.0, 0.0, "P"),
                    _atom(4, "C1'", "RA", "R", 2, 1.0, 2.0, 0.0, "C"),
                    _atom(5, "N9", "RA", "R", 2, 1.0, 3.0, 0.0, "N"),
                    _atom(6, "C8", "RA", "R", 2, 2.0, 3.0, 0.0, "C"),
                    _atom(7, "N7", "RA", "R", 2, 2.0, 4.0, 0.0, "N"),
                    _atom(8, "C5", "RA", "R", 2, 1.0, 4.0, 0.0, "C"),
                    _atom(9, "FE", "FE2", "I", 3, 2.0, 0.0, 0.0, "FE"),
                    _atom(10, "O", "OPC", "W", 4, 8.0, 8.0, 8.0, "O"),
                    "END\n",
                ]),
                encoding="utf-8",
            )

            report = infer_standard_chemistry_definitions(
                path, maximum_frames_by_module={}, total_source_frames=10
            )

            atmosphere = report["definitions"]["ion_atmosphere"]
            self.assertEqual(
                [row["species"] for row in atmosphere["ion_groups"]], ["FE"]
            )
            targets = {
                row["target_id"]: row["atom_indices"]
                for row in atmosphere["target_groups"]
            }
            self.assertEqual(targets["protein"], [0, 1])
            self.assertIn(2, targets["nucleic_acid"])
            self.assertNotIn(9, targets["all_solute"])
            self.assertIn("nucleic_acid_geometry", report["applicable_modules"])

    def test_bulk_mobile_ions_are_not_expanded_into_binding_sites(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ions.pdb"
            path.write_text(
                "".join([
                    _atom(1, "OD1", "ASP", "A", 1, 0.0, 0.0, 0.0, "O"),
                    _atom(2, "OD2", "ASP", "A", 1, 0.0, 2.0, 0.0, "O"),
                    _atom(3, "MG", "MG", "I", 1, 2.0, 0.0, 0.0, "MG"),
                    _atom(4, "NA", "NA", "I", 2, 20.0, 0.0, 0.0, "NA"),
                    _atom(5, "O", "WAT", "W", 1, 3.0, 0.0, 0.0, "O"),
                    "END\n",
                ]),
                encoding="utf-8",
            )
            report = infer_standard_chemistry_definitions(
                path,
                maximum_frames_by_module={},
                total_source_frames=100,
                ion_site_classification_enabled=True,
            )
            inference = report["inference"]
            self.assertEqual(inference["retained_ion_candidate_count"], 1)
            self.assertEqual(inference["excluded_bulk_mobile_ion_count"], 1)
            sites = report["definitions"]["ion_coordination_geometry"]["ion_sites"]
            self.assertEqual([row["site_id"] for row in sites], ["mg-1"])
            self.assertEqual(
                report["definitions"]["ion_coordination_geometry"]["ion_pairs"],
                [],
            )
            statuses = {
                row["site_id"]: row["screening_status"]
                for row in inference["ion_candidates"]
            }
            self.assertEqual(
                statuses,
                {
                    "mg-1": "retained_trajectory_binding_candidate",
                    "na-2": "excluded_bulk_mobile_candidate",
                },
            )
            requested = {
                row["site_id"]: row["trajectory_classification_requested"]
                for row in inference["ion_candidates"]
            }
            self.assertEqual(requested, {"mg-1": True, "na-2": False})
            atmosphere = report["definitions"]["ion_atmosphere"]
            self.assertEqual(
                {row["species"] for row in atmosphere["ion_groups"]},
                {"MG", "NA"},
            )
            self.assertIn("ion_atmosphere", report["applicable_modules"])


if __name__ == "__main__":
    unittest.main()
