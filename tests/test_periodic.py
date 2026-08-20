import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from salsbury_md_analysis.coordinates import CoordinateFrame
from salsbury_md_analysis.periodic import (
    PeriodicFrameProcessor,
    PeriodicReconstructionError,
    load_connectivity,
    make_whole_coordinates,
    minimum_image_displacement,
)


ORTHORHOMBIC = ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0))
TRICLINIC = ((10.0, 0.0, 0.0), (4.0, 8.0, 0.0), (1.0, 2.0, 9.0))


class PeriodicReconstructionTests(unittest.TestCase):
    def test_make_whole_rebuilds_boundary_crossing_bond(self):
        rebuilt = make_whole_coordinates(
            ((9.5, 1.0, 1.0), (0.5, 1.0, 1.0)),
            ORTHORHOMBIC,
            ((0, 1),),
            maximum_bond_length_angstrom=2.0,
            cycle_closure_tolerance_angstrom=1.0e-6,
        )
        self.assertEqual(rebuilt[0], (9.5, 1.0, 1.0))
        self.assertAlmostEqual(rebuilt[1][0], 10.5)

    def test_selective_make_whole_rebuilds_complete_required_component_only(self):
        coordinates = (
            (9.5, 1.0, 1.0), (0.5, 1.0, 1.0),
            (9.5, 4.0, 1.0), (0.5, 4.0, 1.0),
        )
        rebuilt = make_whole_coordinates(
            coordinates,
            ORTHORHOMBIC,
            ((0, 1), (2, 3)),
            maximum_bond_length_angstrom=2.0,
            cycle_closure_tolerance_angstrom=1.0e-6,
            required_atom_indices=(0,),
        )
        self.assertAlmostEqual(rebuilt[1][0], 10.5)
        self.assertEqual(rebuilt[2:], coordinates[2:])

    def test_processor_reports_selective_reconstruction_scope(self):
        processor = PeriodicFrameProcessor(
            "make_whole",
            4,
            ((0, 1), (2, 3)),
            {
                "maximum_bond_length_angstrom": 2.0,
                "cycle_closure_tolerance_angstrom": 1.0e-6,
            },
        )
        observed = processor.process(
            CoordinateFrame(
                0,
                ((9.5, 1.0, 1.0), (0.5, 1.0, 1.0), (9.5, 4.0, 1.0), (0.5, 4.0, 1.0)),
                "angstrom", True, ORTHORHOMBIC,
            ),
            "frame-0",
            required_atom_indices=(0,),
        )
        self.assertAlmostEqual(observed.coordinates_angstrom[1][0], 10.5)
        self.assertEqual(observed.coordinates_angstrom[2][0], 9.5)
        self.assertEqual(processor.report()["reconstructed_component_count"], 1)
        self.assertEqual(processor.report()["reconstructed_atom_count"], 2)

    def test_processor_preserves_compact_dcd_array_storage(self):
        processor = PeriodicFrameProcessor(
            "make_whole",
            4,
            ((0, 1), (2, 3)),
            {
                "maximum_bond_length_angstrom": 2.0,
                "cycle_closure_tolerance_angstrom": 1.0e-6,
            },
        )
        observed = processor.process(
            CoordinateFrame(
                0,
                np.asarray(
                    (
                        (9.5, 1.0, 1.0), (0.5, 1.0, 1.0),
                        (9.5, 4.0, 1.0), (0.5, 4.0, 1.0),
                    ),
                    dtype=float,
                ),
                "angstrom", True, ORTHORHOMBIC,
            ),
            "frame-0",
            required_atom_indices=(0,),
        )
        self.assertIsInstance(observed.coordinates_angstrom, np.ndarray)
        self.assertAlmostEqual(observed.coordinates_angstrom[1, 0], 10.5)
        self.assertAlmostEqual(observed.coordinates_angstrom[2, 0], 9.5)

    def test_preprocessed_policy_accepts_only_verified_cache_frames(self):
        processor = PeriodicFrameProcessor(
            "preprocessed_make_whole",
            2,
            preprocessed_cache_identity={"cache_report_sha256": "a" * 64},
        )
        cached = processor.process(
            CoordinateFrame(
                0,
                ((9.5, 1.0, 1.0), (10.5, 1.0, 1.0)),
                "angstrom",
                True,
                ORTHORHOMBIC,
                "made_whole_molecular_payload_cache",
            ),
            "cache/frame-0",
        )
        self.assertEqual(
            cached.coordinate_representation, "preprocessed_make_whole"
        )
        self.assertEqual(cached.coordinates_angstrom[1][0], 10.5)
        with self.assertRaisesRegex(
            PeriodicReconstructionError, "not a declared made-whole"
        ):
            processor.process(
                CoordinateFrame(
                    1,
                    ((9.5, 1.0, 1.0), (0.5, 1.0, 1.0)),
                    "angstrom",
                    True,
                    ORTHORHOMBIC,
                ),
                "raw/frame-1",
            )

    def test_triclinic_minimum_image_is_nearest_lattice_vector(self):
        displacement = tuple(
            0.9 * TRICLINIC[0][axis] + 0.9 * TRICLINIC[1][axis]
            for axis in range(3)
        )
        observed = minimum_image_displacement(displacement, TRICLINIC)
        expected = tuple(
            -0.1 * TRICLINIC[0][axis] - 0.1 * TRICLINIC[1][axis]
            for axis in range(3)
        )
        for actual, wanted in zip(observed, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_continuous_unwrap_tracks_component_anchor_and_segment_reset(self):
        processor = PeriodicFrameProcessor(
            "unwrap_continuous",
            2,
            ((0, 1),),
            {
                "maximum_bond_length_angstrom": 2.0,
                "cycle_closure_tolerance_angstrom": 1.0e-6,
                "maximum_anchor_displacement_angstrom": 2.0,
            },
        )
        processor.begin_segment(False)
        first = processor.process(
            CoordinateFrame(0, ((9.5, 1.0, 1.0), (0.5, 1.0, 1.0)), "angstrom", True, ORTHORHOMBIC),
            "segment-1/frame-0",
        )
        second = processor.process(
            CoordinateFrame(1, ((0.2, 1.0, 1.0), (1.2, 1.0, 1.0)), "angstrom", True, ORTHORHOMBIC),
            "segment-1/frame-1",
        )
        self.assertEqual(first.coordinate_representation, "unwrap_continuous")
        self.assertAlmostEqual(second.coordinates_angstrom[0][0], 10.2)
        self.assertAlmostEqual(second.coordinates_angstrom[1][0], 11.2)
        processor.begin_segment(False)
        reset = processor.process(
            CoordinateFrame(0, ((0.2, 1.0, 1.0), (1.2, 1.0, 1.0)), "angstrom", True, ORTHORHOMBIC),
            "segment-2/frame-0",
        )
        self.assertAlmostEqual(reset.coordinates_angstrom[0][0], 0.2)

    def test_cycle_inconsistent_with_single_whole_image_fails(self):
        with self.assertRaises(PeriodicReconstructionError):
            make_whole_coordinates(
                ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (8.0, 0.0, 0.0)),
                ORTHORHOMBIC,
                ((0, 1), (1, 2), (2, 0)),
                maximum_bond_length_angstrom=5.0,
                cycle_closure_tolerance_angstrom=1.0e-6,
            )

    def test_anchor_displacement_gate_fails_closed(self):
        processor = PeriodicFrameProcessor(
            "unwrap_continuous",
            2,
            ((0, 1),),
            {
                "maximum_bond_length_angstrom": 2.0,
                "cycle_closure_tolerance_angstrom": 1.0e-6,
                "maximum_anchor_displacement_angstrom": 0.1,
            },
        )
        processor.begin_segment(False)
        processor.process(
            CoordinateFrame(0, ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0)), "angstrom", True, ORTHORHOMBIC),
            "frame-0",
        )
        with self.assertRaises(PeriodicReconstructionError):
            processor.process(
                CoordinateFrame(1, ((1.5, 1.0, 1.0), (2.5, 1.0, 1.0)), "angstrom", True, ORTHORHOMBIC),
                "frame-1",
            )

    def test_json_and_psf_connectivity_are_explicit_and_cardinality_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "bonds.json"
            json_path.write_text(
                json.dumps({
                    "format": "salsbury-bonds-v1",
                    "atom_count": 3,
                    "index_base": 0,
                    "bonds": [[0, 1], [1, 2]],
                }),
                encoding="utf-8",
            )
            bonds, identity = load_connectivity(json_path, 3)
            self.assertEqual(bonds, ((0, 1), (1, 2)))
            self.assertEqual(identity["bond_count"], 2)

            psf_path = root / "structure.psf"
            psf_path.write_text(
                "PSF\n\n       3 !NATOM\n"
                "       1 SEG 1 RES A A 0.0 1.0\n"
                "       2 SEG 1 RES B B 0.0 1.0\n"
                "       3 SEG 1 RES C C 0.0 1.0\n\n"
                "       2 !NBOND: bonds\n       1       2       2       3\n\n"
                "       0 !NTHETA: angles\n",
                encoding="utf-8",
            )
            psf_bonds, _ = load_connectivity(psf_path, 3)
            self.assertEqual(psf_bonds, ((0, 1), (1, 2)))
            with self.assertRaises(PeriodicReconstructionError):
                load_connectivity(psf_path, 4)

            prmtop_path = root / "structure.prmtop"
            prmtop_path.write_text(
                "%VERSION VERSION_STAMP = V0001.000\n"
                "%FLAG POINTERS\n%FORMAT(10I8)\n"
                f"{3:8d}{1:8d}\n"
                "%FLAG BONDS_INC_HYDROGEN\n%FORMAT(10I8)\n"
                f"{0:8d}{3:8d}{1:8d}\n"
                "%FLAG BONDS_WITHOUT_HYDROGEN\n%FORMAT(10I8)\n"
                f"{3:8d}{6:8d}{1:8d}\n",
                encoding="utf-8",
            )
            prmtop_bonds, _ = load_connectivity(prmtop_path, 3)
            self.assertEqual(prmtop_bonds, ((0, 1), (1, 2)))

    def test_reconstruction_requires_cell_and_connectivity(self):
        processor = PeriodicFrameProcessor(
            "make_whole",
            2,
            (),
            {
                "maximum_bond_length_angstrom": 2.0,
                "cycle_closure_tolerance_angstrom": 1.0e-6,
            },
        )
        with self.assertRaises(PeriodicReconstructionError):
            processor.process(
                CoordinateFrame(0, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), "angstrom", True, ORTHORHOMBIC),
                "frame-0",
            )


if __name__ == "__main__":
    unittest.main()
