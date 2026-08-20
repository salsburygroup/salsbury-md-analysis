import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from salsbury_md_analysis.coordinates import (
    CoordinateReadError,
    iter_coordinate_frames,
    iter_dcd_frames,
    iter_gro_frames,
    iter_pdb_frames,
    iter_xyz_frames,
)


def _record(payload: bytes, endian: str = "<") -> bytes:
    marker = struct.pack(f"{endian}i", len(payload))
    return marker + payload + marker


def _write_coordinate_dcd(
    path: Path,
    endian: str = "<",
    with_unit_cell: bool = False,
    fixed_atom_count: int = 0,
) -> None:
    header = bytearray(84)
    header[:4] = b"CORD"
    struct.pack_into(f"{endian}3i", header, 4, 2, 0, 100)
    struct.pack_into(f"{endian}i", header, 36, fixed_atom_count)
    if with_unit_cell:
        struct.pack_into(f"{endian}i", header, 44, 1)
        struct.pack_into(f"{endian}i", header, 80, 24)
    title = struct.pack(f"{endian}i", 1) + b"coordinate reader test".ljust(80)
    payload = _record(bytes(header), endian)
    payload += _record(title, endian)
    payload += _record(struct.pack(f"{endian}i", 2), endian)
    for coordinates in (
        ((0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
        ((0.5, 1.5), (0.0, 0.0), (0.0, 0.0)),
    ):
        if with_unit_cell:
            payload += _record(
                struct.pack(f"{endian}6d", 10.0, 0.0, 10.0, 0.0, 0.0, 10.0),
                endian,
            )
        for axis in coordinates:
            payload += _record(struct.pack(f"{endian}2f", *axis), endian)
    path.write_bytes(payload)


class CoordinateReaderTests(unittest.TestCase):
    def test_xyz_streams_frames_and_applies_declared_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frames.xyz"
            path.write_text(
                "2\nf0\nC 0 0 0\nC 0.1 0 0\n2\nf1\nC 0 0 0\nC 0.2 0 0\n",
                encoding="utf-8",
            )
            frames = list(iter_xyz_frames(path, "nanometer"))
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].coordinates_angstrom[1], (1.0, 0.0, 0.0))
            self.assertEqual(frames[1].frame_index, 1)

    def test_pdb_multimodel_and_gro_coordinates_are_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdb = root / "models.pdb"
            atom_1 = "ATOM      1  C   UNK A   1       1.000   2.000   3.000  1.00  0.00           C\n"
            atom_2 = "ATOM      1  C   UNK A   1       2.000   2.000   3.000  1.00  0.00           C\n"
            pdb.write_text(
                "MODEL        1\n" + atom_1 + "ENDMDL\nMODEL        2\n" + atom_2 + "ENDMDL\nEND\n",
                encoding="utf-8",
            )
            pdb_frames = list(iter_pdb_frames(pdb))
            self.assertEqual(len(pdb_frames), 2)
            self.assertEqual(pdb_frames[1].coordinates_angstrom[0], (2.0, 2.0, 3.0))

            gro = root / "one.gro"
            gro.write_text(
                "synthetic\n1\n    1UNK      C    1   0.100   0.200   0.300\n1.0 1.0 1.0\n",
                encoding="utf-8",
            )
            gro_frame = list(iter_gro_frames(gro))[0]
            self.assertEqual(gro_frame.coordinates_angstrom[0], (1.0, 2.0, 3.0))

    def test_standard_little_and_big_endian_dcd_coordinates_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for endian in ("<", ">"):
                path = root / ("little.dcd" if endian == "<" else "big.dcd")
                _write_coordinate_dcd(path, endian=endian)
                frames = list(iter_dcd_frames(path, "angstrom"))
                self.assertEqual(len(frames), 2)
                self.assertEqual(frames[0].atom_count, 2)
                self.assertIsInstance(frames[0].coordinates_angstrom, np.ndarray)
                self.assertEqual(frames[0].coordinates_angstrom.dtype, np.float64)
                self.assertTrue(frames[0].coordinates_angstrom.flags.owndata)
                self.assertAlmostEqual(frames[1].coordinates_angstrom[1][0], 1.5)

    def test_charmm_dcd_unit_cell_records_are_interpreted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cell.dcd"
            _write_coordinate_dcd(path, with_unit_cell=True)
            frames = list(iter_dcd_frames(path, "angstrom"))
            self.assertEqual(len(frames), 2)
            self.assertTrue(all(frame.periodic_cell_present for frame in frames))
            self.assertEqual(
                frames[0].cell_vectors_angstrom,
                ((10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)),
            )

    def test_dcd_selected_frames_skip_coordinate_decode_but_validate_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected.dcd"
            _write_coordinate_dcd(path, with_unit_cell=True)
            frames = list(iter_dcd_frames(path, "angstrom", {1}))
            self.assertEqual([frame.frame_index for frame in frames], [1])
            self.assertAlmostEqual(frames[0].coordinates_angstrom[1][0], 1.5)
            with self.assertRaises(CoordinateReadError):
                list(iter_dcd_frames(path, "angstrom", {2}))

    def test_fixed_atom_dcd_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixed.dcd"
            _write_coordinate_dcd(path, fixed_atom_count=1)
            with self.assertRaises(CoordinateReadError) as context:
                list(iter_dcd_frames(path, "angstrom"))
            self.assertIn("fixed-atom", str(context.exception))

    def test_dcd_declared_frame_truncation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "truncated.dcd"
            _write_coordinate_dcd(path)
            path.write_bytes(path.read_bytes()[:-8])
            with self.assertRaises(CoordinateReadError):
                list(iter_coordinate_frames(path, "angstrom"))

    def test_unsupported_coordinate_format_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frames.xtc"
            path.write_bytes(b"not xtc")
            with self.assertRaises(CoordinateReadError) as context:
                list(iter_coordinate_frames(path, "angstrom"))
            self.assertIn("unsupported coordinate format", str(context.exception))


if __name__ == "__main__":
    unittest.main()
