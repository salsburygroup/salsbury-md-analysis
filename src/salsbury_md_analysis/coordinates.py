"""Streaming coordinate readers with compact NumPy-backed DCD frames."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Sequence, Set, TextIO, Tuple

import numpy as np

from .preflight import FileProbeError, probe_gro, probe_pdb, probe_xyz


Coordinate = Tuple[float, float, float]
CellVectors = Tuple[Coordinate, Coordinate, Coordinate]
SUPPORTED_COORDINATE_FORMATS = ("pdb", "gro", "xyz", "dcd")
_UNIT_TO_ANGSTROM = {"angstrom": 1.0, "nanometer": 10.0}
DCD_MADE_WHOLE_CACHE_TITLE = (
    b"salsbury-md-analysis made-whole molecular-payload cache"
)


class CoordinateReadError(ValueError):
    """Raised when coordinate records are malformed or unsupported."""


@dataclass(frozen=True)
class CoordinateFrame:
    frame_index: int
    coordinates_angstrom: Sequence[Coordinate]
    source_unit: str
    periodic_cell_present: bool
    cell_vectors_angstrom: Optional[CellVectors] = None
    coordinate_representation: str = "raw"

    @property
    def atom_count(self) -> int:
        return len(self.coordinates_angstrom)


def _scaled(x: float, y: float, z: float, source_unit: str) -> Coordinate:
    try:
        scale = _UNIT_TO_ANGSTROM[source_unit]
    except KeyError as exc:
        raise CoordinateReadError(
            "coordinate unit must be angstrom or nanometer"
        ) from exc
    return x * scale, y * scale, z * scale


def _cell_from_lengths_angles(
    a: float,
    b: float,
    c: float,
    alpha_degrees: float,
    beta_degrees: float,
    gamma_degrees: float,
    source_unit: str,
) -> CellVectors:
    """Return crystallographic cell vectors with lengths normalized to angstrom."""

    values = (a, b, c, alpha_degrees, beta_degrees, gamma_degrees)
    if not all(math.isfinite(value) for value in values):
        raise CoordinateReadError("periodic cell contains a non-finite value")
    if min(a, b, c) <= 0.0:
        raise CoordinateReadError("periodic cell lengths must be positive")
    if not all(0.0 < angle < 180.0 for angle in values[3:]):
        raise CoordinateReadError("periodic cell angles must lie strictly between 0 and 180 degrees")
    alpha = math.radians(alpha_degrees)
    beta = math.radians(beta_degrees)
    gamma = math.radians(gamma_degrees)
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) <= 1.0e-12:
        raise CoordinateReadError("periodic cell gamma angle is degenerate")
    def clean(value: float) -> float:
        return 0.0 if abs(value) <= 1.0e-14 * max(a, b, c) else value

    a_vector = (a, 0.0, 0.0)
    b_vector = (clean(b * math.cos(gamma)), b * sin_gamma, 0.0)
    c_x = c * math.cos(beta)
    c_y = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    c_z_squared = c * c - c_x * c_x - c_y * c_y
    tolerance = 1.0e-10 * c * c
    if c_z_squared <= tolerance:
        raise CoordinateReadError("periodic cell vectors are degenerate")
    c_vector = (clean(c_x), clean(c_y), math.sqrt(c_z_squared))
    return tuple(
        _scaled(vector[0], vector[1], vector[2], source_unit)
        for vector in (a_vector, b_vector, c_vector)
    )  # type: ignore[return-value]


def _pdb_cell(path: Path) -> Optional[CellVectors]:
    try:
        with Path(path).open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line[:6].strip().upper() != "CRYST1":
                    continue
                if len(line) < 54:
                    raise CoordinateReadError(
                        f"PDB CRYST1 record is too short at line {line_number}"
                    )
                try:
                    a = float(line[6:15])
                    b = float(line[15:24])
                    c = float(line[24:33])
                    alpha = float(line[33:40])
                    beta = float(line[40:47])
                    gamma = float(line[47:54])
                except ValueError as exc:
                    raise CoordinateReadError(
                        f"PDB CRYST1 record is malformed at line {line_number}"
                    ) from exc
                return _cell_from_lengths_angles(
                    a, b, c, alpha, beta, gamma, "angstrom"
                )
    except (OSError, UnicodeError) as exc:
        raise CoordinateReadError(str(exc)) from exc
    return None


def _gro_cell(fields: Tuple[float, ...]) -> CellVectors:
    if len(fields) == 3:
        vectors = (
            (fields[0], 0.0, 0.0),
            (0.0, fields[1], 0.0),
            (0.0, 0.0, fields[2]),
        )
    elif len(fields) == 9:
        # GROMACS order: v1x v2y v3z v1y v1z v2x v2z v3x v3y.
        vectors = (
            (fields[0], fields[3], fields[4]),
            (fields[5], fields[1], fields[6]),
            (fields[7], fields[8], fields[2]),
        )
    else:  # pragma: no cover - guarded by probe_gro
        raise CoordinateReadError("GRO box line must contain 3 or 9 values")
    if not all(math.isfinite(value) for vector in vectors for value in vector):
        raise CoordinateReadError("GRO box contains a non-finite value")
    return tuple(
        _scaled(vector[0], vector[1], vector[2], "nanometer")
        for vector in vectors
    )  # type: ignore[return-value]


def _dcd_cell(payload: bytes, endian: str, source_unit: str) -> CellVectors:
    values = struct.unpack(
        f"{endian}6{'d' if len(payload) == 48 else 'f'}", payload
    )
    a, gamma_raw, b, beta_raw, alpha_raw, c = values
    angles_raw = (alpha_raw, beta_raw, gamma_raw)
    if any(value < 0.0 for value in values) or any(value > 180.0 for value in angles_raw):
        raise CoordinateReadError(
            "DCD stores a symmetric-matrix unit cell that is not supported safely"
        )
    if all(abs(value) <= 1.0 for value in angles_raw):
        alpha, beta, gamma = (
            math.degrees(math.acos(max(-1.0, min(1.0, value))))
            for value in angles_raw
        )
    else:
        alpha, beta, gamma = angles_raw
    return _cell_from_lengths_angles(
        a, b, c, alpha, beta, gamma, source_unit
    )


def _pdb_coordinate(line: str, line_number: int) -> Coordinate:
    if len(line) < 54:
        raise CoordinateReadError(
            f"PDB coordinate record is too short at line {line_number}"
        )
    try:
        values = tuple(float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54)))
    except ValueError as exc:
        raise CoordinateReadError(
            f"PDB coordinate is malformed at line {line_number}"
        ) from exc
    return _scaled(values[0], values[1], values[2], "angstrom")


def iter_pdb_frames(path: Path) -> Iterator[CoordinateFrame]:
    """Yield every complete PDB model, or one implicit model, in angstrom."""

    source = Path(path).expanduser().resolve(strict=False)
    try:
        metadata = probe_pdb(source, "trajectory")
    except FileProbeError as exc:
        raise CoordinateReadError(str(exc)) from exc
    expected_atoms = int(metadata["atom_count"])
    cell_vectors = _pdb_cell(source)
    explicit_models = int(metadata["observed_frame_count"]) > 1
    coordinates = []
    in_model = not explicit_models
    frame_index = 0
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = line[:6].strip().upper()
                if record == "MODEL":
                    in_model = True
                    coordinates = []
                    explicit_models = True
                    continue
                if record == "ENDMDL":
                    if len(coordinates) != expected_atoms:
                        raise CoordinateReadError(
                            f"PDB model {frame_index} has {len(coordinates)} atoms; "
                            f"expected {expected_atoms}"
                        )
                    yield CoordinateFrame(
                        frame_index,
                        tuple(coordinates),
                        "angstrom",
                        bool(metadata["periodic_cell_declared"]),
                        cell_vectors,
                    )
                    frame_index += 1
                    coordinates = []
                    in_model = False
                    continue
                if record in {"ATOM", "HETATM"} and in_model:
                    coordinates.append(_pdb_coordinate(line, line_number))
    except (OSError, UnicodeError) as exc:
        raise CoordinateReadError(str(exc)) from exc
    if not explicit_models:
        if len(coordinates) != expected_atoms:
            raise CoordinateReadError(
                f"PDB frame has {len(coordinates)} atoms; expected {expected_atoms}"
            )
        yield CoordinateFrame(
            0,
            tuple(coordinates),
            "angstrom",
            bool(metadata["periodic_cell_declared"]),
            cell_vectors,
        )


def iter_gro_frames(path: Path) -> Iterator[CoordinateFrame]:
    """Yield the single validated GRO frame, converted from nm to angstrom."""

    source = Path(path).expanduser().resolve(strict=False)
    try:
        metadata = probe_gro(source, "trajectory")
        lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    except FileProbeError as exc:
        raise CoordinateReadError(str(exc)) from exc
    except (OSError, UnicodeError) as exc:
        raise CoordinateReadError(str(exc)) from exc
    atom_count = int(metadata["atom_count"])
    coordinates = []
    for line_number, line in enumerate(lines[2 : 2 + atom_count], start=3):
        if len(line) < 44:
            raise CoordinateReadError(
                f"GRO coordinate record is too short at line {line_number}"
            )
        try:
            x, y, z = (float(line[20:28]), float(line[28:36]), float(line[36:44]))
        except ValueError as exc:
            raise CoordinateReadError(
                f"GRO coordinate is malformed at line {line_number}"
            ) from exc
        coordinates.append(_scaled(x, y, z, "nanometer"))
    try:
        cell_values = tuple(float(value) for value in lines[-1].split())
    except ValueError as exc:  # pragma: no cover - guarded by probe_gro
        raise CoordinateReadError("GRO box contains a nonnumeric value") from exc
    yield CoordinateFrame(
        0, tuple(coordinates), "nanometer", True, _gro_cell(cell_values)
    )


def _next_nonblank(handle: TextIO) -> Optional[str]:
    while True:
        line = handle.readline()
        if line == "":
            return None
        if line.strip():
            return line


def iter_xyz_frames(path: Path, source_unit: str) -> Iterator[CoordinateFrame]:
    """Stream a conventional XYZ trajectory using its externally declared unit."""

    source = Path(path).expanduser().resolve(strict=False)
    try:
        metadata = probe_xyz(source)
    except FileProbeError as exc:
        raise CoordinateReadError(str(exc)) from exc
    expected_atoms = int(metadata["atom_count"])
    frame_index = 0
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            while True:
                count_line = _next_nonblank(handle)
                if count_line is None:
                    break
                atom_count = int(count_line.strip())
                comment = handle.readline()
                if comment == "":
                    raise CoordinateReadError("XYZ is truncated before a frame comment")
                coordinates = []
                for atom_index in range(atom_count):
                    line = handle.readline()
                    if line == "":
                        raise CoordinateReadError(
                            f"XYZ is truncated in atom {atom_index + 1} of frame {frame_index}"
                        )
                    fields = line.split()
                    try:
                        coordinates.append(
                            _scaled(
                                float(fields[1]), float(fields[2]), float(fields[3]), source_unit
                            )
                        )
                    except (IndexError, ValueError) as exc:
                        raise CoordinateReadError(
                            f"XYZ coordinate is malformed in frame {frame_index}, "
                            f"atom {atom_index + 1}"
                        ) from exc
                if atom_count != expected_atoms:
                    raise CoordinateReadError(
                        f"XYZ frame {frame_index} has {atom_count} atoms; expected {expected_atoms}"
                    )
                yield CoordinateFrame(
                    frame_index, tuple(coordinates), source_unit, False
                )
                frame_index += 1
    except (OSError, UnicodeError) as exc:
        raise CoordinateReadError(str(exc)) from exc


def _dcd_record(
    handle: BinaryIO,
    endian: str,
    expected_length: Optional[int] = None,
) -> bytes:
    marker = handle.read(4)
    if len(marker) != 4:
        raise CoordinateReadError("DCD is truncated before a Fortran record marker")
    length = struct.unpack(f"{endian}i", marker)[0]
    if length < 0 or length > 512 * 1024 * 1024:
        raise CoordinateReadError(f"DCD record length is implausible: {length}")
    if expected_length is not None and length != expected_length:
        raise CoordinateReadError(
            f"DCD record length is {length}; expected {expected_length}"
        )
    payload = handle.read(length)
    closing = handle.read(4)
    if len(payload) != length or len(closing) != 4:
        raise CoordinateReadError("DCD is truncated within a Fortran record")
    if struct.unpack(f"{endian}i", closing)[0] != length:
        raise CoordinateReadError("DCD Fortran record markers do not match")
    return payload


def _skip_dcd_record(
    handle: BinaryIO, endian: str, expected_length: Optional[int] = None,
) -> None:
    """Validate a Fortran record envelope without decoding its payload."""
    marker = handle.read(4)
    if len(marker) != 4:
        raise CoordinateReadError("DCD is truncated before a Fortran record marker")
    length = struct.unpack(f"{endian}i", marker)[0]
    if length < 0 or length > 512 * 1024 * 1024:
        raise CoordinateReadError(f"DCD record length is implausible: {length}")
    if expected_length is not None and length != expected_length:
        raise CoordinateReadError(
            f"DCD record length is {length}; expected {expected_length}"
        )
    handle.seek(length, 1)
    closing = handle.read(4)
    if len(closing) != 4 or struct.unpack(f"{endian}i", closing)[0] != length:
        raise CoordinateReadError("DCD Fortran record markers do not match")


def iter_dcd_frames(
    path: Path, source_unit: str,
    selected_frame_indices: Optional[Set[int]] = None,
) -> Iterator[CoordinateFrame]:
    """Stream standard 32-bit DCD coordinate records without fixed atoms."""

    source = Path(path).expanduser().resolve(strict=False)
    try:
        with source.open("rb") as handle:
            first = handle.read(4)
            if len(first) != 4:
                raise CoordinateReadError("DCD is too short")
            if struct.unpack("<i", first)[0] == 84:
                endian = "<"
            elif struct.unpack(">i", first)[0] == 84:
                endian = ">"
            else:
                raise CoordinateReadError(
                    "DCD does not begin with a supported 84-byte header record"
                )
            handle.seek(0)
            header = _dcd_record(handle, endian, expected_length=84)
            if header[:4] != b"CORD":
                raise CoordinateReadError(
                    f"structural QC requires CORD coordinates, not {header[:4]!r}"
                )
            control = struct.unpack(f"{endian}20i", header[4:84])
            declared_frames = control[0]
            fixed_atom_count = control[8]
            is_charmm = control[19] != 0
            has_unit_cell = is_charmm and bool(control[10])
            has_fourth_dimension = is_charmm and control[11] == 1
            if declared_frames <= 0:
                raise CoordinateReadError("DCD declared frame count must be positive")
            if fixed_atom_count:
                raise CoordinateReadError(
                    "DCD fixed-atom trajectories are not supported by the current reader"
                )
            title = _dcd_record(handle, endian)
            if len(title) < 4:
                raise CoordinateReadError("DCD title record is malformed")
            title_count = struct.unpack(f"{endian}i", title[:4])[0]
            if title_count < 0 or len(title) != 4 + 80 * title_count:
                raise CoordinateReadError(
                    "DCD title count does not match the title record length"
                )
            title_lines = [
                title[4 + 80 * index : 4 + 80 * (index + 1)].rstrip(b" \x00")
                for index in range(title_count)
            ]
            coordinate_representation = (
                "made_whole_molecular_payload_cache"
                if DCD_MADE_WHOLE_CACHE_TITLE in title_lines
                else "raw"
            )
            atom_record = _dcd_record(handle, endian, expected_length=4)
            atom_count = struct.unpack(f"{endian}i", atom_record)[0]
            if atom_count <= 0:
                raise CoordinateReadError("DCD atom count must be positive")
            coordinate_bytes = 4 * atom_count
            if selected_frame_indices is not None:
                invalid = [
                    index for index in selected_frame_indices
                    if isinstance(index, bool) or not isinstance(index, int)
                    or index < 0 or index >= declared_frames
                ]
                if invalid:
                    raise CoordinateReadError(
                        "selected DCD frame indices must be integer positions within "
                        "the declared frame count"
                    )
            for frame_index in range(declared_frames):
                selected = (
                    selected_frame_indices is None
                    or frame_index in selected_frame_indices
                )
                if not selected:
                    if has_unit_cell:
                        _skip_dcd_record(handle, endian)
                    _skip_dcd_record(handle, endian, expected_length=coordinate_bytes)
                    _skip_dcd_record(handle, endian, expected_length=coordinate_bytes)
                    _skip_dcd_record(handle, endian, expected_length=coordinate_bytes)
                    if has_fourth_dimension:
                        _skip_dcd_record(handle, endian, expected_length=coordinate_bytes)
                    continue
                cell_vectors = None
                if has_unit_cell:
                    cell = _dcd_record(handle, endian)
                    if len(cell) not in {24, 48}:
                        raise CoordinateReadError(
                            f"DCD unit-cell record has unsupported length {len(cell)}"
                        )
                    cell_vectors = _dcd_cell(cell, endian, source_unit)
                x_record = _dcd_record(handle, endian, expected_length=coordinate_bytes)
                y_record = _dcd_record(handle, endian, expected_length=coordinate_bytes)
                z_record = _dcd_record(handle, endian, expected_length=coordinate_bytes)
                if has_fourth_dimension:
                    _dcd_record(handle, endian, expected_length=coordinate_bytes)
                try:
                    scale = _UNIT_TO_ANGSTROM[source_unit]
                except KeyError as exc:
                    raise CoordinateReadError(
                        "coordinate unit must be angstrom or nanometer"
                    ) from exc
                dtype = np.dtype(f"{endian}f4")
                coordinates = np.empty((atom_count, 3), dtype=np.float64)
                coordinates[:, 0] = np.frombuffer(x_record, dtype=dtype) * scale
                coordinates[:, 1] = np.frombuffer(y_record, dtype=dtype) * scale
                coordinates[:, 2] = np.frombuffer(z_record, dtype=dtype) * scale
                yield CoordinateFrame(
                    frame_index,
                    coordinates,
                    source_unit,
                    has_unit_cell,
                    cell_vectors,
                    coordinate_representation,
                )
            if handle.read(1):
                raise CoordinateReadError(
                    "DCD contains data after its declared frame count"
                )
    except OSError as exc:
        raise CoordinateReadError(str(exc)) from exc


def coordinate_format(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return "pdb"
    if suffix == ".gro":
        return "gro"
    if suffix == ".xyz":
        return "xyz"
    if suffix == ".dcd":
        return "dcd"
    raise CoordinateReadError(
        f"unsupported coordinate format {suffix or '<none>'}; supported: "
        + ", ".join(SUPPORTED_COORDINATE_FORMATS)
    )


def iter_coordinate_frames(
    path: Path, declared_coordinate_unit: str,
    selected_frame_indices: Optional[Set[int]] = None,
) -> Iterator[CoordinateFrame]:
    """Dispatch a trajectory reader and normalize its coordinates to angstrom."""

    format_name = coordinate_format(path)
    if format_name == "dcd":
        return iter_dcd_frames(path, declared_coordinate_unit, selected_frame_indices)
    if format_name == "pdb":
        frames = iter_pdb_frames(path)
    elif format_name == "gro":
        frames = iter_gro_frames(path)
    else:
        frames = iter_xyz_frames(path, declared_coordinate_unit)
    if selected_frame_indices is None:
        return frames
    return (
        frame for frame in frames if frame.frame_index in selected_frame_indices
    )


def finite_coordinate_count(frame: CoordinateFrame) -> int:
    """Return the number of atoms with three finite coordinates."""

    return sum(all(math.isfinite(value) for value in coordinate) for coordinate in frame.coordinates_angstrom)
