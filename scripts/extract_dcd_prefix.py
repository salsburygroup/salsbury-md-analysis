#!/usr/bin/env python3
"""Create a local, hash-recorded DCD prefix for bounded regression testing.

The source is read without modification. The output retains the original DCD
header, title, atom count, coordinate records, starting step, and save interval;
only the declared frame count is changed to the requested prefix length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import BinaryIO, Optional, Tuple


class DCDPrefixError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_record(
    handle: BinaryIO, endian: str, expected_length: Optional[int] = None
) -> bytes:
    opening = handle.read(4)
    if len(opening) != 4:
        raise DCDPrefixError("DCD is truncated before a record marker")
    length = struct.unpack(f"{endian}i", opening)[0]
    if length < 0 or length > 512 * 1024 * 1024:
        raise DCDPrefixError(f"implausible DCD record length: {length}")
    if expected_length is not None and length != expected_length:
        raise DCDPrefixError(
            f"DCD record length is {length}; expected {expected_length}"
        )
    payload = handle.read(length)
    closing = handle.read(4)
    if len(payload) != length or len(closing) != 4:
        raise DCDPrefixError("DCD is truncated within a record")
    if struct.unpack(f"{endian}i", closing)[0] != length:
        raise DCDPrefixError("DCD record markers do not match")
    return payload


def write_record(handle: BinaryIO, endian: str, payload: bytes) -> None:
    marker = struct.pack(f"{endian}i", len(payload))
    handle.write(marker)
    handle.write(payload)
    handle.write(marker)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def extract_prefix(
    source: Path,
    output: Path,
    frame_count: int,
    protected_roots: tuple[Path, ...] = (),
) -> dict:
    if any(_is_within(output, root) for root in protected_roots):
        raise DCDPrefixError("refusing to write a regression fixture under a protected root")
    if output.exists():
        raise DCDPrefixError(f"output already exists: {output}")
    with source.open("rb") as input_handle:
        first = input_handle.read(4)
        if len(first) != 4:
            raise DCDPrefixError("DCD is too short")
        if struct.unpack("<i", first)[0] == 84:
            endian = "<"
            byte_order = "little"
        elif struct.unpack(">i", first)[0] == 84:
            endian = ">"
            byte_order = "big"
        else:
            raise DCDPrefixError("DCD does not have a supported 84-byte header")
        input_handle.seek(0)
        header = read_record(input_handle, endian, 84)
        if header[:4] != b"CORD":
            raise DCDPrefixError("only CORD DCD files are supported")
        control = list(struct.unpack(f"{endian}20i", header[4:]))
        declared_frames = control[0]
        fixed_atoms = control[8]
        is_charmm = control[19] != 0
        has_unit_cell = is_charmm and bool(control[10])
        has_fourth_dimension = is_charmm and control[11] == 1
        if fixed_atoms:
            raise DCDPrefixError("fixed-atom DCD files are not supported")
        if frame_count <= 0 or frame_count > declared_frames:
            raise DCDPrefixError(
                f"frame_count must be between 1 and declared count {declared_frames}"
            )
        title = read_record(input_handle, endian)
        atom_record = read_record(input_handle, endian, 4)
        atom_count = struct.unpack(f"{endian}i", atom_record)[0]
        coordinate_bytes = atom_count * 4
        control[0] = frame_count
        output_header = header[:4] + struct.pack(f"{endian}20i", *control)

        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as output_handle:
            write_record(output_handle, endian, output_header)
            write_record(output_handle, endian, title)
            write_record(output_handle, endian, atom_record)
            for _ in range(frame_count):
                if has_unit_cell:
                    cell = read_record(input_handle, endian)
                    if len(cell) not in {24, 48}:
                        raise DCDPrefixError(
                            f"unsupported DCD unit-cell record length: {len(cell)}"
                        )
                    write_record(output_handle, endian, cell)
                for _axis in range(3):
                    write_record(
                        output_handle,
                        endian,
                        read_record(input_handle, endian, coordinate_bytes),
                    )
                if has_fourth_dimension:
                    write_record(
                        output_handle,
                        endian,
                        read_record(input_handle, endian, coordinate_bytes),
                    )
    return {
        "atom_count": atom_count,
        "byte_order": byte_order,
        "source_declared_frame_count": declared_frames,
        "output_declared_frame_count": frame_count,
        "starting_step": control[1],
        "save_interval_steps": control[2],
        "unit_cell_records": has_unit_cell,
        "fourth_dimension_records": has_fourth_dimension,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument(
        "--protected-root",
        action="append",
        default=[],
        type=Path,
        help="Refuse outputs under this root; may be supplied more than once.",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    protected_roots = tuple(
        root.expanduser().resolve(strict=False) for root in args.protected_root
    )
    metadata = extract_prefix(source, output, args.frames, protected_roots)
    report = {
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
        },
        "output": {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        },
        "dcd": metadata,
        "limitations": [
            "The prefix is a technical regression fixture, not a new scientific trajectory.",
            "Frame extraction does not establish equilibration, convergence, or validity.",
        ],
    }
    if args.provenance is not None:
        provenance = args.provenance.expanduser().resolve(strict=False)
        if any(_is_within(provenance, root) for root in protected_roots):
            raise DCDPrefixError("refusing to write provenance under a protected root")
        if provenance.exists():
            raise DCDPrefixError(f"provenance output already exists: {provenance}")
        provenance.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
