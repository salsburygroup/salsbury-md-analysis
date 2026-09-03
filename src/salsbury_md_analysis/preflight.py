"""Read-only topology and trajectory metadata preflight.

The probes in this module inspect file structure and metadata only.  They never
modify an input and never imply equilibration, convergence, or scientific
validity.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Dict, List, Mapping, Optional, TextIO, Tuple

from .manifests import (
    inventory_content_signature_sha256, inventory_system_inputs,
    resolve_manifest_path,
)
from .reporting import issue_record
from .trajectory_contracts import frame_axis_value, normalize_segment_axis


class FileProbeError(ValueError):
    """Raised when a file is unsupported, malformed, or structurally incomplete."""


TOPOLOGY_FORMATS = ("pdb", "gro", "psf", "prmtop")
TRAJECTORY_FORMATS = ("dcd", "pdb", "gro", "xyz")
CONNECTIVITY_FORMATS = ("json", "psf", "prmtop", "parm7")


def _base(path: Path, format_name: str, role: str) -> Dict[str, object]:
    source = Path(path)
    stat = source.stat()
    return {
        "path": str(source),
        "format": format_name,
        "role": role,
        "size_bytes": stat.st_size,
    }


def probe_pdb(path: Path, role: str) -> Dict[str, object]:
    """Inspect PDB atom/model records without interpreting chemistry."""

    source = Path(path)
    model_counts: List[int] = []
    current_count = 0
    explicit_models = False
    inside_model = False
    cryst1 = False
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = line[:6].strip().upper()
                if record == "CRYST1":
                    cryst1 = True
                elif record == "MODEL":
                    if inside_model:
                        raise FileProbeError(f"nested MODEL record at line {line_number}")
                    if current_count:
                        raise FileProbeError(
                            f"atom records occur before MODEL at line {line_number}"
                        )
                    explicit_models = True
                    inside_model = True
                    current_count = 0
                elif record in {"ATOM", "HETATM"}:
                    current_count += 1
                elif record == "ENDMDL":
                    if not inside_model:
                        raise FileProbeError(f"ENDMDL without MODEL at line {line_number}")
                    model_counts.append(current_count)
                    current_count = 0
                    inside_model = False
    except (OSError, UnicodeError) as exc:
        raise FileProbeError(str(exc)) from exc
    if inside_model:
        raise FileProbeError("unterminated MODEL record")
    if explicit_models:
        if current_count:
            raise FileProbeError("atom records occur outside an explicit MODEL block")
        if not model_counts:
            raise FileProbeError("PDB contains MODEL records but no completed models")
    else:
        model_counts = [current_count]
    if not model_counts or model_counts[0] <= 0:
        raise FileProbeError("PDB contains no ATOM or HETATM records")
    if len(set(model_counts)) != 1:
        raise FileProbeError(
            "PDB models have inconsistent atom counts: "
            + ", ".join(str(value) for value in model_counts)
        )
    result = _base(source, "pdb", role)
    result.update(
        {
            "atom_count": model_counts[0],
            "observed_frame_count": len(model_counts),
            "periodic_cell_declared": cryst1,
            "metadata_scope": "records",
        }
    )
    return result


def probe_gro(path: Path, role: str) -> Dict[str, object]:
    """Inspect one GROMACS GRO structure/frame."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            lines = handle.readlines()
    except (OSError, UnicodeError) as exc:
        raise FileProbeError(str(exc)) from exc
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 3:
        raise FileProbeError("GRO file is too short")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise FileProbeError("GRO atom-count line is not an integer") from exc
    if atom_count <= 0:
        raise FileProbeError("GRO atom count must be positive")
    expected_lines = atom_count + 3
    if len(lines) != expected_lines:
        raise FileProbeError(
            f"GRO line count is {len(lines)}; expected {expected_lines} for {atom_count} atoms"
        )
    box_fields = lines[-1].split()
    if len(box_fields) not in {3, 9}:
        raise FileProbeError("GRO box line must contain 3 or 9 numeric values")
    try:
        [float(value) for value in box_fields]
    except ValueError as exc:
        raise FileProbeError("GRO box line contains a nonnumeric value") from exc
    result = _base(source, "gro", role)
    result.update(
        {
            "atom_count": atom_count,
            "observed_frame_count": 1,
            "periodic_cell_declared": True,
            "metadata_scope": "records",
        }
    )
    return result


def probe_psf(path: Path) -> Dict[str, object]:
    """Read the declared atom count from a CHARMM/NAMD PSF."""

    source = Path(path)
    atom_count: Optional[int] = None
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if "!NATOM" not in line.upper():
                    continue
                try:
                    atom_count = int(line.split()[0])
                except (IndexError, ValueError) as exc:
                    raise FileProbeError("PSF !NATOM declaration is malformed") from exc
                break
    except (OSError, UnicodeError) as exc:
        raise FileProbeError(str(exc)) from exc
    if atom_count is None:
        raise FileProbeError("PSF contains no !NATOM declaration")
    if atom_count <= 0:
        raise FileProbeError("PSF atom count must be positive")
    result = _base(source, "psf", "topology")
    result.update({"atom_count": atom_count, "metadata_scope": "header"})
    return result


def probe_prmtop(path: Path) -> Dict[str, object]:
    """Read NATOM from the Amber POINTERS section."""

    source = Path(path)
    in_pointers = False
    values: List[int] = []
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.upper() == "%FLAG POINTERS":
                    in_pointers = True
                    continue
                if not in_pointers:
                    continue
                if stripped.upper().startswith("%FORMAT"):
                    continue
                if stripped.upper().startswith("%FLAG"):
                    break
                for token in stripped.split():
                    try:
                        values.append(int(token))
                    except ValueError as exc:
                        raise FileProbeError("Amber POINTERS section contains a noninteger") from exc
                if values:
                    break
    except (OSError, UnicodeError) as exc:
        raise FileProbeError(str(exc)) from exc
    if not values:
        raise FileProbeError("Amber topology contains no POINTERS/NATOM value")
    atom_count = values[0]
    if atom_count <= 0:
        raise FileProbeError("Amber NATOM must be positive")
    result = _base(source, "prmtop", "topology")
    result.update({"atom_count": atom_count, "metadata_scope": "header"})
    return result


def probe_bond_json(path: Path) -> Dict[str, object]:
    """Inspect portable explicit bond JSON without interpreting chemistry."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FileProbeError(str(exc)) from exc
    required = {"format", "atom_count", "index_base", "bonds"}
    allowed = required | {"provenance"}
    if (
        not isinstance(payload, dict)
        or not required.issubset(payload)
        or not set(payload).issubset(allowed)
    ):
        raise FileProbeError(
            "bond JSON requires format, atom_count, index_base, and bonds; "
            "provenance is optional"
        )
    if payload["format"] != "salsbury-bonds-v1" or payload["index_base"] != 0:
        raise FileProbeError(
            "bond JSON requires format=salsbury-bonds-v1 and index_base=0"
        )
    atom_count = payload["atom_count"]
    if isinstance(atom_count, bool) or not isinstance(atom_count, int) or atom_count <= 0:
        raise FileProbeError("bond JSON atom_count must be a positive integer")
    raw_bonds = payload["bonds"]
    if not isinstance(raw_bonds, list):
        raise FileProbeError("bond JSON bonds must be an array")
    bonds = set()
    for index, raw_bond in enumerate(raw_bonds):
        if not isinstance(raw_bond, list) or len(raw_bond) != 2:
            raise FileProbeError(f"bond JSON bond {index} must contain two indices")
        first, second = raw_bond
        if (
            isinstance(first, bool)
            or isinstance(second, bool)
            or not isinstance(first, int)
            or not isinstance(second, int)
        ):
            raise FileProbeError(f"bond JSON bond {index} indices must be integers")
        if first == second:
            raise FileProbeError(f"bond JSON bond {index} is a self bond")
        if min(first, second) < 0 or max(first, second) >= atom_count:
            raise FileProbeError(
                f"bond JSON bond {index} exceeds atom count {atom_count}"
            )
        bonds.add((min(first, second), max(first, second)))
    if atom_count > 1 and not bonds:
        raise FileProbeError("bond JSON contains no bonds")
    result = _base(source, "salsbury-bonds-v1", "connectivity")
    result.update(
        {
            "atom_count": atom_count,
            "bond_count": len(bonds),
            "index_base": 0,
            "metadata_scope": "records",
        }
    )
    return result


def _next_nonblank(handle: TextIO) -> Tuple[Optional[str], int]:
    skipped = 0
    while True:
        line = handle.readline()
        if line == "":
            return None, skipped
        skipped += 1
        if line.strip():
            return line, skipped


def probe_xyz(path: Path) -> Dict[str, object]:
    """Count frames and atoms in a conventional XYZ trajectory."""

    source = Path(path)
    frame_counts: List[int] = []
    line_number = 0
    try:
        with source.open("r", encoding="utf-8", errors="strict") as handle:
            while True:
                count_line, consumed = _next_nonblank(handle)
                line_number += consumed
                if count_line is None:
                    break
                try:
                    atom_count = int(count_line.strip())
                except ValueError as exc:
                    raise FileProbeError(
                        f"XYZ atom count is not an integer at line {line_number}"
                    ) from exc
                if atom_count <= 0:
                    raise FileProbeError(
                        f"XYZ atom count must be positive at line {line_number}"
                    )
                comment = handle.readline()
                line_number += 1
                if comment == "":
                    raise FileProbeError("XYZ is truncated before the frame comment")
                for atom_index in range(atom_count):
                    line = handle.readline()
                    line_number += 1
                    if line == "":
                        raise FileProbeError(
                            f"XYZ is truncated in atom {atom_index + 1} of frame {len(frame_counts) + 1}"
                        )
                    fields = line.split()
                    if len(fields) < 4:
                        raise FileProbeError(
                            f"XYZ coordinate line {line_number} has fewer than four fields"
                        )
                    try:
                        float(fields[1])
                        float(fields[2])
                        float(fields[3])
                    except ValueError as exc:
                        raise FileProbeError(
                            f"XYZ coordinate line {line_number} contains nonnumeric coordinates"
                        ) from exc
                frame_counts.append(atom_count)
    except (OSError, UnicodeError) as exc:
        raise FileProbeError(str(exc)) from exc
    if not frame_counts:
        raise FileProbeError("XYZ contains no frames")
    if len(set(frame_counts)) != 1:
        raise FileProbeError(
            "XYZ frames have inconsistent atom counts: "
            + ", ".join(str(value) for value in frame_counts)
        )
    result = _base(source, "xyz", "trajectory")
    result.update(
        {
            "atom_count": frame_counts[0],
            "observed_frame_count": len(frame_counts),
            "periodic_cell_declared": False,
            "metadata_scope": "complete_file",
        }
    )
    return result


def _fortran_record(handle, endian: str, expected_length: Optional[int] = None) -> bytes:
    raw_length = handle.read(4)
    if len(raw_length) != 4:
        raise FileProbeError("DCD is truncated before a Fortran record marker")
    length = struct.unpack(f"{endian}i", raw_length)[0]
    if length < 0 or length > 512 * 1024 * 1024:
        raise FileProbeError(f"DCD record length is implausible: {length}")
    if expected_length is not None and length != expected_length:
        raise FileProbeError(f"DCD record length is {length}; expected {expected_length}")
    payload = handle.read(length)
    if len(payload) != length:
        raise FileProbeError("DCD is truncated within a Fortran record")
    raw_closing = handle.read(4)
    if len(raw_closing) != 4 or struct.unpack(f"{endian}i", raw_closing)[0] != length:
        raise FileProbeError("DCD Fortran record markers do not match")
    return payload


def probe_dcd(path: Path) -> Dict[str, object]:
    """Read a conventional 32-bit DCD header without loading coordinates."""

    source = Path(path)
    try:
        with source.open("rb") as handle:
            first = handle.read(4)
            if len(first) != 4:
                raise FileProbeError("DCD is too short")
            little = struct.unpack("<i", first)[0]
            big = struct.unpack(">i", first)[0]
            if little == 84:
                endian = "<"
                byte_order = "little"
            elif big == 84:
                endian = ">"
                byte_order = "big"
            else:
                raise FileProbeError(
                    "DCD does not begin with a supported 84-byte header record"
                )
            handle.seek(0)
            header = _fortran_record(handle, endian, expected_length=84)
            magic = header[:4]
            if magic not in {b"CORD", b"VELD"}:
                raise FileProbeError(f"unsupported DCD header signature: {magic!r}")
            declared_frames, starting_step, save_interval = struct.unpack(
                f"{endian}3i", header[4:16]
            )
            if declared_frames <= 0:
                raise FileProbeError("DCD declared frame count must be positive")
            if save_interval <= 0:
                raise FileProbeError("DCD save interval must be positive")
            title = _fortran_record(handle, endian)
            if len(title) < 4:
                raise FileProbeError("DCD title record is malformed")
            title_count = struct.unpack(f"{endian}i", title[:4])[0]
            if title_count < 0 or len(title) != 4 + 80 * title_count:
                raise FileProbeError("DCD title count does not match the title record length")
            atom_record = _fortran_record(handle, endian, expected_length=4)
            atom_count = struct.unpack(f"{endian}i", atom_record)[0]
            if atom_count <= 0:
                raise FileProbeError("DCD atom count must be positive")
    except OSError as exc:
        raise FileProbeError(str(exc)) from exc
    result = _base(source, "dcd", "trajectory")
    result.update(
        {
            "atom_count": atom_count,
            "declared_frame_count": declared_frames,
            "starting_step": starting_step,
            "save_interval_steps": save_interval,
            "byte_order": byte_order,
            "coordinate_kind": magic.decode("ascii"),
            "metadata_scope": "header_only",
        }
    )
    return result


def probe_topology(path: Path) -> Dict[str, object]:
    suffix = Path(path).suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return probe_pdb(path, "topology")
    if suffix == ".gro":
        return probe_gro(path, "topology")
    if suffix == ".psf":
        return probe_psf(path)
    if suffix in {".prmtop", ".parm7"}:
        return probe_prmtop(path)
    raise FileProbeError(
        f"unsupported topology format {suffix or '<none>'}; supported: {', '.join(TOPOLOGY_FORMATS)}"
    )


def probe_trajectory(path: Path) -> Dict[str, object]:
    suffix = Path(path).suffix.lower()
    if suffix == ".dcd":
        return probe_dcd(path)
    if suffix in {".pdb", ".ent"}:
        return probe_pdb(path, "trajectory")
    if suffix == ".gro":
        return probe_gro(path, "trajectory")
    if suffix == ".xyz":
        return probe_xyz(path)
    raise FileProbeError(
        f"unsupported trajectory format {suffix or '<none>'}; supported: {', '.join(TRAJECTORY_FORMATS)}"
    )


def probe_connectivity(path: Path) -> Dict[str, object]:
    """Inspect an explicitly supplied bond-topology input."""

    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return probe_bond_json(path)
    if suffix == ".psf":
        result = probe_psf(path)
    elif suffix in {".prmtop", ".parm7"}:
        result = probe_prmtop(path)
    else:
        raise FileProbeError(
            f"unsupported connectivity format {suffix or '<none>'}; "
            f"supported: {', '.join(CONNECTIVITY_FORMATS)}"
        )
    result["role"] = "connectivity"
    return result


def _attach_hash(probe: Dict[str, object], inventory_by_path: Mapping[str, Mapping[str, object]]) -> None:
    record = inventory_by_path.get(str(probe["path"]))
    if record is not None:
        probe["sha256"] = record.get("sha256")


def _frame_count(probe: Mapping[str, object]) -> Optional[int]:
    """Return declared or observed frame count from a successful probe."""

    value = probe.get("observed_frame_count", probe.get("declared_frame_count"))
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def preflight_system(
    data: Mapping[str, object], source_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Run format-aware metadata checks for every system-manifest input."""

    manifest_path = Path(source_path).expanduser().resolve(strict=False)
    inventory = inventory_system_inputs(data, manifest_path, hash_content=hash_content)
    inventory_by_path = {
        str(record["resolved_path"]): record for record in inventory["entries"]
    }
    issues: List[Dict[str, str]] = []
    system_reports: List[Dict[str, object]] = []
    systems = data["systems"]
    assert isinstance(systems, list)
    for raw_system in systems:
        assert isinstance(raw_system, dict)
        system_id = str(raw_system["system_id"])
        replica_reports: List[Dict[str, object]] = []
        replicas = raw_system["replicas"]
        assert isinstance(replicas, list)
        for raw_replica in replicas:
            assert isinstance(raw_replica, dict)
            replica_id = str(raw_replica["replica_id"])
            replica_location = f"{system_id}/{replica_id}"
            topology_path = resolve_manifest_path(str(raw_replica["topology"]), manifest_path)
            topology_probe: Optional[Dict[str, object]] = None
            try:
                topology_probe = probe_topology(topology_path)
                _attach_hash(topology_probe, inventory_by_path)
            except FileProbeError as exc:
                issues.append(issue_record("error", "TOPOLOGY_PROBE_FAILED", replica_location, str(exc)))

            connectivity_probe: Optional[Dict[str, object]] = None
            raw_connectivity = raw_replica.get("connectivity")
            if raw_connectivity is not None:
                connectivity_path = resolve_manifest_path(
                    str(raw_connectivity), manifest_path
                )
                try:
                    connectivity_probe = probe_connectivity(connectivity_path)
                    _attach_hash(connectivity_probe, inventory_by_path)
                    if topology_probe is not None and (
                        topology_probe.get("atom_count")
                        != connectivity_probe.get("atom_count")
                    ):
                        issues.append(
                            issue_record(
                                "error",
                                "CONNECTIVITY_ATOM_COUNT_MISMATCH",
                                replica_location,
                                f"topology has {topology_probe.get('atom_count')} atoms; "
                                f"connectivity declares {connectivity_probe.get('atom_count')}",
                            )
                        )
                except FileProbeError as exc:
                    issues.append(
                        issue_record(
                            "error", "CONNECTIVITY_PROBE_FAILED", replica_location, str(exc)
                        )
                    )

            force_field_parameter_probe = None
            raw_force_field_parameters = raw_replica.get("force_field_parameters")
            if isinstance(raw_force_field_parameters, dict):
                raw_files = raw_force_field_parameters.get("files", [])
                if isinstance(raw_files, list):
                    file_rows = []
                    for value in raw_files:
                        parameter_path = resolve_manifest_path(
                            str(value), manifest_path
                        )
                        inventory_row = inventory_by_path.get(str(parameter_path), {})
                        file_rows.append({
                            "declared_path": str(value),
                            "path": str(parameter_path),
                            "size_bytes": inventory_row.get("size_bytes"),
                            "sha256": inventory_row.get("sha256"),
                        })
                    force_field_parameter_probe = {
                        "format": raw_force_field_parameters.get("format"),
                        "files": file_rows,
                        "metadata_scope": "inventory_only",
                    }

            segment_reports: List[Dict[str, object]] = []
            previous_probe: Optional[Dict[str, object]] = None
            previous_axis: Optional[Dict[str, object]] = None
            segments = raw_replica["segments"]
            assert isinstance(segments, list)
            for raw_segment in segments:
                assert isinstance(raw_segment, dict)
                segment_id = str(raw_segment["segment_id"])
                segment_location = f"{replica_location}/{segment_id}"
                trajectory_path = resolve_manifest_path(
                    str(raw_segment["trajectory"]), manifest_path
                )
                trajectory_probe: Optional[Dict[str, object]] = None
                axis = normalize_segment_axis(raw_segment, "ps")
                try:
                    trajectory_probe = probe_trajectory(trajectory_path)
                    _attach_hash(trajectory_probe, inventory_by_path)
                    if trajectory_probe.get("metadata_scope") == "header_only":
                        issues.append(
                            issue_record(
                                "warning", "DCD_HEADER_ONLY", segment_location,
                                "DCD declared metadata was read, but coordinate records and actual frame count were not scanned.",
                            )
                        )
                    if topology_probe is not None and (
                        topology_probe.get("atom_count") != trajectory_probe.get("atom_count")
                    ):
                        issues.append(
                            issue_record(
                                "error", "ATOM_COUNT_MISMATCH", segment_location,
                                f"topology has {topology_probe.get('atom_count')} atoms; "
                                f"trajectory declares {trajectory_probe.get('atom_count')}",
                            )
                        )
                    if connectivity_probe is not None and (
                        connectivity_probe.get("atom_count")
                        != trajectory_probe.get("atom_count")
                    ):
                        issues.append(
                            issue_record(
                                "error",
                                "CONNECTIVITY_TRAJECTORY_ATOM_COUNT_MISMATCH",
                                segment_location,
                                f"connectivity has {connectivity_probe.get('atom_count')} atoms; "
                                f"trajectory declares {trajectory_probe.get('atom_count')}",
                            )
                        )
                    continuous = bool(raw_segment.get("continuous_with_previous", False))
                    dcd_header_step_policy = str(
                        raw_segment.get("dcd_header_step_policy", "continuous")
                    )
                    if continuous:
                        if previous_probe is None or previous_axis is None:
                            issues.append(
                                issue_record(
                                    "warning", "CONTINUITY_NOT_VERIFIED", segment_location,
                                    "continuity is declared but the preceding segment could not be probed",
                                )
                            )
                        else:
                            previous_frames = _frame_count(previous_probe)
                            if previous_frames is None:
                                issues.append(
                                    issue_record(
                                        "warning", "CONTINUITY_NOT_VERIFIED", segment_location,
                                        "continuity is declared but the preceding frame count is unavailable",
                                    )
                                )
                            else:
                                if previous_axis["kind"] != axis["kind"]:
                                    issues.append(
                                        issue_record(
                                            "error", "FRAME_AXIS_KIND_MISMATCH",
                                            segment_location,
                                            "continuous segments cannot switch between physical-time and sample-index axes",
                                        )
                                    )
                                else:
                                    expected_value = frame_axis_value(
                                        previous_axis, previous_frames
                                    )
                                    observed_value = frame_axis_value(axis, 0)
                                    matches = (
                                        observed_value == expected_value
                                        if axis["kind"] == "sample_index"
                                        else math.isclose(
                                            float(observed_value),
                                            float(expected_value),
                                            rel_tol=1.0e-12,
                                            abs_tol=1.0e-12,
                                        )
                                    )
                                    if not matches:
                                        code = (
                                            "SAMPLE_INDEX_CONTINUITY_MISMATCH"
                                            if axis["kind"] == "sample_index"
                                            else "PHYSICAL_TIME_CONTINUITY_MISMATCH"
                                        )
                                        unit = "sample" if axis["kind"] == "sample_index" else "ps"
                                        issues.append(
                                            issue_record(
                                                "error",
                                                code,
                                                segment_location,
                                                f"continuous segment starts at {observed_value} {unit}; "
                                                f"expected {expected_value} {unit} from the preceding "
                                                "segment axis and frame count",
                                            )
                                        )
                        if (
                            previous_probe is not None
                            and previous_probe.get("format") == "dcd"
                            and trajectory_probe.get("format") == "dcd"
                        ):
                            previous_start = int(previous_probe["starting_step"])
                            dcd_previous_frames = int(previous_probe["declared_frame_count"])
                            previous_interval = int(previous_probe["save_interval_steps"])
                            expected_start = previous_start + dcd_previous_frames * previous_interval
                            observed_start = int(trajectory_probe["starting_step"])
                            if observed_start != expected_start:
                                declared_reset = (
                                    dcd_header_step_policy == "reset_per_segment"
                                    and observed_start == previous_start
                                )
                                if declared_reset or (
                                    previous_start == 0 and observed_start == 0
                                ):
                                    basis = (
                                        "the declared reset_per_segment policy"
                                        if declared_reset
                                        else "matching zero-valued headers"
                                    )
                                    issues.append(
                                        issue_record(
                                            "warning", "DCD_HEADER_STEP_RESET", segment_location,
                                            "adjacent DCD headers repeat their starting step under "
                                            f"{basis}; "
                                            "header step continuity is unavailable and continuity "
                                            "rests on the declared frame axis and external lineage",
                                        )
                                    )
                                else:
                                    issues.append(
                                        issue_record(
                                            "error", "DCD_CONTINUITY_MISMATCH", segment_location,
                                            f"continuous segment starts at step {observed_start}; "
                                            f"expected {expected_start} from the preceding DCD header",
                                        )
                                    )
                except FileProbeError as exc:
                    issues.append(
                        issue_record("error", "TRAJECTORY_PROBE_FAILED", segment_location, str(exc))
                    )
                segment_report: Dict[str, object] = {
                        "segment_id": segment_id,
                        "continuous_with_previous": bool(
                            raw_segment.get("continuous_with_previous", False)
                        ),
                        "dcd_header_step_policy": str(
                            raw_segment.get("dcd_header_step_policy", "continuous")
                        ),
                        "frame_axis": axis,
                        "trajectory": trajectory_probe,
                    }
                if axis["kind"] == "physical_time":
                    segment_report["timing"] = axis["timing"]
                else:
                    segment_report["sample_axis"] = axis["sample_axis"]
                segment_reports.append(segment_report)
                previous_probe = trajectory_probe
                previous_axis = axis
            replica_reports.append(
                {
                    "replica_id": replica_id,
                    "topology": topology_probe,
                    "connectivity": connectivity_probe,
                    "force_field_parameters": force_field_parameter_probe,
                    "segments": segment_reports,
                }
            )
        system_reports.append({"system_id": system_id, "replicas": replica_reports})
    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": inventory["manifest_sha256"],
        "content_hashes_included": hash_content,
        "input_content_signature_sha256": (
            inventory_content_signature_sha256(inventory) if hash_content else None
        ),
        "technical_status": "failed" if error_count else "complete",
        "scientific_status": "not evaluated",
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
        "systems": system_reports,
        "limitations": [
            "Preflight metadata does not establish equilibration, convergence, adequate sampling, population meaning, or scientific validity.",
            "DCD support currently inspects declared header metadata only; coordinate records and actual frame counts require a later full reader backend.",
            "PDB, GRO, PSF, PRMTOP, and XYZ probes validate limited structural records, not force-field or chemical correctness.",
            "For continuous segments, the next first-frame time is expected one preceding frame interval after the preceding final frame; duplicated boundary frames require separate segments or an explicit data correction.",
        ],
    }
