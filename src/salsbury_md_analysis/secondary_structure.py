"""External mkdssp adapter with explicit executable and frame provenance."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .hydrogen_bond_chemistry import PROTEIN_RESIDUES
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .trajectory_contracts import (
    TrajectoryContractError,
    frame_axis_value,
    normalize_segment_axis,
)


class SecondaryStructureAnalysisError(ValueError):
    """Raised when DSSP cannot be executed or parsed under its declared contract."""


def build_mkdssp_command(
    executable: str, input_path: Path, output_path: Path, version_text: str
) -> List[str]:
    """Build the version-appropriate classic-DSSP command without a shell."""

    match = re.search(r"\b(\d+)\.(\d+)(?:\.(\d+))?\b", version_text)
    if match is None:
        raise SecondaryStructureAnalysisError(
            "mkdssp version output does not contain a parseable semantic version"
        )
    major = int(match.group(1))
    if major >= 4:
        return [
            executable,
            "--output-format",
            "dssp",
            str(input_path),
            str(output_path),
        ]
    return [executable, "-i", str(input_path), "-o", str(output_path)]


def parse_dssp_text(text: str) -> List[Dict[str, object]]:
    """Parse classic fixed-column DSSP residue assignments."""

    in_table = False
    rows = []
    for line in text.splitlines():
        if line.lstrip().startswith("#") and "RESIDUE" in line and "STRUCTURE" in line:
            in_table = True
            continue
        if not in_table or len(line) < 17:
            continue
        if line[13:14] == "!":
            continue
        residue_token = line[5:11].strip()
        chain_id = line[11:12].strip()
        amino_acid = line[13:14].strip()
        if not residue_token or not amino_acid:
            continue
        code = line[16:17].strip() or "C"
        rows.append({
            "dssp_residue_token": residue_token, "chain_id": chain_id,
            "amino_acid_code": amino_acid, "secondary_structure_code": code,
        })
    if not rows:
        raise SecondaryStructureAnalysisError("mkdssp output contains no parseable residue assignments")
    return rows


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("secondary_structure") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict):
        raise SecondaryStructureAnalysisError("definitions.secondary_structure must be an object")
    required = {"method", "executable", "frame_stride", "maximum_frames"}
    missing = sorted(required.difference(raw))
    unknown = sorted(set(raw).difference(required | {"frame_selection"}))
    if missing:
        raise SecondaryStructureAnalysisError("secondary-structure settings missing: " + ", ".join(missing))
    if unknown:
        raise SecondaryStructureAnalysisError("secondary-structure settings contain unknown fields: " + ", ".join(unknown))
    if raw["method"] != "mkdssp":
        raise SecondaryStructureAnalysisError("method currently supports only mkdssp")
    executable = str(raw["executable"]).strip()
    if not executable:
        raise SecondaryStructureAnalysisError("executable must be nonempty")
    for label in ("frame_stride", "maximum_frames"):
        value = raw[label]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SecondaryStructureAnalysisError(f"{label} must be a positive integer")
    result = dict(raw)
    result["frame_selection"] = normalize_frame_selection(
        raw.get("frame_selection"), int(result["frame_stride"]),
        error_type=SecondaryStructureAnalysisError,
    )
    return result


def _pdb_template_lines(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    records = [line for line in lines if line[:6].strip().upper() in {"ATOM", "HETATM"}]
    if not records:
        raise SecondaryStructureAnalysisError("topology PDB contains no coordinate records")
    return records


def _frame_pdb_payload(
    template: Sequence[str],
    atoms: Sequence[AtomRecord],
    coordinates: Sequence[Sequence[float]],
) -> Tuple[str, Dict[Tuple[str, str], Dict[str, object]]]:
    """Create a DSSP-safe protein PDB and a reversible residue identity map."""

    if len(template) != len(coordinates) or len(atoms) != len(coordinates):
        raise SecondaryStructureAnalysisError("PDB template and trajectory atom counts differ")
    protein_residues = _protein_residue_keys(atoms)
    residue_numbers: Dict[Tuple[str, int, str, str], int] = {}
    next_residue_by_chain: Dict[str, int] = {}
    residue_mapping: Dict[Tuple[str, str], Dict[str, object]] = {}
    output = []
    serial = 0
    for line, atom, coordinate in zip(template, atoms, coordinates):
        residue_key = (
            atom.chain_id,
            atom.residue_number,
            atom.insertion_code,
            atom.residue_name,
        )
        if (
            line[:6].strip().upper() != "ATOM"
            or residue_key not in protein_residues
        ):
            continue
        if residue_key not in residue_numbers:
            next_residue_by_chain[atom.chain_id] = (
                next_residue_by_chain.get(atom.chain_id, 0) + 1
            )
            residue_numbers[residue_key] = next_residue_by_chain[atom.chain_id]
        dssp_residue_number = residue_numbers[residue_key]
        serial += 1
        if serial > 99999 or dssp_residue_number > 9999:
            raise SecondaryStructureAnalysisError(
                "DSSP-safe PDB numbering exceeds the classic PDB field width"
            )
        residue_mapping[(atom.chain_id, str(dssp_residue_number))] = {
            "chain_id": atom.chain_id,
            "residue_number": atom.residue_number,
            "insertion_code": atom.insertion_code,
            "residue_name": atom.residue_name,
            "original_residue_token": f"{atom.residue_number}{atom.insertion_code}",
            "dssp_sequential_residue_number": dssp_residue_number,
        }
        element = atom.element.strip().upper()
        if not element or len(element) > 2 or not element.isalpha():
            raise SecondaryStructureAnalysisError(
                f"atom {atom.atom_index} has no DSSP-safe chemical element"
            )
        padded = line.ljust(80)
        normalized = (
            padded[:6]
            + f"{serial:5d}"
            + padded[11:22]
            + f"{dssp_residue_number:4d}"
            + " "
            + padded[27:30]
            + f"{coordinate[0]:8.3f}{coordinate[1]:8.3f}{coordinate[2]:8.3f}"
            + padded[54:]
        )
        normalized = normalized.ljust(80)
        output.append(normalized[:76] + f"{element:>2s}" + normalized[78:])
    if not output or not residue_mapping:
        raise SecondaryStructureAnalysisError(
            "topology contains no protein ATOM records for DSSP"
        )
    return "\n".join(output + ["END"]) + "\n", residue_mapping


def _protein_residue_keys(
    atoms: Sequence[AtomRecord],
) -> set[Tuple[str, int, str, str]]:
    return {
        (
            atom.chain_id,
            atom.residue_number,
            atom.insertion_code,
            atom.residue_name,
        )
        for atom in atoms
        if atom.residue_name.upper() in PROTEIN_RESIDUES
    }


def _mkdssp_environment(executable: str) -> Tuple[Dict[str, str], Optional[str], str]:
    environment = dict(os.environ)
    declared = environment.get("LIBCIFPP_DATA_DIR")
    if declared:
        return environment, declared, "environment"
    candidate = Path(executable).resolve().parent.parent / "share" / "libcifpp"
    if candidate.is_dir():
        environment["LIBCIFPP_DATA_DIR"] = str(candidate)
        return environment, str(candidate), "executable-prefix"
    return environment, None, "mkdssp-default"


def secondary_structure_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    executable = shutil.which(str(settings["executable"]))
    if executable is None:
        raise SecondaryStructureAnalysisError(
            f"mkdssp executable {settings['executable']!r} is unavailable on PATH"
        )
    version_process = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False, timeout=30
    )
    version_text = (version_process.stdout or version_process.stderr).strip()
    if version_process.returncode != 0 or not version_text:
        raise SecondaryStructureAnalysisError("mkdssp --version failed")
    mkdssp_environment, data_directory, data_directory_source = _mkdssp_environment(
        executable
    )
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(context["system_manifest_path"])
    system = load_json(system_path)
    coordinate_unit = str(project["coordinate_unit"])
    output_time_unit = project.get("time_unit")
    periodic_policy = str(project["periodic_coordinate_policy"])
    frame_selection_plan, frame_selection_report = plan_frame_selection(
        system, system_path, coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        maximum_selected_frames=int(settings["maximum_frames"]),
        error_type=SecondaryStructureAnalysisError,
    )
    issues = [issue for issue in context.get("warnings", []) if isinstance(issue, dict)]
    counts: Dict[tuple, Dict[str, int]] = {}
    evaluated_frames = 0
    frame_reports = []
    normalization_reports: Dict[Tuple[str, str], Dict[str, object]] = {}
    skipped_nonprotein_replicas: List[Dict[str, object]] = []
    applicable_replica_count = 0
    with tempfile.TemporaryDirectory(prefix="salsbury-dssp-") as temporary:
        temporary_path = Path(temporary)
        for raw_system in system["systems"]:
            system_id = str(raw_system["system_id"])
            for replica in raw_system["replicas"]:
                replica_id = str(replica["replica_id"])
                topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
                topology_format, atoms = read_topology_atoms(topology_path)
                if topology_format != "pdb":
                    raise SecondaryStructureAnalysisError("mkdssp adapter currently requires a PDB topology")
                protein_residues = _protein_residue_keys(atoms)
                if not protein_residues:
                    skipped_nonprotein_replicas.append({
                        "system_id": system_id,
                        "replica_id": replica_id,
                        "reason": "no standard protein residue",
                    })
                    issues.append({
                        "severity": "info",
                        "code": "SECONDARY_STRUCTURE_NOT_APPLICABLE",
                        "location": f"{system_id}/{replica_id}",
                        "message": (
                            "DSSP is protein-specific; this DNA/RNA/ligand-only replica "
                            "was retained in the campaign but not passed to mkdssp."
                        ),
                    })
                    continue
                applicable_replica_count += 1
                template = _pdb_template_lines(topology_path)
                processor = PeriodicFrameProcessor.from_replica(
                    project, replica, system_path, len(atoms)
                )
                for segment in replica["segments"]:
                    segment_id = str(segment["segment_id"])
                    trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                    selected_indices = frame_selection_plan[(
                        system_id, replica_id, segment_id,
                    )]
                    axis = normalize_segment_axis(segment, str(output_time_unit) if output_time_unit else None)
                    periodic_frames = 0
                    processor.begin_segment(
                        bool(segment.get("continuous_with_previous", False))
                    )
                    reader_indices = reader_frame_indices(
                        selected_indices, processor.policy
                    )
                    for raw_frame in iter_coordinate_frames(
                        trajectory_path, coordinate_unit, reader_indices
                    ):
                        selected = frame_selected(
                            raw_frame.frame_index, selected_indices,
                            int(settings["frame_stride"]),
                        )
                        if not selected and processor.policy != "unwrap_continuous":
                            continue
                        frame = processor.process(
                            raw_frame,
                            f"{system_id}/{replica_id}/{segment_id}/frame-{raw_frame.frame_index}",
                        )
                        if frame.atom_count != len(atoms):
                            raise SecondaryStructureAnalysisError("trajectory/topology atom count mismatch")
                        periodic_frames += int(frame.periodic_cell_present)
                        if not selected:
                            continue
                        evaluated_frames += 1
                        if evaluated_frames > int(settings["maximum_frames"]):
                            raise SecondaryStructureAnalysisError("maximum_frames gate exceeded")
                        input_path = temporary_path / "frame.pdb"
                        output_path = temporary_path / "frame.dssp"
                        pdb_text, residue_mapping = _frame_pdb_payload(
                            template, atoms, frame.coordinates_angstrom
                        )
                        input_path.write_text(pdb_text, encoding="utf-8")
                        normalization_reports[(system_id, replica_id)] = {
                            "system_id": system_id,
                            "replica_id": replica_id,
                            "topology_atom_count": len(atoms),
                            "dssp_protein_atom_count": sum(
                                (
                                    atom.chain_id,
                                    atom.residue_number,
                                    atom.insertion_code,
                                    atom.residue_name,
                                ) in protein_residues
                                for atom in atoms
                            ),
                            "dssp_residue_count": len(residue_mapping),
                            "normalization": (
                                "HETATM records excluded; ATOM serials and residues were "
                                "renumbered sequentially within each chain for classic-PDB "
                                "compatibility; original residue identities were restored in output"
                            ),
                        }
                        process = subprocess.run(
                            build_mkdssp_command(
                                executable, input_path, output_path, version_text
                            ),
                            env=mkdssp_environment,
                            capture_output=True, text=True, check=False, timeout=120,
                        )
                        if process.returncode != 0 or not output_path.is_file():
                            raise SecondaryStructureAnalysisError(
                                f"mkdssp failed for {system_id}/{replica_id}/{segment_id}/frame-{frame.frame_index}: "
                                + (process.stderr.strip() or f"exit {process.returncode}")
                            )
                        assignments = parse_dssp_text(output_path.read_text(encoding="utf-8", errors="strict"))
                        axis_value = frame_axis_value(axis, frame.frame_index)
                        for assignment in assignments:
                            mapping_key = (
                                str(assignment["chain_id"]),
                                str(assignment["dssp_residue_token"]),
                            )
                            original = residue_mapping.get(mapping_key)
                            if original is None:
                                raise SecondaryStructureAnalysisError(
                                    "mkdssp returned a residue identity absent from the reversible input map: "
                                    + "/".join(mapping_key)
                                )
                            key = (
                                system_id,
                                replica_id,
                                str(original["chain_id"]),
                                int(original["residue_number"]),
                                str(original["insertion_code"]),
                                str(original["residue_name"]),
                                int(original["dssp_sequential_residue_number"]),
                            )
                            code_counts = counts.setdefault(key, {})
                            code = str(assignment["secondary_structure_code"])
                            code_counts[code] = code_counts.get(code, 0) + 1
                        frame_reports.append({
                            "system_id": system_id, "replica_id": replica_id,
                            "segment_id": segment_id, "source_frame_index": frame.frame_index,
                            "axis_kind": axis["kind"], "axis_value": axis_value,
                            "assignment_count": len(assignments),
                        })
                    if periodic_frames and periodic_policy == "allow_wrapped_diagnostic":
                        issues.append({
                            "severity": "warning", "code": "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                            "location": f"{system_id}/{replica_id}/{segment_id}",
                            "message": f"{periodic_frames} periodic frames were passed to mkdssp without molecule reconstruction",
                        })
    populations = []
    for key in sorted(counts):
        total = sum(counts[key].values())
        populations.append({
            "system_id": key[0], "replica_id": key[1], "chain_id": key[2],
            "dssp_residue_token": f"{key[3]}{key[4]}",
            "original_residue_number": key[3],
            "original_insertion_code": key[4],
            "original_residue_name": key[5],
            "dssp_sequential_residue_number": key[6],
            "evaluated_frame_count": total,
            "code_counts": counts[key],
            "code_fractions": {code: count / total for code, count in sorted(counts[key].items())},
        })
    if int(frame_selection_report["selected_frame_count"]) < int(
        frame_selection_report["source_frame_count"]
    ):
        issues.append({
            "severity": "warning", "code": "FRAME_SUBSAMPLING",
            "location": str(source),
            "message": (
                f"DSSP evaluated {frame_selection_report['selected_frame_count']} of "
                f"{frame_selection_report['source_frame_count']} source frames under "
                f"{frame_selection_report['mode']}"
            ),
        })
    return {
        "module_id": "secondary_structure", "technical_status": "complete",
        "scientific_status": (
            "not_applicable" if applicable_replica_count == 0 else "not evaluated"
        ), "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content, "settings": settings,
        "frame_selection": frame_selection_report,
        "implementation": {
            "executable_path": executable,
            "version_output": version_text,
            "command_contract": "positional" if int(re.search(r"\b(\d+)\.", version_text).group(1)) >= 4 else "legacy-i-o",
            "shell": False,
            "libcifpp_data_directory": data_directory,
            "libcifpp_data_directory_source": data_directory_source,
        },
        "evaluated_frame_count": evaluated_frames, "frame_reports": frame_reports,
        "applicable_replica_count": applicable_replica_count,
        "skipped_nonprotein_replicas": skipped_nonprotein_replicas,
        "input_normalization_reports": list(normalization_reports.values()),
        "residue_populations": populations, "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "DSSP is executed externally and its executable path and version output are retained.",
            "The executable's assignment alphabet is preserved; DSSP 4.6 PPII code P requires an explicit mapping before comparison with older DSSP alphabets that report those residues as coil.",
            "Temporary per-frame PDB files are generated locally and removed after execution.",
            "Periodic production DSSP requires explicit connectivity and make_whole or unwrap_continuous preprocessing.",
            "Pooled residue populations must be paired with replica-sensitive convergence analysis.",
        ],
    }


def secondary_structure_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return secondary_structure_project(project_path, hash_content=hash_content)
    except (
        ManifestValidationError, SecondaryStructureAnalysisError, AtomMappingError,
        CoordinateReadError, PeriodicReconstructionError, TrajectoryContractError, OSError, subprocess.SubprocessError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "secondary_structure", "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{"severity": "error", "code": "SECONDARY_STRUCTURE_INVALID", "message": message} for message in messages],
        }
