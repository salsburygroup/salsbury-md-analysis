"""External x3dna-dssr JSON adapter for nucleic-acid structural motifs."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .atom_mapping import AtomMappingError, AtomRecord, read_topology_atoms
from .context import compile_project_context_file
from .coordinates import CoordinateReadError, iter_coordinate_frames
from .frame_sampling import (
    frame_selected, normalize_frame_selection, plan_frame_selection,
    reader_frame_indices,
)
from .manifests import ManifestValidationError, load_json, resolve_manifest_path
from .moments import sample_summary
from .periodic import PeriodicFrameProcessor, PeriodicReconstructionError
from .replica_execution import ReplicaPartial
from .replica_module_execution import (
    execute_replica_final_module,
    merge_frame_selection_reports,
    restore_source_provenance,
    unique_issues,
)
from .trajectory_contracts import (
    TrajectoryContractError,
    frame_axis_value,
    normalize_segment_axis,
)
from .validation import positive_integer


class NucleicAcidStructureError(ValueError):
    """Raised when DSSR execution or JSON interpretation is unsafe."""


def build_dssr_json_command(
    executable: str, input_path: Path, output_path: Path
) -> List[str]:
    """Build the declared x3dna-dssr JSON command without invoking a shell."""

    return [
        executable,
        "--json",
        "--more",
        f"--input={input_path}",
        f"--output={output_path}",
    ]


def parse_dssr_collection_counts(
    payload: Mapping[str, object], fields: Sequence[str]
) -> Dict[str, int]:
    """Count declared top-level DSSR JSON collections without guessing aliases."""

    result = {}
    for field in fields:
        value = payload.get(field)
        if value is None:
            result[field] = 0
        elif isinstance(value, (list, dict)):
            result[field] = len(value)
        elif isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[field] = value
        else:
            raise NucleicAcidStructureError(
                f"DSSR JSON field {field!r} is not a collection or nonnegative count"
            )
    return result


def extract_numeric_json_path(
    payload: object, path: Sequence[str]
) -> List[float]:
    """Extract finite numbers through a declared JSON path with ``*`` wildcards."""

    if not path or any(not isinstance(token, str) or not token for token in path):
        raise NucleicAcidStructureError("DSSR numeric-query path must be nonempty strings")
    current = [payload]
    for token in path:
        following = []
        for value in current:
            if token == "*":
                if isinstance(value, list):
                    following.extend(value)
                elif isinstance(value, dict):
                    following.extend(value[key] for key in sorted(value))
                else:
                    continue
            elif isinstance(value, dict) and token in value:
                following.append(value[token])
        current = following
    result = []
    for value in current:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NucleicAcidStructureError(
                "DSSR numeric-query path resolved to a nonnumeric value"
            )
        number = float(value)
        if not math.isfinite(number):
            raise NucleicAcidStructureError(
                "DSSR numeric-query path resolved to a non-finite value"
            )
        result.append(number)
    return result


def discover_helical_step_parameter_path(
    payload: Mapping[str, object]
) -> Dict[str, object] | None:
    """Find one DSSR object path containing the six standard step parameters.

    DSSR JSON has evolved across releases.  Preparation probes the installed
    executable and freezes the actually observed object path instead of
    guessing a version-specific alias at analysis time.
    """

    required = ("shift", "slide", "rise", "tilt", "roll", "twist")
    candidates: list[tuple[tuple[str, ...], Dict[str, str]]] = []

    def visit(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            normalized = {str(key).lower(): str(key) for key in value}
            if set(required).issubset(normalized):
                fields = {name: normalized[name] for name in required}
                if all(
                    isinstance(value[fields[name]], (int, float))
                    and not isinstance(value[fields[name]], bool)
                    for name in required
                ):
                    candidates.append((path, fields))
            for key, child in value.items():
                visit(child, (*path, str(key)))
        elif isinstance(value, list):
            for child in value:
                visit(child, (*path, "*"))

    visit(payload, ())
    if not candidates:
        return None
    unique = {(path, tuple(sorted(fields.items()))): fields for path, fields in candidates}
    ranked = sorted(
        unique,
        key=lambda item: (
            0 if any("step" in token.lower() for token in item[0]) else 1,
            0 if any(token.lower() in {"stems", "helices"} for token in item[0]) else 1,
            len(item[0]), item[0],
        ),
    )
    path, field_rows = ranked[0]
    return {"object_path": list(path), "fields": dict(field_rows)}


def probe_dssr_reference_duplex(
    executable: str, reference_path: Path, *, timeout_seconds: float = 120.0
) -> Dict[str, object]:
    """Probe one reference structure for DSSR, a duplex stem, and step fields."""

    resolved = shutil.which(executable)
    if resolved is None:
        candidate = Path(executable).expanduser()
        if candidate.is_file():
            resolved = str(candidate.resolve(strict=True))
    if resolved is None:
        return {
            "status": "not_available", "reason": "dssr_not_installed",
            "executable": executable,
        }
    reference = Path(reference_path).expanduser().resolve(strict=True)
    try:
        version = subprocess.run(
            [resolved, "--version"], capture_output=True, text=True,
            check=False, timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "status": "not_available", "reason": "dssr_version_probe_failed",
            "executable": resolved, "detail": str(exc),
        }
    version_text = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or not version_text:
        return {
            "status": "not_available", "reason": "dssr_version_probe_failed",
            "executable": resolved,
        }
    with tempfile.TemporaryDirectory(prefix="salsbury-dssr-probe-") as temporary:
        output = Path(temporary) / "reference.json"
        try:
            run = subprocess.run(
                build_dssr_json_command(resolved, reference, output),
                capture_output=True, text=True, check=False, timeout=timeout_seconds,
            )
            if run.returncode != 0 or not output.is_file():
                return {
                    "status": "not_available", "reason": "dssr_reference_probe_failed",
                    "executable": resolved, "version_output": version_text,
                    "detail": (run.stderr or run.stdout).strip(),
                }
            payload = json.loads(output.read_text(encoding="utf-8", errors="strict"))
        except (OSError, UnicodeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            return {
                "status": "not_available", "reason": "dssr_reference_probe_failed",
                "executable": resolved, "version_output": version_text,
                "detail": str(exc),
            }
    if not isinstance(payload, dict):
        return {
            "status": "not_available", "reason": "dssr_reference_json_invalid",
            "executable": resolved, "version_output": version_text,
        }
    counts = parse_dssr_collection_counts(payload, ("pairs", "helices", "stems"))
    if counts["stems"] < 1:
        return {
            "status": "not_available", "reason": "no_duplex_dna_or_rna",
            "executable": resolved, "version_output": version_text,
            "collection_counts": counts,
        }
    parameters = discover_helical_step_parameter_path(payload)
    if parameters is None:
        return {
            "status": "not_available",
            "reason": "dssr_helical_step_descriptors_unavailable",
            "executable": resolved, "version_output": version_text,
            "collection_counts": counts,
        }
    return {
        "status": "available", "reason": None,
        "executable": str(Path(resolved).resolve(strict=True)),
        "version_output": version_text, "collection_counts": counts,
        "helical_step_parameters": parameters,
    }


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("nucleic_acid_structure") if isinstance(definitions, dict) else None
    required = {
        "method", "executable", "frame_stride", "maximum_frames",
        "timeout_seconds", "json_collection_fields",
    }
    allowed = required | {"numeric_queries", "frame_selection"}
    if not isinstance(raw, dict) or not required.issubset(raw) or not set(raw).issubset(allowed):
        raise NucleicAcidStructureError(
            "definitions.nucleic_acid_structure fields do not match the contract"
        )
    if raw["method"] != "x3dna-dssr-json":
        raise NucleicAcidStructureError("method must be x3dna-dssr-json")
    executable = str(raw["executable"]).strip()
    if not executable:
        raise NucleicAcidStructureError("executable must be nonempty")
    frame_stride = positive_integer(
        raw["frame_stride"], "frame_stride", error_type=NucleicAcidStructureError
    )
    maximum_frames = positive_integer(
        raw["maximum_frames"], "maximum_frames", error_type=NucleicAcidStructureError
    )
    timeout = raw["timeout_seconds"]
    if (
        isinstance(timeout, bool) or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout)) or float(timeout) <= 0.0
    ):
        raise NucleicAcidStructureError("timeout_seconds must be finite and positive")
    fields = raw["json_collection_fields"]
    if (
        not isinstance(fields, list) or not fields
        or any(not isinstance(field, str) or not field.strip() for field in fields)
        or len(set(fields)) != len(fields)
    ):
        raise NucleicAcidStructureError(
            "json_collection_fields must be a nonempty unique string array"
        )
    queries = raw.get("numeric_queries", [])
    if not isinstance(queries, list):
        raise NucleicAcidStructureError("numeric_queries must be an array")
    query_ids = set()
    normalized_queries = []
    for index, query in enumerate(queries):
        if not isinstance(query, dict) or set(query) != {
            "query_id", "path", "missing_policy"
        }:
            raise NucleicAcidStructureError(
                f"numeric query {index} fields do not match the contract"
            )
        query_id = str(query["query_id"]).strip()
        path = query["path"]
        policy = query["missing_policy"]
        if (
            not query_id or query_id in query_ids or not isinstance(path, list)
            or not path or any(not isinstance(token, str) or not token for token in path)
            or policy not in {"skip", "fail"}
        ):
            raise NucleicAcidStructureError("DSSR numeric query is invalid")
        query_ids.add(query_id)
        normalized_queries.append({
            "query_id": query_id, "path": list(path), "missing_policy": policy,
        })
    return {
        "method": "x3dna-dssr-json",
        "executable": executable,
        "frame_stride": frame_stride,
        "frame_selection": normalize_frame_selection(
            raw.get("frame_selection"), frame_stride,
            error_type=NucleicAcidStructureError,
        ),
        "maximum_frames": maximum_frames,
        "timeout_seconds": float(timeout),
        "json_collection_fields": [str(field) for field in fields],
        "numeric_queries": normalized_queries,
    }


def _pdb_payload(
    atoms: Sequence[AtomRecord], coordinates: Sequence[Sequence[float]]
) -> str:
    if len(atoms) != len(coordinates):
        raise NucleicAcidStructureError("topology/trajectory atom count mismatch")
    rows = []
    for serial, (atom, coordinate) in enumerate(zip(atoms, coordinates), start=1):
        if serial > 99999 or not -999 <= atom.residue_number <= 9999:
            raise NucleicAcidStructureError("DSSR PDB numbering exceeds fixed-column limits")
        x, y, z = coordinate
        rows.append(
            f"ATOM  {serial:5d} {atom.atom_name:^4s}{atom.altloc[:1]:1s}"
            f"{atom.residue_name:>3s} {atom.chain_id[:1]:1s}{atom.residue_number:4d}"
            f"{atom.insertion_code[:1]:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00          {atom.element[:2]:>2s}\n"
        )
    return "".join(rows) + "END\n"


def _nucleic_acid_structure_project_serial(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    executable = shutil.which(str(settings["executable"]))
    if executable is None:
        raise NucleicAcidStructureError(
            f"DSSR executable {settings['executable']!r} is unavailable"
        )
    version = subprocess.run(
        [executable, "--version"], capture_output=True, text=True,
        check=False, timeout=float(settings["timeout_seconds"]),
    )
    version_text = (version.stdout or version.stderr).strip()
    if version.returncode != 0 or not version_text:
        raise NucleicAcidStructureError("x3dna-dssr --version failed")
    context = compile_project_context_file(source, hash_content=hash_content)
    system_path = Path(str(context["system_manifest_path"]))
    system = load_json(system_path)
    coordinate_unit = str(project["coordinate_unit"])
    output_time_unit = project.get("time_unit")
    periodic_policy = str(project["periodic_coordinate_policy"])
    frame_selection_plan, frame_selection_report = plan_frame_selection(
        system, system_path, coordinate_unit,
        settings["frame_selection"],  # type: ignore[arg-type]
        frame_stride=int(settings["frame_stride"]),
        maximum_selected_frames=int(settings["maximum_frames"]),
        error_type=NucleicAcidStructureError,
    )
    issues = [
        issue for issue in context.get("issues", []) if isinstance(issue, dict)
    ]
    frames = []
    evaluated = 0
    with tempfile.TemporaryDirectory(prefix="salsbury-dssr-") as temporary:
        root = Path(temporary)
        for raw_system in system["systems"]:
            assert isinstance(raw_system, dict)
            system_id = str(raw_system["system_id"])
            for replica in raw_system["replicas"]:
                assert isinstance(replica, dict)
                replica_id = str(replica["replica_id"])
                topology_path = resolve_manifest_path(str(replica["topology"]), system_path)
                _, atoms = read_topology_atoms(topology_path)
                processor = PeriodicFrameProcessor.from_replica(
                    project, replica, system_path, len(atoms)
                )
                for segment in replica["segments"]:
                    assert isinstance(segment, dict)
                    segment_id = str(segment["segment_id"])
                    trajectory_path = resolve_manifest_path(str(segment["trajectory"]), system_path)
                    selected_indices = frame_selection_plan[(
                        system_id, replica_id, segment_id,
                    )]
                    axis = normalize_segment_axis(
                        segment, str(output_time_unit) if output_time_unit else None
                    )
                    processor.begin_segment(bool(segment.get("continuous_with_previous", False)))
                    periodic_frames = 0
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
                        periodic_frames += int(frame.periodic_cell_present)
                        if not selected:
                            continue
                        evaluated += 1
                        if evaluated > int(settings["maximum_frames"]):
                            raise NucleicAcidStructureError("maximum_frames gate exceeded")
                        input_path = root / "frame.pdb"
                        output_path = root / "frame.json"
                        input_path.write_text(
                            _pdb_payload(atoms, frame.coordinates_angstrom), encoding="ascii"
                        )
                        process = subprocess.run(
                            build_dssr_json_command(executable, input_path, output_path),
                            capture_output=True, text=True, check=False,
                            timeout=float(settings["timeout_seconds"]),
                        )
                        if process.returncode != 0 or not output_path.is_file():
                            raise NucleicAcidStructureError(
                                f"DSSR failed for {system_id}/{replica_id}/{segment_id}/frame-{frame.frame_index}: "
                                + (process.stderr.strip() or f"exit {process.returncode}")
                            )
                        try:
                            payload = json.loads(output_path.read_text(encoding="utf-8"))
                        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                            raise NucleicAcidStructureError(
                                "DSSR output is not valid JSON"
                            ) from exc
                        if not isinstance(payload, dict):
                            raise NucleicAcidStructureError("DSSR JSON root must be an object")
                        counts = parse_dssr_collection_counts(
                            payload, settings["json_collection_fields"]  # type: ignore[arg-type]
                        )
                        numeric_queries = []
                        for query in settings["numeric_queries"]:  # type: ignore[union-attr]
                            values = extract_numeric_json_path(payload, query["path"])
                            if not values and query["missing_policy"] == "fail":
                                raise NucleicAcidStructureError(
                                    f"DSSR numeric query {query['query_id']} resolved no values"
                                )
                            numeric_queries.append({
                                "query_id": query["query_id"],
                                "path": query["path"],
                                "value_count": len(values),
                                "values": values,
                                "summary": sample_summary(values) if values else None,
                            })
                        frames.append({
                            "system_id": system_id,
                            "replica_id": replica_id,
                            "segment_id": segment_id,
                            "source_frame_index": frame.frame_index,
                            "axis_kind": axis["kind"],
                            "axis_value": frame_axis_value(axis, frame.frame_index),
                            "collection_counts": counts,
                            "numeric_queries": numeric_queries,
                            "observed_json_keys": sorted(payload),
                        })
                    if periodic_frames and periodic_policy == "allow_wrapped_diagnostic":
                        issues.append({
                            "severity": "warning",
                            "code": "PERIODIC_COORDINATES_NOT_UNWRAPPED",
                            "location": f"{system_id}/{replica_id}/{segment_id}",
                            "message": f"{periodic_frames} periodic frames were passed to DSSR without connectivity-aware reconstruction",
                        })
    replica_keys = sorted({
        (str(row["system_id"]), str(row["replica_id"])) for row in frames
    })
    summaries = []
    for system_id, replica_id in replica_keys:
        selected = [
            row for row in frames
            if row["system_id"] == system_id and row["replica_id"] == replica_id
        ]
        summaries.append({
            "system_id": system_id,
            "replica_id": replica_id,
            "evaluated_frame_count": len(selected),
            "collection_count_summaries": {
                field: sample_summary([
                    float(row["collection_counts"][field]) for row in selected
                ])
                for field in settings["json_collection_fields"]  # type: ignore[union-attr]
            },
            "numeric_query_summaries": {
                query["query_id"]: (
                    sample_summary([
                        value for row in selected
                        for report in row["numeric_queries"]
                        if report["query_id"] == query["query_id"]
                        for value in report["values"]
                    ])
                    if any(
                        report["query_id"] == query["query_id"] and report["values"]
                        for row in selected for report in row["numeric_queries"]
                    ) else None
                )
                for query in settings["numeric_queries"]  # type: ignore[union-attr]
            },
        })
    if int(frame_selection_report["selected_frame_count"]) < int(
        frame_selection_report["source_frame_count"]
    ):
        issues.append({
            "severity": "warning", "code": "FRAME_SUBSAMPLING",
            "location": str(source),
            "message": (
                f"DSSR evaluated {frame_selection_report['selected_frame_count']} of "
                f"{frame_selection_report['source_frame_count']} source frames under "
                f"{frame_selection_report['mode']}"
            ),
        })
    return {
        "module_id": "nucleic_acid_structure",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(source),
        "project_manifest_sha256": context["project_manifest_sha256"],
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": context["system_manifest_sha256"],
        "input_content_signature_sha256": context["input_content_signature_sha256"],
        "content_hashes_included": hash_content,
        "settings": settings,
        "frame_selection": frame_selection_report,
        "implementation": {
            "executable_path": executable,
            "version_output": version_text,
            "command_contract": "x3dna-dssr --json --more --input= --output=",
            "shell": False,
        },
        "evaluated_frame_count": evaluated,
        "frame_reports": frames,
        "replica_summaries": summaries,
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "DSSR is an external dependency; executable path, version output, command contract, and requested JSON fields are retained.",
            "Declared wildcard numeric queries can retain base-pair, base-step, groove, helical, backbone, or sugar descriptors exposed by the installed DSSR JSON version; paths and missing-value policies are explicit.",
            "Absent declared JSON collections are counted as zero, while malformed present collections fail closed.",
            "Temporary frame PDB and JSON files are removed after execution.",
            "Periodic production DSSR requires connectivity-aware make_whole or unwrap_continuous preprocessing.",
            "Motif counts are descriptive and require replica-sensitive convergence analysis.",
        ],
    }


def _reduce_nucleic_acid_structure_reports(
    partials: Sequence[ReplicaPartial[Dict[str, object]]],
    source_context: Dict[str, object],
) -> Dict[str, object]:
    reports = [partial.value for partial in partials]
    first = dict(reports[0])
    for report in reports[1:]:
        for key in ("module_id", "settings", "implementation"):
            if report.get(key) != first.get(key):
                raise NucleicAcidStructureError(
                    f"replica nucleic-acid reports disagree on {key}"
                )
    first["frame_selection"] = merge_frame_selection_reports([
        report["frame_selection"] for report in reports
        if isinstance(report.get("frame_selection"), dict)
    ])
    for key in ("frame_reports", "replica_summaries"):
        first[key] = [
            row for report in reports for row in report.get(key, [])
        ]
    first["evaluated_frame_count"] = sum(
        int(report.get("evaluated_frame_count", 0)) for report in reports
    )
    maximum = int(first["settings"]["maximum_frames"])  # type: ignore[index]
    if int(first["evaluated_frame_count"]) > maximum:
        raise NucleicAcidStructureError(
            "replica workers collectively exceeded the project maximum_frames gate"
        )
    issues = unique_issues(reports)
    first["issues"] = issues
    first["error_count"] = sum(
        issue.get("severity") == "error" for issue in issues
    )
    first["warning_count"] = sum(
        issue.get("severity") == "warning" for issue in issues
    )
    restore_source_provenance(first, source_context)
    return first


def nucleic_acid_structure_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    """Run DSSR by replica, then merge identity-preserving observations."""

    project = load_json(Path(project_path).expanduser().resolve(strict=False))
    settings = _settings(project)
    selection = settings.get("frame_selection")
    if isinstance(selection, dict) and selection.get("mode") == "auto_resource_budget_v1":
        return _nucleic_acid_structure_project_serial(
            project_path, hash_content=hash_content
        )
    return execute_replica_final_module(
        project_path,
        runner_id="nucleic_acid_structure",
        hash_content=hash_content,
        reducer=_reduce_nucleic_acid_structure_reports,
    )


def nucleic_acid_structure_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return nucleic_acid_structure_project(project_path, hash_content=hash_content)
    except (
        AtomMappingError,
        CoordinateReadError,
        ManifestValidationError,
        NucleicAcidStructureError,
        PeriodicReconstructionError,
        TrajectoryContractError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "nucleic_acid_structure",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(
                Path(project_path).expanduser().resolve(strict=False)
            ),
            "error_count": len(messages),
            "warning_count": 0,
            "issues": [
                {
                    "severity": "error",
                    "code": "NUCLEIC_ACID_STRUCTURE_INVALID",
                    "message": message,
                }
                for message in messages
            ],
        }
