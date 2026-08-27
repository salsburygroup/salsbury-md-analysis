"""Compile explicit project settings and system identities into one stable context."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .manifests import (
    ManifestValidationError,
    inventory_system_inputs,
    inventory_content_signature_sha256,
    load_json,
    resolve_manifest_path,
    sha256_file,
    validate_project,
    validate_system,
)
from .provenance import stable_json_sha256
from .trajectory_contracts import normalize_segment_axis, require_periodic_policy


_TIME_IN_PS = {"fs": 0.001, "ps": 1.0, "ns": 1000.0, "us": 1_000_000.0}


def _normalized_selections(raw: Mapping[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for selection_id in sorted(raw):
        selection = raw[selection_id]
        assert isinstance(selection, dict)
        if "preset" in selection:
            normalized[selection_id] = {"preset": str(selection["preset"])}
        elif "atom_names" in selection:
            names = selection["atom_names"]
            assert isinstance(names, list)
            normalized[selection_id] = {
                "atom_names": sorted(str(name).strip() for name in names)
            }
        else:
            residue_keys = selection["residue_keys"]
            heavy_only = selection["heavy_only"]
            assert isinstance(residue_keys, list)
            assert isinstance(heavy_only, bool)
            normalized[selection_id] = {
                "residue_keys": sorted(
                    ({
                        "chain_id": str(key["chain_id"]),
                        "residue_number": int(key["residue_number"]),
                        "insertion_code": str(key["insertion_code"]),
                    } for key in residue_keys if isinstance(key, dict)),
                    key=lambda key: (
                        key["chain_id"], key["residue_number"], key["insertion_code"]
                    ),
                ),
                "heavy_only": heavy_only,
            }
    return normalized


def _normalized_interval(
    raw: Optional[object], output_unit: str
) -> Optional[Dict[str, object]]:
    if raw is None:
        return None
    assert isinstance(raw, dict)
    source_unit = str(raw["unit"])
    scale = _TIME_IN_PS[source_unit] / _TIME_IN_PS[output_unit]
    return {
        "start": float(raw["start"]) * scale,
        "end": float(raw["end"]) * scale,
        "unit": output_unit,
        "declared_unit": source_unit,
    }


def _system_identity(
    data: Mapping[str, object], output_time_unit: Optional[str]
) -> List[Dict[str, object]]:
    systems_out: List[Dict[str, object]] = []
    systems = data["systems"]
    assert isinstance(systems, list)
    for system in systems:
        assert isinstance(system, dict)
        replicas_out: List[Dict[str, object]] = []
        replicas = system["replicas"]
        assert isinstance(replicas, list)
        for replica in replicas:
            assert isinstance(replica, dict)
            segments_out: List[Dict[str, object]] = []
            segments = replica["segments"]
            assert isinstance(segments, list)
            for segment in segments:
                assert isinstance(segment, dict)
                axis = normalize_segment_axis(segment, output_time_unit)
                segment_out = {
                    "segment_id": str(segment["segment_id"]),
                    "trajectory": str(segment["trajectory"]),
                    "continuous_with_previous": bool(
                        segment.get("continuous_with_previous", False)
                    ),
                    "frame_axis_kind": str(axis["kind"]),
                    "weights": (
                        str(segment["weights"])
                        if segment.get("weights") is not None
                        else None
                    ),
                }
                if axis["kind"] == "physical_time":
                    segment_out["timing"] = axis["timing"]
                else:
                    segment_out["sample_axis"] = axis["sample_axis"]
                segments_out.append(segment_out)
            force_field_parameters = replica.get("force_field_parameters")
            normalized_force_field_parameters = None
            if isinstance(force_field_parameters, dict):
                files = force_field_parameters.get("files", [])
                assert isinstance(files, list)
                normalized_force_field_parameters = {
                    "format": str(force_field_parameters["format"]),
                    "files": [str(path) for path in files],
                }
            replicas_out.append({
                "replica_id": str(replica["replica_id"]),
                "topology": str(replica["topology"]),
                "connectivity": (
                    str(replica["connectivity"])
                    if replica.get("connectivity") is not None
                    else None
                ),
                "force_field_parameters": normalized_force_field_parameters,
                "segments": segments_out,
            })
        systems_out.append({
            "system_id": str(system["system_id"]),
            "replicas": replicas_out,
        })
    return systems_out


def compile_project_context(
    project: Mapping[str, object],
    source_path: Path,
    hash_content: bool = False,
) -> Dict[str, object]:
    """Return a deterministic read-only analysis context or fail closed."""

    project_path = Path(source_path).expanduser().resolve(strict=False)
    validate_project(project, source_path=project_path, check_paths=True)
    issues: List[str] = []

    coordinate_unit = project.get("coordinate_unit")
    time_unit = project.get("time_unit")
    sampling_mode = str(project.get("sampling_mode"))
    selections = project.get("selections")
    periodic_coordinate_policy = project.get("periodic_coordinate_policy")
    if coordinate_unit is None:
        issues.append("coordinate_unit is required to compile an analysis context")
    if time_unit is None and sampling_mode != "AI_ENSEMBLE":
        issues.append("time_unit is required to compile an analysis context")
    if sampling_mode == "AI_ENSEMBLE" and project.get("production_interval") is not None:
        issues.append(
            "production_interval is not valid for AI_ENSEMBLE because sample order is not physical time"
        )
    if selections is None:
        issues.append("selections is required to compile an analysis context")
    elif isinstance(selections, dict):
        required = {"alignment", "analysis"}
        requested = project.get("requested_modules", [])
        if isinstance(requested, list) and "common_atom_mapping" in requested:
            required.add("mapping")
        missing = sorted(required.difference(selections))
        if missing:
            issues.append(
                "selections is missing required semantic roles: " + ", ".join(missing)
            )
    try:
        require_periodic_policy(periodic_coordinate_policy)
    except ValueError as exc:
        issues.append(str(exc))
    if issues:
        raise ManifestValidationError(issues)

    assert isinstance(coordinate_unit, str)
    assert isinstance(selections, dict)
    system_text = str(project["system_manifest"])
    system_path = resolve_manifest_path(system_text, project_path)
    system = load_json(system_path)
    validate_system(system, source_path=system_path, check_paths=True)

    axis_issues: List[str] = []
    for raw_system in system["systems"]:
        assert isinstance(raw_system, dict)
        for raw_replica in raw_system["replicas"]:
            assert isinstance(raw_replica, dict)
            if (
                periodic_coordinate_policy in {"make_whole", "unwrap_continuous"}
                and raw_replica.get("connectivity") is None
            ):
                axis_issues.append(
                    f"{raw_system['system_id']}/{raw_replica['replica_id']} requires an "
                    f"explicit connectivity path for periodic_coordinate_policy="
                    f"{periodic_coordinate_policy}"
                )
            for raw_segment in raw_replica["segments"]:
                assert isinstance(raw_segment, dict)
                location = (
                    f"{raw_system['system_id']}/{raw_replica['replica_id']}/"
                    f"{raw_segment['segment_id']}"
                )
                if sampling_mode == "AI_ENSEMBLE" and raw_segment.get("sample_axis") is None:
                    axis_issues.append(
                        f"{location} must use sample_axis for AI_ENSEMBLE; physical timing must not be invented"
                    )
                if sampling_mode != "AI_ENSEMBLE" and raw_segment.get("timing") is None:
                    axis_issues.append(
                        f"{location} must use physical timing for sampling_mode {sampling_mode}"
                    )
    if axis_issues:
        raise ManifestValidationError(axis_issues)

    system_ids = [str(entry["system_id"]) for entry in system["systems"]]
    reference_system = project.get("reference_system")
    warnings: List[Dict[str, object]] = []
    if reference_system is None:
        if len(system_ids) != 1:
            raise ManifestValidationError((
                "reference_system is required when the system manifest contains multiple systems",
            ))
        reference_system = system_ids[0]
        warnings.append({
            "severity": "warning",
            "code": "REFERENCE_SYSTEM_INFERRED",
            "message": "reference_system was inferred because the manifest contains one system",
        })
    elif reference_system not in system_ids:
        raise ManifestValidationError((
            f"reference_system {reference_system!r} is not present in the system manifest",
        ))

    requested_modules = list(project.get("requested_modules", []))
    contract = {
        "contract_version": 3,
        "project_id": str(project["project_id"]),
        "profile_id": str(project["analysis_profile"]),
        "sampling_mode": sampling_mode,
        "periodic_coordinate_policy": str(periodic_coordinate_policy),
        "periodic_reconstruction": project.get("periodic_reconstruction"),
        "preprocessed_coordinate_source": project.get(
            "preprocessed_coordinate_source"
        ),
        "reference_connectivity": project.get("reference_connectivity"),
        "reference_system": str(reference_system),
        "requested_modules": requested_modules,
        "units": {
            "coordinates": coordinate_unit,
            "time": time_unit if isinstance(time_unit, str) else None,
            "temperature": "kelvin",
        },
        "temperature_kelvin": project.get("temperature_kelvin"),
        "production_interval": (
            _normalized_interval(project.get("production_interval"), time_unit)
            if isinstance(time_unit, str)
            else None
        ),
        "analysis_stride": (
            {"value": project["analysis_stride"], "unit": "frames"}
            if isinstance(project.get("analysis_stride"), int)
            else (
                {"expression": str(project["analysis_stride"])}
                if project.get("analysis_stride") is not None
                else None
            )
        ),
        "selections": _normalized_selections(selections),
        "systems": _system_identity(
            system, time_unit if isinstance(time_unit, str) else None
        ),
    }
    inventory = inventory_system_inputs(
        system, source_path=system_path, hash_content=hash_content
    )
    content_signature = None
    if hash_content:
        content_signature = inventory_content_signature_sha256(inventory)

    return {
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "project_manifest_path": str(project_path),
        "project_manifest_sha256": sha256_file(project_path),
        "system_manifest_path": str(system_path),
        "system_manifest_sha256": sha256_file(system_path),
        "contract": contract,
        "contract_signature_sha256": stable_json_sha256(contract),
        "input_content_signature_sha256": content_signature,
        "content_hashes_included": hash_content,
        "input_inventory": inventory,
        "issues": warnings,
        "error_count": 0,
        "warning_count": len(warnings),
        "limitations": [
            "The context records declared identities, physical-time or sample-index frame axes, units, selections, paths, and periodic-coordinate policy; it does not establish scientific validity.",
            "AI_ENSEMBLE sample indices identify ensemble members only and are not physical time.",
            "Named atom selections are exact atom-name sets, not backend-specific selection expressions.",
            "A string analysis_stride is retained as an unevaluated expression and must be interpreted by a later execution backend.",
        ],
    }


def compile_project_context_file(path: Path, hash_content: bool = False) -> Dict[str, object]:
    """Load and compile one project manifest."""

    source = Path(path)
    return compile_project_context(load_json(source), source, hash_content=hash_content)
