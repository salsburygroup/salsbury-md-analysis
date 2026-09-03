"""Fail-closed validation and deterministic inventories for analysis manifests.

This module intentionally uses only the Python standard library.  It validates
the semantic invariants that matter during routine analysis and publication locking;
JSON Schema files remain the machine-readable interchange specification.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .provenance import stable_json_sha256
from .registry import get_module


MANIFEST_KINDS = ("project", "system", "lock", "output", "regression")
COORDINATE_UNITS = ("angstrom", "nanometer")
TIME_UNITS = ("fs", "ps", "ns", "us")
PERIODIC_COORDINATE_POLICIES = (
    "reject", "allow_wrapped_diagnostic", "make_whole", "unwrap_continuous",
    "preprocessed_make_whole",
)
SELECTION_PRESETS = (
    "all", "backbone", "complex_trace", "heavy", "macromolecular_backbone",
    "molecular_payload", "solute_heavy",
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_SELECTION_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
FORCE_FIELD_PARAMETER_FORMATS = (
    "charmm_parameter_files_v1", "openmm_system_xml_v1", "gromacs_tpr_v1",
)


class DuplicateKeyError(ValueError):
    """Raised when JSON text contains a duplicate object key."""


class ManifestValidationError(ValueError):
    """Raised with every detected semantic validation issue."""

    def __init__(self, issues: Iterable[str]):
        ordered = tuple(str(issue) for issue in issues)
        if not ordered:
            ordered = ("manifest validation failed",)
        self.issues = ordered
        super().__init__("; ".join(ordered))


def _unique_object(pairs: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Dict[str, object]:
    """Load one JSON object while rejecting duplicate keys."""

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError, DuplicateKeyError) as exc:
        raise ManifestValidationError((f"{source}: {exc}",)) from exc
    if not isinstance(payload, dict):
        raise ManifestValidationError((f"{source}: top-level JSON value must be an object",))
    return payload


def _require_object(value: object, label: str, issues: List[str]) -> Optional[Mapping[str, object]]:
    if not isinstance(value, dict):
        issues.append(f"{label} must be an object")
        return None
    return value


def _require_string(value: object, label: str, issues: List[str]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{label} must be a nonempty string")
        return None
    return value


def _require_list(value: object, label: str, issues: List[str]) -> Optional[List[object]]:
    if not isinstance(value, list):
        issues.append(f"{label} must be an array")
        return None
    return value


def _required(data: Mapping[str, object], names: Iterable[str], issues: List[str]) -> None:
    for name in names:
        if name not in data:
            issues.append(f"missing required field: {name}")


def _unknown_fields(
    data: Mapping[str, object], allowed: Iterable[str], label: str, issues: List[str]
) -> None:
    unknown = sorted(set(data).difference(allowed))
    if unknown:
        issues.append(f"{label} contains unknown fields: {', '.join(unknown)}")


def _validate_force_field_parameters(
    value: object, label: str, issues: List[str],
    source_path: Optional[Path], check_paths: bool,
) -> None:
    parameters = _require_object(value, label, issues)
    if parameters is None:
        return
    _unknown_fields(parameters, {"format", "files"}, label, issues)
    _required(parameters, {"format", "files"}, issues)
    format_name = _require_string(parameters.get("format"), f"{label}.format", issues)
    if format_name is not None and format_name not in FORCE_FIELD_PARAMETER_FORMATS:
        issues.append(
            f"{label}.format must be one of: "
            + ", ".join(FORCE_FIELD_PARAMETER_FORMATS)
        )
    files = _require_list(parameters.get("files"), f"{label}.files", issues)
    if files is None:
        return
    if not files:
        issues.append(f"{label}.files must contain at least one file")
    normalized: List[str] = []
    for index, path in enumerate(files):
        _validate_file_path(
            path, f"{label}.files[{index}]", issues, source_path, check_paths,
        )
        if isinstance(path, str):
            normalized.append(path)
    if len(normalized) != len(set(normalized)):
        issues.append(f"{label}.files must not contain duplicates")
    if format_name in {"openmm_system_xml_v1", "gromacs_tpr_v1"} and len(files) != 1:
        issues.append(f"{label}.format={format_name} requires exactly one file")


def _duplicates(values: Iterable[str]) -> List[str]:
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _base_directory(source_path: Optional[Path]) -> Path:
    if source_path is None:
        return Path.cwd().resolve()
    return Path(source_path).expanduser().resolve(strict=False).parent


def resolve_manifest_path(value: str, source_path: Optional[Path]) -> Path:
    """Resolve a manifest path relative to the manifest, never the caller."""

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = _base_directory(source_path) / candidate
    return candidate.resolve(strict=False)


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_file_path(
    value: object,
    label: str,
    issues: List[str],
    source_path: Optional[Path],
    check_paths: bool,
) -> Optional[Path]:
    text = _require_string(value, label, issues)
    if text is None:
        return None
    resolved = resolve_manifest_path(text, source_path)
    if check_paths:
        if not resolved.exists():
            issues.append(f"{label} does not exist: {resolved}")
        elif not resolved.is_file():
            issues.append(f"{label} is not a regular file: {resolved}")
    return resolved


def _validate_string_array(value: object, label: str, issues: List[str]) -> None:
    values = _require_list(value, label, issues)
    if values is None:
        return
    for index, item in enumerate(values):
        _require_string(item, f"{label}[{index}]", issues)


def _validate_selections(value: object, issues: List[str]) -> None:
    selections = _require_object(value, "selections", issues)
    if selections is None:
        return
    if not selections:
        issues.append("selections must contain at least one named selection")
    for selection_id, raw_selection in selections.items():
        prefix = f"selections.{selection_id}"
        if not _SELECTION_ID.fullmatch(selection_id):
            issues.append(
                f"selection id {selection_id!r} must start with a letter and contain only "
                "letters, digits, underscores, or hyphens"
            )
        selection = _require_object(raw_selection, prefix, issues)
        if selection is None:
            continue
        _unknown_fields(
            selection,
            {"preset", "atom_names", "residue_keys", "heavy_only"},
            prefix,
            issues,
        )
        modes = [
            "preset" in selection,
            "atom_names" in selection,
            "residue_keys" in selection or "heavy_only" in selection,
        ]
        if sum(modes) != 1:
            issues.append(
                f"{prefix} must define exactly one of preset, atom_names, or "
                "residue_keys with heavy_only"
            )
            continue
        if "preset" in selection:
            if selection["preset"] not in SELECTION_PRESETS:
                issues.append(
                    f"{prefix}.preset must be one of: {', '.join(SELECTION_PRESETS)}"
                )
            continue
        if "residue_keys" in selection or "heavy_only" in selection:
            if set(selection) != {"residue_keys", "heavy_only"}:
                issues.append(
                    f"{prefix} residue selection requires exactly residue_keys and heavy_only"
                )
                continue
            if not isinstance(selection["heavy_only"], bool):
                issues.append(f"{prefix}.heavy_only must be boolean")
            keys = _require_list(selection["residue_keys"], f"{prefix}.residue_keys", issues)
            if keys is None:
                continue
            if not keys:
                issues.append(f"{prefix}.residue_keys must contain at least one residue")
            normalized_keys = []
            for index, raw_key in enumerate(keys):
                label = f"{prefix}.residue_keys[{index}]"
                key = _require_object(raw_key, label, issues)
                if key is None:
                    continue
                _unknown_fields(
                    key, {"chain_id", "residue_number", "insertion_code"}, label, issues
                )
                _required(key, {"chain_id", "residue_number", "insertion_code"}, issues)
                chain_id = key.get("chain_id")
                residue_number = key.get("residue_number")
                insertion_code = key.get("insertion_code")
                if not isinstance(chain_id, str):
                    issues.append(f"{label}.chain_id must be a string")
                if (
                    isinstance(residue_number, bool)
                    or not isinstance(residue_number, int)
                ):
                    issues.append(f"{label}.residue_number must be an integer")
                if not isinstance(insertion_code, str):
                    issues.append(f"{label}.insertion_code must be a string")
                if (
                    isinstance(chain_id, str)
                    and isinstance(residue_number, int)
                    and not isinstance(residue_number, bool)
                    and isinstance(insertion_code, str)
                ):
                    normalized_keys.append((chain_id, residue_number, insertion_code))
            duplicates = _duplicates(str(value) for value in normalized_keys)
            if duplicates:
                issues.append(f"{prefix}.residue_keys contains duplicate residue identities")
            continue
        names = _require_list(selection["atom_names"], f"{prefix}.atom_names", issues)
        if names is None:
            continue
        if not names:
            issues.append(f"{prefix}.atom_names must contain at least one atom name")
        normalized: List[str] = []
        for index, name in enumerate(names):
            text = _require_string(name, f"{prefix}.atom_names[{index}]", issues)
            if text is not None:
                normalized.append(text.strip())
        duplicates = _duplicates(normalized)
        if duplicates:
            issues.append(
                f"{prefix}.atom_names contains duplicates: {', '.join(duplicates)}"
            )


def validate_project(
    data: Mapping[str, object], source_path: Optional[Path] = None, check_paths: bool = False
) -> None:
    issues: List[str] = []
    allowed = {
        "project_id", "analysis_profile", "system_manifest", "analysis_output_root",
        "reference_system", "production_interval", "analysis_stride",
        "temperature_kelvin", "sampling_mode", "statistical_weights",
        "reference_structure", "common_atom_policy", "definitions",
        "requested_modules", "compute_environment", "protected_locations",
        "coordinate_unit", "time_unit", "selections",
        "periodic_coordinate_policy",
        "periodic_reconstruction", "reference_connectivity",
        "preprocessed_coordinate_source",
    }
    required = {
        "project_id", "analysis_profile", "system_manifest", "analysis_output_root",
        "sampling_mode", "protected_locations",
    }
    _unknown_fields(data, allowed, "project manifest", issues)
    _required(data, required, issues)
    _require_string(data.get("project_id"), "project_id", issues)
    _require_string(data.get("analysis_profile"), "analysis_profile", issues)
    _validate_file_path(data.get("system_manifest"), "system_manifest", issues, source_path, check_paths)

    output_text = _require_string(data.get("analysis_output_root"), "analysis_output_root", issues)
    output_root = (
        resolve_manifest_path(output_text, source_path) if output_text is not None else None
    )
    protected = _require_list(data.get("protected_locations"), "protected_locations", issues)
    protected_paths: List[Path] = []
    if protected is not None:
        protected_texts: List[str] = []
        for index, value in enumerate(protected):
            label = f"protected_locations[{index}]"
            text = _require_string(value, label, issues)
            if text is None:
                continue
            path = Path(text).expanduser()
            if not path.is_absolute():
                issues.append(f"{label} must be an absolute path")
                continue
            protected_texts.append(str(path.resolve(strict=False)))
            protected_paths.append(path.resolve(strict=False))
        duplicate_protected = _duplicates(protected_texts)
        if duplicate_protected:
            issues.append(
                f"protected_locations contains duplicates: {', '.join(duplicate_protected)}"
            )
    if output_root is not None:
        for protected_path in protected_paths:
            if _overlap(output_root, protected_path):
                issues.append(
                    "analysis_output_root overlaps protected location: "
                    f"{output_root} and {protected_path}"
                )

    mode = data.get("sampling_mode")
    valid_modes = {"UNBIASED_MD", "BIASED_MD", "ENHANCED_SAMPLING", "AI_ENSEMBLE"}
    if mode not in valid_modes:
        issues.append(f"sampling_mode must be one of: {', '.join(sorted(valid_modes))}")

    interval = data.get("production_interval")
    if interval is not None:
        interval_object = _require_object(interval, "production_interval", issues)
        if interval_object is not None:
            _unknown_fields(interval_object, {"start", "end", "unit"}, "production_interval", issues)
            _required(interval_object, {"start", "end", "unit"}, issues)
            start = interval_object.get("start")
            end = interval_object.get("end")
            if not isinstance(start, (int, float)) or isinstance(start, bool):
                issues.append("production_interval.start must be numeric")
            if not isinstance(end, (int, float)) or isinstance(end, bool):
                issues.append("production_interval.end must be numeric")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end <= start:
                issues.append("production_interval.end must be greater than start")
            if interval_object.get("unit") not in set(TIME_UNITS):
                issues.append("production_interval.unit must be fs, ps, ns, or us")

    stride = data.get("analysis_stride")
    if stride is not None:
        valid_stride = (
            isinstance(stride, int) and not isinstance(stride, bool) and stride > 0
        ) or (isinstance(stride, str) and bool(stride.strip()))
        if not valid_stride:
            issues.append("analysis_stride must be a positive integer or nonempty string")

    temperature = data.get("temperature_kelvin")
    if temperature is not None and (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or temperature <= 0
    ):
        issues.append("temperature_kelvin must be a positive number")

    coordinate_unit = data.get("coordinate_unit")
    if coordinate_unit is not None and coordinate_unit not in COORDINATE_UNITS:
        issues.append(
            f"coordinate_unit must be one of: {', '.join(COORDINATE_UNITS)}"
        )
    time_unit = data.get("time_unit")
    if time_unit is not None and time_unit not in TIME_UNITS:
        issues.append(f"time_unit must be one of: {', '.join(TIME_UNITS)}")
    periodic_policy = data.get("periodic_coordinate_policy")
    if (
        periodic_policy is not None
        and periodic_policy not in PERIODIC_COORDINATE_POLICIES
    ):
        issues.append(
            "periodic_coordinate_policy must be reject, allow_wrapped_diagnostic, "
            "make_whole, unwrap_continuous, or preprocessed_make_whole"
        )
    preprocessed = data.get("preprocessed_coordinate_source")
    if preprocessed is not None:
        preprocessed_object = _require_object(
            preprocessed, "preprocessed_coordinate_source", issues
        )
        if preprocessed_object is not None:
            _unknown_fields(
                preprocessed_object,
                {"cache_report", "cache_report_sha256"},
                "preprocessed_coordinate_source",
                issues,
            )
            _required(
                preprocessed_object,
                {"cache_report", "cache_report_sha256"},
                issues,
            )
            report_path = _validate_file_path(
                preprocessed_object.get("cache_report"),
                "preprocessed_coordinate_source.cache_report",
                issues,
                source_path,
                check_paths,
            )
            digest = preprocessed_object.get("cache_report_sha256")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                issues.append(
                    "preprocessed_coordinate_source.cache_report_sha256 must be a SHA-256 hex digest"
                )
            elif check_paths and report_path is not None and report_path.is_file():
                try:
                    actual = sha256_file(report_path)
                except OSError as exc:
                    issues.append(str(exc))
                else:
                    if actual.lower() != digest.lower():
                        issues.append(
                            "preprocessed_coordinate_source.cache_report_sha256 does not match cache_report"
                        )
    if periodic_policy == "preprocessed_make_whole" and preprocessed is None:
        issues.append(
            "preprocessed_coordinate_source is required for preprocessed_make_whole"
        )
    if periodic_policy != "preprocessed_make_whole" and preprocessed is not None:
        issues.append(
            "preprocessed_coordinate_source is only valid for preprocessed_make_whole"
        )
    reconstruction = data.get("periodic_reconstruction")
    if reconstruction is not None:
        reconstruction_object = _require_object(
            reconstruction, "periodic_reconstruction", issues
        )
        if reconstruction_object is not None:
            allowed_reconstruction = {
                "maximum_bond_length_angstrom",
                "cycle_closure_tolerance_angstrom",
                "maximum_anchor_displacement_angstrom",
            }
            _unknown_fields(
                reconstruction_object,
                allowed_reconstruction,
                "periodic_reconstruction",
                issues,
            )
            required_reconstruction = {
                "maximum_bond_length_angstrom",
                "cycle_closure_tolerance_angstrom",
            }
            if periodic_policy == "unwrap_continuous":
                required_reconstruction.add("maximum_anchor_displacement_angstrom")
            if periodic_policy in {"make_whole", "unwrap_continuous"}:
                _required(reconstruction_object, required_reconstruction, issues)
            for field in allowed_reconstruction:
                value = reconstruction_object.get(field)
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                ):
                    issues.append(f"periodic_reconstruction.{field} must be finite and positive")
    elif periodic_policy in {"make_whole", "unwrap_continuous"}:
        issues.append(
            "periodic_reconstruction is required for make_whole or unwrap_continuous"
        )
    if data.get("selections") is not None:
        _validate_selections(data["selections"], issues)

    for field in ("statistical_weights", "reference_structure", "reference_connectivity"):
        if data.get(field) is not None:
            _validate_file_path(data[field], field, issues, source_path, check_paths)
    for field in ("reference_system", "common_atom_policy", "compute_environment"):
        if data.get(field) is not None:
            _require_string(data[field], field, issues)
    if data.get("definitions") is not None and not isinstance(data["definitions"], dict):
        issues.append("definitions must be an object")

    requested = data.get("requested_modules")
    if requested is not None:
        modules = _require_list(requested, "requested_modules", issues)
        module_ids: List[str] = []
        if modules is not None:
            for index, value in enumerate(modules):
                module_id = _require_string(value, f"requested_modules[{index}]", issues)
                if module_id is None:
                    continue
                module_ids.append(module_id)
                try:
                    get_module(module_id)
                except KeyError:
                    issues.append(f"requested_modules[{index}] is unknown: {module_id}")
            duplicates = _duplicates(module_ids)
            if duplicates:
                issues.append(f"requested_modules contains duplicates: {', '.join(duplicates)}")

    if issues:
        raise ManifestValidationError(issues)


def validate_system(
    data: Mapping[str, object], source_path: Optional[Path] = None, check_paths: bool = False
) -> None:
    issues: List[str] = []
    _unknown_fields(data, {"systems"}, "system manifest", issues)
    _required(data, {"systems"}, issues)
    systems = _require_list(data.get("systems"), "systems", issues)
    system_ids: List[str] = []
    if systems is not None:
        if not systems:
            issues.append("systems must contain at least one system")
        for system_index, raw_system in enumerate(systems):
            prefix = f"systems[{system_index}]"
            system = _require_object(raw_system, prefix, issues)
            if system is None:
                continue
            _unknown_fields(system, {"system_id", "metadata", "replicas"}, prefix, issues)
            _required(system, {"system_id", "replicas"}, issues)
            system_id = _require_string(system.get("system_id"), f"{prefix}.system_id", issues)
            if system_id is not None:
                system_ids.append(system_id)
            metadata = system.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                issues.append(f"{prefix}.metadata must be an object")
            replicas = _require_list(system.get("replicas"), f"{prefix}.replicas", issues)
            replica_ids: List[str] = []
            if replicas is None:
                continue
            if not replicas:
                issues.append(f"{prefix}.replicas must contain at least one replica")
            for replica_index, raw_replica in enumerate(replicas):
                replica_prefix = f"{prefix}.replicas[{replica_index}]"
                replica = _require_object(raw_replica, replica_prefix, issues)
                if replica is None:
                    continue
                _unknown_fields(
                    replica, {
                        "replica_id", "topology", "connectivity",
                        "force_field_parameters", "segments",
                    }, replica_prefix, issues
                )
                _required(replica, {"replica_id", "topology", "segments"}, issues)
                replica_id = _require_string(
                    replica.get("replica_id"), f"{replica_prefix}.replica_id", issues
                )
                if replica_id is not None:
                    replica_ids.append(replica_id)
                _validate_file_path(
                    replica.get("topology"), f"{replica_prefix}.topology", issues,
                    source_path, check_paths,
                )
                if replica.get("connectivity") is not None:
                    _validate_file_path(
                        replica.get("connectivity"),
                        f"{replica_prefix}.connectivity",
                        issues,
                        source_path,
                        check_paths,
                    )
                if replica.get("force_field_parameters") is not None:
                    _validate_force_field_parameters(
                        replica.get("force_field_parameters"),
                        f"{replica_prefix}.force_field_parameters",
                        issues, source_path, check_paths,
                    )
                segments = _require_list(
                    replica.get("segments"), f"{replica_prefix}.segments", issues
                )
                segment_ids: List[str] = []
                if segments is None:
                    continue
                if not segments:
                    issues.append(f"{replica_prefix}.segments must contain at least one segment")
                for segment_index, raw_segment in enumerate(segments):
                    segment_prefix = f"{replica_prefix}.segments[{segment_index}]"
                    segment = _require_object(raw_segment, segment_prefix, issues)
                    if segment is None:
                        continue
                    _unknown_fields(
                        segment,
                        {
                            "segment_id", "trajectory", "continuous_with_previous",
                            "dcd_header_step_policy", "weights", "timing", "sample_axis",
                        },
                        segment_prefix,
                        issues,
                    )
                    _required(segment, {"segment_id", "trajectory"}, issues)
                    segment_id = _require_string(
                        segment.get("segment_id"), f"{segment_prefix}.segment_id", issues
                    )
                    if segment_id is not None:
                        segment_ids.append(segment_id)
                    _validate_file_path(
                        segment.get("trajectory"), f"{segment_prefix}.trajectory", issues,
                        source_path, check_paths,
                    )
                    if segment.get("weights") is not None:
                        _validate_file_path(
                            segment.get("weights"), f"{segment_prefix}.weights", issues,
                            source_path, check_paths,
                        )
                    continuous = segment.get("continuous_with_previous", False)
                    if not isinstance(continuous, bool):
                        issues.append(f"{segment_prefix}.continuous_with_previous must be boolean")
                    elif segment_index == 0 and continuous:
                        issues.append(
                            f"{segment_prefix}.continuous_with_previous cannot be true for the first segment"
                        )
                    dcd_header_step_policy = segment.get(
                        "dcd_header_step_policy", "continuous"
                    )
                    if dcd_header_step_policy not in {
                        "continuous", "reset_per_segment"
                    }:
                        issues.append(
                            f"{segment_prefix}.dcd_header_step_policy must be "
                            "continuous or reset_per_segment"
                        )
                    has_timing = segment.get("timing") is not None
                    has_sample_axis = segment.get("sample_axis") is not None
                    if has_timing == has_sample_axis:
                        issues.append(
                            f"{segment_prefix} must declare exactly one of timing or sample_axis"
                        )
                    timing = None
                    if has_timing:
                        timing = _require_object(
                            segment.get("timing"), f"{segment_prefix}.timing", issues
                        )
                    if timing is not None:
                        _unknown_fields(
                            timing,
                            {"first_frame_time", "frame_interval", "unit"},
                            f"{segment_prefix}.timing",
                            issues,
                        )
                        _required(
                            timing,
                            {"first_frame_time", "frame_interval", "unit"},
                            issues,
                        )
                        first = timing.get("first_frame_time")
                        interval = timing.get("frame_interval")
                        if (
                            isinstance(first, bool)
                            or not isinstance(first, (int, float))
                            or not math.isfinite(float(first))
                        ):
                            issues.append(
                                f"{segment_prefix}.timing.first_frame_time must be a finite number"
                            )
                        if (
                            isinstance(interval, bool)
                            or not isinstance(interval, (int, float))
                            or not math.isfinite(float(interval))
                            or float(interval) <= 0.0
                        ):
                            issues.append(
                                f"{segment_prefix}.timing.frame_interval must be a finite positive number"
                            )
                        if timing.get("unit") not in TIME_UNITS:
                            issues.append(
                                f"{segment_prefix}.timing.unit must be fs, ps, ns, or us"
                            )
                    sample_axis = None
                    if has_sample_axis:
                        sample_axis = _require_object(
                            segment.get("sample_axis"),
                            f"{segment_prefix}.sample_axis",
                            issues,
                        )
                    if sample_axis is not None:
                        _unknown_fields(
                            sample_axis,
                            {"first_sample_index", "sample_interval"},
                            f"{segment_prefix}.sample_axis",
                            issues,
                        )
                        _required(
                            sample_axis,
                            {"first_sample_index", "sample_interval"},
                            issues,
                        )
                        first_sample = sample_axis.get("first_sample_index")
                        sample_interval = sample_axis.get("sample_interval")
                        if (
                            isinstance(first_sample, bool)
                            or not isinstance(first_sample, int)
                            or first_sample < 0
                        ):
                            issues.append(
                                f"{segment_prefix}.sample_axis.first_sample_index must be a nonnegative integer"
                            )
                        if (
                            isinstance(sample_interval, bool)
                            or not isinstance(sample_interval, int)
                            or sample_interval <= 0
                        ):
                            issues.append(
                                f"{segment_prefix}.sample_axis.sample_interval must be a positive integer"
                            )
                duplicate_segments = _duplicates(segment_ids)
                if duplicate_segments:
                    issues.append(
                        f"{replica_prefix} contains duplicate segment_id values: "
                        f"{', '.join(duplicate_segments)}"
                    )
            duplicate_replicas = _duplicates(replica_ids)
            if duplicate_replicas:
                issues.append(
                    f"{prefix} contains duplicate replica_id values: {', '.join(duplicate_replicas)}"
                )
    duplicate_systems = _duplicates(system_ids)
    if duplicate_systems:
        issues.append(f"duplicate system_id values: {', '.join(duplicate_systems)}")
    if issues:
        raise ManifestValidationError(issues)


def validate_lock(data: Mapping[str, object]) -> None:
    issues: List[str] = []
    allowed = {
        "project_id", "suite_repository", "suite_version", "suite_commit",
        "project_commit", "profile_id", "environment_identity",
        "input_manifest_sha256", "source_manifest_sha256", "output_manifest_sha256",
        "authoritative_data_roots", "external_dependencies", "commands",
        "random_seeds", "owner", "reviewers", "technical_status",
        "scientific_status", "limitations", "frame_budget_sensitivity",
        "replica_diagnostics",
    }
    required = {
        "project_id", "suite_commit", "project_commit", "profile_id",
        "environment_identity", "input_manifest_sha256", "source_manifest_sha256",
        "owner", "technical_status", "scientific_status",
    }
    _unknown_fields(data, allowed, "analysis lock", issues)
    _required(data, required, issues)
    for field in ("project_id", "profile_id", "environment_identity", "owner", "scientific_status"):
        _require_string(data.get(field), field, issues)
    for field in ("suite_commit", "project_commit"):
        value = data.get(field)
        if not isinstance(value, str) or not _GIT_COMMIT.fullmatch(value):
            issues.append(f"{field} must be a 40-character hexadecimal Git commit")
    for field in ("input_manifest_sha256", "source_manifest_sha256", "output_manifest_sha256"):
        value = data.get(field)
        if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
            issues.append(f"{field} must be a 64-character hexadecimal SHA-256")
    if data.get("technical_status") not in {"complete", "failed", "blocked", "skipped", "withdrawn"}:
        issues.append("technical_status is invalid")
    roots = data.get("authoritative_data_roots")
    if roots is not None:
        values = _require_list(roots, "authoritative_data_roots", issues)
        if values is not None:
            for index, value in enumerate(values):
                text = _require_string(value, f"authoritative_data_roots[{index}]", issues)
                if text is not None and not Path(text).expanduser().is_absolute():
                    issues.append(f"authoritative_data_roots[{index}] must be absolute")
    dependencies = data.get("external_dependencies")
    if dependencies is not None:
        values = _require_list(dependencies, "external_dependencies", issues)
        if values is not None:
            for index, value in enumerate(values):
                if not isinstance(value, dict):
                    issues.append(f"external_dependencies[{index}] must be an object")
    seeds = data.get("random_seeds")
    if seeds is not None:
        values = _require_list(seeds, "random_seeds", issues)
        if values is not None:
            for index, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, (int, str)):
                    issues.append(f"random_seeds[{index}] must be an integer or string")
    for field in ("commands", "reviewers", "limitations"):
        if data.get(field) is not None:
            _validate_string_array(data[field], field, issues)
    sensitivity = data.get("frame_budget_sensitivity")
    if sensitivity is not None:
        record = _require_object(
            sensitivity, "frame_budget_sensitivity", issues
        )
        if record is not None:
            _unknown_fields(
                record,
                {"policy", "status", "rationale", "evidence_report_sha256"},
                "frame_budget_sensitivity",
                issues,
            )
            _required(record, {"policy", "status"}, issues)
            if record.get("policy") not in {"off", "recommend", "require"}:
                issues.append(
                    "frame_budget_sensitivity.policy must be off, recommend, or require"
                )
            status = record.get("status")
            if status not in {
                "completed", "skipped", "unavailable", "not_applicable", "planned"
            }:
                issues.append("frame_budget_sensitivity.status is invalid")
            rationale = record.get("rationale")
            if rationale is not None:
                _require_string(rationale, "frame_budget_sensitivity.rationale", issues)
            if status in {"skipped", "unavailable"} and not (
                isinstance(rationale, str) and rationale.strip()
            ):
                issues.append(
                    "frame_budget_sensitivity.rationale is required when skipped or unavailable"
                )
            evidence = record.get("evidence_report_sha256")
            if evidence is not None:
                values = _require_list(
                    evidence, "frame_budget_sensitivity.evidence_report_sha256", issues
                )
                if values is not None:
                    for index, value in enumerate(values):
                        if not isinstance(value, str) or not _SHA256.fullmatch(value):
                            issues.append(
                                "frame_budget_sensitivity.evidence_report_sha256"
                                f"[{index}] must be a 64-character hexadecimal SHA-256"
                            )
            if status == "completed" and not (
                isinstance(evidence, list) and len(evidence) > 0
            ):
                issues.append(
                    "completed frame_budget_sensitivity requires evidence_report_sha256"
                )
            policy = record.get("policy")
            if policy == "require" and status in {"skipped", "not_applicable"}:
                issues.append(
                    "required frame_budget_sensitivity cannot be skipped or not_applicable"
                )
            if (
                policy == "require" and status == "unavailable"
                and data.get("technical_status") not in {"blocked", "failed"}
            ):
                issues.append(
                    "unavailable required frame_budget_sensitivity must block or fail the lock"
                )
    replica_diagnostics = data.get("replica_diagnostics")
    if replica_diagnostics is not None:
        record = _require_object(
            replica_diagnostics, "replica_diagnostics", issues
        )
        if record is not None:
            _unknown_fields(
                record,
                {
                    "policy", "status", "rationale",
                    "additional_replicas_may_be_useful",
                    "evidence_report_sha256",
                },
                "replica_diagnostics",
                issues,
            )
            _required(record, {"policy", "status"}, issues)
            if record.get("policy") not in {"off", "optional"}:
                issues.append("replica_diagnostics.policy must be off or optional")
            status = record.get("status")
            if status not in {
                "completed", "skipped", "unavailable", "not_applicable", "planned"
            }:
                issues.append("replica_diagnostics.status is invalid")
            rationale = record.get("rationale")
            if rationale is not None:
                _require_string(rationale, "replica_diagnostics.rationale", issues)
            if status in {"skipped", "unavailable"} and not (
                isinstance(rationale, str) and rationale.strip()
            ):
                issues.append(
                    "replica_diagnostics.rationale is required when skipped or unavailable"
                )
            additional = record.get("additional_replicas_may_be_useful")
            if additional is not None and not isinstance(additional, bool):
                issues.append(
                    "replica_diagnostics.additional_replicas_may_be_useful must be boolean"
                )
            evidence = record.get("evidence_report_sha256")
            if evidence is not None:
                values = _require_list(
                    evidence, "replica_diagnostics.evidence_report_sha256", issues
                )
                if values is not None:
                    for index, value in enumerate(values):
                        if not isinstance(value, str) or not _SHA256.fullmatch(value):
                            issues.append(
                                "replica_diagnostics.evidence_report_sha256"
                                f"[{index}] must be a 64-character hexadecimal SHA-256"
                            )
            if status == "completed" and not (
                isinstance(evidence, list) and len(evidence) > 0
            ):
                issues.append(
                    "completed replica_diagnostics requires evidence_report_sha256"
                )
    if issues:
        raise ManifestValidationError(issues)


def validate_output(
    data: Mapping[str, object], source_path: Optional[Path] = None, check_paths: bool = False
) -> None:
    issues: List[str] = []
    allowed = {
        "run_id", "suite_commit", "profile_id", "modules", "technical_status",
        "scientific_status", "limitations",
    }
    required = {"run_id", "suite_commit", "profile_id", "modules", "technical_status", "scientific_status"}
    _unknown_fields(data, allowed, "output manifest", issues)
    _required(data, required, issues)
    for field in ("run_id", "suite_commit", "profile_id", "scientific_status"):
        _require_string(data.get(field), field, issues)
    if data.get("technical_status") not in {"complete", "failed", "blocked", "partial"}:
        issues.append("technical_status is invalid")
    modules = _require_list(data.get("modules"), "modules", issues)
    module_ids: List[str] = []
    if modules is not None:
        for index, raw_module in enumerate(modules):
            prefix = f"modules[{index}]"
            module = _require_object(raw_module, prefix, issues)
            if module is None:
                continue
            _unknown_fields(module, {"module_id", "status", "outputs", "warnings"}, prefix, issues)
            _required(module, {"module_id", "status", "outputs"}, issues)
            module_id = _require_string(module.get("module_id"), f"{prefix}.module_id", issues)
            if module_id is not None:
                module_ids.append(module_id)
                try:
                    get_module(module_id)
                except KeyError:
                    issues.append(f"{prefix}.module_id is unknown: {module_id}")
            if module.get("status") not in {"complete", "failed", "blocked", "skipped"}:
                issues.append(f"{prefix}.status is invalid")
            outputs = _require_list(module.get("outputs"), f"{prefix}.outputs", issues)
            if outputs is not None:
                for output_index, raw_output in enumerate(outputs):
                    output_prefix = f"{prefix}.outputs[{output_index}]"
                    output = _require_object(raw_output, output_prefix, issues)
                    if output is None:
                        continue
                    _unknown_fields(output, {"path", "sha256"}, output_prefix, issues)
                    _required(output, {"path", "sha256"}, issues)
                    resolved = _validate_file_path(
                        output.get("path"), f"{output_prefix}.path", issues,
                        source_path, check_paths,
                    )
                    digest = output.get("sha256")
                    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                        issues.append(f"{output_prefix}.sha256 must be a hexadecimal SHA-256")
                    elif check_paths and resolved is not None and resolved.is_file():
                        actual = sha256_file(resolved)
                        if actual.lower() != digest.lower():
                            issues.append(
                                f"{output_prefix}.sha256 mismatch: expected {digest.lower()}, "
                                f"observed {actual}"
                            )
            if module.get("warnings") is not None:
                _validate_string_array(module["warnings"], f"{prefix}.warnings", issues)
        duplicates = _duplicates(module_ids)
        if duplicates:
            issues.append(f"modules contains duplicate module_id values: {', '.join(duplicates)}")
    if data.get("limitations") is not None:
        _validate_string_array(data["limitations"], "limitations", issues)
    if issues:
        raise ManifestValidationError(issues)


def validate_regression(
    data: Mapping[str, object],
    source_path: Optional[Path] = None,
    check_paths: bool = False,
) -> None:
    """Validate a hash-pinned regression case and its approval boundary."""

    issues: List[str] = []
    allowed = {
        "regression_id", "module_id", "project_manifest", "expected_identity",
        "assertions", "approval",
    }
    required = allowed
    _unknown_fields(data, allowed, "regression case", issues)
    _required(data, required, issues)
    _require_string(data.get("regression_id"), "regression_id", issues)
    module_id = _require_string(data.get("module_id"), "module_id", issues)
    if module_id is not None:
        try:
            get_module(module_id)
        except KeyError:
            issues.append(f"module_id is unknown: {module_id}")
        if module_id not in {
            "structural_integrity_qc", "replica_rmsd_rg", "pooled_rmsf", "dccm"
        }:
            issues.append(f"module_id has no regression runner: {module_id}")
    _validate_file_path(
        data.get("project_manifest"), "project_manifest", issues, source_path, check_paths
    )

    identity = _require_object(data.get("expected_identity"), "expected_identity", issues)
    identity_fields = {
        "project_manifest_sha256", "system_manifest_sha256",
        "input_content_signature_sha256",
    }
    if identity is not None:
        _unknown_fields(identity, identity_fields, "expected_identity", issues)
        _required(identity, identity_fields, issues)
        for field in sorted(identity_fields):
            value = identity.get(field)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                issues.append(f"expected_identity.{field} must be a hexadecimal SHA-256")

    assertions = _require_list(data.get("assertions"), "assertions", issues)
    if assertions is not None:
        if not assertions:
            issues.append("assertions must contain at least one assertion")
        for index, raw_assertion in enumerate(assertions):
            prefix = f"assertions[{index}]"
            assertion = _require_object(raw_assertion, prefix, issues)
            if assertion is None:
                continue
            _unknown_fields(
                assertion, {"path", "operator", "expected", "absolute_tolerance"},
                prefix, issues,
            )
            _required(assertion, {"path", "operator", "expected"}, issues)
            path = _require_list(assertion.get("path"), f"{prefix}.path", issues)
            if path is not None:
                if not path:
                    issues.append(f"{prefix}.path must contain at least one component")
                for component_index, component in enumerate(path):
                    if isinstance(component, bool) or not isinstance(component, (str, int)):
                        issues.append(
                            f"{prefix}.path[{component_index}] must be a string or integer"
                        )
            operator = assertion.get("operator")
            if operator not in {"equal", "close", "contains", "is_null"}:
                issues.append(f"{prefix}.operator is invalid")
            tolerance = assertion.get("absolute_tolerance")
            if operator == "close":
                if (
                    isinstance(tolerance, bool)
                    or not isinstance(tolerance, (int, float))
                    or not math.isfinite(float(tolerance))
                    or float(tolerance) < 0.0
                ):
                    issues.append(
                        f"{prefix}.absolute_tolerance must be a finite nonnegative number for close"
                    )
            elif tolerance is not None:
                issues.append(
                    f"{prefix}.absolute_tolerance is allowed only for close"
                )

    approval = _require_object(data.get("approval"), "approval", issues)
    if approval is not None:
        _unknown_fields(
            approval, {"status", "owner", "reviewers", "decision_utc", "notes"},
            "approval", issues,
        )
        _required(approval, {"status", "owner", "reviewers", "notes"}, issues)
        status = approval.get("status")
        if status not in {"candidate", "approved", "retired"}:
            issues.append("approval.status must be candidate, approved, or retired")
        _require_string(approval.get("owner"), "approval.owner", issues)
        reviewers = _require_list(approval.get("reviewers"), "approval.reviewers", issues)
        if reviewers is not None:
            for index, reviewer in enumerate(reviewers):
                _require_string(reviewer, f"approval.reviewers[{index}]", issues)
            if status == "approved" and not reviewers:
                issues.append("approval.reviewers must not be empty when status is approved")
        notes = approval.get("notes")
        if notes is not None:
            _validate_string_array(notes, "approval.notes", issues)
        decision = approval.get("decision_utc")
        if status == "approved" and (
            not isinstance(decision, str) or not decision.strip()
        ):
            issues.append("approval.decision_utc is required when status is approved")
        elif decision is not None and not isinstance(decision, str):
            issues.append("approval.decision_utc must be a string or null")
    if issues:
        raise ManifestValidationError(issues)


def validate_manifest(
    kind: str,
    data: Mapping[str, object],
    source_path: Optional[Path] = None,
    check_paths: bool = False,
) -> None:
    """Validate a manifest kind and raise all detected issues together."""

    if kind == "project":
        validate_project(data, source_path=source_path, check_paths=check_paths)
    elif kind == "system":
        validate_system(data, source_path=source_path, check_paths=check_paths)
    elif kind == "lock":
        validate_lock(data)
    elif kind == "output":
        validate_output(data, source_path=source_path, check_paths=check_paths)
    elif kind == "regression":
        validate_regression(data, source_path=source_path, check_paths=check_paths)
    else:
        raise ValueError(f"unknown manifest kind: {kind}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 of one regular file using bounded memory."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(
    role: str,
    original_path: str,
    resolved: Path,
    system_id: str,
    replica_id: str,
    segment_id: Optional[str],
    hash_content: bool,
) -> Dict[str, object]:
    stat = resolved.stat()
    return {
        "role": role,
        "system_id": system_id,
        "replica_id": replica_id,
        "segment_id": segment_id,
        "manifest_path": original_path,
        "resolved_path": str(resolved),
        "size_bytes": stat.st_size,
        "modified_utc": _datetime.datetime.fromtimestamp(
            stat.st_mtime, tz=_datetime.timezone.utc
        ).isoformat(),
        "sha256": sha256_file(resolved) if hash_content else None,
    }


def inventory_system_inputs(
    data: Mapping[str, object],
    source_path: Path,
    hash_content: bool = False,
) -> Dict[str, object]:
    """Build a deterministic, read-only file inventory for a system manifest."""

    manifest_path = Path(source_path).expanduser().resolve(strict=False)
    validate_system(data, source_path=manifest_path, check_paths=True)
    records: List[Dict[str, object]] = []
    systems = data["systems"]
    assert isinstance(systems, list)
    for system in systems:
        assert isinstance(system, dict)
        system_id = str(system["system_id"])
        replicas = system["replicas"]
        assert isinstance(replicas, list)
        for replica in replicas:
            assert isinstance(replica, dict)
            replica_id = str(replica["replica_id"])
            topology_text = str(replica["topology"])
            records.append(
                _file_record(
                    "topology", topology_text,
                    resolve_manifest_path(topology_text, manifest_path),
                    system_id, replica_id, None, hash_content,
                )
            )
            connectivity = replica.get("connectivity")
            if connectivity is not None:
                connectivity_text = str(connectivity)
                records.append(
                    _file_record(
                        "connectivity", connectivity_text,
                        resolve_manifest_path(connectivity_text, manifest_path),
                        system_id, replica_id, None, hash_content,
                    )
                )
            force_field_parameters = replica.get("force_field_parameters")
            if isinstance(force_field_parameters, dict):
                format_name = str(force_field_parameters.get("format", "unknown"))
                files = force_field_parameters.get("files", [])
                if isinstance(files, list):
                    for file_text_raw in files:
                        file_text = str(file_text_raw)
                        records.append(
                            _file_record(
                                f"force_field_parameter:{format_name}", file_text,
                                resolve_manifest_path(file_text, manifest_path),
                                system_id, replica_id, None, hash_content,
                            )
                        )
            segments = replica["segments"]
            assert isinstance(segments, list)
            for segment in segments:
                assert isinstance(segment, dict)
                segment_id = str(segment["segment_id"])
                trajectory_text = str(segment["trajectory"])
                records.append(
                    _file_record(
                        "trajectory", trajectory_text,
                        resolve_manifest_path(trajectory_text, manifest_path),
                        system_id, replica_id, segment_id, hash_content,
                    )
                )
                weights = segment.get("weights")
                if weights is not None:
                    weights_text = str(weights)
                    records.append(
                        _file_record(
                            "weights", weights_text,
                            resolve_manifest_path(weights_text, manifest_path),
                            system_id, replica_id, segment_id, hash_content,
                        )
                    )
    records.sort(
        key=lambda row: (
            str(row["system_id"]), str(row["replica_id"]),
            str(row["segment_id"] or ""), str(row["role"]), str(row["resolved_path"]),
        )
    )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "content_hashes_included": hash_content,
        "entry_count": len(records),
        "entries": records,
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "limitations": [
            "File presence and hashes do not establish trajectory readability, continuity, equilibration, convergence, or scientific validity."
        ],
    }


def inventory_content_signature_sha256(inventory: Mapping[str, object]) -> str:
    """Return the context-compatible signature of one content-hashed inventory."""

    if inventory.get("content_hashes_included") is not True:
        raise ManifestValidationError((
            "input inventory must include content hashes before a signature is computed",
        ))
    entries = inventory.get("entries")
    if not isinstance(entries, list):
        raise ManifestValidationError(("input inventory entries must be a list",))
    normalized = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestValidationError((
                f"input inventory entry {index} must be an object",
            ))
        signature = entry.get("sha256")
        if not isinstance(signature, str) or _SHA256.fullmatch(signature) is None:
            raise ManifestValidationError((
                f"input inventory entry {index} lacks a valid SHA-256",
            ))
        normalized.append({
            "role": entry.get("role"),
            "system_id": entry.get("system_id"),
            "replica_id": entry.get("replica_id"),
            "segment_id": entry.get("segment_id"),
            "manifest_path": entry.get("manifest_path"),
            "sha256": signature,
        })
    return stable_json_sha256(normalized)
