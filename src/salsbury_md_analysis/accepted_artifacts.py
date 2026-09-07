"""Shared fail-closed validation of completed reports and their companions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping
from .manifests import sha256_file


class ArtifactValidationError(ValueError):
    pass


def expected_system_manifest(module_id, project_path):
    """Resolve explicitly configured cache routing, with source validation."""
    from .manifests import load_json, resolve_manifest_path
    project_path = Path(project_path).resolve(strict=True)
    project = load_json(project_path)
    system = resolve_manifest_path(project["system_manifest"], project_path)
    parallel = project.get("definitions", {}).get("structural_qc", {}).get("parallel_execution", {})
    if module_id == "structural_integrity_qc" and parallel.get("enabled") is True:
        from .coordinate_cache import validate_reusable_coordinate_cache
        cached = resolve_manifest_path(parallel["coordinate_cache_system_manifest"], project_path)
        validated = validate_reusable_coordinate_cache(cached.parent, system)
        if Path(validated["cached_system_manifest"]).resolve() != cached.resolve():
            raise ArtifactValidationError("QC cache routing differs from validated cache")
        return cached
    return system


def _complete(payload, label):
    if not isinstance(payload, dict) or payload.get("technical_status") != "complete":
        raise ArtifactValidationError(f"{label} is not technically complete")
    if payload.get("error_count", 0) != 0 or any(
        isinstance(issue, dict) and issue.get("severity") == "error"
        for issue in payload.get("issues", [])
    ):
        raise ArtifactValidationError(f"{label} contains technical errors")


def validate_complete_report(path: Path, *, expected_module=None, expected_project=None,
                             require_sidecar=False, verify_inputs=False, _seen=None):
    """Validate immutable bytes; never repair or overwrite failed evidence."""
    path = Path(path).resolve(strict=True)
    ancestors = set() if _seen is None else set(_seen)
    if path in ancestors:
        raise ArtifactValidationError("cyclic report provenance")
    ancestors.add(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    _complete(report, str(path))
    if path.name in {"final-resource-summary.json", "final-findings-summary.json"}:
        if not isinstance(report.get("source_report_records"), list):
            raise ArtifactValidationError("final summary lacks source-report integrity records")
        for kind in ("json", "csv", "markdown"):
            if not isinstance(report.get(kind + "_sha256"), str):
                raise ArtifactValidationError("final summary lacks companion output hashes")
    module = report.get("module_id")
    if expected_module and module != expected_module:
        raise ArtifactValidationError(f"report module {module!r} differs from {expected_module!r}")
    if path.name == "report.json" and (not isinstance(module, str) or not module):
        raise ArtifactValidationError("analysis report lacks module identity")
    for prefix in ("project_manifest", "system_manifest"):
        value, digest = report.get(prefix + "_path"), report.get(prefix + "_sha256")
        if module == "coordinate_cache" and prefix == "system_manifest":
            if value != report.get("source_system_manifest"):
                raise ArtifactValidationError("cache source manifest identity differs")
            digest = report.get("source_system_manifest_sha256")
        if value is not None or digest is not None:
            if not isinstance(value, str) or not isinstance(digest, str):
                raise ArtifactValidationError(f"incomplete {prefix} identity")
            source = Path(value).expanduser()
            source = source if source.is_absolute() else path.parent / source
            if not source.is_file() or sha256_file(source) != digest:
                raise ArtifactValidationError(f"stale {prefix} hash")
    if verify_inputs and expected_project is None:
        expected_project = report.get("project_manifest_path")
    if expected_project:
        source = Path(expected_project).resolve(strict=True)
        if report.get("project_manifest_sha256") != sha256_file(source):
            from .upstream_cache import project_module_contract_sha256
            if report.get("module_contract_sha256") != project_module_contract_sha256(module, source):
                raise ArtifactValidationError("report does not match current project/module contract")
        if module == "structural_integrity_qc":
            expected_system = expected_system_manifest(module, source)
            if (Path(report["system_manifest_path"]).resolve() != expected_system.resolve()
                    or report["system_manifest_sha256"] != sha256_file(expected_system)):
                raise ArtifactValidationError("QC report differs from validated source/cache routing")
        signature = report.get("input_content_signature_sha256")
        if signature is not None:
            from .context import compile_project_context_file
            current = compile_project_context_file(source, hash_content=True)
            if current.get("input_content_signature_sha256") != signature:
                raise ArtifactValidationError("report input-content signature differs from current files")
    summary_path = Path(str(path) + ".summary.json")
    legacy_summary = path.parent / "summary.json"
    if not summary_path.is_file() and legacy_summary.is_file():
        summary_path = legacy_summary
    if require_sidecar or summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _complete(summary, str(summary_path))
        if summary.get("report_sha256") != sha256_file(path):
            raise ArtifactValidationError("report/summary hash mismatch")
        if summary.get("module_id", module) != module:
            raise ArtifactValidationError("report/summary module mismatch")
    # Artifact records have explicit file identity. Arbitrary textual paths
    # without a checksum are not evidence of a required companion.
    artifact_base = path.parent
    if module == "state_coordinate_exports":
        artifact_base = Path(report["export_directory"]).resolve(strict=True)
    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            if value.get("artifact_schema") == "salsbury-columnar-table-v1":
                from .columnar_artifacts import load_columnar_table
                load_columnar_table(value, verify_array_hashes=True)
            for field, filename in value.items():
                if field.endswith("_path") and isinstance(filename, str):
                    fingerprint = value.get(field[:-5] + "_sha256")
                    if isinstance(fingerprint, str):
                        candidate = Path(filename).expanduser()
                        candidate = candidate if candidate.is_absolute() else artifact_base / candidate
                        if not candidate.is_file() or sha256_file(candidate) != fingerprint:
                            raise ArtifactValidationError(f"missing or changed companion: {filename}")
            digest = value.get("sha256", value.get("artifact_sha256"))
            name = value.get("path", value.get("artifact_path"))
            if isinstance(name, str) and isinstance(digest, str):
                candidate = Path(name).expanduser()
                candidate = candidate if candidate.is_absolute() else artifact_base / candidate
                if not candidate.is_file() or sha256_file(candidate) != digest:
                    raise ArtifactValidationError(f"missing or changed companion: {name}")
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
    if module == "coordinate_cache":
        from .coordinate_cache import validate_reusable_coordinate_cache
        cache_root = Path(report["cache_output_directory"]).resolve(strict=True)
        validate_reusable_coordinate_cache(cache_root, Path(report["source_system_manifest"]))
        # Cache companion paths are relative to the cache root, not this copied report.
        for entry in report.get("cached_per_system_manifests", {}).values():
            target = (cache_root / entry["path"]).resolve(strict=True)
            if cache_root not in target.parents or sha256_file(target) != entry["sha256"]:
                raise ArtifactValidationError("missing or changed per-system cache manifest")
    else:
        visit(report)
    for record in report.get("source_report_records", []):
        child = Path(record["path"])
        child = child if child.is_absolute() else path.parent / child
        validate_complete_report(child, verify_inputs=True, _seen=ancestors)
    return report


def reports_complete(root: Path, names, task: Mapping[str, object] | None = None) -> bool:
    root = Path(root).resolve(strict=True)
    task = task or {}
    try:
        if not task and (root / "local-execution-plan.json").is_file():
            plan = json.loads((root / "local-execution-plan.json").read_text(encoding="utf-8"))
            matches = [entry for phase in plan.get("phases", []) for entry in phase.get("tasks", [])
                       if set(names) and set(names) == set(entry.get("completion_reports", []))]
            if len(matches) == 1:
                task = matches[0]
        for name in names:
            path = (root / name).resolve(strict=True)
            if root not in path.parents:
                return False
            project = task.get("project_filename")
            if project:
                project = root / str(project)
            validate_complete_report(path, expected_module=task.get("module_id"),
                expected_project=project, require_sidecar=path.name == "report.json")
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
