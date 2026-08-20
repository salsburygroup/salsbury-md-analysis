"""Prespecified integration of nonredundant module outputs without hidden scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Mapping

from .clustering import clustering_kmeans_project_safe
from .convergence import convergence_uncertainty_project_safe
from .manifests import ManifestValidationError, load_json, sha256_file
from .observables import optional_observables_project_safe
from .pca_fes import pca_fes_basins_project_safe
from .rmsd_rg import replica_rmsd_rg_project_safe


class IntegratedAnalysisError(ValueError):
    """Raised when integration entries or output paths are ambiguous."""


_RUNNERS: Dict[str, Callable[..., Dict[str, object]]] = {
    "replica_rmsd_rg": replica_rmsd_rg_project_safe,
    "pca_fes_basins": pca_fes_basins_project_safe,
    "clustering_kmeans": clustering_kmeans_project_safe,
    "optional_observables": optional_observables_project_safe,
    "convergence_uncertainty": convergence_uncertainty_project_safe,
}


def json_pointer(document: object, pointer: str) -> object:
    """Resolve a strict RFC-6901 JSON pointer against a report object."""

    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise IntegratedAnalysisError("value_pointer must be an RFC-6901 pointer beginning with /")
    current = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise IntegratedAnalysisError(f"JSON pointer token {token!r} is absent")
            current = current[token]
        elif isinstance(current, list):
            try:
                index = int(token)
                current = current[index]
            except (ValueError, IndexError) as exc:
                raise IntegratedAnalysisError(f"JSON pointer list token {token!r} is invalid") from exc
        else:
            raise IntegratedAnalysisError(f"JSON pointer cannot descend through {type(current).__name__}")
    return current


def _settings(project: Mapping[str, object]) -> Dict[str, object]:
    definitions = project.get("definitions")
    raw = definitions.get("integrated_comparison") if isinstance(definitions, dict) else None
    if not isinstance(raw, dict) or set(raw) != {"entries", "allow_failed_modules"}:
        raise IntegratedAnalysisError(
            "definitions.integrated_comparison must contain entries and allow_failed_modules"
        )
    if not isinstance(raw["allow_failed_modules"], bool):
        raise IntegratedAnalysisError("allow_failed_modules must be boolean")
    entries = raw["entries"]
    if not isinstance(entries, list) or not entries:
        raise IntegratedAnalysisError("integrated entries must be a nonempty array")
    normalized = []
    ids = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {
            "entry_id", "module_id", "value_pointer", "label", "interpretation_type"
        }:
            raise IntegratedAnalysisError(f"integrated entry {index} fields are incomplete")
        entry_id = str(entry["entry_id"]).strip()
        if not entry_id or entry_id in ids:
            raise IntegratedAnalysisError("integrated entry IDs must be nonempty and unique")
        if entry["module_id"] not in _RUNNERS:
            raise IntegratedAnalysisError(
                f"module {entry['module_id']} is not available to integrated comparison"
            )
        if entry["interpretation_type"] not in {"technical", "descriptive", "exploratory", "inferential"}:
            raise IntegratedAnalysisError("interpretation_type is invalid")
        if not str(entry["label"]).strip():
            raise IntegratedAnalysisError("integrated labels must be nonempty")
        ids.add(entry_id)
        normalized.append(dict(entry))
    return {"entries": normalized, "allow_failed_modules": raw["allow_failed_modules"]}


def integrated_comparison_project(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    settings = _settings(project)
    reports: Dict[str, Dict[str, object]] = {}
    rows = []
    issues = []
    for entry in settings["entries"]:
        module_id = str(entry["module_id"])
        if module_id not in reports:
            reports[module_id] = _RUNNERS[module_id](source, hash_content=hash_content)
        report = reports[module_id]
        technical_status = str(report.get("technical_status"))
        if technical_status != "complete":
            message = f"module {module_id} technical status is {technical_status}"
            if not settings["allow_failed_modules"]:
                raise IntegratedAnalysisError(message)
            rows.append({**entry, "technical_status": technical_status, "value": None, "failure": message})
            issues.append({
                "severity": "warning", "code": "INTEGRATED_SOURCE_FAILED",
                "location": module_id, "message": message,
            })
            continue
        try:
            value = json_pointer(report, str(entry["value_pointer"]))
            rows.append({**entry, "technical_status": technical_status, "value": value, "failure": None})
        except IntegratedAnalysisError as exc:
            if not settings["allow_failed_modules"]:
                raise
            rows.append({**entry, "technical_status": technical_status, "value": None, "failure": str(exc)})
            issues.append({
                "severity": "warning", "code": "INTEGRATED_VALUE_UNAVAILABLE",
                "location": str(entry["entry_id"]), "message": str(exc),
            })
    for module_id, report in reports.items():
        for issue in report.get("issues", []):
            if isinstance(issue, dict):
                issues.append({**issue, "upstream_module_id": module_id})
    return {
        "module_id": "integrated_comparison", "technical_status": "complete",
        "scientific_status": "not evaluated", "project_manifest_path": str(source),
        "project_manifest_sha256": sha256_file(source), "content_hashes_included": hash_content,
        "settings": settings,
        "integration_contract": {
            "aggregation": "none",
            "hidden_composite_score": False,
            "raw_values_preserved": True,
            "failures_preserved": True,
            "interpretation_types_preserved": True,
        },
        "integrated_table": rows,
        "source_modules": [{
            "module_id": module_id,
            "technical_status": report.get("technical_status"),
            "scientific_status": report.get("scientific_status"),
            "project_manifest_sha256": report.get("project_manifest_sha256"),
            "input_content_signature_sha256": report.get("input_content_signature_sha256"),
        } for module_id, report in sorted(reports.items())],
        "error_count": 0,
        "warning_count": sum(issue.get("severity") == "warning" for issue in issues),
        "issues": issues,
        "limitations": [
            "The integrated table does not calculate a hidden composite score or erase upstream failures.",
            "Pointers and interpretation types are prespecified; outcome-driven metric selection remains invalid.",
            "Descriptive and exploratory rows do not become inferential merely because they share a table.",
            "Technical completion does not establish scientific validity.",
        ],
    }


def integrated_comparison_project_safe(
    project_path: Path, hash_content: bool = False
) -> Dict[str, object]:
    try:
        return integrated_comparison_project(project_path, hash_content=hash_content)
    except (ManifestValidationError, IntegratedAnalysisError, OSError) as exc:
        messages = list(exc.issues) if isinstance(exc, ManifestValidationError) else [str(exc)]
        return {
            "module_id": "integrated_comparison", "technical_status": "failed",
            "scientific_status": "not evaluated",
            "project_manifest_path": str(Path(project_path).expanduser().resolve(strict=False)),
            "error_count": len(messages), "warning_count": 0,
            "issues": [{"severity": "error", "code": "INTEGRATED_INVALID", "message": message} for message in messages],
        }
