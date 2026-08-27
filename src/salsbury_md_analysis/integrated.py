"""Prespecified and completed-campaign integration without hidden scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

from .clustering import clustering_kmeans_project_safe
from .convergence import convergence_uncertainty_project_safe
from .perturbation_response import perturbation_response_dynamics_project_safe
from .reweighting import trajectory_reweighting_project_safe
from .allosteric_pathways import allosteric_pathways_project_safe
from .energetic_network_embeddings import energetic_network_embeddings_project_safe
from .multivalent_bridges import multivalent_molecular_bridges_project_safe
from .hydration_density import hydration_density_channels_project_safe
from .pocket_dynamics import ensemble_pocket_dynamics_project_safe
from .interaction_fingerprints import interaction_fingerprints_project_safe
from .spatial_interaction_ensembles import spatial_interaction_ensembles_project_safe
from .interaction_persistence import interaction_persistence_project_safe
from .helical_mechanics import helical_mechanics_project_safe
from .random_feature_koopman import random_feature_koopman_project_safe
from .reactive_paths import reactive_path_ensembles_project_safe
from .finding_picker import prioritize_findings
from .manifests import ManifestValidationError, load_json, sha256_file
from .observables import optional_observables_project_safe
from .pca_fes import pca_fes_basins_project_safe
from .rmsd_rg import replica_rmsd_rg_project_safe


class IntegratedAnalysisError(ValueError):
    """Raised when integration entries or output paths are ambiguous."""


def _system_pair(row: Mapping[str, object]) -> tuple[str, str] | None:
    systems = row.get("system_ids")
    if not isinstance(systems, list):
        return None
    unique = sorted({str(value) for value in systems if str(value)})
    if len(unique) != 2:
        return None
    return unique[0], unique[1]


def _is_cross_system(row: Mapping[str, object]) -> bool:
    systems = row.get("system_ids")
    return (
        isinstance(systems, list)
        and len({str(value) for value in systems if str(value)}) >= 2
    )


def _expected_pairs(
    system_ids: Sequence[str], policy: Mapping[str, object]
) -> List[tuple[str, str]]:
    unique = list(dict.fromkeys(map(str, system_ids)))
    if str(policy.get("mode", "all_pairs")) == "reference_vs_all":
        reference = str(policy.get("reference_system_id", ""))
        if reference not in unique:
            raise IntegratedAnalysisError(
                "reference_vs_all integration requires a declared reference system"
            )
        return [tuple(sorted((reference, other))) for other in unique if other != reference]
    return [
        tuple(sorted((unique[left], unique[right])))
        for left in range(len(unique))
        for right in range(left + 1, len(unique))
    ]


def _source_report_inventory(root: Path) -> List[Dict[str, object]]:
    records = []
    integrated_path = root / "results" / "integrated-comparison" / "report.json"
    for path in sorted((root / "results").glob("**/report.json")):
        if path == integrated_path:
            continue
        report = load_json(path)
        records.append({
            "module_id": str(report.get("module_id", path.parent.name)),
            "report_path": str(path),
            "report_sha256": sha256_file(path),
            "technical_status": report.get("technical_status"),
            "scientific_status": report.get("scientific_status"),
        })
    return records


def _module_comparison_coverage(
    module_accounting: Sequence[Mapping[str, object]],
    comparison_findings: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    comparison_counts: Dict[str, int] = {}
    families: Dict[str, set[str]] = {}
    for row in comparison_findings:
        module_id = str(row.get("module_id", ""))
        comparison_counts[module_id] = comparison_counts.get(module_id, 0) + 1
        families.setdefault(module_id, set()).add(
            str(row.get("comparison_family", "unspecified"))
        )
    coverage = []
    for row in module_accounting:
        module_id = str(row.get("module_id", ""))
        count = comparison_counts.get(module_id, 0)
        role = str(row.get("review_role", "scientific_result"))
        if count:
            disposition = "compared_with_rankable_cross_system_findings"
        elif role == "technical_support":
            disposition = "reviewed_technical_support_not_a_scientific_comparison"
        elif role == "quality_control":
            disposition = "reviewed_as_quality_control_not_promoted_as_a_finding"
        elif role == "interpretive_context":
            disposition = "reviewed_as_interpretive_context"
        else:
            disposition = "reviewed_no_automatic_cross_system_highlight"
        coverage.append({
            **dict(row),
            "cross_system_finding_count": count,
            "comparison_families": sorted(families.get(module_id, set())),
            "comparison_disposition": disposition,
        })
    return coverage


def integrated_comparison_results(root: Path) -> Dict[str, object]:
    """Integrate every completed report in a prepared comparative campaign.

    The campaign finding adapters perform method-aware comparisons.  This
    finalizer records the complete review inventory, preserves every candidate
    comparison, and groups the comparisons by system pair.  It deliberately
    does not subtract arbitrary JSON arrays or invent a composite score.
    """

    analysis_root = Path(root).expanduser().resolve(strict=True)
    config_path = analysis_root / "analysis-config.json"
    coverage_path = analysis_root / "module-coverage.json"
    if not config_path.is_file() or not coverage_path.is_file():
        raise IntegratedAnalysisError(
            "prepared comparison requires analysis-config.json and module-coverage.json"
        )
    config = load_json(config_path)
    coverage = load_json(coverage_path)
    comparison_system_ids = coverage.get("comparison_system_ids")
    if not isinstance(comparison_system_ids, list) or len(comparison_system_ids) < 2:
        raise IntegratedAnalysisError(
            "integrated campaign comparison requires at least two comparison systems"
        )

    # The integrated report does not yet exist, so this call exercises the
    # method-aware direct and cross-report comparison adapters exactly once.
    prioritized = prioritize_findings(
        analysis_root, maximum_findings=1_000_000, write_outputs=False
    )
    if prioritized.get("unreported_candidate_count") != 0:
        raise IntegratedAnalysisError(
            "comparison candidate count exceeds the fail-closed integration bound"
        )
    comparison_findings = [
        dict(row) for row in prioritized.get("findings", [])
        if isinstance(row, dict) and _is_cross_system(row)
    ]
    for row in comparison_findings:
        row.pop("finding_id", None)
        row["integrated_comparison_source"] = True

    pair_groups: Dict[tuple[str, str], List[Mapping[str, object]]] = {}
    for row in comparison_findings:
        pair = _system_pair(row)
        if pair is not None:
            pair_groups.setdefault(pair, []).append(row)
    comparison_policy = config.get("comparisons", {})
    if not isinstance(comparison_policy, dict):
        comparison_policy = {}
    expected_pairs = _expected_pairs(
        [str(value) for value in comparison_system_ids], comparison_policy
    )
    pair_summaries = []
    for pair in sorted(set(expected_pairs).union(pair_groups)):
        rows = pair_groups.get(pair, [])
        pair_summaries.append({
            "left_system_id": pair[0],
            "right_system_id": pair[1],
            "comparison_finding_count": len(rows),
            "module_ids": sorted({str(row.get("module_id")) for row in rows}),
            "comparison_families": sorted({
                str(row.get("comparison_family")) for row in rows
            }),
            "highest_ranked_statement": (
                str(rows[0].get("statement", "")) if rows else None
            ),
            "comparison_disposition": (
                "ranked_cross_system_findings"
                if rows else "reviewed_no_automatic_highlight"
            ),
        })

    inventory = _source_report_inventory(analysis_root)
    complete_inventory = [
        row for row in inventory if row.get("technical_status") == "complete"
    ]
    module_accounting = [
        row for row in prioritized.get("module_accounting", [])
        if isinstance(row, dict)
    ]
    reviewed_report_count = int(prioritized.get("reviewed_report_count", 0))
    unreviewed_complete_report_count = max(
        0, len(complete_inventory) - reviewed_report_count
    )
    if unreviewed_complete_report_count:
        raise IntegratedAnalysisError(
            "one or more completed reports were not included in comparison review"
        )
    failed_inventory = [
        row for row in inventory if row.get("technical_status") != "complete"
    ]
    issues = []
    if failed_inventory:
        issues.append({
            "severity": "warning",
            "code": "INTEGRATED_INCOMPLETE_SOURCE_REPORTS_PRESERVED",
            "message": (
                f"{len(failed_inventory)} non-complete source reports were preserved "
                "but were not treated as scientific results"
            ),
        })
    return {
        "module_id": "integrated_comparison",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "analysis_root": str(analysis_root),
        "analysis_config_path": str(config_path),
        "analysis_config_sha256": sha256_file(config_path),
        "module_coverage_path": str(coverage_path),
        "module_coverage_sha256": sha256_file(coverage_path),
        "comparison_policy": comparison_policy,
        "comparison_system_ids": [str(value) for value in comparison_system_ids],
        "integration_contract": {
            "all_completed_reports_reviewed": True,
            "method_aware_comparisons_only": True,
            "arbitrary_array_subtraction": False,
            "hidden_composite_score": False,
            "source_findings_preserved": True,
            "scientific_interpretation_automated": False,
        },
        "source_report_count": len(inventory),
        "complete_source_report_count": len(complete_inventory),
        "failed_source_report_count": len(failed_inventory),
        "reviewed_report_count": reviewed_report_count,
        "unreviewed_complete_report_count": unreviewed_complete_report_count,
        "comparison_candidate_count": len(comparison_findings),
        "system_pair_count": len(pair_summaries),
        "multi_system_nonpair_finding_count": sum(
            _system_pair(row) is None for row in comparison_findings
        ),
        "system_pair_summaries": pair_summaries,
        "comparison_findings": comparison_findings,
        "module_comparison_coverage": _module_comparison_coverage(
            module_accounting, comparison_findings
        ),
        "source_reports": inventory,
        "error_count": 0,
        "warning_count": len(issues),
        "issues": issues,
        "limitations": [
            "Every completed report is reviewed, but only scientifically matched quantities are compared numerically.",
            "A result without an automatic highlight remains in module comparison coverage; absence of a highlight is not evidence of equivalence.",
            "Comparison candidates are ranked transparently and do not form a composite biological score.",
            "Technical completion does not establish convergence, mechanism, causality, kinetics, or biological significance.",
        ],
    }


def integrated_comparison_results_safe(root: Path) -> Dict[str, object]:
    try:
        return integrated_comparison_results(root)
    except (IntegratedAnalysisError, OSError, ValueError) as exc:
        return {
            "module_id": "integrated_comparison",
            "technical_status": "failed",
            "scientific_status": "not evaluated",
            "analysis_root": str(Path(root).expanduser().resolve(strict=False)),
            "error_count": 1,
            "warning_count": 0,
            "issues": [{
                "severity": "error",
                "code": "INTEGRATED_RESULTS_INVALID",
                "message": str(exc),
            }],
        }


_RUNNERS: Dict[str, Callable[..., Dict[str, object]]] = {
    "replica_rmsd_rg": replica_rmsd_rg_project_safe,
    "pca_fes_basins": pca_fes_basins_project_safe,
    "clustering_kmeans": clustering_kmeans_project_safe,
    "optional_observables": optional_observables_project_safe,
    "convergence_uncertainty": convergence_uncertainty_project_safe,
    "perturbation_response_dynamics": perturbation_response_dynamics_project_safe,
    "trajectory_reweighting": trajectory_reweighting_project_safe,
    "allosteric_pathways": allosteric_pathways_project_safe,
    "energetic_network_embeddings": energetic_network_embeddings_project_safe,
    "multivalent_molecular_bridges": multivalent_molecular_bridges_project_safe,
    "hydration_density_channels": hydration_density_channels_project_safe,
    "ensemble_pocket_dynamics": ensemble_pocket_dynamics_project_safe,
    "interaction_fingerprints": interaction_fingerprints_project_safe,
    "spatial_interaction_ensembles": spatial_interaction_ensembles_project_safe,
    "interaction_persistence": interaction_persistence_project_safe,
    "helical_mechanics": helical_mechanics_project_safe,
    "random_feature_koopman": random_feature_koopman_project_safe,
    "reactive_path_ensembles": reactive_path_ensembles_project_safe,
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
