"""Explicit, hash-bound adoption of unchanged results into a prepared campaign.

Original reports are never rewritten. A receipt binds their original context to
the target context; every consumer revalidates it. This is technical acceptance,
not a scientific-validity certificate or permission to reuse a known bad result.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from .context import compile_project_context_file
from .manifests import content_hash_session, load_json, resolve_manifest_path, sha256_file, stable_json_sha256


SCHEMA = "salsbury-report-adoption-v1"
# These direct schemas have no external fitted-model input. Scalar results also
# bind the trajectory_features definition, not just their own downstream settings.
SUPPORTED = frozenset({
    "ion_atmosphere", "ion_coordination_geometry", "nucleic_acid_geometry",
    "radial_distribution_functions", "trajectory_features",
    "scalar_feature_distributions", "scalar_threshold_states",
})
_SHA = re.compile(r"[a-f0-9]{64}")
_NON_SCIENTIFIC = {
    "project_id", "analysis_output_root", "requested_modules", "protected_locations",
    "compute_environment", "definitions",
}


class ReportReuseError(ValueError):
    """An adoption lacks evidence or changes the original calculation."""


def receipt_path(report_path):
    return Path(str(report_path) + ".adoption.json")


def _definition_closure(module, definitions):
    from .analysis_config import DEPENDENCIES
    result = {}

    def add(name):
        if name in result:
            return
        if not isinstance(definitions.get(name), dict):
            raise ReportReuseError(f"missing dependency definition: {name}")
        result[name] = deepcopy(definitions[name])
        dependencies = set(DEPENDENCIES.get(name, ()))
        if name in {"scalar_feature_distributions", "scalar_threshold_states"}:
            dependencies.add("trajectory_features")
        feature_source = result[name].get("feature_source")
        if feature_source in {"common_pca", "tica", "trajectory_features"}:
            dependencies.discard("common_pca")
            dependencies.add("time_lagged_independent_component_analysis" if feature_source == "tica" else feature_source)
        for dependency in sorted(dependencies):
            add(dependency)

    add(module)
    return result


def scientific_contract(module, project_path):
    """Normalize transport paths only; preserve every selection, axis and weight.

Full current-context validation checks file contents against the native input
signature. The second contract compares relocated manifests by content identity.
Unknown settings are kept verbatim, so unsupported relocations fail closed.
"""
    source = Path(project_path).resolve(strict=True)
    project = load_json(source)
    if project.get("preprocessed_coordinate_source"):
        raise ReportReuseError("cache-backed result adoption is not supported; validate cache lineage separately")
    context = compile_project_context_file(source, hash_content=True)
    system_path = resolve_manifest_path(project["system_manifest"], source)
    system = deepcopy(load_json(system_path))
    entries = context["input_inventory"]["entries"]
    identities = {
        (e["system_id"], e["replica_id"], e["segment_id"], e["role"]):
        {"sha256": e["sha256"], "size_bytes": e["size_bytes"]}
        for e in entries
    }
    for item in system["systems"]:
        for replica in item["replicas"]:
            for role in ("topology", "connectivity"):
                if role in replica:
                    replica[role] = identities[(item["system_id"], replica["replica_id"], None, role)]
            for segment in replica["segments"]:
                for role in ("trajectory", "weights"):
                    if role in segment:
                        segment[role] = identities[(item["system_id"], replica["replica_id"], segment["segment_id"], role)]
    contract = {key: deepcopy(value) for key, value in project.items() if key not in _NON_SCIENTIFIC}
    contract["system_manifest"] = system
    for field in ("reference_structure", "reference_connectivity"):
        if contract.get(field):
            path = resolve_manifest_path(contract[field], source)
            if str(path.resolve()) not in {str(Path(e["resolved_path"]).resolve()) for e in entries}:
                raise ReportReuseError(f"{field} is not covered by the original input-content signature")
            contract[field] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    contract["definitions"] = _definition_closure(module, project.get("definitions", {}))
    return contract, context


def _summary_path(path):
    modern = Path(str(path) + ".summary.json")
    return modern if modern.is_file() else path.parent / "summary.json"


def _original_acceptance(path):
    from .accepted_artifacts import validate_complete_report
    if receipt_path(path).exists():
        raise ReportReuseError("chained adoption is not supported; use the original immutable report")
    report = validate_complete_report(path, require_sidecar=True, verify_inputs=True)
    if report.get("content_hashes_included") is not True:
        raise ReportReuseError("original report does not declare content-hashed inputs")
    if not _SHA.fullmatch(str(report.get("input_content_signature_sha256", ""))):
        raise ReportReuseError("original report lacks a complete input-content signature")
    project = Path(report["project_manifest_path"]).resolve(strict=True)
    system = resolve_manifest_path(load_json(project)["system_manifest"], project)
    if Path(report["system_manifest_path"]).resolve() != system.resolve() or report["system_manifest_sha256"] != sha256_file(system):
        raise ReportReuseError("original report system identity differs from its project")
    # A derived sidecar's own pass is not proof that its upstream context passed.
    for record in load_json(_summary_path(path)).get("source_report_records", []):
        validate_complete_report(Path(record["path"]), verify_inputs=True, require_sidecar=True)
    return report


@content_hash_session()
def review_report(source_report, target_project):
    """Read-only eligibility review. No numerical calculation or output copying."""
    path = Path(source_report).resolve(strict=True)
    target = Path(target_project).resolve(strict=True)
    result = {"source_report_path": str(path), "source_report_sha256": sha256_file(path),
              "target_project_path": str(target), "target_project_sha256": sha256_file(target),
              "eligible": False, "disposition": "insufficient_validation"}
    try:
        report = _original_acceptance(path)
        _check_temporal_policy(report, target)
        module = report["module_id"]
        result["module_id"] = module
        original = Path(report["project_manifest_path"]).resolve(strict=True)
        old_contract, _ = scientific_contract(module, original)
        new_contract, _ = scientific_contract(module, target)
        result["source_scientific_contract_sha256"] = stable_json_sha256(old_contract)
        result["target_scientific_contract_sha256"] = stable_json_sha256(new_contract)
        if old_contract != new_contract:
            result.update(disposition="changed_calculation", reason="inputs, settings, timing, selections or dependency definitions differ")
        elif module not in SUPPORTED:
            result.update(disposition="unsupported_schema", reason=f"fresh-root adoption is not yet validated for {module}")
        else:
            result.update(eligible=True, disposition="unchanged_calculation", reason="original-context validation and content-based scientific contract match")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _check_temporal_policy(report, target_project):
    """Environment-only static mode must not disappear during adoption."""
    config = Path(target_project).parent / "analysis-config.json"
    target_static = (load_json(config).get("execution", {}).get("trajectory_mode") == "static_ensemble") if config.exists() else False
    if target_static and report["module_id"] == "scalar_threshold_states":
        raise ReportReuseError("scalar threshold states cannot be adopted into a static ensemble")
    expected = "disabled: discontinuous static ensemble" if target_static else "ordered segment-local summaries"
    policies = set()
    def visit(value):
        if isinstance(value, dict):
            if "temporal_output_policy" in value:
                policies.add(value["temporal_output_policy"])
            for child in value.values():
                if isinstance(child, (dict, list)): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(report)
    if policies and policies != {expected}:
        raise ReportReuseError("original temporal-output policy differs from the target campaign")


def _relative_companions(report):
    """Only explicit hash-bound companions; absolute paths remain original references."""
    paths = set()

    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for key, name in value.items():
                if not isinstance(name, str):
                    continue
                digest = (value.get(key[:-5] + "_sha256") if key.endswith("_path")
                          else value.get("sha256", value.get("artifact_sha256")) if key in {"path", "artifact_path"} else None)
                if isinstance(digest, str) and not Path(name).is_absolute():
                    p = Path(name)
                    if ".." in p.parts or p == Path("."):
                        raise ReportReuseError(f"relative artifact escapes report directory: {name}")
                    paths.add(p)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
    visit(report)
    return sorted(paths)


def _adopted_summary(source_summary, source_report, target_report):
    """Remap only exact report-link fields; preserve all numerical evidence."""
    def remap(value):
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: (str(target_report) if (key.endswith("_path") or key == "path")
                          and item == str(source_report) else remap(item))
                    for key, item in value.items()}
        return value
    result = remap(load_json(source_summary))
    result["report_adoption"] = {
        "schema": SCHEMA,
        "source_report_path": str(source_report),
        "source_report_sha256": sha256_file(source_report),
        "source_summary_sha256": sha256_file(source_summary),
        "receipt_path": str(receipt_path(target_report)),
        "resources_describe_original_execution": True,
    }
    return result


@content_hash_session()
def adopt_report(source_report, target_project, destination):
    """Publish an unchanged report and receipt into an absent report directory."""
    path = Path(source_report).resolve(strict=True)
    destination = Path(destination).resolve(strict=False)
    if destination.name != "report.json":
        raise ReportReuseError("adoption destination must be a task's report.json")
    if destination.parent.exists():
        raise ReportReuseError("destination report directory already exists; preserve its evidence")
    review = review_report(path, target_project)
    if not review["eligible"]:
        raise ReportReuseError(f"{review['disposition']}: {review['reason']}")
    report = load_json(path)
    source_summary = _summary_path(path)
    receipt = {
        "adoption_schema": SCHEMA, **review,
        "source_summary_path": str(source_summary), "source_summary_sha256": sha256_file(source_summary),
        "target_report_path": str(destination),
        "original_producer_metadata": report.get("execution_resources", {}),
        "validation_implementation_sha256": sha256_file(Path(__file__)),
        "target_analysis_config_sha256": (
            sha256_file(Path(target_project).parent / "analysis-config.json")
            if (Path(target_project).parent / "analysis-config.json").is_file() else None
        ),
        "numerical_payload_rewritten": False, "scientific_acceptance": False,
        "resource_accounting": "report resources are the original computation, not new compute consumed by adoption",
        "limitations": [
            "Original reports, manifests, inputs and absolute companion paths must remain accessible and unchanged.",
            "Contract equality does not certify scientific validity or repair known numerical defects. Review producer versions and relevant fixes before choosing a source result.",
        ],
    }
    destination.parent.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".adoption-", dir=destination.parent.parent))
    # Preserve an interrupted staging directory for diagnosis; never overwrite it.
    shutil.copyfile(path, temporary / "report.json")
    target_summary = temporary / "report.json.summary.json"
    target_summary.write_text(json.dumps(_adopted_summary(source_summary, path, destination),
                                        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["target_summary_sha256"] = sha256_file(target_summary)
    receipt["summary_remap_policy"] = "exact_report_links_and_adoption_marker"
    for relative in _relative_companions(report):
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ReportReuseError(f"companion collides with adoption payload: {relative}")
        shutil.copyfile(path.parent / relative, target)
    receipt_path(temporary / "report.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_adoption(temporary / "report.json", target_project, _staging=True)
    # mkdir+rename under the campaign lock; never replace completed outputs.
    if destination.parent.exists():
        raise ReportReuseError("destination appeared during adoption; staged evidence preserved")
    os.rename(temporary, destination.parent)
    return {**review, "adopted_report_path": str(destination),
            "adoption_receipt_sha256": sha256_file(receipt_path(destination))}


@content_hash_session()
def validate_adoption(path, target_project=None, *, _staging=False):
    """Revalidate the source and both contracts, not a saved 'eligible' flag."""
    path = Path(path).resolve(strict=True)
    receipt = load_json(receipt_path(path))
    from .qualified_dihedral_acceptance import SCHEMA as QUALIFIED_SCHEMA, registered_validation
    if receipt.get("adoption_schema") == QUALIFIED_SCHEMA:
        qualified = registered_validation(path, target_project)
        return {**receipt, "validation_basis": qualified.validation_basis}
    if receipt.get("adoption_schema") != SCHEMA:
        raise ReportReuseError("unknown report-adoption schema")
    if not _staging and str(path) != receipt.get("target_report_path"):
        raise ReportReuseError("adopted report location changed")
    target = Path(target_project or receipt["target_project_path"]).resolve(strict=True)
    if str(target) != receipt["target_project_path"] or sha256_file(target) != receipt["target_project_sha256"]:
        raise ReportReuseError("adoption target project changed")
    config = target.parent / "analysis-config.json"
    if (sha256_file(config) if config.is_file() else None) != receipt.get("target_analysis_config_sha256"):
        raise ReportReuseError("adoption target analysis configuration changed")
    source = Path(receipt["source_report_path"]).resolve(strict=True)
    if sha256_file(path) != receipt["source_report_sha256"] or sha256_file(source) != receipt["source_report_sha256"]:
        raise ReportReuseError("adopted or original report bytes changed")
    summary = _summary_path(source)
    if (str(summary) != receipt["source_summary_path"]
            or sha256_file(summary) != receipt["source_summary_sha256"]
            or sha256_file(_summary_path(path)) != receipt["target_summary_sha256"]):
        raise ReportReuseError("adoption summary changed")
    if load_json(_summary_path(path)) != _adopted_summary(summary, source, receipt["target_report_path"]):
        raise ReportReuseError("adoption summary differs beyond report-link remapping and provenance marker")
    review = review_report(source, target)
    if not review["eligible"]:
        raise ReportReuseError(f"adoption no longer valid: {review['reason']}")
    for key in ("module_id", "source_scientific_contract_sha256", "target_scientific_contract_sha256"):
        if review[key] != receipt.get(key):
            raise ReportReuseError(f"adoption receipt {key} differs from current validation")
    return receipt


@content_hash_session()
def runtime_report(path, target_project):
    """Return a derived in-memory view, with original provenance explicitly retained."""
    receipt = validate_adoption(path, target_project)
    from .qualified_dihedral_acceptance import SCHEMA as QUALIFIED_SCHEMA, registered_validation
    if receipt.get("adoption_schema") == QUALIFIED_SCHEMA:
        qualified = registered_validation(path, target_project)
        return {**deepcopy(qualified.report), "qualified_validation": {
            "basis": qualified.validation_basis, "current_raw_validation": False,
            "current_cache_validation": False, "scientific_acceptance": False,
            "policy_sha256": qualified.policy_sha256}}
    report = load_json(path)
    context = compile_project_context_file(Path(target_project), hash_content=True)
    from .upstream_cache import project_module_contract_sha256
    report["reuse_provenance"] = {"adoption_receipt_path": str(receipt_path(path)),
                                 "adoption_receipt_sha256": sha256_file(receipt_path(path)),
                                 "source_report_path": receipt["source_report_path"],
                                 "source_report_sha256": receipt["source_report_sha256"],
                                 "original_project_manifest_path": report["project_manifest_path"],
                                 "original_project_manifest_sha256": report["project_manifest_sha256"]}
    for key in ("project_manifest_path", "project_manifest_sha256", "system_manifest_path",
                "system_manifest_sha256", "contract_signature_sha256", "input_content_signature_sha256"):
        report[key] = context[key]
    report["module_contract_sha256"] = project_module_contract_sha256(report["module_id"], Path(target_project))
    return report


def adopt_prepared_task(source_report, prepared_root, task_id, *, apply=False):
    """Resolve the target from its native execution plan; never launch a task."""
    root = Path(prepared_root).resolve(strict=True)
    from .user_workflow import campaign_activity, campaign_lock
    if apply:
        activity = campaign_activity(root)
        if activity["local_controller"] != "idle" or activity["slurm_jobs"] or activity["scheduler_query"] == "failed":
            raise ReportReuseError("cannot adopt into an active campaign or one with unknown scheduler state")
    with campaign_lock(root) if apply else nullcontext():
        plan = load_json(root / "local-execution-plan.json")
        tasks = [task for phase in plan["phases"] for task in phase["tasks"] if task["task_id"] == task_id]
        if len(tasks) != 1:
            raise ReportReuseError(f"expected exactly one prepared task: {task_id}")
        task = tasks[0]
        names = task.get("completion_reports", [])
        if len(names) != 1 or not task.get("project_filename"):
            raise ReportReuseError("adoption supports one direct report per task with an explicit project")
        project = (root / task["project_filename"]).resolve(strict=True)
        destination = (root / names[0]).resolve(strict=False)
        if root not in project.parents or root not in destination.parents:
            raise ReportReuseError("prepared target escapes campaign root")
        report = load_json(Path(source_report))
        if report.get("module_id") != task.get("module_id"):
            raise ReportReuseError("source module differs from prepared task")
        review = review_report(source_report, project)
        return adopt_report(source_report, project, destination) if apply else review
