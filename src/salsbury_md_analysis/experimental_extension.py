"""Fail-closed reuse of a prepared main campaign by experimental workflows."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Mapping, Sequence

from .analysis_config import COMMAND_MODULES, DEFAULT_DISABLED_MODULES
from .manifests import (
    load_json, resolve_manifest_path, sha256_file, validate_project,
)


class ExperimentalExtensionError(ValueError):
    """Raised when a main campaign cannot be reused without ambiguity."""


# The final integrated comparison and finding report must be rebuilt after the
# experimental reports have been added.  All other technically complete main
# modules are eligible for immutable reuse.
_RECOMPUTED_AFTER_EXTENSION = {
    "integrated_comparison", "rmsf_permutation_inference",
}


def _complete_report(path: Path) -> Dict[str, object]:
    try:
        report = load_json(path)
    except (OSError, ValueError) as exc:
        raise ExperimentalExtensionError(
            f"upstream report is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(report, dict) or report.get("technical_status") != "complete":
        raise ExperimentalExtensionError(
            f"upstream report is not technically complete: {path}"
        )
    summary_path = Path(str(path) + ".summary.json")
    if not summary_path.is_file():
        raise ExperimentalExtensionError(
            f"upstream report has no immutable summary sidecar: {path}"
        )
    try:
        summary = load_json(summary_path)
    except (OSError, ValueError) as exc:
        raise ExperimentalExtensionError(
            f"upstream report summary is unreadable: {summary_path}: {exc}"
        ) from exc
    if (
        not isinstance(summary, dict)
        or summary.get("technical_status") != "complete"
        or summary.get("report_sha256") != sha256_file(path)
    ):
        raise ExperimentalExtensionError(
            f"upstream report summary is incomplete or hash-mismatched: {path}"
        )
    return report


def _module_contract_sha256(module_id: str, project_path: Path) -> str:
    """Hash science while normalizing source-versus-cache system routing."""

    source = Path(project_path).expanduser().resolve(strict=False)
    project = load_json(source)
    definitions = project.get("definitions")
    if not isinstance(definitions, dict):
        raise ExperimentalExtensionError(
            f"project has invalid definitions while hashing {module_id}"
        )
    definition = definitions.get(module_id)
    if definition is not None and not isinstance(definition, dict):
        raise ExperimentalExtensionError(
            f"project has an invalid {module_id} definition"
        )
    excluded = {
        "project_id", "analysis_output_root", "requested_modules",
        "protected_locations", "definitions", "system_manifest",
        "reference_structure", "reference_connectivity",
    }
    contract = {
        key: value for key, value in project.items() if key not in excluded
    }
    contract["definitions"] = {
        module_id: deepcopy(definition) if isinstance(definition, dict) else None
    }
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_report_path(
    source: Path, task: Mapping[str, object]
) -> tuple[Path, Path, str]:
    """Resolve one task's report, runtime project, and reusable project scope."""

    project_value = task.get("project_filename")
    command = task.get("command")
    if not isinstance(project_value, str) or not isinstance(command, str):
        raise ExperimentalExtensionError(
            "analysis task has no project manifest or command"
        )
    runtime_project = Path(project_value)
    if not runtime_project.is_absolute():
        runtime_project = source / runtime_project
    runtime_project = runtime_project.resolve(strict=True)
    try:
        runtime_relative = str(runtime_project.relative_to(source))
    except ValueError as exc:
        raise ExperimentalExtensionError(
            "analysis task project is outside the upstream campaign"
        ) from exc
    runtime_payload = load_json(runtime_project)
    output_root = runtime_payload.get("analysis_output_root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ExperimentalExtensionError(
            f"analysis task project has no output root: {runtime_relative}"
        )
    report_path = Path(output_root)
    if not report_path.is_absolute():
        report_path = source / report_path
    report_path = report_path / command / "report.json"
    reuse_relative = (
        "project.json"
        if runtime_project.name in {
            "project-cache-base.json", "project-structural-qc-parallel.json",
        }
        else runtime_relative
    )
    return report_path, runtime_project, reuse_relative


def _validate_report_provenance(
    report: Mapping[str, object],
    *,
    module_id: str,
    runtime_project: Path,
) -> None:
    if report.get("module_id") != module_id:
        raise ExperimentalExtensionError(
            f"upstream report module does not match {module_id}"
        )
    reported_project = Path(
        str(report.get("project_manifest_path", ""))
    ).expanduser().resolve(strict=False)
    if (
        reported_project != runtime_project
        or report.get("project_manifest_sha256") != sha256_file(runtime_project)
    ):
        raise ExperimentalExtensionError(
            "upstream report project path or hash does not match its task"
        )
    runtime_payload = load_json(runtime_project)
    system_value = runtime_payload.get("system_manifest")
    if not isinstance(system_value, str) or not system_value.strip():
        raise ExperimentalExtensionError(
            "upstream runtime project has no system manifest"
        )
    runtime_system = resolve_manifest_path(system_value, runtime_project)
    reported_system = Path(
        str(report.get("system_manifest_path", ""))
    ).expanduser().resolve(strict=False)
    if (
        reported_system != runtime_system
        or report.get("system_manifest_sha256") != sha256_file(runtime_system)
    ):
        raise ExperimentalExtensionError(
            "upstream report system path or hash does not match its task"
        )


def inspect_main_campaign(root: Path) -> Dict[str, object]:
    """Return the immutable reports that a new experimental campaign may reuse."""

    source = Path(root).expanduser().resolve(strict=True)
    required = (
        "system.json", "project.json", "analysis-config.json",
        "local-execution-plan.json", "campaign-resource-plan.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ExperimentalExtensionError(
            "upstream main campaign lacks required files: " + ", ".join(missing)
        )
    config = load_json(source / "analysis-config.json")
    if not isinstance(config, dict):
        raise ExperimentalExtensionError("upstream analysis config is invalid")
    if config.get("enable_all_experimental_modules") is True:
        raise ExperimentalExtensionError(
            "--experimental-after-main requires a main campaign, not a campaign "
            "that already enabled all experimental modules"
        )
    plan = load_json(source / "local-execution-plan.json")
    if not isinstance(plan, dict) or not isinstance(plan.get("phases"), list):
        raise ExperimentalExtensionError("upstream local execution plan is invalid")
    resource_plan = load_json(source / "campaign-resource-plan.json")
    resource_rows = resource_plan.get("tasks") if isinstance(
        resource_plan, dict
    ) else None
    if not isinstance(resource_rows, list):
        raise ExperimentalExtensionError(
            "upstream campaign resource plan has no task allocations"
        )
    resource_by_id = {
        str(row["task_id"]): row for row in resource_rows
        if isinstance(row, dict) and isinstance(row.get("task_id"), str)
    }

    reusable = []
    incomplete = []
    for phase in plan["phases"]:
        if not isinstance(phase, dict) or not isinstance(phase.get("tasks"), list):
            raise ExperimentalExtensionError("upstream execution phase is invalid")
        for task in phase["tasks"]:
            if not isinstance(task, dict):
                raise ExperimentalExtensionError("upstream execution task is invalid")
            module_id = task.get("module_id")
            if (
                not isinstance(module_id, str)
                or module_id in DEFAULT_DISABLED_MODULES
                or module_id in _RECOMPUTED_AFTER_EXTENSION
            ):
                continue
            try:
                report_path, runtime_project, reuse_relative = _task_report_path(
                    source, task
                )
                report = _complete_report(report_path)
                _validate_report_provenance(
                    report,
                    module_id=module_id,
                    runtime_project=runtime_project,
                )
            except (ExperimentalExtensionError, OSError, ValueError) as exc:
                incomplete.append({
                    "task_id": task.get("task_id"),
                    "module_id": module_id,
                    "reason": str(exc),
                })
                continue
            reusable_project = source / reuse_relative
            if not reusable_project.is_file():
                incomplete.append({
                    "task_id": task.get("task_id"),
                    "module_id": module_id,
                    "reason": (
                        "no source project exists for the runtime cache project"
                    ),
                })
                continue
            try:
                report_relative = str(report_path.resolve(strict=True).relative_to(source))
            except ValueError:
                incomplete.append({
                    "task_id": task.get("task_id"),
                    "module_id": module_id,
                    "reason": "report or project is outside the upstream campaign",
                })
                continue
            planner_task_ids = [
                str(value) for value in task.get("planner_task_ids", [])
                if str(value) in resource_by_id
            ]
            if not planner_task_ids:
                incomplete.append({
                    "task_id": task.get("task_id"),
                    "module_id": module_id,
                    "reason": "no matching upstream planner allocation",
                })
                continue
            reusable.append({
                "task_id": task.get("task_id"),
                "module_id": module_id,
                "project_relative_path": reuse_relative,
                "runtime_project_relative_path": str(
                    runtime_project.relative_to(source)
                ),
                "report_relative_path": report_relative,
                "report_sha256": sha256_file(report_path),
                "summary_sha256": sha256_file(Path(str(report_path) + ".summary.json")),
                "planner_task_ids": planner_task_ids,
            })

    if incomplete:
        preview = "; ".join(
            f"{row['module_id']}: {row['reason']}" for row in incomplete[:8]
        )
        if len(incomplete) > 8:
            preview += f"; and {len(incomplete) - 8} more"
        raise ExperimentalExtensionError(
            "upstream main campaign is not complete enough for immutable extension "
            "reuse: " + preview
        )
    if not reusable:
        raise ExperimentalExtensionError(
            "upstream main campaign contains no reusable completed module reports"
        )
    external_allocations = {}
    for row in reusable:
        for task_id in row["planner_task_ids"]:
            allocation = resource_by_id[task_id]
            selected = allocation.get("selected_physical_frames_per_replica")
            if not isinstance(selected, list) or not selected:
                raise ExperimentalExtensionError(
                    f"upstream planner task has no selected frame allocation: {task_id}"
                )
            external_allocations[task_id] = {
                "task_id": task_id,
                "module_id": row["module_id"],
                "selected_physical_frames_per_replica": [
                    int(value) for value in selected
                ],
                "integer_stride": int(allocation.get("integer_stride", 1)),
                "frame_selection": deepcopy(allocation.get(
                    "frame_selection", {"mode": "fixed_stride_v1"}
                )),
                "report_relative_path": row["report_relative_path"],
            }
    cache = source / "coordinate-cache"
    cache_path = str(cache) if (
        (cache / "coordinate-cache-report.json").is_file()
        and (cache / "system-cache.json").is_file()
    ) else None
    return {
        "extension_contract_schema": "salsbury-experimental-after-main-v1",
        "technical_status": "complete",
        "scientific_status": "not evaluated",
        "upstream_main_campaign": str(source),
        "upstream_system_manifest": str(source / "system.json"),
        "upstream_system_manifest_sha256": sha256_file(source / "system.json"),
        "upstream_analysis_config_sha256": sha256_file(
            source / "analysis-config.json"
        ),
        "upstream_execution_plan_sha256": sha256_file(
            source / "local-execution-plan.json"
        ),
        "upstream_campaign_resource_plan_sha256": sha256_file(
            source / "campaign-resource-plan.json"
        ),
        "coordinate_cache_input": cache_path,
        "reusable_reports": reusable,
        "reusable_report_count": len(reusable),
        "external_task_allocations": external_allocations,
    }


def bind_system_manifest(
    destination: Path,
    payload: Mapping[str, object],
    contract: Mapping[str, object],
    *,
    upstream_relative_path: str = "system.json",
) -> Path:
    """Bind a generated manifest to its byte-identical upstream main manifest."""

    upstream_root = Path(str(contract["upstream_main_campaign"]))
    upstream = (upstream_root / upstream_relative_path).resolve(strict=True)
    upstream_payload = load_json(upstream)
    if upstream_payload != dict(payload):
        raise ExperimentalExtensionError(
            f"experimental inputs differ from upstream main {upstream_relative_path}"
        )
    destination.symlink_to(upstream)
    return destination


def apply_main_report_reuse(
    extension_root: Path,
    contract: Mapping[str, object],
    project_relative_paths: Sequence[str],
) -> Dict[str, object]:
    """Remove validated main tasks and retain their reports as immutable inputs."""

    upstream_root = Path(str(contract["upstream_main_campaign"]))
    rows = contract.get("reusable_reports")
    if not isinstance(rows, list):
        raise ExperimentalExtensionError("extension contract has no report inventory")
    by_project: Dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if isinstance(row, Mapping):
            by_project.setdefault(str(row["project_relative_path"]), []).append(row)

    linked_reports: Dict[str, Dict[str, object]] = {}
    external_by_project: Dict[str, list[str]] = {}
    for relative in project_relative_paths:
        if relative not in by_project:
            continue
        extension_path = extension_root / relative
        upstream_path = upstream_root / relative
        if not extension_path.is_file() or not upstream_path.is_file():
            continue
        extension_project = load_json(extension_path)
        upstream_project = load_json(upstream_path)
        if not isinstance(extension_project, dict) or not isinstance(
            upstream_project, dict
        ):
            raise ExperimentalExtensionError(f"invalid project manifest: {relative}")
        requested = extension_project.get("requested_modules")
        definitions = extension_project.get("definitions")
        upstream_definitions = upstream_project.get("definitions")
        if not isinstance(requested, list) or not isinstance(definitions, dict):
            raise ExperimentalExtensionError(f"invalid extension project: {relative}")
        if not isinstance(upstream_definitions, dict):
            raise ExperimentalExtensionError(f"invalid upstream project: {relative}")

        external = []
        for row in by_project.get(relative, []):
            module_id = str(row["module_id"])
            if module_id not in requested:
                continue
            upstream_definition = upstream_definitions.get(module_id)
            extension_definition = definitions.get(module_id)
            if (upstream_definition is None) != (extension_definition is None):
                raise ExperimentalExtensionError(
                    f"upstream and extension {module_id} definitions differ in {relative}"
                )
            if isinstance(upstream_definition, dict):
                definitions[module_id] = deepcopy(upstream_definition)
            elif upstream_definition is not None:
                raise ExperimentalExtensionError(
                    f"upstream project {relative} has an invalid {module_id} definition"
                )
            extension_path.write_text(
                json.dumps(extension_project, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # The system manifest is linked to the upstream bytes before this
            # function runs. A contract mismatch therefore exposes a real
            # scientific-definition difference rather than a path artifact.
            if _module_contract_sha256(
                module_id, extension_path
            ) != _module_contract_sha256(module_id, upstream_path):
                raise ExperimentalExtensionError(
                    f"upstream {module_id} contract does not match {relative}"
                )
            external.append(module_id)
            report_relative = str(row["report_relative_path"])
            linked_reports[report_relative] = dict(row)
        if external:
            extension_path.write_text(
                json.dumps(extension_project, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            validate_project(
                extension_project, source_path=extension_path, check_paths=True
            )
            external_by_project[relative] = sorted(set(external))

    return {
        "external_modules_by_project": external_by_project,
        "linked_reports": [linked_reports[key] for key in sorted(linked_reports)],
        "externally_reused_module_ids": sorted({
            module_id for values in external_by_project.values()
            for module_id in values
        }),
    }


def remove_reused_modules(
    extension_root: Path, reuse: Mapping[str, object]
) -> None:
    """Remove externally satisfied modules only after planning has allocated them."""

    raw = reuse.get("external_modules_by_project")
    if not isinstance(raw, Mapping):
        raise ExperimentalExtensionError("reuse project inventory is invalid")
    for relative, module_values in raw.items():
        if not isinstance(relative, str) or not isinstance(module_values, list):
            raise ExperimentalExtensionError("reuse project inventory is invalid")
        path = extension_root / relative
        project = load_json(path)
        requested = project.get("requested_modules")
        if not isinstance(requested, list):
            raise ExperimentalExtensionError(
                f"extension project requested_modules is invalid: {relative}"
            )
        external = {str(value) for value in module_values}
        project["requested_modules"] = [
            str(value) for value in requested if str(value) not in external
        ]
        path.write_text(
            json.dumps(project, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_project(project, source_path=path, check_paths=True)


def filter_base_commands(
    commands: Sequence[str], reuse: Mapping[str, object]
) -> list[str]:
    by_project = reuse.get("external_modules_by_project")
    external = set(
        by_project.get("project.json", [])
        if isinstance(by_project, Mapping) else []
    )
    return [
        command for command in commands
        if COMMAND_MODULES.get(command, command) not in external
    ]


def materialize_report_links(
    extension_root: Path, contract: Mapping[str, object], reuse: Mapping[str, object]
) -> list[str]:
    """Create read-only links to validated reports without modifying main output."""

    upstream_root = Path(str(contract["upstream_main_campaign"]))
    rows = reuse.get("linked_reports")
    if not isinstance(rows, list):
        return []
    generated = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        relative = str(row["report_relative_path"])
        source = upstream_root / relative
        destination = extension_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            raise ExperimentalExtensionError(
                f"experimental extension output already exists: {destination}"
            )
        destination.symlink_to(source)
        generated.append(relative)
        source_summary = Path(str(source) + ".summary.json")
        destination_summary = Path(str(destination) + ".summary.json")
        destination_summary.symlink_to(source_summary)
        generated.append(str(Path(relative + ".summary.json")))
    return generated


def write_extension_contract(
    extension_root: Path,
    contract: Mapping[str, object],
    reuse: Mapping[str, object],
    *,
    integrated_comparison_recomputed: bool,
) -> str:
    payload = {
        **deepcopy(dict(contract)),
        "reuse": deepcopy(dict(reuse)),
        "immutable_upstream": True,
        "main_results_recomputed": False,
        "integrated_comparison_recomputed": bool(
            integrated_comparison_recomputed
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path = extension_root / "experimental-after-main-contract.json"
    path.write_text(encoded, encoding="utf-8")
    return path.name
