"""Report/input dependencies carried from candidate planning into execution."""
from pathlib import Path
from typing import Mapping, Sequence

from .cache_routing import cache_routing_plan
from .manifests import load_json


def annotate_planning_dependencies(tasks: Sequence[dict], *, root: Path,
                                   base_project: Mapping, view_paths: Sequence[Path],
                                   context_paths: Sequence[Path], config: Mapping,
                                   cache_enabled: bool) -> None:
    """Attach bundle dependencies, never introduce broad stage barriers."""
    projects = {"base": base_project}
    for path in [*view_paths, *context_paths]:
        projects[path.stem.removeprefix("project-")] = load_json(path)
    views = {p.stem.removeprefix("project-") for p in view_paths}
    routed = cache_routing_plan(base_project) if cache_enabled else {}
    cache_modules = set(routed.get("cache_project_modules", []))
    cache = next((str(t["task_id"]) for t in tasks if t["module_id"] == "coordinate_cache"), None)
    preflight = next((str(t["task_id"]) for t in tasks if t.get("execution_script") == "run_preflight.slurm"), None)
    by_module = {}
    view_preflights = {}
    for t in tasks:
        bundle = str(t.setdefault("execution_bundle_id", t["task_id"]))
        scope = str(t.get("workflow_id") or "base")
        by_module.setdefault((scope, str(t["module_id"])), set()).add(bundle)
        if t.get("preflight_source_manifest"):
            view_preflights[str(t["preflight_source_manifest"])] = bundle
    module_config = config.get("modules", {})
    for t in tasks:
        module = str(t["module_id"])
        scope = str(t.get("workflow_id") or "base")
        bundle = str(t["execution_bundle_id"])
        dependencies, waits = set(), set()
        project = projects.get(scope, base_project)
        definitions = project.get("definitions", {})
        if module == "coordinate_cache":
            pass
        elif module == "workflow_preflight":
            if t.get("preflight_requires_cache") and cache:
                dependencies.add(cache)
        elif module == "rmsf_permutation_inference":
            waits = set()
            dependencies.update(b for (s, m), values in by_module.items()
                                if m == "pooled_rmsf" for b in values)
        elif module == "workflow_integrated_reporting":
            waits.update(str(r["execution_bundle_id"]) for r in tasks
                         if r["module_id"] not in {
                             "structural_integrity_qc", "rmsf_permutation_inference",
                             "coordinate_cache", "workflow_preflight",
                             "workflow_integrated_reporting", "workflow_final_reporting"})
        elif module == "workflow_final_reporting":
            waits.update(str(r["execution_bundle_id"]) for r in tasks)
        else:
            if scope in views:
                manifest = str(project.get("system_manifest", "system.json"))
                p = view_preflights.get(manifest, preflight)
                if p:
                    dependencies.add(p)
            else:
                if preflight:
                    dependencies.add(preflight)
                if cache and scope == "base" and (module in cache_modules or module == "structural_integrity_qc"):
                    dependencies.add(cache)
            required = set(module_config.get(module, {}).get("depends_on", []))
            if module == "markov_state_models":
                required.update(m for m in ("pca_fes_basins", "clustering_kmeans", "clustering_hdbscan",
                                            "clustering_imwkmeans", "alternative_clustering") if m in definitions)
            elif module in {"representative_frames", "state_coordinate_exports"}:
                source = definitions.get(module, {}).get("source")
                if isinstance(source, str):
                    required.add(source)
            elif module == "grouped_ml":
                required.add("clustering_kmeans")
            elif module in {"scalar_feature_distributions", "scalar_threshold_states"}:
                required.add("trajectory_features")
            for needed in required:
                waits.update(by_module.get((scope, needed), set()))
        t["planning_dependencies"] = {
            "schema": "salsbury-planning-dependencies-v1",
            "depends_on_bundle_ids": sorted(dependencies - {bundle}),
            "wait_for_bundle_ids": sorted(waits - {bundle}),
        }


def bind_execution_dependencies(phases: Sequence[dict], rows: Sequence[Mapping]) -> None:
    """Resolve the same logical bundle edges to emitted executable task IDs."""
    if not rows or not any("planning_dependencies" in r for r in rows):
        return
    if not all("planning_dependencies" in r for r in rows):
        raise ValueError("partial planner dependency contract")
    by_id = {str(r["task_id"]): r for r in rows}
    tasks = [t for phase in phases for t in phase["tasks"]]
    owners = {}
    for task in tasks:
        for rid in task.get("planner_task_ids", []):
            bundle = str(by_id[rid].get("execution_bundle_id", rid))
            if bundle in owners and owners[bundle] != task["task_id"]:
                raise ValueError("planner bundle maps to multiple execution tasks")
            owners[bundle] = task["task_id"]
    if set(owners) != {str(r.get("execution_bundle_id", r["task_id"])) for r in rows}:
        raise ValueError("native schedule does not cover every planner bundle")
    for task in tasks:
        deps, waits = set(), set()
        task["schedule_order_key"] = min(
            str(by_id[rid].get("execution_bundle_id", rid))
            for rid in task.get("planner_task_ids", []))
        for rid in task.get("planner_task_ids", []):
            contract = by_id[rid]["planning_dependencies"]
            deps.update(owners[b] for b in contract["depends_on_bundle_ids"])
            # An optional cache producer can be removed by an explicitly
            # reduced plan. A missing required input still fails above.
            waits.update(owners[b] for b in contract["wait_for_bundle_ids"] if b in owners)
        task["depends_on_task_ids"] = sorted(deps - {task["task_id"]})
        task["wait_for_task_ids"] = sorted(waits - {task["task_id"]})
