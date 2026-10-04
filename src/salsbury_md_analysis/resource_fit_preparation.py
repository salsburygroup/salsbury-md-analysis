"""Core-first opt-in preparation shared by local and comparative workflows."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .analysis_config import make_resource_fit_config
from .manifests import load_json
from .planning_diagnostics import planning_event
from .planning_reuse import reuse_planning_work


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _disabled_switches(requested, selected):
    """Enumerate the exact selected policy, not just switches tried by a search."""
    switches = []
    for key, value in requested.get("modules", {}).items():
        if value.get("enabled") and not selected["modules"][key]["enabled"]:
            switches.append(f"modules.{key}.enabled")
    for key, value in requested.get("clustering", {}).get("methods", {}).items():
        if value.get("enabled") and not selected["clustering"]["methods"][key]["enabled"]:
            switches.append(f"clustering.methods.{key}.enabled")
    if requested.get("community_analysis", {}).get("pald", {}).get("enabled") and not selected.get("community_analysis", {}).get("pald", {}).get("enabled"):
        switches.append("community_analysis.pald.enabled")
    for key, value in requested.get("views", {}).items():
        if value.get("state_trajectory_exports_enabled") and not selected["views"][key]["state_trajectory_exports_enabled"]:
            switches.append(f"views.{key}.state_trajectory_exports_enabled")
    return sorted(switches)


@reuse_planning_work
def prepare_core_first_resource_fit(*, prepare, common, destination: Path,
                                    target_wall_hours, config_path):
    """Validate the native core, then full/reduced scopes, before materializing.

    No analysis is executed. Each trial is a fresh native preparation, so costs,
    dependencies, sampling and emitted schedule are recalculated together.
    Failed-search evidence never becomes a claim of resource infeasibility.
    """
    from .quickstart import QuickstartPlanningError, QuickstartError

    evidence = {}
    trial_directories = {}
    selected_trial = "requested"
    fallback_reason = None
    journal = destination.with_name(destination.name + ".resource-fit-evidence")
    if journal.exists():
        raise QuickstartError(f"resource-fit evidence already exists: {journal}; choose a new versioned directory")
    journal.mkdir(parents=True)

    def preserve(target=journal):
        for name, value in evidence.items():
            output = trial_directories[name]
            # Keep byte-exact sampling inputs after the temporary trial goes
            # away, including when final materialization raises. These are
            # preparation documents, never trajectories or analysis results.
            documents = [output / "fixed-sampling-schedule.json", output / "sampling-plan.json"]
            documents.extend(sorted(output.glob("project*.json")))
            documents.extend(sorted(output.glob("system*.json")))
            hashes = {}
            for source in documents:
                if not source.is_file():
                    continue
                relative = f"{name}/{source.name}"
                destination_file = target / relative
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination_file)
                hashes[relative] = hashlib.sha256(destination_file.read_bytes()).hexdigest()
            value["preparation_documents"] = {
                "sha256": hashes,
                "scope": "byte-exact trial sampling and project/manifest declarations; not analysis results",
            }
            _write(target / f"{name}.json", value)

    with tempfile.TemporaryDirectory(prefix="salsbury-core-first-fit-") as temporary:
        root = Path(temporary)

        def trial(name, *, path=config_path, core=False, reductions=True):
            output = root / name
            trial_directories[name] = output
            planning_event("resource_fit_trial_started", trial=name)
            try:
                prepare(**{**common, "protected_core_only": core or common.get("protected_core_only", False)},
                        output_directory=output, target_wall_hours=target_wall_hours,
                        config_path=path, _recommend_reductions=reductions)
            except QuickstartPlanningError as exc:
                planning_event("resource_fit_trial_finished", trial=name,
                               status="search_failed" if exc.plan.get("planning_search_failed") else "rejected")
                evidence[name] = {"status": "search_failed" if exc.plan.get("planning_search_failed") else "rejected",
                                  "message": str(exc), "config": exc.analysis_config, "plan": exc.plan}
                preserve()
                return exc.analysis_config, exc.plan, exc
            except QuickstartError as exc:
                planning_event("resource_fit_trial_finished", trial=name, status="preparation_error")
                evidence[name] = {"status": "invalid_input_or_preparation_error", "message": str(exc)}
                preserve()
                raise
            config = load_json(output / "analysis-config.json")
            plan = load_json(output / "campaign-resource-plan.json")
            if plan.get("feasibility_status") != "feasible":
                raise QuickstartError("resource-fit trial did not pass native schedule validation")
            evidence[name] = {"status": "validated", "config": config, "plan": plan,
                              "execution_adapter": load_json(output / "execution-adapter.json")}
            preserve()
            planning_event("resource_fit_trial_finished", trial=name, status="validated")
            return config, plan, None

        raw_config = load_json(config_path) if config_path is not None else {}
        fixed = bool(raw_config.get("sampling", {}).get("fixed_schedule_file"))
        if fixed:
            # A frozen scientific schedule may not be reduced by this mode.
            core_config, core_plan = None, None
        else:
            core_config, core_plan, core_error = trial("protected-core", core=True, reductions=False)
            if core_error is not None:
                preserve()
                failure_plan = deepcopy(core_plan)
                failure_plan["protected_core_rejected"] = not core_plan.get("planning_search_failed", False)
                raise QuickstartPlanningError(
                    ("Protected-core search failed; feasibility remains unknown. "
                     if core_plan.get("planning_search_failed") else
                     "No acceptable reduced plan: protected-core preparation did not fit. ")
                    + str(core_error), plan=failure_plan, analysis_config=core_config,
                    output_directory=destination,
                ) from core_error

        requested_config, requested_plan, failure = trial("requested", reductions=not fixed)
        active_config = requested_config
        applied_switches = []
        recommendation = requested_plan.get("method_reduction_recommendation", {})
        if failure is not None:
            if fixed:
                preserve()
                raise failure
            patch = recommendation.get("configuration_patch", {})
            if recommendation.get("recommendation_status") == "feasible_subset_found" and patch:
                applied_switches = list(patch)
                active_config, _, _ = make_resource_fit_config(requested_config, list(patch))
                candidate = root / "reduced-config.json"
                _write(candidate, active_config)
                active_config, _, failure = trial("reduced", path=candidate, reductions=False)
                if failure is None:
                    selected_trial = "reduced"
            if failure is not None:
                # The exact native core has already passed; do not lose it to
                # a stalled coupling search or an imperfect subset estimate.
                active_config = deepcopy(core_config)
                applied_switches = []
                fallback_reason = str(failure)
                selected_trial = "protected-core"

        all_disabled = _disabled_switches(requested_config, active_config)
        # Keep the explicit patch auditable, including dependency-induced loss.
        _, direct, transitive = make_resource_fit_config(
            requested_config, applied_switches or all_disabled)
        active_path = root / "selected-config.json"
        _write(active_path, active_config)
        # Native prepare requires an empty directory. Preserve all trial
        # receipts after materialization, even if that final step fails.
        report = None
        planning_event("resource_fit_materialization_started")
        try:
            report = prepare(**common, output_directory=destination,
                             target_wall_hours=None, config_path=active_path,
                             _recommend_reductions=False,
                             _fixed_sampling_replay=root / selected_trial / "fixed-sampling-schedule.json")
        finally:
            preserve()
            preserve(destination / "resource-fit-evidence")
        final_plan = load_json(destination / "campaign-resource-plan.json")
        selected_plan = evidence[selected_trial]["plan"]
        refinement = selected_plan.get("planning_refinement")
        planning_event("resource_fit_materialization_finished", status=final_plan.get("feasibility_status"))
        fit_report = {
            "report_schema": "salsbury-resource-fit-report-v1",
            "technical_status": "complete",
            "planning_status": "replanned_with_dependency_closed_optional_reduction" if direct else "requested_plan_feasible",
            "automatic_changes_applied": bool(direct),
            "protected_set_preserved": True,
            "protected_core_checked_first": not fixed,
            "protected_core_fallback_used": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "optimality_proven": False,
            "selected_validated_trial": selected_trial,
            "validated_sampling_replayed": True,
            "planning_refinement": refinement,
            "directly_disabled_configuration_switches": direct,
            "disabled_configuration_switches": all_disabled,
            "transitively_disabled_modules": transitive,
            "reduction_decisions": recommendation.get("decisions", []),
            "requested_config": "analysis-config.requested.json",
            "resolved_config": "analysis-config.resource-fit.json",
            "requested_plan": "campaign-resource-plan.requested.json",
            "protected_core_plan": "resource-fit-evidence/protected-core.json" if not fixed else None,
            "final_plan": "campaign-resource-plan.json",
            "final_feasibility_status": final_plan.get("feasibility_status"),
            "original_request_preserved": True,
            "preparation_journal": str(journal),
            "failure_boundary": "A failed or timed-out search is unknown feasibility; only a validated native plan can be emitted.",
        }
        files = {"analysis-config.requested.json": requested_config,
                 "analysis-config.resource-fit.json": active_config,
                 "campaign-resource-plan.requested.json": requested_plan,
                 "resource-fit-report.json": fit_report}
        for name, value in files.items():
            _write(destination / name, value)
        report["generated_files"].extend([*files, *[f"resource-fit-evidence/{name}.json" for name in evidence],
            *[f"resource-fit-evidence/{path}" for value in evidence.values()
              for path in value["preparation_documents"]["sha256"]]])
        report["resource_fit"] = fit_report
        return report
