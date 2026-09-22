"""Prepare a NEMO plan and Slurm preview; never submit or run analyses.

Run with the intended checkout on PYTHONPATH. The output directory must be new.
The DEAC policy is retained; only local environment/output paths are adapted.
"""
import argparse
import json
from pathlib import Path
import sys

from salsbury_md_analysis.quickstart import prepare_standard_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--route", choices=("workstation", "generic", "deac"), default="deac")
    parser.add_argument("--hours", type=float)
    parser.add_argument("--expect-preview-infeasible", action="store_true")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[1]
    fixture = repository / "tutorials/nemo_zinc_finger_workstation/data"
    profile_name = "deac.json" if args.route == "deac" else "generic-template.json"
    profile = json.loads((repository / "profiles/slurm" / profile_name).read_text())
    if args.route == "generic":
        profile["profile_id"] = profile["cluster_name"] = "planning-only-example"
        profile["partitions"] = {key: "example" for key in profile["partitions"]}
        profile["node_policy"] = {"cpus_per_node": 44, "memory_gib_per_node": 185,
                                  "maximum_nodes_per_campaign": 1}
    profile["environment"]["python_executable"] = sys.executable
    profile["environment"]["package_root"] = str(repository)
    profile["paths"]["allowed_output_roots"] = [str(args.output.resolve())]
    profile["paths"]["group_storage_root"] = str(args.output.resolve())
    profile_path = args.output / "deac-local-preview.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    config = json.loads((repository / "tutorials/nemo_zinc_finger_workstation/analysis-config.json").read_text())
    hours = args.hours if args.hours is not None else {"workstation": 2, "generic": 10, "deac": 16}[args.route]
    config["execution"].update({"maximum_hours_per_cpu": hours, "maximum_total_cpu_hours": 2 * hours})
    if args.route != "workstation":
        config["execution"].update({"submission_adapter": "slurm", "slurm_profile": str(profile_path)})
    if args.route == "deac":
        config["execution"]["resource_calibration_catalog"] = str(repository / "profiles/apollo_measured_resource_calibrations_v5.json")
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    output = args.output / "prepared"
    prepare_standard_analysis(
        pdb_path=fixture / "nemo_zinc_finger.pdb",
        psf_path=fixture / "nemo_zinc_finger.psf",
        trajectories=[fixture / "nemo_zinc_finger_1000_frames.dcd"],
        output_directory=output, project_id="nemo-overhead-planning-validation",
        frame_interval_ps=0.2, config_path=config_path,
    )
    plan = json.loads((output / "campaign-resource-plan.json").read_text())
    execution = json.loads((output / "local-execution-plan.json").read_text())
    tasks = [t for p in execution["phases"] for t in p["tasks"]]
    summary = {
        "route": args.route, "campaign_hours": hours,
        "scientific_execution_performed": False, "jobs_submitted": False,
        "feasibility_status": plan["feasibility_status"],
        "estimated_cpu_hours": plan["estimated_selected_cpu_hours"],
        "estimated_wall_hours": plan["estimated_selected_wall_hours_lower_bound"],
        "generated_job_count": len(tasks),
        "unmapped_scripts": [t["script"] for t in tasks if not t["planner_task_ids"]],
        "overhead": [{"script": t["script"], "minutes": t["planned_wall_hours"] * 60,
                      "requested_memory_gib": t["requested_memory_gib"],
                      "timeout_minutes": t["requested_wall_minutes"]}
                     for t in tasks if "preflight" in t["script"] or "finalize_reporting" in t["script"]],
    }
    preview_path = output / "slurm-submission-preview.json"
    if preview_path.exists():
        preview = json.loads(preview_path.read_text())
        summary["preview"] = {key: preview.get(key) for key in (
            "generated_schedule_feasibility_status", "submission_permitted",
            "planner_estimated_dependency_critical_path_hours", "walltime_allocation")}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["unmapped_scripts"]:
        raise SystemExit("generated jobs lack planner estimates")
    if summary["feasibility_status"] != "feasible":
        raise SystemExit("campaign plan is not feasible")
    if args.route != "workstation":
        expected = "infeasible" if args.expect_preview_infeasible else "feasible"
        if summary.get("preview", {}).get("generated_schedule_feasibility_status") != expected:
            raise SystemExit("unexpected scheduler preview feasibility")
        if summary["preview"]["submission_permitted"] is not (not args.expect_preview_infeasible):
            raise SystemExit("unexpected submission-permission result")


if __name__ == "__main__":
    main()
