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
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[1]
    fixture = repository / "tutorials/nemo_zinc_finger_workstation/data"
    profile = json.loads((repository / "profiles/slurm/deac.json").read_text())
    profile["environment"]["python_executable"] = sys.executable
    profile["environment"]["package_root"] = str(repository)
    profile["paths"]["allowed_output_roots"] = [str(args.output.resolve())]
    profile["paths"]["group_storage_root"] = str(args.output.resolve())
    profile_path = args.output / "deac-local-preview.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n")
    config = json.loads((repository / "tutorials/nemo_zinc_finger_workstation/analysis-config.json").read_text())
    config["execution"].update({
        "submission_adapter": "slurm", "slurm_profile": str(profile_path),
        "maximum_hours_per_cpu": 16, "maximum_total_cpu_hours": 32,
        "resource_calibration_catalog": str(repository / "profiles/apollo_measured_resource_calibrations_v5.json"),
    })
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
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["unmapped_scripts"]:
        raise SystemExit("generated jobs lack planner estimates")


if __name__ == "__main__":
    main()
