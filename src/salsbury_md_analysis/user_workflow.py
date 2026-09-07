"""Terminal-first setup, diagnostics, inspection, and reviewed recovery."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

COMMANDS = {"init", "doctor", "plan", "status", "run", "resume"}


def add_parsers(subparsers):
    init = subparsers.add_parser("init", help="Create an editable study and analysis config; never launch work.")
    init.add_argument("directory", type=Path)
    init.add_argument("--pdb", type=Path)
    init.add_argument("--connectivity", type=Path)
    init.add_argument("--trajectory", action="append", type=Path, default=[])
    init.add_argument("--frame-interval-ps", type=float)
    init.add_argument("--cpus", type=int, default=1)
    init.add_argument("--memory-gib", type=float, default=8)
    init.add_argument("--hours", type=float, default=24)
    init.add_argument("--adapter", choices=("local", "slurm", "custom"), default="local")
    init.add_argument("--slurm-profile", type=Path)
    init.add_argument("--interactive", action="store_true", help="Ask for each system and its independent replicas.")
    doctor = subparsers.add_parser("doctor", help="Check dependencies, platform, inputs, and execution tools without running analysis.")
    doctor.add_argument("path", nargs="?", type=Path, help="Study JSON, analysis config, or prepared directory.")
    doctor.add_argument("--json", action="store_true")
    plan = subparsers.add_parser("plan", help="Prepare a study for review; never submit or execute it.")
    plan.add_argument("study", type=Path)
    plan.add_argument("--output", type=Path, help="New prepared directory; defaults to study directory/analysis.")
    for command in ("status", "run", "resume"):
        parser = subparsers.add_parser(command, help={
            "status":"Show task completion, blockers, logs, and next actions.",
            "run":"Execute a reviewed local plan or submit its Slurm jobs.",
            "resume":"Review safe recovery; add --execute to run only unfinished tasks.",
        }[command])
        parser.add_argument("root", type=Path)
        if command == "status":
            parser.add_argument("--json", action="store_true")
        if command == "resume":
            parser.add_argument("--execute", action="store_true")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")


def initialize(args):
    from .analysis_config import default_analysis_config
    from .registry import MODULES
    destination = args.directory.expanduser().resolve()
    if destination.exists():
        raise ValueError("Choose a new setup directory; existing files will not be overwritten.")
    cpus = _positive(args.cpus, "cpus")
    memory = _positive(args.memory_gib, "memory-gib")
    hours = _positive(args.hours, "hours")
    profile = str(args.slurm_profile.expanduser().resolve(strict=True)) if args.slurm_profile else None
    if args.adapter == "slurm" and not profile:
        raise ValueError("Slurm requires --slurm-profile; start from the neutral profile template.")
    systems = []
    if args.interactive:
        count = int(input("Number of systems [1]: ").strip() or "1")
        if count < 1:
            raise ValueError("At least one system is required")
        for index in range(count):
            system_id = input(f"System {index+1} name: ").strip()
            pdb = Path(input("PDB path: ").strip()).expanduser().resolve(strict=True)
            connectivity = Path(input("Bond topology path (PSF, PRMTOP/PARM7, bond JSON): ").strip()).expanduser().resolve(strict=True)
            interval = _positive(float(input("Time between saved frames, ps: ")), "frame interval")
            print("Enter one trajectory per independent replica, one path at a time. Blank line finishes the system.")
            trajectories = []
            while True:
                value = input(f"Replica {len(trajectories)+1} DCD path: ").strip()
                if not value:
                    break
                trajectories.append(str(Path(value).expanduser().resolve(strict=True)))
            systems.append(dict(system_id=system_id, pdb=str(pdb), connectivity=str(connectivity),
                                frame_interval_ps=interval, trajectories=trajectories))
    else:
        if not args.pdb or not args.connectivity or not args.trajectory or args.frame_interval_ps is None:
            raise ValueError("Supply --pdb, --connectivity, --trajectory, and --frame-interval-ps, or use --interactive.")
        systems.append(dict(system_id="system-1", pdb=str(args.pdb.expanduser().resolve(strict=True)),
            connectivity=str(args.connectivity.expanduser().resolve(strict=True)),
            frame_interval_ps=_positive(args.frame_interval_ps,"frame interval"),
            trajectories=[str(p.expanduser().resolve(strict=True)) for p in args.trajectory]))
    for system in systems:
        if not system["system_id"] or not system["trajectories"]:
            raise ValueError("Every system needs a name and at least one independent replica.")
    if len({s["system_id"] for s in systems}) != len(systems):
        raise ValueError("System names must be unique")
    config = default_analysis_config([m.module_id for m in MODULES], [])
    config["execution"].update(maximum_parallel_cpus=cpus, maximum_memory_gib=memory,
        maximum_hours_per_cpu=hours, maximum_total_cpu_hours=cpus*hours,
        submission_adapter=args.adapter, slurm_profile=profile)
    destination.mkdir(parents=True)
    _write_new(destination / "analysis-config.json", config)
    _write_new(destination / "study.json", {"study_schema":"salsbury-study-v1", "project_id":"my-study",
        "config":"analysis-config.json", "systems":systems})
    return {"technical_status":"complete", "study":str(destination / "study.json"),
            "next":f"salsbury-md-analysis doctor {shlex.quote(str(destination / 'study.json'))}",
            "jobs_submitted":False,"execution_started":False}


def _study(path):
    source = Path(path).expanduser().resolve(strict=True)
    value = _json(source)
    if not isinstance(value,dict) or value.get("study_schema") != "salsbury-study-v1":
        raise ValueError("Expected a study.json produced by init")
    if set(value)-{"study_schema","project_id","config","systems"}:
        raise ValueError("Study has unknown fields")
    systems = copy.deepcopy(value.get("systems"))
    if not isinstance(systems,list) or not systems:
        raise ValueError("Study needs a nonempty systems list")
    if not isinstance(value.get("project_id"), str) or not value["project_id"].strip():
        raise ValueError("Study needs a project_id")
    def resolve(raw):
        p = Path(raw).expanduser()
        return str((p if p.is_absolute() else source.parent / p).resolve(strict=True))
    for system in systems:
        if not isinstance(system, dict) or set(system) != {"system_id", "pdb", "connectivity", "frame_interval_ps", "trajectories"}:
            raise ValueError("Each system requires system_id, pdb, connectivity, frame_interval_ps, and trajectories")
        if not isinstance(system["system_id"], str) or not system["system_id"].strip():
            raise ValueError("System names must be nonempty strings")
        system["pdb"] = resolve(system["pdb"])
        system["connectivity"] = resolve(system["connectivity"])
        if not isinstance(system.get("trajectories"), list) or not system["trajectories"]:
            raise ValueError("Every system needs trajectories")
        system["trajectories"] = [resolve(p) for p in system["trajectories"]]
        _positive(system["frame_interval_ps"],"frame interval")
    if len({s["system_id"] for s in systems}) != len(systems):
        raise ValueError("System names must be unique")
    return source, value, systems, Path(resolve(value["config"]))


def diagnose(path=None):
    issues = []
    def issue(level, message, action):
        issues.append({"severity":level,"message":message,"action":action})
    system = platform.system()
    if system not in {"Linux","Darwin"}:
        issue("error","Native Windows execution is not supported.","Use the Linux installation inside WSL2.")
    if not (3,10) <= sys.version_info[:2] <= (3,12):
        issue("warning","This Python version is outside the tested 3.10–3.12 range.","Use the reviewed Python 3.12 environment.")
    config, systems = {}, []
    if path:
        path = Path(path).expanduser().resolve(strict=True)
        if path.is_dir():
            path = path / "analysis-config.json"
        value = _json(path)
        if value.get("study_schema"):
            _, _, systems, config_path = _study(path)
            config = _json(config_path)
        else:
            config = value
        from .analysis_config import load_analysis_config
        from .registry import MODULES
        config = load_analysis_config(config_path if value.get("study_schema") else path,
            [m.module_id for m in MODULES],list(config.get("views",{})))
    packages = {}
    for package, module in (("numpy","numpy"),("scipy","scipy"),("scikit-learn","sklearn"),("psutil","psutil"),("ijson","ijson")):
        available = importlib.util.find_spec(module) is not None
        packages[package] = available
        if not available:
            issue("error",f"Missing {package}.",
                  f"Install the documented runtime requirements ({package}).")
    executable = lambda name: shutil.which(name) or (str(Path(sys.executable).parent / name) if (Path(sys.executable).parent / name).is_file() else None)
    dssp = executable("mkdssp") or executable("dssp")
    if not dssp:
        issue("warning","DSSP is absent; protein secondary structure needs it.","Install dssp in the reviewed Conda environment or explicitly disable secondary_structure.")
    hdbscan_enabled = config.get("clustering",{}).get("methods",{}).get("hdbscan",{}).get("enabled",False)
    if hdbscan_enabled and importlib.util.find_spec("hdbscan") is None:
        issue("error","HDBSCAN is enabled but its package is absent.","Install the hdbscan extra or disable that method.")
    adapter = config.get("execution",{}).get("submission_adapter","local")
    if not executable("bash"):
        issue("error","The generated launchers need bash.","Install bash or use WSL2.")
    if adapter == "slurm":
        profile_path = config.get("execution", {}).get("slurm_profile")
        from .execution_adapters import load_slurm_profile
        profile_data = load_slurm_profile(Path(profile_path)) if profile_path else {}
        for command in (profile_data.get("submit_command", "sbatch"), profile_data.get("status_command", "squeue"), "sacct", "srun"):
            if not executable(command):
                issue("error",f"Slurm command {command} is unavailable.","Run on the cluster submission host or select local mode.")
        profile = config.get("execution",{}).get("slurm_profile")
        if not profile:
            issue("error","No Slurm site profile is configured.","Supply a site-specific Slurm profile.")
        else:
            from .execution_adapters import load_slurm_profile
            load_slurm_profile(Path(profile))
    for item in systems:
        from .preflight import probe_topology, probe_connectivity, probe_trajectory
        topology = probe_topology(Path(item["pdb"]))
        bonds = probe_connectivity(Path(item["connectivity"]))
        if topology["atom_count"] != bonds["atom_count"]:
            issue("error",f"{item['system_id']}: PDB and bond-topology atom counts differ.","Use matching files from the simulation; do not reorder atoms.")
        for trajectory in item["trajectories"]:
            header = probe_trajectory(Path(trajectory))
            if header.get("atom_count") != topology["atom_count"]:
                issue("error", f"{item['system_id']}: trajectory atom count differs: {trajectory}",
                      "Use a matching atom order and topology. A stripped trajectory needs a matching stripped topology.")
    return {"technical_status":"failed" if any(i["severity"]=="error" for i in issues) else "complete",
        "platform":system,"python":sys.version.split()[0],"adapter":adapter,"packages":packages,
        "dssp":dssp,"issues":issues,"next":"Review chemistry and replica grouping, then run plan. Diagnostics do not launch analysis."}


def prepare_study(path, output=None):
    source, study, systems, config_path = _study(path)
    result = diagnose(source)
    if result["technical_status"] != "complete":
        raise ValueError("Doctor found missing requirements. Run doctor on this study and fix the reported errors.")
    destination = Path(output).expanduser().resolve() if output else source.parent / "analysis"
    if len(systems)==1:
        from .quickstart import prepare_standard_analysis
        s = systems[0]
        report = prepare_standard_analysis(pdb_path=Path(s["pdb"]),psf_path=Path(s["connectivity"]),
            trajectories=[Path(p) for p in s["trajectories"]],frame_interval_ps=s["frame_interval_ps"],
            project_id=study["project_id"],output_directory=destination,config_path=config_path)
    else:
        from .comparative_quickstart import prepare_comparative_analysis
        # An immutable normalized request keeps relative-path interpretation independent of cwd.
        request = source.parent / ("comparison-input-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".json")
        _write_new(request,{"request_schema":"salsbury-comparative-analysis-input-v1","systems":systems})
        report = prepare_comparative_analysis(request_path=request,output_directory=destination,
            project_id=study["project_id"],config_path=config_path)
    return {"technical_status":report.get("technical_status","complete"),"output":str(destination),
        "execution_started":False,"jobs_submitted":False,
        "next":f"Read {destination / 'planning-report.md'}, then run: salsbury-md-analysis run {shlex.quote(str(destination))}"}


def workflow_status(root):
    from .accepted_artifacts import reports_complete
    root = Path(root).expanduser().resolve(strict=True)
    plan = _json(root / "local-execution-plan.json")
    tasks = [t for phase in plan["phases"] for t in phase["tasks"]]
    states = {}
    latest = {}
    for path in sorted((root / "local-execution-status").glob("*.json")):
        for phase in _json(path).get("phase_reports", []):
            for record in phase.get("tasks", []):
                latest[record.get("task_id")] = record
    for task in tasks:
        names = task.get("completion_reports",[])
        complete = bool(names) and reports_complete(root,names,task)
        present = [str(name) for name in names if (root / name).exists()]
        states[task["task_id"]] = {"task_id":task["task_id"],"module":task.get("module_id",task.get("command")),
            "state":"complete" if complete else "invalid_output" if present else "not_complete",
            "reports":list(names),"logs":str(root / "logs"), "blocking_tasks":[],
            "last_attempt":latest.get(task["task_id"])}
        attempt = latest.get(task["task_id"], {})
        if states[task["task_id"]]["state"] == "not_complete" and attempt.get("status") in {"failed", "timed_out"}:
            states[task["task_id"]]["state"] = attempt["status"]
    for task in tasks:
        row = states[task["task_id"]]
        if row["state"] != "complete":
            row["blocking_tasks"] = [dep for dep in task.get("depends_on_task_ids",[]) if states.get(dep,{}).get("state") != "complete"]
            if row["state"] == "not_complete" and row["blocking_tasks"]:
                row["state"] = "dependency_blocked"
    activity = campaign_activity(root)
    # Scheduler activity is not acceptance. A live job may have partial files.
    for job in activity["slurm_jobs"]:
        for task_id in job.get("task_ids", []):
            if task_id in states and states[task_id]["state"] != "complete":
                row = states[task_id]
                row["artifact_state"] = row["state"]
                row["state"] = "running" if job["state"] in {"RUNNING", "COMPLETING"} else "queued"
                row.setdefault("slurm_jobs", []).append(job)
    rows = list(states.values())
    counts = dict(Counter(row["state"] for row in rows))
    return {"technical_status":"complete" if counts.get("complete",0)==len(rows) else "incomplete",
        "root":str(root),"counts":counts,"tasks":rows,
        "activity":activity,
        "next":"Open prioritized_findings.md or the interactive report." if counts.get("complete",0)==len(rows)
               else "Check active jobs before recovery. Invalid outputs require diagnosis; resume never overwrites them."}


@contextmanager
def campaign_lock(root):
    """One entry-point lock shared by native local runs and reviewed submission."""
    import fcntl
    with (Path(root) / ".user-workflow.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("A run or submission already owns this campaign. Nothing was started.") from exc
        yield


def _queue_command(root):
    from .execution_adapters import load_slurm_profile
    profile_path = Path(root) / "slurm-profile.json"
    if not profile_path.is_file():
        raise ValueError("Prepared Slurm profile is missing; scheduler state is unknown.")
    return [load_slurm_profile(profile_path)["status_command"],
            "--me", "--noheader", "--format=%i|%T|%Z"]


def campaign_activity(root):
    """Report live activity separately from artifact acceptance."""
    activity = {"local_controller": "idle", "slurm_jobs": [], "scheduler_query": "not_applicable"}
    try:
        with campaign_lock(root):
            pass
    except ValueError:
        activity["local_controller"] = "active"
    config = _json(root / "analysis-config.json")
    if config.get("execution", {}).get("submission_adapter") != "slurm":
        return activity
    try:
        ledger = {}
        for path in sorted((root / "submission-ledgers").glob("*.tsv")):
            for line in path.read_text().splitlines()[1:]:
                fields = line.split("\t")
                if len(fields) == 2:
                    ledger.setdefault(fields[1].split(";")[0], set()).add(fields[0])
        queue = subprocess.run(_queue_command(root),
                               capture_output=True, text=True, check=True)
        activity["scheduler_query"] = "complete"
        for line in queue.stdout.splitlines():
            fields = line.split("|", 2)
            if len(fields) == 3 and Path(fields[2]).resolve() == root:
                activity["slurm_jobs"].append({"job_id": fields[0], "state": fields[1],
                    "task_ids": sorted(ledger.get(fields[0].split("_")[0], set()))})
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        activity["scheduler_query"] = "failed"
        activity["error"] = str(exc)
    return activity


def execute_workflow(root, *, resume=False, execute=True):
    resolved = Path(root).expanduser().resolve(strict=True)
    adapter = _json(resolved / "analysis-config.json")["execution"].get("submission_adapter", "local")
    if adapter == "slurm" and execute:
        with campaign_lock(resolved):
            return _execute_workflow(resolved, resume=resume, execute=execute)
    return _execute_workflow(resolved, resume=resume, execute=execute)


def _execute_workflow(root, *, resume=False, execute=True):
    from .execution_adapters import run_local_workflow
    root = Path(root).expanduser().resolve(strict=True)
    status = workflow_status(root)
    if status["technical_status"] == "complete":
        return dict(status,execution_started=False,jobs_submitted=False)
    if status["counts"].get("invalid_output"):
        raise ValueError("Existing output failed validation. Preserve it and diagnose the cause; automatic overwrite is forbidden.")
    config = _json(root / "analysis-config.json")
    adapter = config["execution"].get("submission_adapter","local")
    if resume and not execute:
        return dict(status,execution_started=False,jobs_submitted=False,
                    next=f"After checking active work: salsbury-md-analysis resume {shlex.quote(str(root))} --execute")
    if adapter == "custom":
        raise ValueError("Custom launchers own submission and recovery. Use launcher-contract.json and run-custom.sh.")
    if adapter == "slurm":
        # Unknown scheduler state must never result in duplicate submissions.
        queue = subprocess.run(_queue_command(root),capture_output=True,text=True,check=True)
        for line in queue.stdout.splitlines():
            fields = line.split("|",2)
            if len(fields)==3 and Path(fields[2]).resolve()==root:
                raise ValueError(f"Campaign still has Slurm job {fields[0]} ({fields[1]}). Nothing was submitted.")
        from .execution_adapters import load_slurm_profile, _slurm_resource_epochs, _slurm_submission_preview, _render_resource_bounded_submit
        plan = _json(root / "local-execution-plan.json")
        unfinished = {r["task_id"] for r in status["tasks"] if r["state"]!="complete"}
        reduced = copy.deepcopy(plan)
        for phase in reduced["phases"]:
            phase["tasks"] = [t for t in phase["tasks"] if t["task_id"] in unfinished]
            for task in phase["tasks"]:
                for key in ("depends_on_task_ids","wait_for_task_ids","resource_predecessor_task_ids"):
                    task[key] = [dep for dep in task.get(key,[]) if dep in unfinished]
        reduced["phases"] = [phase for phase in reduced["phases"] if phase["tasks"]]
        preview = _json(root / "slurm-submission-preview.json")
        if preview.get("generated_schedule_feasibility_status") != "feasible":
            raise ValueError("The prepared Slurm schedule is not feasible; replan before submission.")
        profile = load_slurm_profile(root / "slurm-profile.json")
        epochs = _slurm_resource_epochs(reduced,profile["partitions"],profile["partition_maximum_wall_minutes"],
            profile["partition_maximum_nodes"],profile["resource_policy"],plan.get("node_policy",{}))
        reviewed = _slurm_submission_preview(reduced, epochs, plan.get("node_policy", {}))
        if reviewed.get("generated_schedule_feasibility_status") != "feasible":
            raise ValueError("The unfinished-task schedule does not fit. Replan; no jobs were submitted.")
        script = root / ("reviewed-launch-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")+".sh")
        preview_path = script.with_suffix(".preview.json")
        _write_new(preview_path, reviewed)
        with script.open("x") as handle:
            handle.write(_render_resource_bounded_submit(root,profile,root / "slurm-profile.json",epochs,True,
                bool(config["execution"].get("autorecovery",True))).replace(
                    'PREVIEW="$ROOT/slurm-submission-preview.json"', f'PREVIEW={shlex.quote(str(preview_path))}'))
        result = subprocess.run(["bash",str(script)],cwd=root,check=False)
        return {"technical_status":"complete" if result.returncode==0 else "failed", "submission_exit_code":result.returncode,
                "jobs_submitted":True if result.returncode==0 else "possibly_partial_check_submission_ledger",
                "launch_script":str(script),"task_count":len(unfinished)}
    if adapter != "local":
        raise ValueError(f"Unsupported adapter: {adapter}")
    return run_local_workflow(root)


def run_command(args):
    try:
        if args.command=="init": result=initialize(args)
        elif args.command=="doctor": result=diagnose(args.path)
        elif args.command=="plan": result=prepare_study(args.study,args.output)
        elif args.command=="status": result=workflow_status(args.root)
        else: result=execute_workflow(args.root,resume=args.command=="resume",execute=getattr(args,"execute",True))
        if getattr(args,"json",False):
            print(json.dumps(result,indent=2))
        else:
            print(f"{args.command}: {result.get('technical_status','complete')}")
            for issue in result.get("issues",[]): print(f"{issue['severity'].upper()}: {issue['message']}\n  {issue['action']}")
            if "counts" in result:
                print(" | ".join(f"{k}: {v}" for k,v in result["counts"].items()))
                for task in result["tasks"]: print(f"{task['state']:20} {task['task_id']}"+(f" <- {', '.join(task['blocking_tasks'])}" if task["blocking_tasks"] else ""))
                print("Live activity: " + json.dumps(result.get("activity", {})))
            if args.command=="plan":
                report = Path(result["output"]) / "planning-report.md"
                if report.is_file(): print(report.read_text())
            if result.get("next"): print(result["next"])
        return 2 if result.get("technical_status")=="failed" else 0
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError) as exc:
        print(f"{args.command}: {exc}",file=sys.stderr)
        return 2
