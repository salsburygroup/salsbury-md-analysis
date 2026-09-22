"""Provisional workload models for generated preflight and reporting jobs.

These are execution overheads, not sampled scientific observations. Estimates
use metadata only: planning must not hash or reread trajectory coordinates.
"""

from pathlib import Path
import math
from typing import Mapping, Sequence

from .manifests import load_json, resolve_manifest_path
from .coordinate_cache import coordinate_cache_system_manifest_filename


MODEL_ID = "orchestration-workload-v1"


def manifest_workload(path: Path) -> dict:
    """Count reads, including duplicate paths the runtime hashes repeatedly."""
    document = load_json(path)
    reads = []
    largest_topology = 0
    replicas = 0
    for system in document["systems"]:
        for replica in system["replicas"]:
            replicas += 1
            for role in ("topology", "connectivity"):
                if replica.get(role):
                    source = resolve_manifest_path(replica[role], path)
                    size = source.stat().st_size
                    reads.append(size)
                    largest_topology = max(largest_topology, size)
            for segment in replica["segments"]:
                for role in ("trajectory", "weights"):
                    if segment.get(role):
                        source = resolve_manifest_path(segment[role], path)
                        reads.append(source.stat().st_size)
    return {
        "input_bytes_read": sum(reads), "input_file_reads": len(reads),
        "replica_count": replicas, "largest_topology_bytes": largest_topology,
        "source": "input_file_metadata",
    }


def _task(script: str, module: str, seconds: float, memory: float,
          stage: int, workload: Mapping, time_factor: float) -> dict:
    return {
        "task_id": f"orchestration:{script}", "workflow_id": "orchestration",
        "module_id": module, "task_scope": "orchestration_overhead",
        "execution_script": script, "dependency_stage": stage,
        "effective_cpu_cap": 1, "intrinsic_cpu_cap": 1,
        # The budget allocator requires positive work counts. This is ONE
        # invocation, explicitly not a frame or an independent replica.
        "source_frames_per_replica": [1], "minimum_frames_per_replica": 1,
        "maximum_frames_per_replica": 1, "required_integer_stride": 1,
        "work_unit": "job_invocation", "sampling_applicable": False,
        "cpu_seconds_per_physical_frame": 0.0,
        # Charge I/O elapsed as occupied single-core time conservatively.
        "fixed_cpu_hours": seconds * time_factor / 3600.0,
        "estimated_peak_memory_gib": memory,
        "memory_size_scaling_applied": True,
        "measured_memory_observation_scaling_eligible": False,
        "priority_weight": 1.0, "member_observation_multiplier": 1,
        "calibration_status": "provisional_workload_model",
        "calibration_id": MODEL_ID,
        "resource_model": {
            "model_id": MODEL_ID, "workload": dict(workload),
            "estimated_unpadded_wall_seconds": seconds,
            "time_safety_factor": time_factor,
            "independently_validated": False,
            "cluster_memory_padding_included": False,
        },
    }


def orchestration_tasks(root: Path, views: Sequence[Path],
                        analysis_tasks: Sequence[Mapping], config: Mapping,
                        *, time_safety_factor: float,
                        maximum_atom_count: int = 85_206,
                        coordinate_cache_enabled: bool = False) -> list[dict]:
    """Generate one protected overhead row per generated orchestration script."""
    manifests = {}
    for view in views:
        source_name = str(load_json(view)["system_manifest"])
        workload = manifest_workload(root / source_name)
        name = source_name
        if coordinate_cache_enabled:
            systems = load_json(root / source_name)["systems"]
            cache_name = ("system-cache.json" if source_name == "system.json" else
                          coordinate_cache_system_manifest_filename(systems[0]["system_id"]))
            cache_input = config.get("execution", {}).get("coordinate_cache_input")
            cache_root = Path(cache_input) if cache_input else Path("coordinate-cache")
            name = str(cache_root / cache_name)
            existing = root / name
            if cache_input and existing.is_file():
                workload = manifest_workload(existing)
            else:
                # Views are rewritten to future caches after planning. Account
                # for their real script identities now; bound bytes by full
                # source coverage rather than treating absent caches as empty.
                view_id = view.stem.removeprefix("project-")
                frames = max((sum(t["source_frames_per_replica"])
                              for t in analysis_tasks if t.get("workflow_id") == view_id), default=0)
                cache_bytes = (16 * maximum_atom_count * frames
                               + 512 * maximum_atom_count * workload["replica_count"])
                workload["input_bytes_read"] = max(workload["input_bytes_read"], cache_bytes)
                workload["largest_topology_bytes"] = max(
                    workload["largest_topology_bytes"], 256 * maximum_atom_count)
                workload["source"] = "full_source_coverage_cache_upper_bound"
        if name != "system.json":
            manifests[name] = workload
    workloads = [("run_preflight.slurm", manifest_workload(root / "system.json"), 0)]
    workloads.extend((f"run_view_preflight_{i}.slurm", manifests[name], 2)
                     for i, name in enumerate(sorted(manifests)))
    result = []
    for script, workload, stage in workloads:
        # 64 MiB/s is a conservative shared-storage planning allowance, not a
        # claimed fitted bandwidth. Include startup, stat and topology probes.
        seconds = (60.0 + workload["input_bytes_read"] / (64 * 1024**2)
                   + 2.0 * workload["input_file_reads"]
                   + 2.0 * workload["replica_count"])
        # Hashing is chunked. Bound the additional large-file working-set
        # allowance, but do not cap topology/parser memory for large systems.
        memory = (0.5 + min(3.5, workload["input_bytes_read"] / 1024**3)
                  + 12 * workload["largest_topology_bytes"] / 1024**3)
        result.append(_task(script, "workflow_preflight", seconds, memory,
                            stage, workload, time_safety_factor))
    reporting = config.get("reporting", {})
    picker = reporting.get("finding_picker_enabled", True)
    # Sequential alternative-clustering fits publish one bundled report.
    # Counting every fit as a full report inflated reporting memory estimates.
    count = len({t.get("execution_bundle_id", t.get("task_id", str(i)))
                 for i, t in enumerate(analysis_tasks)})
    atom_scale = max(0.1, math.sqrt(maximum_atom_count / 85_206.0))
    workload = {"planned_analysis_task_count": len(analysis_tasks),
                "planned_report_bundle_count": count, "view_count": len(views),
                "finding_picker_enabled": bool(picker),
                "maximum_atom_count": maximum_atom_count}
    seconds = 60.0 + 2.0 * count
    memory = 1.0
    if picker:
        # Includes native presentation artifacts and finding extraction, NOT
        # the separate interactive repository or scientific state exports.
        seconds += 120.0 + 15.0 * count + 30.0 * len(views)
        memory = 1.0 + (0.25 * count + 0.5 * len(views)) * atom_scale
    final_stage = max((int(t["dependency_stage"]) for t in analysis_tasks), default=2) + 1
    if len(load_json(root / "system.json")["systems"]) > 1:
        result.append(_task("run_reporting_integrated_comparison.slurm",
                            "workflow_integrated_reporting", 60.0 + 3.0 * count,
                            1.0 + 0.05 * count * atom_scale,
                            final_stage, workload, time_safety_factor))
        final_stage += 1
    result.append(_task("run_finalize_reporting.slurm", "workflow_final_reporting",
                        seconds, memory, final_stage,
                        workload, time_safety_factor))
    return result
