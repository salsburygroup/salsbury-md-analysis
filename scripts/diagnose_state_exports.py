"""Bounded, read-only export preparation profiler; never writes scientific outputs.

Run in the task's existing saved-report environment. Requires a new output
directory outside the study's analysis/output and protected input locations.
The deadline stops diagnostics, not scientific jobs, and is not a runtime estimate.
"""

import argparse
import cProfile
import json
import math
import os
import pstats
import resource
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from salsbury_md_analysis.manifests import load_json, resolve_manifest_path, sha256_file
from salsbury_md_analysis.periodic import PeriodicFrameProcessor
from salsbury_md_analysis.state_coordinate_exports import state_coordinate_exports_project
from salsbury_md_analysis.upstream_cache import _ENVIRONMENT_VARIABLES


def run(project_path, output_directory, maximum_seconds):
    if not math.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be finite and positive")
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("bounded profiling requires POSIX interval timers")
    project_path = Path(project_path).resolve(strict=True)
    output = Path(output_directory).resolve()
    project = load_json(project_path)
    protected = [resolve_manifest_path(str(project["analysis_output_root"]), project_path)]
    protected.extend(resolve_manifest_path(str(p), project_path) for p in project.get("protected_locations", []))
    # Keep diagnostic evidence outside the existing study tree, even if no
    # protected_locations were declared. Never make a directory in a science root.
    protected.append(project_path.parent)
    if any(output == p or p in output.parents for p in protected):
        raise ValueError("diagnostic output must be outside the study and protected locations")
    output.mkdir(parents=False, exist_ok=False)
    profile = cProfile.Profile()
    started, cpu_started = time.perf_counter(), time.process_time()
    phases, policies, processed = [], Counter(), Counter()
    receipt = {"diagnostic_schema": "state-export-timing-v1", "scientific_status": "not_evaluated", "scientific_results_saved": False, "coordinate_files_written": 0, "maximum_seconds": maximum_seconds}

    def phase(name):
        phases.append({"phase": name, "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started})

    original_factory = PeriodicFrameProcessor.from_replica
    original_process = PeriodicFrameProcessor.process

    def factory(*args, **kwargs):
        processor = original_factory(*args, **kwargs)
        policies[processor.policy] += 1
        return processor

    def process(self, *args, **kwargs):
        processed[self.policy] += 1
        return original_process(self, *args, **kwargs)

    def expired(signum, frame):
        raise TimeoutError("bounded diagnostic deadline reached; no exports were written")

    previous_handler = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("an existing interval timer prevents bounded profiling")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, maximum_seconds)
    try:
        profile.enable()
        phase("diagnostic_identity_hashing")
        receipt["project_path"] = str(project_path)
        receipt["project_sha256"] = sha256_file(project_path)
        source = project["definitions"]["state_coordinate_exports"]["source"]
        declared = os.environ.get(_ENVIRONMENT_VARIABLES[source])
        if declared:
            source_path = Path(declared).expanduser().resolve(strict=True)
            receipt["source_report_path"] = str(source_path)
            receipt["source_report_sha256"] = sha256_file(source_path)
        with patch.object(PeriodicFrameProcessor, "from_replica", side_effect=factory), patch.object(PeriodicFrameProcessor, "process", process):
            receipt["result"] = state_coordinate_exports_project(project_path, hash_content=True, diagnostic_only=True, phase_observer=phase)
        receipt["technical_status"] = "complete"
    except Exception as exc:
        receipt.update(technical_status="timeout" if isinstance(exc, TimeoutError) else "error", error_type=type(exc).__name__, error=str(exc))
    finally:
        profile.disable()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        phase("diagnostic_stopped")
        receipt.update(phases=phases, effective_replica_processor_policies=dict(policies), processed_frames_by_policy=dict(processed), wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu_started)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt["process_peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
        receipt["memory_scope"] = "whole diagnostic process high-water mark, including imports"
        receipt["limitations"] = ["No export-write timing", "Profiling adds overhead", "A timeout is a lower bound, not a completed runtime estimate", "No scientific acceptance implied"]
        profile.dump_stats(str(output / "profile.pstats"))
        with (output / "profile.txt").open("x") as stream:
            pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(60)
        (output / "timing.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=float, required=True)
    args = parser.parse_args()
    result = run(args.project, args.output_directory, args.maximum_seconds)
    print(json.dumps({"technical_status": result["technical_status"], "evidence": str(args.output_directory)}))
    raise SystemExit(0 if result["technical_status"] == "complete" else 1)
