"""Isolated resource accounting for one command and its descendants.

This supervisor forwards stdout/stderr without buffering scientific payloads.
RSS sampling is a lower bound, never a qualified memory replacement model.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    destination = Path(arguments.pop(0))
    if arguments and arguments[0] == "--":
        arguments.pop(0)
    started = time.monotonic()
    child = subprocess.Popen(arguments)
    peak = 0
    samples = 0
    try:
        import psutil
    except ImportError:
        psutil = None
    while child.poll() is None:
        if psutil is not None:
            try:
                parent = psutil.Process(child.pid)
                processes = [parent, *parent.children(recursive=True)]
                resident = 0
                for process in processes:
                    try:
                        resident += process.memory_info().rss
                    except (psutil.Error, OSError):
                        pass
                peak = max(peak, resident)
                samples += 1
            except (psutil.Error, OSError):
                pass
        time.sleep(0.02)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    largest_child_bytes = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    payload = {
        "measurement_schema": "salsbury-isolated-command-resources-v2",
        "measurement_available": True,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "total_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "wall_seconds": time.monotonic() - started,
        "maximum_resident_memory_mib": max(peak, largest_child_bytes) / (1024 ** 2),
        "sampled_process_tree_peak_mib": peak / (1024 ** 2) if samples else None,
        "largest_child_peak_mib": largest_child_bytes / (1024 ** 2),
        "memory_sample_count": samples,
        "memory_sample_interval_seconds": .02 if samples else None,
        "memory_measurement_scope": "sampled_process_tree_and_largest_child" if samples else "largest_child_only",
        "memory_replacement_qualified": False,
        "measurement_scope": "isolated invocation; child CPU includes reaped descendants; RSS scope is explicit",
        "child_exit_code": child.returncode,
    }
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
