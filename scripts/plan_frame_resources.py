#!/usr/bin/env python3
"""Plan all-frame or reported balanced subsampling from a retained pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from salsbury_md_analysis.resource_planning import (
    calibrate_from_benchmarks,
    recommend_frame_budget,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path, nargs="+")
    parser.add_argument("--evaluated-frame-count", type=int, action="append")
    parser.add_argument("--total-source-frames", type=int, required=True)
    parser.add_argument("--replica-count", type=int, required=True)
    parser.add_argument("--target-wall-hours", type=float, default=4.0)
    parser.add_argument("--target-memory-gib", type=float, default=16.0)
    parser.add_argument("--minimum-frames-per-replica", type=int, default=100)
    parser.add_argument(
        "--sensitivity-check-policy",
        choices=("off", "recommend", "require"),
        default="off",
        help=(
            "Optional: off, nonblocking recommend, or explicit project-owner "
            "require; the planner never schedules the comparison."
        ),
    )
    parser.add_argument("--calibration-id")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    benchmarks = [
        json.loads(path.read_text(encoding="utf-8")) for path in arguments.benchmark
    ]
    calibration = calibrate_from_benchmarks(
        benchmarks,
        evaluated_frame_counts=arguments.evaluated_frame_count,
        calibration_id=arguments.calibration_id,
    )
    plan = recommend_frame_budget(
        calibration,
        total_source_frames=arguments.total_source_frames,
        replica_count=arguments.replica_count,
        target_wall_seconds=arguments.target_wall_hours * 3600.0,
        target_memory_mib=arguments.target_memory_gib * 1024.0,
        minimum_frames_per_replica=arguments.minimum_frames_per_replica,
        sensitivity_check_policy=arguments.sensitivity_check_policy,
    )
    payload = {"calibration": calibration, "plan": plan}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
