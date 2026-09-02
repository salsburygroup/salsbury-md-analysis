#!/usr/bin/env python3
"""Validate independent runtime observations against planner CPU bounds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from salsbury_md_analysis.planner_calibration_models import (
    validate_runtime_holdouts,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("holdouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve(strict=True)
    holdout_path = args.holdouts.expanduser().resolve(strict=True)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    holdouts = json.loads(holdout_path.read_text(encoding="utf-8"))
    holdouts["content_sha256"] = _sha256(holdout_path)
    result = validate_runtime_holdouts(model, holdouts)
    result["model_sha256"] = _sha256(model_path)

    output = args.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "technical_status": result["technical_status"],
        "scientific_status": result["scientific_status"],
        "output": str(output),
        "model_sha256": result["model_sha256"],
        "holdout_evidence_sha256": result["holdout_evidence_sha256"],
        "point_count": result["point_count"],
        "maximum_observed_to_planning_upper_ratio": result[
            "maximum_observed_to_planning_upper_ratio"
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
