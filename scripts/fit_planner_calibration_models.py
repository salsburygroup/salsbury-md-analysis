#!/usr/bin/env python3
"""Fit planner CPU models from a verified three-size evidence matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from salsbury_md_analysis.planner_calibration_models import (
    fit_size_length_models,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-safety-factor", type=float, default=1.5)
    args = parser.parse_args()
    result = fit_size_length_models(
        args.evidence, residual_safety_factor=args.residual_safety_factor
    )
    destination = args.output.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "technical_status": "complete",
        "scientific_status": "runtime evidence only",
        "output": str(destination),
        "source_evidence_sha256": result["source_evidence_sha256"],
        "modules": sorted(result["models"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
