#!/usr/bin/env python3
"""Extend a hash-bound resource catalog with censored timeout evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from salsbury_md_analysis.resource_calibrations import (
    build_resource_calibration_catalog,
    redact_resource_calibration_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-catalog", action="append", type=Path, default=[])
    parser.add_argument("--timeout-record", action="append", type=Path, default=[])
    parser.add_argument("--sidecar", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--redact-source-paths", action="store_true")
    args = parser.parse_args()
    catalog = build_resource_calibration_catalog(
        args.sidecar,
        timeout_records=args.timeout_record,
        base_catalogs=args.base_catalog,
    )
    if args.redact_source_paths:
        catalog = redact_resource_calibration_catalog(catalog)
    args.output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
