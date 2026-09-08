"""Publish checksum companions for derived reports without changing their bytes."""
from __future__ import annotations

import json
from pathlib import Path

from .accepted_artifacts import validate_complete_report
from .finding_picker import finding_sidecar_evidence
from .manifests import sha256_file


def write_derived_report_sidecar(report_path: Path, source_paths) -> Path:
    path = Path(report_path).resolve(strict=True)
    report = validate_complete_report(path)
    if report.get("module_id") not in {"rmsf_permutation_inference", "integrated_comparison"}:
        raise ValueError("derived publication only supports RMSF inference and integrated comparison")
    sources = sorted({Path(value).resolve(strict=True) for value in source_paths})
    if not sources or path in sources:
        raise ValueError("derived publication requires separate source files")
    report_records, input_records = [], []
    for source in sources:
        record = {"path": str(source), "sha256": sha256_file(source)}
        if source.name == "report.json":
            validate_complete_report(source)
            report_records.append(record)
        else:
            input_records.append(record)
    payload = {
        "sidecar_schema": "salsbury-derived-report-sidecar-v1",
        "technical_status": "complete", "module_id": report["module_id"],
        "report_path": str(path), "report_sha256": sha256_file(path),
        "report_size_bytes": path.stat().st_size,
        "source_report_records": report_records, "input_records": input_records,
        "resource_accounting": "derived reporting; source analysis measurements are counted separately",
        "finding_evidence": finding_sidecar_evidence(report, path),
    }
    sidecar = Path(str(path) + ".summary.json")
    if sidecar.exists():
        if json.loads(sidecar.read_text()) != payload:
            raise ValueError("existing derived sidecar differs; refusing overwrite")
    else:
        with sidecar.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    validate_complete_report(path, require_sidecar=True)
    return sidecar


if __name__ == "__main__":
    import sys
    write_derived_report_sidecar(Path(sys.argv[1]), sys.argv[2:])
