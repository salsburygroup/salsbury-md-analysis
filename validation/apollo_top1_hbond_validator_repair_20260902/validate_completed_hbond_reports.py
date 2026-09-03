#!/usr/bin/env python3
"""Validate and install completed hydrogen-bond reports without recomputation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from salsbury_md_analysis.frame_sampling import integer_stride_selected_count


EXPECTED_SETTINGS = {
    "candidate_harmonization": "intersection_by_atom_identity_v2",
    "chemistry_policy": "automatic_topology_templates_v1",
    "interaction_scope": "all_solute",
    "output_mode": "sparse_spatial_observed_union_v3",
    "water_policy": "exclude",
}

EXPECTED_CUTOFFS = {
    ("primary", 3.0, 150.0),
    ("sensitivity_da3_angle120", 3.0, 120.0),
    ("sensitivity_da3_angle135", 3.0, 135.0),
    ("sensitivity_da3.2_angle120", 3.2, 120.0),
    ("sensitivity_da3.2_angle135", 3.2, 135.0),
    ("sensitivity_da3.2_angle150", 3.2, 150.0),
    ("sensitivity_da3.5_angle120", 3.5, 120.0),
    ("sensitivity_da3.5_angle135", 3.5, 135.0),
    ("sensitivity_da3.5_angle150", 3.5, 150.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_if_absent_or_identical(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise SystemExit(f"existing destination differs: {destination}")
        return
    os.link(source, destination)


def expected_members(system_ids: list[str]) -> set[tuple[str, str]]:
    return {
        (system_id, f"replica-{replica}")
        for system_id in system_ids
        for replica in range(1, 4)
    }


parser = argparse.ArgumentParser()
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
parser.add_argument("--expected-report-sha256", required=True)
parser.add_argument("--expected-summary-sha256", required=True)
parser.add_argument("--expected-project-sha256", required=True)
parser.add_argument("--expected-stride", type=int, required=True)
parser.add_argument("--expected-systems", required=True)
parser.add_argument("--install-report", type=Path, action="append", required=True)
parser.add_argument("--install-summary", type=Path, action="append", required=True)
args = parser.parse_args()

if len(args.install_report) != len(args.install_summary):
    raise SystemExit("report and summary destination counts differ")
if sha256(args.report) != args.expected_report_sha256:
    raise SystemExit("completed report hash differs from the recorded failed-job artifact")
if sha256(args.summary) != args.expected_summary_sha256:
    raise SystemExit("completed summary hash differs from the recorded failed-job artifact")

report = json.loads(args.report.read_text(encoding="utf-8"))
summary = json.loads(args.summary.read_text(encoding="utf-8"))
systems = [item for item in args.expected_systems.split(",") if item]
members = expected_members(systems)

if report.get("technical_status") != "complete":
    raise SystemExit("report is not technically complete")
if report.get("module_id") != "hydrogen_bond_discovery":
    raise SystemExit("report module is not hydrogen_bond_discovery")
if report.get("scientific_status") != "not evaluated":
    raise SystemExit("report unexpectedly claims a scientific status")
if int(report.get("error_count", -1)) != 0:
    raise SystemExit("report contains errors")
if int(report.get("warning_count", -1)) != 0:
    raise SystemExit("report contains warnings")
if report.get("content_hashes_included") is not True:
    raise SystemExit("report does not include content hashes")
if report.get("project_manifest_sha256") != args.expected_project_sha256:
    raise SystemExit("project-manifest hash differs from the frozen request")
if summary.get("technical_status") != "complete":
    raise SystemExit("summary is not technically complete")
if summary.get("module_id") != "hydrogen_bond_discovery":
    raise SystemExit("summary module is not hydrogen_bond_discovery")
if summary.get("report_sha256") != sha256(args.report):
    raise SystemExit("summary report hash does not match the report")
if int(summary.get("report_size_bytes", -1)) != args.report.stat().st_size:
    raise SystemExit("summary report size does not match the report")

settings = report.get("settings")
selection = report.get("frame_selection")
if not isinstance(settings, dict) or not isinstance(selection, dict):
    raise SystemExit("report lacks settings or frame-selection metadata")
for key, value in EXPECTED_SETTINGS.items():
    if settings.get(key) != value:
        raise SystemExit(f"hydrogen-bond setting differs: {key}")
if int(settings.get("frame_stride", -1)) != 1:
    raise SystemExit("legacy frame_stride is not fixed at one")
configured = settings.get("frame_selection")
if not isinstance(configured, dict):
    raise SystemExit("settings lack frame-selection configuration")
if configured.get("mode") != "integer_stride_per_replica_v1":
    raise SystemExit("frame-selection mode differs")
if int(configured.get("stride", -1)) != args.expected_stride:
    raise SystemExit("configured stride differs")
if int(selection.get("resolved_integer_stride", -1)) != args.expected_stride:
    raise SystemExit("resolved stride differs")

observed_cutoffs = {
    (
        str(row.get("cutoff_id")),
        float(row.get("maximum_donor_acceptor_distance_angstrom")),
        float(row.get("minimum_donor_hydrogen_acceptor_angle_degrees")),
    )
    for row in settings.get("cutoff_definitions", [])
}
if observed_cutoffs != EXPECTED_CUTOFFS:
    raise SystemExit("hydrogen-bond cutoff definitions differ")

replicas = selection.get("replicas")
if not isinstance(replicas, list):
    raise SystemExit("report lacks replica frame selections")
observed_members = {
    (str(row.get("system_id")), str(row.get("replica_id"))) for row in replicas
}
if observed_members != members or len(replicas) != len(members):
    raise SystemExit("system/replica coverage is incomplete")

expected_total = 0
for row in replicas:
    source_count = int(row.get("source_frame_count", -1))
    expected_count = integer_stride_selected_count(source_count, args.expected_stride)
    observed_count = int(row.get("selected_frame_count", -1))
    if observed_count != expected_count:
        raise SystemExit("per-replica count differs from the complete-interval generator")
    spacing = row.get("selection_spacing")
    if not isinstance(spacing, dict):
        raise SystemExit("replica lacks selection-spacing metadata")
    expected_last = (expected_count - 1) * args.expected_stride
    expected_spacing = {
        "kind": "exact_integer_stride",
        "last_source_frame_forced": False,
        "maximum_source_frame_gap": args.expected_stride,
        "minimum_source_frame_gap": args.expected_stride,
        "starts_at_replica_frame_zero": True,
    }
    for key, value in expected_spacing.items():
        if spacing.get(key) != value:
            raise SystemExit(f"replica spacing differs: {key}")
    segments = row.get("segments")
    if not isinstance(segments, list) or not segments:
        raise SystemExit("replica lacks segment frame-selection metadata")
    if sum(int(segment.get("selected_frame_count", -1)) for segment in segments) != expected_count:
        raise SystemExit("segment counts do not sum to the complete-interval count")
    if int(segments[0].get("first_selected_source_frame_index", -1)) != 0:
        raise SystemExit("replica selection does not start at frame zero")
    if int(segments[-1].get("last_selected_source_frame_index", -1)) != expected_last:
        raise SystemExit("replica last frame differs from the complete-interval generator")
    expected_total += expected_count

if int(selection.get("selected_frame_count", -1)) != expected_total:
    raise SystemExit("frame-selection total differs from complete-interval counts")
if int(report.get("evaluated_frame_count", -1)) != expected_total:
    raise SystemExit("evaluated-frame total differs from complete-interval counts")
for key in (
    "conceptual_candidate_count",
    "candidate_count",
    "materialized_observed_candidate_count",
    "present_event_count",
    "explicit_geometry_evaluation_count",
    "spatial_neighbor_pair_count",
):
    if int(report.get(key, -1)) <= 0:
        raise SystemExit(f"report has invalid {key}")

for report_destination, summary_destination in zip(
    args.install_report, args.install_summary, strict=True
):
    link_if_absent_or_identical(args.report, report_destination)
    link_if_absent_or_identical(args.summary, summary_destination)

print(
    json.dumps(
        {
            "technical_status": "complete",
            "scientific_status": "not evaluated",
            "evaluated_frame_count": expected_total,
            "expected_systems": systems,
            "expected_stride": args.expected_stride,
            "project_manifest_sha256": args.expected_project_sha256,
            "report_sha256": sha256(args.report),
            "summary_sha256": sha256(args.summary),
            "installed_reports": [str(path) for path in args.install_report],
            "installed_summaries": [str(path) for path in args.install_summary],
        },
        indent=2,
        sort_keys=True,
    )
)
