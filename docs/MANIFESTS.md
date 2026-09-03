# Manifests and provenance

Status: **experimental**

The manifest layer records technical lineage for molecular-dynamics projects.
It does not claim that a trajectory is readable, equilibrated, converged, or
scientifically valid.

## Manifest types

- **Project**: analysis profile, system manifest, output root, sampling mode,
  requested modules, and protected locations.
- **System**: systems, replicas, topologies, continuous trajectory segments,
  mandatory per-segment physical timing, and optional statistical-weight files.
- **Output**: module statuses, output paths and SHA-256 values, technical
  status, scientific status, warnings, and limitations.
- **Analysis lock**: exact suite/project commits, environment identity,
  manifest hashes, data roots, owners, review state, limitations, and the
  optional frame-budget sensitivity policy/status/evidence or skip rationale.
  Replica diagnostics are recorded separately and default to off.

`frame_budget_sensitivity` records `off`, `recommend`, or `require` separately
from `completed`, `skipped`, `unavailable`, `planned`, or `not_applicable`.
Completed checks retain report hashes; skipped and unavailable checks require a
rationale. The field records a publication decision but does not turn a
budget-sensitivity comparison into proof of trajectory convergence. Publication
locks may use `off` with `not_applicable`; the B-versus-2B comparison is not a
mandatory publication gate unless the project owner explicitly selects
`require`.

`replica_diagnostics` records `off` or `optional`. When selected, it is
exploratory and may indicate that additional independent simulations could be
useful for a particular estimator. Replica agreement and leave-one-replica-out
are not scientific acceptance gates.

The JSON Schema files under `schemas/` define interchange structure. The Python
validator adds semantic checks that JSON Schema alone does not conveniently
express.

## Validation

```bash
PYTHONPATH=src python3 -m salsbury_md_analysis \
  validate-manifest system path/to/system.json --check-paths
```

Use `--json` for a machine-readable report. A valid manifest exits with status
0; invalid JSON or semantic failures exit with status 2.

Validation rejects:

- duplicate JSON keys and unknown fields;
- missing or empty required identifiers;
- duplicate system, replica, segment, requested-module, or output-module IDs;
- unknown suite module IDs;
- a first trajectory segment marked continuous with a nonexistent predecessor;
- a DCD header-step policy other than `continuous` or `reset_per_segment`;
- missing, non-finite, or nonpositive segment timing;
- malformed SHA-256 values or Git commits;
- relative authoritative data roots and protected locations;
- analysis output roots that overlap a protected location;
- missing or non-file references when `--check-paths` is supplied.

For output manifests, `--check-paths` also recomputes every recorded SHA-256 and
fails on a mismatch.

Relative file paths are resolved from the manifest's directory, not the shell's
current directory.

Coordinate-analysis project manifests must also declare
`periodic_coordinate_policy`. `reject` blocks periodic frames;
`allow_wrapped_diagnostic` is diagnostic only; `make_whole` and
`unwrap_continuous` perform connectivity-aware in-suite reconstruction. The
latter policies require explicit reconstruction gates plus a connectivity path
for every replica. See [Periodic coordinate reconstruction](PERIODIC_COORDINATES.md).

## Read-only input inventory

```bash
PYTHONPATH=src python3 -m salsbury_md_analysis \
  inventory-system path/to/system.json --hash-content > input-inventory.json
```

The command validates that all topology, connectivity, trajectory, and weight paths exist and
are regular files. It prints a deterministic JSON inventory with resolved paths,
sizes, modification times, and streamed SHA-256 values when `--hash-content` is
used. It never edits an input.

Hashing trajectories can be expensive. Without `--hash-content`, the inventory
still hashes the small manifest itself but records `null` content hashes for its
referenced files.

## Scientific boundary

File presence, metadata, and byte hashes demonstrate identity and technical
lineage only. The future preflight module must separately inspect topology and
trajectory formats, atom counts, frames, boxes, duration, ordering, overlap,
and continuity. Later modules must independently evaluate equilibration,
sampling, uncertainty, population meaning, and scientific interpretation.
