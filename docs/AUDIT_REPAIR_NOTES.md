# Audit repairs awaiting release

These changes are on a repair branch. They are not changes to an existing
release tag.

The planner now schedules indivisible replica workers in waves and adds serial
work separately. The local runner dispatches ready tasks across the dependency
graph instead of waiting for an entire level to finish. Memory planning uses
the configured task multiplier and one reservation per node. Generated
resource-token schedules reserve a conservative upper bound on occupied nodes;
they report that allowance separately. Local reservations are estimates, not
an operating-system memory limit.

Completion and reuse checks verify module identity, project provenance,
report/summary checksums, and declared companion files. Coordinate-cache reuse
also verifies cached topology, connectivity, and trajectory checksums.
Original source identities use the recorded content hash when present, or the
recorded size and modification time otherwise. Checking large files can add
I/O time. These checks do not establish scientific validity.

Downstream reports now record their own project checksum and preserve the
upstream project's checksum separately. Final resource and finding summaries
include output-file hashes and source-report records. Older final summaries
without this evidence are not accepted as newly validated output. Preserve
them and generate a new reporting snapshot; do not overwrite accepted work.

Resource measurement runs in an isolated supervisor. CPU accounting excludes
earlier, unrelated child processes. Memory records distinguish sampled
process-tree RSS from the largest child's RSS. If the operating system blocks
process inspection, collection falls back to the latter. Neither measurement
is used to lower a memory model without explicit qualification.

The picker no longer fills category quotas or forces ten headlines. It ranks
effects within their comparison families and uses corrected statistical
support where available. The documentation explains the eligibility rules and
their limits; a within-family percentile is not biological importance.

PCA, tICA, convergence, and feature-distribution figures plot named scientific
quantities. Generic numeric diagnostics no longer satisfy primary-figure
coverage. An explicitly unavailable result gets a reason card and table, not
an invented quantity.

## Verification and remaining scope

The repair audit includes the full unit suites, planner boundary cases,
installed NEMO execution, main-result reuse by experimental, and offline viewer
tests. Optional OpenMM connectivity is tested separately and remains optional.
The NEMO calculations are software acceptance tests, not new scientific claims.

Live multi-node Slurm replay, cumulative retry accounting across separate
allocations, and fresh end-to-end trajectories for every biomolecular class
remain separate validation work. Existing synthetic and module tests do not
replace those runs. Do not describe this audit as proof of every estimator or
every cluster configuration.

Run the shared-contract check when synchronizing the core branches:

```bash
python scripts/check_shared_core.py \
  --main-source /path/to/main/src/salsbury_md_analysis \
  --experimental-source /path/to/experimental/src/salsbury_md_analysis
```

It checks the audited shared implementations while allowing the named
experimental dispatch additions. It is not a claim that every file in the
branches should be identical.
