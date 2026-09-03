# Finalization acceptance record — 2026-09-02

This record separates software readiness from scientific acceptance. It covers
the current release branch, the bounded TOP1 equivalence test, the TBA recovery
chain, the TOP1 hydrogen-bond installation repair, and the Apollo planning
sweeps. It does not approve a biological conclusion.

## Software release gate

Status: **pass for branch integration**.

- Live GitHub refs at the gate were `main` at
  `3fdf0bc2296c95679c0bbe17dd28ca9516358174` and `experimental` at
  `84438d8d938cd1953c3d1edfa4eb86b7dda77fb2`.
- The 14 hydrogen-bond redesign commits and the five later stride and
  ion-atmosphere commits were already present on live `main` under merged
  commit IDs. Direct diffs of `frame_sampling.py`, `ion_atmosphere.py`, its
  tests, and the frame-sampling documentation against the TOP1 worktree were
  empty. The worktree branch named `main` was not pushed.
- The full local suite ran 680 tests with zero failures and one expected skip.
- The hydrogen-bond redesign retains protein–protein, protein–DNA, and DNA–DNA
  strata; per-system chemistry; cross-system identity matching; periodic
  minimum-image geometry; sparse implicit-zero accounting; and closed failure
  on incompatible chemistry.
- Ion-atmosphere analysis uses exact local minimum-image distances on only the
  planner-selected frames. It does not request continuous full-system
  unwrapping. Structural analyses keep their own reconstruction policies.
- Structural QC runs one worker per replica. Its frame count remains an
  explicit scientific choice; resource planning does not reward unnecessary
  full-frame analysis.
- Task recovery is enabled by default, bounded by policy, can be disabled, and
  retains every failed attempt and terminal diagnostic.

## TOP1 hydrogen-bond equivalence and production evidence

The bounded scalar-versus-spatial gate passed before production: 36 immutable
beginning, middle, and end frames; 2,487 observed identities; and 22,000
present events matched with zero identity, cutoff-mask, distance, or angle
differences.

Production jobs `8298182` and `8298183` completed their calculations but their
external wrapper rejected valid complete-interval counts. The saved reports
were technically complete, had zero errors and zero warnings, and retained
these exact counts:

| Report | Stride | Replicas | Frames per replica | Total frames | SHA-256 |
|---|---:|---:|---:|---:|---|
| TOP1 D | 15 | 12 | 1,333 | 15,996 | `52523fcf0ff5bad8bcad33c4dea2f1f6e4d795e529d52bed7e239c8a88bb92ef` |
| TOP1 T | 29 | 6 | 689 | 4,134 | `64b3be98fbc00874aa295fba0a35e3fa048f54dc9d10225ca3b22e3bb538ecd3` |

The naïve expected totals, 16,008 and 4,140, were wrong. Validator-only job
`8298362` imported `integer_stride_selected_count`, checked every replica and
endpoint, and completed in two seconds with exit code `0:0`. Its stderr was
empty. The accepted reports are hard links to the saved calculation outputs,
so their bytes did not change. The repair did not read a trajectory.

Before that validator ran, independently submitted complete-interval jobs
`8298283` (T) and `8298284` (D) had installed valid reports at the historical
prepared-result paths. Both jobs completed with exit code `0:0`; their stderr
files contain only the passing 39-test preflight. Fresh validation and
installation gates pass for both reports. Their file hashes differ from the
validator-only copies because the files retain run-specific provenance and
resource records, but their scientific payloads match exactly: project,
system, input, and comparison-contract signatures; evaluated-frame counts;
and conceptual, candidate, materialized, event, geometry, and spatial-work
counts.

Validator attempt `8298341` therefore failed closed when it found existing,
nonidentical bytes. It did not overwrite either report. The historical
prepared-result paths now contain the validated complete-interval reports, so
downstream reporting does not require a path remap. The versioned attempt03
copies, the failed attempt, and all original evidence remain immutable.

## TBA recovery evidence

The checkpointed structural-QC continuation `8296182` completed. The accepted
report covers 20 replicas and 2,500 selected frames from 200,000 source frames,
using stride 80. Its technical status is complete, its error count is zero,
and its single warning states that coordinate and chemical checks used the
declared sample. The report SHA-256 is
`154e997289f18181429301fe7a3dc8f85a305ee65594697e827c211055453863`.

Authoritative reporting v3 failed because a required convergence report path
did not exist. Versioned v4 also failed and was retained. Versioned v5
completed as job `8298046` with exit code `0:0`; it reviewed 121 reports across
33 modules, produced 904 candidates, retained 697 QC records, and reported zero
silent omissions. The final summary SHA-256 is
`b6a6afb474e2b0b22f70ca0ae0989c93d657a07f74a62f515fa074b60d7e1e57`.

## Planner and scheduler boundary

The Apollo curves use 44 cores and 185 GiB per node, no more than 16 nodes per
campaign, and a fixed padded Slurm request of `7-00:00:00` for every
planner-backed job. Predicted walltime is the expected elapsed time after
allocation; it is not the requested limit and excludes queue delay. The
planner starts from the maximum task inventory and memory demand, then reduces
the useful-node ceiling when tasks cannot use more resources.

The primary TOP1 recommendation keeps a shared stride contract: D uses six
nodes and predicts 121.09 hours; T uses three nodes and predicts 123.00 hours.
If both allocations begin together, time until both finish is 123.00 hours,
with nine requested nodes and 97.57% combined information. The complete curves,
including the 75%, 80%, 90%, 95%, 99%, and 100% thresholds, are in
`apollo_node_sweet_spots_20260902.json`. No planning-sweep job was submitted.
The reviewed allocation-efficient choices are one node for current TBA, three
for current TREX, one for current thrombin, five for projected TREX, and one
for projected thrombin. The 168-hour request is only the hard feasibility
ceiling; it is not an optimization target.

## Scientific acceptance

Status: **not evaluated**.

TBA v5, TOP1 D/T hydrogen bonds, the TREX fixtures, and the thrombin fixture are
technical evidence. They do not establish convergence, metastability, kinetics,
mechanism, or biological importance. The TOP1 T-dihedral replacement was still
running when this record was written. TOP1 downstream reporting remains held
until that result and the installed hydrogen-bond reports pass explicit human
review. TREX remains unsuitable for biological conclusions without human
acceptance of its earlier scientific-QC concerns. No TREX or thrombin campaign
was submitted.
