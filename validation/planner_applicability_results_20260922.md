# Planner repair validation: merge blocked

The bounded repair is implemented and pushed in draft PR #129. It is **not merged**.
The small-system checks passed, but the required runtime gate did not.

## What passed

All 15 small-system holdouts passed their original frozen timing bounds and used
the planned frame counts: RMSD/Rg, RMSF, individual PCA, DCCM, and DSSP at 66,
250, and 1,000 selected frames. All DSSP frames retained all 28 expected residues.
The 1,000-frame DSSP check took 796.7 seconds.

Two medium-system checks also completed successfully. The larger checks were
runtime pilots, not scientifically sufficient samples or accepted analyses.

| Holdout | Original frozen estimate | Observed runtime | Decision |
| --- | ---: | ---: | --- |
| medium-replica_rmsd_rg-66 | 3873.6 s | 1523.9 s | Pass |
| medium-pooled_rmsf-66 | 2888.4 s | 1527.6 s | Pass |
| medium-dccm-66 | 226.7 s | >303 s; stopped | Original bound failed |
| large-dccm-30, corrected fixture | 248.3 s | >369.0 s | Bound failed; censored |

The medium DCCM result is a lower bound from an unfinished job, not a completed
measurement. The later source-reading guard increases that estimate, but the
written acceptance rule retains the stricter original bound. It cannot be used
to retroactively pass this test. The corrected large DCCM result is likewise
not a completed scientific output.

The 104,300-atom case is outside the repair's transfer envelope and retains the
legacy estimate exactly. Its overrun therefore exposes an existing planner
limitation; it is not evidence that the patch reduced that estimate.

## Planning comparison

With the same NEMO inputs and resource ceilings, current main forecasts 11.263
hours and the candidate forecasts 7.882 hours. Both retain the same 32 scheduled
analysis tasks. Structural QC, RMSD/Rg, DSSP, and convergence/uncertainty increase
from 500 to 1,000 selected frames; no task loses coverage. These are forecasts,
not measurements of a newly executed full workflow.

## Regression checks and scope

The tested candidate is `60ab90e629e141e4fb8386a590556937b6900164`.
All 14 GitHub checks for that commit passed. The ten new policy tests cover
fallbacks, source-length allowance, idempotence, DSSP process costs, and unchanged
memory and sampling rules. Cluster full-suite results are recorded below.

The initial two cluster-suite attempts each encountered two environment errors:
local recovery-wrapper tests inherited a Slurm job ID and therefore wrote a
scheduler-named log instead of the expected `local.jsonl`. The final suite was
rerun without inherited Slurm variables in the test subprocess; source and tests
were unchanged. Original failure logs are preserved.

Final candidate suite in the corrected test environment:

```text
Ran 809 tests in 52.316 seconds
OK (2 skipped)
```

Scientific estimator sources and original MD files were unchanged. This repair
changes resource estimates, not estimator speed or scientific acceptance.

## Failed attempts and stopping rule

The first harness attempt had frame-selection and report-collection errors;
its outputs were excluded. The first large DCCM attempt was also invalid:
the inherited fixture allowed at most 44 atoms while selecting 588. A separate
test copy raised only that safety cap to 588, retained the same selection and
timing bound, and froze its source and input hashes before execution.

After the original medium timing bound failed, four unstarted checks were
canceled: medium PCA and large RMSD/Rg, RMSF, and PCA. They are **not validated**.
The corrected large check had a predeclared timeout 120 seconds beyond its
frozen prediction. No failed observation was fitted into new coefficients.
All jobs from this bounded validation batch have ended.

## Next decision

Keep PR #129 unmerged. A follow-up repair should distinguish raw-input
reconstruction from cached-coordinate analysis and validate the cost of reading
the full source independently of the selected-frame count. It needs a newly
frozen validation round; these failed holdouts must remain in the evidence.
Do not weaken sampling floors or remove periodic-reconstruction checks.

The scientific-research-premortem and project-state-transition-controller gates
kept the draft from merging after a timing failure. Scientific status: not evaluated.

Canonical Obsidian publication remains pending the designated publisher;
the local evidence and deferred handoff are not a canonical update.

Supporting evidence: `planner-validation-evidence.json` and the preserved
cluster evidence archive. No production campaign was submitted by the new
planning comparison.
