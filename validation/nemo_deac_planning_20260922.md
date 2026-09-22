# NEMO DEAC planning validation

The original DEAC tutorial's one-hour budget is rejected by the planner at
core revision `72e50ba8cc5b541abafa9d218f73eba1e5931870`. The revised 16-hour
budget passes planning and a Slurm preview with the same environment, inputs,
enabled methods, and sampling policy. This is preparation evidence. No analysis
jobs were submitted and no new runtime measurement was made.

## Validation

The environment passed `pip check`; DSSP reported version 4.6.1. The supplied
fixture contains 423 atoms and 1,000 frames. The study declares a 0.2 ps saved
frame interval. Both budgets retain two CPUs, 32 GiB aggregate memory, 8 GiB
scratch, the supplied DEAC profile, and the Apollo v5 calibration catalog.

| Check | Result |
| --- | --- |
| Original one-hour plan | Exit 2; infeasible; execution not authorized by planner |
| Sixteen-hour plan | Exit 0; feasible; 10.4934 estimated CPU-hours; 11.6718 estimated science-path hours |
| Sixteen-hour scheduler preview | Exit 0; feasible; execution not started; jobs not submitted |
| Recovery instructions copied from the revised tutorial | Passed; original failed directory retained; configuration backup retained; new output passed preview |
| Revised tutorial setup and planning blocks, from a fresh study | Passed; preview feasible; no submission ledger |
| Documentation suite | 14 tests passed on DEAC |
| Calibration suite | 11 tests passed on DEAC |

The revised plan contains 36 sampling tasks and no tasks below the declared
sampling floor. Three tasks exhaust a short input source; this does not establish
scientific adequacy. The scheduler preview models a 13.4712-hour dependency path,
within the 16-hour campaign limit, using at most two simultaneous CPUs and 8 GiB
of simultaneous reserved memory. Per-job timeout limits are not added together
as a prediction of elapsed time.

The original terminal error recommends `--target-wall-hours 16`. That option
belongs to the lower-level preparation command; the tutorial uses `plan`, whose
budget comes from the study configuration. The revised tutorial shows how to
change both budget fields and prepare a new directory.

## Why the estimate is large

Controlled planning comparisons retained the fixture, requested methods,
resource caps, and sampling policy. The table reports minimum-coverage costs,
so extra frame allocation under the 16-hour cap does not drive the comparison.

| Calibration input | Minimum CPU-hours | Minimum science-path hours | Recommended campaign ceiling |
| --- | ---: | ---: | ---: |
| Full Apollo v5 catalog | 10.4043 | 11.5788 | 16 hours |
| Same catalog, completed records only | 7.1113 | 3.9158 | 6 hours |
| No catalog overlay | 0.9962 | 0.7586 | 2 hours |

The last two rows are diagnostic counterfactuals, not approved replacement
settings. They do not establish actual runtime or justify deleting timeout
evidence. The generic model also rejects a one-hour campaign after reserves.

The full-catalog minimum assigns 4.1232 CPU-hours to RMSD/Rg, 3.0913 to pooled
RMSF, and 1.9670 to individual PCA. Their fixed components are 4.1175, 3.0750,
and 1.9506 hours respectively. Together, these three estimates account for
88.2% of the 10.4043 CPU-hour minimum.

`load_resource_calibration_catalog` groups records by module. Its
`_conservative_affine_cpu_model` builds a nonnegative envelope using frame
counts and completed or censored CPU time. The intercept then enters each
eligible task's fixed cost. For the three methods above, the recorded workload
multiplier is 1.0. These inherited fixed costs are not NEMO timing measurements.

`_apply_measured_resource_calibrations` also imports timeout-derived wall-time
floors. The resource planner scales those floors by selected-frame ratio and
allocated CPU counts. In this plan, structural QC has a 3.7507-hour wall floor
despite a 0.0051 CPU-hour estimate; secondary structure has a 4.9501-hour wall
floor despite a 0.0495 CPU-hour estimate. This explains why the critical path
is large even for the small fixture. CPU and wall estimates use different
conservative evidence and are not measurements from a single NEMO execution.

Removing censored records changes the fitted envelope, so individual intercepts
can increase even while the aggregate prediction decreases. A completed-only
catalog is not a valid estimate of the timeout contribution by simple
subtraction at the individual-method level.

The code implements its conservative envelope, but the catalog's transfer to a
small tutorial is insufficiently specific. The next calibration repair should
establish applicability by workload and implementation context, retain relevant
censored bounds, and check predictions against an independent measured NEMO
run. This investigation does not change the planner, catalog, scientific
sampling rules, or execution policy.

The historical Apple-silicon acceptance record reports 475 seconds and 0.1551
CPU-hours for an older 28-report run. Different code, hardware, and optional
modules prevent treating that record as the current DEAC runtime.

## Reproduction and boundaries

Use the revised DEAC tutorial's fresh-study commands, first with `--hours 1`
to reproduce the rejection, then with `--hours 16` in a separate study. Keep the
failed plan. Run the documented recovery block against the failed study and
preview its new output. Stop before the submission section.

For the diagnostic comparisons, copy the same study/configuration into new
directories at the 16-hour ceiling. In one copy set
`execution.resource_calibration_catalog` to null. In another use a separately
named catalog containing only entries whose `evidence_status` is
`complete_execution` (the schema also treats an omitted value as complete).
Record the derived catalog's hash and diagnostic-only status. Run `plan` only
and compare `minimum_known_cpu_hours`, `minimum_wall_hours_lower_bound`, and
`permissive_minimum_resource_request`.

The accompanying JSON records the exact source, fixture and catalog hashes,
checks, and acceptance scope. Wang's original files remain unchanged. Technical
planning validation is complete; scientific analysis and actual runtime
validation were not performed.
