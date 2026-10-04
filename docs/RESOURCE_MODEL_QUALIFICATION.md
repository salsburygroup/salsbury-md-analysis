# Runtime, memory and resource waits

Changing a cost estimate does not change an existing campaign's frames, chemistry
or scientific minimums. Prepare a new plan to use a revised model; keep older
plans and failed attempts.

## iMWKMeans

The pooled iMWKMeans estimate includes fit observations, feature dimensions,
k values, Minkowski powers, initializations, the iteration ceiling and silhouette
cap. Nonquadratic Lp centers use 100 bisection passes in the current implementation.
Initialization and those calculations appear in `imwkmeans_runtime_model`.

The model separates a 300-second setup/read/report allowance from fitting and
linear all-observation assignment. That allowance is an engineering estimate,
not measured setup time. The iteration ceiling bounds modeled work; it does not
predict convergence. The configured task time factor applies once. Campaign
Slurm headroom remains a separate allowance inside the user's time limit.

The provisional reference rate is 0.5 seconds per fit observation for three
dimensions, k=2–12, p=1.5/2/3, four initializations, 500 iterations and a
1,000-observation silhouette cap. Other grids scale by their declared work.
Completed 40,000-observation commands took up to 18,428.5 seconds in the inspected
incident. Their complete grid signatures were unavailable. They support raising
the former proxy, but do not establish a transferable fitted model or a completion
time for the two pooled commands that timed out.

New sidecars retain `imwkmeans_workload`: implementation, settings, fit count and
assignment count. Compatible completed timings can raise the workload envelope.
A timeout contributes a lower bound with an explicit 1.5 model-uncertainty factor;
it is never relabeled as completion. Records without the workload signature cannot
calibrate the grid model. Whole-command wall time is charged conservatively as
occupied single-core time, not reported as measured CPU usage or kernel time.
Held-out checks report coverage without changing the fitted envelope. Covering
a timeout bound does not validate its eventual completion time.

## SASA and water-network memory

A batch-level Slurm peak may combine unrelated analyses. The catalog retains
such peaks as diagnostics but does not apply them as task or worker measurements.

A smaller request requires at least two completed, context-matched measurements
explicitly qualified as peak measurements. Each must identify one analysis
command and cover the target atom count, frame ceiling, worker concurrency and
memory-relevant settings. Missing fields, a different implementation, a smaller
measured workload or a sampled RSS lower bound cannot authorize a reduction.
SASA and water reports retain source atom counts; sidecars retain their memory
workload. Raw-source and native-cache contexts remain distinct.

Before sampling is chosen, qualification uses the full supplied frame ceiling.
These modules do not shrink qualified measurements again using generic atom or
observation scaling. Without qualified evidence, the existing baseline stays.
The DEAC policy remains **1.5× task memory plus 1 GiB once per node**, applied by
the planner. Neither launcher adds another memory factor.

## Resource waits and task outcomes

Local execution uses the same CPU/memory token reservations as the native planner
and Slurm launcher. A later small task cannot take tokens reserved for another
task merely because they are momentarily free. Capacity is released when the
preceding task terminates. Resource waits are completion-only: failure releases
the reservation without blocking unrelated science. Required input dependencies
still require success. The packing algorithm and per-node limits are unchanged.

| Task record | Meaning | Runtime-calibration eligibility |
| --- | --- | --- |
| `complete` / `recovered_complete` | Execution and required report checks passed | Eligible with input/workload evidence |
| `timed_out` | A child ran and exceeded its task or campaign limit | Right-censored bound only |
| `not_started_deadline` | No child started before the deadline | Excluded |
| `skipped_dependency` | A required input task failed | Excluded |
| `failed` | Execution, configuration or output validation failed | Excluded |
| `reused_complete` | Existing output passed reuse checks | No new measurement |

`execution_started` and `runtime_calibration_eligible` make this distinction
machine-readable. Never-started records retain pending input/resource waits and
zero execution time. If a retry cannot start before the deadline, the last
executed attempt's outcome remains the task's outcome. Original logs and attempts
remain available.
