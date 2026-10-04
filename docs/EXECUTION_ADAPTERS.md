# Local and Slurm execution

Scientific configuration and execution-site configuration are separate. The
analysis config chooses an adapter, while a Slurm profile describes one cluster.
Neither adapter changes module selection, frame strides, definitions, dependencies,
or report/hash contracts.

Choose `local` when you want the package to manage work on the current computer,
`slurm` when a Slurm scheduler should manage it, and `custom` when an external
launcher should consume the generated dependency and resource contract.
The scientific plan is the same either way; only the way resources are requested
and jobs are launched changes.

Every prepared campaign writes `planning-report.md` and `planning-report.json`.
The Markdown report starts with an analysis-family table: numeric cells are
effective integer strides over the original trajectories, while `Off`, `Deferred`,
`Not applicable`, and `Not scheduled` remain distinct. The JSON report retains
the cache, projection, and method-local stride components; exact per-replica frame
counts; selected totals; retained time spacing; and sampling-floor status.

Required preflight and final-reporting jobs have their own
[workload estimates](ORCHESTRATION_RESOURCE_ESTIMATES.md). The planner includes
their costs in the campaign budget and lists them separately from sampled
analyses. Their timeout allowances are separate from estimated durations.
See [runtime and memory qualification](RESOURCE_MODEL_QUALIFICATION.md) for
iMWKMeans grid costs, task-scoped memory evidence, local resource reservations
and the distinction between a timeout and a task that never started.

To compare several prepared envelopes in the compact matrix form, run:

```bash
salsbury-md-analysis report-plan-matrix \
  --plan "8 h reduced=/plans/8h/planning-report.json" \
  --plan "24 h reduced=/plans/24h/planning-report.json" \
  --plan "48 h reduced=/plans/48h/planning-report.json" \
  --plan "168 h complete=/plans/168h/planning-report.json" \
  --output analysis-plan-matrix.md
```

To inspect the prepared plan without using either executor, add `--plan-only`
to `prepare-analysis` or `prepare-comparison`. Preparation still validates and
writes the campaign artifacts. It reports `execution_started: false` and
`jobs_submitted: false`, returns the complete plan, and leaves the local or
Slurm launch command for a separate reviewed step.
If the requested CPU cap exceeds the resolved workflow's useful concurrent
width, this output includes `REQUESTED_CPUS_EXCEED_USEFUL_PARALLELISM` and the
effective cap. Generated launchers use the effective cap, including Slurm array
concurrency and any multiprocess coordinate-cache request.

The structural-integrity task is internally replica-parallel when its validated
lossless coordinate cache is enabled. The planner caps it at the number of
replicas, the campaign CPU ceiling, and the number of workers that fit inside
the aggregate memory ceiling. The execution adapter requests those CPU slots
and the workers reuse the already-written cache; the task does not unwrap or
decode the original solvated trajectories a second time. Generated workflows
retain a dedicated structural-QC runtime project with one stable shard per
replica. Before launch, the execution adapter verifies that its worker cap
matches the planner and scheduler; a missing or inconsistent declaration fails
closed rather than falling back to serial execution. Structural QC depends
on cache completion, but unrelated modules do not depend on structural QC or
the cache unless their declared inputs require them.

Every planned task whose module has trajectory-ensemble semantics carries an
`ensemble_parallelism_contract`. The scheduler may partition structural QC,
RMSD/Rg, individual PCA, and continuous unwrapping into complete replica-local
work. For pooled RMSF, DCCM, common PCA, TICA, FES, clustering, and MSMs,
replica workers are limited to mergeable statistics, identity-preserving
features, or ordered segment records. Their primary result is finalized once at
the contract's pooled scope. A task that declares an independent replica
partition for one of those pooled estimators is rejected before launch. See
[`ENSEMBLE_PARALLELISM.md`](ENSEMBLE_PARALLELISM.md).

## Failures and preparation checks

### Diagnosing slow planning

Add `--diagnose-planning` to your existing `prepare-analysis` or
`prepare-comparison` command. For example, keeping the same inputs and limits:

```bash
salsbury-md-analysis prepare-comparison systems.json \
  --output my-plan --project-id my-study --config analysis-config.json \
  --plan-only --diagnose-planning --planning-stack-interval-seconds 60
```

This writes a Python stack snapshot every 60 seconds (configurable, at least
one second). Each invocation gets a separate directory beside the output:
`my-plan.planning-diagnostics/<process-id>-<unique-suffix>/`. Its path is printed
on standard error, leaving the command's normal JSON output unchanged.

- `stack-traces.txt`: periodic Python call stacks, an exception traceback if one
  escapes preparation, and best-effort SIGTERM and fatal-signal traces where
  supported. An existing process-level fatal handler keeps its original output
  destination; `metadata.json` records which handler owns those traces.
- `events.jsonl`: preparation boundaries, core-first trials, refinement progress
  and exact-input cache counters.
- `metadata.json`: Python/package versions, process ID, elapsed time, process CPU
  time and exit status on normal completion. CPU time excludes child processes.

The mode uses Python's standard-library watchdog, not per-call tracing. It
creates no diagnostic files or watchdog when off, and needs no extra package.
Snapshots briefly interrupt execution; shorter intervals and more threads cost
more. Start with 60 seconds. The mode does not change sampling, resource limits,
deadlines, pruning decisions or submission behavior, and does not recover a
stopped planner. SIGTERM keeps its prior termination behavior; SIGKILL cannot
be captured. An interrupted session may retain `status: running` in its metadata;
that is not evidence of a live process or a feasible plan. A later diagnostic
write failure warns on standard error without replacing the preparation result.

Stacks contain Python file paths, function names and line numbers, not local
variables or trajectory arrays. They do not profile native C/BLAS execution.
Exception messages can include input paths; review the bundle before sharing it.

### Worker receipts and preparation checks

Each local or generated Slurm worker saves an atomic, attempt-scoped status
receipt when it starts and ends. Failed workers are visible before independent
workers finish. Receipts identify the prepared task contract; changing that
contract prevents an old receipt from being treated as its current outcome.
`status` reports allocation activity separately, so a running Slurm allocation
does not conceal a failed task. Accepted final reports take precedence over old
failure evidence. For older runs without receipts, complete failed temporary
JSON reports can supply diagnostic evidence when the report belongs to one
task and its module matches. Their attempt freshness is explicitly unverified;
partial JSON and unrelated reports are ignored. Temporary files are never
accepted as completed results.

Preparation and native launch check the exact worker project path, accepting
absolute paths or paths relative to the analysis directory, including spaces.
A missing or double-prefixed path stops launch. Final selected PCA observation
counts are checked against grouped-learning and representative-selection
limits. These checks include equivalent oligomer members and use projection
counts, not the smaller PCA fit count. A conflicting explicit limit is reported
without changing sampling or raising the limit. Review it and replan resources.
For tICA-derived clustering, omitted short segments do not contribute to the
consumer count; this check waits for segment counts when they are unavailable.
Cache inputs that have not yet been materialized are checked against the final
allocation; their available files are checked again before execution.

## Local desktop or workstation

Local mode is the default. It needs Python 3.10 or newer plus the package's analysis
dependencies, but it does not need Slurm:

```json
{
  "config_schema": "salsbury-analysis-config-v1",
  "execution": {
    "submission_adapter": "local",
    "maximum_parallel_cpus": 8,
    "maximum_hours_per_cpu": 24,
    "maximum_memory_gib": 64,
    "maximum_scratch_gib": 256,
    "planning_utilization": 1.0,
    "pilot_budget_fraction": 0.0,
    "finalization_headroom_fraction": 0.0,
    "time_safety_factor": 1.5,
    "well_calibrated_memory_uncertainty_factor": 1.0,
    "poorly_calibrated_memory_uncertainty_factor": 1.0,
    "censored_timeout_safety_factor": 1.5
  }
}
```

Prepare with that config and run `./run-local.sh`. The dependency-aware executor
dispatches ready tasks across the full dependency graph and atomically reserves
both CPU slots and planner-derived memory. Phase labels organize the report;
they do not delay a ready task behind unrelated work. Their combined reservations cannot
exceed `maximum_parallel_cpus` or `maximum_memory_gib`.
The two named calibration factors default to 1.0. An explicit override remains
available, but the default no longer adds 1.25 before the site's multiplier.
The planner applies DEAC's 1.5 factor once to each task working set, rounds the
task request up to whole GiB, and holds 1 GiB separately per node. Slurm passes
the padded task request through without multiplying it again.

The resource-token schedule conservatively holds that reserve for each node
that could be occupied concurrently, bounded by the permitted nodes, task node
counts, and available CPUs. It prints this upper bound and the aggregate reserve
in `resource_token_policy`. Co-located tasks do not each incur another 1 GiB.
The local runner holds one node reserve. A custom launcher must honor the
same task requests and node reserve. These are reservation limits; local mode
does not impose an operating-system RSS limit on a process that outgrows its
estimate.

The memory value is an aggregate campaign ceiling rather than a per-task limit,
a prediction, or an amount preallocated at startup. Each task keeps both its
working-set estimate and safety-adjusted reservation in
`campaign-resource-plan.json`, and completed reports record measured peak
resident memory for later calibration.
`maximum_hours_per_cpu` is the complete local campaign wall-time deadline, and each
task also receives its planner-derived deadline. Each attempt receives unique logs
and a retained JSON record under `local-execution-status/`. A terminal failure
blocks only tasks that require its successful output. Technically complete module outputs are revalidated and
reused by the same worker logic used on Slurm.

Local mode is also the simplest portability check. A wheel-installed v80
candidate completed the generated workflow for a real 100-frame TBA trajectory
on one Apollo CPU in about four minutes, with roughly 162 MiB peak resident
memory. That bounded run establishes that the installed local adapter and its
dependency order work without Slurm; it is not a runtime promise for larger
systems and carries `scientific_status: not evaluated`.

The fitted CPU model also passed two independent, multi-system TOP1
hydrogen-bond runs that were not used for fitting. The D comparison used 15,996
selected frames and consumed 70.5% of its planning upper bound. The T comparison
used 4,134 selected frames and consumed 80.5%. The check used topology sizes,
source lengths, selected-frame counts, and pre-coordinate endpoint counts. It
did not read coordinates or materialize the Cartesian candidate set. The
versioned calculation is in
`validation/planner_final_holdout_acceptance_20260902.json`. Because both runs
remained below the existing upper bounds, this check did not change the fitted
coefficients.

### Automatic task recovery

New campaigns retry a failed or timed-out task once by default:

```json
{
  "execution": {
    "autorecovery": true,
    "maximum_task_attempts": 2
  }
}
```

Set `execution.autorecovery` to `false` to allow only the first attempt.
`execution.maximum_task_attempts` must be from 1 through 5. Plans created before
the recovery fields existed keep their original one-attempt behavior.

Recovery is task-local. A successful report is hash-checked and reused; it is
not rerun because another task failed. A local task is retried only within the original campaign deadline and resource
request. Slurm requeues receive a fresh allocation and are bounded by attempt
count; cumulative campaign-wall accounting across requeues is not yet enforced. Local execution records one
stdout file, stderr file, exit code, timing record, and byte count per attempt
under `local-execution-status/`. A task that succeeds after a failure is labeled
`recovered_complete`, and its accepted report can release scientific
dependents.

Exit code zero is necessary but insufficient when a task declares completion
reports. The local and Slurm recovery wrappers require every declared JSON file
to match their module and project contract, carry no technical errors, and pass
report/summary and companion-file hash checks. Recorded input signatures are
revalidated for local reuse; hashing large source files adds read I/O.
Missing, stale, invalid, or failed reports produce recovery exit code 66 and cannot release a
success-dependent task.

The Slurm launcher writes the same attempt events under
`autorecovery-status/`. Ordinary nonzero exits are retried inside the existing
allocation. When Slurm sends the generated pre-timeout signal, the wrapper
records the event and requests a scheduler requeue if another attempt remains.
Module checkpoints stay in the prepared campaign directory, so a compatible
checkpoint can resume after requeue. If the attempt cap is reached, the task
fails and success-dependent jobs remain blocked or are cancelled by Slurm.
Completion-only and unrelated jobs keep their declared graph behavior.

Automatic recovery addresses transient execution failures. It does not change
inputs, frame selection, chemistry, estimator settings, wall limits, or
scientific acceptance. Read the retained stderr, resource use, partial files,
and dependency outcome before making a versioned repair for a terminal failure.

## External launcher

Set `execution.submission_adapter` to `custom` when a site launcher, workflow
engine, container service, or another scheduler should start the generated work:

```json
{
  "config_schema": "salsbury-analysis-config-v1",
  "execution": {
    "submission_adapter": "custom",
    "maximum_parallel_cpus": 16,
    "maximum_hours_per_cpu": 24,
    "maximum_memory_gib": 128
  }
}
```

Preparation writes `launcher-contract.json`. Each task has a stable ID, a
`depends_on_task_ids` list containing only reports or data it consumes, and an
optional `wait_for_task_ids` list for completion-only ordering. The
numbered levels are a topological presentation of that graph, not a rule that
all work in one level succeeds or fails together. A launcher may run ready tasks
concurrently while their summed `cpu_slots` and `requested_memory_gib` remain
within the contract envelope. For each task the contract supplies the script,
argument vector, working directory, compatibility environment, timeout, planner
task IDs, true prerequisites, and expected completion reports. A nonzero exit,
timeout, or missing accepted report skips only descendants that name that task
in `depends_on_task_ids`; completion-only consumers such as final report
collation still run, and unrelated work remains eligible.

The user-supplied executable receives the contract path as its only argument:

```bash
export SALSBURY_MD_ANALYSIS_CUSTOM_LAUNCHER=/absolute/path/to/my-launcher
./run-custom.sh
```

Worker scripts retain Slurm-compatible variable names for portability. The
external launcher assigns unique `SLURM_JOB_ID` values, a stable
`SLURM_ARRAY_JOB_ID` for related array elements, and its site name in
`SLURM_CLUSTER_NAME`; the contract supplies the remaining task environment.
`custom` mode prepares the work but does not run or submit it automatically.

## When the requested memory is too small

Preparation checks every enabled task at its technical minimum. If even one
cannot fit, it stops before generating a runnable campaign and writes a
`memory-feasibility-report.json` with the largest estimate, exact shortfall,
rounded-up memory recommendation, oversized tasks, and the narrowest config
switches that would remove them. It does not silently disable an analysis or
reduce its technical frame minimum.

Users who explicitly prefer a reduced campaign can add
`--auto-disable-to-fit-memory` to `prepare-analysis` or `prepare-comparison`.
The initializer then preserves the requested config, disables only the listed
module or clustering-method switches and their dependents, and replans. Review
`analysis-config.requested.json`, `analysis-config.memory-fit.json`, and
`memory-feasibility-report.json` before launching. The fallback addresses
memory only; CPU-hour, critical-path, calibration, and scratch limits still
fail closed.

Use `--auto-disable-optional-to-fit-resources` when the user wants the same
explicit reduction across CPU, critical-path wall time, and memory. The
initializer validates the protected core first, including preprocessing,
representative structures and reporting. If that plan does not fit, it stops
before exploring optional analyses. It then tries the full requested scope;
when necessary, it removes optional configuration bundles and recalculates
sampling, reporting costs and dependencies after each change. Tied bottlenecks
can require several removals before wall time falls. Trajectory writing may be
disabled separately; representative structures remain protected.

The full-scope trial checks scientific-minimum feasibility before refining
sampling. If it does not fit, reduction ranks optional removals using already-priced
blocking work and scientific-priority weights, then rebuilds and checks one
dependency-closed removal at a time, then refines the fitting reduced subset.
Each check still enforces overall scientific floors, PCA/clustering
source consistency, dependencies, memory padding and per-node limits. A parent
PCA projection must supply its downstream fits; reducing it to its own smaller
floor does not make those fits legitimately source-limited.
If coupled integer streams make a protected-core minimum probe fail, the
coupling refinement is required before rejecting that core. If this search is
interrupted, feasibility remains unknown.

Native preparation limits sampling refinement to 512 new schedule constructions
per planning call, shared across its coupling iterations. Minimum-feasibility
checks and final validation still run; exact schedule-cache hits do not consume
this allowance. Change `planning.maximum_refinement_schedule_calls` in the
analysis configuration to adjust it. Zero keeps the validated minimum candidate
without extra sampling refinement. The setting counts work, not seconds, so it
does not guarantee a planning deadline.

When refinement reaches the limit, preparation retains its validated minimum
candidate and records `planning_refinement.status: validated_minimum_fallback`,
the reason and call counts in the plan. This is a feasible fallback, not a claim
of maximum information or optimality. The final launcher uses the selected
trial's validated sampling schedule without repeating that optimization; input
bindings, settings, scientific floors and the native schedule are checked again.
User-supplied fixed schedules remain unchanged.

Exact repeated calculations share a bounded in-memory cache within that
preparation. Every task field and resolved planner argument is part of the key;
changed sampling, costs, dependencies or resource limits require a fresh result.
Returned plans are isolated copies. Nothing is reused from a previous invocation.
The sampling-plan cache holds at most 32 results and 16 MiB of serialized content.
A separate schedule cache holds at most 128 results and 8 MiB. Python heap usage
is larger than these serialized sizes. Failed searches are not cached. Diagnostic
mode records both caches' hit/miss counts, schedule-cache evictions, and the
inner scheduler cache's hits and misses. Ranking is a heuristic: it avoids optimizing every alternative but
does not prove the largest possible retained set. The final native preparation
still validates the selected configuration and its emitted schedule.

Review `analysis-config.resource-fit.json` and `resource-fit-report.json` for
every disabled switch and whether the protected fallback was used;
`planning-report.md` gives the final strides. The completed protected preparation is retained if an optional
search fails. Search errors and iteration histories are recorded separately
from resource rejections, and the result is not claimed to be an optimal
subset. Trial receipts are saved beside the output in
`<output>.resource-fit-evidence/` as each trial finishes, then copied into the
completed output. A stopped preparation submits nothing. A fixed sampling
schedule is checked unchanged, without optional pruning.

Slurm requests can be larger than the estimated working set because the
planner reads explicit adjustments from the site profile. It applies those
terms before testing memory feasibility. A profile may also declare its CPU and
memory per node. Planning then rejects an adjusted task request that cannot fit
one node and assigns each task global and per-node CPU and padded-memory tokens.
The sum of concurrently held tokens cannot exceed either the campaign envelope
or a node's CPU and memory shape. A task waits only for its own scientific
inputs, explicit completion waits, and prior users of the tokens it needs.
Among dependency-ready tasks, the adapter schedules the earliest resource fit
first and uses the remaining declared critical path to break ties. This lets a
downstream-critical task use an available node instead of waiting behind an
unrelated long task that happened to appear earlier in the generated plan.
`submit.sh` uses `afterany` for token and completion-only predecessors, so a
failed job releases capacity without becoming an accidental scientific gate.
It uses `afterok` only for a task's `depends_on_task_ids` and
asks Slurm to terminate a descendant whose required job failed instead of
leaving it pending indefinitely. The complete mapping remains visible in
`scheduler-resource-requests.json`.

Ordinary module relationships appear as `wait_for_task_ids`. They delay a
consumer until a possible cache producer finishes, but they do not require that
producer to succeed. The worker validates any completed cache against the
current project, system, content signature, report hash, and sidecar; otherwise
it unsets the cache and recomputes from the project inputs. RMSF permutation is
submitted separately with RMSF as its only success-required report.
Integrated comparison is also separate and does not wait for structural QC.
Before submission, run:

```bash
./submit.sh --preview
```

This prints `slurm-submission-preview.json` and exits without calling Slurm. The
preview gives the exact job and dependency counts, configured CPU and
aggregate-memory caps, resource-token edges, peak scheduled resources, the planner's
estimated dependency critical path, and the sum of scheduler time-limit
reservations. It warns when the prepared dependency and token schedule cannot use
all requested cores. If the generated dependency/resource critical path exceeds
the campaign wall limit, the preview marks the schedule infeasible and
`submit.sh` refuses to submit it. Running `./submit.sh` prints the same contract
immediately before the first submission.

## Slurm cluster

Set `submission_adapter` to `slurm` and provide `slurm_profile`:

```json
{
  "config_schema": "salsbury-analysis-config-v1",
  "execution": {
    "submission_adapter": "slurm",
    "slurm_profile": "../slurm/my-cluster.json",
    "maximum_parallel_cpus": 32,
    "maximum_hours_per_cpu": 24,
    "finalization_headroom_fraction": 0.0
  }
}
```

Site profiles may declare `partition_maximum_wall_minutes`,
`partition_maximum_nodes`, and a `long_wall` partition role. A generated request
that exceeds its preferred partition's wall or node limit is routed
automatically to `long_wall`; if no acceptable fallback is configured,
preparation fails before scheduler submission. The supplied DEAC profile records
the 24-hour, one-node `small` limits and routes longer or multi-node work to
`large` while leaving ordinary short jobs on `small`.

The profile schema is `salsbury-slurm-profile-v1`. It records scheduler submit,
status, and cancel commands; account, Unix group, QoS, and role-specific partitions;
Python and package paths; environment setup commands and variables; shared-write
umask; storage and scratch roots; conservative resource policy metadata; and an
optional `node_policy` with `cpus_per_node`, `memory_gib_per_node`, and
`maximum_nodes_per_campaign`. The generic template leaves the node shape null;
the supplied DEAC profile uses a conservative 44-CPU, 185-GiB node shape. These
are editable profile values, not hard-coded planner constants; use the real node
shape for another cluster, or leave the fields null when no homogeneous shape is
available. The adapter converts every planner task estimate to a time and memory
request using the profile preferences. The shipped profiles set
`walltime_safety_factor` to 1.0 because the planner has already applied its 1.5
task-time factor. The scheduler therefore does not multiply planner estimates a
second time. Profiles with another value fail validation instead of silently
restoring the duplicate factor. Scheduler-only padding belongs in
`walltime_overhead_minutes`. Individual jobs retain that overhead and their
minimum timeout. Their timeout sum is a diagnostic, not a campaign runtime
estimate, and no longer rejects an otherwise feasible schedule.

The campaign allocation recommendation uses the final dependency/resource
schedule, including the planner's existing uncertainty adjustment:

```text
requested hours = ceil(estimated scheduled hours × 4/3)
```

Both profiles expose `campaign_walltime_headroom_fraction` (default one-third)
and `campaign_walltime_rounding_minutes` (default 60). For example, an
already-buffered estimate of 11.75 hours requests 16 hours, even if the planning
budget was 48 hours. The user ceiling includes the full allowance and rounding.
For a 48-hour ceiling, the planner reserves at most 36 estimated execution hours
before selecting strides or proposing optional reductions. These execution
hours already include task-level model uncertainty and explicit preflight and
reporting costs. Fresh configurations add no utilization, pilot or finalization
reserve. Existing explicit reserve settings remain in force and are reported
separately in `time_allowance_accounting`.
For a non-integer ceiling, the planner first rounds the usable allocation down
to the configured interval, then divides by the headroom factor. The preview
refuses submission if the final schedule plus the full allowance exceeds the
ceiling; it never trims the allowance to make a plan pass. Scientific minima,
chemistry and memory padding are unchanged. Fresh planning can select different
strides within the smaller execution budget. Queue waiting and a separately
launched interactive build are excluded. Local-only execution retains its
existing budget; this allocation allowance applies to Slurm planning.

Native preparation validates the generated task/dependency schedule before it
enables a launcher. `native-schedule-validation.json` records the schedule hash,
task coverage, CPU-hour and wall-time budgets, final estimate, and any failures.
Every logical task must map to exactly one execution task; one serial worker
can contain several logical clustering methods. The same dependency and
CPU/memory packing calculation supplies the Slurm preview and acceptance check,
including the configured memory reserve once per node.

Fresh candidate allocation and adapter validation use the same task dependencies
and CPU/per-node-memory token scheduler. Independent tasks can overlap without
waiting for an entire analysis stage. When more nodes are allowed, smaller-node
placements remain candidates for unchanged task requests. The chosen schedule
is a constructed feasible schedule, not proof of a globally optimal runtime.
Older task metadata without dependency contracts retains the legacy stage model.
Scientific floors, CPU limits, memory limits and missing-calibration checks
remain enforced. Fixed sampling remains fixed.

If no cache
stride is selected, preparation saves the rejected candidate diagnostics and
evaluates the protected-core reduction recommendation. It does not export a
fixed sampling schedule or enable a launcher. Setting
`fail_if_minimum_coverage_unaffordable` to `false` cannot make an unselected
cache search executable. Legacy independently authored workflows without a
native validation record keep their
existing validation path. Estimates exclude queue delays and do not guarantee
execution time.

For a campaign that fits on one node, use the generated shared-allocation route:

```bash
./submit.sh --single-allocation --preview  # Inspect; submits nothing.
./submit.sh --single-allocation            # Submit after reviewing the plan.
```

`run-campaign.slurm` requests the computed campaign time and the existing
CPU/memory envelope. It runs the native local task-DAG executor on the allocated
node, with the same shorter deadline enforced inside the job. It does not
submit child Slurm jobs. Check `single_allocation.submission_permitted` in the
preview. A multi-node plan, incompatible node capacity or unavailable partition
is refused for this route; use the normal distributed `./submit.sh` instead.
Normal per-task submissions keep individual task timeouts, not a 16-hour limit
on every task. Separate queued jobs do not share one enforceable elapsed-time
deadline; the campaign value is their runtime recommendation, not a guarantee
about calendar completion.

Previously prepared directories are unchanged. Prepare into a new directory
after upgrading; do not overwrite accepted plans or edit their scientific
budget merely to shorten a Slurm request.

`scheduler-resource-requests.json` records every mapped planner task, safety
margin, selected partition, final request, and resource-token schedule.
`slurm-submission-preview.json` also records the planned node count, each task's
conceptual node assignment, per-node padded reservation, runtime estimate,
campaign time request and diagnostic timeout path. Replica-final modules
and coordinate-cache construction can use an
identity-preserving `srun` worker group across several nodes; pooled reducers
still run once after all replica workers finish. Non-distributed modules remain
single-node jobs, and independent tasks provide additional cross-node
parallelism. The canonical `submit.sh` submits individual array
elements when needed so that both CPU and memory are bounded across all jobs that
can run at the same time. Only requests that cross
`large_memory_threshold_gib` use the `large_memory` partition role. Use the
generated `submit.sh` for a planned campaign: it applies the task-specific node,
task-count, CPU, memory, and time requests. Individual worker scripts are
implementation artifacts, not a substitute for the resource-bounded launcher.
`slurm-submission-preview.json` is the concise preflight view of that complete
mapping; `execution_started: false` and `jobs_submitted: false` describe the
preview itself, not the state after `./submit.sh` is executed.
Copy
`profiles/slurm/generic-template.json`, review every value with the cluster owner,
then prepare and run `./submit.sh`. The exact normalized profile is retained beside
the generated workflow as `slurm-profile.json`.

Setup commands are literal reviewed shell lines and therefore belong only in a
trusted, version-controlled profile. Scheduler command fields accept one executable
name or absolute path, environment variable names are validated, and additional
directives must begin with `#SBATCH --`.

## Optional capacity advice

Capacity inspection is separate from preparation and submission. Run it only when
you want a live planning answer:

```bash
salsbury-md-analysis advise-slurm-capacity prepared-analysis \
  --wall-hours 24 --format markdown
```

The command reads `campaign-resource-plan.json`, `scheduler-resource-requests.json`,
and `slurm-profile.json`. The scheduler-request manifest limits the calculation to
planner rows that have generated execution tasks, avoiding double counting of
pooled planning rows that were replaced by per-system chemistry tasks. It first
reports the maximum parallelism that the workflow graph can use, the live Slurm
and account/QoS ceilings it can discover, and the smaller recommended CPU count.
It then reruns the saved resource allocation in memory using that CPU count and
the supplied duration. Saved task definitions are inputs, but saved PCA projection
counts are not fixed downstream inputs. The adviser regenerates each view's PCA
projection, replaces every associated clustering-fit source stream, and repeats
the full allocation until the counts agree. It records the coupling iterations
and fails if a clustering fit has no projection parent or the coupled allocation
cannot be stabilized. The result includes each method's integer stride, selected
frames, estimated CPU-hours, observation-scaled memory, largest scheduler request,
and the largest exact resource-wave memory total. The saved plan also separates
CPU-hour utilization from wall-time utilization and reports why allocation
stopped. Repeating the calculation with a longer duration cannot reduce any
task's frame coverage when the tasks, CPU cap, memory cap, and reserve fractions
are unchanged. The clustering-fit source streams are regenerated separately for
each duration rather than carried over from the prepared campaign.

Live inspection uses only `scontrol`, `sacctmgr`, and `squeue`. It does not call
`sbatch` or `scancel`. `--offline` skips those queries and replans from saved
evidence only. `--cpu-ceiling` applies a lower personal or project limit, and
`--maximum-memory-gib` tests a different aggregate concurrent-memory ceiling without changing the
prepared campaign.

Before submission, the queue section can say whether nodes currently have room
for the largest request and summarize queue pressure. That is not a start-time
reservation because priority, fair-share, backfill, and later submissions can
change placement. After submission, repeat `--job-id JOB_ID` for pending jobs to
include Slurm's own projected start times. JSON is the default output and is the
best interface for ChatGPTWork; `--format markdown` gives a shorter human-readable
summary.

This command is optional. Local execution, generic Slurm submission, and every
analysis module work without invoking it, and it adds no Python dependency.

## Salsbury-group DEAC default

Use `profiles/analysis/deac-default.json`. It selects
`profiles/slurm/deac.json`, currently configured for:

- Slurm account `salsburygrp`, Unix group `salsburyGrp`, and QoS `normal`;
- `small` for routine analysis roles and automatic routing to `large` at 96 GiB;
- `/opt/scyld/slurm/bin` scheduler commands;
- `/deac/phy/salsburyGrp` group storage and shared-write `umask 0002`;
- the dedicated versioned v76 group analysis environment and measured Apollo
  calibration catalog;
- 32 parallel CPUs and a 24-hour complete-campaign planning envelope.

The measured-resource catalog accepts only hash-bound complete report sidecars
and explicitly labeled right-censored timeout records. Timeout target frames are
not completed frame coverage. Their elapsed CPU and wall times are lower bounds
that receive the configured censored-timeout safety factor before planning.
Wall lower bounds scale with selected frames and never assume speedup from more
CPUs than the failed attempt validated. A multi-CPU timeout's MaxRSS remains
aggregate diagnostic evidence; it is not replayed as one worker's memory.
Repeated complete measurements may replace a legacy memory baseline only when
their measurement scope explicitly qualifies that replacement. Unqualified,
one-off, and censored evidence cannot lower it. Fresh invocation supervisors
separate each command's CPU accounting from earlier commands. RSS records say
whether they describe the largest child or a sampled process-tree peak; sampled
RSS remains a lower bound and cannot authorize a smaller memory request. Replica-parallel jobs declare the full
worker population independently from active concurrency. CPU, aggregate memory,
per-node memory, and node-count limits determine the active workers, and any
remaining workers run in explicit waves without changing selected frames.

The current catalog adds held-out size-and-length CPU models for structural QC,
direct hydrogen-bond discovery, and ion-atmosphere analysis. Each model fits a
fixed term, an exact topology-atom by source-frame term, and a selected-work
term. Structural QC uses selected topology-atom frames. Hydrogen-bond planning
uses a spatial endpoint-pair proxy and never prices the full donor-acceptor
Cartesian universe. Ion-atmosphere planning uses ion-target minimum-image pairs
and a conservative topology-atom proxy before execution. The middle-size
system was excluded from fitting and had to fall below the planning upper bound
at 20, 50, and 100 selected frames. A project outside the measured atom-count,
source-length, or selected-work range still requires a local pilot.

Structural QC uses one worker per replica, bounded by campaign CPU and memory.
Each worker reads only the complete-interval frame indices chosen by the same
generator used by the planner. Its frame-local `make_whole` reconstruction does
not trigger a separate full-trajectory unwrapping pass. Other analyses retain
continuous unwrapping when their own coordinate contract requires it. Routine
structural QC therefore does not require every saved frame unless the resolved
sampling plan explicitly selects every frame.

The Unix group and path fields are retained provenance and operator configuration;
the launcher does not run `chgrp`, move data, or change input permissions. Update the
profile if DEAC changes its partitions, account/QoS policy, or validated environment.
