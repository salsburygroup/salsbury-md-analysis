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
    "planning_utilization": 0.85,
    "pilot_budget_fraction": 0.05,
    "finalization_headroom_fraction": 0.05,
    "time_safety_factor": 1.5,
    "memory_safety_factor": 1.25,
    "censored_timeout_safety_factor": 1.5
  }
}
```

Prepare with that config and run `./run-local.sh`. The dependency-aware executor
runs phases in order and atomically reserves both CPU slots and planner-derived
memory for independent tasks within a phase. Their combined reservations cannot
exceed `maximum_parallel_cpus` or `maximum_memory_gib`.
The memory value is an aggregate campaign ceiling rather than a per-task limit,
a prediction, or an amount preallocated at startup. Each task keeps both its
working-set estimate and safety-adjusted reservation in
`campaign-resource-plan.json`, and completed reports record measured peak
resident memory for later calibration.
`maximum_hours_per_cpu` is the complete local campaign wall-time deadline, and each
task also receives its planner-derived deadline. Each attempt receives unique logs
and a retained JSON record under `local-execution-status/`; a failed attempt never
launches later phases. Technically complete module outputs are revalidated and
reused by the same worker logic used on Slurm.

Local mode is also the simplest portability check. A wheel-installed v80
candidate completed the generated workflow for a real 100-frame TBA trajectory
on one Apollo CPU in about four minutes, with roughly 162 MiB peak resident
memory. That bounded run establishes that the installed local adapter and its
dependency order work without Slurm; it is not a runtime promise for larger
systems and carries `scientific_status: not evaluated`.

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

Slurm requests can be larger than the estimated working set because the site
profile adds explicit safety margins. Preparation applies those margins before
testing memory feasibility. It then packs individual array elements and ordinary
jobs into deterministic resource waves. The sum of CPU slots and the sum of
buffered memory requests in one wave cannot exceed the two campaign caps.
`submit.sh` uses `afterany` between resource waves so a failed job releases the
next allocation. It uses `afterok` only for a task's `depends_on_task_ids` and
asks Slurm to terminate a descendant whose required job failed instead of
leaving it pending indefinitely. The complete mapping remains visible in
`scheduler-resource-requests.json`.
Before submission, run:

```bash
./submit.sh --preview
```

This prints `slurm-submission-preview.json` and exits without calling Slurm. The
preview gives the exact job and dependency-wave counts, configured CPU and
aggregate-memory caps, peak resources in any generated wave, the planner's
estimated dependency critical path, and the sum of scheduler time-limit
reservations. It warns when the prepared dependency and memory waves cannot use
all requested cores. Running `./submit.sh` prints the same contract immediately
before the first submission.

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
    "finalization_headroom_fraction": 0.05
  }
}
```

Site profiles may declare `partition_maximum_wall_minutes` and a `long_wall`
partition role. A generated request that exceeds its preferred partition's
declared limit is then routed automatically to `long_wall`; if no acceptable
fallback is configured, preparation fails before scheduler submission. The
supplied DEAC profile records the 24-hour `small` limit and routes longer work
to `large` while leaving ordinary short jobs on `small`.

The profile schema is `salsbury-slurm-profile-v1`. It records scheduler submit,
status, and cancel commands; account, Unix group, QoS, and role-specific partitions;
Python and package paths; environment setup commands and variables; shared-write
umask; storage and scratch roots; and conservative resource policy metadata. The
adapter converts every planner task estimate to a time and memory request using the
profile safety factors. `scheduler-resource-requests.json` records every mapped
planner task, the safety margin, selected partition, final request, and exact
aggregate-resource wave. The canonical `submit.sh` submits individual array
elements when needed so that both CPU and memory are bounded across all jobs that
can run at the same time. Only requests that cross
`large_memory_threshold_gib` use the `large_memory` partition role. The generated
worker retains the largest request as a safe direct-submission fallback, while
`submit.sh` applies the task-specific overrides used for a normal campaign launch.
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
not completed frame coverage. Their elapsed CPU time is a cost lower bound that
receives the configured censored-timeout safety factor before planning.

The Unix group and path fields are retained provenance and operator configuration;
the launcher does not run `chgrp`, move data, or change input permissions. Update the
profile if DEAC changes its partitions, account/QoS policy, or validated environment.
