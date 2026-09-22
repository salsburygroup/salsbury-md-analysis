# Analyze your own trajectories on a Slurm cluster

Read [Resource settings and planning limits](../RESOURCE_PLANNING.md) before
choosing a budget or interpreting planner and scheduler estimates.

This is a how-to guide for running your files through a Slurm scheduler. It
assumes that the core package and optional interactive viewer are installed in
an environment visible from both the login and compute nodes. Use the
[NEMO cluster tutorial](../nemo_zinc_finger_cluster/README.md) to configure and
test a site profile with supplied inputs first. WFU users should use the
separate [DEAC tutorial](../nemo_zinc_finger_deac/README.md).

## Prepare a site profile

Copy [`generic-template.json`](../../profiles/slurm/generic-template.json) to a
location you control on shared storage. Fill in the settings for your cluster:

- submit, status, and cancel commands;
- account, Unix group, QoS, and partitions;
- partition wall-time and node limits;
- the absolute Python executable and optional package checkout;
- environment setup commands and variables;
- shared output and scratch roots;
- node CPU and memory shape; and
- scheduler-only time and memory padding.

Set `resource_policy.walltime_safety_factor` to `1.0`. The campaign planner has
already applied its task-time factor. Use `walltime_overhead_minutes` for an
explicit scheduler-only allowance.

The generic profile contains placeholders rather than safe defaults for your
site. Confirm the values with the cluster documentation or administrators.
Keep the profile beside the campaign as provenance.

## Gather shared inputs

Place the PDB, matching bond topology, and trajectories on storage visible from
every compute node. Supply one DCD per independent replica or continuous
segment and obtain the saved-frame interval from the simulation settings. The
PDB, topology, and trajectories must use the same atom order.

Do not place a prepared campaign under node-local scratch. The planner may use
scratch for temporary work when the site profile permits it, but completion
reports, logs, ledgers, and accepted results belong on durable shared storage.

## Create the Slurm study

```bash
salsbury-md-analysis init /shared/path/my-study \
  --pdb /shared/path/inputs/system.pdb \
  --connectivity /shared/path/inputs/system.psf \
  --trajectory /shared/path/inputs/replica-1.dcd \
  --trajectory /shared/path/inputs/replica-2.dcd \
  --frame-interval-ps 100 \
  --cpus 16 \
  --memory-gib 64 \
  --hours 24 \
  --adapter slurm \
  --slurm-profile /shared/path/config/my-cluster.json
```

The numbers above are illustrative campaign ceilings, not calibrated requests
for your files. Use a workload- and hardware-matched catalog when available,
review the generated estimates, and require successful planning before
execution. Follow the [budget-recovery recipe](../RESOURCE_PLANNING.md#if-planning-rejects-the-budget)
when a limit is insufficient; retain the scientific scope unless you explicitly
choose a different analysis.

Use `--interactive` for several systems or conditions. Review the resulting
`study.json` and `analysis-config.json`, including the input paths, replica
grouping, chemistry, comparisons, module switches, exports, resource limits,
and selected profile.

## Check and plan on the submission host

```bash
salsbury-md-analysis doctor /shared/path/my-study/study.json
salsbury-md-analysis plan /shared/path/my-study/study.json
```

`doctor` checks scheduler commands, software, optional executables, file
headers, and atom counts. `plan` writes the prepared directory without
submitting. Review `planning-report.md`, `module-coverage.json`,
`automatic-chemical-context.json`, `campaign-resource-plan.json`,
`scheduler-resource-requests.json`, and the retained `slurm-profile.json`.

The plan reports selected physical frames, effective raw strides, task
dependencies, partitions, CPUs, memory, nodes, and time. Resolve an infeasible
plan before submission. Do not treat a queue estimate as a reservation or a
software plan as evidence that the scientific sampling is adequate.

## Preview without submitting

```bash
cd /shared/path/my-study/analysis
./submit.sh --preview
less slurm-submission-preview.json
```

Confirm that the generated schedule is feasible and that every request matches
site policy. The preview submits no jobs.

An optional read-only capacity check can report current node fit and queue
pressure:

```bash
salsbury-md-analysis advise-slurm-capacity \
  /shared/path/my-study/analysis \
  --wall-hours 24 \
  --cpu-ceiling 16 \
  --format markdown
```

## Submit and monitor

The following command submits the reviewed Slurm jobs:

```bash
salsbury-md-analysis run /shared/path/my-study/analysis
```

Returned job IDs are recorded in `submission-ledgers/`. Submission does not
mean that the campaign completed. Check the scheduler and accepted artifacts:

```bash
squeue --me
salsbury-md-analysis status /shared/path/my-study/analysis
salsbury-md-analysis status /shared/path/my-study/analysis --json
```

Do not submit a duplicate campaign while jobs are queued or running. If work
stops, inspect `status`, retained attempt logs, and scheduler state. Review
recovery without executing it:

```bash
salsbury-md-analysis resume /shared/path/my-study/analysis
```

Use `--execute` only after confirming that no earlier job remains active and
that the proposed recovery is correct. The recovery reuses accepted reports and
preserves failed evidence.

## Build the interactive report

Use the companion's
[cluster interactive guide](https://github.com/salsburygroup/salsbury-md-analysis-interactive/blob/main/tutorials/cluster/README.md)
for the completion check, report build, archive, transfer, and review steps.

After `status` reports all scheduled tasks complete, build on an approved
compute allocation or a suitable workstation. The separate viewer build is
not covered by the core campaign budget; follow site policy for login-node use:

```bash
salsbury-md-analysis-interactive /shared/path/my-study/analysis
```

Transfer the complete `interactive-report/` directory to a workstation and
open `index.html`. The viewer reads completed reports; it does not rerun the
analysis.

`technical_status: complete` means the software outputs passed their declared
contracts. Scientific conclusions still require review of chemistry, sampling,
convergence, uncertainty, and method assumptions.
