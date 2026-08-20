# Local and Slurm execution

Scientific configuration and execution-site configuration are separate. The
analysis config chooses an adapter, while a Slurm profile describes one cluster.
Neither adapter changes module selection, frame strides, definitions, dependencies,
or report/hash contracts.

In other words, choose `local` when you want the package to manage work on the
current computer, and choose `slurm` when a cluster scheduler should manage it.
The scientific plan is the same either way; only the way resources are requested
and jobs are launched changes.

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

The profile schema is `salsbury-slurm-profile-v1`. It records scheduler submit,
status, and cancel commands; account, Unix group, QoS, and role-specific partitions;
Python and package paths; environment setup commands and variables; shared-write
umask; storage and scratch roots; and conservative resource policy metadata. The
adapter converts every planner task estimate to a time and memory request using the
profile safety factors. An array receives the maximum request among its elements,
because Slurm cannot request different resources for individual elements of one
array. `scheduler-resource-requests.json` records every mapped planner task, margin,
array aggregation, selected partition, and final request. Requests that cross
`large_memory_threshold_gib` automatically use the `large_memory` partition role.
Copy
`profiles/slurm/generic-template.json`, review every value with the cluster owner,
then prepare and run `./submit.sh`. The exact normalized profile is retained beside
the generated workflow as `slurm-profile.json`.

Setup commands are literal reviewed shell lines and therefore belong only in a
trusted, version-controlled profile. Scheduler command fields accept one executable
name or absolute path, environment variable names are validated, and additional
directives must begin with `#SBATCH --`.

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
