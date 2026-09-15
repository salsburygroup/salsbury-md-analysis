# Analyze your own trajectories on a workstation

This is a how-to guide for a local analysis. It assumes that you have installed
a reviewed Salsbury MD Analysis version or recorded source commit and that the
`salsbury-md-analysis` command is available. Use the
[NEMO workstation tutorial](../nemo_zinc_finger_workstation/README.md) if you
want a complete supplied example first.

## Gather the inputs

For each molecular system, identify:

- one PDB with the atom names and reference coordinates;
- the matching bond topology as PSF, PRMTOP/PARM7, or
  `salsbury-bonds-v1` JSON;
- one DCD for each independent replica or continuous segment; and
- the time between saved frames in picoseconds, taken from the simulation
  settings rather than inferred from a rewritten trajectory header.

The PDB, topology, and every trajectory must use the same atom order. Keep
independent replicas separate so time-lagged analyses never create transitions
across replica boundaries.

## Create the study

Choose CPU, aggregate memory, and complete-campaign wall limits that your
workstation can sustain:

```bash
salsbury-md-analysis init my-study \
  --pdb /absolute/path/system.pdb \
  --connectivity /absolute/path/system.psf \
  --trajectory /absolute/path/replica-1.dcd \
  --trajectory /absolute/path/replica-2.dcd \
  --frame-interval-ps 100 \
  --cpus 4 \
  --memory-gib 16 \
  --hours 8
```

For several systems or conditions, use `init my-study --interactive` with the
same resource flags. The terminal wizard asks for each system and its separate
trajectories.

`init` writes `my-study/study.json` and
`my-study/analysis-config.json`. It does not run an analysis. Review the input
paths, replica grouping, frame interval, automatic chemistry, module switches,
state exports, comparisons, and resource limits before continuing.

## Check and plan

```bash
salsbury-md-analysis doctor my-study/study.json
salsbury-md-analysis plan my-study/study.json
```

`doctor` checks the Python environment, optional executables, file headers, and
matching atom counts. It cannot confirm that protonation, ligand identity,
replica grouping, or the scientific comparison is correct.

`plan` writes `my-study/analysis` without executing it. Read:

- `planning-report.md` for selected frames, effective raw strides, disabled or
  deferred methods, and the campaign request;
- `automatic-chemical-context.json` for inferred molecular groups;
- `module-coverage.json` for every applicable, inapplicable, disabled, and
  deferred module; and
- `campaign-resource-plan.json` for task estimates, dependencies, memory, CPU,
  and time.

An infeasible plan is not runnable. Change the inputs, configuration, or
resource limits deliberately and create a new prepared directory. The planner
does not silently discard protected analyses or lower scientific frame minima.

## Run and check completion

```bash
salsbury-md-analysis run my-study/analysis
salsbury-md-analysis status my-study/analysis
salsbury-md-analysis status my-study/analysis --json
```

Local execution honors the prepared dependency graph and aggregate CPU and
memory limits. A zero exit code is not the only completion check: declared
reports and their hashes must also pass validation.

If the run stops, inspect `status` and the named logs. Review recovery without
executing it:

```bash
salsbury-md-analysis resume my-study/analysis
```

Run `resume my-study/analysis --execute` only after reviewing the proposed
unfinished work. Accepted reports are reused. Invalid outputs are preserved and
must be diagnosed rather than overwritten.

Automatic recovery is controlled by `execution.autorecovery`. Read the
prepared configuration and recovery policy before enabling or disabling it.

## Read and browse the results

Start with `prioritized_findings.md`, `planning-report.md`,
`analysis_resource_and_frame_table.md`, and the files under
`presentation-artifacts/`. Inspect structural QC before interpreting molecular
states, distributions, interactions, or representative structures.

After the core campaign is complete, the optional companion package can build
an offline browser:

```bash
salsbury-md-analysis-interactive my-study/analysis
```

Open `my-study/analysis/interactive-report/index.html`. Copy or share the whole
`interactive-report/` directory so its evidence links remain intact.

`technical_status: complete` means the software contract passed. It does not
establish convergence, equilibrium populations, kinetics, mechanism, or
publication readiness.
