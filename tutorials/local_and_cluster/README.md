# First terminal run: local or Slurm

This tutorial uses the unreleased candidate's `init → doctor → plan → run`
workflow. For installation, troubleshooting, and file conventions, use the
[terminal guide](../../docs/TERMINAL_WORKFLOW.md). No AI assistant is required.

## Inputs

Use your simulation's PDB and matching PSF, PRMTOP/PARM7, or bond JSON, plus
one DCD for each independent replica. Find the saved-frame interval in the
simulation configuration; supply it in ps. Do not join unrelated replicas or
discontinuous segments for kinetic analysis.

Paths containing spaces work when quoted:

```bash
salsbury-md-analysis init "my first study" \
  --pdb "input files/system.pdb" \
  --connectivity "input files/system.psf" \
  --trajectory "input files/replica 1.dcd" \
  --trajectory "input files/replica 2.dcd" \
  --frame-interval-ps 100 \
  --cpus 4 --memory-gib 16 --hours 8
```

For a comparison, use `--interactive` instead of the file flags and enter all
systems. The wizard writes `study.json` and a full editable
`analysis-config.json`. Review chemistry, replica grouping, module switches,
and resource limits before planning.

## Review before execution

The older `prepare-analysis --plan-only` command remains available. The new
`plan` command is always review-only. For a Slurm preview without submission,
`./submit.sh --preview` remains available in a prepared workflow.

```bash
salsbury-md-analysis doctor "my first study/study.json"
salsbury-md-analysis plan "my first study/study.json"
```

The terminal prints a plan, not results. Check the effective raw stride and
frames for each method, off/deferred reasons, and padded resource requests.
An insufficient envelope stops preparation with an explanation. Do not treat
an infeasible proposal as a runnable plan.

The prepared directory also has the chemical-context, preflight, module
coverage, sampling, and resource records. Planning does not modify your
simulation files. Structural QC may use a validated coordinate cache;
continuous unwrapping scans required source frames before a strided cache is
written. QC then reads its planned retained frames, with replica workers.
Those are different stages with separately accounted costs.

## Execute and recover

```bash
salsbury-md-analysis run "my first study/analysis"
salsbury-md-analysis status "my first study/analysis"
# Review unfinished work; this does not execute it:
salsbury-md-analysis resume "my first study/analysis"
# Execute the reviewed recovery:
salsbury-md-analysis resume "my first study/analysis" --execute
```

Accepted outputs are reused. Invalid outputs are preserved and require
diagnosis. Automatic retries stay within the prepared limits; they do not
invent new scientific settings or erase failure evidence.
Set `execution.autorecovery` to `false` to disable automatic retries.

For Slurm, create a site profile from
[`generic-template.json`](../../profiles/slurm/generic-template.json), fill in
your cluster's settings, and pass `--adapter slurm --slurm-profile /path/site.json`
to `init`. Plan on the submission host. Review the scheduler preview before
`run` submits anything. Each successful submission is recorded in a task/job
ledger. Use the DEAC profile only for that cluster.

## Read and share

Start with `prioritized_findings.md` and the figures/CSV tables in
`presentation-artifacts/`. Keep QC separate from physical findings. The
resource table reports measured CPU time, peak memory, and frame coverage.

To browse results, install the optional interactive package and run:

```bash
salsbury-md-analysis-interactive "my first study/analysis"
```

Open the generated HTML. Share the whole report folder so its evidence links
remain intact. Technical completion does not establish scientific validity;
interpretation still needs your review.
