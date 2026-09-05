# Run without an AI assistant

These commands are in main candidate 0.1.3rc1 and experimental candidate
0.2.0a3. Both are unreleased. The published v0.1.2 tag is unchanged and does
not contain them. No AI service, account, or API key is needed.
Use Linux, macOS, or a Linux environment inside Windows WSL2.
Native Windows execution is not supported. WSL2 acceptance remains a release
check until it has been exercised on a Windows host.

## Install once

Use Python 3.12. From the candidate source directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
salsbury-md-analysis doctor
```

This installs the Python dependencies used by the default modules. For a
pinned deployment, install the candidate wheel and keep its SHA-256 and the
platform's resolved dependency list with the results. Do not substitute a
moving Git branch for the version used by an accepted campaign.

Protein secondary structure also needs `mkdssp`. The repository's
`environment.yml` supplies a reviewed Conda environment including DSSP.
X3DNA-DSSR is separately licensed and is not bundled; intrinsic DNA geometry
does not require it. HDBSCAN and OpenMM connectivity preparation remain optional.
See [dependencies and licenses](../DEPENDENCIES_AND_LICENSES.md).

## Create the study

You need a PDB, its matching bond topology (PSF, PRMTOP/PARM7, or bond JSON),
one DCD per independent replica, and the time between saved frames in ps.
Atom order must agree. Do not concatenate independent replicas or disconnected
trajectory pieces and call them one continuous replica.

```bash
salsbury-md-analysis init my-study \
  --pdb system.pdb --connectivity system.psf \
  --trajectory rep1.dcd --trajectory rep2.dcd \
  --frame-interval-ps 100 --cpus 8 --memory-gib 32 --hours 24
```

For several systems, use `init my-study --interactive` with the same resource
flags. It asks for each system's name, structure, topology, frame interval, and
replicas. The interactive setup is ordinary terminal input, not an AI chat.

`my-study/study.json` is the editable input list. Relative input paths in that
file resolve against its directory. `my-study/analysis-config.json` contains
module switches, protected dependencies, comparisons, optional analyses,
exports, and resource limits. Change these before planning. A module disabled
by configuration is distinct from one that is inapplicable or cannot fit.
Protected prerequisites cannot silently disappear to make a budget pass.

## Check, plan, then decide

```bash
salsbury-md-analysis doctor my-study/study.json
salsbury-md-analysis plan my-study/study.json
```

Neither command executes analyses or submits jobs. `doctor` checks platform,
Python packages, external commands, file headers, and matching atom counts.
It cannot verify that a protonation state, ligand, replica grouping, or
scientific comparison is the one you intended.

`plan` prints `planning-report.md` and writes a prepared directory at
`my-study/analysis`. Review its per-method **effective raw stride**, selected
frames, off/deferred reasons, and padded CPU, memory, and wall-time requests.
For an insufficient budget it exits with an error and preserves the proposed
plan and explanations. Do not run an infeasible plan. A time estimate excludes
queue wait and is not a promise that a cluster will start or finish on schedule.

Planning uses the existing scientific-minimum and two-stage cache/method
stride rules. The new commands do not introduce a second planner.

## Run and inspect

```bash
salsbury-md-analysis run my-study/analysis
salsbury-md-analysis status my-study/analysis
salsbury-md-analysis status my-study/analysis --json
```

Local execution stays on the current computer and uses the prepared aggregate
limits. `status` separates validated completion artifacts from live controller
and Slurm activity. It lists missing/invalid outputs, known failed attempts,
and true prerequisite blockers. An absent report alone does not prove a job
failed. Inspect the named logs when a task needs attention.

For Slurm, start with `profiles/slurm/generic-template.json`, not the DEAC
profile. Fill in your site's account, partitions, time limits, node capacities,
Python environment, and storage roots. Pass its absolute path with
`init --adapter slurm --slurm-profile /path/to/site.json`. Plan on the cluster
submission host. `run` prints the scheduler preview, submits it, and records
each returned job ID in `submission-ledgers/`. A partial submission is not a
completed campaign. The DEAC profile is an example for that cluster only.

Advanced users can keep their own launcher: the prepared
`launcher-contract.json` describes tasks, dependencies, completion checks,
commands, and resources. Set the adapter to `custom`; this front end will not
submit or resume work owned by another launcher.

## Resume without repeating accepted work

```bash
salsbury-md-analysis resume my-study/analysis
salsbury-md-analysis resume my-study/analysis --execute
```

The first command is a review only. The second reuses validated completed
reports and runs unfinished work. It refuses active Slurm work, an unavailable
scheduler query, and existing outputs that fail acceptance. Diagnose and
preserve failed evidence before preparing a versioned repair. It does not
raise limits, change sampling, repair chemistry, or erase partial output.
The local lock also covers the older `run-local-workflow` entry point.

Each resumed invocation has the prepared execution envelope; it does not refund
CPU time spent on earlier attempts. Retain attempt logs when reporting total
campaign cost. Moving a runnable campaign to a different filesystem may require
a fresh plan because validated source/cache paths are explicit; copying a
dashboard is different and is supported offline.

## Read the results

Start with `prioritized_findings.md`, the figures and CSV tables under
`presentation-artifacts/`, and `analysis_resource_and_frame_table.md`.
QC has its own report. JSON retains the complete evidence and provenance; it
is not the intended first page for a person.

The optional viewer runs **after** core main or experimental:

```bash
salsbury-md-analysis-interactive my-study/analysis
```

Open the generated `index.html` locally. Copy the entire report directory,
including `evidence/`, when sharing it. HTML alone does not contain every raw
file. The viewer is not another analysis engine and does not restart a campaign.

## If something goes wrong

| Symptom | Next step |
|---|---|
| Missing package or executable | Read `doctor`; install the named dependency or explicitly disable an optional module. |
| Too little memory/time | Read the plan's minimum request and reduction proposal; edit the config and plan into a new directory. |
| Invalid output | Preserve it, inspect stderr and the acceptance reason, and prepare an isolated repair. |
| Slurm query fails | Check cluster access. Resume refuses to guess whether earlier jobs are active. |
| No primary clustering | Read the comparison contract. Different observations, geometry, missing labels, or unresolved legacy evaluation can prevent a fair ranking. |
| Missing dashboard evidence | Copy the complete report folder, not just its HTML file. |
