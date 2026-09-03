# Analyze a zinc-finger trajectory on a workstation

This tutorial takes you from three simulation files to a planned and executed
analysis. The example is a real 1,000-frame subset of a published
Salsbury-group simulation of the 28-residue zinc-finger domain of human NEMO.
It contains the protein, its hydrogens, and one zinc ion.

The source publication showed that short zinc-finger trajectories miss rare
conformations found by longer simulations. This exercise is deliberately small
enough for a modest workstation. Its config sets 32 GiB as a hard campaign
ceiling, not as an expected allocation: the accepted run peaked near 300 MiB,
and the current conservative plan estimates a 2.4 GiB task maximum before
execution-adapter overhead. It teaches the workflow but does **not** reproduce
the paper or provide enough sampling for a scientific conclusion.

## What you will learn

You will:

1. identify the structure, connectivity, trajectory, and physical time input;
2. let the generic workflow recognize a protein-plus-zinc system;
3. inspect its automatic module and frame/resource plan before execution;
4. run applicable analyses locally with a two-CPU, one-hour ceiling; and
5. distinguish technical completion from scientific interpretation.

No NEMO-specific analysis code is used. The same `prepare-analysis` command is
the normal entry point for another protein, nucleic acid, complex, ligand,
cofactor, or ion-containing system.

## The four inputs

| Input | File or value | Why it is needed |
| --- | --- | --- |
| Coordinates and atom identities | `data/nemo_zinc_finger.pdb` | Names the 423 atoms and supplies the reference coordinates. |
| Explicit bonds | `data/nemo_zinc_finger.psf` | Supplies 426 bonds, including the chemically prepared protein topology. No bonds are guessed by distance. |
| Time-ordered coordinates | `data/nemo_zinc_finger_1000_frames.dcd` | Contains source frames 0–999 in their original order and retains unit-cell records. |
| Saved-frame interval | `0.2 ps` | Comes from the matching archived NAMD settings: a 2 fs timestep and `dcdfreq 100`. |

The DCD header was rewritten by a trajectory plugin and cannot establish its
physical interval by itself. Supplying the independently recorded interval is
therefore mandatory. Full hashes, derivation details, publication identifiers,
and limitations are in
[`data/FIXTURE_PROVENANCE.json`](data/FIXTURE_PROVENANCE.json).
The group-generated PSF and tutorial trajectory subset are available under CC
BY 4.0 as described in the repository's [`LICENSE-DATA.md`](../../LICENSE-DATA.md).
The PDB-derived starting coordinates retain their PDB 2JVX provenance.

## 1. Create an environment

From the repository root, create a fresh Python 3.10–3.12 environment and
install the package with its conventional clustering dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[clustering]'
```

HDBSCAN and PaLD are off by default. OpenMM is unnecessary because this example
already has a PSF. DSSP is optional; if `mkdssp` is not installed, the generic
workflow omits that external-tool module and records why.

## 2. Prepare the analysis

Still at the repository root, choose a new output directory and run:

```bash
salsbury-md-analysis prepare-analysis \
  --pdb tutorials/nemo_zinc_finger/data/nemo_zinc_finger.pdb \
  --psf tutorials/nemo_zinc_finger/data/nemo_zinc_finger.psf \
  --trajectory tutorials/nemo_zinc_finger/data/nemo_zinc_finger_1000_frames.dcd \
  --frame-interval-ps 0.2 \
  --project-id nemo-zinc-finger-tutorial \
  --config tutorials/nemo_zinc_finger/analysis-config.json \
  --output nemo-zinc-finger-tutorial-run
```

Preparation reads the inputs but does not modify them. It writes manifests,
the inferred chemical context, topology-derived conformational views, integer
frame strides, a shared campaign resource plan, and local worker scripts into
the new output directory. It fails instead of silently replacing a nonempty
output directory.

## 3. Read the plan before running

These are the most useful planning files:

- `module-coverage.json`: which analyses are enabled, inapplicable, or skipped;
- `automatic-chemical-context.json`: the inferred protein, ion, and analysis
  groups;
- `sampling-plan.json`: selected source frames and every integer stride;
- `campaign-resource-plan.json`: estimated CPU time, memory, and the shared
  two-CPU/one-hour campaign ceiling;
- `memory-feasibility-report.json`: an exact shortfall and the smallest module
  switches needed only when the requested memory ceiling cannot hold every
  technical minimum;
- `conformational-views.json`: the coordinate bases used for PCA, FES, and
  clustering; and
- `analysis-config.json`: the complete resolved configuration, including
  defaults not written in the small tutorial override.

The workflow should identify one protein system and one zinc ion. Zinc geometry
and species-resolved ion-atmosphere analysis are applicable. Water-network and
nucleic-acid modules are not, because those atoms are absent. The tutorial
configuration leaves all applicable methods enabled, keeps the optimized
coordinate cache on `auto`, and disables multi-frame state-trajectory exports
to keep the exercise small. Representative structures remain enabled.

### If your memory ceiling is lower

The value in `execution.maximum_memory_gib` is the maximum memory that all
concurrently running tasks may reserve. It is not the planner's prediction and
it is not a requirement to have that much installed memory. Read the individual
task estimates in `campaign-resource-plan.json`; the local or Slurm adapter adds
its documented scheduling safety margin when it builds the final requests.

If an enabled task cannot meet its technical minimum under your ceiling,
preparation fails before launching work. The failure states the largest
estimate, the shortfall, the rounded-up ceiling needed to retain everything,
and the exact module or clustering-method switches that would have to be turned
off. The requested analysis is never silently reduced.

To accept that reduced analysis explicitly, copy the tutorial config, lower
`execution.maximum_memory_gib`, choose a fresh output directory, and add:

```bash
--auto-disable-to-fit-memory
```

to the same `prepare-analysis` command. The new directory preserves
`analysis-config.requested.json`, writes the complete reduced configuration to
`analysis-config.memory-fit.json`, and explains every change in
`memory-feasibility-report.json`. Review those files before running
`./run-local.sh`. This option does not lower frame minima and does not hide a
remaining CPU-time, wall-time, calibration, or scratch-space failure.

As a bounded check of this path, the current planner was given a 1.5 GiB
ceiling for this fixture. It reported a 2.4 GiB SASA minimum, recommended a
3 GiB ceiling to retain every enabled analysis. Only with the explicit flag did
it write a reduced config with SASA off and a 1.2 GiB largest remaining
minimum. These numbers are calibration-dependent examples, not universal
requirements.

## 4. Run locally

```bash
cd nemo-zinc-finger-tutorial-run
./run-local.sh
```

The local adapter honors the same dependency graph and resource accounting as
the Slurm adapter. It may run independent tasks concurrently, but it will not
exceed the configured aggregate two-CPU or 32 GiB memory limits. The memory
ceiling is deliberately conservative and is not preallocated; actual peak use
is reported after the run.

## 5. Check completion and results

Start with:

- the newest JSON file under `local-execution-status/`, which reports whether
  every scheduled stage completed and identifies each task's log;
- `preflight.report.json`, which records input and topology checks;
- `analysis_resource_and_frame_table.md`, which gives CPU time, peak memory,
  selected physical frames, and clustering observation counts by method; and
- `prioritized_findings.md`, which points to notable FES, clustering, RMSF,
  interaction, and ion results without declaring them biologically important.

Detailed machine-readable reports live under `results/`. For this system,
inspect the structural-QC report before any plotted result, then the PCA/FES and
clustering reports, and finally the zinc geometry and ion-atmosphere reports.
Representative structures are written for observed states even though the
tutorial turns off multi-frame state trajectories.

A report with `technical_status: complete` passed its software contract. The
tutorial's `scientific_status` remains `not evaluated`: 1,000 early frames from
one trajectory cannot establish convergence, equilibrium populations,
metastability, kinetics, zinc affinity, or a biological binding mechanism.

### Reference acceptance run

The tutorial was exercised from a clean wheel installation on an Apple silicon
workstation with Python 3.12. The known-good run completed 28 module
reports in 7 minutes 55 seconds, used 0.155 measured CPU-hours, and reached a
maximum measured resident memory of 300 MiB. Every error log was empty and no
temporary report remained. The planner used all 1,000 frames for each
frame-based result except the shared PCA basis, which used a uniform integer
stride of 2 (500 frames) and projected all 1,000 frames afterward. These values
are acceptance evidence for this fixture and machine, not general performance
guarantees. The hash-bound summary is retained in
[`validation/nemo_zinc_finger_tutorial_acceptance.json`](../../validation/nemo_zinc_finger_tutorial_acceptance.json).

A separate release-candidate run installs the built wheel into an empty Python
3.12 environment, runs the full tutorial, checks every report and sidecar hash,
and previews the Slurm plan without submitting it. Its record is
[`validation/release_candidate_20260902.json`](../../validation/release_candidate_20260902.json).

## 6. Use your own trajectory

Replace the PDB, PSF/connectivity file, DCD paths, and frame interval in the
preparation command. Repeat `--trajectory` once per independent replica. Keep
one continuous segment per trajectory for time-lagged analyses, and never join
MSM transitions across replica or restart boundaries.

For a full study, increase the CPU/time envelope in a copied config, retain the
entire accepted production interval, and review automatic chemistry and common
atom selections before execution. Use `prepare-comparison` when comparing
controls, variants, ligands, or conditions on a shared PCA basis.

## Citation and provenance

The simulation subset is associated with:

> Godwin R, Gmeiner W, Salsbury FR Jr. Importance of long-time simulations for
> rare event sampling in zinc finger proteins. *Journal of Biomolecular
> Structure and Dynamics*. 2016;34(1):125–134.
> <https://doi.org/10.1080/07391102.2015.1015168>

The starting NMR structure is [PDB 2JVX](https://www.rcsb.org/structure/2JVX),
“Solution Structure of human NEMO zinc finger,”
<https://doi.org/10.2210/pdb2JVX/pdb>.

Please also cite the Salsbury MD Analysis release or commit used for the run.
The full source and derived-file hashes are retained in the fixture provenance
record. A separate explicit data-license decision is required before this
scientific fixture is included in a public repository release.
