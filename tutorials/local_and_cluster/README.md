# Run the same analysis locally or on a cluster

This guide covers the normal terminal workflow for a new molecular-dynamics
system. The planner, local runner, and Slurm runner use the same task graph and
the same complete-interval frame indices. Moving a prepared analysis to a
cluster does not change which physical frames a module reads.

Technical completion means that the commands and report contracts passed. It
does not establish scientific validity, convergence, metastability, kinetics,
binding, or biological importance.

## 1. Install the command

Use Python 3.10 through 3.12 in a clean environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'salsbury-md-analysis[clustering]'
salsbury-md-analysis --help
```

For a source checkout, replace the installation line with
`python -m pip install -e '.[clustering]'` from the repository root.

## 2. Identify the inputs

Provide one atom-order-matched structure, one explicit bond topology, and one
trajectory for each independent replica. Supported topology/connectivity
combinations include PDB plus PSF, Amber PRMTOP/PARM7, or portable
`salsbury-bonds-v1` JSON. Production trajectories use DCD. Record the physical
time between saved frames from the simulation settings; a trajectory header is
not always sufficient evidence.

Do not join separate replicas or restart segments across a time-lagged
transition. Keep the original files immutable and write the analysis to a new
directory.

## 3. Prepare and inspect without running

Start with `--plan-only`:

```bash
salsbury-md-analysis prepare-analysis \
  --pdb system.pdb \
  --connectivity system.psf \
  --trajectory replica-1.dcd \
  --trajectory replica-2.dcd \
  --frame-interval-ps 10 \
  --project-id example-study \
  --config analysis-config.json \
  --output example-study-analysis \
  --plan-only
```

Review these files before execution:

- `preflight.report.json` for file, atom-order, cell, and topology checks;
- `automatic-chemical-context.json` for protein, nucleic-acid, ligand,
  cofactor, solvent, and ion assignments;
- `module-coverage.json` for enabled, inapplicable, and deferred methods;
- `sampling-plan.json` for exact source-frame indices and integer strides;
- `campaign-resource-plan.json` for calibrated CPU, memory, wall, and spatial
  work estimates; and
- `planning-report.md` for the compact human review.

Stop if the composition, chemistry, time interval, or selected production
range is wrong. A technically valid input can still be the wrong scientific
input.

### Does structural QC need every saved frame?

Usually no. Structural QC uses its planner-selected complete-interval indices,
one worker per replica, and frame-local whole-molecule repair. It does not make
an extra full-system unwrapped copy. Hydrogen-bond and ion-atmosphere work also
reads the planned frames and uses periodic minimum-image geometry. A module
reads every saved frame only when its accepted plan selects every frame or its
documented calculation requires that interval.

Reducing structural-QC sampling should follow the plan's stated scientific
minimums. Raising wall time can be appropriate when the minimum itself is
required. Do not lower QC weight merely to hide a timeout; first check whether
the estimate, worker layout, or failed input is responsible.

## 4. Run on a workstation

Prepare again without `--plan-only`, using a fresh output directory, then run:

```bash
cd example-study-analysis
./run-local.sh
```

The local runner respects the configured aggregate CPU and memory limits.
Independent tasks can run together, while each replica's structural-QC work
uses one worker.

## 5. Understand automatic recovery

Prepared campaigns set `execution.autorecovery` to `true` by default and allow
two attempts unless the config says otherwise. A failed or timed-out task keeps
its first attempt log, retries within the same declared limits, and reuses an
accepted report or checkpoint when that is safe. Dependent work is released
only after the required report passes. The final status distinguishes a first-
attempt completion from `recovered_complete`.

To require a single attempt, set:

```json
{
  "execution": {
    "autorecovery": false,
    "maximum_task_attempts": 1
  }
}
```

If recovery also fails, inspect every attempt's stderr, exit code, peak memory,
elapsed time, partial output, and dependency state before changing the plan.
Make the smallest versioned repair and keep the failure evidence.

## 6. Move the workflow to Slurm

Copy `profiles/slurm/generic-template.json`. Fill in values confirmed by your
cluster owner: scheduler commands, account, group, QoS, partitions, node size,
environment setup, and allowed storage roots. Point
`execution.slurm_profile` in the analysis config to that copy and set
`execution.submission_adapter` to `slurm`.

Prepare with `--plan-only` first. In the prepared directory, inspect the
scheduler requests without submitting:

```bash
./submit.sh --preview
```

To compare one through sixteen 44-core, 185-GiB nodes without submitting work,
run:

```bash
salsbury-md-analysis plan-node-sweep campaign-resource-plan.json \
  --cpus-per-node 44 \
  --memory-gib-per-node 185 \
  --maximum-nodes 16 \
  --maximum-wall-hours 168 \
  > node-sweep.json
```

The sweep first plans at the maximum allocation. It reads the complete enabled
task graph, replica-worker caps, dependency stages, execution bundles, and
safety-adjusted memory requests. It then checks smaller allocations only up to
the largest node count that the maximum-node schedule can use. Larger points
remain in the curve as explicit replays and cannot gain information.

Read `task_inventory_ceiling` for that maximum useful node count. Read
`threshold_sensitivity` for the first node count reaching 75%, 80%, 90%, 95%,
99%, or 100% of the best information score. Each threshold row also reports
the mean and median multiple of the registered scientific minimum. These are
planning comparisons, not evidence of adequate sampling.

Pareto filtering is always active. The default `minimum_nodes` policy selects
the smallest-node point on the front. Pass
`--pareto-selection-policy balanced` to give equal weight to the active Pareto
dimensions. The selected point is recorded in
`operational_balance`; no jobs are submitted by this command.

To compare modeled wall time and information within a fixed node limit, add
`--pareto-objectives walltime_information`. The default
`nodes_walltime_information` mode also treats node count as a Pareto objective.
Both modes apply `--maximum-nodes` before computing the front.

The preview can refuse submission even when the local tutorial completed. A
cluster's minimum job reservation and scheduler overhead can make a short
workstation wall limit impossible. In that case, prepare a new plan with at
least the reported campaign-wall requirement and inspect it again; do not
bypass the refusal. Only after the preview matches local policy should you run
`./submit.sh`.
Submission uses the planned dependencies and task-specific CPU, memory, and
wall requests. A terminal Slurm failure is handled under the same bounded
recovery policy; a dependency is not accepted because its predecessor merely
exited.

Never submit a full hydrogen-bond candidate-by-frame table. The current engine
keeps the endpoint universe implicit, locates only spatially possible events,
and accounts for implicit zeros in the downstream comparison.

## 7. Verify completion

Before accepting a run, check:

1. the final local or Slurm status says every required task is complete or
   `recovered_complete`;
2. every declared dependency points to the accepted predecessor report;
3. unexpected stderr and error arrays are empty;
4. report hashes and source-content hashes match;
5. the frame count, physical interval, and source indices agree across the
   planner, workers, and final report;
6. measured CPU, memory, wall time, and spatial-work counters are present; and
7. no temporary report or partial result was mistaken for an accepted output.

Then perform scientific review separately. Examine equilibration, replica
agreement, sampling adequacy, method assumptions, and the question-specific
acceptance criteria before drawing a conclusion.

## 8. Use the tests and examples

Run the complete source test suite with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```

The NEMO tutorial supplies a redistributable protein-plus-zinc case. The test
suite also covers protein, DNA, modified residues, ligands, ions, periodic
cells, sparse hydrogen-bond accounting, planner/worker frame agreement, and
recovery failure injection.

Two additional retained-input fixtures were used for bounded technical checks:

- TREX control and 8OG systems retained protein, a DNA oligomer, modified DNA,
  four Mg ions, sodium, and chloride. The check found all four directed
  protein/DNA hydrogen-bond strata, preserved the control/8OG residue-name
  difference under positional mapping, and did not materialize a pre-coordinate
  candidate table.
- The verified non-`SCREEN/` thrombin input contained protein, sodium, and
  chloride. It therefore tested only those entities; it did not support a
  ligand or cofactor claim.

In all three bounded fixtures, translating ions by full lattice vectors left
exact minimum-image distances unchanged within `1.1e-14` angstrom. No
simulation job or full campaign was submitted. The hash-bound result is in
[`validation/trex_thrombin_technical_fixture_acceptance.json`](../../validation/trex_thrombin_technical_fixture_acceptance.json).
TREX remains unsuitable for biological conclusions until its earlier
scientific-QC concerns receive explicit human acceptance.

For method definitions and citations, see
[`docs/METHODS_AND_CITATIONS.md`](../../docs/METHODS_AND_CITATIONS.md). Cite the
software release or commit used for each analysis, and record any external
method software and simulation inputs needed to reproduce the result.
