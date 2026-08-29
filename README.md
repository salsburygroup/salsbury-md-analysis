# Salsbury MD Analysis

Reusable Python tools for molecular-dynamics trajectory analysis, teaching, and reproducible computational research.

If you have a reference PDB, the matching bond topology, and one or more DCD
trajectories, the quickest place to start is [Use](#use). The workflow will
inspect the files, identify the analyses that apply, plan a bounded run, and
write local and Slurm launchers. It will not modify the simulation inputs.

## Status

Version 0.2.0a2 is an **experimental branch candidate**. It contains
58 registered MD, core, and reporting modules. Every registered module has an
implementation and automated tests, but none is yet marked `supported` for
unreviewed production or publication use. Technical completion and scientific
validity are reported separately.

## Included analysis

The comprehensive, nonredundant `standard_md_v1` profile includes:

- manifests, input inventories, atom mapping, preflight, structural QC, and
  connectivity-aware make-whole or continuous unwrapping;
- replica RMSD and radius of gyration, pooled RMSF, DCCM, nonlinear information
  measures, information dynamics, and correlation networks;
- default-off DFI/DCI perturbation response, exact frame-aligned trajectory
  reweighting, contact-occupancy allosteric pathways, and solvent/ion/ligand
  multivalent molecular-bridge networks, protein-only residue interaction-energy
  heat-kernel embeddings from Amber, CHARMM, or serialized OpenMM parameters,
  and segment-safe reactive-path
  ensembles with DTW route clustering and explicit transition-sufficiency gates,
  plus chemically typed frame-level interaction fingerprints, aligned spatial
  interaction superfeatures, DSSR-gated duplex helical mechanics, censored
  temporal interaction persistence, aligned water/ion density and geometric
  channel candidates, ensemble geometric pocket dynamics, and seed-gated
  random-feature nonlinear kinetic sensitivity;
- individual and shared-basis PCA, TICA, and free-energy or occupancy
  landscapes, with observed representative frames and structures and optional
  immutable state trajectories;
- eleven separately switchable conventional clustering methods. Ten are on by
  default; KMeans uses dependency-free deterministic strat_all and
  strat_reduced NANI initialization, while seeded KMeans++ remains available
  for compatibility. HDBSCAN is an optional noise-aware sensitivity. Ward and
  quality-threshold run only when they can assign every evaluated observation
  exactly. PaLD is kept separate because it describes local-depth and
  strong-tie communities rather than an ordinary full-trajectory partition,
  and it is off by default;
- separate Markov models for the best kinetically validated complete
  clustering and for the PCA-FES states, with lag, connectivity,
  implied-timescale, Chapman--Kolmogorov, time-blocked VAMP-E, and
  block-bootstrap diagnostics. Optional HDBSCAN dense-core and PaLD sampled
  models remain clearly labeled sensitivities and never replace the two primary
  reports;
- explicit, Scott, Freedman-Diaconis, and Rice scalar histograms, plus declared
  threshold states with sensitivity scans and segment-safe residence runs;
- dihedrals, explicit and topology-template-discovered direct hydrogen bonds,
  grouped chemical-identity comparison across nonidentical condition topologies,
  scalable one-water-mediated networks, bond-pattern clustering, nested grouped
  logistic/elastic-net classification, reusable trajectory
  features, native contacts, radial distribution functions, optional
  observables, protein and nucleic-acid secondary structure, intrinsic DNA-ring
  and stacking geometry, DSSR descriptors, generic bound-ion and ion-pair
  geometry, and SASA;
- grouped-validation machine learning, convergence and uncertainty diagnostics,
  RMSF permutation inference, and non-aggregating integrated reports.

The exact module contracts, inputs, outputs, status, and interpretation limits
are generated from the code registry in
[`docs/generated/MODULE_REFERENCE.md`](docs/generated/MODULE_REFERENCE.md).
The final non-docking legacy destination review is summarized in
[`docs/LEGACY_REVIEW.md`](docs/LEGACY_REVIEW.md); it closes repository placement
without treating experimental modules as scientifically supported.
The current independent numerical and real-trajectory evidence, including its
nonpassing scientific gates, is summarized in
[`docs/SCIENTIFIC_VALIDATION.md`](docs/SCIENTIFIC_VALIDATION.md).

## Install

The base package requires Python 3.10 or newer, NumPy, and SciPy:

```bash
python -m venv .venv
./.venv/bin/python -m pip install -e .
```

For the reviewed Python stack plus HDBSCAN and DSSP, use the Conda environment:

```bash
micromamba create --prefix ./.venv --file environment.yml \
  --override-channels --channel conda-forge --strict-channel-priority
./.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

Platform locks and their validation limits are documented in
[`environments/README.md`](environments/README.md).
Required, optional, external, build-only, and validation-only dependencies—and
their license and redistribution boundaries—are consolidated in
[`DEPENDENCIES_AND_LICENSES.md`](DEPENDENCIES_AND_LICENSES.md).
The x3dna-dssr adapter requires a separately obtained executable; it is not
redistributed by this repository.
The quick-start initializer discovers `mkdssp`/`dssp` either on `PATH` or next
to the active Python interpreter, which covers the reviewed Conda environment
without requiring a separate executable argument.
It similarly discovers `x3dna-dssr` but schedules helical mechanics only when a
reference probe verifies a DSSR duplex stem and the installed JSON exposes all
six required step descriptors. `--dssr-executable` can declare an explicit
installation for either single-system or comparison preparation; comparisons
gate and plan helical mechanics independently for each system.

The bundled NEMO tutorial trajectory and group-generated topology are licensed
separately under CC BY 4.0; the starting PDB retains its PDB provenance. See
[`LICENSE-DATA.md`](LICENSE-DATA.md) and the fixture's recorded provenance.

## Use

New users can begin with the
[`NEMO zinc-finger tutorial`](tutorials/nemo_zinc_finger/README.md). It uses a
small, hash-recorded piece of a published Salsbury-group protein–zinc
simulation to demonstrate preparation, planning, local execution, and careful
interpretation without requiring access to the cluster archive.

The command line is `salsbury-md-analysis`. During development it can also be
run directly from the source tree:

```bash
PYTHONPATH=src python -m salsbury_md_analysis list-modules
PYTHONPATH=src python -m salsbury_md_analysis --help
PYTHONPATH=src python -m salsbury_md_analysis preflight-system path/to/system.json --hash-content
PYTHONPATH=src python -m salsbury_md_analysis structural-qc path/to/project.json --hash-content
PYTHONPATH=src python -m salsbury_md_analysis common-pca path/to/project.json --hash-content
PYTHONPATH=src python -m salsbury_md_analysis build-coordinate-cache path/to/system.json --output path/to/new-cache
PYTHONPATH=src python -m salsbury_md_analysis prepare-unwrapped-cache path/to/system.json --output path/to/lossless-cache
PYTHONPATH=src python -m salsbury_md_analysis write-scientific-minimums-template --output scientific-minimums.json
PYTHONPATH=src python -m salsbury_md_analysis compare-hydrogen-bonds path/to/comparison-request.json
PYTHONPATH=src python -m salsbury_md_analysis plan-automatic-sampling path/to/system.json --simulation-kind unbiased_md
PYTHONPATH=src python -m salsbury_md_analysis plan-frame-resources pilot-100.json pilot-500.json --total-source-frames 30000 --replica-count 3
PYTHONPATH=src python -m salsbury_md_analysis run-regression path/to/regression-case.json
```

For a routine workstation or cluster case, one command inspects an atom-order-matched PDB,
explicit bond topology, and one DCD per replica, then writes validated manifests, the automatic
sampling report, portable staged workers, and both local and Slurm launchers into a new directory.
Local execution is the safe default:

```bash
PYTHONPATH=src python -m salsbury_md_analysis prepare-analysis \
  --pdb system.pdb --psf system.psf \
  --trajectory replica-1.dcd --trajectory replica-2.dcd --trajectory replica-3.dcd \
  --frame-interval-ps 10 --project-id my-study --output my-study-analysis
cd my-study-analysis
./run-local.sh
```

Add `--plan-only` to `prepare-analysis` or `prepare-comparison` when you want
the complete resource and sampling plan before deciding whether to run it. The
command validates inputs and writes the prepared directory, but it does not
start local workers or submit scheduler jobs. Its JSON response includes the
full `campaign-resource-plan.json`, a compact `planning_summary`, and the
reviewed command that would run the prepared campaign. The prepared directory's
`planning-report.md` gives the quicker human review: one row per analysis family,
effective raw strides over the original trajectories, and distinct `Off`,
`Deferred`, and `Not applicable` states. `planning-report.json` retains exact
per-task and per-replica details. Several plans can be combined with
`report-plan-matrix` into an 8-hour/24-hour/48-hour/168-hour-style comparison.

If the requested CPU count is above the dependency graph's useful parallel
ceiling, the response includes
`REQUESTED_CPUS_EXCEED_USEFUL_PARALLELISM` with the useful and excess core
counts. The planning record keeps the requested count, but the generated local,
custom, and Slurm launchers use the smaller effective count; Slurm array widths
and multiprocess worker requests are capped accordingly. If the envelope cannot
retain protected preparation and structural-integrity checks at their minima,
planning returns
`planning_outcome: no_acceptable_reduced_plan`; it does not propose disabling
those checks. The response also reports a
`protected_subset_minimum_request`: padded CPUs, aggregate memory, and whole
wall hours for the best dependency-closed subset that retains every protected
module under the supplied CPU and memory caps. This is a permissive execution
floor, not evidence that the trajectory is converged or scientifically
adequate; the scientific question may require a larger request.

That prepared directory also contains `prepared/system.json`. Pass that
manifest to `prepare-unwrapped-cache` when you want to create a reusable,
stride-1 continuously unwrapped cache before running any analyses.

Supported compositions, input-format boundaries, automatic chemistry routing,
and the behavior for RNA, oligomers, ligands/cofactors, ions, water, membranes,
and unknown polymers are summarized in
[`docs/GENERAL_BIOMOLECULAR_SYSTEMS.md`](docs/GENERAL_BIOMOLECULAR_SYSTEMS.md).

For work on WFU's DEAC cluster, add
`--config profiles/analysis/deac-default.json` during preparation and run
`./submit.sh`. That analysis config selects the validated
`profiles/slurm/deac.json` cluster profile. Other groups can copy
`profiles/slurm/generic-template.json` and set their account, Unix group, QoS,
partitions, scheduler commands, Python/environment setup, and storage paths.
The active choice is recorded in `execution-adapter.json`; local, Slurm, and
custom-launcher modes execute the same worker scripts, dependency order, frame selections, atomic
outputs, hashes, and resource instrumentation. Local execution enforces the
configured aggregate CPU and memory caps; Slurm requests are derived from the same
planner estimates and retained in `scheduler-resource-requests.json`. See
[`docs/EXECUTION_ADAPTERS.md`](docs/EXECUTION_ADAPTERS.md).

On a Slurm login node, the optional `advise-slurm-capacity` command can inspect
a prepared campaign before submission. It reports the cluster ceiling, the
smaller workflow-useful CPU maximum, sampling and memory estimates for a supplied
duration, current node fit, and queue pressure. It never runs during normal
preparation and never submits, cancels, or changes a job.

`--connectivity` is an equivalent, clearer spelling of `--psf` and also accepts
portable `salsbury-bonds-v1` JSON plus Amber PRMTOP/PARM7 inputs. This allows a
validated bond graph exported from another engine to enter the same generic
workflow without fabricating a PSF.

If none of those connectivity files was retained, an explicitly selected
OpenMM preparation fallback can create the portable bond JSON used by the
analysis workflow:

```bash
PYTHONPATH=src python -m salsbury_md_analysis prepare-analysis \
  --pdb system.pdb --generate-connectivity-openmm \
  --trajectory replica1.dcd --trajectory replica2.dcd \
  --frame-interval-ps 10 --project-id example --output analysis-example
```

This fallback requires the optional `openmm-connectivity` dependency only while
preparing the reusable bond JSON. It uses standard OpenMM residue templates and
explicit PDB connectivity, never a distance cutoff. For modified residues,
ligands, or cofactors, prefer the simulation topology; alternatively provide
reviewed residue XML with repeated `--openmm-bond-definitions`. The generated
graph is fail-closed for isolated atoms in multi-atom residues and remains
subject to topology-owner review. OpenMM does not natively write a parameterized
PSF here, and the later analysis jobs do not require OpenMM.

For matched controls, mutations, ligands, or a larger variant panel, declare
two or more systems in one `salsbury-comparative-analysis-input-v1` JSON file
and prepare one shared-basis workflow:

```bash
PYTHONPATH=src python -m salsbury_md_analysis prepare-comparison comparison.json \
  --project-id variant-panel --output variant-panel-analysis
cd variant-panel-analysis
./run-local.sh
```

Each request system supplies `system_id`, `pdb`, `trajectories`,
`frame_interval_ps`, and either `psf` or a portable explicit-bond JSON
`connectivity` file (plus optional `first_frame_time_ps`). Replica lengths may
differ. By default, preparation generates both (1) topology-aware shared views
using one common-atom PCA basis and grid across every declared system and (2)
separate per-system PCA/FES/clustering views with that system's replicas pooled.
This distinguishes changes on a common comparison coordinate system from states
best resolved on each system's own basis. Corresponding per-system branches use
balanced physical-frame budgets across conditions. Equivalent oligomer members
may expand the observation set but are never relabeled as replicas. The same
config supports all-pairs or reference-versus-all reporting for panels such as
20 variants.

Comparative preparation also schedules RMSF permutation inference automatically
when that module is enabled. Each declared simulation replica contributes one
exchangeable RMSF profile. Frames, time blocks, and symmetry-equivalent oligomer
members are not promoted to independent units. A pair with fewer than two
replicas per system is reported as insufficient rather than tested with
pseudoreplicated frames.

The default preparation target is one campaign envelope of 16 parallel CPUs
for 24 wall hours. `execution.maximum_parallel_cpus` and
`execution.maximum_hours_per_cpu` in `analysis-config.json` configure that
envelope; `--target-wall-hours` is a command-line override for the latter. It
is not a separate allowance for every method. The estimate includes a 1.5
timing safety factor. Only 85% of the raw CPU-hour envelope is normally planned,
with separate pilot and finalization reserves removed before scientific work is
allocated. Slurm's additional per-job time margin is a timeout threshold, not
extra planned science time. The generated `campaign-resource-plan.json` applies
the campaign limits across base trajectory estimators, inherited base analyses,
and every enabled PCA/FES/clustering/state view. `sampling-plan.json` records the applied
direct-method selections, while each view project records its shared upstream
selection. Both files report selected frame counts, coverage, and any
subsampling.
Methods without a retained portable calibration use explicitly labeled
conservative proxies. Transferred measurements retain the task's declared
workload scaling. The
coordinate-cache estimate scales with source atom count; common PCA scales with
Cartesian feature count and oligomer-member expansion. A small system therefore
does not inherit the unchanged runtime of the larger calibration system. If the
technical minima still do not fit, preparation prints the CPU and critical-path
shortfall and a calculated wall-time retry bound; it never lengthens the
campaign silently. Coarse legacy memory tiers are transferred from the
85,206-atom reference with a conservative square-root atom-count factor, a 10%
fixed-allocation floor, and a 4x upper bound. Measured maxima instead scale from
their measured observation coverage with a square-root relationship and the
same 10% floor. This avoids charging a tiny solute or short run the unchanged
allowance of a large, long calibration while retaining substantial headroom.

`execution.maximum_memory_gib` is the maximum simultaneous memory request for
the complete campaign, not an allowance for every job. The planner first turns
each estimated working set into a safety-adjusted scheduler request. With the
DEAC profile this is `ceil(1.5 × working set + 1 GiB)`, with a 2 GiB minimum.
It then packs independent tasks into resource waves whose summed CPU and
memory requests stay within the configured campaign caps. Resource waves wait
for completion of the preceding wave so a failure releases the allocation;
success-only dependencies are added separately for the reports a task truly
consumes. A lower memory cap
can therefore increase the integer strides or serialize work even when every
individual task fits. The local executor and the generated `submit.sh` enforce
the same limits.

For an insufficient memory cap, `campaign-resource-plan.json` and
`memory-feasibility-report.json` state (1) the largest enabled technical-minimum
working-set estimate, (2) the safety-adjusted memory request that would retain all enabled
work, and (3) every module or clustering-method switch that cannot fit.
Preparation remains fail-closed by
default. If the user explicitly accepts a reduced campaign, add
`--auto-disable-to-fit-memory` to either `prepare-analysis` or
`prepare-comparison`. The initializer preserves
`analysis-config.requested.json`, turns off the narrowest oversized
configuration switches and anything that depends on them, replans without
lowering frame minima, and writes the
fully explicit result to `analysis-config.memory-fit.json`. If the reduced
campaign still violates CPU or wall-time limits, it still fails.
Trajectory execution subsampling is never random: it is deterministic,
replica-balanced, and spread over the full time range. Every production
trajectory selector receives one exact integer stride over each concatenated
replica timeline; frame zero is retained and segment boundaries do not restart
the stride. Each method must meet its fixed minimum samples per replica/system.
Order-dependent methods also enforce a method-specific maximum physical-time
separation between retained frames; thermodynamic estimators do not. Runtime pilots calibrate resource cost only; the planner does not infer
autocorrelation times or event rates. The campaign planner advances stride upgrades through an absolute
CPU-and-wall resource frontier. Replanning the same tasks with a longer wall
limit extends that allocation path, so no task loses frames merely because a
previously unaffordable method becomes affordable. The exact retained counts
must satisfy both the total CPU-hour and dependency-stage wall-time limits.
Each alternative-clustering family receives its own explicit integer
fit stride according to its scaling profile; PCA projection and fit allocations
are reapplied until every clustering source count exactly matches the final PCA
projection count. A discrete-stride cycle is resolved by deriving a conservative
projection ceiling from that replan, not by restoring a saved projection count.
Families performed
serially by one view command are accounted as one execution bundle for wall
time while remaining separate logical allocations. All strides operate over
each replica-member timeline. Seeded random focal-observation
sampling is reserved for bounded silhouette estimates against the full fitted
partition.

`resource_budget_utilization` reports CPU-hour and wall-time use separately.
A campaign can nearly exhaust its wall-time allowance while leaving many raw
CPU-hours unused when dependency stages or serial methods cannot use all cores.
`allocation_saturation` records whether every frame ceiling was reached, memory
blocked the remaining work, or the next deterministic stride step would exceed
the campaign envelope. The planner does not duplicate analyses just to occupy
idle cores.

Preparation also classifies the reference chemistry and creates complementary
conformational projects automatically. Protein–nucleic-acid complexes declare
global common-heavy, chemistry-defined interface, and C-alpha/C1-prime trace
views; protein-only systems declare global common-heavy and trace views. The
trace view and its trajectory exports are off by default but independently
configurable. PCA basis planning uses both coordinate-feature count and
trajectory length, and chooses exact dense or gated leading-component
computation accordingly. Basis
fitting and projection have independent, explicitly reported integer strides;
the planner uses every source frame when it fits the campaign envelope. Each
view retains separate PCA/FES/cluster labels and auditable
resource decisions in `conformational-views.json`. See
[`docs/CONFORMATIONAL_VIEWS.md`](docs/CONFORMATIONAL_VIEWS.md).
Methods that operate on derived PCA, clustering, hydrogen-bond, or RMSF output
inherit the upstream frame selection. Their post-processing time is bounded by
explicit observation/state/grid limits. Retained Apollo measurements calibrate
the demonstrated TREX view methods; unmeasured derived methods are labeled as
provisional complexity models rather than as portable benchmarks.

The automatic workflow can infer the broad molecular composition and schedule
analyses that do not require a project-specific scientific selection. It cannot
know, for example, which residues define the biological site of interest, which
ligand atoms define a reaction coordinate, which ions belong to a particular
binding shell, or how the ring atoms of a new modified base should be grouped.
When an analysis needs that information, the workflow defers it instead of
guessing. Supply the relevant selection in an explicit project definition to
enable that analysis. The generated `module-coverage.json` records which
modules were scheduled, disabled, or deferred and gives the reason for every
deferral.

`submit.sh` performs the quick path checks immediately, then schedules the
full content-hashing preflight as a small cluster job. Three dependent method
arrays start only after their prerequisites succeed: direct analyses, derived
analyses, and K-means-dependent state exports. Expensive common PCA, DCCM,
RMSD/Rg, and K-means results are computed once; downstream processes reuse a
complete report only when its module identity, resolved project/system paths,
manifest hashes, and input-content signature match the current run. A declared
cache mismatch fails instead of silently recomputing or accepting unrelated
data. Each submission freshly hashes the declared inputs. If a retained
preflight exists, the refreshed report must be byte-identical before cached
module results are eligible for reuse. The structured preflight includes every supplied topology, explicit
connectivity file, and trajectory; it hashes each input and checks topology,
connectivity, and trajectory atom cardinality before analysis. Generated
launchers retain both the exact Python executable and package
source used to create them, so running `prepare-analysis` directly from a
source checkout remains reproducible on compute nodes. If that installation is
intentionally moved, set `SALSBURY_MD_ANALYSIS_PYTHON` and
`SALSBURY_MD_ANALYSIS_PYTHONPATH` when submitting.

### Turn analyses on or off

Pass a partial JSON configuration to `prepare-analysis` or
`prepare-comparison` with `--config CONFIG.json`. Anything not listed in the
partial file keeps its default. Each scientific module uses
`modules.<module_id>.enabled`; conformational views and their optional state
trajectory exports have separate switches; and each clustering method can be
controlled independently. For example:

```json
{
  "config_schema": "salsbury-analysis-config-v1",
  "enable_all_experimental_modules": true
}
```

That master opt-in enables all thirteen default-off experimental methods. Explicit
`modules.<module_id>.enabled: false` entries take precedence, and normal input,
applicability, and external-tool gates still apply.
If DFI/DCI functional-site nodes are supplied, its required macromolecular-trace
view is enabled automatically while trace-defined trajectory export remains off.
Without those nodes, DFI/DCI is reported as unavailable rather than guessing a
biological functional site.

Individual controls can be combined with the other configuration sections:

```json
{
  "config_schema": "salsbury-analysis-config-v1",
  "modules": {
    "solvent_accessible_surface_area": {"enabled": true},
    "water_mediated_hydrogen_bond_networks": {"enabled": false}
  },
  "views": {
    "macromolecular_trace": {
      "enabled": false,
      "state_trajectory_exports_enabled": false
    }
  },
  "clustering": {
    "methods": {
      "hdbscan": {"enabled": false},
      "ward": {"enabled": true}
    }
  },
  "community_analysis": {
    "pald": {
      "enabled": false,
      "community_msm_enabled": false
    }
  },
  "inference": {
    "ion_site_classification_enabled": true
  }
}
```

Use the file during preparation:

```bash
salsbury-md-analysis prepare-analysis ... --config my-analysis-config.json
```

Run `salsbury-md-analysis list-modules` to see the module identifiers. The
prepared campaign writes the complete resolved configuration to
`analysis-config.json` and records every enabled, disabled, or deferred module
in `module-coverage.json`. Disabling an upstream module also disables analyses
that depend on it; required preflight, provenance, and atom-mapping checks
cannot be turned off, and neither can structural-integrity QC. Each complete
module row includes a generated `protected` flag, `depends_on`, and
`turning_off_also_disables`, while `module_groups` arranges the switches as
infrastructure, quality/motion, conformational bases, states/kinetics,
internal geometry, interactions/solvent/ions, and integration. This metadata
is generated from the workflow graph and cannot be edited independently of the
module switches. The full set of module, view, comparison, export,
inference, execution, and reporting controls is described in
[`docs/CONFIGURATION_AND_FINAL_REPORTING.md`](docs/CONFIGURATION_AND_FINAL_REPORTING.md).

Sampling floors live in a separate file so module choices and scientific
minimums are not mixed together. Generate a complete editable copy with:

```bash
salsbury-md-analysis write-scientific-minimums-template \
  --output scientific-minimums.json
```

Every method lists `minimum_frames_per_replica`,
`minimum_frames_overall_per_system` (pooled across that system's replicas), and
`maximum_time_gap_between_retained_frames_ns`. Point
`sampling.scientific_minimums_file` at the reviewed file. Values may be kept or
made stricter; the public workflow refuses a lower frame floor or a looser
positive time-gap gate.

The default remains all automatically applicable modules and high-detail
conformational views. Each instrumented result includes measured CPU,
wall-time, memory, host, job, physical-frame, and symmetry-expanded observation
evidence. A final dependent job writes the consolidated resource/frame table
and transparent prioritized-finding report. The picker accounts for every
completed report as a ranked scientific-candidate source, QC evidence,
interpretive context, technical support, or a reviewed report with an explicit
reason that no automatic highlight was produced. It can compare completed
per-system reports as well as combined multi-system reports, keeps the selected
evidence paths visible, and never labels descriptive extrema as statistically
significant. See [`docs/FINDING_PICKER.md`](docs/FINDING_PICKER.md).

Interactive browsing is provided separately by
[`salsbury-md-analysis-interactive`](https://github.com/salsburygroup/salsbury-md-analysis-interactive).
The core package does not install or invoke the viewer. Install the companion
only when you want to turn a completed campaign into a self-contained offline
HTML report; the JSON, CSV, Markdown, structures, and method reports produced
here remain the scientific record.

The generated workflow is safely resumable. A technically complete method
report is validated and reused on resubmission only after the freshly rehashed
preflight matches the retained one. An existing malformed or incomplete final
report is never overwritten, and a newly completed report is installed with an
atomic fail-if-present link so concurrent submissions cannot silently replace
one another. Failed-job temporary reports remain for diagnosis.

Commands inspect declared inputs and emit JSON reports to standard output. They
do not overwrite trajectories or source data. Large trajectories, restricted
structures, full results, credentials, and private cluster paths must not be
committed here.

Large solvated campaigns can build one connectivity-aware, unaligned
`molecular_payload` cache and reuse it for solute-only conformational work.
The cache retains protein, nucleic acid, hydrogens, ligands, cofactors, and
ions, while water-dependent analyses continue to read the immutable solvated
source. `prepare-unwrapped-cache` is the explicit preprocessing-only mode: it
continuously reconstructs every source frame, saves a stride-1 payload, and
stops. Set `execution.coordinate_cache_input` to that cache directory in later
analysis configs. Reuse validates the source manifest, topology, connectivity,
trajectory path/size/modification identity, complete report, and all-frame
retention before planning; water-dependent methods still use the original
solvated inputs. Cache construction is atomic and fail-if-present. See
[`docs/COORDINATE_CACHE.md`](docs/COORDINATE_CACHE.md).

Inexpensive streaming modules use every frame when `frame_stride` is one.
Expensive modules can instead use deterministic, full-timespan pooled ceilings
allocated equally across replicas while reporting exactly what was available
and evaluated. RMSD/Rg is the explicit per-replica exception.
The resource planner fits fixed scan overhead and incremental selected-frame
cost from retained pilots on the actual method/project workload, prefers all
frames, and emits an `auto_resource_budget_v1` contract only when the declared
wall-time envelope requires it. Budget-doubling sensitivity defaults to `off`
and is optional. It is reported as `off`, `recommend`, or `require` rather than silently enforced.
It is never scheduled automatically; `require` is an explicit project-owner
choice, not a default condition for publication use.
The supported modules, TREX 30,000-frame interpretation, method-specific pilot
ranges, continuous-unwrapping safeguard, and escalation rules are documented in
[`docs/FRAME_SAMPLING.md`](docs/FRAME_SAMPLING.md).

The default scientific presentation order is FES, FES basin conformations,
silhouette-selected clustering and populations, cluster conformations, RMSF
profiles plus RMSF-colored structures, and then other physically meaningful
results. Non-RMSD scalar time series are histogram-first using Scott's rule;
their time-series views are secondary. See
[`docs/REPORTING_STANDARD.md`](docs/REPORTING_STANDARD.md).

For a modified residue or other non-standard chemistry, preserve the matching
force-field connectivity rather than guessing bonds from coordinates. When an
OpenMM `system.xml` and its atom-order-matched PDB are available, run
`scripts/export_openmm_system_connectivity.py topology.pdb system.xml bonds.json`
to make an auditable portable bond graph. The converter uses the
atom-order-matched OpenMM PDB topology as the default graph and inventories
System-only `HarmonicBondForce`, `CustomBondForce`, and constraint pairs. It
excludes all System-only pairs by default because they may reflect an atom-order
mismatch, restraint, angle constraint, or rigid geometry rather than a
covalent bond. Include a reviewed category only with its separately named
`--include-harmonic-force-only-pairs`, `--include-custom-bonds`, or
`--include-constraint-only-pairs` switch.
If a reviewed serialized OpenMM `State` contains the accepted coordinates,
`scripts/export_openmm_state_pdb.py` can write an atom-order-matched PDB without
changing the original topology or checkpoint.

## Validate the repository

```bash
PYTHONPATH=src python scripts/generate_docs.py --check
PYTHONPATH=src python -m unittest discover -s tests -v
python -m build
```

The remaining release gates are tracked in
[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).

## License

Original source code is BSD-3-Clause. Original documentation and teaching
material are CC BY 4.0 under [`LICENSE-DOCS.md`](LICENSE-DOCS.md). Historical or
third-party material must pass a separate rights review before inclusion. The
review retained for this alpha is documented in
[`RELEASE_RIGHTS_REVIEW.md`](RELEASE_RIGHTS_REVIEW.md).
