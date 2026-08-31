# Configuration and final reporting

`prepare-analysis` writes a complete `analysis-config.json`. Its versioned
schema is `salsbury-analysis-config-v1`, and its default is
`all_applicable`: every applicable scientific analysis and high-detail
conformational view that can be selected without inventing chemistry is
enabled; the deliberately coarse trace view is the documented exception.
Thirteen active-development methods are also deliberate exceptions:
`perturbation_response_dynamics`, `trajectory_reweighting`,
`allosteric_pathways`, `multivalent_molecular_bridges`,
`energetic_network_embeddings`,
`reactive_path_ensembles`, `interaction_fingerprints`,
`spatial_interaction_ensembles`, `interaction_persistence`,
`random_feature_koopman`, `helical_mechanics`, `hydration_density_channels`,
and `ensemble_pocket_dynamics` are present in
every generated configuration but are off until explicitly enabled. The
pathway network itself is trajectory-derived by default.
Set the master switch to opt into all thirteen at once:

```json
{
  "config_schema": "salsbury-analysis-config-v1",
  "enable_all_experimental_modules": true
}
```

The master switch is applied before individual module entries, so an explicit
`modules.<module_id>.enabled: false` can still leave a method out. It does not
bypass topology applicability, required scientific inputs, resource gates, or
external-program checks; for example, helical mechanics still requires DSSR
and a detected duplex, while energetic-network embeddings require an Amber
PRMTOP/PARM7, CHARMM PSF plus matching parameter files, or a serialized OpenMM
System XML with one standard `NonbondedForce`. Every route also requires an
explicit bond graph for cpptraj-style exclusions. Reweighting remains off at
planning time without a declared frame-weight file, and allosteric pathways
remain off without reviewed source and sink nodes. Bridge and hydration-density
tasks are omitted when the reference topology contains none of their enabled
particle or mediator species. These decisions are reported as unavailable or
not applicable while the original configuration intent remains visible.
The command-line equivalent is `--with-experimental-modules` on
`prepare-analysis` and `prepare-comparison`. It adds applicable experimental
methods to the normal main-module workflow for the new campaign. It does not
write into, or reuse results from, a completed campaign directory.
Supply a reviewed partial or complete configuration with:

```bash
salsbury-md-analysis prepare-analysis ... --config my-analysis-config.json
```

The `planning` section provides two independent controls:

```json
{
  "config_schema": "salsbury-analysis-config-v1",
  "planning": {
    "module_selection": "protected_core_only",
    "stride_mode": "uniform_cache_stride"
  }
}
```

`module_selection: protected_core_only` disables every non-protected module
after resolving the normal dependency graph. It keeps the protected modules,
their prerequisites, enabled conformational views needed by common PCA, and
observed representative-frame output. Optional clustering methods, PaLD, and
state-trajectory exports are off in this mode. The equivalent command-line
switch is `--protected-core-only`.

`stride_mode: uniform_cache_stride` tests one deterministic cache stride at a
time and requires every enabled method, PCA projection, and clustering fit to
consume that retained stream without further frame subsampling. A candidate is
rejected if the retained stream is below a scientific minimum, exceeds a
method's declared frame ceiling, violates an ordered-method time-gap limit, or
does not fit the campaign resources. The planner tests from finest to coarsest
and stops at the first feasible candidate. If none fits, preparation fails
closed or the existing optional-method reduction can propose a smaller enabled
set. The equivalent command-line switch is `--uniform-cache-stride`. The
default `balanced_per_method` mode keeps the two-stage cache plus method-stride
optimization because it usually retains more information across methods in a
fixed resource envelope. Uniform mode requires a newly planned
`planned_strided` cache; preparation rejects cache-off, lossless-cache, and
external-cache configurations instead of pretending they satisfy this mode.

Each `modules.<module_id>` entry has an `enabled` flag and an `options` object.
Options replace generated definition fields and then undergo the normal strict
project-schema validation. Disabling an upstream module also disables its
dependent analyses; the resolved config turns off affected conformational
views before project construction, and `module-coverage.json` records the
decision. The protected scientific core cannot be disabled: provenance,
preflight, common-atom mapping, structural-integrity QC, RMSD/Rg, pooled RMSF,
individual and shared/common PCA, DCCM, PCA free-energy surfaces, and observed
representative-frame selection. Comparative preparation additionally protects
`integrated_comparison`: every comparison campaign must review and account for
all completed result reports before the finding picker runs. These protections
are passed into stride planning and optional-module reduction, so a plan that
cannot fit the core returns `NO_ACCEPTABLE_REDUCED_PLAN`. Each topology-derived
view has its own `enabled` flag and optional per-view `module_options`.
The complete generated config also gives each module a generated `protected`
flag, its direct `depends_on` list, and its `turning_off_also_disables` list.
`module_groups` places the switches
in numbered sections: required infrastructure, quality/motion,
conformational bases, states/kinetics, internal geometry,
interactions/solvent/ions, and integration. These fields explain the graph;
only `enabled` and `options` are editable. Changing an explanatory dependency
field is rejected because it would make the config disagree with execution.
The high-detail common-heavy and interface views are enabled by default.
`macromolecular_trace` PCA and its state trajectories are off by default but
can be enabled independently for a deliberately coarse diagnostic.
DFI/DCI currently runs only on that trace view because its scientific unit is
one representative macromolecular node per residue. Activation examples and
the external weight schema and optional external-network override are in
[`EXPERIMENTAL_METHODS.md`](EXPERIMENTAL_METHODS.md).

`hydrogen_bonds` is the optional manual fixed-feature interface. Routine
campaigns use `hydrogen_bond_discovery`, which infers donor, bonded hydrogen,
and acceptor candidates from topology connectivity and chemical identity.
`representative_structures` is likewise an optional coordinate-space
mean/medoid utility. Routine state workflows use `representative_frames` and
`state_coordinate_exports`; the former selects observed frames nearest the
declared cluster center or FES basin root in that analysis feature space. The
export module is protected because it materializes representative state
structures. Its separate `state_trajectory_exports_enabled` view option is on
by default for non-trace conformational views and off for the trace view. It may
be turned off without disabling representative structures.

The planner models coordinate writing from the configured maximum output frames
and representative count. Measurements from the retired repeated-read exporter
are not applied to every PCA projection; state-assignment sampling still
inherits the pooled conformational-view plan.

The `clustering.methods` object lists all eleven conventional partitioning
methods explicitly:
`kmeans`, `hdbscan`, `intelligent_minkowski_weighted_kmeans`, `pam`,
`minkowski_weighted_pam`, `ward`, `gaussian_mixture`,
`variational_gaussian_mixture`, `affinity_propagation`, `mean_shift`,
and `quality_threshold`. Ten complete or sampled-partition methods are on by
default and each method has its own `enabled` flag. HDBSCAN is available but
off by default because its noise-censored dense-core partition cannot represent
the complete trajectory; when enabled, its MSM is a conditional sensitivity
that never enters primary MSM selection. Ward and quality-threshold remain on
by default but are skipped before their quadratic fit whenever the planner
cannot retain every observation; quality-threshold is also omitted if no
configured cutoff assigns every observation. PaLD is intentionally separate under
`community_analysis.pald`: its
sampled cohesion, local-depth, and strong-tie community network is off by
default because the direct calculation is cubic. Enabling PaLD never makes it
eligible for conventional best-partition selection. Its separately labeled
sampled-community MSM is controlled by `community_msm_enabled` and is also off
by default. `clustering.feature_space` controls both conventional clustering
and the optional PaLD feature basis; it defaults to `tica` and can be changed
to `common_pca` for a declared geometric sensitivity analysis.
The PaLD planner uses a separately measured single-CPU 500-observation
calibration and scales time cubically and matrix memory quadratically with its
configured observation ceiling. The normal workflow safety factor is then
applied; PaLD is never costed as ordinary linear per-frame clustering.

The conventional alternative-clustering planner uses per-algorithm
performance models. PAM and Minkowski-weighted PAM use a quadratic,
feature-count-aware pairwise-memory model; affinity propagation uses its own
dense-similarity quadratic model. Mean shift and the Gaussian and variational
Gaussian mixtures remained below the planner's 1-GiB working-set floor in the
accepted calibration and therefore keep that floor. Each algorithm has an
independent integer fit stride and runtime curve. Ward and quality-threshold
retain provisional estimates and are skipped unless every observation can be
fit exactly. The calibration records timing and resident memory only: cluster
labels, centers, populations, representatives, scores, and biological results
were discarded. The cluster profile's memory adjustment is applied after these
working-set models, in the same way as for other modules.

The `execution` section describes one campaign envelope. For example,
`maximum_parallel_cpus: 32` and `maximum_hours_per_cpu: 24` means at most 32
CPUs for 24 wall hours, or 768 raw CPU-hours before the configured utilization
and pilot/finalization reserves. It does **not** mean 24 hours for every method.
`time_safety_factor`, the two named memory-calibration factors, and
`censored_timeout_safety_factor` are independently configurable.
`well_calibrated_memory_uncertainty_factor` defaults to `1.0` for models backed
by repeated completed evidence. `poorly_calibrated_memory_uncertainty_factor`
defaults to `1.25` for weak, single-run, censored-only, or unmeasured models.
The former analysis-config `memory_safety_factor` remains a deprecated alias for
the latter and cannot be supplied with it. These named factors are part of the
task working-set model, not a cluster adjustment.

The planner applies the selected cluster profile's memory adjustment once after
the task model is complete. On DEAC the final reservation is
`ceil(1.5 × uncertainty-adjusted working set per node + 1 GiB)` for every
allocated node. The scheduler and custom-launcher adapters consume that
per-node value without changing it. There is no additional unnamed or
scheduler-side memory padding. All terms are recorded in
`resource_safety_margins`.
The default 1.5 time factor is applied to modeled task costs before frame
allocation. The planner then uses only the configured utilization fraction
(normally 0.85) and removes pilot and finalization reserves. The shipped Slurm
profiles set their scheduler wall-time factor to 1.0, so that factor is not
applied again. Their 15-minute per-job overhead and 30-minute minimum must fit
inside `maximum_hours_per_cpu`, which is the final padded end-to-end execution
ceiling. The execution adapter retains the largest uniform fraction of the
overhead whose serialized dependency-path kill limits fit inside the ceiling.
It does not alter sampling or enabled modules. If planner estimates plus minimum
job limits cannot fit, submission is refused.
`finalization_headroom_fraction` reserves campaign capacity for dependency
barriers, summaries, hashes, and fail-closed acceptance rather than allocating
the entire envelope to scientific estimators. A timed-out job is retained as a
right-censored lower bound: its target frames are not counted as completed
coverage, and its elapsed time is never mislabeled as a successful runtime.
For a public or shared source tree, build the distributable catalog with
`build-resource-calibration-catalog --redact-source-paths`. The redacted copy
retains measured CPU, memory, frame coverage, evidence hashes, and timeout
censoring semantics while removing private paths, scheduler IDs, and hostnames.
The full catalog remains separately hash-pinned as internal provenance.
Two or more complete measurements for a module qualify the largest completed
RSS to replace an older built-in memory baseline and receive the
well-calibrated uncertainty factor. A single completed run and timeout-only
evidence can raise a memory floor but cannot lower it; those cases receive the
poorly calibrated factor. An upper prediction bound, when fitted, is part of
the working-set model. The planner then applies the cluster profile's memory
adjustment once. The report records every rule and term applied. A
parallel task keeps its per-worker estimate separate from its intrinsic worker
count; if the campaign cannot hold every worker at once, the planner schedules
worker waves and increases the wall estimate without reducing frame coverage.
Direct
estimators receive small method-specific runtime pilots (normally 10--100
frames per replica at the TREX reference size and fewer for substantially
larger systems). Those pilots calibrate cost and memory only; they are not
scientific minima, convergence thresholds, or production recommendations.
The planner does not estimate autocorrelation times or event rates. It enforces
each method's fixed minimum sample count and, for order-dependent methods only,
the maximum retained-frame temporal separation using the declared frame
interval before allocating additional
deterministic full-timespan samples within the shared envelope. Preparation writes
`campaign-resource-plan.json`, which inventories base direct tasks, inherited
base tasks, and all enabled conformational-view tasks. Downstream FES,
clustering, and state methods share the selected physical-frame identities of
their view PCA; equivalent oligomer members add observations but do not add
physical frames or replicas. Every replan regenerates the PCA projection count
and uses it as the source count for each downstream clustering fit. The planner
iterates until those counts agree exactly. If discrete stride upgrades alternate
between two allocations, it derives the componentwise lower projection ceiling
from that cycle and replans the fits; it never falls back to a saved count.
A two-stage plan first selects a cache stride from the raw trajectory, then an
integer method stride over that cache for each estimator. Effective raw stride
is their product. Candidates that violate any protected module's scientific
floor are pruned before CPU or memory planning. An optional module that cannot
meet its own floor makes that complete candidate fail closed; dependency-closed
reduction may then recommend disabling the module and replan the remaining
workflow. The planner never prices a scientifically invalid protected-core
stride as if it were executable.
A configured envelope that cannot fund the small
technical minima fails closed by default.

Method-reduction advice preserves every module marked `protected: true`,
including protection through dependency closure. If those retained tasks do
not fit, the terminal and saved plan report `No acceptable reduced plan` and
recommend a larger envelope. A reduced configuration is never manufactured by
turning off structural-integrity QC.

The terminal summary and `campaign-resource-plan.json` also report the padded
minimum resource request for the best protected dependency-closed subset.
Without a physical-node policy, CPU and aggregate memory come from its busiest
resource stage. With a node policy, the replay-safe request retains the CPU,
memory, and node envelope used to validate the printed wall time; reducing that
envelope can change lane packing. `modeled_peak_utilization` separately reports
the CPUs, memory, and occupied nodes in the modeled schedule. Requested wall
time is the science critical path divided by the usable science fraction after
pilot and finalization reserves, then rounded up to a whole hour. A status of
`requires_larger_wall_time` means the subset fits the supplied CPU and memory
caps but needs a longer campaign ceiling. The accompanying warning states that
this permissive minimum is an execution floor, not a convergence,
equilibration, or biological-validity claim.

The method floors are independently configurable. Create a complete policy
file with `write-scientific-minimums-template`, review its per-replica,
pooled-overall-per-system, and ordered-method time-gap values, and set
`sampling.scientific_minimums_file` to that path. The public policy allows a
user or publication workflow to raise count floors or tighten positive
time-gap maxima. It rejects changes that weaken the packaged standard. The
resolved policy ID, source path, and SHA-256 are copied into the campaign plan.

Stride upgrades follow a progressive absolute resource frontier. For an
otherwise identical plan, increasing the wall limit extends the shorter
allocation instead of replacing its inexpensive upgrades with newly affordable
expensive work. Per-task frame coverage is therefore nondecreasing across a
duration series such as 8, 24, 48, and 168 hours. Priority weights decide among
upgrades that become affordable at the same frontier; they do not retract an
earlier task's frames.

CPU-hours and wall time are reported separately in
`resource_budget_utilization`. A low CPU-hour fraction is not automatically
unused scientific work: serial estimators, dependency barriers, and per-task CPU
caps can saturate wall time while many requested cores are idle.
`allocation_saturation` identifies the stop reason, the groups at their frame
ceilings, memory-blocked groups, and the wall allowance required by the next
stride upgrade. No duplicate calculation is added merely to consume the raw
CPU-hour envelope.
When `maximum_parallel_cpus` exceeds the largest concurrently useful stage,
`resource_warnings` reports `REQUESTED_CPUS_EXCEED_USEFUL_PARALLELISM`, the
useful ceiling, and the excess requested cores. The requested value remains in
the provenance record. The resolved execution cap is reduced to the useful
ceiling, and generated local, custom, and Slurm launchers use that effective
value. The warning states both counts before execution.
That failure includes the minimum calibrated critical path, the science wall
and CPU allowances after reserves, the configured campaign ceiling, and the
smallest calculated `--target-wall-hours` retry bound. The number is guidance,
not an automatic change to the user's resource request.

Memory starts with a per-task working-set estimate at its technical minimum. Legacy tier values are
referenced to the retained 85,206-atom solvated benchmark system, then adjusted
by a conservative square-root atom-count factor. The factor cannot fall below
0.1 or rise above 4.0, so fixed library allocations retain headroom and very
large systems do not extrapolate without bound. Measured maxima are transferred
by the measured observation coverage instead: the square-root observation
factor also has a 0.1 floor. A power-law method keeps its declared
observation-based exponent. Every task records the reference value, applicable
scaling model, and final selected-observation estimate. The execution profile
then converts each working set to a buffered request. The DEAC rule is
`ceil(1.5 × working set + 1 GiB)`, with a 2 GiB minimum. The campaign's
`maximum_memory_gib` limits the sum of those requests in each concurrent wave.
It is not repeated for every job. CPU and memory limits are both considered
while allocating frames and estimating dependency-stage wall time.

A Slurm profile can also declare `node_policy.cpus_per_node`,
`node_policy.memory_gib_per_node`, and an optional
`node_policy.maximum_nodes_per_campaign`. Node memory is checked against the
safety-adjusted scheduler request, not the unpadded working-set estimate.
Replica-parallel tasks are split into node-sized worker groups; other tasks
must fit one node. Concurrent resource lanes are then bin-packed so no node's
padded CPU or memory reservation exceeds its declared shape. The campaign's
`maximum_memory_gib` remains an aggregate simultaneous-memory limit: a task
that reserves 181 GiB on each of two nodes consumes 362 GiB of that envelope.
If the campaign gives only aggregate CPU and memory caps, the planner derives
the node-count ceiling from those caps and the profile's node shape. The
protected-core minimum request reports its modeled node count as well as
aggregate CPUs, memory, and wall time.

When one or more safety-adjusted technical minima exceed `maximum_memory_gib`, the plan's
`memory_feasibility` section reports the largest raw working set, the required
buffered request, its shortfall,
a whole-GiB rounded recommendation, the oversized task rows, and the minimal
set of module or individual clustering-method switches that must be disabled
at that cap. The default behavior does not alter the user's analysis. It writes
the failed planning evidence and
stops before any analysis is launched.

An explicit reduced-campaign workflow is available with:

```bash
salsbury-md-analysis prepare-analysis ... --config analysis.json \
  --auto-disable-to-fit-memory
```

The same flag is available on `prepare-comparison`; it applies one explicit
resolved configuration to the complete shared-basis and per-system comparison
campaign.

This performs a disposable planning pass, turns off each narrowest
configuration switch whose minimum cannot fit, materializes dependency-driven
disables, and replans. The final
directory contains `analysis-config.requested.json`,
`analysis-config.memory-fit.json`, and `memory-feasibility-report.json`, so all
on/off decisions are reviewable. `coordinate_cache` is represented by
`execution.coordinate_cache: off` when it is the incompatible feature.
Technical frame minima and scientific gates are never relaxed. The fallback
addresses memory only; a remaining CPU-hour, critical-path, calibration, or
scratch-space failure remains fail-closed.

To let the planner apply dependency-closed optional reductions for CPU,
critical-path wall time, or memory, use:

```bash
salsbury-md-analysis prepare-analysis ... --config analysis.json \
  --auto-disable-optional-to-fit-resources
```

The comparison initializer accepts the same flag. This mode preserves the
requested config and plan, applies only the planner-recommended optional
switches and their dependents, and writes `analysis-config.resource-fit.json`
and `resource-fit-report.json`. It succeeds only if all protected modules fit
without relaxing their configured sampling minima.

Measured catalogs describe the systems on which they were collected; they are
not universal per-frame constants. Planner tasks may therefore carry an
auditable workload multiplier before a catalog rate or memory maximum is
applied. The coordinate cache uses source atom count. Common PCA uses Cartesian
feature count and member-observation expansion. Methods driven by projected
observations rather than raw atoms keep their observation-based model. The
applied multiplier and reference workload remain in
`campaign-resource-plan.json`.
The generated local, Slurm, and custom-launcher adapters derive a task graph
from each module's real report inputs. `execution.submission_adapter` selects `local`, `slurm`,
or `custom`; local is the default.
Slurm mode requires a validated `execution.slurm_profile`, which keeps account,
partition, QoS, environment, command, and storage conventions outside scientific
configuration. The supplied `profiles/analysis/deac-default.json` selects the
Salsbury-group DEAC profile. Custom mode hands `launcher-contract.json` to the
executable named by `SALSBURY_MD_ANALYSIS_CUSTOM_LAUNCHER`. All adapters use the
same workers and output contracts.
The Slurm adapter records task-specific scheduler requests in
`scheduler-resource-requests.json` and routes sufficiently large requests through
the profile's large-memory role. Its canonical launcher submits deterministic
resource epochs. Each dependency level is packed into serial lanes that stay
within both `maximum_parallel_cpus` and the aggregate `maximum_memory_gib`; the
next epoch waits for completion of the preceding epoch only to release and
reuse resources. A task waits for successful completion only of its declared
data prerequisites. The launcher refuses submission when its generated
critical-path estimate or serialized scheduler kill-limit path exceeds the
campaign wall limit. Queue wait and scheduler backfill are not counted as
analysis wall time.

`advise-slurm-capacity` is an optional, read-only step for prepared Slurm
campaigns. It can replace the configured CPU count with the smaller of the live
scheduler ceiling and the workflow's useful parallelism, then rerun the saved
sampling allocation for a requested duration. Because selected observations are
recomputed, observation-scaled memory is recomputed as well. The command reports
per-task and exact planned resource-wave memory separately; it does not alter the
config, regenerate workers, or submit work.

The `exports` section separates feature/alignment atoms from coordinate output.
The default coordinate payload retains the complete non-water molecular system
(including hydrogens and chain-associated ligands, cofactors, and ions), while
PCA/FES uses its declared common-heavy Cartesian basis. Clustering uses tICA
coordinates derived from that basis by default; the alignment and analysis
atom selections remain independently recorded.
Equivalent oligomer members are exported on one canonical member topology and
pooled with member provenance; they are never counted as independent replicas.

The `comparisons` section supports `all_pairs` (suitable for a batch such as 20
variants) or `reference_vs_all`. Inferential candidates with p-values are
corrected by Benjamini-Hochberg within declared comparison families. A
difference without a supported p-value is retained as descriptive and is never
labeled statistically significant. `run_shared_basis_comparisons` controls the
common-atom, common-PCA-basis/common-grid branches.
`run_per_system_analysis` controls additional system-isolated
PCA/FES/clustering branches. Both default to true. Replicas are pooled within a
system, while matched per-system branch families receive the same per-replica
physical-frame budget. These independent-basis landscapes complement rather
than replace the shared-basis comparison and cannot be overlaid as though their
PC axes were identical.

Every generated analysis runs in one fresh instrumented process. Its JSON
report records host, execution job/task identity, requested CPUs/memory/time, wall
time, CPU time, and peak resident memory. Final report collation waits for every
enabled task to finish without treating unrelated failures as prerequisites. It
records incomplete or failed inputs rather than hiding them and
writes:

- `analysis_resource_and_frame_table.csv`, `.json`, and `.md`;
- `results/integrated-comparison/report.json` for comparative campaigns;
- `prioritized_findings.csv`, `.json`, and `.md`;
- compact finalizer status reports.

Interactive browsing is provided by the separate
`salsbury-md-analysis-interactive` companion repository. The core package does
not install or invoke the viewer; its JSON, CSV, Markdown, structures, and
method reports remain the scientific record.
Root-level `*-availability.json` records are included alongside completed
module reports, so an optional method that cannot run is shown as unavailable
rather than silently absent.

The resource summary, integrated comparison, finding picker, and optional RMSF
permutation inference run as isolated final-reporting components. One component
may report failure without preventing the others from examining completed
upstream reports; the finalizer still returns a failed technical status when
any component fails.

Each instrumented report also contains a `planner_benchmark` adapter accepted
directly by `plan-frame-resources`. It uses original selected physical frames
as the linear frame-planner unit and separately records member-expanded
observations. Quadratic clustering still requires its observation-specific
multi-budget calibration; a doubled symmetry representation is not silently
treated as twice as many independent frames.

The resource table distinguishes original selected physical frames from
symmetry-expanded member observations. For methods with distinct computational
budgets, the table also separates PCA
basis physical frames/member observations, clustering fit observations, full
assignment observations, and silhouette-evaluation observations. Thus a
1,000-observation silhouette estimate is not reported as though the underlying
30,000-frame/60,000-member assignment had been subsampled. Every included
analysis report must be technically complete, instrumented, and have exact
frame/observation accounting or finalization fails closed.
Each worker creates a compact `report.json.summary.json` while the full report
is already in memory. The sidecar is bound to the installed report path and
SHA-256 and contains only resource accounting plus transparent finding
evidence. Finalization streams the full-report hash but reads the compact
sidecar, avoiding another multi-gigabyte JSON parse. Existing reports without
a sidecar retain a backward-compatible full-report fallback.

Instrumented trajectory-feature jobs store large numeric record tables as
hash-bound NumPy columnar artifacts beside the installed report. The JSON report
contains each artifact manifest path and SHA-256 instead of repeating every
numeric value in Python dictionaries. Downstream scalar-distribution and
threshold-state modules validate those hashes and read the arrays through
read-only memory maps. Scott, Rice, and explicit-bin scalar summaries use
constant-summary-memory streaming passes; exact Freedman-Diaconis quartiles
retain only the scalar value buffer needed for the quantiles. Threshold
populations, sensitivity counts, transitions, and residence summaries are
reduced in one segment-safe pass. Their frame assignments and residence runs
are also written as hash-bound numeric columns rather than rebuilt as large
lists of dictionaries. Legacy inline trajectory-feature reports remain
readable.

The finding picker uses the documented presentation order (FES, FES
conformations, silhouette-scored clustering, cluster conformations, RMSF, then
other physical results) and uses deterministic
lexicographic ranking rather than an opaque composite score. It is a triage
aid: report pointers, evidence class, effects, and corrected p-values remain
visible for scientific review. When the corresponding completed reports are
present, it names the highest-RMSF atom in each system, the largest pairwise
atom-level RMSF change, the strongest DCCM pair and pairwise DCCM change, and
the most occupied or most changed direct hydrogen bond. These extrema are
explicitly descriptive unless an upstream inferential method supplies a valid
p-value; the picker never converts a large effect into statistical
significance. The same transparent triage covers residue SASA, declared
observables, nucleic-acid and ion geometry, RDF peaks, circular dihedral
differences, water-mediated bridges, and every enabled default-off experimental
method. Experimental hydration components, geometric pockets, interaction
fingerprints and their censored persistence events, seed-gated random-feature
nonlinear kinetics, pathways, bridges, DFI/DCI, reweighting diagnostics,
reactive paths, and helical mechanics enter the same evidence-linked descriptive triage;
their effect sizes are never relabeled as statistical significance. When systems were executed into
separate reports, it selects the highest evaluated-frame coverage for each
system, compares those reports, and records every contributing path. This
keeps shorter RDF/resource pilots out of a comparison when a higher-coverage
completed report is available.
The JSON output records `method_evidence_coverage` for every completed report
and availability record considered. An unavailable method contributes zero
candidates and cannot become a significant or descriptive finding merely
because it was requested.
