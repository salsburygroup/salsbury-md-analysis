# Configuration and final reporting

`prepare-analysis` writes a complete `analysis-config.json`. Its versioned
schema is `salsbury-analysis-config-v1`, and its default is
`all_applicable`: every applicable scientific analysis and high-detail
conformational view that can be selected without inventing chemistry is
enabled; the deliberately coarse trace view is the documented exception.
Supply a reviewed partial or complete configuration with:

```bash
salsbury-md-analysis prepare-analysis ... --config my-analysis-config.json
```

Each `modules.<module_id>` entry has an `enabled` flag and an `options` object.
Options replace generated definition fields and then undergo the normal strict
project-schema validation. Disabling an upstream module also disables its
dependent analyses; the generated `module-coverage.json` records this rather
than producing a workflow that will fail later. Preflight, provenance, and
common-atom mapping infrastructure cannot be disabled. Each topology-derived
view has its own `enabled` flag and optional per-view `module_options`.
The high-detail common-heavy and interface views are enabled by default.
`macromolecular_trace` PCA and its state trajectories are off by default but
can be enabled independently for a deliberately coarse diagnostic.

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

The `execution` section describes one campaign envelope. For example,
`maximum_parallel_cpus: 32` and `maximum_hours_per_cpu: 24` means at most 32
CPUs for 24 wall hours, or 768 raw CPU-hours before the configured utilization
and pilot/finalization reserves. It does **not** mean 24 hours for every method.
`time_safety_factor`, `memory_safety_factor`, and
`censored_timeout_safety_factor` are independently configurable.
The analysis-config `memory_safety_factor` adjusts measured working-set
calibrations. It is distinct from the execution profile's scheduler-memory
factor and fixed overhead; both are recorded in `resource_safety_margins`.
The default 1.5 time factor is applied to modeled task costs before frame
allocation. The planner then uses only the configured utilization fraction
(normally 0.85) and removes pilot and finalization reserves. A Slurm profile may
add a further per-job wall-time margin; that is a timeout request, not additional
planned analysis time.
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
Direct
estimators receive small method-specific runtime pilots (normally 10--100
frames per replica at the TREX reference size and fewer for substantially
larger systems). Those pilots calibrate cost and memory only; they are not
scientific minima, convergence thresholds, or production recommendations.
The planner then allocates additional deterministic, full-timespan samples
within the shared envelope and reports every reduction. Preparation writes
`campaign-resource-plan.json`, which inventories base direct tasks, inherited
base tasks, and all enabled conformational-view tasks. Downstream FES,
clustering, and state methods share the selected physical-frame identities of
their view PCA; equivalent oligomer members add observations but do not add
physical frames or replicas. Every replan regenerates the PCA projection count
and uses it as the source count for each downstream clustering fit. The planner
iterates until those counts agree exactly. If discrete stride upgrades alternate
between two allocations, it derives the componentwise lower projection ceiling
from that cycle and replans the fits; it never falls back to a saved count.
A configured envelope that cannot fund the small
technical minima fails closed by default.

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

Measured catalogs describe the systems on which they were collected; they are
not universal per-frame constants. Planner tasks may therefore carry an
auditable workload multiplier before a catalog rate or memory maximum is
applied. The coordinate cache uses source atom count. Common PCA uses Cartesian
feature count and member-observation expansion. Methods driven by projected
observations rather than raw atoms keep their observation-based model. The
applied multiplier and reference workload remain in
`campaign-resource-plan.json`.
The generated local and Slurm adapters run base stages before conformational stages.
`execution.submission_adapter` selects `local` or `slurm`; local is the default.
Slurm mode requires a validated `execution.slurm_profile`, which keeps account,
partition, QoS, environment, command, and storage conventions outside scientific
configuration. The supplied `profiles/analysis/deac-default.json` selects the
Salsbury-group DEAC profile. Both adapters run the same workers and output contracts.
The Slurm adapter records task-specific scheduler requests in
`scheduler-resource-requests.json` and routes sufficiently large requests through
the profile's large-memory role. Its canonical launcher submits deterministic
dependency waves. Each wave stays within both `maximum_parallel_cpus` and the
aggregate `maximum_memory_gib`; downstream work waits for all jobs in the preceding
wave. This controls active compute allocation; queue
wait and scheduler backfill are not counted as analysis wall time.

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
time, CPU time, and peak resident memory. A final `afterok` job waits for every
base and conformational-view stage, verifies all expected reports exist, and
writes:

- `analysis_resource_and_frame_table.csv`, `.json`, and `.md`;
- `prioritized_findings.csv`, `.json`, and `.md`;
- compact finalizer status reports.

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

The finding picker uses the documented
presentation order—FES, FES conformations, silhouette-scored clustering,
cluster conformations, RMSF, then other physical results—and uses deterministic
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
differences, and water-mediated bridges. When systems were executed into
separate reports, it selects the highest evaluated-frame coverage for each
system, compares those reports, and records every contributing path. This
keeps shorter RDF/resource pilots out of a comparison when a higher-coverage
completed report is available.
