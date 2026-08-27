# Frame coverage and expensive-analysis budgets

Frame selection is an explicit scientific contract, not an implicit speed
optimization. Nearly all estimators pool selected frames across replicas, so
method ceilings are **total pooled-frame ceilings**, not per-replica ceilings.
The planner allocates a pooled ceiling equally across replicas. Replica-resolved
RMSD/Rg is the explicit exception and keeps a per-replica ceiling. The toolkit
defaults to `fixed_stride_v1`; with `frame_stride: 1`, every available frame is
evaluated when it fits the method envelope.

For the current three-replica TREX lesion data, manifests that point to the raw
10-ps trajectories expose 10,000 frames per replica and 30,000 total.  Manifests
that point to the 100-ps aggregate trajectories expose 1,000 frames per replica
and 3,000 total.  The code reports what the supplied files contain; it
does not silently treat an aggregate trajectory as the raw trajectory.

## Supported policies

```json
"frame_stride": 1,
"frame_selection": {
  "mode": "integer_stride_per_replica_v1",
  "stride": 20
}
```

`integer_stride_per_replica_v1` applies exactly `0, stride, 2*stride, ...` over
each replica's concatenated segment order. It retains frame zero, does not force
the final frame, and requires `frame_stride: 1` so two sampling rules cannot be
mixed. The planner starts with a budget-derived stride, calculates the exact
retained count for every replica, and adjusts the integer stride until the real
count satisfies the resource ceiling. The requested budget is therefore a hard
ceiling rather than a promise of an unattainable sample count. Every execution
report records the exact stride, count, and coverage.

`uniform_per_replica_budget_v1` is retained only so frozen historical projects
can be reproduced. It uses a deterministic near-stride with possibly unequal
gaps and is never emitted by the current generic workflow.

Alternative-clustering fits use
`algorithm_specific_integer_stride_v1`. Every enabled algorithm has its own
explicit plan and, when runnable, an `integer_stride_per_replica_member_v1`
stride rather than a shared nominal observation budget. Execution applies each
stride continuously over every concatenated replica-member timeline and does
not restart at segment boundaries. Algorithms capable of out-of-sample
assignment then assign all projected observations; Ward and quality-threshold
clustering are skipped when a full-observation fit is not affordable. The
outer campaign planner treats every runnable family as an independent
nonlinear allocation, recomputes its CPU and memory cost at every candidate
stride, and couples those choices to the PCA projection selection. It then
reapplies the projection and replans until every exact stride and retained count
stops changing. Families executed serially by one view command share an
execution bundle: their CPU time and bundle wall time are summed, but they do
not incorrectly inflate useful parallel CPU capacity. The retained bundled
TREX timing is divided into explicitly provisional per-family calibrations
until isolated family pilots replace it; this calibration uncertainty is
separate from the exactness of the selected integer strides.

An input-aware planner may instead resolve an automatic resource envelope:

```json
"frame_selection": {
  "mode": "auto_resource_budget_v1",
  "target_wall_seconds": 14400,
  "estimated_seconds_per_frame": 0.25,
  "minimum_frames_per_replica": 500,
  "fixed_overhead_seconds": 30,
  "safety_factor": 1.5,
  "sensitivity_check_policy": "off",
  "estimated_peak_memory_mib": 2048,
  "target_memory_mib": 16384,
  "calibration_id": "site-and-method-specific-benchmark-id"
}
```

The estimate must come from a retained, method- and environment-specific
calibration. The planner evaluates all frames when they fit. Otherwise it
resolves the finest common integer stride inside the wall-time envelope.
It reports the requested and resolved modes, full and selected wall-time
estimates, calibration ID, resource envelope, exact coverage, and the reason
subsampling was triggered. A frame-independent memory excess fails instead of
pretending that fewer frames will fix a quadratic feature/atom matrix.
The separate bounded silhouette calculation is an intentional exception: when
an exact silhouette is too expensive, a seeded random set of focal observations
is scored against the full fitted partition and the estimate is labeled as
such. It does not refit clustering on that focal sample.

`sensitivity_check_policy` is `off`, `recommend`, or `require`. `off` is the
default. A strict B-versus-2B check is optional even for publication work; its completed, skipped,
or unavailable state and any rationale belong in the publication analysis
lock. The planner never schedules the doubled-budget run automatically. A
publication lock remains valid with `off`/`not_applicable`, or with a skipped
recommended check and a recorded rationale. `require` is used only when the
project owner explicitly promotes the check to a project gate. An all-frame
result does not need frame-budget sensitivity.

## Minimal-input automatic plan

The higher-level planner needs only the system manifest, simulation kind, and
the two explicit owner choices. It inspects topology records and trajectory
headers read-only, then emits a plan for every registry module:

```bash
PYTHONPATH=src python -m salsbury_md_analysis \
  plan-automatic-sampling system.json --simulation-kind unbiased_md
```

Add `--b-vs-2b` only when a base-versus-doubled sensitivity comparison is
wanted. Add `--replica-diagnostics` only for optional exploratory diagnostics;
they are not recommended acceptance measures. Replica agreement and
leave-one-replica-out are never scientific gates. Analysis modules may report
time-block or autocorrelation diagnostics after execution, but the planner does
not estimate autocorrelation times or event rates and does not use a short pilot
to lower sampling. The plan reports every subsampled method,
selected/source counts, coverage fraction, pooled versus per-replica scope,
and inherited downstream frame identities.

## Scientific minima used by the planner

Each framed method has a permissive feasibility floor: a minimum number of
retained samples per physical replica and a minimum pooled count per system.
There is no universal effective-sample-size threshold, minimum trajectory
duration, or required percentage of the timeline. Short trajectories remain
eligible when they contain enough saved observations. Duration, selected span,
stride, and spacing are reported as provenance.

The planner uses a maximum retained-frame separation only for information
dynamics, tICA, and MSMs. It also reads each project's configured lags and
minimum valid pair or transition counts. These requirements are evaluated on
replica- and member-segment-safe sequences. The planner does not estimate
autocorrelation times or event rates. Runtime pilots calibrate CPU and memory
costs only.

| Methods | Frames per replica | Frames per system | Temporal rule |
|---|---:|---:|---|
| Structural-integrity QC | 100 | 500 | Count floor; continuity preprocessing still scans every raw frame |
| Replica RMSD/Rg | 100 | 100 | Count floor; order and times are retained for reporting |
| RMSF, dihedrals, nucleic-acid geometry, optional observables, scalar distributions, H-bond pattern/comparison, grouped regularized classification, RMSF permutation | 200 | 1,000 | Count floor; no spacing gate |
| H bonds, automatic H-bond discovery, ion coordination, ion atmosphere, RDF, trajectory features | 200 | 1,000 | Count floor; no spacing gate |
| Water-mediated H-bond networks, DSSP, nucleic-acid structure, SASA | 100 | 500 | Count floor; no spacing gate |
| DCCM | 250 | 1,000 | Count floor; no spacing gate |
| Individual/common PCA | 250 | 1,000 | Count floor; no spacing gate |
| Generalized correlation, correlation networks, FES, clustering, representatives/exports, grouped ML | 250 | 1,000 | Count floor; no spacing gate |
| PaLD, when explicitly enabled | 20 | 100 | Count floor; no spacing gate |
| Information dynamics | 500 | 2,000 | At most 0.50 ns between retained frames plus configured lag and total valid-pair minimum |
| tICA | 500 | 2,000 | At most 0.50 ns between retained frames plus configured lag and valid pairs in every segment |
| MSMs | 500 | 2,000 | At most 0.50 ns between retained frames plus largest configured lag and total valid-transition minimum |
| Scalar threshold-state dynamics | 250 | 1,000 | Ordered, segment-safe series; selected spacing is reported, with no universal gap gate |
| Convergence/uncertainty | 250 | 250 | Ordered per-replica series; duration and spacing are reported, with no universal gap gate |

The complete per-module contract, including inherited upstream sampling and
source-limited failures, is written into `sampling-plan.json` and
`campaign-resource-plan.json`. Publication-specific analysis may raise these
floors in its locked configuration; reducing them changes the scientific
standard and is not an automatic resource-planning action.

Thermodynamic and ensemble estimators have no time-gap rule because randomizing
their stride-1 frame order would not change the estimator. Their count floors
are deliberately lenient feasibility minima, not convergence claims. Ordered
scalar-state and convergence reports preserve their temporal lineage without a
universal duration or gap threshold; project-specific validation may impose a
stronger requirement.

The automatic plan gives every directly sampled estimator an exact integer
stride. Modules with the concatenated-replica selection interface receive
`integer_stride_per_replica_v1`; older streaming interfaces receive the same
resolved integer as `frame_stride`. In both cases the planner recomputes the
count, coverage, and selected-wall-time estimate after resolving the stride.
The one-command initializer writes the resolved stride into the project rather
than leaving it as an advisory value.

Concatenated-replica stride modules emit:

- total source, selected, and coverage counts;
- source, selected, coverage, and endpoint indices for each segment;
- source, selected, coverage, and exact stride for each replica; and
- a `FRAME_SUBSAMPLING` warning whenever selected frames are fewer than source
  frames.

Integer-stride modules emit source, decoded, and evaluated frame counts for
each segment, retain original source-frame indices and physical times, and
emit the same subsampling warning. A selected DCD payload is decoded; an
unselected DCD payload is record-validated and skipped at the reader level.

For `make_whole`, an independent frame can be skipped before coordinate
reconstruction.  For `unwrap_continuous`, every raw frame is decoded and passed
through the continuous image-history calculation; only the expensive estimator
is sampled.  This preserves scientific continuity while reducing estimator
cost.

Connectivity reconstruction is also selection-aware. A solute-only observable
rebuilds every complete bonded component containing one of its declared atoms,
but does not rebuild unrelated waters or ions. Water-mediated networks are
different: every selected frame screens all water oxygens with a periodic
cell-list so exchanging water identities remain eligible, while only
cutoff-near waters create sparse edges and paths. Coordinate frames are not
retained after those derived records are formed. Selected DCD records are
decoded into compact numeric arrays rather than per-coordinate Python objects,
and scoped make-whole/continuous-unwrapping preserves that compact backing.
The current reader still reads all atom coordinates in each selected DCD frame;
atom-selective DCD payload reads remain a future optimization.

The balanced contract is available for common and individual PCA, DCCM, SASA,
automatic direct hydrogen-bond discovery, one-water networks, RDF, DSSP, and
DSSR. For PCA, basis fitting and projection have independent contracts: a
balanced basis sample can be sensitivity-tested while
`projection_frame_selection` uses every source frame. Downstream FES,
clustering, MSM, and representative exports inherit those exact projection
identities rather than resampling independently. Structural QC, RMSD/Rg, RMSF,
trajectory features, dihedrals, nucleic-acid geometry, and ion geometry remain
all-frame streaming analyses by default, but the automatic planner may assign
and report a non-unit integer stride when the all-frame runtime exceeds the
declared method envelope.

## Method-specific production starting ceilings

These are production sampling ceilings, not the 10--100-frame technical runtime
pilots and not universal scientific defaults. Use
all frames whenever the measured cost is acceptable.  When a budget is needed,
increase it and compare the scientific summaries before treating a result as
stable.

| Analysis | Initial total pooled ceiling at the 85,199-atom reference size | Current basis |
|---|---:|---|
| Streaming scalar/local geometry, RDF, RMSF | 100,000 | linear or streamed implementation; use all 30,000 TREX frames where tested |
| DCCM on the declared 474-atom selection | 30,000 | 30,000-frame TREX run completed; selected-atom matrix controls scaling |
| Common PCA on the declared 474-atom selection | 30,000 | 30,000 basis and projection frames completed on TREX |
| Quadratic alternative-clustering sweep | planner-selected sampled fit; all frames assigned afterward | pending 750/1,500/3,000-fit TREX calibration; one normal fit budget, optional B-versus-2B |
| Automatic direct H bonds on the templated TREX system | 30,000 | 30,000 frames and 64,640 candidates completed |
| Other PCA, explicit H bonds, and comparable moderate methods | 10,000 | conservative pooled default pending matched full-frame evidence |
| One-water networks, 960-point SASA, DSSP, or DSSR | 1,000 | expensive neighbor, surface, or external-executable work |

For three equally long replicas, 10,000 pooled frames resolves to 3,333 per
replica (9,999 retained) and 1,000 resolves to 333 per replica (999 retained).
The one-frame remainder is intentionally not assigned to a favored replica.
These are example operational maxima, not minimum scientific sample sizes.

A production run should retain the pilot and final selected coverage records;
when a doubled-budget check is chosen, retain it as well. Budget agreement is a
sensitivity check, not proof of
equilibration, independence, convergence, or adequate biological sampling.

Alternative clustering uses per-algorithm observation-fit plans rather than
discarding unsampled frames from its final populations. The retained bundled
calibration supplies total timing evidence, while algorithm-specific linear or
quadratic provisional profiles participate directly in the globally coupled
CPU, wall-time, and dynamic-memory optimization. Distinct algorithms may
receive the same stride when the converged discrete allocation happens to
select it; they are nevertheless optimized separately. Each report records the
fitted and fully assigned counts separately. A second budget is generated only
when the sensitivity policy is explicitly `recommend` or `require`.

Resource gates remain separate from frame selection.  For example,
`maximum_feature_observations`, `maximum_observations`, `maximum_frames`, and
water-network path/pair limits fail closed if even the selected workload is too
large.  They must be sized from a read-only inventory and retained in the
analysis lock.
