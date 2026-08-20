# Changelog

## 0.0.1.dev84

- Included the current and immediately preceding dependency-test records in the
  source archive. A packaging regression test now checks that the record named
  by the current development version exists and is explicitly listed in
  `MANIFEST.in`.
- Kept the dev83 analysis implementation unchanged; dev84 corrects package
  contents and is the clean private-repository candidate.

## 0.0.1.dev83

- Added the explicit `scientific_status: not evaluated` boundary to the final
  resource table, prioritized-findings output, and reporting-disabled marker.
  Generated finalizers now fail closed if either user-facing final summary
  omits that boundary.
- Made short, high-dimensional common-PCA plans use their full bounded sample
  space when the basis has at most 128 frames. The planner reports the expanded
  subspace explicitly, while larger calculations retain truncated
  oversampling. This prevents avoidable residual-gate failures without forming
  a dense feature covariance or changing the selected frames.

## 0.0.1.dev82

- Added a non-personal GitHub no-reply address to the package metadata so
  strict setuptools metadata validation succeeds without publishing a private
  contact address.
- Kept the dev81 analysis implementation unchanged; the new candidate is a
  packaging-metadata correction only.
- Rewrote the main clustering and MSM overview in plain language while keeping
  the method count, defaults, exact-assignment rules, and scientific limits
  unchanged.

## 0.0.1.dev81

- Corrected serialized OpenMM-system connectivity for systems that use angle
  or rigid-geometry constraints or whose serialized System cannot independently
  prove atom-order agreement with a PDB. The exporter now uses the
  atom-order-matched PDB topology as the default covalent graph and inventories
  System-only harmonic, custom, and constraint pairs without admitting them
  automatically. This prevents noncovalent or order-mismatched pairs from
  corrupting hydrogen-bond, make-whole, or molecular-graph analysis. Each
  System-only category remains available through a separate, explicitly
  reviewed opt-in switch with provenance.

## 0.0.1.dev80

- Unified conservative residue-name routing across generic selection,
  composition inference, conformational views, oligomer detection, automatic
  chemistry, hydrogen bonds, and state exports. Canonical RNA, common water
  models, histidine variants, and common ion aliases now receive consistent
  treatment without replacing explicit topology chemistry.
- Added a general-biomolecular-systems guide describing automatic behavior and
  fail-closed limits for proteins, DNA/RNA, protein–nucleic-acid complexes,
  oligomers, ligands/cofactors, ions, water, membranes, carbohydrates, and
  unknown polymers.
- Separated macromolecular comparison classes from full chemical composition,
  allowing ligand-bound and ligand-free systems to share a common-atom basis
  while preserving the ligand/cofactor distinction in every system record.
- Made serialized OpenMM-system connectivity more conservative: custom
  bond-force pairs are excluded unless explicitly requested. Added an
  immutable-state-to-PDB helper for
  accepted checkpoint coordinates.
- Added versioned measured-resource calibration catalogs that keep completed
  execution sidecars distinct from right-censored timeout evidence. Timeout
  target coverage is never reported as completed coverage; observed CPU cost is
  retained as a lower bound and receives a configurable censoring safety factor
  for planning.
- Added catalog extension from hash-verified prior catalogs, duplicate-evidence
  rejection, and a command-line builder accepting complete sidecars, timeout
  records, and base catalogs.
- Reserved configurable finalization headroom inside the single campaign CPU
  and wall-time envelope, and made the timing, memory, and timeout-censoring
  safety factors explicit in analysis configurations and generated plans.
- Added an Apollo v2 measured-resource catalog derived from separately
  preserved, hash-pinned TOP1 timeout evidence while retaining the earlier
  validated v1 catalog as immutable input. Raw scheduler identifiers and
  campaign paths are not shipped in the reusable source or package artifacts.
- Added public-safe resource-catalog redaction that preserves evidence hashes,
  measured costs, memory, coverage, and censoring status while removing private
  paths, scheduler identifiers, and hostnames. Publication/campaign-specific
  TREX and TBA harnesses and cached outputs are excluded from the v80 reusable
  source snapshot and remain preserved in their older locked lineages.

## 0.0.1.dev79

- Raised the generated FES state-trajectory total-frame guard to 500 while
  retaining the declared per-state integer stride. Together with the 250-state
  guard and the default basis-derived stride, this prevents round-up across
  many small basins from contradicting the generated total-frame ceiling.

## 0.0.1.dev78

- Added fail-closed compatibility for cached upstream reports that predate
  recorded module-contract hashes. Reuse now requires the cached report's
  original project manifest to remain present, match its recorded SHA-256, and
  yield the same module-specific contract as the current recovery manifest.
- Raised newly generated FES state-export and MSM state-capacity guards from 50
  to 250, matching the default total exported-frame ceiling. The observed state
  count and configured guard remain explicit in reports; kinetic validation
  gates are unchanged.

## 0.0.1.dev77

- Added opt-in, fail-closed iMWK-Means grid checkpoint/restart through
  `SALSBURY_MD_ANALYSIS_IMWKMEANS_CHECKPOINT`. Each completed k/p candidate,
  including all declared initialization ranks, is installed atomically and
  bound to the project, inputs, and complete clustering settings. A restart
  reuses only signature-matched candidates and recomputes any interrupted
  candidate without changing the fitted grid or scientific gates.

## 0.0.1.dev76

- Fail closed during campaign preparation when grouped decision-tree
  diagnostics cannot form their declared minimum number of segment/time-block
  groups. The planner now records the block size, group minimum, available
  group ceiling, and required per-replica frame minimum; equivalent oligomer
  members never manufacture independent groups.

## 0.0.1.dev75

- Fail closed during campaign preparation when the enabled information-dynamics
  estimators cannot provide their declared minimum number of segment-safe lag
  pairs. The exact lag, pair requirement, available-pair ceiling, and derived
  per-replica frame minimum now enter the campaign resource task contract.

## 0.0.1.dev74

- Connected every generated base, automatic-chemistry, and conformational-view
  task to its campaign-planner wall-time and peak-memory estimate. Slurm workers
  now receive safety-margined requests, with array-wide maxima and exact mapping
  provenance retained in `scheduler-resource-requests.json`.
- Added configurable automatic large-memory routing. The DEAC profile sends a
  worker array to the `large` role when any element reaches 96 GiB, without
  changing frame selection or relaxing a scientific gate.
- Extended local execution to reserve CPU and memory atomically, enforce both
  per-task and whole-campaign deadlines, export scheduler-equivalent request
  metadata to instrumentation, and retain those requests in attempt status.
- Moved the DEAC profile contract from a campaign-owned Python path to the
  dedicated versioned group environment at
  `/deac/phy/salsburyGrp/software/salsbury-md-analysis/environments/v74`.

## 0.0.1.dev73

- Added one portable execution contract with an explicit `local` or `slurm`
  configuration switch. Local execution now follows the generated dependency
  graph, enforces the campaign CPU and wall-time caps, preserves attempt-specific
  logs/status evidence, and runs without Slurm.
- Added strict cluster-profile validation and profile-driven Slurm account, QoS,
  partition, command, Python/environment, shared-write, and storage metadata.
  Included a generic template plus Salsbury-group DEAC analysis and Slurm profiles.
- Kept both adapters on the identical generated worker scripts, atomic output
  installation, content-hash reuse gates, sampling decisions, resource sidecars,
  and final reporting contract.

## 0.0.1.dev72

- Repair member-resolved state-coordinate exports by carrying the exact
  canonical-member alignment mapping into the export manifest. The mapping is
  now fail-closed, per system/replica/member, and content signed.
- Correct final resource/frame accounting for scalar distributions and scalar
  threshold states by deduplicating exact source-frame identities repeated
  across feature/state assignment series. Feature-frame work units remain
  available as observations but can no longer be mislabeled as trajectory
  frames, including when an older embedded planner benchmark carries the same
  ambiguity.

## 0.0.1.dev71 - 2026-08-18 candidate

- Extended the physical-frame accounting repair to reports whose exact frame
  coverage is recorded by unique top-level system/replica/segment rows rather
  than a top-level frame-selection object. Dihedral, multi-observable, and
  related work-unit totals can no longer be mislabeled as physical frames.

## 0.0.1.dev70 - 2026-08-18 candidate

- Corrected final resource and frame accounting so exact physical-frame
  selection takes precedence over module-specific atom-frame, residue-frame,
  candidate, surface-point, or feature observation counts. A regression test
  covers the 100-frame, 238,000 atom-frame SASA case found by the v68 matrix.

## 0.0.1.dev69 - 2026-08-18 candidate

- Added an explicit, optional OpenMM preparation fallback for systems lacking
  PSF, PRMTOP/PARM7, or bond JSON connectivity. It writes a reusable,
  hash-provenanced bond JSON from standard residue templates, PDB connectivity,
  and optionally reviewed residue-definition XML, while failing closed for
  incomplete multi-atom residues. Later analysis does not require OpenMM.

## 0.0.1.dev68 - 2026-08-18 candidate

- Completed generic dependency-stage coverage for the automatic chemistry
  command contract by adding nucleic-acid structure and geometry, and made the
  regression test assert the entire command map rather than a hand-maintained
  subset.

## 0.0.1.dev67 - 2026-08-18 candidate

- Assigned every automatically inferred chemistry command to the generic
  dependency-safe execution stages. Ion atmosphere, ion geometry, RDF,
  trajectory features, observables, Scott-first scalar distributions, and
  threshold states can now be prepared and submitted instead of failing
  closed as unmapped commands.

## 0.0.1.dev66 - 2026-08-18 candidate

- Generalized the single-system initializer from PSF-only connectivity to the
  same explicit PSF, portable bond-JSON, and Amber PRMTOP/PARM7 formats already
  supported by the analysis engine and comparative initializer. The
  `--connectivity` spelling is now available while `--psf` remains a compatible
  alias.

## 0.0.1.dev65 - 2026-08-18 candidate

- Added species-resolved periodic ion-atmosphere analysis with automatic
  inference for topology-supported cations and anions, including K, Mg, Zn,
  Na, Ca, Cl, and Fe when present. Geometric shell occupancy is kept separate
  from biological binding and oxidation-state interpretation.
- Added hash-bound measured resource calibration catalogs and conservative
  CPU, memory, physical-frame, and symmetry-observation overlays for the
  generic resource planner.
- Made single-system generic preparation infer applicable chemistry modules,
  including ion-atmosphere, ion-coordination, RDF, and nucleic-acid geometry,
  while retaining reviewable definitions and explicit non-applicability.

## 0.0.1.dev64 - 2026-08-18 candidate

- Added exact physical-frame and symmetry-expanded observation accounting to
  the multi-state MSM report so instrumented execution can produce its
  hash-bound resource sidecar without weakening the accounting gate.

## 0.0.1.dev63 - 2026-08-18 candidate

- Kept over-fragmented clustering partitions in the geometric clustering
  inventory while skipping only their MSM construction when the declared
  `maximum_states` limit is exceeded; one unsuitable method can no longer
  abort the other clustering and FES state models.
- Made grouped ML consume the KMeans assignment table's unit-independent
  `feature_values` contract, so configured tICA and trajectory-feature spaces
  receive the same validated grouped classification path as common PCA.

## 0.0.1.dev62 - 2026-08-18 candidate

- Separated structural-QC execution validity from scientific review. Readable,
  internally executable trajectories continue through the complete workflow
  when coordinate or chemical thresholds are exceeded; reports retain those
  observations as `qc_status: review_required`, leave scientific status not
  evaluated, and require explicit human judgment.
- Replaced residue-order peptide inference with explicit PSF/PRMTOP/bond-JSON
  C-N connectivity. Protein segments that share a PDB chain identifier can no
  longer create invented peptide or omega outliers across a segment boundary.
- Made the conservative steric screen topology-aware by excluding explicit 1-2
  and 1-3 pairs. If connectivity is unavailable, peptide, omega, and steric
  checks are explicitly not evaluated instead of producing misleading counts.

## 0.0.1.dev61 - 2026-08-18 candidate

- Unified CHARMM ion residue aliases across automatic chemistry and
  hydrogen-bond chemistry. `SOD`, `POT`, and `CLA` (plus charged/common
  aliases) are now classified as excluded ions rather than provisional
  ligands, allowing dynamic DNA-cation shell analysis to select sodium and
  potassium correctly without admitting ions as hydrogen-bond endpoints.

## 0.0.1.dev60 - 2026-08-18 candidate

- Hydrogen-bond comparisons now accept an optional `system_id` for each
  condition, filter a shared multi-system discovery report to that system, and
  fail closed when a multi-system report is supplied without explicit system
  selection. This prevents silent cross-condition pooling in batch studies.
- Automatic chemical perception now treats TOP1 EdU residue label `EDU` as a
  uracil-like nucleic acid and recognizes CHARMM `HSD`, `HSE`, and `HSP` as
  protein histidine protonation states.

## 0.0.1.dev59 - 2026-08-18 candidate

- Added exact source-frame, symmetry-observation, and representative-subset
  accounting to representative-frame reports so instrumented production jobs
  can emit hash-bound resource sidecars without treating the small exported
  representative set as the source analysis sample.
- Made state-coordinate export alignment use one exact all-topology common
  alignment intersection at the declared coverage gate. Variant systems may
  retain different molecular payload atoms while their exported coordinates
  remain aligned on the same protein/nucleic-acid structural basis.

## 0.0.1.dev58 - 2026-08-18 candidate

- Made the structural-QC inter-frame displacement basis an explicit named
  selection and set the generic workflow to inferred solute heavy atoms.
  Full coordinate, finite-value, extent, near-coincident, and configured
  chemical-integrity checks remain unchanged, while physically diffusing bulk
  water and mobile ions no longer create false macromolecular jump failures.
- Added report evidence for the displacement selection and selected atom count,
  plus a regression test in which a large bulk-water displacement does not
  conceal or redefine the solute structural gate.

## 0.0.1.dev57 - 2026-08-17 candidate

- Kept short conditional-trajectory replicas in pooled system information
  estimates while explicitly skipping only underpowered replica-level
  estimates.  Expected short ion-retention episodes no longer abort an entire
  information-correlation analysis.
- Extended symmetry-expanded oligomer-member PCA to use the exact ordered
  analysis-atom intersection across every system topology.  The alignment
  identity remains strict, variant-specific excluded atoms are reported, and
  member observations remain separately accounted from physical frames.
- Raised generic stage-one and alternative-clustering memory planning to 32
  GiB after a 79,789-projection Apollo job exhausted 16 GiB.  Frame selection,
  clustering settings, and scientific gates are unchanged.

## 0.0.1.dev56 - 2026-08-17 candidate

- Added exact source-frame, selected-frame, state-observation, and metric-value
  accounting to convergence reports so instrumented planner calibration cannot
  conflate frames with the number of scalar diagnostics per frame.
- Replaced the quadratic long-series autocorrelation loop with an
  overlap-normalized FFT implementation after a direct-versus-FFT numerical
  equivalence test.  Short series retain the transparent direct calculation;
  frames, ESS rules, split-mean gates, and block definitions are unchanged.
- Added bounded-memory production SASA output.  All selected frames and all
  surface atoms are still evaluated at the declared resolution; exact total
  timeseries, Scott-rule total distributions, and per-atom/per-residue first
  and second moments are retained without millions of duplicated Python
  dictionaries.  Full per-frame atom/residue detail remains opt-in.
- Made bounded SASA output the generic quickstart default while preserving an
  explicit `full_atom_timeseries` mode for small or publication-locked jobs.
- Corrected cross-variant common PCA mapping semantics.  The declared 95%
  structural-homology gate remains enforced on the alignment basis, while the
  analysis basis is the exact all-topology atom intersection.  Variant-specific
  excluded atoms, raw reference coverage, and the complete common-basis
  contract are reported instead of rejecting a scientifically valid shared
  heavy-atom basis because variants have different chemistry.

## 0.0.1.dev55 - 2026-08-16 candidate

- Added exact report-level physical-frame accounting to DCCM-derived
  correlation networks.  Instrumented execution now verifies the upstream
  selected count against the independent replica/segment evaluated-frame sum
  and no longer rejects a complete network report for missing accounting.
- Added `sparse_packed_v2` as the generic direct-hydrogen-bond default.  Each
  present event retains candidate identity, cutoff membership, donor-acceptor
  distance, and donor-hydrogen-acceptor angle in a fixed-width, base64-encoded
  payload while removing duplicated per-frame bond identifiers, cutoff lists,
  and Python geometry dictionaries.  Candidate chemistry remains frozen in
  the separate dictionary, absent events remain explicit evaluated zeros, and
  all selected frame locators are retained.
- Packed nonzero sensitivity-cutoff occupancy counts per segment instead of
  expanding one JSON dictionary per candidate, cutoff, and segment.  The
  candidate index, cutoff index, present count, evaluated count, and cutoff
  definitions remain exactly recoverable, while the primary occupancy table
  stays directly readable for reporting and finding selection.
- Added fail-closed selected-versus-evaluated frame reconciliation and explicit
  physical-frame, symmetry-observation, and candidate-frame workload
  accounting to direct hydrogen-bond reports.
- Reduced sparse hydrogen-bond candidate memory by keeping the compact frozen
  candidate dictionary and one atom-chemistry dictionary after discovery,
  rather than retaining a full nested chemistry record for every possible
  donor-hydrogen-acceptor triple.  Comparison, grouped classification, and
  finding selection accept the packed representation; finding selection now
  uses exact occupancy aggregates instead of rescanning multi-gigabyte frame
  matrices.
- Made cross-replica automatic-candidate harmonization incremental: only the
  running intersection, running union, and current replica set coexist, and
  harmonization enumerates compact atom-index triples rather than constructing
  nested output dictionaries.  Exact common, union, and per-replica exclusion
  counts are unchanged.  The sparse coordinate-evaluation path now builds the
  compact frozen candidate rows and one deduplicated atom-chemistry dictionary
  directly, so full per-candidate nested chemistry records are never
  materialized.
- Retained the dev54 connectivity-continuous unwrapping, rigid-body-invariant
  structural QC, DSSP element-column, and bounded nucleic-acid-memory repairs.

## 0.0.1.dev54 - 2026-08-16 candidate

- Changed generic and comparative production projects, including the optimized
  molecular-payload coordinate cache, from per-frame make-whole reconstruction
  to connectivity-aware component-continuous unwrapping.  Whole bonded
  components can no longer jump by periodic box vectors between decoded
  frames; exact component-anchor displacement gates remain fail-closed.
- Changed structural-integrity frame displacement to the maximum per-atom
  residual after a proper Kabsch rigid-body superposition.  Harmless global
  translation and rotation no longer masquerade as coordinate corruption,
  while internal or component-relative discontinuities remain errors.
- Made DSSP frame PDB generation write the topology-derived chemical element
  into classic PDB columns 77--78 even when the source PDB leaves those
  columns blank.  Invalid or missing inferred elements fail before mkdssp.
- Bounded nucleic-acid-geometry memory by retaining one raw scalar metric table
  and exact histograms/residence summaries without duplicating full ring-fit
  vectors, per-metric assignment rows, or individual residence-run rows.  This
  preserves every selected frame while removing the 60,000-frame 32 GB OOM
  failure mode.

## 0.0.1.dev53 - 2026-08-16 candidate

- Added one-time, topology/connectivity-derived hydrogen-bond candidate
  planning. Generic workflows now scale hydrogen-bond runtime by the exact
  common candidate universe, set the feature-observation gate to the selected
  frame count times that universe, and fail before reading trajectory
  coordinates when an explicit gate cannot accommodate the declared plan.
  The runtime model now scales the retained compiled-sparse timing by exact
  candidate count with an I/O floor for small candidate universes.
- Retained the v52 DNA torsion definitions and automatic cross-system
  hydrogen-bond candidate harmonization as the canonical generic behavior.

- Replaced post-hoc alternative-clustering fit ceilings with a globally
  coupled iterative campaign allocation. Every runnable clustering family is
  now a separate nonlinear logical task with its own exact integer stride,
  dynamic memory estimate, and balance group. Families executed serially by
  one view command share an execution bundle, so wall time is summed without
  falsely counting them as parallel jobs. The planner reapplies PCA projection
  selections and replans until all task strides and exact counts stabilize;
  nonconvergence fails closed. Ward and quality-threshold retain their
  all-observation-or-explicit-skip contract.

- Added an explicitly secondary MSM sensitivity for HDBSCAN dense cores. Noise
  observations split kinetic segments, no transition crosses a noise gap, and
  the partial model reports retained coverage and conditional interpretation.
  It can never enter or win primary MSM selection; HDBSCAN is now off by
  default while remaining independently switchable. Ward and quality-threshold
  are skipped before their quadratic work whenever an exact all-observation fit
  is unaffordable; quality-threshold is also skipped unless every observation
  is assigned. Complete exact results remain normal primary candidates.

- Separated PaLD from the eleven conventional clustering competitors. The new
  optional `pald_community_analysis` module reports a regularly strided,
  replica/member-aware cohesion matrix, local depths, the universal strong-tie
  threshold, connected communities, cores, boundary cohesion, and strongest
  intercommunity ties. It is off by default; its separately labeled sampled-
  community MSM is independently off by default and never competes for the
  selected conventional clustering MSM. A one-CPU 500-observation calibration
  now gives PaLD its own cubic-time/quadratic-memory planner model.

- Added atomic, connectivity-aware molecular-payload DCD caches for large
  campaigns. A cache reads the solvated source once, preserves frame and timing
  identities, retains complete solute/ligand/cofactor/ion coordinates, removes
  bulk water, writes portable subset connectivity, and remains deliberately
  unaligned so every downstream view applies its own alignment. Water-dependent
  analyses continue to use the original solvated trajectories.

- Added a scheduler-neutral hard campaign CPU/wall/memory allocator and wired
  its allocations into generated direct-estimator and conformational-view
  frame selections. `campaign-resource-plan.json` now inventories base direct,
  inherited base, PCA/FES/clustering, and state-postprocessing tasks under one
  limit; matched observation consumers inherit one upstream selection.
- Replaced fixed 1,000-frame pilot floors with 10--100-frame method-specific
  technical pilots that shrink for systems larger than the TREX calibration;
  pilots are explicitly not scientific sufficiency criteria.
- Added automatic pooled oligomer-member interface comparative views while
  retaining physical-frame and member-observation accounting separately.
- Separated conformational feature/alignment selections from exported molecular
  payloads. Complete-solute exports now retain hydrogens and canonicalize full
  member protein/DNA units rather than exporting only PCA feature atoms.
- Added configuration contracts for per-system/shared comparisons, oligomer
  pooling, exports, and campaign-wide execution limits.
- Removed the duplicate coarse common-PCA/FES/clustering branch from comparative
  quickstarts; configured topology-aware shared views are now the sole automatic
  conformational branches, with trace analysis still off by default.
- Comparative quickstarts now also generate explicit system-isolated
  PCA/FES/clustering view families by default. Each family pools that system's
  replicas, includes oligomer-member views when detected, receives its own
  content-hashed preflight, and is budgeted together with the shared-basis
  comparison under the one campaign envelope.
- Generated Slurm launchers now dependency-batch per-system preflights and view
  families, run views after base stages, and cap simultaneous CPU reservations
  at `execution.maximum_parallel_cpus` instead of multiplying the configured
  ceiling by the number of view projects.

- Frame accounting now prefers an explicit report-level evaluated/observation
  count before recursive diagnostic fallbacks, preventing per-candidate
  hydrogen-bond segment records from inflating a 4,200-frame workload.
- Generated workers now create report-hash-bound compact resource/finding
  sidecars and validate both artifacts on reuse. Finalizers stream report
  hashes and avoid reparsing multi-gigabyte SASA or clustering JSON solely to
  build summary tables.
- Final resource tables now separate PCA basis, model-fit, full-assignment,
  and silhouette-evaluation workloads and recover member-expanded information
  analysis counts from replica rows. Finalization fails closed on incomplete,
  uninstrumented, or frame-unaccounted analysis reports.
- Expanded transparent finding triage across SASA, declared observables,
  nucleic-acid and ion geometry, RDFs, dihedrals, and water-mediated networks.
  It reports atom/residue/interaction-level descriptive extremes and pairwise
  differences without relabeling them as statistically significant.
- Comparative quickstart requests now accept either a PSF or validated portable
  `salsbury-bonds-v1` connectivity JSON per system, allowing immutable matched
  studies to reuse their content-hashed bond topology without conversion.
- Final resource tables recover exact physical/member frame coverage from an
  instrumented report's embedded planner benchmark when the scientific report
  does not expose a canonical top-level frame-count field.
- Structural-integrity planning now uses the observed 300-frame full-chemistry
  Apollo runtime (4.430551 s/frame). The calibration is explicitly labeled
  timing-only because every frame was evaluated but the trajectory failed the
  prespecified omega and displacement QC gates.

- Added a self-service `prepare-comparison` workflow for two or more systems,
  including unequal replica lengths and large variant panels. It generates
  outcome-independent common-heavy, interface, trace, and strict equivalent-
  oligomer views on shared PCA bases/common grids, while retaining per-system,
  replica, physical-frame, and oligomer-member provenance.
- Generalized quickstart resource and conformational-view planning from one
  equal replica length to the exact per-replica frame-count vector.
- Replaced optional-observable and secondary-structure timing proxies with
  retained Apollo measurements from all 30,000 frames and 300 uniformly
  distributed frames, respectively; automatic plans still apply their 1.5
  timing safety factor and report any subsampling.

- Accelerate structural-integrity near-coincident and element-radius clash
  searches with deterministic SciPy cKDTree candidate generation while
  retaining exact thresholds, exclusions, counts, and sorted examples. This
  removes the all-atom Python neighbor-loop bottleneck exposed by the retained
  85,199-atom TREX resource benchmark.
- Cover every direct planner method in the retained resource benchmark harness;
  explicit indexed hydrogen bonds and optional observables can no longer be
  omitted from calibration runs.
- Validate the default all-applicable configuration and both all-pairs and
  reference-versus-all finding selection for a 20-system variant batch.

- Fixed conformational-view workers to export the content-hashed preflight
  report whenever they reuse an upstream common-PCA report. This preserves the
  cache signature gate for FES, clustering, information, and lagged analyses.
- Added bounded deterministic refinement to large-view truncated PCA. The
  solver records residuals at each declared power-iteration budget, preserves
  the original frame selection and residual gate, and fails closed if the last
  attempt is still inaccurate.
- Fixed conformational-view submission final-job reporting so shell variables
  expand before dependent finalization is submitted; a regression test now
  rejects the previous literal `${VIEW_*}` output.
- Extended transparent finding triage to name the most flexible atom per
  system, largest pairwise RMSF change, strongest DCCM pair/difference, and
  highest or most changed direct-hydrogen-bond occupancy. These remain
  descriptive unless an upstream method supplies valid inferential p-values.

- Added strict equivalent-oligomer detection and a symmetry-expanded canonical
  member view. Each protein or protein-DNA member is independently made whole,
  aligned, projected, clustered, assigned to FES basins, and exported with
  `member_id`; original physical frames and independent replicas are never
  inflated by the member multiplier.
- Made TICA, information dynamics, MSM, FES blocking, and quadratic-clustering
  sampling member-safe. Continuous PCA/TICA score cross-correlations and
  categorical state concordance/Cramer's V are calculated across matched
  physical frames without causal interpretation.
- Added `salsbury-analysis-config-v1`, generated by default and accepted with
  `prepare-analysis --config`, with all-applicable defaults, dependency-aware
  module disabling, per-module/per-view options, and all-pairs or
  reference-versus-all comparison policy.
- Added per-analysis measured host, Slurm, CPU, wall-time, peak-memory, physical
  frame, and symmetry-expanded observation evidence, plus a final dependency-
  gated CSV/JSON/Markdown resource table and transparent FES-first finding
  prioritizer with Benjamini-Hochberg correction only where p-values exist.
- Added direct resource-planner benchmark adapters to instrumented reports and
  signed alternative-clustering workload identity for quadratic pilot reuse.
- Added deterministic replica-balanced sampled fitting for quadratic
  alternative-clustering sweeps, algorithm-specific assignment of every source
  frame, explicit approximate-versus-exact extension labels, full populations,
  and optional cross-budget adjusted-Rand sensitivity.
- Replaced the shared alternative-clustering fit sample in generated workflows
  with automatically scaled, per-algorithm exact integer strides; Ward and
  quality-threshold retain their full-fit-or-skip contract.
- Added fixed-plus-quadratic multi-pilot calibration and resource-envelope
  planning for alternative clustering. Routine quick starts request one fit
  budget; B-versus-2B remains opt-in, and cubic PaLD remains separately gated.
- Reused Minkowski distance matrices and Ward linkage across parameter values
  and vectorized PAM/MWPAM medoid updates so an extensive sampled sweep does
  not repeat avoidable quadratic Python work.
- Allowed a complete content-hashed common-PCA report to be reused by a
  different downstream project only when a conservative module-specific
  scientific-contract hash matches exactly; project-path/hash reuse remains
  the default exact case.
- Normalized exact residue-key selections in compiled project contexts instead
  of assuming every non-preset selection was an atom-name list.
- Made each generated conformational-view project bind its semantic `analysis`
  and `alignment` roles to the exact selections declared by its PCA contract.
- Added outcome-independent automatic global common-heavy, chemical-interface,
  and macromolecular-trace conformational views, including exact residue-key
  selection and protein-only versus protein–nucleic-acid routing.
- Added comparative multi-topology view planning that harmonizes modification
  positions, unions reference-contacting residues across conditions, and
  reports both per-reference and common atom counts before trajectory outcomes.
- Added feature-aware PCA resource planning and a deterministic randomized
  leading-component solver for large common-heavy views. Basis sampling remains
  balanced across each replica's full time span, every source frame is projected
  by default, and numerical/resource gates fail closed.
- Added selection-scoped immutable state-coordinate exports and automatic
  solute-only FES-basin trajectories/representatives to each generated
  conformational-view workflow.
- Added outcome-independent automatic hydrogen-bond candidate harmonization
  for homologous multi-system comparisons. The common classification matrix
  can now retain atom-index triples present in every replica before any
  coordinates or occupancies are evaluated, while explicitly reporting
  system-specific chemical candidates that remain covered by per-system
  discovery rather than encoding chemical absence as a conformational signal.
- Split generated cluster execution into preflight plus three dependency-safe
  arrays and added fail-closed, hash-matched reuse of common-PCA, DCCM,
  RMSD/Rg, and K-means reports so derived methods do not repeat expensive
  trajectory scans. Every cached content signature must also match a freshly
  rehashed preflight, preventing silent reuse after in-place input changes.

## 0.0.1.dev0 - unreleased

- Added a one-command PDB/PSF/DCD initializer with pre-output PSF connectivity
  geometry validation, method-specific wall-time planning, deterministic
  full-timespan frame selection, reproducible Slurm launchers, and explicit
  accounting for every deferred question-specific module.
- Include each explicitly supplied PSF, PRMTOP/PARM7, or portable bond JSON in
  the structured preflight report with its content hash and a fail-closed atom
  cardinality comparison against both topology and trajectory.
- Make generated preflight and module launchers safely resumable: complete
  reports are validated and reused, incomplete or malformed finals are never
  overwritten, and new finals use an atomic fail-if-present installation.
- Discover `mkdssp`/`dssp` beside the active Python interpreter as well as on
  `PATH`, and fail during preparation when an explicitly supplied executable is
  invalid. Conda-prefix installations therefore work without manual PATH edits.
- Added reader-level DCD record skipping for fixed-stride structural QC,
  RMSD/Rg, RMSF, and dihedral execution while preserving source/evaluated frame
  identities and validating skipped record envelopes.
- Recognize the legacy three-character CHARMM PDB water label `TIP` consistently
  as solvent in generic selections, quick-start composition, direct-interaction
  chemistry, and water-mediated-network identity. This prevents truncated
  `TIP3` residues from entering solute-heavy mappings.
- Define each discovered water by its oxygen atom and explicit O--H
  connectivity rather than by a possibly wrapped PDB residue identifier. Large
  legacy solvated PDB files can therefore reuse residue numbers without merging
  distinct waters or miscounting system composition.
- Made matched-validation launchers and report-derived builders require an
  explicit `technical_status: complete`; missing status now fails closed.
- Added a generic two-condition direct-hydrogen-bond comparator that groups
  connectivity-equivalent donor hydrogens before occupancy, preserves
  per-replica evidence, requires identical interaction scopes and cutoff
  definitions, and aligns nonidentical topologies only through explicit,
  fail-closed chemical-homolog mappings supplied by the project.
  Candidate-eligible unobserved interactions are zeros, while topology-specific
  chemical absences are null and excluded from numeric difference ranking.
- Documented separate condition-specific discovery and separate interaction
  scopes (for example protein--nucleic-acid and protein--protein) so a focused
  candidate universe cannot silently omit another scientifically required
  interaction family.
- Made cluster interpretation explicitly report-and-retain: strong association
  with system, replica, starting conformer, or preparation lineage remains a
  scientific characteristic of the descriptive partition and never discards
  assignments, populations, representative structures, or state trajectories.
  Only metastable-state and kinetic-state claims require independent evidence.

- Added a minimal-input, method- and system-size-aware automatic sampling plan
  whose ceilings apply to the pooled estimator across replicas, except for
  replica-resolved RMSD/Rg; every subsample is explicitly reported.
- Made B-versus-2B and exploratory replica diagnostics opt-in and added
  autocorrelation-adjusted effective-sample-size uncertainty without treating
  replica agreement or leave-one-replica-out as acceptance measures.
- Added smoothing-resolved FES-basin silhouette diagnostics, a Scott-rule
  histogram-first presentation adapter for all non-RMSD scalar time series,
  a versioned scientific reporting order, and RMSF PDB-B-factor/VMD-cartoon
  exports.

- Added reported automatic resource-envelope frame selection with an all-frame
  preference, deterministic replica-balanced subsampling, retained calibration
  and cost estimates, and optional `off`/`recommend`/`require` frame-budget
  sensitivity policy.
- Decode selected DCD coordinate records into compact NumPy arrays and preserve
  that backing through scoped periodic reconstruction, avoiding large transient
  Python-float/tuple populations for solvated trajectories while retaining the
  same atom indexing and coordinate values.
- Added an installed resource-planning command that fits fixed DCD/trajectory
  scan overhead plus incremental estimator and output costs from one or more
  retained pilots on the actual method/project workload.
- Vectorized sparse direct hydrogen-bond distance screening and limited angle
  evaluation to cutoff-near candidates while retaining exact triclinic imaging.
- Added an isolated frame-coverage benchmark harness that records wall/CPU
  time, peak resident memory, scheduler/environment identity, compact observed
  counts, and a compressed full module report without changing input data.
- Made connectivity-aware reconstruction observable-scoped: every complete
  bonded component that influences a declared observable is rebuilt, while
  unrelated solvent is left out of solute-only estimators. Water-network
  execution still screens all exchanging water oxygens per selected frame with
  exact periodic geometry and retains only sparse cutoff-near records.
- Changed trajectory-feature execution to retain derived values and frame
  identities rather than whole solvated coordinate frames.
- Replaced nested-Python Cartesian covariance, DCCM covariance, and PCA power
  iteration with vectorized online moments and symmetric LAPACK eigensolving,
  retaining numerical-rank, deterministic-orientation, and residual gates.
- Separated PCA basis-fit sampling from projection sampling, allowing a
  balanced, sensitivity-tested covariance basis to project every source frame
  for FES, clustering, state assignment, trajectory export, and representative
  structures.
- Added solvent-aware `complex_trace`, `macromolecular_backbone`, and
  `solute_heavy` project-selection presets; the existing `backbone` preset now
  excludes common solvent and monatomic-ion residue names instead of selecting
  water oxygens named `O`.
- Distinguished DCD segment-header step resets from genuine nonzero DCD step
  inconsistencies during preflight; reset headers now warn and defer to the
  explicit continuous frame axis and external lineage.
- Limited SASA van der Waals-radius validation to atoms actually selected as
  surfaces or occluders, so excluded ions do not invalidate a solute-only SASA
  calculation.
- Added automatic topology-template direct hydrogen-bond discovery for standard
  protein and nucleic-acid chemistry, with scope-level selection, auditable
  provisional ligand fallback, and prespecified distance/angle sensitivity
  reporting; retained explicit atom-index discovery for publication locks.
- Added opt-in chunked sparse direct-hydrogen-bond output with an explicit-zero
  contract while preserving dense legacy output as the compatibility default.
- Added scalable one-water-mediated hydrogen-bond networks with automatic
  solute/water chemistry, cell-list neighbor search, exact triclinic geometry,
  cutoff sensitivity, direct coincidence, water-exchange-aware residence,
  representative frames, and descriptive network summaries.
- Added explicit 8-oxoG (`8OG`, `8OX`, and `OX3`) nucleic-acid templates,
  including the lesion O8 acceptor and N7-H donor, so modified DNA is not
  silently treated as a provisional ligand.
- Added an OpenMM-System connectivity exporter that derives portable bond JSON
  from explicit force-field bond terms and constraints without residue-template
  or distance inference. Water-edge pairing now partitions by declared
  interaction scope before within-water expansion.
- Added a deterministic uniform per-replica frame-budget policy with selected
  DCD payload skipping, explicit source/selected-frame coverage reporting, and
  a public-safe all-replica published-Miller TREX1 lesion execution record.
- Extended the balanced frame-budget contract to PCA, DCCM, SASA, automatic
  direct hydrogen bonds, RDF, DSSP, and DSSR; all retain full-frame defaults,
  emit per-replica and per-segment coverage, warn on subsampling, and preserve
  every intermediate frame required by continuous unwrapping.
- Cross-validated automatic protein/DNA hydrogen-bond chemistry, geometry,
  nine-cutoff decisions, and replica occupancies on 75 hash-pinned real TREX
  frames against MDTraj and OpenMM; retained two independently confirmed
  terminal DNA donor bonds omitted by MDTraj's PDB bond inference.
- Established the MD-only `salsbury-md-analysis` distribution, Python package,
  and command-line name.
- Registered the 38-module `standard_md_v1` trajectory-analysis profile.
- Separated docking functionality for the future `salsbury-docking`
  repository.
- Removed private cluster paths, internal migration catalogs, and
  project-specific validation evidence from the public repository surface.
- Added fail-closed manifests, topology/trajectory preflight, structural QC,
  connectivity-aware make-whole and continuous unwrapping, analysis methods,
  regression contracts, generated references, environment specifications, and
  public repository policies.
- Closed technical destinations for all 434 reviewed non-docking legacy
  capabilities, while preserving scientific-support gates and routing 62
  docking capabilities to the separate docking review.
- Added the reusable legacy gaps found during source-body review: richer group
  distances, automatic independent-axis FES binning, segment-safe lagged
  correlation, correlation-profile clustering, and autocorrelation sequences.
- Cross-validated the original 38-module snapshot plus automatic hydrogen-bond
  discovery through 29 independent
  real-trajectory, reference-library, formula, legacy-family, and reporting
  cases; retained nonpassing convergence and input-quality findings.
- Added a public-safe scientific-validation snapshot while keeping private
  trajectories, storage paths, and full evidence in controlled records.
- Refused project-level SASA below 240 sphere points, warned below the
  independently checked 960-point setting, and documented the observed
  resolution sensitivity.
- Documented and tested preservation of DSSP 4.6's PPII `P` code and corrected
  structural-QC documentation to match the implemented chemical checks.
- Added multi-smoothness FES minima/catchment sensitivity while preserving raw
  histograms and frame identities at every declared Gaussian width.
- Added exact-or-seeded-estimated silhouettes, broader clustering parameter
  grids, and immutable state trajectory/representative-structure exports with
  collision refusal and checksummed lineage.
- Added triclinic, volume-normalized RDFs and reference-defined native-contact
  fractions to close reusable AMD analysis gaps.
- Added explicit/Scott/FD/Rice scalar feature histograms and segment-safe,
  boundary-censored residence runs.
- Added a shell-free, versioned x3dna-dssr JSON adapter for replica-resolved
  nucleic-acid structural motif counts and numeric descriptors.
- Added individual DNA-ring plane/torsion geometry, fused-ring and stacking
  relationships, and replica/block/stationarity reporting with independent
  Scott/FD/Rice distributions.
- Added ion-agnostic bound-site and ion-pair geometry with ligand occupancy,
  minimum-image distances, ideal-polyhedron continuous-shape matching, and
  shared-ligand bridge descriptors.
