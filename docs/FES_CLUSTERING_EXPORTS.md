# FES, clustering, and state-coordinate exports

## Histogram and FES contract

`pca_fes_basins` bins PC1/PC2 (or another explicitly declared component pair)
independently. Automatic choices are `scott`, `freedman_diaconis`, and `rice`.
Scott widths use the declared standard formula on each axis:

`h = 3.5 * population_standard_deviation * n^(-1/3)`.

The report preserves rule widths, unclamped and selected bin counts, grid gates,
raw counts, probabilities, and frame identities. For unweighted unbiased MD,
relative free energy is `-k_B T ln(surface_density / maximum_surface_density)`.
AI ensembles remain nonthermodynamic occupancy landscapes. Biased and enhanced
sampling fail closed until a validated weighting model is available.

`smoothing_sigmas_bins` declares every Gaussian smoothing level to evaluate and
`primary_smoothing_sigma_bins` selects the primary report alias. Each level has
its own minima, deterministic steepest-density catchments, frame assignments,
replica populations, and segment-block populations. Cross-tabulations and the
adjusted Rand index compare each alternate catchment partition with the primary
one. Raw histogram counts are identical at every smoothing level: smoothing
changes only the surface used to locate minima and route catchments.

Every smoothing level also reports an exact or seeded estimated silhouette for
its assigned basin labels when at least two basins are present. This is a
secondary geometric-separation diagnostic in the selected PCA plane. It never
defines a basin, chooses a smoothing level, proves metastability, or replaces
the population and smoothing-sensitivity reports. Per-system surfaces carry
the same diagnostic on their locally normalized, shared-grid basin labels.
The default FES silhouette budget evaluates 1,000 seeded focal frames against
the full assigned partition when more than 1,000 observations are present;
the evaluated indices and seed are retained. This keeps all 30,000 or more
frames in the basin definition and populations while bounding the secondary
`O(B*N)` separation diagnostic.

The pooled surface also fixes a common grid for independently normalized
per-system conditional surfaces on the same PCA basis. Grid coordinates can be
compared directly; absolute free-energy/occupancy offsets and local basin IDs
cannot. Pooled basin assignments include system/replica population tables and
pairwise descriptive system differences.

`scalar_feature_distributions` applies the same explicit, Scott,
Freedman-Diaconis, or Rice choices to any declared scalar output from
`trajectory_features`. It retains rule widths, raw and clamped bin counts, raw
frame assignments, and histogram counts. Contiguous bin residence runs never
cross a declared segment; runs touching either segment boundary are labeled
censored and excluded from complete-run summaries.

## Clustering selection

The generic workflow clusters tICA coordinates by default so the candidate
partitions emphasize slow continuous modes rather than maximum Cartesian
variance alone. PCA coordinates remain available as a configured geometric
sensitivity analysis. tICA itself does not require clusters or discrete states;
it is fitted before clustering from segment-safe time-lagged feature pairs.

The default-off `random_feature_koopman` module is a separate nonlinear
sensitivity over selected TICA coordinates. It approximates a declared
Gaussian kernel with random Fourier dictionaries, repeats the complete grid
over several prespecified feature-map seeds, and withholds a model unless both
contiguous-block held-out VAMP-E and recovered slow-subspace stability gates
pass. It does not replace the linear TICA coordinates used by clustering, does
not select a discrete partition, and does not bypass the MSM validation below.

KMeans scans declared `k_values`. New quick starts use the dependency-free,
deterministic Stratified NANI initializers `nani_strat_all` and
`nani_strat_reduced`; legacy manifests without `initialization_methods` retain
the seeded KMeans++ behavior. NANI ranks frames by the complementary MSD of the
remaining ensemble and selects initial centers by stable rank stratification.
`strat_all` covers the full complementary-MSD ordering, while
`strat_reduced` first restricts selection to the configured high-density
fraction. Neither NANI path uses a clustering random seed. The report preserves
the initial frame indices, complementary-MSD values, effective candidate count,
duplicate skips, and adjusted-Rand agreement between enabled initialization
strategies. `nani_percentage` must supply at least `k` candidate frames for
every requested `k`; the analysis fails closed rather than silently increasing
the percentage. When exact silhouette is too large, the separately recorded
`silhouette_random_seeds` generate several reproducible focal-frame samples.
The report preserves every sample, summarizes the score distribution, and
requires unanimous agreement on the winning `k`; disagreement fails closed.
Sampling never changes any already-fitted candidate partition, although it can
affect which candidate would be selected.

iMWK-Means scans `k`,
Minkowski `p`, and initialization ranks. HDBSCAN scans minimum cluster size and
minimum samples while retaining noise coverage. The alternative family can scan
the relevant grids for PAM, weighted PAM, Ward, Gaussian and variational
mixtures, affinity propagation, mean shift, and quality-threshold clustering.
PaLD is a separate, optional community analysis rather than a twelfth
partitioning competitor. When explicitly enabled, it uses a bounded regular
sample to report the cohesion matrix, local depth, mutual-cohesion strong-tie
network, connected communities, community cores, boundary observations, and
strongest intercommunity ties. A separately labeled PaLD-community MSM may be
enabled on those sampled labels; it never becomes the selected conventional
clustering MSM state definition.

Every selected partition with at least two retained clusters reports silhouette,
Calinski-Harabasz, and Davies-Bouldin evidence where applicable. Silhouette is
exact when the observation count is within the declared limit. Above that limit,
the KMeans suite evaluates several prespecified seeded subsets of focal
observations against all members of the complete fitted partition and labels
the result as estimated; every seed and evaluated index is retained. The
winning `k` must be identical across samples. Selection maximizes silhouette
only among valid algorithm-specific candidates and never treats geometric
separation as evidence of metastability or kinetics.

All clustering runners can instead use exact columns from
`trajectory_features`, including ion-site or ion-pair distances. The manifest
records feature IDs and one-based value indices; assignments retain the complete
system/replica/segment/frame identity and report system-by-state populations.

Strong association between cluster labels and system, replica, starting
conformer, or preparation lineage is itself a scientific property of the
fitted partition. It must be shown with the population evidence and carried
into representatives and state trajectories. Without independent sampling and
kinetic evidence, the partition describes ensemble or preparation separation
rather than metastable states.

## Markov-model selection and validation

Every enabled clustering method is reported under its own name. Full-coverage
partitions are eligible for primary segment-safe transition-model selection.
HDBSCAN may also produce a separately labeled dense-core MSM sensitivity: every
noise observation ends a kinetic segment, so transitions are never inferred
across an unassigned interval. Its stationary populations, residence behavior,
and timescales are conditional on the retained core observations and cannot be
interpreted as full-trajectory estimates. Ward and quality-threshold are skipped
unless the exact fit includes every observation. Quality-threshold is also
skipped unless every observation falls within a selected cluster cutoff. A full
exact Ward or fully assigned quality-threshold partition is a normal primary
candidate; no approximate all-frame extension or sampled MSM is produced for
these methods. Cohesion-only PaLD output stays
in its separate community module. The best primary clustering MSM is selected
by kinetic evidence in this order: all declared validation gates, mean held-out
VAMP-E, lower Chapman--Kolmogorov error, lower implied-timescale variation,
geometric score, and fewer states.

The primary PCA-FES basin partition is always modeled and reported separately.
It is a sensitivity state definition, not a competitor that can erase the best
clustering result. Each family reports transition counts and connectivity,
reversible and nonreversible estimators, implied timescales across lag values,
Chapman--Kolmogorov tests, contiguous time-block VAMP validation, and optional
time-block bootstrap intervals. No transition crosses a declared trajectory
segment or bootstrap-block boundary. Passing numerical gates does not by
itself establish adequate sampling, metastability, kinetics, or mechanism.

## Coordinate export contract

`state_coordinate_exports` converts a declared FES smoothing partition or
selected clustering partition into coordinate files:

- one multi-frame PDB or XYZ trajectory for each state/system/replica group;
- one or more observed representative PDB structures nearest the fitted center,
  medoid, or FES minimum;
- a checksummed export manifest containing every source frame identity and input
  lineage.

The default `representative_frames` result is an observed frame nearest the
declared state center in the state-defining feature space. For K-means that is
the point nearest the fitted centroid; for an FES basin it is the point nearest
the density-basin root. This is not generally the full-coordinate RMSD medoid,
which minimizes average pairwise RMSD to all state members. The separate
`representative_structures` function can calculate that coordinate-space
medoid for an explicitly supplied, prealigned ensemble, but it is quadratic in
the number of frames and is not a separate default campaign analysis.

An optional `coordinate_selection` names a project selection to materialize.
Routine conformational-view workflows use the full solute-heavy selection, so
FES trajectories retain the protein–nucleic-acid context while omitting bulk
water and free ions. If the field is absent, the backwards-compatible behavior
exports every topology atom. The selected atom identities and topology hash are
retained in the export evidence.

The exporter streams source trajectories read-only, applies the project's
declared connectivity-aware periodic reconstruction, and never combines unlike
system/replica topologies. It writes to
`analysis_output_root/08_clustering/state_coordinate_exports/<export_id>` using a
temporary sibling directory followed by an atomic rename. `existing_output_policy`
must be `fail`; an existing export ID is never overwritten or merged. Publication
repositories should lock selected export manifests and parameters rather than
forking the reusable implementation.

State assignments, silhouettes, smoothing stability, and representatives remain
descriptive until sampling, convergence, and kinetics are independently valid.
