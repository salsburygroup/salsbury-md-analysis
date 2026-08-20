# Representative frames

`representative_frames` consolidates representative-frame selection for fitted
KMeans clusters and PCA-FES basins. It selects observed frames only; it does not
construct arithmetic average coordinates or write structures, trajectories,
figures, or publication-specific files. Generic coordinate materialization is
provided separately by `state_coordinate_exports`.

## Contract

The project definition declares:

- `source`: `clustering_kmeans` or `pca_fes_basins`;
- `representatives_per_state`: number of observed frames retained per state;
- `maximum_states`: fail-closed state-count resource gate;
- `maximum_candidates`: fail-closed observation-count resource gate.

For KMeans, candidates are ranked by squared distance in the exact declared
clustering feature space. For PCA basins, candidates are ranked by squared
distance in the declared two-component PCA plane to the deterministic basin
root. Ties resolve by system, replica, segment, and source-frame identity.

The report contains immutable frame locators, source-module lineage, distance,
and representative rank. `fes_smoothing_sigma_bins` can select one declared FES
smoothing partition rather than the primary alias. The separately tested
`state_coordinate_exports` adapter writes new, collision-protected coordinate
artifacts; a paper's chosen figures, labels, and layouts belong in that paper's
publication repository.

## Scientific boundary

A nearest-center observed frame is a descriptive representative of a fitted
state. It is not an average structure and does not prove that the state is
metastable, converged, mechanistically meaningful, or sufficiently sampled.
Every representative inherits the source module's feature, preprocessing,
scaling, clustering, binning, and basin-definition choices.
