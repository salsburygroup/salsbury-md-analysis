# Time-lagged independent component analysis

Module ID: `time_lagged_independent_component_analysis`  
CLI: `salsbury-md-analysis tica PROJECT.json`

The TICA implementation consumes declared components from the suite's shared
common-PCA feature basis. It constructs lag pairs separately within every
trajectory segment; it never joins the last frame of one segment to the first
frame of another.

The estimator uses the average covariance of the two lag-pair endpoints and a
symmetrized lagged covariance. The generalized eigenproblem is solved by
covariance whitening with an explicit eigenvalue cutoff and relative diagonal
regularization. Returned components include generalized-eigen residuals,
loadings, projections, eigenvalues, and implied-timescale diagnostics.

Current kinetic execution requires:

- `sampling_mode=UNBIASED_MD`;
- physical rather than sample-index timing;
- one common evaluated frame interval across segments;
- a declared positive lag and minimum pairs per segment;
- explicitly selected common-PCA components.

TICA output remains experimental until lag sensitivity, feature sensitivity,
stationarity, convergence, and downstream state-model validation pass on a
locked scientific dataset.
