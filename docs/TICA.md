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

## Short continuous segments

The default `short_segment_policy` is `error`: every supplied segment must
contain at least `lag_frames + minimum_pairs_per_segment` projected observations.
For example, a lag of 10 and a minimum of 10 pairs require 20 observations.
To omit shorter streams from tICA only, set this preparation-config option:

```json
{
  "modules": {
    "time_lagged_independent_component_analysis": {
      "options": {"short_segment_policy": "omit"}
    }
  }
}
```

This is a fragment to add to your existing analysis config. In a project
manifest, the same field belongs under
`definitions.time_lagged_independent_component_analysis`.
The lag, pair minimum, covariance cutoff and PCA settings stay unchanged.
Filtering happens after the shared PCA calculation; it never removes frames
from that PCA basis or from independent analyses.

The report's `segment_eligibility` records each system, replica, segment and
oligomer-member stream, its observation and pair counts, and the reason for
each exclusion. Excluded source-frame identities are retained. Totals use all
supplied PCA observations as the denominator. `feature_lineage` binds the
unchanged PCA payload by SHA-256. Downstream tICA-based clustering receives only
eligible projections and carries the eligibility ledger in its feature contract.
Its populations therefore describe that eligible subset, not every input frame.

Omission does not repair broken time axes: nonfinite, repeated, irregular or
incompatible physical times still fail, including in short streams. If no
eligible stream remains, tICA fails without lowering its minimum. Unrelated
analysis work remains independent. Discontinuous frames must first be represented
as their real continuous segments; this option does not make a concatenated
static ensemble suitable for kinetic analysis.

TICA output remains experimental until lag sensitivity, feature sensitivity,
stationarity, convergence, and downstream state-model validation pass on a
locked scientific dataset.
