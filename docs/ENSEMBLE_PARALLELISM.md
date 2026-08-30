# Replica parallelism and pooled estimators

Independent simulation replicas can be read concurrently. They are not, by
default, separate definitions of the mean structure, covariance basis, state
space, or comparison model.

The package records an `ensemble_parallelism_contract` for planned and
scheduled tasks. The contract states what a replica worker may return and the
scope at which the primary result must be finalized. A scheduler request that
marks a pooled estimator as an independently finalized replica partition fails
before execution.

## Global coordinate estimators

Common PCA, pooled RMSF, and DCCM use a common atom map and one declared
alignment convention.

- A common-PCA replica worker may produce a count, mean, and centered Cartesian
  second moment, or identity-preserving basis samples. The states are merged
  before calculating the global mean, covariance matrix, and PCA basis.
- A pooled-RMSF worker may produce coordinate count, mean, and centered second
  moments. Frame-pooled RMSF is calculated from the merged system state, not by
  averaging independently centered replica RMSFs.
- A DCCM worker may produce displacement count, mean, and cross-second moments.
  The system-pooled DCCM is calculated after those states are merged. A local
  replica DCCM may still be reported as a labeled diagnostic.

The streaming merge formulas include the between-replica mean correction. They
are numerically equivalent to processing the selected aligned frames in one
serial stream, apart from floating-point summation order.

## Pooled feature and state models

Replica workers may extract common-PCA or tICA features while retaining system,
replica, member, segment, frame, and physical-time identity. Centering,
standardization, FES grid construction, conventional clustering, PaLD, and
state selection are then performed on the complete declared view or on one
explicit replica-balanced sample from that view. The software does not fit an
independent clustering model to each replica and attempt to reconcile labels
afterward. Frames from different replicas can therefore receive the same state
label under one fitted model.

Common PCA supports two explicit weighting contracts. `frame` gives every
selected frame equal weight. `replica_equal` gives every replica distribution
equal total weight even when selected frame counts differ. Both produce one
shared mean and basis.

## Ordered estimators

TICA, information dynamics, residence analysis, and Markov models preserve
replica, member, and segment boundaries. Lag pairs, transition counts, and
residence runs are formed within those boundaries. Their sufficient statistics
or count matrices may then be pooled for one declared-view model. No transition
is created between the end of one replica and the beginning of another.

Continuous unwrapping has the opposite boundary: it must scan frames in order
within one continuous replica history, but different replicas can be unwrapped
concurrently. Unwrapping state is never shared between replicas.

## Replica-level diagnostics

Structural QC, RMSD/Rg, individual PCA, and convergence diagnostics may produce
complete replica-local results. Their campaign reports preserve replica
identity. Individual-PCA coordinates from separately fitted bases are not
treated as a common feature space.

This execution contract controls numerical meaning. It does not make trajectory
frames independent samples, establish convergence, or replace a declared
replica-level uncertainty model.
