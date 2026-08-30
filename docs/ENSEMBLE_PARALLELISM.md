# Replica parallelism and pooled estimators

Independent simulation replicas can be read concurrently. They are not, by
default, separate definitions of the mean structure, covariance basis, state
space, or comparison model.

The package records an `ensemble_parallelism_contract` for planned and
scheduled tasks. The contract states what a replica worker may return and the
scope at which the primary result must be finalized. A scheduler request that
marks a pooled estimator as an independently finalized replica partition fails
before execution.

The shared replica-worker/reducer layer validates unique system/replica
identities, complete continuous-segment identities, contiguous shard ordinals,
the scheduler CPU limit, and stable reduction order. Critical base modules use
this layer for SASA, water-mediated networks, hydrogen-bond discovery, DCCM,
secondary structure, RMSF, RMSD/Rg, structural QC, and ion analyses. The
reducer is method-specific: some concatenate identity-preserving observations,
while RMSF and DCCM merge exact streaming moments with the between-replica mean
correction.

The experimental branch uses the same executor for DSSR frame geometry and
multivalent bridge extraction. Multivalent occupancies are reduced after
replica extraction, bridge residence runs remain inside their original
segments, feature indices are harmonized globally, and the retained detailed
hyperedges are selected from the pooled records. Experimental allosteric-contact
and recurrent-pocket inputs use the validated molecular-payload cache when
their atom and parameter contracts permit it. Hydration-density and recurrent
pocket regions still finalize from pooled voxel counts; independently finalized
replica regions are prohibited because their labels would not define one shared
spatial model.

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

K-means-like and probabilistic models may fit once on the declared pooled
training observations and then assign identity-preserving replica chunks with
that single fitted model. HDBSCAN labels the complete pooled set it fits,
including explicit noise. Ward and quality-threshold methods are skipped when
their fit budget cannot cover every observation they claim to partition;
neither receives invented out-of-sample labels. PaLD remains a separate pooled
sample community analysis rather than a substitute trajectory partition.

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
