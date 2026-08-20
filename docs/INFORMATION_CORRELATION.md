# Generalized correlation and mutual information

Module ID: `generalized_correlation_and_information`  
CLI: `salsbury-md-analysis information-correlation PROJECT.json`

This module provides a nonlinear dependence analysis distinct from Cartesian
DCCM. It currently consumes explicitly selected scalar common-PCA projections
and reports empirical mutual information in nats, normalized mutual
information, and the scalar generalized-correlation transform.

The estimator uses deterministic quantile histogram bins. Constant-feature
normalized entries are `null`, not zero. Every report records requested and
actual bin counts, marginal entropies, feature lineage, and separate replica
and system-pooled matrices.

Histogram mutual information is sample- and bin-sensitive. Analyses must repeat
reasonable bin counts, retain replica-level results, and avoid treating frames
as independent uncertainty units. The matrices are symmetric and cannot
establish direction, causality, or mechanism. Residue-vector and explicit
distance-feature providers remain future extensions of the same estimator
contract rather than separate copies of the algorithm.
