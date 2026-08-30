# Principal-component analysis

The experimental PCA layer provides two deliberately different analyses:

- `individual-pca` fits one PCA basis independently for each replica. It is a
  within-replica visualization and fluctuation diagnostic. Its component scores
  must not be compared numerically between replicas because every replica has a
  different mean and basis.
- `common-pca` maps one atom set across every declared topology, fits a single
  shared basis, and projects every evaluated frame onto that basis. Its scores
  can be placed on common axes, subject to the sampling and mapping limitations
  below.

Both commands separate basis fitting from projection and write no project or
trajectory files. The basis pass fits a covariance model or leading-component
subspace. The projection pass evaluates the separately declared projection
frames. File size and modification time are checked within and between passes;
a change fails the run.

## Required configuration

Individual PCA uses `definitions.individual_pca`:

```json
{
  "alignment_selection": "alignment",
  "analysis_selection": "analysis",
  "minimum_reference_coverage": 1.0,
  "frame_stride": 1,
  "frame_selection": {
    "mode": "integer_stride_per_replica_v1",
    "stride": 5
  },
  "projection_frame_stride": 1,
  "projection_frame_selection": {"mode": "fixed_stride_v1"},
  "maximum_features": 900,
  "component_count": 10,
  "minimum_evaluated_frames_per_replica": 100
}
```

`frame_stride` and `frame_selection` define the covariance/basis-fit sample.
`projection_frame_stride` and `projection_frame_selection` independently define
which frames are projected after the basis is fixed. Thus a large trajectory
can fit a balanced, sensitivity-tested basis on a bounded sample while still
projecting every source frame for FES assignment, clustering, state exports,
and representative-structure extraction. If the projection fields are omitted,
they default to the basis-fit contract for backward compatibility. Both
contracts and both evaluated counts are emitted in the report.

Common PCA uses the same fields under `definitions.common_pca` and also
requires:

```json
{
  "basis_weighting": "replica_equal"
}
```

`basis_weighting` has two exact meanings:

- `frame`: each evaluated frame has equal weight. Longer replicas influence the
  basis more strongly.
- `replica_equal`: each replica distribution has equal total weight regardless
  of frame count. Within each replica, its evaluated frames remain equally
  weighted.

The report records each replica's numerical basis contribution. Neither choice
is universally correct. Comparative work should run both when unequal sampling
could change the result.

Routine project preparation selects the PCA resource path from both the number
of Cartesian features and the number of available frames. Small views use the
exact dense covariance solver. Large common-heavy or interface views use the
deterministic leading-component solver:

```json
{
  "method": "randomized_truncated_svd_v1",
  "oversampling": 12,
  "power_iterations": 4,
  "power_iteration_schedule": [4, 8, 12],
  "random_seed": 20260812,
  "maximum_sample_matrix_elements": 25000000,
  "maximum_relative_residual": 0.001
}
```

This solver retains all atoms in the declared view but computes only the
leading components required downstream. The basis-fit frames are selected by a
deterministic full-timespan, per-replica plan; projection normally retains all
source frames. The report records the sample-matrix dimensions, solver seed,
power iterations, every bounded refinement attempt, orthonormality error, exact
covariance-action residuals, and the independent basis/projection counts. The
solver starts at four power iterations and may reuse the same deterministic
range-finder state at eight and twelve iterations. It never changes the
residual tolerance or frame selection; failure after the last declared attempt
remains fail-closed.

When the complete planned basis contains no more than 128 frames, preparation
expands the declared oversampling so the range finder spans the full bounded
sample space. This avoids an avoidable residual failure for short,
high-dimensional trajectories without forming a feature-by-feature covariance
matrix. Larger bases keep the 12-vector oversampling shown above. The resource
plan records the applied subspace size and policy; this numerical choice does
not turn a short trajectory into adequate scientific sampling.

## Coordinate and numerical contract

Every evaluated frame is rigidly fitted to the explicit `reference_structure`
using `alignment_selection`. PCA features are the fitted Cartesian coordinates
of `analysis_selection`, in reference atom order. Common PCA intersects that
selection across all topologies under `common_atom_policy`; individual PCA maps
each topology separately.

Covariance is the population covariance of evaluated frames, with denominator
`N`, in square angstroms. Eigenvalues therefore sum to the total fitted
Cartesian variance apart from numerical tolerance. Scores have angstrom units.
Every component includes:

- eigenvalue and explained/cumulative variance fraction;
- atom-resolved x/y/z loadings;
- eigensolver call count, convergence flag, and directly evaluated residual norm;
- a deterministic sign convention: the first largest-absolute loading is
  positive.

The sign is reproducible but not physically meaningful. A component and its
negative describe the same axis. Near-degenerate components may rotate within a
nearly equal-eigenvalue subspace, so subspaces, rather than individual vectors,
require sensitivity comparison.

The exact implementation uses vectorized online population covariance and
NumPy's symmetric LAPACK eigensolver; its memory scales quadratically with
Cartesian feature count. The leading-component implementation works directly
with a bounded sample-by-feature matrix and avoids constructing the dense
feature covariance. `maximum_features` and the selected solver's memory gate
are mandatory. A component below the fixed eigenvalue gate is omitted and
reported as a numerical-rank warning. Every returned eigenpair is sign-oriented
deterministically and checked against a direct residual gate. Near-degenerate
subspaces still require subspace-level sensitivity rather than comparison of
individual eigenvectors. Solver constants are emitted in `solver_contract`.

## Frame axes, segments, and periodic coordinates

MD projection rows retain system, replica, segment, source-frame index,
normalized physical time, and time unit. `AI_ENSEMBLE` rows instead retain a
declared `sample_index`. They do not contain a fabricated time field. System
segments must declare exactly one of `timing` or `sample_axis`, and the project
sampling mode must agree with that declaration. The basis and projection
strides each apply independently to the zero-based source-frame index of every
segment.

`periodic_coordinate_policy: "reject"` blocks periodic input and
`allow_wrapped_diagnostic` remains diagnostic only. `make_whole` and
`unwrap_continuous` preprocess both PCA covariance and projection passes from
explicit connectivity. See
[Periodic coordinate reconstruction](PERIODIC_COORDINATES.md) for gates and
multicomponent limitations.

## Outputs and interpretation boundary

Individual PCA reports a mean, components, and complete projection time series
for each replica. Common PCA additionally reports the global common-atom mean,
basis weights, per-replica and frame-pooled system projection summaries, and
mean-score differences from the declared reference system.

These outputs are descriptive. Frame-pooled standard deviations are not
replica-level uncertainty. PCA does not identify thermodynamic basins, free
energies, kinetics, mechanisms, convergence, or adequate sampling. Those claims
require the later FES, clustering, state-model, and convergence stages plus an
appropriate sampling/weight model.

The 474-atom macromolecular-trace solver has completed retained real-trajectory
technical validation. The larger common-heavy and chemistry-defined interface
paths require separate retained validation before their results can be treated
as accepted scientific evidence. Both modules therefore remain `experimental`.
Private technical validation evidence belongs in the group's validation
records or a publication repository, not in this public toolkit.

The small, non-scientific teaching inputs under `examples/pca_fixture/` exercise
both commands and deliberately use unequal replica lengths so the global basis
weighting is visible. They are software fixtures, not molecular evidence.
