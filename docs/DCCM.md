# Dynamic cross-correlation matrices

Status: **experimental**

This stage-5 module calculates Cartesian displacement cross-correlation on a
globally common, fitted atom basis. It reports per-replica matrices,
frame-pooled system matrices, and differences from the declared reference
system.

```bash
PYTHONPATH=src python3 -m salsbury_md_analysis \
  dccm path/to/project.json --hash-content
```

The command is read-only and emits JSON to standard output.

## Definition

For atoms `i` and `j`, the matrix entry is

`<delta_r_i dot delta_r_j> / sqrt(<|delta_r_i|^2><|delta_r_j|^2>)`,

where displacement is measured from each atom's mean fitted position. Values
are bounded to `[-1, 1]` after numerical roundoff correction.

This is Cartesian dot-product DCCM. It is not generalized correlation, mutual
information, causality, directionality, or a significance test.

## Required project contract

```json
{
  "reference_system": "control",
  "periodic_coordinate_policy": "reject",
  "definitions": {
    "dccm": {
      "alignment_selection": "alignment",
      "analysis_selection": "analysis",
      "minimum_reference_coverage": 1.0,
      "frame_stride": 1,
      "maximum_atoms": 500,
      "minimum_evaluated_frames_per_replica": 100,
      "minimum_variance_angstrom2": 1e-12
    }
  }
}
```

`maximum_atoms` is mandatory because calculation time and matrix memory scale
quadratically. The gate is checked before trajectory analysis.

Per-frame dot-product covariance and replica merging use vectorized online
moments, so coordinate frames are not retained. The atom-by-atom output matrix
and working covariance remain quadratic in the declared analysis selection;
frame coverage can therefore be all-frame for a bounded trace selection even
when a much larger atom selection must use a sensitivity-tested frame budget.

`minimum_variance_angstrom2` prevents division by negligible positional
variance. If either atom in a pair falls below the gate, that entry is JSON
`null`. It is not reported as zero correlation. Undefined-entry counts and
valid off-diagonal summaries accompany each matrix.

## Pooling and differences

Each replica retains its own DCCM. The system matrix merges streaming moments
across replicas and is frame weighted; it is not an uncertainty estimate.
Reference-system differences are simple entrywise subtraction and remain
`null` wherever either contributing matrix entry is undefined.

## Scientific limits

- Correlation does not establish causality, directionality, mechanism, or
  functional importance.
- Pooled frames are not independent uncertainty units.
- Results can be sensitive to alignment, atom selection, sampling balance,
  state mixtures, and trajectory length.
- Periodic coordinates fail closed under `reject`; `allow_wrapped_diagnostic`
  remains diagnostic. Production runs use connectivity-aware `make_whole` or
  `unwrap_continuous` preprocessing.
- Each segment reports its declared timing and evaluated physical-time range.
- No block-sensitivity or statistical-significance calculation is implemented
  in this experimental slice.
- The current real-project candidate covers structural QC, not DCCM, and is
  not approved.
- A technically complete matrix does not establish equilibration, convergence,
  adequate sampling, or scientific validity.
