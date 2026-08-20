# Legacy review closure

The checksum-collapsed non-docking legacy catalog is technically destination-closed as of 2026-08-11.

- **434** MD/core/reporting capabilities received a final repository destination.
- **280** have a reusable public-suite successor.
- **154** are non-estimator scaffolds, utilities, project/publication locks, test evidence, or historical artifacts and are not part of the public estimator API.
- **62** docking/screening capabilities were excluded from this decision and routed to the separate `salsbury-docking` review.
- **0** MD destination decisions remain unresolved.

The machine-readable public record is [`legacy_review_summary.json`](../legacy_review_summary.json). The detailed private ledger retains source checksums, row-level decisions, evidence scope, recommended owners, and future release gates; legacy source itself is not copied into this repository.

## What closure means

Closure approves where each reviewed capability belongs. It does not mean every old script was reproduced byte-for-byte, nor does it promote any module from `experimental` to `supported`.

The reusable successors have two evidence levels:

- **84** rows belong to method families with a locked direct numerical or formula validation. This is family-scoped evidence, not universal equivalence across every old parameterization.
- **196** rows have a reviewed suite contract and unit-tested successor without a claim that the old code was run numerically row by row.

Four apparent analysis candidates were confirmed to be abstract or writer scaffolds and removed from the public-estimator count. Three previously metadata-limited rows were resolved by source-body inspection as a historical text utility or project/publication locks.

## Reusable methods added during final review

- Cross-group minimum, mean, and maximum distances; minimum-mean group distance; and COM/COG distance variants.
- Independent-axis Scott, Freedman-Diaconis, and Rice bin selection for PCA occupancy/FES histograms.
- Segment-safe source-at-time-*t* to target-at-time-*t*+lag cross-correlation.
- Optional HDBSCAN clustering of signed correlation profiles or absolute-correlation similarity distances.
- Explicit overlap-normalized autocorrelation sequences for convergence diagnostics.

These are general methods with explicit contracts and tests. Project paths, plotting wrappers, output writers, and publication-specific drivers remain outside the public core.

## What remains

There is no remaining legacy destination review within the cataloged MD scope. Release qualification remains separate: each experimental module still needs independent scientific regression, documentation review, and explicit promotion before it is called supported. Newly discovered legacy code or genuinely new analyses enter through the normal extension process rather than silently changing this frozen review.
