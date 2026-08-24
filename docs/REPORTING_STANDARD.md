# Scientific reporting standard

The machine-readable policy is
[`reporting/reporting_standard_v1.json`](../reporting/reporting_standard_v1.json).
It is a presentation contract, not a claim that a displayed difference is
statistically, mechanistically, or biologically important.

## Scalar time-series rule

Every finite scalar time-series result except RMSD is presented first as a
histogram using Scott's rule. The corresponding time series is retained as a
secondary presentation. RMSD is the explicit exception: it remains a
replica-resolved time series first because drift, discontinuities, and
replica-specific behavior are central to its interpretation.

The generic JSON-in/JSON-out adapter applies this rule to any frame-indexed
scalar records:

```bash
PYTHONPATH=src python -m salsbury_md_analysis \
  summarize-timeseries timeseries-request.json > presentation.json
```

Scott bin widths are computed from pooled finite observations across the
declared systems and replicas, with declared minimum and maximum bin-count
gates. Source-frame and segment identities remain attached. A constant field
is reported as `not_estimable`; the suite does not invent a histogram width.
Histogramming does not make correlated trajectory frames independent.

## Result order

1. Free-energy surfaces for unbiased equilibrium MD, or explicitly labeled
   occupancy landscapes when thermodynamic FES is not justified. Matched
   systems use one shared PCA basis and grid.
2. Observed structures and state trajectories from the top-populated FES
   basins, plus basins that are distinctive between systems.
3. Clustering selected from declared parameter sweeps using exact or seeded
   estimated silhouette evidence, with populations by system, replica, and
   state. The best kinetically evaluated clustering model and the FES-basin
   transition model are both retained; one never replaces the other.
4. Observed structures and state trajectories from the top-populated or
   system-distinctive clusters.
5. RMSF profiles, a VMD `NewCartoon` colored by RMSF, and a PDB whose B-factor
   field contains RMSF in angstrom.
6. Other results with clear physical significance between systems, or
   physically interesting results for a single system, with their uncertainty
   and interpretation limits.

The originating priority list repeated FES conformations at ranks two and
four. To keep the production suite comprehensive but nonredundant, the profile
stores FES conformations once and uses rank four for the analogous clustering
representatives.

## Selection cautions

FES basin silhouette can be calculated, but it is a secondary diagnostic of
separation in the chosen PCA plane. Basins remain defined by free-energy or
occupancy minima and their deterministic catchments. The score cannot select
the smoothing level, establish thermodynamic states, or replace smoothing and
population sensitivity.

Clustering silhouette is used to select among valid declared candidates.
Above the exact limit, a seeded subset of focal observations is compared with
all members of the fitted partition and the result is labeled estimated.
Neither form establishes metastability or kinetics.

For Markov analysis, geometric silhouette is only a clustering diagnostic.
Full-coverage clustering partitions are ranked using kinetic gates and
time-blocked VAMP-E, Chapman--Kolmogorov, and implied-timescale diagnostics.
The primary FES basin partition is evaluated and reported separately because a
free-energy catchment is not automatically a Markov state.

A clustering result is not discarded because its labels are strongly
associated with replica, system, starting conformer, or preparation lineage.
The suite retains the partition, population tables, representatives, and state
trajectories and reports that association as part of the scientific result.
Without independent kinetic and sampling evidence, the labels describe
ensemble/preparation separation rather than validated metastable or kinetic
states.
