# Figures, tables, and structures

Every technically complete analysis report produces at least one labeled figure.
The workflow also writes a CSV table when the report contains tabular numerical
results. These files are derived views of the scientific JSON reports, not
replacements for them.

The final reporting stage writes the files under `presentation-artifacts/` and
records them in `presentation-artifacts/presentation-manifest.json`. Each
manifest entry has a stable artifact identifier, analysis class, purpose,
human-readable title, relative path, source-report path and hash, and enough
context to identify the system, conformational view, state, or comparison it
shows. The finalizer stops if a completed report has no presentation adapter or
if a file fails its recorded size or hash check.

Run the presentation stage directly on a completed analysis root with:

```bash
salsbury-md-analysis build-presentation-artifacts /path/to/analysis-root
```

The command refuses to overwrite an existing `presentation-artifacts/`
directory. Supply `--output /new/versioned/path` when rebuilding presentation
files from immutable results.

## What the primary figures show

- Free-energy surfaces use the configured primary smoothing level, labeled PCA
  axes, and a relative-free-energy legend. Separate tables retain smoothing
  sensitivity and basin definitions.
- FES basins and clustering states include per-system population tables and
  stacked bar charts. Clustering model tables identify the method, feature
  source, state count, and silhouette value.
- DCCM reports include system matrices and pairwise difference matrices with
  the mapped atom labels retained in CSV tables.
- RMSD remains a replica-resolved time series. Radius of gyration is presented
  first as a Scott-rule histogram, with its time series kept as a secondary
  view.
- RMSF reports include profiles and pairwise differences. RMSF-colored
  structures remain separate coordinate artifacts.
- Hydrogen bonds, hydration networks, ion analyses, SASA, secondary structure,
  internal coordinates, RDFs, information measures, networks, kinetic models,
  and other completed modules each receive a method-appropriate figure and a
  table when their report exposes tabular values.
- A module without a specialized adapter receives a labeled numerical summary
  and table. If no truthful numerical presentation can be made, the reporting
  stage fails instead of inventing a plot.

## Findings and exact links

Finding records use `salsbury-finding-target-v1` to name the intended analysis,
purpose, and context. After the presentation manifest is complete, the finding
picker resolves each target to exact figure, table, or structure records. A
finding cannot silently point at an unrelated panel. Supporting records such as
PCA basis summaries and failed kinetic validations remain searchable but do not
displace scientific findings in the opening section.

`state_coordinate_exports` can calculate ion stability within each aligned
state. Ion sites are matched without relying on atom identity, then filtered by
state occupancy, positional RMSF, and minimum frame count. Representative
structures contain the complete non-solvent molecular payload and only ions
assigned to stable state sites. The report records the thresholds and every
retained or excluded site.

The optional `salsbury-md-analysis-interactive` package reads this manifest and
copies the referenced files into its portable report. The core package remains
usable without the interactive viewer.
