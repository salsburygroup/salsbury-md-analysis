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
  axes, and a relative-free-energy legend. CSV tables retain the full grids at
  every saved smoothing level, plus smoothing sensitivity and basin populations.
- FES basins and clustering states include per-system population tables and
  stacked bar charts. Clustering model tables identify the method, feature
  source, state count, and silhouette value.
- DCCM reports include system matrices and pairwise difference matrices with
  the mapped atom labels retained in full-matrix CSV tables. A separate table
  lists the 50 largest off-diagonal differences.
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

## Replotting FES and DCCM results

The following files are ordinary UTF-8, comma-separated text. Import them into
Origin, Excel, R, Python, or another plotting program; no JSON reader or
interactive viewer is required.

| Result | File under `presentation-artifacts/` | Contents |
| --- | --- | --- |
| Primary pooled FES | `pca-fes-basins/<view>/primary-fes.csv` | Every grid cell at the primary smoothing level |
| Other pooled FES smoothing levels | `pca-fes-basins/<view>/grids/pooled-smoothing-<sigma>.csv` | Every grid cell at each saved alternative level |
| Per-system FES on the shared basis | `pca-fes-basins/<view>/grids/<system-token>-smoothing-<sigma>.csv` | Each system's separately normalized grid at every saved level |
| Individual-system DCCM | `dccm/<system>.csv` | All N × N entries, including the diagonal and both triangles |
| Full DCCM difference | `dccm/comparisons/<left>-minus-<right>-matrix.csv` | Both source correlations and left minus right for every matrix entry |
| Largest DCCM differences | `dccm/comparisons/<left>-minus-<right>.csv` | The existing top-50 off-diagonal summary, not the full matrix |

Paths for separately run per-system conformational views include an additional
`per-system/<system-token>/` directory. Tokens include a short hash where needed
to distinguish system names that would otherwise produce the same filename.
The presentation manifest records the exact path and original system identifier.

FES tables identify the system, PCA components, smoothing level, normalization
scope, and temperature when those are recorded in the source report. Coordinates,
bin widths, and grid bounds are in ångströms. Energy is in kcal/mol and probability
density is in Å⁻². Each row retains the raw count and probability, smoothed count
and probability, basin ID, and reported energy or occupancy score. Use the
zero-based `x_bin` and `y_bin` columns, or the labeled bin centers, to reconstruct
the grid; do not assume it is square.

For nonthermodynamic results, the occupancy-score column is populated instead
of inventing an energy. Missing or nonfinite energy, score, or correlation values
are blank, with status columns distinguishing missing, NaN, positive infinity,
negative infinity, and values not reported. A measured zero remains zero.
Missing legacy metadata is left blank. The exporter does not interpolate,
renormalize, resmooth, round, or refit the results.

Per-system FES grids use their own normalization and energy zero. Their common
PCA basis and grid support distribution comparisons, not comparisons of absolute
free energies. Basin IDs from independently constructed surfaces need not refer
to the same conformation.

DCCM tables are in long form: `atom_i` is the zero-based matrix row and
`atom_j` is the zero-based column. Correlations and their differences are
dimensionless. Atom labels include chain, residue, insertion code, and atom
name when supplied; older reports without identities retain matrix indices.
Reconstruct the square array from these indices to make a heat map. Full
matrices can create large CSV files; rows are written incrementally without
building a second tabular copy in memory.

To add these tables to an older completed analysis, rebuild only its
presentation files in a new directory:

```bash
salsbury-md-analysis build-presentation-artifacts /path/to/completed-analysis \
  --output /path/to/new-presentation-files
```

This reads the saved reports without rerunning trajectories or changing accepted
results. It exports only grids and matrices present in those reports. Each new
CSV is registered with its content hash and source-report hash.

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
