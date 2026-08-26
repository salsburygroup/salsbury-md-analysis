# Interactive result browser

Version 0.2 adds an offline, interactive front end to the normal finalization
stage. It is a presentation layer over the existing JSON, CSV, structures, and
figures. Those source artifacts remain the scientific record.

## What is generated

When `reporting.interactive_report_enabled` is `true` (the default), a completed
local or Slurm campaign writes:

- `interactive-report/index.html`, the self-contained result browser;
- `interactive-report/manifest.json`, which records the HTML hash, source-report
  hashes, included assets, and the technical/scientific status boundary; and
- `final-interactive-report-summary.json`, the finalizer's completion record.

Open `interactive-report/index.html` in a current browser. It does not need a
web server or internet connection, and it does not send structures or results
to an external service.

The browser presents:

1. QC errors and warnings plus the highest-ranked findings;
2. every ranked finding with system, module, evidence level, effect, and raw
   report link;
3. FES surfaces at every retained smoothing level and per-system surface;
4. clustering populations and silhouette evidence;
5. RMSF and affordable DCCM views plus pre-rendered figures;
6. availability/metric panels for every default-off experimental module,
   aligned hydration/ion density projections, pocket-persistence bars, and
   interaction-fingerprint occupancy bars when those outputs are available;
7. interactive representative PDB structures;
8. every module report, including modules that produced no ranked finding;
9. measured CPU, memory, frame selection, and observation accounting; and
10. resolved configuration, chemical context, conformational views, QC, and
   provenance.

## Molecular viewer boundary

The built-in viewer is deliberately dependency-free. It reads PDB atom records,
supports rotation and zoom, filters atoms, colors by element, chain, or B-factor,
and highlights atom/residue text matches. Its CA/P trace lines are visual guides,
not inferred chemical bonds. Download the linked PDB and use VMD, ChimeraX, or
another full molecular package when bond topology, surfaces, measurements, or
publication rendering are needed.

Representative structures are included in the HTML only up to explicit bounded
asset limits. Omitted structures and figures remain linked and are listed in the
provenance panel. Multi-frame state trajectories are never embedded in the HTML.

## Large reports

The finalizer never treats browser convenience as permission for unbounded
memory use. A JSON report larger than 128 MB is hash-indexed and represented by
its compact summary sidecar rather than loaded into the browser. The raw report
remains linked. This affects only the interactive preview; it does not remove,
change, or reclassify the analysis result.

## Configuration

To turn the front end off while retaining the normal JSON/CSV/Markdown reports:

```json
{
  "config_schema": "salsbury-analysis-config-v1",
  "reporting": {
    "interactive_report_enabled": false
  }
}
```

To build it later from a completed campaign:

```bash
salsbury-md-analysis build-interactive-report path/to/analysis-root
```

Generation is immutable. If the output directory already exists, its manifest
and HTML hash must validate before it is reused. A partial or changed directory
fails closed instead of being overwritten.

## Interpretation

The result browser does not promote technical completion into scientific
validity. Automated findings without a supported adjusted p-value remain
descriptive. FES basins, smoothing, clustering, silhouettes, state populations,
representatives, correlations, and ion geometry still require review of
sampling, convergence, chemistry, uncertainty, and the underlying method report.
