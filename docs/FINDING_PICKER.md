# Prioritized findings and complete module accounting

The finding picker puts results that merit attention near the beginning of a
campaign report. It does not decide whether a scientific claim is valid.

Every technically complete module report receives one of five dispositions:

- `ranked_candidates`: the module produced one or more deterministic finding
  candidates;
- `quality_control`: the result belongs in the QC and interpretation channel;
- `interpretive_context`: the result supplies a basis, representation, or
  representative artifact used to interpret another analysis;
- `technical_support`: the result records provenance, mapping, caching, or
  execution state rather than a scientific observation; or
- `reviewed_no_automatic_highlight`: the scientific report was reviewed by the
  picker but did not meet a declared automatic highlight rule.

The JSON output records this in `module_accounting`. It also records
`reviewed_report_count`, `reviewed_module_count`, and `silent_omission_count`.
A successful picker run has zero silent omissions. Reports without highlights
remain linked and are not treated as empty, unimportant, or scientifically
negative results.

Comparative campaigns first write
`results/integrated-comparison/report.json`. That mandatory finalizer reviews
every completed report, preserves method-aware cross-system candidates, groups
them by system pair, and records modules that produced no automatic highlight.
The finding picker then takes cross-system candidates from this integrated
report instead of independently recreating them. Single-system findings remain
eligible, so a comparison campaign can still highlight a noteworthy result
that occurs in only one system.

The integration step compares matched scientific summaries produced by each
module. It does not subtract arbitrary arrays, treat differently defined state
labels as equivalent, or calculate a composite biological score. A module with
no standardized comparison remains visible in `module_comparison_coverage`;
that disposition does not imply that the systems are equivalent.

## Ranking and statistical language

Candidates are ordered by the documented presentation categories, followed by
inferential status, adjusted p-value, absolute effect size, module, and
statement. The picker does not use an opaque composite score.

When supported p-values are available, Benjamini-Hochberg correction is applied
within the declared comparison family. Only a candidate with an adjusted
p-value at or below the configured alpha is labeled statistically significant.
Single-system extrema, state populations, silhouettes, correlations,
information measures, and threshold-state occupancies are normally descriptive
or exploratory.

`reporting.maximum_findings` limits the ranked list. The output separately
reports `candidate_count`, `reported_count`, and
`unreported_candidate_count`, so truncation cannot be mistaken for complete
candidate coverage. The raw reports remain the scientific record.

## QC channel

Structural-integrity and convergence results are kept out of the scientific
ranking. Their statuses and report links appear in `quality_control_records`.
Warnings and errors reported by other modules also enter this channel. This
keeps an urgent QC problem visible without presenting it as a biological
finding.

## Configuration

The picker is enabled by default:

```json
{
  "reporting": {
    "finding_picker_enabled": true,
    "maximum_findings": 50
  }
}
```

Turning the picker off does not disable the underlying analyses or remove their
reports. It only suppresses the consolidated prioritized-finding outputs.
