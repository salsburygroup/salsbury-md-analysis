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

The opening section contains 10, 11, or 12 findings. The first 10 are always
shown. Ranks 11 and 12 extend the opening section only when a finding at that
boundary is statistically significant after Benjamini-Hochberg correction.
The highlighted report always contains 50 findings when at least 50 candidates
exist, so the secondary section contains 40, 39, or 38 findings respectively.
The JSON records the selected count, any boundary promotions, and the selection
reason. A smaller campaign is marked
`candidate_limited` and presents every available candidate without inventing
entries. Every candidate beyond the first 50 remains searchable in the
interactive report and is written to the JSON and CSV outputs. The output
reports headline, secondary, additional-candidate, and total searchable counts
so the 50-item presentation limit cannot be mistaken for full candidate
coverage. The raw module reports remain the scientific record.

Quality-control and interpretation records remain in the JSON and interactive
report and are also written to `prioritized_findings_qc.md`. Keeping the full
QC ledger separate prevents it from overwhelming the shorter findings report.

## QC channel

Structural-integrity and convergence results are kept out of the scientific
ranking. Convergence records show the ESS reference, the number of RMSD and Rg
series above that reference, and links to every quantitative diagnostic. They
do not label the campaign converged or unconverged. Warnings and errors reported
by other modules also enter this channel. This keeps an urgent QC problem
visible without presenting it as a biological finding.

## Configuration

The picker is enabled by default:

```json
{
  "reporting": {
    "finding_picker_enabled": true,
    "minimum_headline_findings": 10,
    "headline_findings": 12,
    "maximum_findings": 50
  }
}
```

Turning the picker off does not disable the underlying analyses or remove their
reports. It only suppresses the consolidated prioritized-finding outputs.

`minimum_headline_findings` and `headline_findings` set the allowed range. Both
must be from 10 through 12, and the minimum cannot exceed the maximum. The
defaults let the evidence choose among 10, 11, and 12. `maximum_findings` is
fixed at 50 in campaign configuration. The command-line headline override fixes
the count only for bounded diagnostics and compatibility testing; reports
created with an override identify the presentation-contract status as
`explicit_override`.
