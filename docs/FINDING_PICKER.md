# Prioritized findings and complete module accounting

The finding picker puts results that merit attention near the beginning of a
campaign report. It does not decide whether a scientific claim is valid.

Every technically complete module report receives one of six dispositions:

- `ranked_candidates`: the module produced one or more deterministic finding
  candidates;
- `quality_control`: the result belongs in the QC and interpretation channel;
- `interpretive_context`: the result supplies a basis, representation, or
  representative artifact used to interpret another analysis;
- `supporting_context`: the result remains searchable but cannot displace a
  scientific finding in the headline or secondary sections;
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

## Selecting the opening results

The picker builds separate queues for free-energy surfaces, structural
representatives, clustering, coupled interactions, RMSF, and other physical
measurements. It interleaves those queues so that a large family of FES state
populations cannot fill the opening section. It also rotates among comparison
families inside each category. Effect magnitudes are compared only within the
same method-specific family, where the values have the same units and meaning.
The picker does not combine unlike effects into a score.

Entries carry a `ranking_role`. Scientific findings are presentation-eligible.
Failed kinetic-model validations, PCA or tICA basis descriptions, grouped-ML
diagnostics, and coordinate-export records remain searchable as validation or
interpretive context. They do not compete with physical findings for the first
50 positions. State-coordinate exports are linked to matching FES and
clustering entries through `companion_artifact_paths`, so a reader can move
from a population or cluster result to its observed representative structures.

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
separately reports presentation-eligible and supporting-context counts, along
with headline, secondary, additional-candidate, and total searchable counts.
The 50-item presentation limit therefore cannot be mistaken for full candidate
coverage. The raw module reports remain the scientific record.

`evidence_bundles` link findings that concern the same system pair or molecular
entity across more than one module. A bundle lists the contributing modules and
finding identifiers. It does not average effects, rank mechanisms, or claim
that the linked observations have a common cause.

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

Replica-level RMSF permutation inference is also summarized in the QC channel.
The summary states how many comparisons were evaluated, how many familywise
p-values were at or below 0.05, and the minimum familywise p-value. Descriptive
RMSF differences stay visible even when replica-level inference does not
support a statistical label.

## Searchable context

Per-system and conformational-view paths are restored as explicit `system_ids`,
`view_ids`, and `context_label` fields when an older report did not store them.
Information-analysis dimensions are labeled as principal components when that
is the feature basis. Atom-pair network findings use chain, residue, and atom
labels when the source report supplies the mapping.

Older compact hydrogen-bond sidecars are supported. The picker reconstructs
pairwise occupancy comparisons from their stored occupancy evidence without
rerunning trajectory analysis or modifying the completed report.

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
