# Reuse completed results in a repaired campaign

Prepare the repaired campaign first. Review a completed source report against
one exact `task_id` from the new `local-execution-plan.json`:

```bash
salsbury-md-analysis adopt-report /old/prepared/results/example/report.json \
  --prepared /new/prepared --task 'EXACT_TASK_ID'
```

The default is a read-only review. It returns `eligible`, `disposition`, a reason,
source/target identities and scientific-contract hashes. An eligible result has
passed original-context acceptance and has the same inputs and calculation in
the new context. Repeat the command with `--apply` to adopt it. The command does
not submit or run analysis. It refuses active campaigns, unknown scheduler state,
and an existing destination report directory; failed or partial evidence is not
overwritten.

The first supported schemas are ion atmosphere, ion coordination geometry,
nucleic-acid geometry, RDF, trajectory features, scalar feature distributions
and scalar threshold states. Other schemas return `unsupported_schema` or a
more specific validation/change diagnostic. Cache-backed outputs, coordinate
caches, aggregate comparisons and derived inference need their own validated
lineage and are not imported by this command.

## What is checked

| Check | Required match |
| --- | --- |
| Original acceptance | Complete report and summary; authentic project and system manifests; current input bytes match the original content signature; companion hashes and recorded upstream sources validate |
| Input identity | Topology, connectivity, trajectories and weights match by content, preserving system/replica/segment identities and order |
| Scientific settings | Selections, atom definitions, timing, continuity and DCD header policy, sampling, weights, temperature, periodic reconstruction and module settings match |
| Dependencies | Scalar results also bind the trajectory-feature definition; PCA-dependent comparison detects changed upstream definitions, including basis weighting |
| References | Structure/connectivity references are content-identical and covered by the original input-content signature |
| Target | The prepared task names this module and destination; the target project and analysis config remain hash-bound |

Only transport paths and non-scientific project labels/output/compute metadata
are excluded from the comparison. Unknown settings are retained, so an
unrecognized change fails closed. Unrelated PCA settings do not invalidate a
direct ion or scalar calculation, but changed features invalidate their scalar
summaries. An invalid original configuration is not repaired by adoption.

`--apply` copies the original numerical report byte-for-byte and copies any
declared relative companions. In the copied summary it remaps exact report-link
fields to the new report location, leaving its numerical evidence unchanged.
Validation reconstructs that remapping from the original summary and checks
both hashes. A provenance marker identifies the original report and requires
its adoption receipt. Absolute companions stay at their original paths.
A separate `report.json.adoption.json` receipt records the source hashes and
the target binding. Native task-completion and upstream-cache checks revalidate
that receipt, the original context, the current context and companions before
reuse. Deleting the receipt does not bypass normal target acceptance.

The upstream loader can return an in-memory view bound to the target context,
with the original report and receipt identified in `reuse_provenance`. It does
not edit the saved numerical report. Reports consumed directly retain their
original project identities. Keep the original inputs, manifests, reports and
absolute companions accessible; this is not a self-contained archival export.

Reported CPU and memory measurements belong to the original calculation.
The local executor records accepted tasks as `reused_complete` with zero new
execution wall time. Preserve both attempts when reporting campaign cost.

## Limits

Contract equality is technical reuse evidence, not scientific acceptance.
Review the original producer version and relevant numerical fixes before
choosing a source report. Historical reports may lack a producer commit; the
receipt preserves the producer metadata available and records the adoption
validator's hash, but cannot reconstruct a missing software version. A known
incorrect numerical result must be rerun, even if its inputs are unchanged.

This command does not lower thresholds, invent missing analyses, certify
unrecorded upstream calculations, or accept a repaired coordinate cache on the
strength of an older failed cache. Resolve those dependencies separately.

## Qualified-derived dihedral results

`adopt-qualified-dihedral` accepts a separately reviewed dihedral result whose
raw-input and coordinate-cache identities are historical. It checks the retained
outputs and their evidence without decoding trajectories. Ordinary `adopt-report`
and raw-report validation keep their current-input checks.

```bash
salsbury-md-analysis adopt-qualified-dihedral /qualified/report.json \
  --prepared /campaign --task 'EXACT_TASK_ID' \
  --qualification-receipt /qualified/proof.json \
  --policy /qualified/approved-policy.json \
  --policy-sha256 APPROVED_POLICY_SHA256
```

This command reviews the evidence without writing. Add `--apply` only after the
review succeeds and the campaign is idle. The caller supplies a policy hash
approved outside this command. An `accepted` or `eligible` flag in a report does
not grant approval; the toolkit does not generate an approved policy.

The policy binds the exact prepared plan, task, configuration, project, system,
settings, historical input signature and qualification proof. Its six approval
anchors cover historical lineage, cache qualification, numerical qualification,
result review, assembly review and the scientific owner. Review the actual
contents of those records before approving the policy hash. Hash verification
cannot authenticate a reviewer or decide whether their scientific reasoning is
sound.

The proof inventories the report, summary, numerical supplement, population
evidence and recursive retained dependencies. Every entry has an absolute,
canonical, non-symlink path, SHA-256, byte count and `json` or `opaque` type. JSON
references with paired path/hash fields and input/output hash maps must resolve
within that inventory or the policy's historical input identities. The byte
budget bounds the total retained inventory, not process memory. All retained
files and their original paths must remain accessible. This is not a portable
archive or a certificate that a current coordinate cache is unchanged.

The normalized population evidence uses schema
`salsbury-qualified-dihedral-population-v1`. It contains:

- `population`: the proof's exact ordered segments, selected frame indices,
  torsion series, selected-frame/series/observation totals and histogram-bin count.
- `series_definitions`: one row per series, in the same order, with system,
  replica, chain, residue, insertion-code and angle identity plus four distinct
  zero-based `atom_indices`.
- `degeneracy_masks`: a flattened Boolean array; `true` means the observation
  was excluded as degenerate. This must come from the retained qualification,
  not from an assumption that every torsion was valid.
- `group_bounds`: contiguous half-open bounds into that mask, one per series.
  Each group covers the selected frames of its own system/replica. Its
  nondegenerate count must equal the reported series count.

The validator checks identities, ordering, frame bounds, masks, counts,
histogram boundaries and denominators. It preserves the numerical report and
source sidecar byte-for-byte; it does not recompute torsions or circular
statistics. The synthetic test in `tests/test_qualified_dihedral_acceptance.py`
shows the complete policy/proof contract. Its synthetic approvals are test data,
not templates for approving a scientific result.

Publication requires the native campaign lock, known-idle scheduler state and
terminal submission history. It writes a new versioned bundle under
`.qualified-reports/.versions/`, then atomically creates a task registration
without replacing the failed report or editing the prepared plan. Interrupted
unregistered bundles remain evidence and do not count as completed results.
Duplicate publication is rejected. Native completion, runtime reuse, resource
summary and findings checks revalidate the registered evidence on each use.

The validation basis is explicitly `retained-qualified-outputs;
historical-raw-and-cache-identities`. Historical producer work, qualification
work and zero-coordinate scalar assembly stay separate. Helper-generated finding
candidates remain in the original sidecar but are withheld from the runtime
findings view. Technical adoption does not release scientific findings or an
interactive report. Dihedrals remains a leaf result; this command adds no
upstream coordinate-cache mapping.
