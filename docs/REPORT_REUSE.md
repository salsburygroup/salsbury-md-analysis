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
