# Resource settings and planning limits

Every tutorial uses the core planner and execution adapter. Tutorial settings
define the budget they may use; they do not replace calibration or measure a
job's requirements. Replan when inputs, enabled analyses, code, environment,
hardware, or calibration data change.

## NEMO starting budgets

| Route | Concurrent CPU ceiling | Aggregate memory ceiling | Campaign elapsed-time ceiling | Model source |
| --- | ---: | ---: | ---: | --- |
| Workstation | 2 | 32 GiB | 2 hours | Built-in planner models |
| Generic Slurm | 2 | 32 GiB | 10 hours | Built-in models plus site scheduler policy |
| WFU DEAC | 2 | 32 GiB | 16 hours | Apollo measured calibration catalog v5 plus DEAC policy |

These settings retain the enabled analyses and their frame minima. The generic
Slurm allowance accommodates the generic profile's per-job timeout floors
along the dependency path. It must still pass the preview for your site. The
DEAC allowance includes calibrated analysis costs, censored timing floors, and
provisional preflight/reporting estimates; it is not a measurement that the
small fixture needs 16 hours. The two own-data tutorials
use illustrative budgets only. None of these examples sets a universal budget
for another job.

`maximum_parallel_cpus` limits concurrent campaign use. One CPU per job can
therefore coexist with a two-CPU campaign. `maximum_memory_gib` limits aggregate
concurrent reservations; 32 GiB does not mean each job requests 32 GiB. Read the
individual requests and peak concurrent totals. A recommendation may retain the
original input caps to preserve validated lane packing; check
`request_replay_policy` before treating it as a minimum-memory recommendation.

## Check the plan and the generated requests

Read `campaign-resource-plan.json`, `scheduler-resource-requests.json` for
Slurm, and the generated scripts. Require successful preparation and a feasible
plan. On Slurm, inspect `slurm-submission-preview.json` after
`submit.sh --preview`: `generated_schedule_feasibility_status` must be
`feasible` and `submission_permitted` must be `true`. The preview command can
exit zero while reporting an infeasible schedule. New plans explicitly budget
setup and reporting under **Required execution overhead** in `planning-report.md`.
Queue waiting is separate. A sum of scheduler time limits is not a runtime forecast. Check
`walltime_allocation.contract`: the generic profile enforces a padded
end-to-end reservation ceiling, whereas the current DEAC profile grants the
full campaign limit to each planner-backed job and checks the estimated
dependency-chain duration. Their feasibility rules therefore differ. For the
tested generic profile with DSSP available, an eight-hour campaign passed
analysis planning but failed preview because its minimum serialized timeout path was 9.5 hours;
the ten-hour example passes, including the preferred 9.73-hour timeout path.
Do not infer feasibility from the analysis planner's two-hour recommendation alone.

Preflight estimates now depend on input bytes, file reads, replicas, and topology
size. Reporting estimates depend on report bundles, views, system size, and
enabled output components. These are conservative workload models, not fitted
job-specific calibrations. The scheduler uses the planner's reservations;
its timeout is not a runtime estimate. The configured time and memory factors
are applied once, with the separate per-node memory reserve retained.
See [the overhead model](../docs/ORCHESTRATION_RESOURCE_ESTIMATES.md).

Use `execution.resource_calibration_catalog` for an appropriate validated
catalog. Record observed overhead costs to refine these provisional models.
New-format plans reject missing preflight or final-reporting task mappings.
Previously prepared directories retain their legacy requests: prepare into a
new directory after upgrading rather than editing old scripts.

Planning-only checks with DSSP available mapped all 32 jobs. The workstation
model estimated 1.12 elapsed hours within its two-hour ceiling; the DEAC model
estimated 11.97 hours within a 16-hour ceiling. The latter is close to its
12-hour working allowance after campaign reserves, so replan after any change
in enabled tools or outputs. Neither figure is a measured runtime. See
[the validation record](../validation/tutorial_orchestration_20260922.md).

## If planning rejects the budget

Read the failure reasons and
`permissive_minimum_resource_request.recommended_request` in the failed
`campaign-resource-plan.json`. An infeasible preparation has no analysis jobs
to resume. Preserve that directory.

For `prepare-analysis`, repeat preparation with the reviewed
`--target-wall-hours` recommendation and a new `--output` directory. That flag
does not belong to `plan`. For a study created with `init`, update both
`execution.maximum_hours_per_cpu` and `execution.maximum_total_cpu_hours` in
the study config. This example changes time only; separately review any CPU,
memory, scratch, or node shortfall:

```bash
export STUDY_DIR="/absolute/path/to/your/study"
export REVISED_HOURS="12"  # replace with the reviewed recommendation
python - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["STUDY_DIR"])
path = root / "analysis-config.json"
original = path.read_bytes()
config = json.loads(original)
hours = float(os.environ["REVISED_HOURS"])
if not math.isfinite(hours) or hours <= 0:
    raise ValueError("REVISED_HOURS must be finite and positive")
with (root / "analysis-config.before-budget-change.json").open("xb") as backup:
    backup.write(original)
config["execution"]["maximum_hours_per_cpu"] = hours
config["execution"]["maximum_total_cpu_hours"] = (
    hours * config["execution"]["maximum_parallel_cpus"]
)
path.write_text(json.dumps(config, indent=2) + "\n")
PY
export ANALYSIS_DIR="$STUDY_DIR/analysis-replanned"
salsbury-md-analysis plan "$STUDY_DIR/study.json" --output "$ANALYSIS_DIR"
```

Use unused backup and output names for each attempt. In the NEMO cluster
tutorials, set `NEMO_ANALYSIS="$ANALYSIS_DIR"` after successful planning. Use
that directory for preview, run, status, recovery, viewer build, and archive.
Use the accepted budget in capacity checks too. Do not edit an old prepared
directory or disable methods merely to silence an infeasible-plan error.

The interactive HTML build happens after core completion. It has no calibrated
CPU, memory, or time request in the core plan. Follow the companion's report
resource guidance, measure representative builds, and keep that cost separate
from trajectory analysis and core final reporting.
