# Analyze the NEMO zinc-finger fixture on a Slurm cluster

This tutorial runs the supplied NEMO zinc-finger example with the current
`main` branches of Salsbury MD Analysis and its interactive viewer. It uses a
Slurm profile that you fill in for your own cluster. WFU users should follow
the separate [DEAC tutorial](../nemo_zinc_finger_deac/README.md), which already
contains the Salsbury group account, paths, partitions, and environment setup.

The example contains 1,000 frames from a published simulation of the
28-residue human NEMO zinc-finger domain. It is suitable for learning the
workflow, but it is not enough to establish convergence, equilibrium
populations, rare-state sampling, zinc affinity, or a biological mechanism.

## Before you start

You need:

- a Slurm account and a writable directory on storage shared by the login and
  compute nodes;
- the correct account, QoS, partitions, node limits, and scheduler commands for
  your site; and
- Python 3.10-3.12 and an environment that compute nodes can read.

Ask your cluster administrators for site values you do not know. The generic
profile deliberately leaves them blank. A guessed partition, storage root, or
node shape can produce an invalid plan.

`main` changes over time. The steps below save the exact core and viewer commit
SHAs used for the run. For a long-lived or published campaign, install reviewed
commits or released wheels instead of relying on a later `main` checkout.

## 1. Create a shared workspace

Set this to a new directory on your cluster's shared project storage:

```bash
export CLUSTER_WORK="/shared/project/path/salsbury-md-analysis-main"
mkdir -p "$CLUSTER_WORK"
cd "$CLUSTER_WORK"
```

Do not use node-local scratch for the source, environment, prepared campaign,
logs, or accepted reports. Temporary task data may use scratch when the site
profile permits it.

## 2. Clone and record both current main branches

```bash
git clone --branch main --single-branch \
  https://github.com/salsburygroup/salsbury-md-analysis.git core
git clone --branch main --single-branch \
  https://github.com/salsburygroup/salsbury-md-analysis-interactive.git interactive

git -C core rev-parse HEAD | tee CORE_MAIN_COMMIT.txt
git -C interactive rev-parse HEAD | tee INTERACTIVE_MAIN_COMMIT.txt
```

Do not update either checkout inside a prepared campaign. Create a new recorded
environment and campaign when you deliberately change revisions.

## 3. Install the core and viewer together

Create the environment on shared storage with a Python build compatible with
your compute nodes:

```bash
python3 -m venv "$CLUSTER_WORK/.venv"
source "$CLUSTER_WORK/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$CLUSTER_WORK/core[clustering]" \
  -e "$CLUSTER_WORK/interactive"
python -m pip check
salsbury-md-analysis --version
salsbury-md-analysis-interactive --version
```

DSSP is optional. If your site supplies `mkdssp`, load its module or add its
environment setup to the profile. Otherwise, the planner records that the
secondary-structure module is deferred.

## 4. Make a profile for your cluster

Copy the generic template:

```bash
cp "$CLUSTER_WORK/core/profiles/slurm/generic-template.json" \
  "$CLUSTER_WORK/my-cluster.json"
```

Edit `my-cluster.json` and replace every site placeholder. Confirm:

- `profile_id`, `cluster_name`, and the submit, status, and cancel commands;
- the account, Unix group, QoS, and partitions required at your site;
- partition wall-time and node limits;
- `paths.group_storage_root`, `paths.scratch_root`, and every allowed output
  root;
- CPUs and memory per node and the maximum campaign node count; and
- scheduler-only memory and wall-time padding.

Keep `resource_policy.walltime_safety_factor` at `1.0`; the campaign planner has
already applied its task-time factor. Use `walltime_overhead_minutes` for a
scheduler-only allowance.

Set the environment paths to this recorded checkout. The following block
changes only those two fields:

```bash
CLUSTER_WORK="$CLUSTER_WORK" "$CLUSTER_WORK/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CLUSTER_WORK"]).resolve()
path = root / "my-cluster.json"
profile = json.loads(path.read_text(encoding="utf-8"))
profile["environment"]["python_executable"] = str(root / ".venv/bin/python")
profile["environment"]["package_root"] = str(root / "core")
path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
PY

python -m json.tool "$CLUSTER_WORK/my-cluster.json" | less
```

Do not continue while required profile values are `null` or still say
`replace-with-your-cluster-name`.

## 5. Create the NEMO study

```bash
export NEMO_STUDY="$CLUSTER_WORK/nemo-zinc-finger-cluster"
export CORE_CMD="$CLUSTER_WORK/.venv/bin/salsbury-md-analysis"

"$CORE_CMD" init "$NEMO_STUDY" \
  --pdb "$CLUSTER_WORK/core/tutorials/nemo_zinc_finger_workstation/data/nemo_zinc_finger.pdb" \
  --connectivity "$CLUSTER_WORK/core/tutorials/nemo_zinc_finger_workstation/data/nemo_zinc_finger.psf" \
  --trajectory "$CLUSTER_WORK/core/tutorials/nemo_zinc_finger_workstation/data/nemo_zinc_finger_1000_frames.dcd" \
  --frame-interval-ps 0.2 \
  --cpus 2 \
  --memory-gib 32 \
  --hours 1 \
  --adapter slurm \
  --slurm-profile "$CLUSTER_WORK/my-cluster.json"
```

`init` creates editable study and analysis configuration files. It does not
read the trajectory, run analysis, or submit jobs.

Apply the bounded settings used by the workstation example:

```bash
NEMO_STUDY="$NEMO_STUDY" "$CLUSTER_WORK/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["NEMO_STUDY"]).resolve()

study_path = root / "study.json"
study = json.loads(study_path.read_text(encoding="utf-8"))
study["project_id"] = "nemo-zinc-finger-cluster-main"
study_path.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")

config_path = root / "analysis-config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
config["views"]["global_common_heavy"] = {
    "enabled": True,
    "state_trajectory_exports_enabled": False,
    "module_options": {},
}
config["execution"]["maximum_scratch_gib"] = 8.0
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY
```

Read `study.json` and `analysis-config.json`. Confirm the three input paths,
the independently recorded `0.2 ps` frame interval, single-replica grouping,
Slurm adapter, profile path, resource ceilings, enabled modules, and disabled
multi-frame state exports.

## 6. Check the environment and plan

Run these commands on the login or submission host:

```bash
"$CORE_CMD" doctor "$NEMO_STUDY/study.json"
"$CORE_CMD" plan "$NEMO_STUDY/study.json"
```

`doctor` checks the software, scheduler commands, file headers, and matching
atom counts. It cannot validate the chemistry, replica grouping, or scientific
question.

`plan` reads the inputs and writes `$NEMO_STUDY/analysis`; it submits nothing.
Review:

- `planning-report.md` for methods, frames, effective raw strides, and resource
  requests;
- `module-coverage.json` for applicable, inapplicable, disabled, and deferred
  work;
- `automatic-chemical-context.json` for the inferred protein and zinc groups;
- `campaign-resource-plan.json` and `scheduler-resource-requests.json` for task
  estimates and final Slurm requests; and
- `slurm-profile.json` for the profile retained with the campaign.

The fixture should contain one protein and one zinc ion. Water and nucleic-acid
modules should be inapplicable because those atoms are absent. Resolve an
infeasible plan before submission.

## 7. Preview without submitting

```bash
cd "$NEMO_STUDY/analysis"
./submit.sh --preview
less slurm-submission-preview.json
```

The preview must report a feasible generated schedule. Check dependencies,
partitions, CPUs, memory, nodes, time, and the number of jobs. This preview
submits no jobs and does not reserve capacity.

If supported by your site, add a read-only capacity check:

```bash
"$CORE_CMD" advise-slurm-capacity "$NEMO_STUDY/analysis" \
  --wall-hours 1 \
  --cpu-ceiling 2 \
  --format markdown
```

## 8. Submit and monitor

The following command submits the reviewed jobs to Slurm:

```bash
"$CORE_CMD" run "$NEMO_STUDY/analysis"
```

Returned job IDs are stored in `submission-ledgers/`. Submission is not
completion. Check both scheduler activity and accepted outputs:

```bash
squeue --me
"$CORE_CMD" status "$NEMO_STUDY/analysis"
"$CORE_CMD" status "$NEMO_STUDY/analysis" --json
```

Do not submit a duplicate campaign while jobs are active. If work stops, review
the named logs, scheduler state, and proposed recovery:

```bash
"$CORE_CMD" resume "$NEMO_STUDY/analysis"
```

Add `--execute` only after confirming that no earlier job remains active and
that the recovery plan is correct. Accepted reports are reused; failed evidence
is preserved.

## 9. Build the interactive report

For the complete build, transfer, and review sequence, use the companion
[NEMO cluster interactive tutorial](https://github.com/salsburygroup/salsbury-md-analysis-interactive/blob/main/tutorials/nemo_zinc_finger_cluster/README.md).

After `status` reports every scheduled task complete:

```bash
"$CLUSTER_WORK/.venv/bin/salsbury-md-analysis-interactive" \
  "$NEMO_STUDY/analysis"
```

Transfer the whole `interactive-report/` directory to your workstation and
open `index.html`. Its evidence links are relative, so copying only the HTML
file breaks the report. The viewer reads accepted results and does not rerun the
analysis.

## 10. Interpret the result carefully

Inspect structural QC before distributions, molecular states, interactions,
or representative structures. A module marked `technical_status: complete`
passed its software contract. The fixture's `scientific_status` remains
`not evaluated`; technical completion does not establish a publishable
scientific conclusion.

Keep the commit files, environment, site profile, study and configuration,
planning records, submission ledger, logs, reports, and complete interactive
report with any accepted use of the run.
