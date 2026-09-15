# Run the current main code on WFU DEAC

This tutorial starts with the NEMO zinc-finger teaching files, runs the current
`main` branch of Salsbury MD Analysis through DEAC's Slurm scheduler, and builds
the offline interactive report with the current `main` branch of the companion
viewer.

The example is a 1,000-frame subset of a published Salsbury-group simulation of
the 28-residue human NEMO zinc-finger domain. It is small enough to learn the
cluster workflow. It tests software behavior; it does not establish convergence,
equilibrium populations, rare-state sampling, zinc affinity, or a biological
mechanism.

Use the
[workstation NEMO tutorial](../nemo_zinc_finger_workstation/README.md) for the
input provenance and a local run. The
[generic cluster tutorial](../nemo_zinc_finger_cluster/README.md) covers Slurm
without DEAC account names, paths, and partitions. This page covers the
DEAC-specific differences:
installing both current checkouts in one environment, binding the Slurm profile
to that environment, reviewing a scheduler preview, submitting, checking
status, and downloading the complete browser.

## Before you start

You need:

- a DEAC account with access to the `salsburyGrp` Unix group;
- permission to write under `/deac/phy/salsburyGrp`; and
- enough quota for the source checkouts, Conda environment, prepared campaign,
  results, and interactive report.

The checked-in DEAC profile names the validated shared `v76` environment. That
environment is useful provenance for earlier accepted work, but it is not the
current `main` checkout. This tutorial makes a copy of the profile and points it
to a new environment without modifying `v76`.

`main` moves. The commands below record the exact core and viewer commits used
for the run. Keep those files with the results. For a long-lived or published
campaign, install reviewed commit SHAs or wheels instead of relying on a later
checkout of `main`.

## 1. Log in and choose a group-storage workspace

Log in from your computer, then enter Bash so the commands below have the same
syntax regardless of your DEAC login shell:

```bash
ssh apollo
bash
```

Choose a new directory under the group filesystem. This example uses your DEAC
username. Change the first line if your work belongs in an existing project
directory.

```bash
export DEAC_WORK="/deac/phy/salsburyGrp/$USER/salsbury-md-analysis-main"
mkdir -p "$DEAC_WORK"
cd "$DEAC_WORK"
```

Keep the source, environment, prepared campaign, and results on group storage.
The supplied profile permits analysis output under `/deac/phy/salsburyGrp`; it
does not permit a campaign rooted in your home directory or `/scratch`.

## 2. Clone both current main branches and record them

```bash
git clone --branch main --single-branch \
  https://github.com/salsburygroup/salsbury-md-analysis.git core
git clone --branch main --single-branch \
  https://github.com/salsburygroup/salsbury-md-analysis-interactive.git interactive

git -C core rev-parse HEAD | tee CORE_MAIN_COMMIT.txt
git -C interactive rev-parse HEAD | tee INTERACTIVE_MAIN_COMMIT.txt
```

Do not run `git pull` inside a prepared or accepted campaign. A later commit may
change code, dependencies, planning, or report contracts. Create and record a
new environment and campaign when deliberately testing a later revision.

## 3. Create one environment for the core and viewer

DEAC has a Salsbury-group Conda installation. Create a separate environment in
this workspace from the core repository's reviewed dependency file, then
install both checkouts:

```bash
export DEAC_CONDA="/deac/phy/salsburyGrp/software/miniconda3/bin/conda"

"$DEAC_CONDA" env create \
  --prefix "$DEAC_WORK/.venv" \
  --file "$DEAC_WORK/core/environment.yml"

"$DEAC_WORK/.venv/bin/python" -m pip install --no-build-isolation \
  -e "$DEAC_WORK/core" \
  -e "$DEAC_WORK/interactive"

"$DEAC_WORK/.venv/bin/python" -m pip check
"$DEAC_WORK/.venv/bin/salsbury-md-analysis" --version
"$DEAC_WORK/.venv/bin/salsbury-md-analysis-interactive" --version
"$DEAC_WORK/.venv/bin/mkdssp" --version
```

The environment file installs Python 3.12, the default numerical dependencies,
HDBSCAN, and DSSP. DSSP is an external executable; installing only the Python
packages would leave protein secondary structure deferred.

## 4. Bind a private copy of the DEAC profile to this checkout

Make a local profile copy. The short Python block changes only the execution
environment paths. It preserves the DEAC account, group, QoS, partitions,
scheduler commands, node shape, storage policy, and scheduler padding from the
checked-in profile.

```bash
cp "$DEAC_WORK/core/profiles/slurm/deac.json" \
  "$DEAC_WORK/deac-current-main.json"

DEAC_WORK="$DEAC_WORK" "$DEAC_WORK/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DEAC_WORK"]).resolve()
path = root / "deac-current-main.json"
profile = json.loads(path.read_text(encoding="utf-8"))
profile["environment"]["python_executable"] = str(root / ".venv/bin/python")
profile["environment"]["package_root"] = str(root / "core")
path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
PY
```

Inspect the copy before using it:

```bash
"$DEAC_WORK/.venv/bin/python" -m json.tool \
  "$DEAC_WORK/deac-current-main.json" | less
```

The expected DEAC settings are account `salsburygrp`, Unix group
`salsburyGrp`, QoS `normal`, routine partition `small`, long or multi-node
partition `large`, and scheduler commands under `/opt/scyld/slurm/bin`.

## 5. Create the NEMO study

`init` writes an editable study and analysis configuration. It does not read
trajectory coordinates, run analysis, or submit jobs.

```bash
export NEMO_STUDY="$DEAC_WORK/nemo-zinc-finger-deac"
export CORE_CMD="$DEAC_WORK/.venv/bin/salsbury-md-analysis"

"$CORE_CMD" init "$NEMO_STUDY" \
  --pdb "$DEAC_WORK/core/tutorials/nemo_zinc_finger_workstation/data/nemo_zinc_finger.pdb" \
  --connectivity "$DEAC_WORK/core/tutorials/nemo_zinc_finger_workstation/data/nemo_zinc_finger.psf" \
  --trajectory "$DEAC_WORK/core/tutorials/nemo_zinc_finger_workstation/data/nemo_zinc_finger_1000_frames.dcd" \
  --frame-interval-ps 0.2 \
  --cpus 2 \
  --memory-gib 32 \
  --hours 1 \
  --adapter slurm \
  --slurm-profile "$DEAC_WORK/deac-current-main.json"
```

Apply the two bounded NEMO settings from the workstation tutorial and attach
the measured DEAC calibration catalog. This keeps multi-frame state trajectory
exports off while retaining representative structures.

```bash
DEAC_WORK="$DEAC_WORK" NEMO_STUDY="$NEMO_STUDY" \
  "$DEAC_WORK/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DEAC_WORK"]).resolve()
study_root = Path(os.environ["NEMO_STUDY"]).resolve()

study_path = study_root / "study.json"
study = json.loads(study_path.read_text(encoding="utf-8"))
study["project_id"] = "nemo-zinc-finger-deac-main"
study_path.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")

config_path = study_root / "analysis-config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
config["views"]["global_common_heavy"] = {
    "enabled": True,
    "state_trajectory_exports_enabled": False,
    "module_options": {},
}
config["execution"]["maximum_scratch_gib"] = 8.0
config["execution"]["resource_calibration_catalog"] = str(
    root / "core/profiles/apollo_measured_resource_calibrations_v5.json"
)
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY
```

Read `study.json` and `analysis-config.json` before planning. Confirm the three
input paths, the independently recorded `0.2 ps` saved-frame interval, the
single-replica grouping, the Slurm adapter, the copied profile path, resource
limits, enabled modules, and disabled multi-frame state exports.

## 6. Check the environment and make a plan

```bash
"$CORE_CMD" doctor "$NEMO_STUDY/study.json"
"$CORE_CMD" plan "$NEMO_STUDY/study.json"
```

`doctor` checks software, scheduler commands, input headers, and matching atom
counts. It cannot decide whether the chemistry, trajectory grouping, or
scientific question is correct.

`plan` reads the inputs and writes the prepared campaign at
`$NEMO_STUDY/analysis`. It does not submit jobs. Review at least:

- `planning-report.md` for methods, effective raw strides, selected frames,
  off or deferred work, and resource requests;
- `module-coverage.json` for automatic, inapplicable, disabled, and deferred
  modules;
- `automatic-chemical-context.json` for the inferred protein and zinc groups;
- `campaign-resource-plan.json` and `scheduler-resource-requests.json` for
  task estimates and final Slurm requests; and
- `slurm-profile.json` for the immutable profile copy retained with the
  campaign.

For this fixture, the chemistry should contain one protein and one zinc ion.
Water and nucleic-acid modules should be inapplicable because those atoms are
absent. `secondary_structure` should not be deferred when the new environment's
`mkdssp` is available.

## 7. Preview Slurm without submitting

```bash
cd "$NEMO_STUDY/analysis"
./submit.sh --preview
less slurm-submission-preview.json
```

The preview must say the generated schedule is feasible. Check the job count,
dependencies, partitions, CPU and memory requests, node assignments, and
critical path. A preview is not a reservation and submits nothing.

An optional read-only capacity check can add current node fit and queue
pressure:

```bash
"$CORE_CMD" advise-slurm-capacity "$NEMO_STUDY/analysis" \
  --wall-hours 1 \
  --cpu-ceiling 2 \
  --format markdown
```

Queue conditions can change after this check.

## 8. Submit the reviewed plan

The next command submits Slurm jobs. Run it only after reviewing the study,
configuration, plan, and scheduler preview.

```bash
"$CORE_CMD" run "$NEMO_STUDY/analysis"
```

The launcher records returned job IDs in `submission-ledgers/`. A successful
submission is not a completed analysis.

Check scheduler activity and accepted outputs separately:

```bash
/opt/scyld/slurm/bin/squeue --me
"$CORE_CMD" status "$NEMO_STUDY/analysis"
"$CORE_CMD" status "$NEMO_STUDY/analysis" --json
```

Do not resubmit because a report is still absent while its job is queued or
running. If work stops, first run `status` and inspect the named logs. A recovery
review does not execute anything:

```bash
"$CORE_CMD" resume "$NEMO_STUDY/analysis"
```

Only after confirming that no earlier DEAC job remains active should you run
the reviewed recovery with `--execute`. Accepted reports are reused; invalid or
changed outputs fail closed and are not overwritten.

## 9. Build the interactive report after completion

Wait until `status` reports every scheduled task complete. Then run the viewer
on the DEAC login node; it reads the accepted core reports and does not rerun
the trajectory analysis.

```bash
"$DEAC_WORK/.venv/bin/salsbury-md-analysis-interactive" \
  "$NEMO_STUDY/analysis"
```

The browser is written to:

```text
nemo-zinc-finger-deac/analysis/interactive-report/index.html
```

Keep the entire `interactive-report/` directory because its evidence links are
relative. To download it through DEAC Open OnDemand or another file-transfer
client, first make one archive:

```bash
tar -C "$NEMO_STUDY/analysis" -czf \
  "$NEMO_STUDY/nemo-interactive-report.tar.gz" \
  interactive-report
```

Download the archive, extract it on your computer, and open `index.html` in a
current browser. The report is self-contained and does not send structures or
results to an external service.

## 10. Read the result without overclaiming it

Start with structural QC, then inspect the linked figures, tables, and
representative structures behind each prioritized finding. The interactive
report organizes the evidence; it does not rank scientific importance or
replace review of sampling, chemistry, uncertainty, and method assumptions.

A module marked `technical_status: complete` passed its software contract. The
NEMO fixture's `scientific_status` remains `not evaluated`. Keep the two commit
files, Conda environment, copied Slurm profile, study/configuration files,
planning records, submission ledger, logs, reports, and the complete interactive
report with any accepted use of the run.
