# Tutorial resource validation, September 22, 2026

PR #125 was merged at `5b15931d913c8026a59a865d91de3651a32db22b` before these
follow-up changes. The companion baseline was
`1813df6f0f032a0af6aa01373b052301b489e72e`.

## Accepted checks

| Route | CPU ceiling | Memory ceiling | Wall ceiling | Preparation | Slurm preview |
| --- | ---: | ---: | ---: | --- | --- |
| Core workstation | 2 | 32 GiB | 2 h | Feasible | Not applicable |
| Interactive workstation preparation | 2 | 32 GiB | 2 h | Feasible | Not applicable |
| Generic Slurm | 2 | 32 GiB | 12 h | Feasible | Feasible; submission permitted |
| DEAC with Apollo v5 catalog | 2 | 32 GiB | 16 h | Feasible | Feasible; submission permitted |

The exact tutorial preparation blocks ran in isolated source snapshots on
DEAC, with Python 3.12 and DSSP 4.6.1. Generic profile site fields came from
the known DEAC profile; its generic scheduler resource policy was retained.
This is one validated site configuration, not a claim about every cluster.
The workstation preparations were checked on Linux, not rerun on macOS.

The generic four-hour attempt passed analysis planning but failed preview:
the minimum serialized scheduler timeout path was 11 hours. The twelve-hour
successor has an estimated execution path of 3.06049 hours and a padded timeout
path of 11.18333 hours. Both previews exited zero, so the JSON feasibility and
submission-permission fields were checked explicitly. The DEAC preview uses
the different `full_campaign_limit_per_planner_backed_job` contract and reports
a 13.47115-hour estimated path within its sixteen-hour ceiling.

The shared budget-recovery recipe was executed against a copy of the rejected
generic study. It preserved the exact original configuration in an exclusive
backup, updated elapsed and total CPU limits to 12 and 24 hours, and prepared
a new directory whose Slurm preview passed. Failed evidence remains intact.

All 35 targeted tests pass: 14 core documentation, 3 tutorial-resource,
11 calibration, and 7 companion tutorial tests. Shell blocks throughout both
tutorial trees pass syntax checks. The initial transferred snapshot included
macOS AppleDouble metadata that caused documentation scans to fail. A new
metadata-free snapshot passed; no tests or source checks were weakened.

## Reproduction and limits

Run the documented preparation blocks with the recorded package environment;
do not execute the later `run`, `run-local.sh`, or submission steps for this
planning check. For Slurm, run `submit.sh --preview` and require both
`generated_schedule_feasibility_status: feasible` and
`submission_permitted: true`.

```bash
python -m unittest discover -s tests -p test_docs.py -v
python -m unittest discover -s tests -p test_tutorial_resources.py -v
python -m unittest discover -s tests -p test_resource_calibrations.py -v
# In the companion checkout:
python -m unittest discover -s tests -p 'test_tutorial*.py' -v
```

[Machine-readable evidence](tutorial_resources_20260922.json) records the
configuration and tutorial hashes, case results, retained evidence locations,
and test outcomes. No analysis jobs were submitted, no viewer build was timed,
and no scientific acceptance is claimed. Planner algorithms, calibration
catalogs, enabled methods, and frame minima are unchanged. Unmatched preflight
and ordinary final-reporting estimates remain a core implementation limitation;
the tutorials now describe it instead of presenting fixed defaults as
job-specific calibration.
