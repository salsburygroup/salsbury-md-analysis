# Tutorial checks after overhead planning repair

These are planning and scheduler-preview results, not observed runtimes. No
analysis jobs were submitted. This record supersedes the resource recommendations
in `tutorial_resources_20260922.md`, whose earlier evidence is retained.

All cases use the bundled NEMO fixture, two CPUs, 32 GiB aggregate memory,
the tutorial's enabled methods and minima, and an existing Python 3.12
environment with DSSP 4.6.1. Each case maps all 32 generated jobs, including
the base preflight, view preflight, and final reporting. The local test
environment without DSSP maps 31 jobs and is not interchangeable with this one.

| Route | Campaign ceiling | Estimated CPU-hours | Planner elapsed estimate | Scheduler preview |
| --- | ---: | ---: | ---: | --- |
| Workstation | 2 h | 1.4703 | 1.1182 h | Not applicable |
| Generic Slurm | 10 h | 1.4703 | 1.1182 h | Feasible; submission permitted |
| Generic Slurm, negative control | 8 h | 1.4703 | 1.1182 h | Infeasible; submission prohibited |
| DEAC with Apollo v5 catalog | 16 h | 10.7840 | 11.9739 h | Feasible; submission permitted |

The generic profile requires a 9.5-hour minimum serialized timeout path and
9.73 hours with preferred allowances. Its ten-hour recommendation reflects
that timeout contract, not an expectation that the fixture takes ten hours.
The example profile uses 44-core, 185-GiB nodes and a one-node maximum;
partition and executable paths are test placeholders. Recheck a real site profile.

The DEAC profile gives each planner-backed job the campaign ceiling as timeout
headroom and checks the estimated dependency path instead of adding those
timeouts. The 11.97-hour planner estimate is close to the 12-hour working
allowance left by its campaign reserves. Any changed tool or output selection
needs a fresh plan. Neither the calibration catalog nor the overhead repair
establishes a measured runtime for this tutorial.

The base and view preflights estimate 1.70 minutes each; final reporting
estimates 17.575 minutes. DEAC padded task requests are 1, 1, and 3 GiB,
respectively, with the profile's per-node reserve separate. These overhead
coefficients remain provisional, not an independently validated fitted model.
The separate interactive HTML build is not included.

## Reproduction

Use the repository root and an unused output directory for every command:

```bash
PYTHONPATH=src python scripts/validate_orchestration_planning.py /path/to/check-local --route workstation
PYTHONPATH=src python scripts/validate_orchestration_planning.py /path/to/check-generic --route generic
PYTHONPATH=src python scripts/validate_orchestration_planning.py /path/to/check-rejected --route generic --hours 8 --expect-preview-infeasible
PYTHONPATH=src python scripts/validate_orchestration_planning.py /path/to/check-deac --route deac
```

The validator prepares workflows and examines preview artifacts; it does not
submit or run them. Source identity, snapshot hash, and exact result values
are in [the machine-readable receipt](tutorial_orchestration_20260922.json).
