# Independent-user acceptance

The candidate adds a terminal workflow; it is not a claim that every operating
system or biomolecular setup has passed a fresh end-to-end test.

## Evidence recorded on 5 September 2026

| Check | Result |
|---|---|
| Main source suite | 735 tests passed |
| Experimental source suite | 830 tests run; one optional-dependency test skipped |
| Shared implementation contracts | 150 matched, with documented experimental additions |
| Fresh macOS wheel installs | Main and experimental passed dependency checks |
| Installed NEMO test | Each branch completed 31 tasks and 28 reports from a 100-frame protein–zinc fixture |
| Plan-only and recovery | Planning launched no analysis; completed resume preserved report hashes and timestamps |
| Offline viewer | Chrome navigation, images, packaged links, and no-network checks passed |
| TBA presentation replay | Preserved the 706 candidates in the current source index; no trajectory analysis rerun |
| Fresh Linux wheel installs and suites | Main: 735 run, one OpenMM skip; experimental: 830 run, one OpenMM skip; no failures |
| Installed Linux terminal workflow | Each branch completed 31 tasks and 28 reports; accepted resume preserved outputs |
| Fresh candidate two-node Slurm test | Each branch completed 13 jobs, 10 analysis reports, and 4 supporting artifacts |
| Windows WSL2 | Deferred: no Windows test host is available; WSL2 is not validated |
| Unfamiliar-person tutorial trial | Pending; automated execution is not a substitute |

The fresh Slurm trials tested main `3185cda` and experimental `acfbe3f`.
They ran sequentially, each within two physical hosts, four allocated CPUs,
and 12 GiB of requested memory. Each used three identical copies of a
100-frame fixture, with all strides set to 1. These copies test dispatch and
pooling, not independent replicas. Report, input, cache, and companion hashes
passed; accepted resume submitted no new work.

The validation launcher explicitly restricted physical hosts. Logical node
slots alone do not bind separate Slurm jobs to a fixed set of hosts. A site
requiring that restriction must provide a placement policy or shared
allocation. Node-loss, OOM, and long checkpoint recovery were not injected in
these live trials.

The NEMO run is a software fixture, not an assessment of its molecular
ensemble. The source suites cover other chemistry, oligomer, ion, planner,
and failure cases; that is different from a fresh full campaign on each type
of system.

## Before a public release

1. Retain the accepted macOS, Linux, and Slurm evidence with the tested revisions.
   Repeat affected checks if executable code or dependencies change.
2. Keep WSL2 outside the validated-platform claim until a Windows-host test
   is available. Native Windows execution is unsupported.
3. Have someone who did not develop the package follow the tutorial without
   an AI assistant. Record every undocumented input or manual repair.
4. Review the opening findings against the complete figures and tables.
   A successful ranking test does not establish biological importance.
5. Freeze the accepted wheel hashes, source revision, dependency list, and
   external-executable versions. Tag that candidate only after review.

## Human trial checklist

Use the [review worksheet](USER_TRIAL.md) to record the result.

Start with a clean machine or environment. Record the OS and Python version.
Use the tutorial to install, create a study, change its resource envelope,
check inputs, plan without running, execute, inspect status, resume completed
work, and copy the report to another folder. Open figures, CSV tables, and
representative structures from the copied report.

For the report review, find the selected clustering method, its comparison
scope, the reason it was selected or left unranked, the alternatives, and the
separate FES results. Identify the evidence behind three headlines. Note
anything that required knowledge absent from the documentation.

Do not mark this trial complete from an automated script or a developer's
familiarity with the package.
