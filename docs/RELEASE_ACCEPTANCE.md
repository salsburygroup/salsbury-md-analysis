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
| Fresh Linux candidate test | Pending: transfer of the new candidate to the validation host was blocked |
| Windows WSL2 | Pending a real Windows-host test |
| Unfamiliar-person tutorial trial | Pending; automated execution is not a substitute |

The earlier multi-node Slurm acceptance tested the preceding candidate. Keep
that evidence, but do not label it a live test of the new submission wrapper.
The current source tests exercise its dependency, resource, ledger, active-job,
and recovery contracts.

The NEMO run is a software fixture, not an assessment of its molecular
ensemble. The source suites cover other chemistry, oligomer, ion, planner,
and failure cases; that is different from a fresh full campaign on each type
of system.

## Before a public release

1. Repeat the installed test on Linux, including the reviewed Slurm wrapper.
2. Run the tutorial inside WSL2; do not advertise native Windows execution.
3. Have someone who did not develop the package follow the tutorial without
   an AI assistant. Record every undocumented input or manual repair.
4. Review the opening findings against the complete figures and tables.
   A successful ranking test does not establish biological importance.
5. Freeze the accepted wheel hashes, source revision, dependency list, and
   external-executable versions. Tag that candidate only after review.

## Human trial checklist

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
