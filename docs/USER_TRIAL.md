# First-user review worksheet

Ask someone who did not develop the package to follow the
[terminal guide](TERMINAL_WORKFLOW.md) and
[NEMO tutorial](../tutorials/nemo_zinc_finger/README.md) without an AI assistant.
Record obstacles before offering help. An automated replay does not complete
this review.

## Record the environment

- Date and reviewer:
- Operating system and version:
- Python version:
- Package versions, source revisions, or wheel hashes:
- Local computer or Slurm site:
- Available CPUs and memory; chosen campaign time limit:
- Optional dependencies present or absent:

Use Linux or macOS. WSL2 is untested; native Windows is unsupported.
Candidate-only commands require the candidate package, not the older release
tag. Do not overwrite an existing study or accepted result.

## Follow the workflow

| Check | Pass, blocked, or unclear | What happened; exact command or message |
|---|---|---|
| Install in a fresh environment; run `python -m pip check` | | |
| Identify the PDB, matching connectivity, replica DCDs, and saved-frame interval | | |
| Create a study and find the editable input and analysis configuration files | | |
| Run `doctor`; distinguish missing requirements from optional-tool warnings | | |
| Change a module switch and a resource limit before planning | | |
| Plan without executing or submitting work | | |
| Find each method's effective raw stride, selected frames, and off/deferred reason | | |
| Find padded task requests and the aggregate resource boundary | | |
| Run an accepted plan and locate its status, logs, and completion evidence | | |
| Resume a completed study without repeating accepted tasks | | |
| Find figures, CSV tables, and representative structures | | |
| Build the optional viewer after analysis completes | | |
| Copy the whole report folder and open its evidence links offline | | |

For Slurm, configure the site's account, partitions, capacities, executables,
and storage paths before planning. A strict physical-host limit needs a site
placement policy or a shared allocation. Do not submit a production study as
part of this review.

## Review the findings

Use a completed study whose scientific context you understand. The short NEMO
tutorial checks usability; it does not establish meaningful ensemble results.

1. Follow three headline links to the figure and table that support them.
2. Find the separate FES results and primary clustering partition for a
   comparable view. Identify its method, evaluated observations, population
   table, and representative structures.
3. Expand alternatives. If clustering is unranked, can you find the reason?
4. Check whether a method score or QC warning has displaced a physical
   difference on the opening page.
5. Name any large or relevant difference that the headlines missed.
6. Name any headline whose absolute change is too small to merit its rank,
   even if its standardized effect or within-method rank is high.
7. Check that atom identities, replica coverage, units, and comparison
   direction are clear enough to interpret each selected result.

The picker can order measured differences. It cannot supply a universal
biological-importance scale across different observables.

## Return the review

- Steps requiring undocumented knowledge:
- Commands or error messages needing clarification:
- Broken or confusing figure, table, or structure links:
- Headlines to keep, demote, or replace, with reasons:
- Could you complete the bounded workflow without assistance?
- Remaining blocker and proposed documentation or software correction:

Keep this worksheet incomplete until a person performs the trial. Keep
technical completion and scientific interpretation as separate review outcomes.
