# Production-readiness assessment

Assessment date: 2026-08-25
Candidate: `0.1.1`

## Decision boundary

This branch contains the proposed `v0.1.1` experimental patch release. It is not
presented as a supported scientific release.
Every registered module remains `experimental`; technical completion, scheduler
success, and scientific validity are separate statuses.

The source contains 45 of 45 MD/core/reporting modules in the registry and
standard profile. The prior `v0.1.0` dependency test is retained in
`validation/v0.1.0_dependency_test.json`. Earlier candidate evidence remains in
`validation/v0.1.0a1_dependency_test.json`. The later memory-planning update,
including its exact source hashes, 457-test result, generated-document check,
low-memory NEMO preparation, and private-remote readback, is recorded separately
in `validation/v0.1.0a1_memory_planner_update_test.json`.
Generated references must match the registry, profiles, schemas, and command
parser before a candidate can be frozen.

## What the generic workflow can do

- Prepare an atom-order-matched PDB plus PSF, PRMTOP/PARM7, or portable bond
  JSON and one DCD per replica without modifying the inputs.
- Optionally create bond JSON through OpenMM when standard templates or reviewed
  residue definitions are available; later analysis does not require OpenMM.
- Run locally without Slurm or through a separately configured Slurm profile.
- Infer conservative applicability for proteins, DNA, RNA,
  protein–nucleic-acid complexes, equivalent oligomers, ligands/cofactors,
  common water models, and supported ion species.
- Build and reuse a connectivity-aware made-whole complete-solute coordinate
  cache while leaving water-dependent analyses on the solvated source.
- Plan deterministic integer-stride sampling inside one configurable campaign
  CPU, memory, scratch, and wall-time envelope and report measured resources and
  exact physical/member-expanded frame coverage.
- Produce per-system and common-basis comparisons, FES and clustering results,
  observed representatives, optional state trajectories, separate FES and
  selected-clustering MSM diagnostics, and non-aggregating final reports.

The supported input and composition boundaries are documented in
[`docs/GENERAL_BIOMOLECULAR_SYSTEMS.md`](docs/GENERAL_BIOMOLECULAR_SYSTEMS.md).
Membranes, glycans, and unknown polymers are retained as generic solute but do
not receive invented specialized chemistry. Unusual covalent or protonation
states require the original simulation topology and explicit project review.

## Retained evidence

| Gate | Current evidence | Interpretation |
|---|---|---|
| Registry/profile | 45 of 45 modules with exact-set tests | Coverage contract, not scientific approval |
| Dependency-equipped suite | Machine-readable release record in `validation/v0.1.0_dependency_test.json`, with earlier candidate evidence retained separately | Software behavior in the recorded environments only |
| Memory-planner update | Source-hash-bound 457-test, low-memory preparation, and GitHub Actions record in `validation/v0.1.0a1_memory_planner_update_test.json` | Current update behavior on the recorded workstation and CI matrix; scientific status remains unevaluated |
| GitHub Actions | Six successful Linux/macOS jobs for Python 3.10--3.12 plus the successful isolated artifact-build/install job | Technical portability and packaging gate for the tested head; rerun after every later commit |
| Installed local workflow | One-CPU, 100-frame TBA result in `validation/v80_installed_local_execution_test.json` | Local-adapter portability check, not a performance or scientific claim |
| Teaching workflow | 1,000-frame NEMO protein-plus-zinc run in `validation/nemo_zinc_finger_tutorial_acceptance.json` | End-to-end tutorial acceptance, not converged sampling or a biological conclusion |
| Generated documentation | Deterministic `scripts/generate_docs.py --check` | Documentation agrees with code metadata |
| Package artifacts | Rebuild and scan the wheel and source archive from the exact retained snapshot; keep their hashes with the candidate evidence | Installability, not method validity |
| Generic acceptance matrix | Protein-only, DNA-only, protein–DNA oligomer, ligand/cofactor, and multi-ion rows are required | Bounded technical generality check |
| Real trajectories | Hash-bounded private TREX, TOP1, TBA, and thrombin execution evidence is retained outside the reusable source | Project evidence; no automatic biological conclusion |
| Scientific status | Every registry module remains `experimental` | Human scientific review is still required per method/project |

TREX trajectories with known scientific defects may be used only as codebase and
resource-planning stress evidence and must remain labeled failed-science evidence.
Withdrawn TOP1 T0/T1 systems are excluded; retained TOP1 intrinsic-DNA evidence
uses D0/D1 only. The retained five-row acceptance gate exercises protein-only,
DNA-only, protein-DNA oligomer, ligand/cofactor, and multi-ion paths. It is a
bounded software test. Every current module and final-summary JSON uses
`scientific_status: not evaluated` unless a later, separate human review record
says otherwise.

## Before a future public release

1. Do not change repository visibility or tag a public release until Linux/macOS
   CI for Python 3.10--3.12 and the isolated artifact-install job pass for the
   exact candidate commit.
2. Retain the final source-archive and wheel hashes with the release evidence.
3. Keep the source and dependency provenance decision in
   `RELEASE_RIGHTS_REVIEW.md` and re-open it when third-party material is added.
4. Protect the public default branch and add a backup reviewer when one is
   available.
5. Keep the NEMO tutorial's source hashes, explicit CC BY 4.0 data license, and
   PDB provenance with the release. Its technical acceptance does not imply
   adequate sampling or scientific validation.

Independent module-by-module scientific review remains the gate for changing a
module from `experimental` to `supported`; it is not silently implied by making
the experimental source visible.

PyPI, conda-forge, containers, Zenodo DOIs, and formal community governance are
not required for the first group-supported release. Repository visibility,
package upload, DOI creation, and public announcement remain separate explicit
actions. Preparing or merging this alpha does not authorize any of them.

## Support gate

A module may be promoted from `experimental` only when its unit/integration and
installed-package checks pass, an appropriate frozen real-project regression
meets predeclared tolerances, limitations and resource bounds are documented,
and both a named scientific reviewer and software maintainer approve it. A paper
should lock the accepted commit and project configuration in its own repository;
publication-specific scripts must not become a divergent copy of the general
suite.
