# Production-readiness assessment

Assessment date: 2026-08-20  
Candidate: `0.0.1.dev84`

## Decision boundary

This candidate is ready for a **private group repository and continued
acceptance testing**. It is not yet a public, supported scientific release.
Every registered module remains `experimental`; technical completion, scheduler
success, and scientific validity are separate statuses.

The source contains 45 of 45 MD/core/reporting modules in the registry and
standard profile. The exact current software-test result, Python environment,
duration, and skip count are recorded in `validation/v84_dependency_test.json`.
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
| Dependency-equipped suite | Machine-readable record in `validation/v84_dependency_test.json` | Software behavior in the recorded environment only |
| Installed local workflow | One-CPU, 100-frame TBA result in `validation/v80_installed_local_execution_test.json` | Local-adapter portability check, not a performance or scientific claim |
| Generated documentation | Deterministic `scripts/generate_docs.py --check` | Documentation agrees with code metadata |
| Package artifacts | Rebuild and scan the wheel and source archive from the exact retained snapshot; keep their hashes with the candidate evidence | Installability, not method validity |
| Generic acceptance matrix | Protein-only, DNA-only, protein–DNA oligomer, ligand/cofactor, and multi-ion rows are required | Bounded technical generality check |
| Real trajectories | Hash-bounded private TREX, TOP1, TBA, and thrombin execution evidence is retained outside the reusable source | Project evidence; no automatic biological conclusion |
| Scientific status | Every registry module remains `experimental` | Human scientific review is still required per method/project |

TREX trajectories with known scientific defects may be used only as codebase and
resource-planning stress evidence and must remain labeled failed-science evidence.
Withdrawn TOP1 T0/T1 systems are excluded; retained TOP1 intrinsic-DNA evidence
uses D0/D1 only. The dev84 five-row acceptance gate exercises protein-only,
DNA-only, protein-DNA oligomer, ligand/cofactor, and multi-ion paths. It is a
bounded software test. Every dev84 module and final-summary JSON uses
`scientific_status: not evaluated` unless a later, separate human review record
says otherwise.

## Before a public or supported release

1. Protect the private repository's default branch and name a primary maintainer
   plus backup reviewer in `CODEOWNERS`.
2. Run CI across Python 3.10–3.12 on Linux and macOS, build/install the artifacts
   in empty environments, and retain exact logs and hashes.
3. Complete the five-row generic acceptance matrix and verify report hashes,
   errors, measured CPU/memory, and source/selected frame coverage.
4. Have independent scientific reviewers approve each method's definitions,
   reference comparison, tolerances, convergence limits, and interpretation.
5. Add a lawful distributable trajectory tutorial fixture or a documented
   download procedure before general community release.
6. Complete contribution-rights and dependency-license review for every retained
   historical component.

PyPI, conda-forge, containers, Zenodo DOIs, and formal community governance are
not required for the first group-supported release. No public repository,
release tag, package upload, DOI, or public announcement is authorized for this
candidate.

## Support gate

A module may be promoted from `experimental` only when its unit/integration and
installed-package checks pass, an appropriate frozen real-project regression
meets predeclared tolerances, limitations and resource bounds are documented,
and both a named scientific reviewer and software maintainer approve it. A paper
should lock the accepted commit and project configuration in its own repository;
publication-specific scripts must not become a divergent copy of the general
suite.
