# Publication repository policy

## Decision

The general `salsbury-md-analysis` repository contains reusable MD estimators,
generic trajectory adapters, schemas, teaching examples, tests, and maintained
documentation. It remains private until the project owner authorizes a public
release. Every publication-specific workflow and frozen parameter set belongs
in its own repository.

A publication repository depends on an exact reviewed release or commit of
`salsbury-md-analysis`; it does not copy or privately fork a general suite
module. A method developed during a project may first live in that project's
repository. If it is generally useful, it is generalized, reviewed, tested, and
added here without project-specific assumptions.

## Boundary

| General MD toolkit | Publication repository |
|---|---|
| Reusable estimators and algorithms | Exact scientific questions and configuration |
| Portable topology and trajectory adapters | Project orchestration and scheduler scripts |
| Generic schemas and validation | Checksummed data manifests and access instructions |
| Small redistributable teaching fixtures | Figures, tables, and manuscript-specific plots |
| General numerical regression tests | Locked expected results for the paper |
| Maintained method documentation | Citation narrative and author contributions |

Private storage paths, manuscript labels, target or mutant names, restricted
data, and one-off plotting layouts are prohibited in the reusable source,
whether the repository is currently private or later released publicly.

## Publication lock

Each publication repository should contain at least:

```text
README.md
CITATION.cff
LICENSE
analysis-lock.json
environment/
manifests/
config/
scripts/
tests/
expected-results/
```

The analysis lock records the exact toolkit version and commit, publication
repository commit, interpreter and dependency lock, input and configuration
hashes, random seeds, and accepted output hashes. Large or restricted data stay
in authoritative project storage and are referenced by manifest and checksum.

## Freeze rules

1. Generalize reusable functionality in `salsbury-md-analysis`.
2. Pin an exact reviewed toolkit commit in the publication repository.
3. Run from checksummed manifests and locked configuration.
4. Record accepted results and create an immutable submission tag.
5. Create a separate final-publication or correction tag; never move an
   existing tag.
6. Later toolkit improvements do not rewrite an earlier publication lock.

This lets the general toolkit remain comprehensive and improve continuously
while each paper retains an exact, reproducible software variant.
