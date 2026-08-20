# Release rights and provenance review

Review date: 2026-08-20  
Candidate: `0.1.0a1`

## Decision

The retained source is cleared for public source distribution under the
repository licenses. Freddie R. Salsbury Jr., as project owner, confirms that
the Salsbury group has the rights needed to release the group-written code and
original teaching material in this repository.

This is a provenance record, not a transfer of rights in software or data that
the repository does not distribute.

## What was checked

- The source, scripts, tests, and examples contain no vendored package trees or
  third-party source headers. References to algorithms and external programs do
  not copy their implementations into this repository.
- The included molecular fixtures are small original synthetic software and
  teaching fixtures. No private trajectory, collaborator structure, or
  publication data file is distributed.
- NumPy, SciPy, scikit-learn, and HDBSCAN are installed dependencies rather
  than copied source. Their installed package metadata identifies permissive
  BSD-compatible licensing; their own license terms continue to apply.
- OpenMM is an optional preparation dependency and is not redistributed.
  `mkdssp` and x3dna-dssr are separately installed external programs and are not
  included in this repository.
- Validation records retain bounded metrics, hashes, and dataset descriptions,
  but not private coordinates or private filesystem locations.
- The DEAC profile intentionally documents a group account, partitions, and
  storage roots. It contains no password, token, private key, or user credential.

## Ongoing rule

New contributions must be original, appropriately licensed, or accompanied by
an explicit permission and attribution record. Publication data and
project-specific scripts remain in their separately reviewed repositories.
