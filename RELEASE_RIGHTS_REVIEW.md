# Release rights and provenance review

Review date: 2026-09-03
Candidate: `0.1.2`

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
- Most included molecular fixtures are small original synthetic software and
  teaching fixtures. The NEMO tutorial additionally distributes a bounded,
  hash-recorded subset of a published Salsbury-group simulation. The
  group-generated PSF and trajectory subset are explicitly released under CC
  BY 4.0 in `LICENSE-DATA.md`. The PDB-derived starting coordinates retain
  their PDB 2JVX provenance and are not claimed as original group data.
- NumPy, SciPy, scikit-learn, and HDBSCAN are installed dependencies rather
  than copied source. Their installed package metadata identifies permissive
  BSD-compatible licensing; their own license terms continue to apply.
- OpenMM is an optional preparation dependency and is not redistributed.
  `mkdssp` and x3dna-dssr are separately installed external programs and are not
  included in this repository.
- Validation records retain bounded metrics, hashes, and dataset descriptions,
  but no private coordinates or private filesystem locations. The only real
  coordinates distributed by this repository are the explicitly reviewed NEMO
  tutorial fixture described above.
- The DEAC profile intentionally documents a group account, partitions, and
  storage roots. It contains no password, token, private key, or user credential.
- The 0.1.2 presentation code, tables, figures, and state-ion visualization
  logic are original group software. They add no third-party source or private
  molecular coordinates.

## Ongoing rule

New contributions must be original, appropriately licensed, or accompanied by
an explicit permission and attribution record. Publication data and
project-specific scripts remain in their separately reviewed repositories.
