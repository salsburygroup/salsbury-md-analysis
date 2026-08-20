# Replica-resolved RMSD and radius of gyration

Status: **experimental**

This stage-3 slice streams every declared trajectory while retaining its
system, replica, segment, and frame identity. It aligns a named atom selection
to an explicit PDB or GRO reference, reports RMSD for a separately named
selection, and reports mass-weighted radius of gyration for a third named
selection.

```bash
PYTHONPATH=src python3 -m salsbury_md_analysis \
  rmsd-rg path/to/project.json --hash-content
```

The command is read-only and emits JSON to standard output. The output keeps
technical completion separate from scientific validation.

## Required project contract

The project must declare `reference_structure`, `common_atom_policy`, the named
selections, and all settings below:

```json
{
  "reference_structure": "reference.pdb",
  "common_atom_policy": "strict",
  "periodic_coordinate_policy": "reject",
  "selections": {
    "alignment": {"preset": "backbone"},
    "analysis": {"preset": "heavy"}
  },
  "definitions": {
    "replica_rmsd_rg": {
      "alignment_selection": "alignment",
      "rmsd_selection": "analysis",
      "rg_selection": "analysis",
      "minimum_reference_coverage": 1.0,
      "frame_stride": 1
    }
  }
}
```

Selections use portable `all`, `backbone`, `complex_trace`,
`macromolecular_backbone`, `heavy`, and `solute_heavy` presets or exact
atom-name lists. Alignment, RMSD, and Rg selections are resolved separately,
but each resolved selection uses one reference-ordered atom intersection shared
by every declared replica topology. This prevents different variants from being
measured on different permissive subsets. Mapping rejects duplicate identities,
enforces the all-topology reference-coverage gate, and emits a stable SHA-256
signature for each target topology. `strict` identity includes residue name;
`position` permits a residue-name substitution but reports it.

## Calculations

Alignment uses an unweighted least-squares rigid-body rotation and translation.
The implementation follows Horn's unit-quaternion solution: the optimal
rotation is obtained from the largest-eigenvalue eigenvector of a symmetric
4-by-4 matrix. See the [original 1987 paper](https://doi.org/10.1364/JOSAA.4.000629).

For each evaluated frame, the report includes:

- alignment RMSD in angstrom;
- RMSD of the declared RMSD selection after applying the alignment transform;
- mass-weighted radius of gyration in angstrom on the mapped target selection;
- replica, segment, and zero-based frame identities;
- manifest-declared physical frame time and time unit;
- selection coverage and mapping signatures;
- optional topology, trajectory, reference, and aggregate input hashes.

Atomic masses are resolved from topology elements using a fixed internal table.
Unknown or empty elements fail closed rather than receiving guessed masses.
Radius of gyration is invariant to the alignment transform, so it is calculated
directly from the selected trajectory coordinates.

## Scientific limits

- PDB and GRO topologies are supported. Trajectories are limited to the suite's
  documented PDB, GRO, XYZ, and restricted standard 32-bit DCD readers.
- `reject` blocks periodic frames and `allow_wrapped_diagnostic` remains
  diagnostic only. Production periodic RMSD/Rg uses `make_whole` or
  `unwrap_continuous` with explicit topology connectivity and reconstruction
  gates. See [Periodic coordinate reconstruction](PERIODIC_COORDINATES.md).
- Alignment is unweighted. Radius of gyration is mass weighted.
- Every system-manifest segment must declare `first_frame_time`,
  `frame_interval`, and `unit`; the report retains both source frame index and
  normalized physical time.
- Frame subsampling is explicit. It is not a convergence or independence rule.
- Internal finite-coordinate and atom-count checks do not replace the preceding
  structural-integrity stage.
- The synthetic regression validates software behavior only. A hash-pinned
  thrombin-project structural-QC candidate has passed technically, but no
  real-project RMSD/Rg regression is approved yet.
- RMSD is mainly structural-stability and QC evidence. A technically complete
  report does not establish equilibration, convergence, adequate sampling,
  mechanism, functional importance, or scientific validity.
