# Periodic coordinate reconstruction

Periodic production analysis is explicit and connectivity aware. The suite
does not infer general covalent bonds from coordinate distances.

## Policies

`periodic_coordinate_policy` has five values:

- `reject` blocks every frame that declares a periodic cell.
- `allow_wrapped_diagnostic` preserves wrapped coordinates and emits warnings.
  It is retained for technical diagnosis, not production interpretation.
- `make_whole` reconstructs every bonded component independently in every
  frame using explicit topology bonds and exact minimum-image bond vectors.
- `unwrap_continuous` first makes every component whole, then follows the
  lowest-index atom of each component through the nearest periodic image across
  frames. State continues across segments only when
  `continuous_with_previous` is true.
- `preprocessed_make_whole` accepts only a toolkit-written molecular-payload
  cache whose DCD marker, cache-report SHA-256, cache-report status, coordinate
  representation, and cached-system-manifest hash all match. It avoids
  reconstructing coordinates that the cache builder has already made whole.

The implementation handles orthorhombic and triclinic cells. Nearest images are
found by an exact, reciprocal-cell-bounded lattice search. Singular or
pathologically ill-conditioned cells fail closed. DCD length/angle and
angle-cosine cell records are supported; unsupported symmetric-matrix DCD cell
records are rejected instead of guessed.

## Manifest contract

Every replica using a reconstruction policy declares a complete connectivity
file:

```json
{
  "replica_id": "replica-1",
  "topology": "equilibrated.pdb",
  "connectivity": "system.psf",
  "segments": []
}
```

Supported connectivity formats are CHARMM/NAMD PSF, Amber PRMTOP/PARM7, and a
portable zero-based JSON form:

```json
{
  "format": "salsbury-bonds-v1",
  "atom_count": 3,
  "index_base": 0,
  "bonds": [[0, 1], [1, 2]]
}
```

The connectivity atom count must exactly equal the topology and trajectory
atom count. Connectivity files are included in the deterministic system input
inventory and content signature. A periodic reference structure additionally
declares `reference_connectivity` in the project manifest.

A preprocessed cache uses a different fail-closed declaration:

```json
{
  "periodic_coordinate_policy": "preprocessed_make_whole",
  "preprocessed_coordinate_source": {
    "cache_report": "coordinate-cache-report.json",
    "cache_report_sha256": "64-lowercase-or-uppercase-hex-digits"
  }
}
```

Ordinary periodic DCDs fail this policy. The cache remains unaligned, so each
analysis still performs its own declared structural alignment. Content-hashed
preflight remains required to protect the cached topology, connectivity, and
trajectory bytes; this policy only removes duplicate reconstruction work.

Reconstruction gates are explicit:

```json
{
  "periodic_coordinate_policy": "unwrap_continuous",
  "periodic_reconstruction": {
    "maximum_bond_length_angstrom": 3.0,
    "cycle_closure_tolerance_angstrom": 0.0001,
    "maximum_anchor_displacement_angstrom": 30.0
  }
}
```

The bond-length gate detects incorrect connectivity, wrong atom order, wrong
units, and unresolved imaging. The cycle-closure gate rejects a bond graph that
cannot be represented consistently as whole components. The anchor-displacement
gate catches excessive apparent motion between saved frames; it cannot prove
that the true displacement was less than half a box, so save frequency remains
a scientific input assumption.

## Exporting portable connectivity from an OpenMM PDB

Install the optional converter dependency and export once:

```bash
python -m pip install -e '.[openmm-connectivity]'
python scripts/export_openmm_connectivity.py equilibrated.pdb equilibrated.bonds.json
```

OpenMM builds standard-residue bonds from atom and residue names and includes
explicit PDB connectivity. The exporter records the OpenMM version and source
PDB SHA-256, and fails when an atom in a multi-atom residue is left isolated.
This is template-derived explicit topology, not a general distance-cutoff bond
guess. Nonstandard residues should instead use the simulation PSF/PRMTOP or
reviewed OpenMM bond definitions.

The same guarded exporter is available directly from the generic workflow when
no connectivity file exists:

```bash
salsbury-md-analysis prepare-analysis --pdb equilibrated.pdb \
  --generate-connectivity-openmm --trajectory production.dcd \
  --frame-interval-ps 10 --project-id example --output analysis-example
```

The generated `generated-connectivity/*.bonds.json` is retained with its source
hash and OpenMM version. Subsequent analysis needs that JSON, not OpenMM. A
parameterized PSF is not synthesized because OpenMM PDB topology alone does not
establish force-field atom types, charges, or nonstandard residue parameters.

## Scientific boundaries

Making components whole does not place separate, nonbonded components into a
unique common image. `unwrap_continuous` preserves the first frame's component
images and follows them thereafter. Therefore the first frame must already
represent the intended multicomponent assembly. Pair, group-contact, and
hydrogen-bond geometry uses triclinic minimum-image vectors for periodic frames.
Whole-assembly RMSD, radius of gyration, PCA, DCCM, or DSSP selections spanning
separate nonbonded components still require an intentionally prepared first
frame and sensitivity review.

These operations fix coordinate representation. They do not establish
equilibration, convergence, adequate sampling, populations, mechanism, or
scientific validity. The distinction between `whole` and `nojump` follows the
same conceptual split documented by
[GROMACS `gmx trjconv`](https://manual.gromacs.org/documentation/2025.1/onlinehelp/gmx-trjconv.html).
