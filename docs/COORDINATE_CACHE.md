# Reusable coordinate caches

`build-coordinate-cache` creates an atomic, reusable coordinate representation
for analyses that do not require bulk solvent. It reads each original solvated
trajectory segment once, reconstructs complete bonded components with explicit
connectivity, removes water, and writes an unaligned `molecular_payload` DCD.
The payload retains protein, nucleic acid, hydrogens, ligands, cofactors, and
ions in their original atom order.

```bash
PYTHONPATH=src python -m salsbury_md_analysis build-coordinate-cache \
  path/to/system.json --output path/to/new-cache
```

The output directory must not already exist. It is assembled under a temporary
sibling directory and installed with one atomic rename only after every frame,
topology, connectivity file, manifest, and report has been written and
validated. A failed build removes only its private temporary directory and
never edits the source manifest, topology, connectivity, or trajectory.

## What the cache preserves

- original system, replica, segment, and frame identities;
- segment timing and declared continuity fields;
- DCD frame count, starting step, save interval, and periodic cell;
- source PDB atom identity and order for the retained payload;
- an explicit subset bond graph with source-connectivity provenance;
- SHA-256 hashes for every generated DCD, topology, connectivity file, and
  cache manifest.

The cache is deliberately unaligned. Each scientific view must still apply its
own declared protein, nucleic-acid, interface, or oligomer-member alignment.
PCA feature atoms and exported coordinate payload atoms remain separate: a
common-heavy PCA may export a representative containing the complete solute.

## Methods that must retain the solvated source

Water-mediated hydrogen bonds, water or solvent RDFs, hydration-shell
observables, and any analysis whose declared atom groups include bulk water
must read the original solvated trajectories. The cache is not permission to
drop chemically required solvent. A campaign may therefore use the cache for
solute conformational work while scheduling water-dependent methods from the
immutable original inputs.

## Acceptance boundary

A completed cache report proves only that the computational representation was
written consistently. Before a new cache implementation is accepted, compare
source and cached coordinates after the same make-whole operation on frozen
frames, verify atom identities and timing, and record CPU, memory, and storage
use. Coordinate equivalence is not evidence that downstream analyses are
scientifically valid, converged, or adequately sampled.
