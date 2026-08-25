# Structural-integrity QC

Status: **experimental**

The first coordinate-level analysis module streams every declared trajectory,
detects execution-invalid inputs, and records explicit coordinate and optional
chemical-integrity review findings. It is the start of standard-analysis stage
2, not a substitute for project-specific chemical review.

```bash
PYTHONPATH=src python3 -m salsbury_md_analysis \
  structural-qc path/to/project.json --hash-content
```

The command is read-only and emits JSON to standard output. It consumes the
compiled project context, preserving system, replica, segment, and
manifest-declared physical-time identities.

## Supported coordinate records

- PDB implicit or multi-model coordinates, interpreted in angstrom;
- one-frame GRO coordinates, converted from nanometer to angstrom;
- multi-frame XYZ coordinates using the project's declared coordinate unit;
- standard 32-bit `CORD` DCD coordinate records, with little- or big-endian
  markers, using the project's declared coordinate unit.

The DCD reader rejects fixed-atom trajectories and 64-bit record markers. It
can skip a declared CHARMM fourth-dimension block and decodes standard
length/angle or angle-cosine unit-cell records. Unsupported symmetric-matrix
cell records fail closed. DCD conventions vary among CHARMM, NAMD, X-PLOR, and
LAMMPS, so the declared unit and periodic behavior remain explicit limitations.
Reference implementations and format
notes:

- https://www.ks.uiuc.edu/Research/vmd/plugins/doxygen/dcdplugin_8c-source.html
- https://docs.mdanalysis.org/stable/documentation_pages/coordinates/DCD.html

XTC, TRR, NetCDF, and other formats still fail clearly rather than being
silently converted.

## Required project gates

The project manifest must declare the following under
`definitions.structural_qc`:

```json
{
  "definitions": {
    "structural_qc": {
      "near_coincident_distance_angstrom": 0.5,
      "maximum_near_coincident_pairs_per_frame": 0,
      "maximum_absolute_coordinate_angstrom": 1000000.0,
      "maximum_frame_atom_displacement_angstrom": null,
      "frame_stride": 1
    }
  }
}
```

No scientific threshold is silently chosen by the command. The displacement
gate is optional and applies after the declared periodic preprocessing policy;
it should remain `null` under wrapped diagnostic execution when coordinates can
cross a boundary. `frame_stride` affects near-coincident-pair evaluation only;
atom count, finite-coordinate, extent, and displacement checks still inspect
every streamed frame.

## Current gates

- trajectory readability and declared frame completion;
- per-frame topology/trajectory atom-count agreement;
- finite coordinate values;
- maximum absolute coordinate extent;
- near-coincident atom pairs using a memory-bounded spatial grid;
- optional raw atom displacement between consecutive frames;
- optional peptide-link continuity and trans-omega geometry checks defined
  only by explicit PSF/PRMTOP/bond-JSON C-N connectivity;
- optional C-alpha chirality, element-radius steric-clash, and declared
  covalent-link checks;
- explicit reporting of evaluated frames, normalized units, examples, and
  technical failures and nonblocking human-review findings.
- declared segment timing and the physical-time range evaluated.

## Scientific limits

Chemical checks run only when `chemical_integrity` is explicitly configured;
their thresholds must be frozen before comparative outcome review. Peptide and
omega checks never infer bonds from residue order, numbering, or a shared chain
identifier. The conservative steric screen excludes same-residue and explicit
topology 1-2 and 1-3 pairs and does not replace specialized treatment of
cofactors, metals, interfaces, unusual residues, or force-field chemistry.
Near-coincident searches do not add periodic images between separate
components.

Unreadable coordinates, invalid manifests, atom-count disagreement, and
nonfinite coordinates remain technical failures because the requested
calculation cannot be executed reliably. Threshold exceedances are reported as
`qc_status: review_required`, with `human_review_status: pending`; they do not
make the command fail and do not block downstream analyses. The machine always
leaves `scientific_status` as `not evaluated`. Only explicit human review may
judge scientific usability or failure. A report with no observed findings does
not establish equilibration, convergence, sampling adequacy, or scientific
validity.
