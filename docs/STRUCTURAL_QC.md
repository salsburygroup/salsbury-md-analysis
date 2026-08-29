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

The command never modifies trajectory, topology, connectivity, or manifest
inputs. It emits JSON to standard output and, when checkpointing is enabled,
writes restart records under the declared analysis output directory. It
preserves system, replica, segment, and manifest-declared physical-time
identities.

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
      "frame_stride": 1,
      "checkpointing": {
        "enabled": true,
        "within_segment_interval_seconds": 7200.0
      }
    }
  }
}
```

No scientific threshold is silently chosen by the command. The displacement
gate is optional and applies after the declared periodic preprocessing policy;
it should remain `null` under wrapped diagnostic execution when coordinates can
cross a boundary. `frame_stride` defines the exact integer-stride sample used by
the coordinate and chemical gates. DCD preflight still validates every record
envelope, including records whose coordinate payload is not decoded.

## Checkpoint and restart behavior

Checkpointing is enabled by default. The module writes one atomic, compressed
checkpoint whenever it completes a trajectory segment. If a single segment is
still running after 7,200 seconds (2 hours), it also writes an in-progress
checkpoint and refreshes it after each additional 2 hours of work in that same
segment. A segment that finishes in less than 2 hours receives only its normal
completion checkpoint.

Run the same command again after an interruption. Completed segments are loaded
without reopening their trajectories, and an interrupted segment resumes after
its last recorded frame. Continuous periodic reconstruction state is restored
with the segment accumulators, so resume does not create a new unwrapping
boundary.

Checkpoints are stored at
`ANALYSIS_OUTPUT_ROOT/structural-qc/checkpoints/CHECKPOINT_ID/`. Each record is
content-hashed and tied to the project manifest, system manifest, declared input
signature, and structural-QC implementation. A damaged or mismatched checkpoint
fails closed; changed inputs or code receive a different checkpoint identity.
Set `checkpointing.enabled` to `false` when restart files are not wanted. The
interval remains explicit in the configuration even when checkpointing is off.

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
