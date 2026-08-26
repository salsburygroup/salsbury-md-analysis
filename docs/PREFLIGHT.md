# Topology and trajectory preflight

Status: **experimental**

The preflight command performs read-only structural checks on every topology
and trajectory named in a validated system manifest:

```bash
PYTHONPATH=src python3 -m salsbury_md_analysis \
  preflight-system path/to/system.json --hash-content
```

It prints JSON and exits with status 0 when no technical errors are found. A
malformed manifest, unsupported format, malformed file, atom-count mismatch, or
DCD continuity mismatch exits with status 2. Warnings do not change a complete
technical status into a failure.

## Supported metadata probes

| Format | Role | Current checks |
|---|---|---|
| PDB | topology or trajectory | ATOM/HETATM count, consistent MODEL sizes, MODEL/ENDMDL structure, CRYST1 presence |
| GRO | topology or one-frame trajectory | declared atom count, complete record count, numeric 3- or 9-value box |
| PSF | topology | positive `!NATOM` declaration |
| PRMTOP/PARM7 | topology | positive Amber `POINTERS`/`NATOM` value |
| XYZ | trajectory | every frame, consistent positive atom count, complete numeric coordinate records |
| DCD | trajectory | byte order, Fortran record markers, CORD/VELD signature, declared frames, start step, save interval, title record, atom count |

Topology and trajectory atom counts must agree. For consecutive DCD segments
marked `continuous_with_previous`, the declared next starting step must equal:

```text
previous_start + previous_declared_frames * previous_save_interval
```

Every segment also declares `first_frame_time`, `frame_interval`, and `unit`.
For all probed formats, a continuous segment must begin at:

```text
previous_first_frame_time + previous_frame_count * previous_frame_interval
```

Units are normalized before comparison. This contract assumes no duplicated
boundary frame. DCD step continuity is checked independently when both adjacent
segments are DCD.

Replica `force_field_parameters` inputs—CHARMM parameter/stream files,
serialized OpenMM System XML, or a declared GROMACS TPR—are included in the
deterministic inventory and content hash. General preflight records their
format, paths, sizes, and hashes; the energetic-network preparation probe then
performs the source-specific semantic parsing and availability decision.

## Deliberate limits

- This preflight command does not scan DCD coordinate records. Its frame count
  is the header declaration, not an observed count. The separate experimental
  `structural-qc` command streams and verifies a restricted standard 32-bit DCD
  subset.
- XTC, TRR, NetCDF, MDTraj HDF5, and other binary trajectories fail as
  unsupported. They are not inferred from file size or extension alone.
- PDB/GRO/PSF/PRMTOP probes do not establish force-field correctness,
  connectivity, chemistry, residue identity, chirality, or covalent state.
- XYZ has no standard time or periodic-cell semantics.
- Preflight says nothing about equilibration, convergence, adequate sampling,
  statistical independence, population meaning, or biological interpretation.

The coordinate-reader backend can extend trajectory coverage without changing
this deliberately metadata-only, fail-closed report contract.
