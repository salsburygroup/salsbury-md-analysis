# Replica-aware RMSF and uncertainty

Status: **experimental**

This stage-4 module fits every evaluated frame to an explicit reference and
calculates atomic root-mean-square fluctuation on one atom basis shared by all
declared topologies. It keeps frame-pooled, per-replica, and time-block
estimates separate.

```bash
PYTHONPATH=src python3 -m salsbury_md_analysis \
  rmsf path/to/project.json --hash-content
```

The command is read-only and emits JSON to standard output.

After retaining that JSON report, export a residue-colored PDB and VMD view:

```bash
PYTHONPATH=src python -m salsbury_md_analysis \
  export-rmsf-visualization pooled-rmsf.json SYSTEM_ID results/rmsf
```

The exporter writes `results/rmsf.rmsf_bfactor.pdb` and
`results/rmsf.rmsf_cartoon.vmd.tcl`. It refuses to overwrite either output
unless `--overwrite` is explicit. The default maps the mean RMSF of analyzed
atoms in each residue onto all atoms of that residue, which lets VMD render a
`NewCartoon` colored by `Beta`. The numeric field remains RMSF in angstrom; it
is not converted into a crystallographic Debye-Waller B factor.

## Definition

For each fitted atom position, the reported RMSF is

`sqrt(<|r - <r>|^2>)`.

The trajectory frames are treated as the population being summarized, so the
positional moment uses denominator `N`, not the sample denominator `N - 1`.
This matches the common definition of RMSF as the standard deviation of fitted
atomic positions described by the
[GROMACS `gmx rmsf` documentation](https://manual.gromacs.org/current/onlinehelp/gmx-rmsf.html).

## Required project contract

```json
{
  "reference_structure": "reference.pdb",
  "common_atom_policy": "strict",
  "periodic_coordinate_policy": "reject",
  "definitions": {
    "pooled_rmsf": {
      "alignment_selection": "alignment",
      "analysis_selection": "analysis",
      "minimum_reference_coverage": 1.0,
      "frame_stride": 1,
      "time_block_size_frames": 1000,
      "include_partial_final_block": false,
      "minimum_replicas_for_uncertainty": 2
    }
  }
}
```

The alignment and analysis selections are resolved separately, but each uses
one reference-ordered intersection across all replica topologies. The coverage
gate therefore applies to the genuinely shared basis, not to a separate basis
for each variant.

## Distinct estimators

The report deliberately does not collapse the following quantities:

- `frame_pooled_rmsf_angstrom` uses all evaluated frames and therefore gives
  replicas with more frames more weight;
- each replica has its own RMSF profile;
- `replica_rmsf_summary_angstrom` gives each available replica one RMSF
  estimate and reports their mean, sample SD, and SEM;
- time-block RMSF profiles reset at segment boundaries and are summarized
  separately.

Sample SD and SEM are `null` for a single estimate. The suite does not turn one
replica into uncertainty by treating its frames as independent replicates.

Blocks are defined by evaluated frame count, and every block also reports its
manifest-declared physical start and end time. A partial final block is either
included with `complete: false` or discarded with an explicit frame count.

## Scientific limits

- Frame-pooled and replica-balanced RMSF are different estimators.
- Time blocks are convergence diagnostics and are not automatically
  statistically independent.
- Periodic coordinates fail closed under `reject`; `allow_wrapped_diagnostic`
  remains diagnostic. Production runs use connectivity-aware `make_whole` or
  `unwrap_continuous` preprocessing.
- The scientific report remains per atom; the visualization exporter provides
  explicit residue-mean or atom-level B-factor-field mapping for PDB references.
- The synthetic fixtures validate software arithmetic only. The current
  real-project candidate covers structural QC, not RMSF, and is not approved.
- RMSF can describe structural variability but does not by itself establish
  equilibration, convergence, functional importance, mechanism, or scientific
  validity.
