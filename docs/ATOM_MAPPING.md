# Common-atom mapping

Status: **experimental**

The common-atom mapper creates an explicit, deterministic index correspondence
between one PDB or GRO reference topology and one or more target topologies.
Every invocation requires a mapping policy, selection, and minimum reference
coverage:

```bash
PYTHONPATH=src python3 -m salsbury_md_analysis map-common-atoms \
  reference.pdb variant-a.pdb variant-b.pdb \
  --policy position \
  --selection heavy \
  --minimum-reference-coverage 0.95 \
  --hash-content
```

The JSON report contains zero-based indices, original serials, the common map,
full reference/target exclusions, per-topology coverage, optional file hashes,
and a stable SHA-256 signature of the policy, selection, and mapping.

## Policies

- `strict`: chain, residue number, insertion code, residue name, atom name, and
  alternate-location identifier must all match.
- `position`: chain, residue number, insertion code, atom name, and
  alternate-location identifier must match; residue name is deliberately
  ignored. Every mapped residue-name substitution is reported as a warning.

`position` is intended for already validated variants that preserve numbering.
It does not infer sequence alignment or prove homology.

## Selections

- `all`: every ATOM/HETATM or GRO atom.
- `backbone`: non-solvent/non-ion atoms named N, CA, C, or O. This avoids
  accidentally treating water oxygens named `O` as protein backbone.
- `complex_trace`: non-solvent/non-ion atoms named CA or C1', providing a
  compact protein-plus-nucleic-acid trace.
- `macromolecular_backbone`: the protein backbone names plus the standard
  nucleic-acid phosphate/sugar backbone names, excluding common solvent and
  monatomic-ion residue names.
- `heavy`: excludes atoms declared or heuristically named as hydrogen while
  retaining solvent and ions.
- `solute_heavy`: applies the heavy-atom rule and also excludes common solvent
  and monatomic-ion residue names.

The heavy-atom element inference and solvent/ion residue-name list are selection
conveniences, not chemical typing. Nonstandard solvent or ion residue names
must be reviewed in the emitted atom map before scientific use.

Project selections may also lock exact residue identities with `residue_keys`
records containing chain ID, residue number, and insertion code, plus an
explicit `heavy_only` flag. The automatic chemical-interface planner uses this
contract to retain complete reference residues selected by prespecified
chemistry and distance rules. It never selects residues from PCA scores,
cluster membership, occupancy differences, or another trajectory outcome.

## Fail-closed behavior

Mapping fails when:

- a topology is malformed or is not PDB/GRO;
- a selection produces no atoms;
- any selected identity is duplicated under the chosen policy;
- no selected identity is common to every topology;
- common coverage is below the explicit threshold.

GRO lacks chain and insertion-code fields, and its residue numbers can wrap.
Any resulting duplicate identity fails rather than being matched by atom order.

## Scientific boundary

A technically complete map does not establish that whole-protein PCA, DCCM,
RMSF differences, or another common-basis comparison is scientifically valid.
Weakly homologous systems may require a prespecified domain-local map or may be
unsuitable for common-basis comparison altogether.
