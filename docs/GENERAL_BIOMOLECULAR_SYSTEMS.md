# General biomolecular systems

Here, “generic” means that a user should not need to write project-specific
Python for a routine first analysis. You provide the structure, bond topology,
trajectories, and saved-frame interval. The software inspects their composition,
chooses the analyses that apply, and shows those choices before execution.

It recognizes common proteins, DNA, RNA, protein–nucleic-acid complexes,
equivalent oligomers, ligands/cofactors, water, and common mono- or multivalent
ions. That classification is deliberately modest: it does not decide which
residue matters, whether an ion is biologically bound, a ligand's protonation,
an element's oxidation state, or what a publication should claim.

The output uses two related labels. `macromolecular_classification` records the
shared comparison basis—protein, nucleic acid, or a protein–nucleic-acid
complex. The more specific `system_classification` also records additional
solute such as a ligand or cofactor. This lets ligand-bound and ligand-free
systems share a common macromolecular basis without pretending that their full
chemical compositions are identical.

## What you need to provide

The one-command initializer currently accepts:

- one atom-order-matched PDB reference;
- one explicit connectivity source: PSF, Amber PRMTOP/PARM7, or
  `salsbury-bonds-v1` JSON; and
- one standard, non-fixed-atom DCD per replica plus the saved-frame interval.

The PDB, connectivity, and DCD must describe the same atoms in the same order.
The initializer checks this before it creates the analysis directory.

If connectivity was not retained, `--generate-connectivity-openmm` may build a
portable bond JSON from standard OpenMM residue templates and explicit PDB
connectivity. OpenMM is needed only for that preparation step. Modified residues,
covalent ligands, and unusual cofactors should use the simulation's original
topology or reviewed OpenMM residue definitions; coordinate-distance bond guessing
is deliberately unavailable.

For a retained OpenMM `system.xml`, the system-connectivity exporter uses
the atom-order-matched OpenMM PDB topology as the default covalent graph. It
inventories System-only `HarmonicBondForce`, `CustomBondForce`, and constraint
pairs but excludes them by default. System-only force pairs can expose an
atom-order mismatch or specialized force construction; custom forces may be
restraints; constraint-only pairs may hold angles or rigid geometry. Each
category has a separately named explicit review option. A reviewed
serialized OpenMM `State` can also supply accepted coordinates for an
atom-order-matched PDB while leaving the original checkpoint untouched.

The lower-level manifest and coordinate readers also accept PDB trajectories,
one-frame GRO coordinates, and XYZ trajectories with declared units. Those
formats do not yet have the same one-command preparation interface as PDB/DCD.

Local execution is the default and does not require Slurm. Cluster execution uses
the same generated workers with a site-specific Slurm profile. See
[Local and Slurm execution](EXECUTION_ADAPTERS.md).

## What the workflow chooses automatically

| Composition | Automatic behavior | Boundary requiring review |
|---|---|---|
| Protein | Protein alignment, global solute-heavy conformation, structural metrics, interactions, SASA, and DSSP when `mkdssp` is available | Nonstandard amino acids and protonation still require correct topology |
| DNA or RNA | Canonical DNA/RNA names are recognized; intrinsic ring/stacking geometry, ion atmosphere, interactions, and global conformation are prepared when applicable | DSSR is optional and separately licensed; modified-base definitions remain reviewable |
| Protein–DNA/RNA | Adds an outcome-independent chemical-interface view and shared-basis comparison paths | Reference numbering and common-atom mapping must be scientifically meaningful |
| Equivalent oligomer | Strict topology/geometry detection may pool independently aligned member observations while preserving `member_id` and physical-frame identity | Members are not independent replicas; ambiguous equivalence fails closed |
| Ligand or cofactor | Retained in global solute-heavy analysis and complete-solute representative exports; its presence is recorded separately from the macromolecular comparison basis; polar atoms can enter generic interaction/ion target groups | Protonation, tautomer, covalency, and ligand-specific features are not guessed |
| Water and ions | Common water and ion aliases are excluded from solute PCA; ions remain in the molecular payload and are analyzed by species when present | Shells and geometric retention do not establish biological binding or oxidation state |
| Membrane, carbohydrate, or unknown polymer | Preserved as other solute for generic global metrics and exports | No specialized membrane, lipid, glycan, or polymer chemistry is inferred; provide explicit project definitions |

Shared name handling covers common CHARMM, AMBER, PDB, and GROMACS labels,
including canonical `RA/RC/RG/RU`, histidine variants `HSD/HSE/HSP`, waters such
as `TIP`, `TP3`, `OPC`, and `TIP5P`, and common ions such as Li, Na, K, Mg, Ca,
Mn, Fe, Co, Ni, Cu, Zn, Cl, Br, and iodide. Residue names route calculations;
atom identity and supplied bonds remain authoritative.

## What happens to coordinates and exported structures

Periodic production trajectories require explicit connectivity-aware make-whole
or continuous-unwrapping treatment. The reusable coordinate cache stores one
made-whole, unaligned molecular payload containing protein, nucleic acid,
hydrogens, ligands, cofactors, and ions. Water-dependent modules continue to read
the immutable solvated trajectories.

PCA or clustering features and alignment atoms are separate from exported atoms.
Representative structures are always written for enabled state analyses. Optional
state trajectories contain the complete configured molecular payload, aligned by
the protein/nucleic-acid feature structure; they are not restricted to the PCA
heavy atoms. Nearby water export is independently configured. The
`macromolecular_trace` PCA view and its trajectory exports are off by default and
can be enabled separately.

## Minimal generic run

```bash
PYTHONPATH=src python -m salsbury_md_analysis prepare-analysis \
  --pdb system.pdb --connectivity system.psf \
  --trajectory replica-1.dcd --trajectory replica-2.dcd \
  --frame-interval-ps 10 --project-id example --output example-analysis
cd example-analysis
./run-local.sh
```

For a Slurm site, copy `profiles/slurm/generic-template.json`, set only reviewed
site/account/partition/environment paths, reference it from an analysis config,
and run the generated `submit.sh`. The Salsbury-group DEAC profile is supplied as
a separate site configuration and is not used by local defaults.

Before execution, review `composition.json`, `automatic-chemical-context.json`,
`conformational-views.json`, `analysis-config.json`, `campaign-resource-plan.json`,
and `sampling-plan.json`. Every trajectory selection is a deterministic integer
stride balanced across replicas/member timelines and is reported with source and
selected frame counts.

You do not need to read every generated file before a routine first run. The
most useful sequence is:

1. Check `composition.json` for correctly recognized protein, nucleic acid,
   ligand/cofactor, water, and ion residues.
2. Check `conformational-views.json` for the intended global, interface, and
   oligomer-member views.
3. Check `campaign-resource-plan.json` for the total envelope and any methods
   that were reduced, skipped, or deferred.
4. Check `sampling-plan.json` for the exact integer stride and retained frame
   count used by each method.
5. After execution, begin with `prioritized_findings.md`, then follow its links
   into the full reports and `analysis_resource_and_frame_table.md`.

If the composition is wrong, stop before submitting the analysis. Fix the
residue naming or supply a reviewed project definition; do not reinterpret an
incorrect automatic classification after the fact.

## What still needs human judgment

A successful preparation, test, scheduler run, or technically complete report is
not scientific validation. The generic workflow deliberately reports unknown or
inapplicable chemistry instead of inventing it. A publication-specific repository
should lock the accepted suite commit, input hashes, selections, parameters,
scientific review, and interpretation without forking general reusable methods.
