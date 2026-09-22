# Secondary structure

Module ID: `secondary_structure`  
CLI: `salsbury-md-analysis secondary-structure PROJECT.json`

The module executes the declared `mkdssp` binary on temporary per-frame PDB
files. It records the resolved executable, version output, command convention,
input normalization, evaluated frame identities, and residue-level assignment
populations. It does not substitute a heuristic classifier when DSSP is
unavailable.

The DSSP assignment alphabet is versioned scientific data. In particular,
DSSP 4.6 can emit `P` for polyproline-II helix. The toolkit retains `P`; it does
not silently convert it to coil. A comparison with an older implementation
must declare its mapping. In the independent trajectory check, the retained
DSSP 4.6 populations agreed exactly with MDTraj after explicitly mapping `P` to
the older coil category; before that version mapping, 90.56% of residue
populations matched exactly.

Periodic production trajectories require connectivity-aware `make_whole` or
`unwrap_continuous` preprocessing. Pooled populations also require
replica-sensitive convergence and uncertainty analysis. Agreement between DSSP
implementations does not establish that a structural transition is converged,
significant, or mechanistically important.

## MD naming compatibility and residue coverage

The adapter writes a temporary protein-only PDB with normalized serials,
residue numbering, and element fields. Non-coordinate comments are omitted.
Histidine aliases `HSD/HSE/HSP` and `HID/HIE/HIP` become `HIS` only in that
temporary input. Paired terminal `OT1/OT2` names become `O/OXT`; incomplete
pairs or collisions with existing `O/OXT` fail with an explicit error. Original
coordinates, topology files, protonation, atom order, and residue identities
are preserved. Reports retain the name mappings and original residue names.

Terminal caps `ACE/NME` are excluded from DSSP input and recorded separately;
they are not amino-acid residues with expected assignments. Every evaluated
frame must return exactly one assignment for each normalized protein residue. Missing, duplicate, or unexpected residues fail the module
before populations are accumulated. Missing-residue errors identify the
original residues and source frame. Inspect backbone completeness, naming, and
geometry instead of interpreting a partial table as complete secondary
structure. This also applies to incomplete or modified residues that DSSP
cannot assign; the adapter does not invent assignments.

The NEMO fixture exposed both naming cases: DSSP 4.6.1 otherwise omitted HSD22
and terminal GLU28, returning 26 of 28 residues despite a successful exit.
Name-only normalization restores all 28 on the reference structure. Its raw
PDB also contains a nonstandard `REMARK` record that direct DSSP invocation
rejects; the adapter's temporary PDB excludes it. The source fixture remains
unchanged.
