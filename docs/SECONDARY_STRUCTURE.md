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
