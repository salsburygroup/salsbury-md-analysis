# Maintainer validation

This directory contains validation harnesses for general toolkit methods when
the authoritative fixture cannot be distributed publicly. These programs are
not user analysis commands, are not registered workflow modules, and are not
installed as console entry points.

`run_hydrogen_bond_discovery_cross_validation.py` checks automatic hydrogen-
bond chemistry and geometry against independent MDTraj and OpenMM engines on a
retained TREX control fixture. The script records expected input hashes and
writes only bounded, path-redacted evidence. Running it requires authorized
access to that immutable fixture plus MDTraj 1.11.1 and OpenMM 8.5.2.

The retained report is
[`../hydrogen_bond_discovery_cross_validation.json`](../hydrogen_bond_discovery_cross_validation.json).
It establishes scoped implementation agreement, not adequate sampling,
energetic importance, a biological mechanism, or publication readiness.

External users can run the public, dependency-light check in
[`../public/run_hydrogen_bond_synthetic_validation.py`](../public/run_hydrogen_bond_synthetic_validation.py).
The public check does not replace the real-trajectory validation.
