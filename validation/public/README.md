# Public validation

The scripts here are small, distributable validation checks that anyone can
run from a source checkout. They use only the package's declared base runtime
dependencies and public synthetic inputs.

Run the hydrogen-bond check from the repository root:

```bash
PYTHONPATH=src python validation/public/run_hydrogen_bond_synthetic_validation.py
```

This check independently recomputes simple donor--acceptor distances and
donor--hydrogen--acceptor angles, tests known present and absent geometries,
and confirms template-based donor/acceptor discovery. It is intentionally
small and deterministic. It complements, but does not replace, the retained
real-trajectory maintainer validation.
