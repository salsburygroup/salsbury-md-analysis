# Hash-pinned regressions

Status: **experimental**

Regression cases bind a project manifest and analysis module to the SHA-256 of
the project manifest, system manifest, and aggregate input-content signature.
They evaluate explicit JSON-path assertions with exact or absolute-tolerance
comparisons.

```bash
PYTHONPATH=src python -m salsbury_md_analysis \
  run-regression path/to/regression-case.json
```

The runner reads declared inputs and writes its report only to standard output.
Each case has `candidate`, `approved`, or `retired` status. `approved` requires
a named reviewer and decision timestamp; passing assertions never changes the
approval state or establishes scientific validity.

The repository contains only small redistributable software fixtures. Private
trajectory regressions and their raw storage locations stay in controlled
group validation records. A publication repository may include checksummed
manifests and lawful expected results for its own locked data.

An accepted real-data regression must record:

- immutable input and manifest hashes;
- exact toolkit and environment identities;
- selections, units, periodic treatment, and parameters;
- platform and external-executable provenance;
- predeclared tolerances and expected negative gates;
- a named software reviewer and scientific reviewer; and
- a clear distinction between technical execution and scientific acceptance.
