# Compiled analysis context

Status: **experimental**

The compiled context turns the project and system manifests into one stable,
machine-readable contract before an analysis backend reads coordinates. It
keeps system, replica, and continuous-segment identities explicit and requires
the project to declare coordinate units, time units, named selections, and a
periodic-coordinate policy. Every segment supplies physical first-frame time
and frame interval.

```bash
PYTHONPATH=src python3 -m salsbury_md_analysis \
  compile-context path/to/project.json --hash-content
```

The command is read-only and writes JSON only to standard output. It resolves
the system manifest relative to the project manifest, validates all input paths,
and returns:

- the normalized semantic contract and its stable SHA-256 signature;
- system, replica, and segment identities in declared order;
- the reference-system decision;
- named `alignment`, `analysis`, and, when requested, `mapping` selections;
- explicit coordinate and time units;
- normalized per-segment physical timing and periodic-coordinate policy;
- a deterministic input inventory;
- optional content hashes and an input-content signature;
- separate technical and scientific statuses.

## Selection rules

Each named selection must use exactly one portable form:

```json
{
  "selections": {
    "alignment": {"preset": "backbone"},
    "analysis": {"preset": "heavy"},
    "custom_site": {"atom_names": ["CA", "CB", "CG"]}
  }
}
```

Portable presets are `all`, `backbone`, `complex_trace`,
`macromolecular_backbone`, `heavy`, and `solute_heavy`. An `atom_names` rule is an
exact set of names. Backend-specific expressions are deliberately excluded from
this shared layer because the same expression can mean different things in
different trajectory libraries.

## Fail-closed behavior

Compilation fails when units, segment timing, periodic-coordinate policy, or
required semantic selections are absent, input files do not exist, an
explicitly named reference system is absent, or a
multi-system project omits `reference_system`. A one-system project may infer
its sole system as the reference, but the report records that inference as a
warning.

## Limits

The context does not load coordinates, interpret string-valued stride
expressions, align structures, or judge chemistry, sampling, convergence, or
scientific validity. Content hashes record exact bytes, not scientific meaning.
