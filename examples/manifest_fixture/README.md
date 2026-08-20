# Manifest validation fixture

This is an original, synthetic one-atom fixture for learning and software tests.
It is not a molecular simulation and must not be used to validate an analysis
algorithm.

From the repository root:

```bash
PYTHONPATH=src python3 -m salsbury_md_analysis \
  validate-manifest project examples/manifest_fixture/project.json --check-paths

PYTHONPATH=src python3 -m salsbury_md_analysis \
  validate-manifest system examples/manifest_fixture/system.json --check-paths

PYTHONPATH=src python3 -m salsbury_md_analysis \
  inventory-system examples/manifest_fixture/system.json --hash-content

PYTHONPATH=src python3 -m salsbury_md_analysis \
  compile-context examples/manifest_fixture/project.json --hash-content

PYTHONPATH=src python3 -m salsbury_md_analysis \
  preflight-system examples/manifest_fixture/system.json --hash-content

PYTHONPATH=src python3 -m salsbury_md_analysis \
  structural-qc examples/manifest_fixture/project.json --hash-content

PYTHONPATH=src python3 -m salsbury_md_analysis \
  rmsd-rg examples/manifest_fixture/project.json --hash-content

PYTHONPATH=src python3 -m salsbury_md_analysis \
  rmsf examples/manifest_fixture/project.json --hash-content

PYTHONPATH=src python3 -m salsbury_md_analysis \
  dccm examples/manifest_fixture/project.json --hash-content

PYTHONPATH=src python3 -m salsbury_md_analysis map-common-atoms \
  examples/manifest_fixture/data/one-atom.pdb \
  examples/manifest_fixture/data/one-atom-variant.pdb \
  --policy position --selection heavy \
  --minimum-reference-coverage 1.0 --hash-content

PYTHONPATH=src python3 -m salsbury_md_analysis \
  validate-manifest output examples/manifest_fixture/output-manifest.json --check-paths
```

`analysis-lock.template.json` is intentionally not a valid publication lock.
Replace every angle-bracket placeholder with an exact commit, environment
identity, hash, owner, and reviewed status before validation.

The project also demonstrates explicit coordinate/time units, portable named
selections, declared structural-QC gates, and explicit
reference/mapping/RMSD-Rg/RMSF/DCCM settings. Its one-atom outputs test
execution and deliberate zero-variance handling only; they are not a scientific
validation dataset. The absolute protected path in `project.json` is a
conspicuous non-existent teaching placeholder. Real projects must list their
actual protected data roots.
