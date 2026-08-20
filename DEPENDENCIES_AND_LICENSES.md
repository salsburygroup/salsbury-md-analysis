# Dependencies and licenses

This page separates what is required to run Salsbury MD Analysis from optional
scientific tools and packages used only to validate the implementation. The
toolkit does not copy or redistribute these third-party projects. Each remains
subject to its own license.

## Salsbury MD Analysis licenses

- Source code, including code embedded in the documentation: BSD-3-Clause;
  see [`LICENSE`](LICENSE).
- Original documentation and teaching material: CC BY 4.0; see
  [`LICENSE-DOCS.md`](LICENSE-DOCS.md).

## Required runtime

| Dependency | Supported version | Why it is needed | License |
| --- | --- | --- | --- |
| Python | 3.10 or newer | Runs the package and command-line interface. | Python Software Foundation License Version 2, with separately licensed components documented by Python. |
| NumPy | `>=2.0,<3` | Arrays, linear algebra, coordinate calculations, and numerical summaries. | The main project is BSD-3-Clause. Installed distributions also identify separately licensed 0BSD, MIT, Zlib, and CC0 components; binary builds can include additional runtime libraries and their notices. |
| SciPy | `>=1.12,<2` | Statistics, spatial searches, filtering, optimization, and scientific numerical routines. | BSD-3-Clause. Binary builds can depend on separately licensed numerical runtime libraries. |

Official license sources: [Python](https://github.com/python/cpython/blob/main/LICENSE),
[NumPy](https://github.com/numpy/numpy/blob/main/pyproject.toml), and
[SciPy](https://github.com/scipy/scipy/blob/main/LICENSE.txt).

## Full-workflow and optional scientific dependencies

| Dependency | Supported/reviewed version | When it is needed | License and redistribution boundary |
| --- | --- | --- | --- |
| scikit-learn | `>=1.6,<2`; reviewed at 1.9.0 | Clustering and other modules implemented through scikit-learn. | BSD-3-Clause; installed dependency, not copied into this repository. |
| HDBSCAN | `>=0.8.44,<0.9`; reviewed at 0.8.44 | The optional HDBSCAN clustering method. HDBSCAN is off by default and can be enabled explicitly. | BSD-3-Clause; installed dependency, not copied into this repository. |
| DSSP (`mkdssp`) | reviewed at 4.6.1 | Protein secondary-structure assignment when that module is enabled. | BSD-2-Clause; separately installed executable. |
| OpenMM | `>=8.5,<8.6`; optional | Connectivity preparation when no reviewed PSF, PRMTOP/PARM7, or portable bond JSON is available. Later analysis does not require OpenMM once the bond graph has been prepared. | OpenMM contains components under MIT, LGPL, and other licenses described in its license inventory. It is not bundled here. |
| X3DNA-DSSR | provider-managed version; optional | The external DSSR-backed nucleic-acid structure module. Intrinsic nucleic-acid geometry and ion analysis do not require it. | Separately licensed by Columbia University and available free of charge for qualifying academic use. It is not open-source software and must not be copied into this repository or redistributed with the package. |

Official license sources: [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING),
[HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan/blob/master/LICENSE),
[DSSP](https://github.com/PDB-REDO/dssp/blob/trunk/LICENSE),
[OpenMM](https://github.com/openmm/openmm/blob/master/docs-source/licenses/Licenses.txt),
and [X3DNA-DSSR](https://x3dna.org/about/about-3dna-dssr).

## Installation and build tools

| Tool | Why it appears in the reviewed environment | License |
| --- | --- | --- |
| setuptools | Build backend; `pyproject.toml` requires setuptools 42 or newer. | MIT |
| pip | Installs the package and its dependencies. | MIT, with license notices for its vendored components. |
| wheel | Builds or inspects wheel artifacts; it is not a scientific runtime dependency. | MIT |

Official license sources: [setuptools](https://github.com/pypa/setuptools/blob/main/LICENSE),
[pip](https://github.com/pypa/pip/blob/main/pyproject.toml), and
[wheel](https://github.com/pypa/wheel/blob/main/LICENSE.txt).

Conda or micromamba is a supported way to create the reviewed environment, but
neither is required to run the package. A standard Python virtual environment
can be used instead.

## Validation-only references

The ordinary test suite uses the Python standard-library `unittest` framework
and the selected package dependencies. It does not require MDTraj.

`validation/maintainer/run_hydrogen_bond_discovery_cross_validation.py` is a maintainer
validation program, not a user analysis command and not part of the generic
workflow. It uses MDTraj 1.11.1 as an independent distance/angle and topology
reference and OpenMM 8.5.2 as an independent force-field/template reference.
Its purpose is to detect shared implementation mistakes in automatic
donor/acceptor discovery and hydrogen-bond geometry. MDTraj is licensed under
LGPL-2.1-or-later; OpenMM has the mixed license inventory described above.
The retained bounded result is
[`validation/hydrogen_bond_discovery_cross_validation.json`](validation/hydrogen_bond_discovery_cross_validation.json).

This kind of independent cross-validation is normal and desirable when a
scientific package implements its own numerical or chemical logic. Users do
not need to repeat it for routine analysis. Maintainers should rerun it when
the relevant chemistry templates, geometry definitions, coordinate handling,
or comparison engine changes.

Anyone can run the smaller public synthetic check at
`validation/public/run_hydrogen_bond_synthetic_validation.py`. It uses only the
base runtime, recomputes the simple reference geometry independently, and does
not require access to retained TREX inputs. It complements rather than replaces
the real-trajectory maintainer validation.

MDTraj license source: [MDTraj](https://github.com/mdtraj/mdtraj).

## Transitive dependencies and redistribution

The platform lock files under `environments/locks/` record exact resolved
packages, including transitive numerical and operating-system libraries. Those
locks are reproducibility records, not a complete third-party license report.
Package managers install those components under their own licenses.

Publishing this source package does not redistribute its separately installed
dependencies. If a future release distributes a container, bundled Conda
environment, offline wheelhouse, or executable application, generate and ship
an artifact-specific software bill of materials and third-party license notice
from the exact resolved artifacts. Do not bundle X3DNA-DSSR without an express
redistribution license from its provider.
