# Analysis environments

`environment.yml` at the repository root is the reviewed full-feature
environment specification. It includes the optional HDBSCAN implementation and
the external `mkdssp` executable. It does not bundle x3dna-dssr or OpenMM.

Create the local environment without changing the system Python:

```bash
micromamba create --prefix ./.venv --file environment.yml \
  --override-channels --channel conda-forge --strict-channel-priority
./.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

The `locks/` directory records platform-specific explicit package solutions.
The macOS lock is the environment used for local validation. The Linux lock was
resolved for a Linux 5.14/glibc 2.34 cluster platform, but it has not yet passed
an approved compute-node installation and regression. Treat it as a deployment
candidate until the same software and trajectory checks pass there.

`nucleic_acid_structure` requires a separately installed x3dna-dssr executable
whose resolved path and version are recorded in the project output. Obtain and
license that program through its provider; do not copy it into this BSD source
repository. `openmm-connectivity` is needed only to export portable bond JSON
from an OpenMM build and is exposed as a separate package extra. All intrinsic
nucleic-acid and ion-geometry methods run with the base NumPy/SciPy package.

Environment reproducibility does not establish scientific validity. Published
work must additionally record the suite commit, project lock, module settings,
input hashes, executable versions, and accepted regression criteria.
