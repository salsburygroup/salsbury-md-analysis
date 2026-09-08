# Observable validation example

These are synthetic counterexamples, not molecular-simulation results. Both
series spend half their observations in each state. The fast series switches
15 times; the slow series switches three times. The simulation-protocol labels
are illustrative inputs for exercising the comparison contract.

From the repository root after installing the package:

```sh
salsbury-md-analysis convergence examples/observable_validation/project.json
salsbury-md-analysis compare-numerical-protocols examples/observable_validation/comparison.json
```

The first command reports occupancy, lagged behavior, and within-series
uncertainty. The second reports finite-window differences and exploratory
bootstrap intervals. Neither command establishes physical validity.

The source hashes in the JSON files match the supplied `*-values.txt` files.
See [the method and configuration guide](../../docs/OBSERVABLE_VALIDATION.md).
