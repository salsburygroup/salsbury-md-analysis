# Contributing

The goal is to keep one comprehensive general toolkit without turning it into a
collection of paper-specific scripts. A useful contribution should make a
method easier to understand, test, reuse, and review on more than one project.

Do not commit trajectories, restricted structures, credentials, private
collaborator material, cluster paths, installed environments, or source with an
unclear license. These boundaries protect collaborators and keep a future
community release portable.

## Adding or changing an analysis

A contribution must include:

1. a scientific definition and interpretation limits;
2. explicit inputs, selections, units, parameters, outputs, and failure states;
3. a review of existing modules showing why the change is not redundant;
4. implementation tests, negative tests, and a small lawful fixture;
5. an appropriate scientific-regression plan;
6. registry and `standard_md_v1` updates when the method belongs in the
   standard workflow;
7. generated reference documentation and any environment changes; and
8. human scientific and software review.

New methods begin as `experimental`. Only independently reviewed and validated
functionality may be marked `supported`.

Project-specific code belongs in a separate publication repository. Generalize
it here only after removing target names, storage assumptions, plotting choices,
and paper-specific parameters.
