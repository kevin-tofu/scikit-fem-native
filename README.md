# skfem-native

[![PyPI](https://img.shields.io/pypi/v/skfem-native.svg)](https://pypi.org/project/skfem-native/)
[![Python](https://img.shields.io/pypi/pyversions/skfem-native.svg)](https://pypi.org/project/skfem-native/)
[![CI](https://github.com/kevin-tofu/skfem-native/actions/workflows/ci.yml/badge.svg)](https://github.com/kevin-tofu/skfem-native/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/skfem-native/badge/?version=latest)](https://skfem-native.readthedocs.io/en/latest/)
[![License: LGPL-3.0](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](https://github.com/kevin-tofu/skfem-native/blob/main/LICENSE)

`skfem-native` provides `skfemntv`, a native numerical assembly backend with a
scikit-fem-style Python API.  It keeps finite-element formulations in readable
Python while accelerating reusable assembly, geometry, and sparse-scatter
kernels in native code.

## Motivation

The project exists to improve the performance of Python assembly workflows
represented by scikit-fem without replacing their Python programming model.
Weak forms, nonlinear constitutive updates, contact formulations, and solver
policy remain ordinary Python components; stable and reusable numerical kernels
are moved to native code where doing so provides a practical benefit.

This boundary is becoming more important in the era of AI-assisted coding.
Application-specific nonlinear formulations can now be developed, reviewed,
and revised rapidly when they remain small, explicit Python components.
Libraries such as scikit-fem therefore become more valuable, not less: they
provide a transparent mathematical vocabulary while avoiding the cost of
embedding every new formulation in a monolithic compiled solver.

`skfem-native` aims to combine that flexibility with faster assembly.  It is a
selectable backend for Python finite-element applications, not a separate owner
of their material models or physical formulations.

## Installation

```bash
python -m pip install skfem-native
```

The distribution is named `skfem-native`; import it as `skfemntv`.

## Quick start

```python
import skfemntv
from skfemntv.helpers import dot, grad

mesh = skfemntv.MeshTet()
basis = skfemntv.Basis(mesh, skfemntv.ElementTetP1())

@skfemntv.BilinearForm
def diffusion(u, v, w):
    return dot(grad(u), grad(v))

matrix = skfemntv.asm(diffusion, basis)
```

Applications can select `skfem.asm` or `skfemntv.asm` at their backend
boundary while preserving the weak-form call site.  `skfemntv` is not yet a
complete replacement for every scikit-fem feature, so backend selection should
remain explicit.

For the supported mesh, element, basis, and assembly subset, the compatibility
target is stronger: changing

```python
import skfem
```

to

```python
import skfemntv as skfem
```

should preserve public DOF ordering and numerical results.  Higher-order DOFs
are numbered by topological entity, following scikit-fem: vertices first,
shared edges or facets next, and element interiors last.  This makes assembled
vectors and matrices directly interchangeable without a coordinate
permutation.

Solver conveniences such as `solve`, `enforce`, and `penalize` are outside the
native assembly backend.  Applications may continue to use scikit-fem or SciPy
for these operations.

## Capabilities

- functional, linear, bilinear, and cross-bilinear assembly
- meshes, elements, bases, tabulation, and quadrature
- caller-supplied tensor coefficient assembly
- threaded native kernels and sparse scatter
- cut-cell and implicit-interface quadrature
- interface supermeshes and contact-facet search

## Assembly performance

The following reference run compares warm-cache Poisson P1 matrix and
right-hand-side assembly using identical meshes and spaces.  Results depend on
hardware, thread affinity, and package versions; they are an example rather
than a universal performance guarantee.

![Poisson assembly scaling](https://raw.githubusercontent.com/kevin-tofu/skfem-native/main/benchmarks/compare-with-skfem/results/poisson-linux-x86_64.png)

See the [benchmark methodology](benchmarks/compare-with-skfem/README.md) and
[recorded environment and values](benchmarks/compare-with-skfem/results/poisson-linux-x86_64.md)
for details.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the motivation, architecture boundary,
and development principles.

## Documentation

Build the Sphinx documentation locally:

```bash
python -m pip install -e '.[docs]'
sphinx-build -M html docs docs/_build
```

Open `docs/_build/html/index.html` after the build completes.

## Development

```bash
python -m pip install -e '.[test]'
pytest -q
```
