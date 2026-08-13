# skfem-native

### Native assembly for scikit-fem-style Python workflows

[![PyPI](https://img.shields.io/pypi/v/skfem-native.svg)](https://pypi.org/project/skfem-native/)
[![Python](https://img.shields.io/pypi/pyversions/skfem-native.svg)](https://pypi.org/project/skfem-native/)
[![CI](https://github.com/kevin-tofu/skfem-native/actions/workflows/ci.yml/badge.svg)](https://github.com/kevin-tofu/skfem-native/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/skfem-native/badge/?version=latest)](https://skfem-native.readthedocs.io/en/latest/)
[![License: LGPL-3.0](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](https://github.com/kevin-tofu/skfem-native/blob/main/LICENSE)

**skfem-native** is an independent native assembly backend for
[scikit-fem](https://github.com/kinnala/scikit-fem)-style Python workflows.  It
is distributed on PyPI as `skfem-native` and imported as `skfemntv`; it is not
an official scikit-fem distribution.  It keeps finite-element formulations in
readable Python while accelerating reusable assembly, geometry, and
sparse-scatter kernels in native code.

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

## Built on the work of scikit-fem

This project exists because
[scikit-fem](https://github.com/kinnala/scikit-fem) demonstrated how effective
a finite-element library can be when mathematical forms remain concise,
readable Python.  scikit-fem is implemented in pure Python, which makes its
environment straightforward to install, inspect, and reproduce across
development machines, notebooks, and CI systems.  Its small and composable API
makes the connection between a weak formulation and the assembled system
unusually clear.  That transparency is valuable for teaching and research, but
also for production engineering: new elements, multiphysics terms, nonlinear
material models, and contact formulations can be inspected and changed without
hiding the governing equations behind a large generated or compiled framework.
See the [scikit-fem documentation](https://scikit-fem.readthedocs.io/) for its
installation guide, examples, and API reference.

scikit-fem also provides the practical foundation on which `skfem-native` is
built: its public abstractions, element conventions, DOF ordering, examples,
and extensive body of implementation work define the compatibility target of
this project.  `skfem-native` does not seek to replace that contribution.  It
explores how selected assembly kernels can be accelerated while preserving the
Python-first model that makes scikit-fem useful.

We are grateful to the scikit-fem authors and contributors.  The usefulness of
this backend is a direct consequence of the careful design and sustained work
of that community.

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

Scalar anisotropic diffusion uses the same readable helper syntax as
scikit-fem:

```python
from skfemntv.helpers import dot, grad, mul

@skfemntv.BilinearForm
def anisotropic_diffusion(u, v, w):
    return dot(mul(w.diffusion, grad(u)), grad(v))

A = np.array([[2.0, 0.4], [0.1, 0.8]])
matrix = skfemntv.asm(anisotropic_diffusion, basis, diffusion=A)
```

The recommended public layout is the scikit-fem component-first form
`(dim, dim, entities, quadrature)`.  A constant `(dim, dim)` tensor is also
accepted and broadcast over all integration points.  The entity-first native
layout `(entities, quadrature, dim, dim)` remains accepted for low-level
compatibility, but is not the preferred user-facing representation.  The
native kernel uses that internal layout so each quadrature-local tensor is
contiguous in memory.  Nonsymmetric tensors are retained rather than silently
symmetrized.  This first typed operation targets scalar H1 fields;
vector-field fourth-order constitutive tensors remain separate work.

Coefficient components follow scikit-fem's component-first convention as
well: `w.material[0]` selects axis 0.  Multiple named fields can be used in a
single typed form, for example:

```python
@skfemntv.BilinearForm
def material_form(u, v, w):
    return (
        w.material[0] * dot(u, v)
        + w.material[1] * ddot(grad(u), grad(v))
        + dot(mul(w.diffusion, grad(u)), grad(v))
    )
```

Component indices must be integers.  Missing fields and out-of-range
components are reported with the coefficient name before assembly.

The mathematical compatibility boundary is also available at runtime:

```python
skfemntv.supports("space.h1")       # True
skfemntv.supports("space.hcurl")    # False (declared as planned)
skfemntv.require_capability("space.hcurl")  # precise capability error
```

Use `skfemntv.capabilities()` for the complete machine-readable registry.
Experimental capabilities require `include_experimental=True`; solver policy
is explicitly marked external.

### Experimental H(curl) simplex slices

The lowest-order first-family Nédélec element is available as a deliberately
small experimental API for affine triangular meshes:

```python
import numpy as np
from scipy.sparse.linalg import spsolve
import skfemntv

mesh = skfemntv.MeshTri.init_tensor(
    np.linspace(0.0, 1.0, 9), np.linspace(0.0, 1.0, 7)
)
basis = skfemntv.AffineTriN1Basis(mesh, intorder=3)
assembler = skfemntv.TriN1Assembler(basis)
linear_assembler = skfemntv.TriN1LinearAssembler(basis)
matrix = assembler.assemble_maxwell(
    mass_coefficient=1.0, curl_coefficient=0.05
).copy()

boundary = basis.boundary_dofs()
free = np.setdiff1d(np.arange(basis.N), boundary)
load = linear_assembler.assemble_vector_load(
    lambda x: np.array((0.0 * x[0], np.sin(np.pi * x[0])))
).copy()
solution = np.zeros(basis.N)
solution[free] = spsolve(matrix[free][:, free], load[free])
```

Analytic vector fields can be interpolated by their globally oriented edge
moments and evaluated at basis quadrature points:

```python
def field(x):
    return np.array((0.0 * x[0], x[0] * x[1]))

edge_values = basis.interpolate_edge_moments(field)
values = basis.evaluate(edge_values)       # (component, cell, quadrature)
curls = basis.evaluate_curl(edge_values)   # (cell, quadrature)
points = basis.global_coordinates          # (component, cell, quadrature)
```

The tetrahedral counterpart uses the same explicit workflow:

```python
axis = np.linspace(0.0, 1.0, 5)
mesh = skfemntv.MeshTet.init_tensor(axis, axis, axis)
basis = skfemntv.AffineTetN1Basis(mesh, intorder=3)
assembler = skfemntv.TetN1Assembler(basis)
linear_assembler = skfemntv.TetN1LinearAssembler(basis)

# TetN1 integration is cell-parallel; applications retain thread ownership.
with skfemntv.thread_limit(4):
    matrix = assembler.assemble_maxwell(
        mass_coefficient=1.0, curl_coefficient=0.05
    ).copy()

load = linear_assembler.assemble_vector_load(
    lambda x: np.array((1.0 + 0.0 * x[0], x[0], -x[1]))
).copy()
boundary = basis.boundary_dofs()
```

TetN1 values and vector curls both use
`(basis, component, cell, quadrature)` publicly; evaluated fields and curls
use `(component, cell, quadrature)`.  The complete runnable example is
`examples/hcurl_tet_n1_maxwell.py`.

These slices include oriented global edge numbering, affine covariant Piola
mapping, mass/curl-curl/Maxwell CSR assembly, coefficient fields, memory
preflight, and boundary-edge selection.  They support only affine triangles and
tetrahedra with one lowest-order tangential moment per edge.  Neither basis is
accepted by the general `Basis` or `asm`; curved mappings, higher order, and
solver policy are not provided.  Both bases support edge-moment interpolation
and quadrature-point evaluation.  The triangle example is
`examples/hcurl_tri_n1_maxwell.py`.

`TriN1Assembler` uses NumPy/SciPy integration.  `TetN1Assembler` uses a fused
cell-parallel C++ integration kernel and the same fixed-CSR ownership model.
Names describe their mathematical spaces rather than implementation details.
Assembly methods reuse
and overwrite one result object, so call `.copy()` before another assembly when
results must be retained.  Public basis values use component-first shape
`(basis, component, cell, quadrature)` and curls use
`(basis, cell, quadrature)` for triangles and
`(basis, component, cell, quadrature)` for tetrahedra; coefficient fields use
`(cell, quadrature)`.

`basis.geometry_diagnostics` reports minimum signed/absolute Jacobian
determinants, minimum cell area/volume, an edge-based aspect indicator, and the
number of negatively oriented cells.  An optional `max_aspect_ratio` constructor
argument rejects meshes beyond an application-selected quality limit.  Tests
cover distorted meshes and mixed cell orientations, including manufactured
PDE convergence.  With this sharply limited scope, `space.hcurl` is declared
experimental; this does not imply support for general meshes or form syntax.

### Assembly memory preflight

Estimate native bilinear-assembler memory before its CSR pattern and scatter
map are allocated:

```python
estimate = skfemntv.estimate_bilinear_memory(basis)

print(skfemntv.format_bytes(
    estimate.construction_peak_total_bytes_upper_bound
))

if not estimate.fits_in(available_bytes, safety_factor=1.25):
    raise MemoryError("assembly does not fit the configured memory budget")
```

The same check can be enforced atomically before native CSR and scatter-map
allocation:

```python
assembler = skfemntv.NativeBilinearForm(
    basis,
    memory_limit_bytes=16 * 1024**3,
    memory_safety_factor=1.25,
)
```

`NativeCrossBilinearForm` accepts the same options.  A failed check raises
`AssemblyMemoryBudgetError` containing the estimate, configured budget,
safety-adjusted requirement, and largest estimated allocation.  The default
safety factor is 1.25.  Supplying no limit preserves existing behavior and does
not impose a process-wide policy.  Segmented `CutCellBasis` bilinear and cross
assemblers use active-cell counts and flattened cut quadrature in the same
preflight and budget API.

The report separates memory already retained by the basis from persistent
native assembler allocations and temporary CSR-pattern construction memory.
Its CSR nonzero count is a conservative, allocation-free upper bound.  Python
object overhead, allocator fragmentation, and coefficient arrays supplied at
assembly time are intentionally excluded and listed in the report assumptions.

Use `estimate_bilinear_memory(test_basis, trial_basis)` for a rectangular
cross-bilinear block.  For mixed spaces,
`estimate_composite_bilinear_memory(basis, field_pairs=...)` estimates the
selected independently cached blocks; omitting `field_pairs` assumes all
ordered field pairs are assembled once.  Repeat a field pair when distinct
value/gradient contraction kinds create multiple cached native blocks.

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
