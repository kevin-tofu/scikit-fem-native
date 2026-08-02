# skfem-native

`skfem-native` (`import skfemntv`) is a compact C++ finite-element assembly engine
with a scikit-fem-compatible Python API.

> `skfemntv` is an independent project.  It is not part of, maintained by, or
> affiliated with the scikit-fem project.  scikit-fem is used only by the test
> suite as a numerical and API reference; it is never a runtime dependency.

Compatibility applies to the documented subset, not to every scikit-fem API.
Unsupported forms and operations raise `skfemntv.UnsupportedNativeForm`; there is
no runtime fallback to scikit-fem or Python element assembly.

## Installation

Install the latest release from PyPI:

```bash
python -m pip install skfem-native
```

The distribution name is `skfem-native`; the Python import name is `skfemntv`:

```python
import skfemntv

# Convenient when porting a form from scikit-fem's supported subset:
import skfemntv as skfem
```

PyPI provides prebuilt CPython 3.10--3.14 wheels for Linux x86_64 (glibc 2.27
or newer), Windows AMD64, macOS arm64, and macOS x86_64.  On these platforms,
`pip` selects the matching wheel automatically; users do not need a C++
compiler.  Other platforms, including Linux arm64, Alpine/musl, Windows
32-bit, and PyPy, currently have no prebuilt wheel and may require a source
build toolchain.

## API contract

### scikit-fem-compatible subset

| Area | Supported API |
|---|---|
| Forms | `BilinearForm`, `LinearForm`, `Functional`, `asm` |
| Form helpers | `dot`, `ddot`, `grad`, `div`, `sym_grad`, `trace` |
| Meshes | `MeshTri`, `MeshTri2`, `MeshQuad`, `MeshQuad2`, `MeshTet`, `MeshTet2`, `MeshWedge1`, `MeshHex`, `MeshHex2` |
| Mesh topology | cached `facets`, `t2f`, `f2t`, `boundary_facets`, facet/element predicates |
| Elements | Tri/Quad/Tet/Hex P0, nodal P1/P2 or Q1/Q2, Wedge6, `ElementDG`, `ElementVector`, `ElementComposite` |
| Bases | `Basis`, `FacetBasis`, `InteriorFacetBasis`, `interpolate`, `get_dofs`, composite splitting |
| Form context | `w.x`, facet `w.n`, user scalars, arrays, callables, interpolated fields |
| Mixed forms | Expanded signatures such as `u, p, v, q, w` |
| Rectangular forms | `asm(form, trial_basis, test_basis)` across different orders or components |
| Interfaces | `jump`, `avg`, `normal_grad`, value and gradient contractions |

### skfemntv extensions

These APIs are useful native-assembly features but are not expected to run
unchanged with upstream scikit-fem.

| API | Purpose |
|---|---|
| `TriangleSupermesh` | Nonmatching surface quadrature and rectangular coupling |
| `SupermeshSearch` | Retained planar topology across geometry updates |
| `TriangleSupermesh.update()` | Refit/rebuild overlap data after motion |
| Interface `Functional` | Integrate gap, normals, and two-sided traces |
| `NativeAssembler` and native kernels | Direct residual/tangent evaluation |
| `MeshPyramid1` / `ElementPyramid1` | Pyramid5 volume and mixed-face assembly (not provided by scikit-fem) |

### Intentionally out of scope

- linear and nonlinear solvers;
- condensation and pressure-pinning policy;
- contact, mortar, or Nitsche formulation selection;
- automatic fallback for unsupported scikit-fem forms;
- importing scikit-fem at runtime.

The core H1 assembly engine exposes reusable SciPy CSR matrices and supports
Tri3/Tri6, Quad4/Quad9, Tet4/Tet10, Wedge6, Pyramid5, and Hex8/Hex27 volume and facet integration,
mixed fields, and nonmatching surface coupling.

Reproducible scaling reports live under `benchmarks/`.  In addition to the
Poisson comparison, `benchmarks/nonlinear-assembly/neo_hookean.py` measures
fused Tet4 Neo-Hookean residual/tangent assembly against numerically equivalent
scikit-fem forms, including one- and four-thread native series.

The topology-independent Neo-Hookean kernel is also regression-tested with
native Tet10, Hex27, Wedge6, and Pyramid5 bases.  Tet10, Hex27, and Wedge6
residuals and consistent tangents are compared with equivalent scikit-fem
forms at integration orders 2, 4, and 6; Pyramid5 uses finite-difference
consistent-tangent checks because it is an skfemntv extension.  Every topology
checks serial/parallel agreement, inverted deformation, and singular geometry.

Stateful small-strain J2 plasticity uses the same fused assembly shape.  The
input state remains committed until the caller accepts the returned trial
state, so a failed Newton step can be rolled back by retaining the old object:

```python
material = skfemntv.J2Plasticity(
    young_modulus=210e3,
    poisson_ratio=.3,
    yield_stress=250.,
    hardening_modulus=1e3,
)
assembler = skfemntv.MaterialAssembler(basis, material)
state = assembler.initial_state()

result = assembler.assemble(u, state, num_threads=4)
# commit after convergence
state = result.trial_state
```

Material state is a contiguous
`(integration_points, material.state_size)` buffer.  J2 exposes named
zero-copy views for plastic strain and accumulated equivalent plastic strain.
State-free `LinearElasticity` uses the same `MaterialAssembler` workflow with
`state_size == 0`.

One-branch Standard Linear Solid viscoelasticity uses six viscous-strain state
components and a backward-Euler material update:

```python
material = skfemntv.StandardLinearSolid(
    equilibrium_modulus=1000.,
    branch_modulus=500.,
    poisson_ratio=.3,
    relaxation_time=2.,
    time_step=.1,
)
assembler = skfemntv.MaterialAssembler(basis, material)
state = assembler.initial_state()
result = assembler.assemble(u, state, num_threads=4)
state = result.trial_state
```

The material's `time_step` is the default.  Adaptive stepping and cutback pass
an evaluation-time override without rebuilding the CSR pattern, coloring, or
geometry:

```python
trial = assembler.assemble(u, state, time_step=dt, num_threads=4)
# retain state when rejected; commit trial.trial_state when accepted
```

The fused call returns the internal-force residual, a consistent CSR tangent,
and integration-point plastic strain history.  It supports any three-component
H1 `Basis` whose tabulated gradients fit the native element-size limit,
including the Tet and Hex orders implemented by skfemntv.

Stateful material assembly is regression-tested on Tet4, Tet10, Wedge6,
Pyramid5, Hex8, and Hex27.  Tet10, Wedge6, Pyramid5, and Hex27 coverage includes distorted geometry,
integration orders 2, 4, and 6, serial/parallel agreement, finite-difference
consistent tangents, and J2 comparison with scikit-fem forms.

Wedge6 and Pyramid5 support mixed triangular/quadrilateral boundary and
interior facets in one `FacetBasis`.  A common collapsed-square quadrature
keeps the per-facet arrays rectangular; `mesh._facet_sizes` identifies the
three- or four-node topology represented by each padded `mesh.facets` column.

Pyramid5 is an explicit skfemntv extension rather than part of the
scikit-fem-compatible subset;
its rational shape functions are tested by volume, partition-of-unity,
constant-strain, material-tangent, and serial/parallel checks.

Native element loops use one thread by default.  Geometry tabulation can use
the shared native thread pool explicitly:

```python
skfemntv.set_num_threads(4)
basis = skfemntv.Basis(mesh, element)
```

Use `skfemntv.get_num_threads()` to inspect the current setting.  Explicit thread
selection keeps benchmarks reproducible and avoids oversubscription when a
surrounding application already uses parallel workers.

When an application supports both scikit-fem and skfemntv, branch explicitly for
native-only controls:

```python
if getattr(skfem, "has_capability", lambda name: False)("native_threads"):
    with skfem.thread_limit(4):
        basis = skfem.Basis(mesh, element)
else:
    basis = skfem.Basis(mesh, element)
```

The requested limit is capped by the CPU affinity visible to the process.

### Package and release checks

Update the single project-version source before creating a GitHub Release:

```bash
./scripts/upgrade_version.py 0.1.1 --dry-run
./scripts/upgrade_version.py 0.1.1
```

The script accepts normalized PEP 440 releases, updates only
`[project].version` in `pyproject.toml`, and rejects invalid or unchanged
versions.  The release tag must then be `v0.1.1`.

The same distribution check used by GitHub Actions can be run locally.  It
builds an sdist and a native wheel, validates their metadata, installs the
wheel into a clean temporary environment, and runs the full test suite against
the installed package:

```bash
python -m pip install --upgrade build twine
python tools/package_check.py
```

Use `--smoke-only` for the faster PR check that installs and imports the wheel
without repeating the full test suite; `--skip-tests` validates only build
metadata.

To build the complete Linux wheel matrix locally, install Docker and
`cibuildwheel`, then run:

```bash
python -m pip install --upgrade cibuildwheel
python tools/build_wheels.py --platform linux --arch x86_64
```

The same wrapper runs natively on macOS and Windows.  A single-version build is
useful while iterating:

```bash
python tools/build_wheels.py --platform linux --arch x86_64 --python 3.12
python tools/build_wheels.py --platform macos --arch arm64 --python 3.12
python tools/build_wheels.py --platform macos --arch x86_64 --python 3.12
python tools/build_wheels.py --platform windows --arch AMD64 --python 3.12
```

Linux uses Docker to produce repaired manylinux wheels.  macOS wheels must be
built on macOS and Windows wheels on Windows; they cannot be produced by the
Linux Docker build.

`tools/local_ci.py` is the cross-platform entry point matching the GitHub
stages.  `fast` installs the editable package and runs tests; `package` builds
and installs a wheel in a clean environment; `wheel` invokes cibuildwheel for
the detected native architecture; and `all` runs every stage:

```bash
python tools/local_ci.py fast
python tools/local_ci.py package
python tools/local_ci.py wheel --python 3.12
python tools/local_ci.py all --python 3.12
```

The normal PR workflow intentionally tests only Python 3.10/3.14 on Linux and
Python 3.14 on macOS arm64 and Windows AMD64.  The manually dispatched
`Full validation` workflow covers every supported Python on Linux, macOS arm64
and Intel, Windows AMD64, and a wheel-install smoke test on all four targets.

GitHub Releases whose tag is `v<version>` trigger `.github/workflows/workflow.yml`.
The workflow verifies the tag against `pyproject.toml`, builds CPython 3.10--3.14
wheels for Linux x86_64, macOS arm64, macOS x86_64, and Windows AMD64, and
publishes through PyPI Trusted Publishing.  Configure a protected GitHub
environment named `pypi` before the first release.

### License

`skfem-native` is distributed under the permissive BSD-3-Clause license, the
same license family used by scikit-fem.  Commercial use, modification, and
redistribution are permitted subject to the conditions in `LICENSE`.

```python
from skfemntv import LinearElasticity, NativeAssembler

assembler = NativeAssembler(
    coordinates, connectivity, element_dofs,
    LinearElasticity(young_modulus=210e9, poisson_ratio=0.3),
)
out = assembler.evaluate(u)
print(out.residual, out.tangent)
```

With the independent scikit-fem-style API:

```python
basis = Basis(mesh, ElementVector(ElementTetP1()))
assembler = NativeAssembler.from_basis(
    basis, LinearElasticity(young_modulus=210e9, poisson_ratio=0.3)
)
```

Compressible Neo-Hookean assembly uses Lamé parameters and returns an
analytically consistent tangent:

```python
from skfemntv import NeoHookean

kernel = NeoHookean.from_young_poisson(young_modulus=100.0, poisson_ratio=0.3)
assembler = NativeAssembler.from_basis(basis, kernel)
out = assembler.assemble(u, state=None, loads=external_force)
```

See `examples/neo_hookean_tet4.py` for a complete Newton solve.

Quadratic Tet10 and Hex27 nodal H1 elements are available through the
independent mesh, mapping, basis, facet, interpolation, and form APIs.
The same APIs support two-dimensional Tri3/Tri6 and Quad4/Quad9 elements,
including edge normals, curved quadratic edges, and mixed-order P2/P1 or
Q2/Q1 Taylor--Hood fields:

```python
mesh = skfem.MeshTri2.from_mesh(
    skfem.MeshTri.init_tensor(x, y)
)
element = (
    skfem.ElementVector(skfem.ElementTriP2())
    * skfem.ElementTriP1()
)
basis = skfem.Basis(mesh, element, intorder=4)
```

Volume and facet bases accept arbitrary integration orders.  Orders above the
built-in low-order rules use tensor-product Gauss rules for Quad/Hex cells and
Duffy-transformed Gauss rules for Tri/Tet cells.  A scikit-fem-style custom
reference quadrature `(points, weights)` can also be supplied directly:

```python
basis = skfem.Basis(mesh, element, intorder=8)

points, weights = basis.quadrature
fbasis = skfem.FacetBasis(
    mesh, element, quadrature=(facet_points, facet_weights)
)
ibasis = skfem.InteriorFacetBasis(
    mesh, element, side=0,
    quadrature=(facet_points, facet_weights),
)
```

Custom points use reference-cell coordinates and have shape `(dim, nq)`;
weights have shape `(nq,)`.  Both sides of an interior facet retain the same
quadrature-point ordering.

Cell integration can be restricted to material or physical regions using the
same element-subset API as scikit-fem.  The assembled vector or matrix keeps
the global space size; elements outside the selection contribute zero:

```python
steel = skfem.Basis(mesh, element, elements=steel_elements)
rubber = basis.with_elements(rubber_elements)

matrix = skfem.asm(steel_form, steel)
matrix += skfem.asm(rubber_form, rubber)
```

Integer element indices and Boolean masks are accepted.  `tind` records the
selected global element indices, while `nelems` reports their count.

Interior edge forms use the same two-sided basis-list convention and
`jump(w, u)` helper as scikit-fem:

```python
side = [
    skfem.InteriorFacetBasis(mesh, element, side=i)
    for i in (0, 1)
]

@skfem.BilinearForm
def penalty(u, v, w):
    return dot(jump(w, u), jump(w, v))

matrix = skfem.asm(penalty, side, side)
```

Different trial and test spaces assemble a rectangular block without requiring
an `ElementComposite`:

```python
velocity = skfem.Basis(
    mesh, skfem.ElementVector(skfem.ElementTriP2())
)
pressure = skfem.Basis(mesh, skfem.ElementTriP1())

@skfem.BilinearForm
def divergence(u, q, w):
    return div(u) * q

# Shape: (pressure.N, velocity.N)
B = skfem.asm(divergence, velocity, pressure)
```

Value, gradient, and row/column divergence blocks use retained native CSR
patterns.  The same two-Basis form supports compatible `FacetBasis` objects.

Cellwise constants and discontinuous nodal spaces use the familiar element
wrappers:

```python
cellwise = skfem.Basis(mesh, skfem.ElementTriP0())
dg = skfem.Basis(mesh, skfem.ElementDG(skfem.ElementTriP1()))
```

P0 volume assembly is available on Tri, Quad, Tet, and Hex meshes.
`ElementDG` supports the nodal elements above.  Interior jump assembly is
available on Tri3/Tri6, Quad4/Quad9, Tet4/Tet10, and Hex8/Hex27 through
`InteriorFacetBasis`.

Facet arguments use global facet numbers, matching scikit-fem:

```python
left = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], 0.0),
    boundaries_only=True,
)
fbasis = skfem.FacetBasis(mesh, element, facets=left)
boundary_dofs = basis.get_dofs(facets=left)
```

Selections are immutable first-class regions while remaining accepted anywhere
an integer ID array is accepted.  Regions support union, intersection,
difference, and complement:

```python
mesh = mesh.with_subdomains({
    "left": lambda x: x[0] < 0.5,
    "loaded": [3, 4, 7],
}).with_boundaries({
    "wall": lambda x: np.isclose(x[0], 0.0),
})

active = mesh.subdomains["left"] | mesh.subdomains["loaded"]
volume_basis = skfem.Basis(mesh, element, elements=active)
wall_basis = skfem.FacetBasis(mesh, element, facets="wall")
```

`CellRegion`, `FacetRegion`, and `NodeRegion` preserve sorted global IDs,
selection diagnostics, and the entity count needed for complement.  Named
subdomains can also be selected directly using `elements="left"`; restricting
a Basis does not renumber its global DOFs.

Pass `normal=` to orient selected facets.  For an interior facet this selects
the parent side whose outward normal is aligned with the requested direction;
for an exterior facet it retains the only parent and records a normal sign:

```python
interface = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], 0.5),
    normal=np.array([1.0, 0.0]),
)
oriented = skfem.FacetBasis(mesh, element, facets=interface)
```

The resulting `FacetRegion.sides` and `FacetRegion.normal_signs` are immutable
per-facet metadata.  `FacetBasis` applies both, including mixed orientations in
one region.

Vector and composite DOFs can be selected by component without relying on
stride assumptions:

```python
boundary = basis.get_dofs("wall")
x_dofs = boundary.all("u^1")
in_plane = boundary.keep(["u^1", "u^2"]).all()

# Native numeric selectors are convenient for generated mixed spaces.
velocity_x = mixed_basis.get_dofs(
    "wall", fields=0, components={0: 0}
).all()
pressure = mixed_basis.get_dofs("wall", fields=1).all()
```

Simple vector names follow scikit-fem (`u^1`, `u^2`, ...).  Composite groups
use unambiguous names such as `field0^1` and `field1^1`; they are available in
`DofsView.groups`.  `all(names)`, `keep(names)`, `drop(names)`, `components=`,
and `fields=` all return sorted global DOF IDs.

The cached `mesh.facets`, `mesh.t2f`, and `mesh.f2t` arrays are shared by
repeated Basis construction.  `mesh.interior_facets()` is a small skfemntv
convenience returning `np.flatnonzero(mesh.f2t[1] != -1)`.

Level-set sign classification is separate from cut-cell integration.  A
callable or one scalar per global mesh node classifies every cell as inside,
outside, cut, or touching.  The returned regions preserve global cell IDs and
can be passed directly to `Basis`:

```python
level_set = skfem.LevelSet(lambda x: x[0] ** 2 + x[1] ** 2 - .25)
classification = level_set.classify(mesh)
active_basis = skfem.Basis(
    mesh, element, elements=classification.active,
)
active_dofs = classification.active_dofs(active_basis)
background_facets = classification.active_facets(mesh)
active_boundary = classification.active_boundary_facets(mesh)
ghost_candidates = classification.ghost_facets(mesh)
```

The convention is negative-inside.  Every connectivity node is sampled, so
high-order edge, face, and interior nodes participate in classification.
`CUT` means that both signs were sampled; `TOUCHING` means at least one value
is within tolerance without a sampled sign crossing.  Classification itself
does not construct quadrature.  `ghost_facets` is only the active-interior
candidate set incident to cut cells; stabilization layers, penalty parameters,
and the weak form remain user choices.

Affine Tri3 and Tet4 meshes also provide the first cut-volume quadrature stage:

```python
inside = level_set.cut_quadrature(mesh, side="inside", intorder=2)
outside = level_set.cut_quadrature(mesh, side="outside", intorder=2)

for cell in classification.cut:
    local = inside.cell_slice(cell)
    x = inside.points[local]
    weight = inside.weights[local]

cut_basis = skfem.CutCellBasis(
    skfem.Basis(mesh, element), inside,
)
field = cut_basis.interpolate(solution)
integral = cut_basis.integrate(field.value)
```

`cell_offsets` stores variable quadrature counts without per-cell padding.
Physical and reference points, positive physical weights, background cell IDs,
and consistently oriented level-set normals are immutable local arrays.  The
Order one integrates constant and linear physical fields exactly.  Order two
uses positive standard simplex rules, and higher orders use positive
Duffy-transformed Gauss rules.  Curved/high-order cuts and implicit-interface
quadrature are not yet supported.  `CutCellBasis` tabulates affine TriP1/TetP1
shape values and physical gradients directly on the flattened rule, maps every
point to global element DOFs, and interpolates scalar or vector coefficients
without constructing padded element-by-quadrature arrays.  `Functional`,
`LinearForm`, and `BilinearForm` use the same public form syntax and execute in
the native C++ assemblers, including `num_threads=`.  Each real cut point is
presented as a one-point native entity; there is no Python assembly fallback or
zero-weight padding.

The supported form subset is intentionally source-compatible with scikit-fem:

```python
import numpy as np
import skfemntv as skfem
from skfemntv.helpers import dot

basis = skfem.Basis(
    skfem.MeshTet(),
    skfem.ElementVector(skfem.ElementTetP1()),
)

@skfem.LinearForm
def body_force(v, w):
    return dot(w.force, v)

force = np.array([0.0, 0.0, -1.0])[:, None, None]
rhs = skfem.asm(body_force, basis, force=force)
```

Physical quadrature coordinates are available as `w.x`.  `FacetBasis` also
provides its outward unit normal as `w.n`; both may participate in NumPy
expressions evaluated before native assembly:

```python
def load(x):
    return np.stack((1.0 + x[0], x[1] ** 2, -0.5 * x[2]))

@skfem.LinearForm
def varying_load(v, w):
    return dot(load(w.x), v)

@skfem.LinearForm
def pressure_normal(v, w):
    return dot(w.n, v)

@skfem.BilinearForm
def weighted_mass(u, v, w):
    return (1.0 + w.x[0] ** 2) * dot(u, v)
```

Supported value and gradient contractions are dispatched to native assembly.
Unsupported forms raise `skfemntv.UnsupportedNativeForm`; `skfemntv.asm` never silently
falls back to Python assembly.  Use upstream `skfem.asm` explicitly for an
independent reference assembly.  The same native API works with `FacetBasis`
for surface tractions.

Nonmatching coplanar P1 triangle surfaces can be coupled through a reusable
supermesh:

```python
supermesh = skfem.TriangleSupermesh(
    master_points,
    master_triangles,
    slave_points,
    slave_triangles,
    components=3,
)
coupling = supermesh.assemble()
```

Triangle intersections and overlap quadrature are generated once.  Repeated
coupling assembly reuses a rectangular CSR pattern and runs the cross-basis
quadrature and scatter in C++.

Master and slave spaces may have different component counts.  For example, a
scalar multiplier can be coupled to a three-component displacement with a
component tensor:

```python
supermesh = skfem.TriangleSupermesh.from_facets(
    multiplier_facet_basis,
    displacement_facet_basis,
)

# Shape: (..., multiplier components, displacement components).
# Leading entity/quadrature axes are broadcast automatically.
coupling = supermesh.assemble_tensor(np.array([[[1.0, 2.0, 3.0]]]))
```

Tensor coefficients need not be symmetric.  Reversing the master and slave
spaces and transposing the component axes produces the transpose coupling
matrix.

Cross-basis value and full physical-gradient contractions share the same
native fixed-CSR assembler:

```python
# coefficient axes:
# (row component, column component, column spatial direction)
value_gradient = supermesh.assemble_cross(
    coefficient,
    row_kind="value",
    column_kind="gradient",
)

gradient_gradient = supermesh.assemble_cross(
    diffusivity,
    row_kind="gradient",
    column_kind="gradient",
)
```

All four value/value, value/gradient, gradient/value, and gradient/gradient
combinations are supported.  Mixed value-gradient contractions require a
tensor coefficient because the spatial direction must be explicit.  A scalar
gradient-gradient coefficient contracts matching component and spatial axes.
The same contractions are available through interface forms:

```python
from skfemntv.helpers import avg, ddot, dot, grad, jump

@skfem.BilinearForm
def gradient_jump(u, v, w):
    return w.kappa * ddot(jump(grad(u)), jump(grad(v)))

@skfem.BilinearForm
def directional_flux(u, v, w):
    return dot(jump(v), dot(w.beta, avg(grad(u))))
```

The dependency-free contraction loop is isolated behind C++ basis and
coefficient views in `cross_contraction.hpp`.  This keeps the public and
assembly APIs stable if a specialized tensor/SIMD backend is selected later.

For direct planar triangle surfaces, AABB sweep, polygon clipping, fan
triangulation, and quadrature generation run in C++ with the GIL released.
Typed vectors use amortized append with pre-reserved capacity and are converted
to NumPy arrays once after construction; no per-intersection `hstack` or
full-array reallocation is performed.  The structured-grid scaling benchmark
is available as:

```sh
python benchmarks/supermesh_builder.py 16 32 64 128
```

Planar topology can be retained across geometry updates:

```python
search = skfem.SupermeshSearch(
    master_triangles,
    slave_triangles,
)
integration = search.build(master_points, slave_points)

for master_x, slave_x in deformed_coordinates:
    integration = search.update(master_x, slave_x)
    matrix = integration.assemble()
```

When the integration-triangle DOF maps are unchanged,
`CrossBilinearAssembler` updates shape values and weights while retaining its
CSR pattern and scatter map.  Sliding that changes overlap pairs rebuilds the
pattern safely.  A fully open interface is represented by empty quadrature and
a correctly sized zero matrix, so subsequent closing can reuse the same search
object.  Update diagnostics report created and disappeared overlap pairs,
pattern reuse, and update count.

For curved Tet10/Hex27 facets, search geometry is refined adaptively:

```python
supermesh = skfem.TriangleSupermesh.from_facets(
    master_basis,
    slave_basis,
    geometry_tolerance=1e-4,
    max_subdivision_level=6,
    projection_tolerance=1e-6,
)
print(supermesh.diagnostics)
```

The chord-error tolerance controls only reusable intersection geometry; parent
high-order shape values and gradients are still evaluated isoparametrically at
every generated overlap quadrature point.

`TriangleSupermesh.from_facets(master, slave)` accepts independent Tet4,
Tet10, Hex8, and Hex27 `FacetBasis` objects.  Search geometry is triangulated,
while shape values at overlap quadrature points are evaluated from the full
parent isoparametric element; high-order coupling is therefore scattered
directly to the original element DOFs.

Interface forms remain formulation-neutral:

```python
from skfemntv.helpers import avg, dot, jump, normal_grad

@skfem.BilinearForm
def interface_form(u, v, w):
    return (
        w.a * dot(jump(u), jump(v))
        + w.b * dot(avg(normal_grad(u)), jump(v))
    )

matrix = skfem.asm(
    interface_form,
    master_basis,
    slave_basis,
    integration=supermesh,
    a=penalty_like_coefficient,
    b=flux_like_coefficient,
)
```

Vector linear-elasticity fluxes use a quadrature-local constitutive tensor;
the package still leaves consistency signs, averaging, and penalty selection
to the user:

```python
from skfemntv.helpers import (
    avg, dot, grad, isotropic_traction_tensor, jump,
)

@skfem.BilinearForm
def consistency(u, v, w):
    traction = isotropic_traction_tensor(
        w.n_master, w.lame_lambda, w.lame_mu
    )
    return dot(jump(v), dot(traction, avg(grad(u))))

@skfem.BilinearForm
def adjoint_consistency(u, v, w):
    traction = isotropic_traction_tensor(
        w.n_master, w.lame_lambda, w.lame_mu
    )
    return dot(jump(u), dot(traction, avg(grad(v))))
```

Both value-gradient and gradient-value terms use the native CSR scatter.  The
second matrix is the transpose of the first for matching coefficients, which
allows symmetric or nonsymmetric Nitsche variants to be composed explicitly.
The regression suite compares complete symmetric Poisson and linear-elastic
interface matrices against scikit-fem facet integrals.  It also checks equal
and opposite elastic resultants and quadratic-field convergence on successively
refined nonmatching P1 traces.

Interface and mortar assembly accept the same bounded per-call thread control
as volume assembly:

```python
matrix = skfem.asm(
    interface_form, master_basis, slave_basis,
    integration=supermesh, num_threads=4,
)
result = supermesh.assemble_mortar("dual", num_threads=4)
```

Direct planar supermesh construction and retained-coordinate updates also
accept `num_threads`.  AABB candidates are generated deterministically, then
their clipping and quadrature records are processed in contiguous thread-local
chunks and concatenated in original candidate order:

```python
supermesh = skfem.TriangleSupermesh(
    master_points, master_triangles,
    slave_points, slave_triangles,
    num_threads=4,
)
supermesh.update(moved_master, moved_slave, num_threads=4)
```

Search/build scaling is available through
`python benchmarks/supermesh_builder.py 32 64 128 --threads 1,2,4`.

The cross assembler colors overlap entities by row DOFs.  Entities within a
color own disjoint CSR rows, allowing lock-free scatter while preserving a
bitwise-stable result.  Requested thread counts are capped by the CPUs visible
to the process.  Scaling can be measured locally with:

```sh
python benchmarks/supermesh_parallel.py \
  --cells 32,64,128 --threads 1,2,4 --repeat 3 \
  --multiplier dual --output benchmarks/supermesh-parallel.csv
```

Interface `LinearForm` uses the same reusable supermesh quadrature and returns
one vector containing the master DOFs followed by the slave DOFs:

```python
@skfem.LinearForm
def interface_load(v, w):
    return dot(w.traction, jump(v))

rhs = skfem.asm(
    interface_load,
    master_basis,
    slave_basis,
    integration=supermesh,
    traction=traction,
)
```

Scalar overlap quantities use `Functional` without requiring bases at
assembly time:

```python
master_trace, slave_trace = supermesh.interpolate(
    master_solution,
    slave_solution,
)

@skfem.Functional
def interface_energy(w):
    jump_value = w.slave - w.master
    return 0.5 * w.penalty * jump_value**2 + w.gap**2

energy = skfem.asm(
    interface_energy,
    integration=supermesh,
    master=master_trace,
    slave=slave_trace,
    penalty=penalty,
)
```

Interface functionals expose `w.x`, `w.n_master`, `w.n_slave`, and `w.gap`.
`TriangleSupermesh.interpolate()` evaluates both value traces and, for
`from_facets()`, their physical gradients at overlap quadrature points.

`avg(v)`, `jump(grad(v))`, `avg(grad(v))`, and normal-gradient traces are
assembled natively as well.  The weights supplied by `jump` produce equal and
opposite master/slave resultants for a constant interface traction.

Supermesh forms receive geometry evaluated at every overlap quadrature point:

```python
@skfem.LinearForm
def varying_interface_load(v, w):
    return dot(load(w.x), jump(v))

@skfem.LinearForm
def master_pressure(v, w):
    return dot(w.n_master, jump(v))

@skfem.BilinearForm
def gap_weighted_penalty(u, v, w):
    return (1.0 + w.gap ** 2) * dot(jump(u), jump(v))
```

`w.n_master` is the master outward unit normal and `w.n_slave` is the exactly
opposing interface convention.  The independently evaluated slave outward
normal is checked before pairing; `orientation_mismatch_count` and
`maximum_normal_opposition_error` report surfaces whose raw normals are not
opposed.  Thus flux kernels always receive `n_master == -n_slave` while bad
component orientation remains visible in diagnostics.
`w.gap` is the signed search-geometry separation measured along the master
normal.  Curved Tet10 and Hex27 normals are evaluated from the original
isoparametric facets rather than copied from the tessellated search triangles.

Mortar constraints have a dedicated sparse result:

```python
result = supermesh.assemble_mortar("slave")

Bm = result.master_matrix
Bs = result.slave_matrix
B = result.coupling_matrix  # [Bm, -Bs]
```

The multiplier selector accepts `"slave"` and `"master"` P1 traces,
`"overlap_p0"`, and a facet-local `"dual"`/biorthogonal trace.  The latter
uses only small parent-facet Gram matrices.  All global outputs are CSR blocks;
the implementation does not form a multiplier-sized dense matrix.

`supermesh.master_trace` and `supermesh.slave_trace` expose shape values,
physical gradients, paired outward normals, quadrature weights, physical
coordinates, DOF maps, and parent facet/element indices at the same physical
quadrature points.  These quadrature-local arrays are suitable for custom
Nitsche stress-flux kernels such as `sigma(u) @ normal`.

User parameters, callables, and geometry values share one numerical
quadrature-expression context:

```python
def pressure_law(x, gap, scale):
    return scale * (1.0 + x[0] + gap ** 2)

@skfem.LinearForm
def nonlinear_pressure(v, w):
    pressure = w.pressure_law(w.x, w.gap, w.scale)
    return dot(pressure * w.n_master, jump(v))

rhs = skfem.asm(
    nonlinear_pressure,
    master_basis,
    slave_basis,
    integration=supermesh,
    pressure_law=pressure_law,
    scale=1.7,
)
```

NumPy ufuncs are supported as coefficient expressions, so scalar, vector, and
tensor parameters can be combined with coordinates, normals, and gap before
the resulting contiguous coefficient is passed to native integration.

Volume and facet bilinear forms may contain multiple value and gradient terms:

```python
@skfem.BilinearForm
def reaction_diffusion(u, v, w):
    return (
        w.reaction * (1.0 + w.x[0] ** 2) * dot(u, v)
        + w.diffusivity * np.exp(-w.x[1])
          * ddot(grad(u), grad(v))
    )
```

Terms of the same kind are added as quadrature coefficients.  The combined
value and gradient coefficients are then assembled in one native traversal,
with one CSR zeroing and scatter pass.

Composite nodal H1 fields use the same expanded signature as scikit-fem:

```python
element = skfem.ElementTetP1() * skfem.ElementTetP1()
basis = skfem.Basis(mesh, element)

@skfem.BilinearForm
def coupled(u1, u2, v1, v2, w):
    return (
        (1.0 + w.x[0]) * u1 * v1
        + 2.0 * u2 * v2
        + 0.3 * u2 * v1
        - 0.4 * u1 * v2
        + w.diffusion * dot(grad(u1), grad(v1))
    )
```

`ElementComposite` interleaves its nodal subfield DOFs and caches each native
rectangular block assembler.  It supports component-compatible value and
gradient contractions as well as mixed-order vector-scalar divergence blocks
using the usual mixed-form notation:

```python
element = (
    skfem.ElementVector(skfem.ElementTetP2())
    * skfem.ElementTetP1()
)
basis = skfem.Basis(mesh, element, intorder=4)

@skfem.BilinearForm
def mixed(u, p, v, q, w):
    return (
        w.mu * ddot(grad(u), grad(v))
        - p * div(v)
        - q * div(u)
    )
```

Both `p * div(v)` and `q * div(u)` are assembled as native rectangular
gradient-value blocks; no Python element loop or runtime scikit-fem fallback
is used.  Mixed-order nodal fields share one geometry and quadrature context;
for example, the Taylor–Hood pair above places velocity DOFs on Tet10 vertices
and edges while pressure DOFs remain on its four vertices.

Composite right-hand sides use the corresponding expanded `LinearForm`
signature:

```python
@skfem.LinearForm
def load(v, q, w):
    return (
        dot(w.force, v)
        + w.source * q
        + ddot(w.flux, grad(v))
    )

rhs = skfem.asm(
    load, basis,
    force=force,
    source=source,
    flux=flux,
)
```

Value and full-gradient terms are grouped per subfield and each cached native
linear assembler is traversed once.

Mixed-field DOFs can be separated without introducing a solver dependency:

```python
velocity_basis, pressure_basis = basis.split_bases()
velocity_indices, pressure_indices = basis.split_indices()

velocity_boundary = velocity_indices[
    velocity_basis.get_dofs().all()
]
pressure_boundary = pressure_indices[
    pressure_basis.get_dofs().all()
]
```

`split_bases()` returns independently numbered field bases, while
`split_indices()` maps their local ordering into the assembled composite
matrix and vector.  `get_dofs()` selects boundary DOFs by default and also
accepts explicit facet connectivity, element indices, or node indices.
Condensation, pressure pinning, and linear solution remain user-side choices.

Coordinate predicates and named boundaries follow the same workflow:

```python
mesh = mesh.with_boundaries({
    "inlet": lambda x: np.isclose(x[0], 0.0),
    "outlet": lambda x: np.isclose(x[0], 1.0),
})
basis = skfem.Basis(mesh, element, intorder=4)

inlet = basis.get_dofs("inlet").all()
outlet = basis.get_dofs(
    lambda x: np.isclose(x[0], 1.0)
).all()
```

Boundary predicates are evaluated at boundary-facet centers.  The same named
selection is preserved by `split_bases()` and maps correctly back through
`split_indices()`.

DOF vectors can be interpolated onto the basis quadrature points:

```python
field = basis.interpolate(solution)
values = field.value
gradients = field.grad
divergence = field.div
```

For a composite basis, interpolation returns one field per subelement:

```python
velocity, pressure = basis.interpolate(solution)

@skfem.LinearForm
def residual(v, q, w):
    return (
        ddot(grad(w.velocity), grad(v))
        + w.pressure * q
    )

r = skfem.asm(
    residual, basis,
    velocity=velocity,
    pressure=pressure,
)
```

Interpolated scalar fields can also be passed as quadrature coefficients to a
`BilinearForm`, enabling solution-dependent native assembly without adding a
nonlinear solver policy to `skfemntv`.

The mixed APIs are tested together on a P2/P1 Taylor--Hood Stokes system:
native bilinear and linear assembly, named velocity-boundary extraction,
pressure pinning, user-side SciPy solution, composite interpolation, and
dissipation evaluation are compared end-to-end against scikit-fem.  Linear
solution and condensation remain outside the package.

Scalar volume and surface quantities use `Functional`:

```python
@skfem.Functional
def energy(w):
    return (
        0.5 * ddot(grad(w.u), grad(w.u))
        + w.pressure**2
    )

velocity, pressure = basis.interpolate(solution)
value = skfem.asm(
    energy, basis,
    u=velocity,
    pressure=pressure,
)
```

The same API works with `FacetBasis`, including `w.x`, `w.n`, and fields
interpolated directly on its quadrature points.  A functional expression is
evaluated to one scalar per quadrature point and its weighted reduction runs
in C++ with the GIL released.

`skfemntv` only traces and assembles the requested jump, weighted average, value,
and outward-normal-gradient contractions; it does not select a Nitsche,
mortar, or contact formulation.

Build and test with:

```sh
python -m pip install -e '.[test]'
pytest
```
