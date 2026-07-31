# skfem-native

`skfem-native` (`import skfn`) is a compact C++ finite-element assembly engine
with a scikit-fem-compatible Python API.

> `skfn` is an independent project.  It is not part of, maintained by, or
> affiliated with the scikit-fem project.  scikit-fem is used only by the test
> suite as a numerical and API reference; it is never a runtime dependency.

Compatibility applies to the documented subset, not to every scikit-fem API.
Unsupported forms and operations raise `skfn.UnsupportedNativeForm`; there is
no runtime fallback to scikit-fem or Python element assembly.

## API contract

### scikit-fem-compatible subset

| Area | Supported API |
|---|---|
| Forms | `BilinearForm`, `LinearForm`, `Functional`, `asm` |
| Form helpers | `dot`, `ddot`, `grad`, `div`, `sym_grad`, `trace` |
| Meshes | `MeshTri`, `MeshTri2`, `MeshQuad`, `MeshQuad2`, `MeshTet`, `MeshTet2`, `MeshHex`, `MeshHex2` |
| Mesh topology | cached `facets`, `t2f`, `f2t`, `boundary_facets`, facet/element predicates |
| Elements | Tri/Quad/Tet/Hex P0, nodal P1/P2 or Q1/Q2, `ElementDG`, `ElementVector`, `ElementComposite` |
| Bases | `Basis`, `FacetBasis`, `InteriorFacetBasis`, `interpolate`, `get_dofs`, composite splitting |
| Form context | `w.x`, facet `w.n`, user scalars, arrays, callables, interpolated fields |
| Mixed forms | Expanded signatures such as `u, p, v, q, w` |
| Rectangular forms | `asm(form, trial_basis, test_basis)` across different orders or components |
| Interfaces | `jump`, `avg`, `normal_grad`, value and gradient contractions |

### skfn extensions

These APIs are useful native-assembly features but are not expected to run
unchanged with upstream scikit-fem.

| API | Purpose |
|---|---|
| `TriangleSupermesh` | Nonmatching surface quadrature and rectangular coupling |
| `SupermeshSearch` | Retained planar topology across geometry updates |
| `TriangleSupermesh.update()` | Refit/rebuild overlap data after motion |
| Interface `Functional` | Integrate gap, normals, and two-sided traces |
| `NativeAssembler` and native kernels | Direct residual/tangent evaluation |

### Intentionally out of scope

- linear and nonlinear solvers;
- condensation and pressure-pinning policy;
- contact, mortar, or Nitsche formulation selection;
- automatic fallback for unsupported scikit-fem forms;
- importing scikit-fem at runtime.

The core H1 assembly engine exposes reusable SciPy CSR matrices and supports
Tri3/Tri6, Quad4/Quad9, Tet4/Tet10, Hex8/Hex27, volume and facet integration,
mixed fields, and nonmatching surface coupling.

```python
from skfn import LinearElasticity, NativeAssembler

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
from skfn import NeoHookean

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

The cached `mesh.facets`, `mesh.t2f`, and `mesh.f2t` arrays are shared by
repeated Basis construction.  `mesh.interior_facets()` is a small skfn
convenience returning `np.flatnonzero(mesh.f2t[1] != -1)`.

The supported form subset is intentionally source-compatible with scikit-fem:

```python
import numpy as np
import skfn as skfem
from skfn.helpers import dot

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
Unsupported forms raise `skfn.UnsupportedNativeForm`; `skfn.asm` never silently
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
from skfn.helpers import avg, ddot, dot, grad, jump

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
from skfn.helpers import avg, dot, jump, normal_grad

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

`w.n_master` and `w.n_slave` are the independent outward unit normals;
`w.gap` is the signed search-geometry separation measured along the master
normal.  Curved Tet10 and Hex27 normals are evaluated from the original
isoparametric facets rather than copied from the tessellated search triangles.

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
nonlinear solver policy to `skfn`.

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

`skfn` only traces and assembles the requested jump, weighted average, value,
and outward-normal-gradient contractions; it does not select a Nitsche,
mortar, or contact formulation.

Build and test with:

```sh
python -m pip install -e '.[test]'
pytest
```
