# C++ Acceleration Audit — 2026-10-15

## Candidate inventory

| Priority | Operation | Current path | Why native code may help | Scope / caution |
|---|---|---|---|---|
| Done | Curved tetrahedral surface evaluation | Tet4/Tet10 inverse mapping, physical gradients, and normals used while tessellating curved facet supermeshes now have an optional native kernel. | 2.15x faster complete self-coupling build and assembly on the measured warped Tet10 case. | Polygon clipping and overlap search remain Python. H(curl) remains a profiling candidate, but its measured work is mostly NumPy's compiled `einsum`. |
| Done | High-order basis tabulation | Hex27 reference shape values and gradients now use a C++ kernel in `Basis._geometry`. | 382–666x faster table generation for 27–729 quadrature points in the local microbenchmark. | Geometry transformation was already native; this moves only reference table creation. Simplex quadratic tables remain Python. |
| Done | Linear Tri3 to P2 mesh promotion | C++ creates lexicographically ordered unique edge nodes and P2 connectivity for `Basis` auto-promotion. | 84–91x faster promotion and 17.4x faster full P2 Basis construction on 2,048 triangles. | Python fallback remains for older extensions; named boundaries and subdomains are remapped in Python. |
| Done | Composite H1 DOF assembly | C++ now numbers shared nodes, builds per-field and composite element DOFs, and fills global DOF locations. | 2.85–7.47x faster full P2/P1 Basis construction for 128–8,192 triangles. | Python retains element-local coordinate ordering and falls back when an older extension is loaded. |
| Done | Tri3 facet topology | C++ builds lexicographically ordered facets, element-to-facet IDs, and two-sided adjacency used by mesh refinement. | 5.47–5.87x faster refinement at 2,048–32,768 input triangles. | Exact NumPy facet order and boundary/subdomain IDs are preserved. |
| Done | `LevelSet.cut_quadrature` on Tri3 | Native cell clipping and quadrature kernel, with Python fallback and other mesh families unchanged. | Locally 305–524x faster over the measured sizes. | Python reference parity and cut workflow tests pass. `interface_quadrature` is still Python. |
| Done | Triangle point location and field evaluation | SciPy cKDTree still finds candidate cells; C++ handles candidate bounds, affine/quadratic inverse mapping, hit selection, field values, and gradients. | Locally 307–361x faster for 5,000 queries on 512–2,048 curved triangles. | Tetrahedral evaluation retains the Python implementation. |

## Completed implementation goal

Move the dominant Tri3 `LevelSet.cut_quadrature` cell loop to C++ while preserving the Python API and exact quadrature contract. Keep unsupported mesh families on the current Python path. Compare timings at increasing mesh sizes and run the existing cut-quadrature parity suite.

## Baseline

Local median of five `cut_quadrature(..., intorder=2)` calls for `phi=x+0.37y-0.413`:

| Resolution | Triangles | Cut cells | Quadrature points | Time |
|---:|---:|---:|---:|---:|
| 32 | 2,048 | 88 | 1,662 | 227 ms |
| 64 | 8,192 | 176 | 6,132 | 883 ms |
| 128 | 32,768 | 350 | 23,463 | 3,471 ms |

These timings include classification and output construction. They are a local baseline, not a cross-machine performance claim.

## Tri3 implementation result

`LevelSet.cut_quadrature` now dispatches Tri3 meshes to the C++ kernel when the
updated extension is available; older extensions and other supported simplex
families retain the existing Python implementation. It returns the same CSR
cell offsets, point counts, physical and reference coordinates, weights, and
interface normals. Native output was checked against the Python path per cell
through area and polynomial moments, with both inside/outside sides and orders
1, 2, 3, and 5.

The repeatable benchmark is `benchmarks/cutfem/cut_quadrature_backend.py`.
On the same local environment, medians of three runs at integration order 2:

| Resolution | Triangles | Python | C++ | Speedup |
|---:|---:|---:|---:|---:|
| 32 | 2,048 | 225.7 ms | 0.741 ms | 304.6x |
| 64 | 8,192 | 874.1 ms | 1.790 ms | 488.3x |
| 128 | 32,768 | 3,457.4 ms | 6.594 ms | 524.4x |

The next high-value candidate is high-order basis tabulation. Profile basis
construction first to isolate that work from mesh setup and geometry.

## Curved triangle point-location implementation result

`Basis.evaluate_at` now dispatches 2D simplex bases to C++ when the updated
extension is available. Python keeps SciPy's `cKDTree` candidate search; the
native kernel filters candidates, applies the affine seed and quadratic
isoparametric Newton iteration, resolves shared-cell hits by global cell ID,
and computes P0/P1/P2 scalar or vector values and physical gradients. Older
extensions and tetrahedron evaluation continue through Python.

`benchmarks/point_evaluation_backend.py` compares both paths and verifies the
full result before timing. With 5,000 deterministic queries in the interior of
a warped MeshTri2 square, medians of two runs:

| Resolution | Triangles | Python | C++ | Speedup |
|---:|---:|---:|---:|---:|
| 16 | 512 | 3,977.8 ms | 11.0 ms | 360.6x |
| 32 | 2,048 | 3,961.7 ms | 12.9 ms | 307.3x |

The timing includes candidate search and result allocation. The interior query
box keeps the generated points inside the curved boundary.

## Hex27 tabulation implementation result

Profiling isolated reference table creation from the already-native geometry
transformation. The former Python loop spent about 2.1 ms at 27 quadrature
points and 52.2 ms at 729 points in a preliminary 100-repeat timing. The native
kernel now creates the same `(nq, 27)` shape table and `(nq, 27, 3)` reference
gradient table. The repeatable comparison is
`benchmarks/basis_tabulation_backend.py`; after parity checks, a 20-repeat run
measured:

| Quadrature points | Python | C++ | Speedup |
|---:|---:|---:|---:|
| 27 | 1.249 ms | 0.003 ms | 381.7x |
| 125 | 5.745 ms | 0.010 ms | 570.1x |
| 343 | 15.741 ms | 0.024 ms | 665.8x |
| 729 | 33.299 ms | 0.050 ms | 666.2x |

Native output agrees with the prior Python loop to `2e-15` absolute and relative
tolerance. The Hex27 interpolation tests, including comparison to scikit-fem,
also pass. The microbenchmark excludes mesh setup and geometry mapping; gains
for complete Basis construction will therefore be smaller. Remaining quadratic
simplex tabulation should be profiled separately before porting.

## Composite H1 DOF assembly implementation result

`Basis._init_composite` delegates global numbering and array construction to
`assemble_composite_dofs` in C++. The kernel preserves the old global order
(mesh node, then field, then component), field-local `element_dofs`, common
node DOFs, composite local row order, and physical DOF locations. Python still
determines the small element-local coordinate layout and keeps the previous
implementation as a fallback for older native extensions. Exact native-versus-
Python array checks cover scalar and vector P2/P1 fields; existing scikit-fem
interpolation, assembly, and projection comparisons also pass.

The repeatable benchmark is `benchmarks/composite_dofs_backend.py`. It uses a
prebuilt MeshTri2 mesh and compares complete P2/P1 Basis construction, with
three-run medians:

| Triangles | Python DOF path | C++ DOF path | Speedup |
|---:|---:|---:|---:|
| 128 | 3.016 ms | 1.060 ms | 2.85x |
| 2,048 | 31.429 ms | 4.749 ms | 6.62x |
| 8,192 | 118.969 ms | 15.916 ms | 7.47x |

At this profile point, the next measured candidate was `_build_topology` during
mesh refinement. It has since been ported as described below.

## Mesh preparation profiling and P2 promotion result

Profiling on local structured triangle meshes identified two sizeable paths.
Uniformly refining 8,192 Tri3 cells took 29.7 ms; about 27 ms was spent
building facet topology through `_build_topology` and `numpy.unique`. Promoting
a 2,048-cell linear mesh to MeshTri2 took 30.6 ms through `MeshTri2.from_mesh`.
A P2/P1 composite Basis on an existing 2,048-cell MeshTri2 took 35.0 ms, of
which 25 ms was in `Basis._init_composite`; the two native geometry calls
accounted for about 3 ms total. Composite numbering and refinement topology
were then ported with explicit array-order and region parity checks.

The high-frequency P2 auto-promotion path used when constructing a P2 Basis
from a linear mesh is now native. `promote_tri_p2_mesh` produces the same
lexicographic unique-edge numbering and local P2 node order as the prior
Python implementation. `_promote_linear_p2_mesh` keeps the Python path when
an older extension lacks this function; named boundaries and subdomains
continue through the existing preservation logic. Tests compare exact point
and cell arrays and check named-region preservation. The repeatable benchmark
is `benchmarks/p2_promotion_backend.py`; medians of three runs:

| Triangles | Python promotion | C++ promotion | Speedup |
|---:|---:|---:|---:|
| 128 | 3.442 ms | 0.039 ms | 87.93x |
| 2,048 | 52.616 ms | 0.578 ms | 91.06x |
| 8,192 | 212.238 ms | 2.539 ms | 83.59x |

For a full P2 Basis on 2,048 triangles, construction decreased from 55.128 ms
to 3.174 ms (17.37x). The full Basis comparison also checks identical
`element_dofs`. Composite DOF numbering has since been moved to C++, followed
by refinement topology with exact facet-order parity.

## Full Hex27 Basis construction impact

`benchmarks/basis_construction_backend.py` measures repeated `Basis` creation
on a prebuilt MeshHex2 mesh with the Python reference tabulator temporarily
substituted for the native one. It checks the integrated volume
`sum(det(J) * W)` on both paths. Median of three local runs:

| Hex elements | Integration order | Quadrature points | Python tabulator, full Basis | C++ tabulator, full Basis | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 27 | 1.470 ms | 0.227 ms | 6.48x |
| 1 | 8 | 125 | 7.504 ms | 1.990 ms | 3.77x |
| 1 | 12 | 343 | 19.681 ms | 4.617 ms | 4.26x |
| 64 | 8 | 125 | 12.568 ms | 7.060 ms | 1.78x |

The smaller end-to-end gains on larger meshes are expected: C++ geometry
mapping scales with element count, while reference shape tables are shared by
all elements. A `cProfile` run for the 64-element, order-8 case attributed about
5 ms to native geometry mapping and 3 ms to Python tensor quadrature setup;
the Hex27 tabulator was below the profiler's 1 ms display resolution. The
remaining Python work worth addressing first is quadrature creation and
reuse, while the dominant per-element geometry path is already native. This
profile does not justify translating that small quadrature setup loop to C++
yet; cacheability or vectorization should be assessed first. Mesh refinement
topology and composite DOF construction have now been addressed for the
measured Tri3/H1 paths.

## Tri3 facet topology implementation result

`_TopologyMesh._build_topology` dispatches linear triangle connectivity to
`build_tri_facet_topology` when the updated extension is present. The C++
kernel sorts canonical edge pairs lexicographically, then reproduces the
NumPy implementation's local-edge-major/cell-major `t2f` layout and adjacent
element ordering in `f2t`. Other mesh families and older extensions retain
their Python paths. Exact comparisons cover all three topology arrays, and a
refinement parity test checks points, cells, topology, named boundaries, and
subdomains against the Python implementation. Empty triangle connectivity is
also covered.

The repeatable benchmark is `benchmarks/mesh_topology_backend.py`. It includes
uniform refinement on fresh meshes and reports three-run medians:

| Input triangles | Python refinement | C++ topology refinement | Speedup |
|---:|---:|---:|---:|
| 2,048 | 7.258 ms | 1.328 ms | 5.47x |
| 8,192 | 29.391 ms | 5.236 ms | 5.61x |
| 32,768 | 117.557 ms | 20.029 ms | 5.87x |

The refinement connectivity generation remains Python and is now a smaller
share of the measured call. Broader remaining candidates are unsupported
supermesh construction paths and H(curl) reference evaluation.

## Curved tetrahedral surface evaluation result

Profiling a warped one-cell Tet10 self-coupling (`geometry_tolerance=.05`,
subdivision level 3) attributed about 593 ms of 686 ms total to supermesh
initialization. Repeated volume-basis evaluation inside curved facet
tessellation accounted for roughly 248 ms, while surface evaluation and
recursive subdivision accounted for most of the rest. A separate H(curl)
profile showed that `einsum` and covariant Piola operations dominate its
reference evaluation; those operations already execute in compiled NumPy, so
translating the surrounding Python setup was not the first useful native task.

The new `evaluate_tet_surface_points` kernel handles inverse mapping for Tet4
and Tet10 geometry, physical basis gradients, and outward normals. The Python
implementation remains the fallback for unsupported elements and older
extensions. Direct native-versus-Python value, gradient, and normal checks and
the complete existing supermesh test module pass. The repeatable benchmark is
`benchmarks/curved_supermesh_backend.py`; on a warped Tet10 facet self-coupling
with tolerance `.01` and subdivision level 4, three-run medians were:

| Search triangles per side | Python complete build + assembly | C++ surface evaluator | Speedup |
|---:|---:|---:|---:|
| 130 | 1,332.876 ms | 621.329 ms | 2.15x |

The benchmark checks parent-facet selection, quadrature coordinates and
weights, and the assembled coupling matrix. More of the curved supermesh
pipeline remains Python, so this is a partial acceleration; broader meshes
should be profiled before moving polygon clipping or overlap search to C++.
