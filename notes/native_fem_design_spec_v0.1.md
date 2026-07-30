# Native FEM Assembly Engine
## Design Specification v0.1

**Status:** Draft  
**Primary language:** Python API + C++ execution core  
**Primary target:** nonlinear FEM assembly  
**Initial integration target:** scikit-fem-compatible data structures  
**Solver strategy:** reuse SciPy, PETSc, MUMPS, Hypre, PARDISO, or other external solvers

---

## 1. Purpose

This project provides a high-performance native assembly engine for finite element analysis while preserving a lightweight Python-facing workflow similar to scikit-fem.

The project does **not** primarily aim to create another general-purpose nonlinear solver. Its central purpose is to accelerate the part that remains expensive even when the linear or nonlinear solver is delegated to PETSc or another mature library:

- element-wise degree-of-freedom gathering;
- quadrature-point kinematics;
- constitutive-law evaluation;
- history-variable updates;
- element residual evaluation;
- consistent tangent evaluation;
- global sparse assembly.

The intended execution model is:

```text
Python describes the analysis
        ↓
C++ evaluates and assembles the physics
        ↓
PETSc / SciPy / external solvers solve the system
```

The core operation should be a coarse-grained native call:

```python
evaluation = assembler.evaluate(u, state, parameters)

R = evaluation.residual
K = evaluation.tangent
trial_state = evaluation.trial_state
```

A single call should perform the complete element and quadrature loops in native code.

---

## 2. Motivation

Python-based FEM libraries are highly effective for research, prototyping, optimization, reduced-order modeling, and access to internal matrices. Their performance becomes problematic when significant work remains inside Python-level element or quadrature loops.

For a nonlinear static step, the dominant operations may be decomposed as:

```text
1. Gather element displacement u_e
2. Evaluate kinematics at quadrature points
3. Update material response
4. Compute element residual r_e
5. Compute element tangent K_e
6. Scatter r_e and K_e into global objects
7. Solve K Δu = -R
```

Step 7 can be delegated effectively to existing solver ecosystems. This project targets Steps 1–6.

The main design assumption is:

> Solver performance is not the only bottleneck. Native element evaluation and sparse assembly must also be available without requiring users to adopt a large monolithic FEM framework.

---

## 3. Goals

### 3.1 Primary goals

1. Preserve a Python workflow compatible in spirit with scikit-fem.
2. Execute element loops, quadrature loops, constitutive updates, and sparse assembly in C++.
3. Accept the current global solution vector `u` through a low-overhead interface.
4. Return or populate the global residual and tangent matrix.
5. Support history-dependent material models with committed and trial states.
6. Permit direct access to matrices, vectors, element contributions, and state variables.
7. Remain independent of a specific linear or nonlinear solver.
8. Support both SciPy CSR and PETSc matrix/vector backends.
9. Keep the initial implementation small enough to validate independently.

### 3.2 Secondary goals

- Enable future matrix-free operators.
- Enable OpenMP element-level parallelism.
- Support user-provided native element kernels.
- Provide a path toward contact, mortar, and mixed formulations.
- Remain suitable for component-mode synthesis, Craig–Bampton ROM, and custom reduction methods.

---

## 4. Non-goals

The initial project will not:

- implement a new sparse direct solver;
- implement a new Krylov solver;
- replace PETSc SNES;
- provide a full automatic weak-form compiler;
- compile arbitrary Python functions to C++;
- support every finite element type;
- support MPI-distributed meshes in the first release;
- support GPU execution in the first release;
- provide production-grade general contact immediately;
- reproduce the complete scikit-fem API;
- become a complete replacement for CalculiX, FEniCSx, MFEM, or deal.II.

These exclusions are deliberate. The project is initially an **assembly engine**, not a full multiphysics platform.

---

## 5. High-level architecture

```text
┌──────────────────────────────────────────────┐
│ Python analysis layer                        │
│                                              │
│ Mesh / Basis / DOF selection / BCs / Loads  │
│ Newton loop or PETSc SNES configuration      │
│ ROM / optimization / post-processing         │
└──────────────────────┬───────────────────────┘
                       │ one coarse-grained call
                       ▼
┌──────────────────────────────────────────────┐
│ Native C++ assembly layer                    │
│                                              │
│ DOF gather                                   │
│ Element loop                                 │
│ Quadrature loop                              │
│ Kinematics                                   │
│ Material update                              │
│ Residual and tangent evaluation              │
│ State trial update                           │
│ Sparse scatter                               │
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│ Linear algebra backend                       │
│                                              │
│ SciPy CSR / PETSc Mat and Vec                │
│ MUMPS / Hypre / PARDISO / KSP                │
└──────────────────────────────────────────────┘
```

The Python layer controls the analysis, but no Python callback may be invoked per element or per quadrature point in the performance-critical path.

---

## 6. Compatibility strategy

The project should reuse or accept scikit-fem concepts where practical:

- mesh coordinates;
- connectivity;
- finite-element selection;
- basis and DOF numbering;
- boundary and subdomain selection;
- quadrature definitions;
- post-processing conventions.

Compatibility is divided into three levels.

### Level 1: data compatibility

The native assembler accepts mesh, connectivity, DOF maps, and basis-related arrays generated from scikit-fem.

This is mandatory for the MVP.

### Level 2: API familiarity

The Python-facing API resembles scikit-fem:

```python
basis = Basis(mesh, ElementVector(ElementTetP1()))

assembler = NativeAssembler(
    basis=basis,
    kernel=NeoHookean(mu=..., lmbda=...),
)

evaluation = assembler.evaluate(u)
```

This is strongly preferred.

### Level 3: arbitrary Form compatibility

Existing arbitrary `BilinearForm` or `LinearForm` Python functions execute natively without modification.

This is explicitly not an initial goal because arbitrary Python cannot be executed efficiently inside a C++ element loop. A future compiled DSL may support a restricted form language.

---

## 7. Primary Python API

### 7.1 Construction

```python
assembler = NativeAssembler(
    basis=basis,
    kernel=kernel,
    matrix_backend="scipy",
    scalar_type="float64",
)
```

Initial constructor inputs:

- `basis`: compatible basis or exported basis data;
- `kernel`: native element kernel descriptor;
- `matrix_backend`: `"scipy"` or `"petsc"`;
- `scalar_type`: initially `"float64"` only;
- optional region/material mapping;
- optional initial state allocation.

### 7.2 Evaluation

```python
evaluation = assembler.evaluate(
    u=u,
    committed_state=state,
    parameters=parameters,
    loads=loads,
)
```

Returned object:

```python
@dataclass
class NativeEvaluation:
    residual: ArrayLike
    tangent: SparseMatrixLike
    trial_state: StateHandle | None
    diagnostics: EvaluationDiagnostics
```

The implementation may support preallocated output:

```python
assembler.evaluate_into(
    u=u,
    residual=R,
    tangent_values=K.data,
    committed_state=state,
    trial_state=trial_state,
)
```

`evaluate_into` is preferred for repeated Newton iterations because it avoids repeated allocation.

### 7.3 Residual and tangent are evaluated together

The main API must evaluate residual and tangent in a single traversal:

```python
evaluation = assembler.evaluate(u)
```

The public API may expose convenience accessors, but internally it must avoid:

```python
R = assembler.assemble_residual(u)
K = assembler.assemble_tangent(u)
```

when that would repeat kinematics and constitutive calculations.

An optional mode may allow residual-only evaluation for line search:

```python
evaluation = assembler.evaluate(u, mode="residual")
```

Supported modes:

- `"residual_tangent"`
- `"residual"`
- future: `"tangent_action"`

---

## 8. Native element-kernel interface

The initial system uses predefined or user-compiled C++ kernels rather than arbitrary Python forms.

Conceptual interface:

```cpp
struct ElementContext {
    int element_id;
    int material_id;

    const double* coordinates;
    const int* element_dofs;

    const double* shape_values;
    const double* reference_shape_gradients;
    const double* quadrature_weights;

    int num_nodes;
    int num_dofs;
    int num_quadrature_points;
    int spatial_dimension;
};

struct KernelInput {
    const double* element_u;
    const double* element_u_previous;
    const double* committed_state;
    const double* parameters;

    double time;
    double time_step;
    double load_factor;
};

struct KernelOutput {
    double* element_residual;
    double* element_tangent;
    double* trial_state;

    double* optional_stress;
    double* optional_energy;
};

class ElementKernel {
public:
    virtual ~ElementKernel() = default;

    virtual void evaluate(
        const ElementContext& context,
        const KernelInput& input,
        KernelOutput& output,
        EvaluationMode mode
    ) const = 0;
};
```

The actual implementation may use templates instead of virtual dispatch in the element loop. Runtime polymorphism should not impose a per-quadrature-point virtual-call overhead.

Preferred implementation options:

1. template-specialized assembler per kernel;
2. function pointer selected outside the element loop;
3. variant-based dispatch outside the element loop;
4. virtual dispatch only once per element if benchmarking shows negligible impact.

---

## 9. Native assembly algorithm

### 9.1 Initialization

Initialization is allowed to be relatively expensive because it occurs once.

```text
1. Validate mesh and DOF maps
2. Export or copy immutable mesh data
3. Precompute quadrature data
4. Precompute reference basis values and gradients
5. Build global sparsity pattern
6. Build element-local-entry → CSR-position scatter map
7. Allocate state arrays
8. Allocate residual and tangent storage
9. Select kernel implementation
```

### 9.2 Evaluation

```text
evaluate(u, committed_state)

1. Validate input vector shape and type
2. Zero global residual
3. Zero tangent values
4. For each element:
   a. Gather u_e
   b. Gather previous element fields if needed
   c. For each quadrature point:
      i.   Compute geometry/Jacobian
      ii.  Compute physical gradients
      iii. Compute strain or deformation gradient
      iv.  Evaluate constitutive model
      v.   Update trial state
      vi.  Accumulate r_e
      vii. Accumulate K_e
   d. Scatter r_e into global residual
   e. Scatter K_e into global tangent values
5. Return residual, tangent, trial state, diagnostics
```

Python must not be re-entered during Steps 4a–4e.

---

## 10. Sparse matrix strategy

### 10.1 Fixed sparsity pattern

For standard material and geometric nonlinearities, the global sparsity pattern is normally fixed even though its values change.

The MVP will therefore:

1. construct CSR `indptr` and `indices` once;
2. construct an element scatter map once;
3. clear only `values` during each evaluation;
4. add local tangent entries directly to known CSR positions.

Conceptually:

```cpp
values[scatter_map[element][local_i][local_j]] += Ke[local_i][local_j];
```

The MVP must not generate a full COO triplet list and convert it to CSR at every Newton iteration.

### 10.2 Matrix ownership

For the SciPy backend:

- C++ owns or writes `indptr`, `indices`, and `values`;
- Python exposes them as a `scipy.sparse.csr_matrix`;
- repeated evaluations reuse the same structural arrays.

For the PETSc backend:

- the first implementation may assemble a sequential PETSc matrix;
- a direct PETSc insertion path may be added after the SciPy path is stable;
- unnecessary CSR-to-PETSc copies should be avoided where possible.

### 10.3 Contact and changing structure

General contact may change couplings. It is outside the initial MVP.

Future options include:

- over-allocated candidate-contact sparsity;
- dynamic PETSc insertion;
- block contact operators;
- matrix-free contact contributions.

---

## 11. Parallel assembly

### 11.1 MVP

The first implementation is single-threaded C++.

This is intentional because it isolates:

- Python-loop removal;
- kernel efficiency;
- sparse scatter efficiency;
- memory-layout effects.

### 11.2 OpenMP phase

The first parallel target is shared-memory OpenMP.

Potential strategies:

1. atomic scatter into CSR;
2. element coloring;
3. thread-local residual and matrix buffers;
4. row ownership by thread;
5. block assembly.

Element coloring is the preferred initial scalable strategy because it avoids concurrent writes for elements of the same color.

The selected method must be benchmarked rather than assumed.

### 11.3 MPI

Distributed mesh ownership, ghost DOFs, halo updates, and distributed assembly are deferred. The API must avoid decisions that make MPI impossible, but MPI support is not required for v0.x.

---

## 12. State-variable model

History-dependent materials require explicit state handling.

The native layer owns or manages:

```text
committed_state[element, quadrature_point, state_component]
trial_state[element, quadrature_point, state_component]
```

The core rules are:

1. `evaluate` reads committed state.
2. `evaluate` writes trial state.
3. residual-only reevaluations must not overwrite committed state.
4. line searches may produce multiple trial states.
5. state is committed only after solver acceptance.
6. rejected steps discard the trial state.

Python-facing usage:

```python
evaluation = assembler.evaluate(u, committed_state=state)

if converged:
    state = assembler.commit(evaluation.trial_state)
else:
    assembler.discard(evaluation.trial_state)
```

A future solver adapter may hide this lifecycle, but the assembly engine itself must expose it explicitly.

### 12.1 Initial state layout

Initial canonical layout:

```text
[element][quadrature_point][state_component]
```

stored contiguously with `state_component` as the fastest-changing index.

This must be benchmarked against structure-of-arrays layouts for vectorization-sensitive kernels.

---

## 13. Memory layout

### 13.1 Coordinates

Preferred initial layout:

```text
coordinates[node][spatial_dimension]
```

contiguous C-order `float64`.

### 13.2 Connectivity

```text
connectivity[element][local_node]
```

contiguous integer array.

### 13.3 Element DOF map

```text
element_dofs[element][local_dof]
```

contiguous integer array.

### 13.4 Reference basis data

```text
shape_values[quadrature_point][local_node]
reference_gradients[quadrature_point][local_node][reference_dimension]
```

For a fixed element family, these arrays are shared by all elements.

### 13.5 Element scratch storage

Element vectors and matrices should use stack storage or reusable thread-local buffers when sizes are known at compile time.

Dynamic allocation inside the element or quadrature loop is prohibited.

---

## 14. Python/C++ boundary

The binding layer may use nanobind or pybind11.

Requirements:

- accept contiguous NumPy `float64` vectors;
- inspect buffers without element-wise Python access;
- avoid copying `u` when safe;
- reuse output arrays;
- release the Python GIL during native evaluation;
- return clear errors for invalid shape, dtype, or contiguity;
- not retain unsafe references to temporary Python buffers.

The binding call should be coarse-grained:

```python
assembler.evaluate_into(u, state, residual, tangent_values)
```

not fine-grained:

```python
for element in elements:
    native_element_call(...)
```

---

## 15. Boundary conditions and loads

### 15.1 Essential boundary conditions

Initial options:

1. return the unconstrained residual and tangent and let Python/scikit-fem/PETSc handle condensation;
2. optionally apply native row/column modification using a supplied constrained-DOF list.

The MVP should first support unconstrained native assembly because it keeps the engine independent of a solver convention.

### 15.2 External loads

Initial supported load representation:

- preassembled global external force vector supplied by Python.

Then:

```text
R(u) = R_internal(u) - F_external
```

Later versions may support native:

- body forces;
- surface tractions;
- follower loads;
- pressure loads;
- centrifugal loads.

Follower loads belong in the nonlinear native layer because they may contribute to the tangent.

---

## 16. Initial kernel set

### v0.1 kernels

1. 3D linear elasticity, Tet4
2. 3D compressible Neo-Hookean hyperelasticity, Tet4

### v0.2 kernels

3. 3D small-strain J2 plasticity, Tet4
4. geometric nonlinear truss or beam reference kernel

### Later kernels

- Hex8 linear elasticity;
- Hex8 hyperelasticity;
- mixed displacement-pressure elements;
- anisotropic elasticity;
- viscoplasticity;
- damage;
- thermal coupling;
- penalty contact;
- Nitsche contact;
- mortar interfaces.

The first kernel set is intentionally narrow.

---

## 17. Solver integration

The assembly engine will not prescribe a nonlinear solver.

### 17.1 SciPy example

```python
for iteration in range(max_iterations):
    out = assembler.evaluate(u, committed_state=state)

    R = apply_bc_to_residual(out.residual)
    K = apply_bc_to_matrix(out.tangent)

    du = scipy.sparse.linalg.spsolve(K, -R)
    u += du
```

### 17.2 PETSc example

The engine provides callbacks or adapters for:

- residual evaluation;
- Jacobian evaluation;
- matrix reuse;
- trial-state handling.

PETSc SNES integration is an adapter layer, not the engine core.

### 17.3 Matrix-free future path

Future API:

```python
operator = assembler.linearized_operator(u, state)
y = operator @ x
```

The initial priority remains explicit residual and CSR tangent assembly because it is easier to validate and precondition.

---

## 18. Validation requirements

Correctness is more important than benchmark speed.

### 18.1 Element-level tests

For each kernel:

- rigid-body motion test where applicable;
- patch test;
- symmetry test where tangent symmetry is expected;
- energy consistency;
- comparison against analytical element results;
- finite-difference check of the element tangent.

Tangent check:

```text
K_e v ≈ [r_e(u + εv) - r_e(u)] / ε
```

### 18.2 Global tests

- comparison with scikit-fem linear elasticity;
- comparison with a trusted hyperelastic reference;
- Newton convergence-rate verification;
- mesh refinement study;
- load-step refinement study;
- state commit/rollback test;
- repeated evaluation without allocation-growth test.

### 18.3 Regression tests

Each release must retain:

- numerical reference values;
- matrix sparsity checks;
- residual norms;
- iteration counts where stable;
- serialization compatibility for state only if promised.

---

## 19. Performance requirements

Initial performance goals are targets, not release guarantees.

### Functional performance rules

1. One Python-to-C++ transition per native evaluation.
2. No Python callbacks inside element loops.
3. No dynamic allocation inside quadrature loops.
4. No sparsity-pattern regeneration during Newton iterations.
5. Residual and tangent evaluated together by default.
6. Reusable output and scratch buffers.
7. GIL released during native evaluation.
8. Immutable geometry and basis data retained natively after initialization.

### Initial benchmark targets

For sufficiently large Tet4 meshes:

- native assembly should be materially faster than a Python-level element-loop baseline;
- target 3× or greater speedup over the corresponding scikit-fem nonlinear assembly path where comparable;
- native overhead should become negligible relative to kernel work for large meshes;
- memory growth over repeated evaluations should be constant.

A claimed speedup must include:

- mesh size;
- DOF count;
- element count;
- compiler and flags;
- CPU;
- number of threads;
- matrix backend;
- whether geometry data were cached;
- whether tangent and residual were both evaluated.

---

## 20. Error handling and diagnostics

`EvaluationDiagnostics` should include:

```python
@dataclass
class EvaluationDiagnostics:
    element_count: int
    quadrature_evaluations: int
    assembly_seconds: float
    kernel_seconds: float | None
    scatter_seconds: float | None
    invalid_element_count: int
    material_failure_count: int
```

Material kernels must report recoverable failures, such as local return-mapping nonconvergence, without crashing the process.

Possible status model:

```cpp
enum class KernelStatus {
    Success,
    InvalidJacobian,
    MaterialNonconvergence,
    InvalidState,
    NumericalFailure
};
```

The assembler reports the first failure and aggregate counts.

---

## 21. Extensibility model

### 21.1 Built-in kernels

Distributed with the project and covered by full validation.

### 21.2 Native plugin kernels

Advanced users may compile an external kernel against a stable C++ kernel ABI or API.

The initial project may postpone ABI stability and support source-level plugin compilation only.

### 21.3 Future compiled form DSL

A future restricted form language may generate element kernels:

```text
Python expression subset
        ↓
intermediate representation
        ↓
generated C++
        ↓
compiled native kernel
```

This is a future project phase, not a prerequisite for the assembly engine.

---

## 22. Proposed package structure

```text
native_fem/
├── python/
│   └── native_fem/
│       ├── __init__.py
│       ├── assembler.py
│       ├── evaluation.py
│       ├── state.py
│       ├── kernels.py
│       ├── scipy_backend.py
│       ├── petsc_backend.py
│       └── skfem_adapter.py
├── cpp/
│   ├── include/native_fem/
│   │   ├── assembler.hpp
│   │   ├── element_context.hpp
│   │   ├── element_kernel.hpp
│   │   ├── state.hpp
│   │   ├── csr_pattern.hpp
│   │   └── kernels/
│   │       ├── linear_elastic_tet4.hpp
│   │       └── neo_hookean_tet4.hpp
│   └── src/
│       ├── assembler.cpp
│       ├── csr_pattern.cpp
│       └── bindings.cpp
├── tests/
│   ├── element/
│   ├── global/
│   ├── tangent/
│   └── performance/
├── examples/
│   ├── linear_tet4.py
│   ├── neo_hookean_tet4.py
│   └── petsc_newton.py
├── benchmarks/
├── CMakeLists.txt
├── pyproject.toml
└── DESIGN.md
```

---

## 23. MVP definition

The MVP is complete when all of the following are true:

1. A scikit-fem Tet4 vector basis can be exported to the native assembler.
2. NumPy displacement vector `u` can be passed without element-wise Python processing.
3. A C++ Tet4 linear-elastic kernel assembles residual and tangent.
4. CSR sparsity and scatter maps are created once and reused.
5. A SciPy CSR matrix is exposed to Python.
6. Native output agrees with scikit-fem within defined tolerances.
7. A Neo-Hookean Tet4 kernel produces a verified consistent tangent.
8. Newton solution of a small hyperelastic test converges.
9. Repeated assembly performs no unbounded allocation.
10. Benchmark results and reproduction instructions are published.

### Explicit MVP exclusions

- plasticity;
- contact;
- OpenMP;
- PETSc direct assembly;
- MPI;
- Hex8;
- arbitrary form compilation;
- general multiphysics.

---

## 24. Development sequence

### Phase 0: feasibility spike

- Export Tet4 basis and DOF data from scikit-fem.
- Implement C++ residual/tangent assembly for linear elasticity.
- Return a SciPy CSR matrix.
- Compare correctness and timing.

### Phase 1: stable native assembly core

- fixed CSR pattern;
- scatter map;
- reusable buffers;
- diagnostics;
- packaging and wheels;
- finite-difference tangent tests.

### Phase 2: nonlinear kernel

- Neo-Hookean Tet4;
- residual-only and residual-plus-tangent modes;
- state-free nonlinear example;
- Newton demonstration.

### Phase 3: history-dependent state

- J2 plasticity;
- committed/trial state;
- rollback tests;
- local constitutive failure reporting.

### Phase 4: solver and performance adapters

- PETSc matrix/vector adapter;
- SNES example;
- OpenMP experiments;
- coloring or thread-local scatter.

### Phase 5: interfaces and contact research

- native kernel plugin API;
- contact candidate infrastructure;
- penalty/Nitsche experiments;
- matrix-free operator prototype.

---

## 25. Key design decisions

### Decision 1

**The project is an assembly engine, not a solver suite.**

### Decision 2

**The main performance boundary is one Python call per complete assembly evaluation.**

### Decision 3

**Element and quadrature loops execute entirely in native code.**

### Decision 4

**Residual and tangent are evaluated together by default.**

### Decision 5

**The global sparsity pattern is precomputed and reused.**

### Decision 6

**Arbitrary existing scikit-fem forms are not automatically compiled in the MVP.**

### Decision 7

**The initial implementation is deliberately narrow: Tet4, float64, CPU, single process.**

### Decision 8

**Direct access to residuals, matrices, states, and element-level data remains a first-class requirement.**

---

## 26. Open design questions

The following questions should be answered through prototypes and benchmarks:

1. Should bindings use nanobind or pybind11?
2. Should C++ own CSR arrays or write into Python-owned arrays?
3. Is the first PETSc integration based on CSR conversion or direct PETSc insertion?
4. Does element coloring outperform atomic scatter for target meshes?
5. What state layout performs best for plasticity kernels?
6. Should fixed-size element kernels use Eigen, plain arrays, or generated loops?
7. How much scikit-fem `Basis` information should be cached versus independently reconstructed?
8. Should boundary condensation remain entirely outside the native layer?
9. What stable interface is required for user-defined native kernels?
10. At what point is a restricted compiled-form DSL justified?

None of these questions blocks the Phase 0 feasibility spike.

---

## 27. Initial acceptance benchmark

The first public benchmark should compare:

- scikit-fem reference assembly;
- native single-thread C++ assembly;
- optional native OpenMP assembly later.

Test cases:

1. Tet4 3D linear elasticity;
2. Tet4 3D Neo-Hookean material;
3. multiple mesh sizes spanning overhead-dominated to compute-dominated regimes.

Report:

- element count;
- DOF count;
- nonzero count;
- residual-only time;
- residual-plus-tangent time;
- peak memory;
- numerical error relative to reference;
- Newton iteration behavior for the nonlinear case.

---

## 28. Summary

The project fills a specific gap:

> retain Python-level control and matrix transparency while moving the nonlinear FEM element evaluation and sparse assembly bottleneck into C++.

Its initial value does not depend on creating a new nonlinear solver, supporting every element, or compiling arbitrary weak forms. A narrow native assembler with a well-defined kernel interface, reusable CSR structure, and scikit-fem data adapter is sufficient to demonstrate the concept.

The recommended first implementation is:

```text
scikit-fem Tet4 Basis
        ↓
C++ linear-elastic / Neo-Hookean kernel
        ↓
C++ residual and tangent assembly
        ↓
reused SciPy CSR structure
        ↓
external Newton and linear solver
```

If this path demonstrates both correctness and meaningful assembly speedup, stateful material models, PETSc integration, OpenMP, and contact can be added without changing the central architecture.
