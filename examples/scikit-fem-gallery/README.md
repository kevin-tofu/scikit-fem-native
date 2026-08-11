# scikit-fem Gallery compatibility

These scripts reproduce selected examples from the
[scikit-fem Gallery](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html)
with both `skfem` and `skfemntv` assembly.

The two backends receive identical coordinates, connectivity, polynomial
spaces, quadrature order, boundary conditions, and solver inputs.  Each script
checks the assembled operators and the final solution and exits with an error
when the configured tolerance is exceeded.

Initial coverage:

- Example 1: two-dimensional Poisson equation with unit load
- Example 9: three-dimensional Poisson equation
- Example 19: heat equation with quadratic quadrilateral elements and the
  Crank--Nicolson method

Run every comparison:

```bash
python examples/scikit-fem-gallery/run_all.py
```

The examples are compatibility checks rather than copied source files.  Plotting
and optional example-specific preconditioners are omitted so the comparison
focuses on assembly and numerical results.
