from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_example():
    path = Path(__file__).parents[1] / "examples" / "linear_elasticity_standalone.py"
    spec = importlib.util.spec_from_file_location("linear_elasticity_standalone", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_elasticity_example_solves_without_kktkit():
    module = _load_example()
    mesh, basis, displacement = module.solve_cube(divisions=2)

    fixed = basis.get_dofs(nodes=lambda x: np.isclose(x[2], 0.0)).all()
    np.testing.assert_allclose(displacement[fixed], 0.0)
    assert mesh.nelements > 0
    assert displacement.shape == (basis.N,)
    assert np.all(np.isfinite(displacement))
    assert np.linalg.norm(displacement) > 0.0
