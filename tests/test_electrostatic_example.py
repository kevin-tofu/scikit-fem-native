from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_example():
    path = Path(__file__).parents[1] / "examples" / "electrostatic_potential.py"
    spec = importlib.util.spec_from_file_location("electrostatic_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_electrostatic_example_solves_parallel_plate_problem():
    module = _load_example()
    mesh, basis, potential = module.solve_capacitor(np.linspace(0.0, 1.0, 7))

    left = basis.get_dofs(nodes=lambda x: np.isclose(x[0], 0.0)).all()
    right = basis.get_dofs(nodes=lambda x: np.isclose(x[0], 1.0)).all()
    np.testing.assert_allclose(potential[left], 1.0)
    np.testing.assert_allclose(potential[right], 0.0)
    assert np.all(potential >= -1.0e-12)
    assert np.all(potential <= 1.0 + 1.0e-12)
    assert mesh.nelements > 0
