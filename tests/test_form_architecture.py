"""Guard the typed-form module boundaries established during stabilization."""

from __future__ import annotations

import ast
from pathlib import Path

import skfemntv
from skfemntv._errors import UnsupportedNativeForm as InternalFormError


PACKAGE = Path(__file__).parents[1] / "python" / "skfemntv"
FORM_INTERNALS = {
    "_coefficients",
    "_composite_fields",
    "_errors",
    "_form_compiler",
    "_form_parameters",
    "_form_terms",
    "_h1_fields",
    "_interface_terms",
}


def _relative_imports(module: str) -> set[str]:
    tree = ast.parse((PACKAGE / f"{module}.py").read_text())
    return {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module is not None
    }


def test_typed_form_internal_modules_do_not_depend_on_dispatcher():
    for module in FORM_INTERNALS:
        assert "forms" not in _relative_imports(module), module


def test_typed_form_internal_dependency_graph_is_acyclic():
    graph = {
        module: _relative_imports(module) & FORM_INTERNALS
        for module in FORM_INTERNALS
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str):
        if module in visiting:
            raise AssertionError(f"typed-form import cycle reaches {module}")
        if module in visited:
            return
        visiting.add(module)
        for dependency in graph[module]:
            visit(dependency)
        visiting.remove(module)
        visited.add(module)

    for module in graph:
        visit(module)


def test_public_api_does_not_export_private_form_nodes():
    assert skfemntv.UnsupportedNativeForm is InternalFormError
    assert len(skfemntv.__all__) == len(set(skfemntv.__all__))
    for name in skfemntv.__all__:
        assert name == "__version__" or not name.startswith("_"), name
        assert hasattr(skfemntv, name), name
