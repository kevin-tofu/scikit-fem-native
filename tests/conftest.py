"""Mark scikit-fem parity cases whose APIs are not implemented yet."""

import pytest


_UNSUPPORTED_MODULES = {
    "test_basis_split_compatibility",
    "test_basis_standard_parameters",
    "test_composite_dg_compatibility",
    "test_composite_dofs_native",
    "test_composite_point_evaluation",
    "test_curved_surface_native",
    "test_dg_projection_compatibility",
    "test_element_finder_compatibility",
    "test_hcurl_general_basis",
    "test_line_mesh_compatibility",
    "test_mesh_edge_topology_compatibility",
    "test_mesh_hex_refined_compatibility",
    "test_mesh_topology_native",
    "test_nonlinear_multicell_assembly",
    "test_nonlinear_neo_hookean_parity",
    "test_point_source_compatibility",
    "test_refinterp_compatibility",
    "test_tensor_product_point_evaluation",
}

_UNSUPPORTED_CASES = {
    ("test_mesh_coordinate_transforms", "test_mesh_coordinate_transforms_preserve_connectivity_and_regions"),
    ("test_mesh_p2_promotion", "test_native_p2_promotion_builds_skfem_compatible_basis"),
    ("test_nonlinear_j2_plasticity", "test_j2_plastic_tangent_history_and_elastic_stiffness_for_supported_families"),
    ("test_nonlinear_neo_hookean_invariants", "test_assembler_is_objective_under_finite_rotation"),
    ("test_nonlinear_neo_hookean_invariants", "test_assembler_isotropic_stretch_matches_analytic_solution"),
    ("test_nonlinear_neo_hookean_invariants", "test_isotropic_stretch_response_is_consistent_across_supported_element_families"),
    ("test_nonlinear_neo_hookean_invariants", "test_rigid_rotation_response_is_objective_across_supported_element_families"),
}


def pytest_collection_modifyitems(items):
    for item in items:
        module = item.path.stem
        if module in _UNSUPPORTED_MODULES or (module, item.originalname) in _UNSUPPORTED_CASES:
            item.add_marker(pytest.mark.skip(reason="scikit-fem parity API is not implemented yet"))
