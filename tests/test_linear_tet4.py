import numpy as np

from skfn import LinearElasticTet4, NativeAssembler


def one_tet():
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)
    dofs = np.arange(12, dtype=np.int64).reshape(1, 4, 3)
    return coordinates, connectivity, dofs


def test_rigid_translation_has_zero_residual():
    assembler = NativeAssembler(*one_tet(), LinearElasticTet4(1000.0, 0.3))
    u = np.tile([0.2, -0.4, 1.1], 4)
    out = assembler.evaluate(u)
    np.testing.assert_allclose(out.residual, 0.0, atol=1e-13)
    np.testing.assert_allclose(out.tangent.toarray(), out.tangent.toarray().T)


def test_tangent_matches_finite_difference_and_structure_is_reused():
    assembler = NativeAssembler(*one_tet(), LinearElasticTet4(1234.0, 0.25))
    rng = np.random.default_rng(4)
    u, direction = rng.normal(size=12), rng.normal(size=12)
    out = assembler.evaluate(u)
    data_id = id(out.tangent.data)
    eps = 1e-7
    fd = (
        assembler.evaluate(u + eps * direction, mode="residual").residual.copy()
        - assembler.evaluate(u, mode="residual").residual.copy()
    ) / eps
    np.testing.assert_allclose(fd, out.tangent @ direction, rtol=2e-8, atol=2e-6)
    assert id(assembler.evaluate(u).tangent.data) == data_id


def test_external_load_is_subtracted():
    assembler = NativeAssembler(*one_tet(), LinearElasticTet4(10.0, 0.2))
    load = np.arange(12, dtype=np.float64)
    out = assembler.evaluate(np.zeros(12), loads=load)
    np.testing.assert_array_equal(out.residual, -load)


def test_assemble_is_primary_evaluation_api():
    assembler = NativeAssembler(*one_tet(), LinearElasticTet4(10.0, 0.2))
    u = np.linspace(0.0, 0.1, 12)
    assembled = assembler.assemble(u, None)
    evaluated = assembler.evaluate(u)
    np.testing.assert_allclose(assembled.residual, evaluated.residual)
    np.testing.assert_allclose(
        assembled.tangent.toarray(), evaluated.tangent.toarray()
    )


def test_evaluate_into_caller_owned_storage():
    assembler = NativeAssembler(*one_tet(), LinearElasticTet4(80.0, 0.22))
    u = np.linspace(0.0, 0.1, 12)
    residual = np.full(12, np.nan)
    values = np.full(assembler.tangent.nnz, np.nan)
    diagnostics = assembler.evaluate_into(
        u, residual, tangent_values=values
    )
    expected = assembler.evaluate(u)
    np.testing.assert_allclose(residual, expected.residual)
    np.testing.assert_allclose(values, expected.tangent.data)
    assert diagnostics.element_count == 1


def test_evaluate_into_rejects_implicit_output_copy():
    assembler = NativeAssembler(*one_tet(), LinearElasticTet4(80.0, 0.22))
    with np.testing.assert_raises(TypeError):
        assembler.evaluate_into(
            np.zeros(12), np.empty(12, dtype=np.float32)
        )
