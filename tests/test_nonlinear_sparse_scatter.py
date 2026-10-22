import numpy as np
import pytest

from skfemntv._skfn import assemble_nonlinear_global


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_nonlinear_scatter_accepts_signed_index_dtypes(dtype):
    dofs = np.array([[0, 1]], dtype=np.int64)
    residual = np.array([[2., 3.]])
    tangent = np.array([[[1., 2.], [3., 4.]]])
    scatter = np.arange(4, dtype=dtype).reshape(1, 2, 2)

    assembled_residual, assembled_tangent = assemble_nonlinear_global(
        dofs, residual, tangent, scatter, 2, 4
    )

    np.testing.assert_array_equal(assembled_residual, [2., 3.])
    np.testing.assert_array_equal(assembled_tangent, [1., 2., 3., 4.])


def test_nonlinear_scatter_rejects_unsigned_and_noncontiguous_indices():
    dofs = np.array([[0, 1]], dtype=np.int64)
    residual = np.zeros((1, 2))
    tangent = np.zeros((1, 2, 2))
    unsigned = np.arange(4, dtype=np.uint32).reshape(1, 2, 2)
    with pytest.raises(ValueError, match="int32 or int64"):
        assemble_nonlinear_global(dofs, residual, tangent, unsigned, 2, 4)

    noncontiguous = np.arange(4, dtype=np.int32).reshape(1, 2, 2).transpose(0, 2, 1)
    with pytest.raises(ValueError, match="C-contiguous"):
        assemble_nonlinear_global(dofs, residual, tangent, noncontiguous, 2, 4)
