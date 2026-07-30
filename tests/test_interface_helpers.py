import numpy as np

from skfn.helpers import avg, jump, normal_grad
from skfn.interface import InterfaceField


def test_jump_and_weighted_average_are_formulation_neutral():
    field=InterfaceField(master=np.array([3.,5.]),slave=np.array([1.,2.]))
    np.testing.assert_array_equal(jump(field),[2.,3.])
    np.testing.assert_allclose(avg(field,weights=(.25,.75)),[1.5,2.75])


def test_normal_gradient_uses_each_side_outward_normal():
    field=InterfaceField(
        master=np.zeros(1),slave=np.zeros(1),
        master_gradient=np.array([[2.],[3.],[4.]]),
        slave_gradient=np.array([[5.],[6.],[7.]]),
        master_normal=np.array([[1.],[0.],[0.]]),
        slave_normal=np.array([[-1.],[0.],[0.]]),
    )
    derivative=normal_grad(field)
    np.testing.assert_array_equal(derivative.master,[2.])
    np.testing.assert_array_equal(derivative.slave,[-5.])
