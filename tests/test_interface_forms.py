import numpy as np

import skfemntv
from skfemntv.helpers import avg, ddot, dot, grad, jump, normal_grad


def interface():
    linear=skfemntv.MeshTet()
    quadratic=skfemntv.MeshTet2.from_mesh(linear)
    master=skfemntv.FacetBasis(
        quadratic,skfemntv.ElementVector(skfemntv.ElementTetP2()),intorder=4
    )
    slave=skfemntv.FacetBasis(
        linear,skfemntv.ElementVector(skfemntv.ElementTetP1()),intorder=4
    )
    integration=skfemntv.TriangleSupermesh.from_facets(master,slave)
    return master,slave,integration


def test_jump_form_matches_trace_block_assembly():
    master,slave,integration=interface()

    @skfemntv.BilinearForm
    def form(u,v,w):
        return w.alpha*dot(jump(u),jump(v))

    actual=skfemntv.asm(
        form,master,slave,integration=integration,alpha=3.5
    )
    expected=integration.assemble_traces(
        (1.,-1.),(1.,-1.),coefficient=3.5
    )
    np.testing.assert_allclose(actual.toarray(),expected.toarray())
    constant=np.ones(master.N+slave.N)
    np.testing.assert_allclose(actual@constant,0.,atol=2e-14)


def test_average_normal_gradient_against_jump_is_user_composable():
    master,slave,integration=interface()

    @skfemntv.BilinearForm
    def form(u,v,w):
        return dot(avg(normal_grad(u),weights=(.3,.7)),jump(v))

    actual=form.assemble(
        master,slave,integration=integration
    )
    expected=integration.assemble_traces(
        (1.,-1.),(.3,.7),
        row_kind="value",column_kind="normal_gradient",
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=2e-14,atol=2e-14
    )


def test_interface_form_sum_builds_no_prescribed_formulation():
    master,slave,integration=interface()

    @skfemntv.BilinearForm
    def arbitrary(u,v,w):
        return (
            w.a*dot(jump(u),jump(v))
            + w.b*dot(avg(normal_grad(u)),jump(v))
        )

    actual=arbitrary.assemble(
        master,slave,integration=integration,a=2.,b=-.4
    )
    expected=(
        integration.assemble_traces((1.,-1.),(1.,-1.),coefficient=2.)
        +integration.assemble_traces(
            (1.,-1.),(.5,.5),
            row_kind="value",column_kind="normal_gradient",
            coefficient=-.4,
        )
    )
    np.testing.assert_allclose(actual.toarray(),expected.toarray())


def test_full_gradient_jump_form_dispatches_to_cross_kernel():
    master,slave,integration=interface()

    @skfemntv.BilinearForm
    def form(u,v,w):
        return w.kappa*ddot(jump(grad(u)),jump(grad(v)))

    actual=skfemntv.asm(
        form,master,slave,integration=integration,kappa=1.7
    )
    expected=integration.assemble_traces(
        (1.,-1.),(1.,-1.),
        row_kind="gradient",column_kind="gradient",
        coefficient=1.7,
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=3e-14,atol=3e-14
    )


def test_direction_tensor_mixed_form_dispatches_to_cross_kernel():
    master,slave,integration=interface()
    beta=np.zeros((3,3,3))
    beta[0,0,0]=1.
    beta[1,1,0]=-2.
    beta[2,2,0]=.5

    @skfemntv.BilinearForm
    def form(u,v,w):
        return dot(jump(v),dot(w.beta,avg(grad(u))))

    actual=skfemntv.asm(
        form,master,slave,integration=integration,beta=beta
    )
    expected=integration.assemble_traces(
        (1.,-1.),(.5,.5),
        row_kind="value",column_kind="gradient",
        coefficient=beta,
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=3e-14,atol=3e-14
    )


def test_interface_linear_jump_has_equal_and_opposite_resultants():
    master,slave,integration=interface()
    traction=np.array([.4,-1.2,2.3])

    @skfemntv.LinearForm
    def load(v,w):
        return dot(w.traction,jump(v))

    actual=skfemntv.asm(
        load,master,slave,integration=integration,traction=traction
    )
    expected=integration.assemble_linear_trace(
        (1.,-1.),coefficient=traction
    )
    np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-14)
    master_resultant=actual[:master.N].reshape(-1,3).sum(axis=0)
    slave_resultant=actual[master.N:].reshape(-1,3).sum(axis=0)
    np.testing.assert_allclose(
        master_resultant+slave_resultant,0.,atol=3e-14
    )


def test_interface_linear_average_full_gradient_is_composable():
    master,slave,integration=interface()
    tensor=np.arange(9,dtype=float).reshape(3,3)/7.

    @skfemntv.LinearForm
    def load(v,w):
        return ddot(w.tensor,avg(grad(v),weights=(.25,.75)))

    actual=load.assemble(
        master,slave,integration=integration,tensor=tensor
    )
    expected=integration.assemble_linear_trace(
        (.25,.75),trace_kind="gradient",coefficient=tensor
    )
    np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-14)


def test_interface_linear_normal_gradient_uses_both_sides():
    master,slave,integration=interface()
    flux=np.array([1.,-.5,.25])

    @skfemntv.LinearForm
    def load(v,w):
        return dot(w.flux,jump(normal_grad(v)))

    actual=skfemntv.asm(
        load,master,slave,integration=integration,flux=flux
    )
    expected=integration.assemble_linear_trace(
        (1.,-1.),trace_kind="normal_gradient",coefficient=flux
    )
    np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-14)


def test_interface_linear_coordinate_context_evaluates_at_overlap_points():
    master,slave,integration=interface()

    def load(x):
        return np.stack((1.+x[0],x[1]**2,-x[2]),axis=0)

    @skfemntv.LinearForm
    def form(v,w):
        return dot(load(w.x),jump(v))

    actual=skfemntv.asm(
        form,master,slave,integration=integration
    )
    coefficient=load(np.moveaxis(
        integration.global_coordinates,-1,0
    ))
    expected=integration.assemble_linear_trace(
        (1.,-1.),coefficient=coefficient
    )
    np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-14)


def test_interface_master_normal_is_available_to_linear_form():
    master,slave,integration=interface()

    @skfemntv.LinearForm
    def form(v,w):
        return dot(w.n_master,jump(v))

    actual=form.assemble(
        master,slave,integration=integration
    )
    expected=integration.assemble_linear_trace(
        (1.,-1.),
        coefficient=np.moveaxis(integration.master_normals,-1,0),
    )
    np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(
        np.linalg.norm(integration.master_normals,axis=2),1.,
        rtol=2e-14,atol=2e-14,
    )


def test_interface_gap_can_weight_bilinear_form():
    master,slave,integration=interface()

    @skfemntv.BilinearForm
    def form(u,v,w):
        return (1.+w.gap**2)*dot(jump(u),jump(v))

    actual=skfemntv.asm(
        form,master,slave,integration=integration
    )
    expected=integration.assemble_traces(
        (1.,-1.),(1.,-1.),coefficient=1.+integration.gap**2
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=3e-14,atol=3e-14
    )


def test_user_callable_composes_parameters_coordinates_gap_and_normal():
    master,slave,integration=interface()

    def pressure_law(x,gap,scale):
        return scale*(1.+x[0]+gap**2)

    @skfemntv.LinearForm
    def form(v,w):
        traction=(
            w.pressure_law(w.x,w.gap,w.scale)*w.n_master
        )
        return dot(traction,jump(v))

    actual=skfemntv.asm(
        form,master,slave,integration=integration,
        pressure_law=pressure_law,scale=1.7,
    )
    pressure=pressure_law(
        np.moveaxis(integration.global_coordinates,-1,0),
        integration.gap,1.7,
    )
    expected=integration.assemble_linear_trace(
        (1.,-1.),
        coefficient=np.moveaxis(
            pressure[...,None]*integration.master_normals,-1,0
        ),
    )
    np.testing.assert_allclose(actual,expected,rtol=3e-14,atol=3e-14)


def test_user_parameters_compose_with_numpy_gap_expression():
    master,slave,integration=interface()

    @skfemntv.BilinearForm
    def form(u,v,w):
        coefficient=w.alpha*np.exp(-w.gap/w.length)
        return coefficient*dot(jump(u),jump(v))

    actual=skfemntv.asm(
        form,master,slave,integration=integration,
        alpha=2.5,length=.3,
    )
    expected=integration.assemble_traces(
        (1.,-1.),(1.,-1.),
        coefficient=2.5*np.exp(-integration.gap/.3),
    )
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=3e-14,atol=3e-14
    )
