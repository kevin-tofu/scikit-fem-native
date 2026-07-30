import numpy as np

import skfn
from skfn.helpers import avg, dot, jump, normal_grad


def interface():
    linear=skfn.MeshTet()
    quadratic=skfn.MeshTet2.from_mesh(linear)
    master=skfn.FacetBasis(
        quadratic,skfn.ElementVector(skfn.ElementTetP2()),intorder=4
    )
    slave=skfn.FacetBasis(
        linear,skfn.ElementVector(skfn.ElementTetP1()),intorder=4
    )
    integration=skfn.TriangleSupermesh.from_facets(master,slave)
    return master,slave,integration


def test_jump_form_matches_trace_block_assembly():
    master,slave,integration=interface()

    @skfn.BilinearForm
    def form(u,v,w):
        return w.alpha*dot(jump(u),jump(v))

    actual=skfn.asm(
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

    @skfn.BilinearForm
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

    @skfn.BilinearForm
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
