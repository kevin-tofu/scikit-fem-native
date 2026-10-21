import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

import skfemntv
from skfemntv.helpers import ddot,div,sym_grad


def assembler():
    points=np.array([
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
        [0.,0.,0.,1.],
    ])
    cells=np.array([[0],[1],[2],[3]],dtype=np.int64)
    mesh=skfemntv.MeshTet(points,cells)
    basis=skfemntv.Basis(
        mesh,skfemntv.ElementVector(skfemntv.ElementTetP1(),dim=3),
        intorder=2,
    )
    return basis,skfemntv.NeoHookeanTet4Assembler(basis,mu=2.,lambda_=3.)


def test_reference_state_has_zero_force_energy_and_symmetric_tangent():
    basis,model=assembler()
    result=model.evaluate(np.zeros(basis.N))
    np.testing.assert_allclose(result.residual,0.,atol=1e-14)
    assert result.energy==pytest.approx(0.,abs=1e-14)
    np.testing.assert_allclose(
        result.tangent.toarray(),result.tangent.toarray().T,
        rtol=1e-13,atol=1e-13,
    )


def test_consistent_tangent_matches_residual_directional_difference():
    basis,model=assembler()
    rng=np.random.default_rng(27)
    displacement=.04*rng.normal(size=basis.N)
    direction=rng.normal(size=basis.N)
    result=model.evaluate(displacement)
    step=1e-7
    finite_difference=(
        model.evaluate(displacement+step*direction).residual
        -model.evaluate(displacement-step*direction).residual
    )/(2.*step)
    energy_difference=(
        model.evaluate(displacement+step*direction).energy
        -model.evaluate(displacement-step*direction).energy
    )/(2.*step)
    tangent_action=result.tangent@direction
    np.testing.assert_allclose(
        tangent_action,finite_difference,rtol=2e-7,atol=2e-8
    )
    assert energy_difference==pytest.approx(
        float(result.residual@direction),rel=2e-7,abs=2e-9
    )


def test_zero_displacement_tangent_matches_linear_elasticity():
    basis,model=assembler()

    @skfemntv.BilinearForm
    def linear_elasticity(u,v,w):
        return 2.*model.mu*ddot(sym_grad(u),sym_grad(v)) \
            +model.lambda_*div(u)*div(v)

    expected=linear_elasticity.assemble(basis)
    actual=model.evaluate(np.zeros(basis.N)).tangent
    np.testing.assert_allclose(
        actual.toarray(),expected.toarray(),rtol=2e-13,atol=2e-13
    )


def test_newton_solve_converges_for_a_displacement_controlled_tet():
    basis,model=assembler()
    fixed_nodes=np.array([0,1,2],dtype=np.int64)
    fixed_dofs=np.concatenate([
        3*fixed_nodes+component for component in range(3)
    ])
    free=np.setdiff1d(np.arange(basis.N),fixed_dofs)
    external=np.zeros(basis.N)
    external[3*3+2]=0.03
    displacement=np.zeros(basis.N)
    for _ in range(12):
        result=model.evaluate(displacement)
        residual=result.residual-external
        if np.linalg.norm(residual[free])<1e-11:
            break
        increment=np.zeros_like(displacement)
        increment[free]=spsolve(
            result.tangent[free][:,free],-residual[free]
        )
        displacement+=increment
    else:
        pytest.fail("Newton iteration did not converge")
    assert displacement[3*3+2]>0.
    assert model.evaluate(displacement).energy>0.


def test_nonpositive_deformation_jacobian_is_rejected():
    basis,model=assembler()
    displacement=np.zeros(basis.N)
    # Reflect the fourth node through the base plane.
    displacement[3*3+2]=-2.
    with pytest.raises(ValueError,match=r"nonpositive det\(F\)"):
        model.evaluate(displacement)
