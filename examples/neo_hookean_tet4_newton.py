"""Small total-Lagrangian Neo-Hookean Tet4 solve with an external Newton loop."""

import numpy as np
from scipy.sparse.linalg import spsolve

import skfemntv as fem


def main():
    points=np.array([
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
        [0.,0.,0.,1.],
    ])
    cells=np.array([[0],[1],[2],[3]],dtype=np.int64)
    mesh=fem.MeshTet(points,cells)
    basis=fem.Basis(
        mesh,fem.ElementVector(fem.ElementTetP1(),dim=3),intorder=2
    )
    assembler=fem.NeoHookeanTet4Assembler(basis,mu=2.,lambda_=3.)

    # Clamp the base triangle and pull the top vertex in the z direction.
    fixed_nodes=np.array([0,1,2],dtype=np.int64)
    fixed_dofs=np.concatenate([
        3*fixed_nodes+component for component in range(3)
    ])
    free=np.setdiff1d(np.arange(basis.N),fixed_dofs)
    external=np.zeros(basis.N)
    external[3*3+2]=.03
    displacement=np.zeros(basis.N)

    for iteration in range(1,13):
        state=assembler.evaluate(displacement)
        residual=state.residual-external
        free_norm=np.linalg.norm(residual[free])
        if free_norm<1e-11:
            break
        displacement[free]+=spsolve(
            state.tangent[free][:,free],-residual[free]
        )
    else:
        raise RuntimeError("Newton iteration did not converge")

    state=assembler.evaluate(displacement)
    print(f"Newton iterations: {iteration-1}")
    print(f"top z displacement: {displacement[11]:.8e}")
    print(f"strain energy: {state.energy:.8e}")
    print(f"free residual norm: {np.linalg.norm((state.residual-external)[free]):.3e}")


if __name__=="__main__":
    main()
