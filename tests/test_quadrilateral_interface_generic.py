import numpy as np
import pytest

import skfemntv


def _surface(mesh,element,height):
    facets=mesh.facets_satisfying(
        lambda x: np.isclose(x[2],height),boundaries_only=True
    )
    return skfemntv.FacetBasis(
        mesh,skfemntv.ElementVector(element),facets=facets,intorder=4
    )


def test_interface_supermesh_assembles_nonmatching_quad_traces():
    master_mesh=skfemntv.MeshHex.init_tensor([0.,1.],[0.,1.],[0.,1.])
    slave_mesh=skfemntv.MeshHex.init_tensor(
        [0.,.5,1.],[0.,.5,1.],[1.,2.]
    )
    master=_surface(master_mesh,skfemntv.ElementHex1(),1.)
    slave=_surface(slave_mesh,skfemntv.ElementHex1(),1.)
    integration=skfemntv.InterfaceSupermesh.from_facets(master,slave)

    assert integration.diagnostics.master_search_triangle_count==2
    assert integration.diagnostics.slave_search_triangle_count==8
    assert integration.diagnostics.overlap_area==pytest.approx(1.)
    np.testing.assert_allclose(
        integration.assemble()@np.ones(slave.N),
        skfemntv.NativeLinearForm(master).assemble(value=np.ones(3))[0],
        rtol=4.e-13,atol=4.e-13,
    )
