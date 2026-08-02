import numpy as np
import pytest

import skfemntv
from skfemntv.helpers import dot,jump


def _disjoint_interfaces(count=200):
    points=[];triangles=[]
    for entity in range(count):
        x=2.*entity
        first=len(points)
        points.extend(((x,0.,0.),(x+1.,0.,0.),(x,1.,0.)))
        triangles.append((first,first+1,first+2))
    points=np.asarray(points,dtype=float).T
    triangles=np.asarray(triangles,dtype=np.int64).T
    return skfemntv.TriangleSupermesh(
        points,triangles,points,triangles
    )


def test_parallel_cross_scatter_is_bitwise_deterministic():
    integration=_disjoint_interfaces()
    serial=integration.assemble(num_threads=1).copy()
    for _ in range(3):
        parallel=integration.assemble(num_threads=4).copy()
        assert np.array_equal(serial.indptr,parallel.indptr)
        assert np.array_equal(serial.indices,parallel.indices)
        assert np.array_equal(serial.data,parallel.data)


@pytest.mark.parametrize("multiplier",["slave","master","overlap_p0","dual"])
def test_parallel_mortar_spaces_match_serial(multiplier):
    integration=_disjoint_interfaces()
    serial=integration.assemble_mortar(multiplier,num_threads=1)
    parallel=integration.assemble_mortar(multiplier,num_threads=4)
    for serial_block,parallel_block in zip(
        (serial.master_matrix,serial.slave_matrix,serial.coupling_matrix),
        (parallel.master_matrix,parallel.slave_matrix,parallel.coupling_matrix),
    ):
        assert np.array_equal(serial_block.indptr,parallel_block.indptr)
        assert np.array_equal(serial_block.indices,parallel_block.indices)
        assert np.array_equal(serial_block.data,parallel_block.data)


def test_interface_form_accepts_per_call_thread_count():
    integration=_disjoint_interfaces()

    @skfemntv.BilinearForm
    def penalty(u,v,w):
        return 2.5*dot(jump(u),jump(v))

    serial=skfemntv.asm(
        penalty,None,None,integration=integration,num_threads=1
    )
    parallel=skfemntv.asm(
        penalty,None,None,integration=integration,num_threads=4
    )
    np.testing.assert_array_equal(serial.indptr,parallel.indptr)
    np.testing.assert_array_equal(serial.indices,parallel.indices)
    np.testing.assert_array_equal(serial.data,parallel.data)


def test_supermesh_rejects_invalid_thread_count():
    integration=_disjoint_interfaces(1)
    with pytest.raises(ValueError,match="positive integer"):
        integration.assemble(num_threads=0)
    with pytest.raises(ValueError,match="positive integer"):
        integration.assemble_mortar(num_threads=True)
