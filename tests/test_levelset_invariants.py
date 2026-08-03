import itertools

import numpy as np
import pytest

import skfemntv


def _simplex_measure(points):
    dimension=points.shape[0]
    jacobian=points[:,1:]-points[:,[0]]
    divisor=2. if dimension==2 else 6.
    return abs(np.linalg.det(jacobian))/divisor


@pytest.mark.parametrize("dimension",[2,3])
def test_all_simplex_sign_patterns_partition_measure(dimension):
    mesh=skfemntv.MeshTri() if dimension==2 else skfemntv.MeshTet()
    total=_simplex_measure(mesh.p[:,mesh.t[:,0]])
    magnitudes=np.linspace(.7,1.3,dimension+1)

    for signs in itertools.product((-1.,1.),repeat=dimension+1):
        values=np.asarray(signs)*magnitudes
        level_set=skfemntv.LevelSet(values,tolerance=0.)
        inside=level_set.cut_quadrature(mesh,intorder=2)
        outside=level_set.cut_quadrature(mesh,side="outside",intorder=2)
        interface=level_set.interface_quadrature(mesh,intorder=2)

        assert inside.weights.sum()+outside.weights.sum()==pytest.approx(
            total,rel=2.e-13,abs=2.e-14
        )
        assert np.all(inside.weights>0.)
        assert np.all(outside.weights>0.)
        for rule in (inside,outside,interface):
            assert rule.cell_offsets[0]==0
            assert rule.cell_offsets[-1]==len(rule.weights)
            assert np.all(np.diff(rule.cell_offsets)>=0)
            assert np.all(np.isfinite(rule.points))
            assert np.all(np.isfinite(rule.weights))
            assert np.all(np.isfinite(rule.normals))
        crossing=len(set(signs))==2
        assert (len(interface.weights)>0)==crossing


@pytest.mark.parametrize("dimension",[2,3])
def test_rigid_transform_and_scale_covariance(dimension):
    rng=np.random.default_rng(20260803+dimension)
    base=skfemntv.MeshTri() if dimension==2 else skfemntv.MeshTet()
    values=np.linspace(-1.1,.9,dimension+1)
    original=skfemntv.LevelSet(values,tolerance=0.)
    volume=original.cut_quadrature(base,intorder=3)
    surface=original.interface_quadrature(base,intorder=3)
    orthogonal,_=np.linalg.qr(rng.normal(size=(dimension,dimension)))
    if np.linalg.det(orthogonal)<0.:orthogonal[:,0]*=-1.
    scale=3.7;translation=rng.normal(size=dimension)
    transformed_points=scale*orthogonal@base.p+translation[:,None]
    mesh=type(base)(transformed_points,base.t.copy())
    transformed=skfemntv.LevelSet(values,tolerance=0.)
    transformed_volume=transformed.cut_quadrature(mesh,intorder=3)
    transformed_surface=transformed.interface_quadrature(mesh,intorder=3)

    assert transformed_volume.weights.sum()==pytest.approx(
        scale**dimension*volume.weights.sum(),rel=3.e-13
    )
    assert transformed_surface.weights.sum()==pytest.approx(
        scale**(dimension-1)*surface.weights.sum(),rel=3.e-13
    )
    original_moment=volume.weights@volume.points
    expected_moment=scale**dimension*(
        scale*orthogonal@original_moment
        +translation*volume.weights.sum()
    )
    np.testing.assert_allclose(
        transformed_volume.weights@transformed_volume.points,
        expected_moment,rtol=3.e-13,atol=3.e-13,
    )
    np.testing.assert_allclose(
        transformed_surface.normals,
        (orthogonal@surface.normals.T).T,
        rtol=2.e-13,atol=2.e-13,
    )


def test_tet_global_relabel_and_local_permutation_are_invariant():
    mesh=skfemntv.MeshTet()
    values=np.array([-.8,.4,1.2,-.3])
    reference=skfemntv.LevelSet(values,tolerance=0.)
    reference_inside=reference.cut_quadrature(mesh,intorder=3)
    reference_surface=reference.interface_quadrature(mesh,intorder=3)
    permutation=np.array([2,0,3,1])
    inverse=np.empty_like(permutation);inverse[permutation]=np.arange(4)
    relabeled=skfemntv.MeshTet(
        mesh.p[:,permutation],inverse[mesh.t]
    )
    relabeled_values=values[permutation]
    other=skfemntv.LevelSet(relabeled_values,tolerance=0.)

    assert other.cut_quadrature(relabeled,intorder=3).weights.sum()==pytest.approx(
        reference_inside.weights.sum(),rel=2.e-13
    )
    assert other.interface_quadrature(
        relabeled,intorder=3
    ).weights.sum()==pytest.approx(reference_surface.weights.sum(),rel=2.e-13)


def test_circle_volume_and_interface_measure_converge_under_refinement():
    radius=.31;exact_area=np.pi*radius**2;exact_length=2.*np.pi*radius
    errors=[]
    for resolution in (8,16,32):
        axis=np.linspace(0.,1.,resolution+1)
        mesh=skfemntv.MeshTri.init_tensor(axis,axis)
        level_set=skfemntv.LevelSet(
            lambda x:(x[0]-.5)**2+(x[1]-.5)**2-radius**2,
            tolerance=0.,
        )
        area=level_set.cut_quadrature(mesh,intorder=2).weights.sum()
        length=level_set.interface_quadrature(mesh,intorder=2).weights.sum()
        errors.append((abs(area-exact_area),abs(length-exact_length)))
    assert errors[1][0]<errors[0][0] and errors[2][0]<errors[1][0]
    assert errors[1][1]<errors[0][1] and errors[2][1]<errors[1][1]
    assert errors[-1][0]<2.e-3
    assert errors[-1][1]<4.e-3


def test_random_tet_planes_preserve_partition_and_normal_orientation():
    rng=np.random.default_rng(41517);mesh=skfemntv.MeshTet()
    total=1./6.
    for _ in range(40):
        normal=rng.normal(size=3);normal/=np.linalg.norm(normal)
        offset=rng.uniform(.15,.55)
        values=normal@mesh.p-offset
        if np.all(values>0.) or np.all(values<0.):continue
        level_set=skfemntv.LevelSet(values,tolerance=0.)
        inside=level_set.cut_quadrature(mesh,intorder=2)
        outside=level_set.cut_quadrature(mesh,side="outside",intorder=2)
        interface=level_set.interface_quadrature(mesh,intorder=2)
        assert inside.weights.sum()+outside.weights.sum()==pytest.approx(
            total,abs=3.e-14
        )
        np.testing.assert_allclose(
            interface.normals,
            np.broadcast_to(normal,interface.normals.shape),
            rtol=3.e-13,atol=3.e-13,
        )


def test_sphere_volume_and_surface_measure_converge_under_refinement():
    radius=.28
    exact_volume=4.*np.pi*radius**3/3.
    exact_area=4.*np.pi*radius**2
    errors=[]
    for resolution in (3,5,7):
        axis=np.linspace(0.,1.,resolution+1)
        mesh=skfemntv.MeshTet.init_tensor(axis,axis,axis)
        level_set=skfemntv.LevelSet(
            lambda x:(x[0]-.5)**2+(x[1]-.5)**2+(x[2]-.5)**2-radius**2,
            tolerance=0.,
        )
        volume=level_set.cut_quadrature(mesh,intorder=2).weights.sum()
        area=level_set.interface_quadrature(mesh,intorder=2).weights.sum()
        errors.append((abs(volume-exact_volume),abs(area-exact_area)))
    assert errors[1][0]<errors[0][0] and errors[2][0]<errors[1][0]
    assert errors[1][1]<errors[0][1] and errors[2][1]<errors[1][1]
