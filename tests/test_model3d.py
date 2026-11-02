"""Verification of the 3-D corner-balancing model.

These tests are the model's certificate: energy conservation, the momentum
invariant of Proposition 1, and agreement between the analytic linearization
used for control design and a finite-difference Jacobian of the nonlinear
model.
"""

import numpy as np
import pytest

from cubli.model3d import (
    Cubli3D,
    quat_from_axis_angle,
    quat_mul,
    quat_normalize,
    quat_to_rot,
    rk4_step,
)
from cubli.params import NOMINAL


def _tilted_state(m: Cubli3D, angle: float, axis=(1.0, -1.0, 0.0), w=None, Om=None):
    q = quat_mul(m.equilibrium_quat(), quat_from_axis_angle(np.asarray(axis), angle))
    return m.state(q=q, w=w, Om=Om)


def test_equilibrium_is_an_equilibrium():
    m = Cubli3D(coulomb=False)
    x = m.state()
    dx = m.f(x, np.zeros(3))
    assert np.allclose(dx, 0.0, atol=1e-12)
    # center of mass exactly above the pivot
    n_I = quat_to_rot(x[0:4]) @ m.n_eq
    assert np.allclose(n_I, [0.0, 0.0, 1.0], atol=1e-12)


def test_energy_is_conserved_without_friction_or_torque():
    p = NOMINAL.__class__(b_pivot=0.0, b_wheel=0.0, tau_coulomb=0.0)
    m = Cubli3D(p, coulomb=False)
    x = _tilted_state(m, np.deg2rad(25.0), w=[0.3, -0.2, 0.1], Om=[120.0, -60.0, 40.0])
    E0 = m.energy(x)
    dt = 2e-4
    for _ in range(int(5.0 / dt)):
        x = rk4_step(m, x, np.zeros(3), dt)
    assert abs(m.energy(x) - E0) / abs(E0) < 1e-8


def test_vertical_momentum_is_conserved_even_under_wheel_torque():
    """Proposition 1: internal actuators cannot change the vertical momentum."""
    p = NOMINAL.__class__(b_pivot=0.0, b_wheel=0.0, tau_coulomb=0.0)
    m = Cubli3D(p, coulomb=False)
    x = _tilted_state(m, np.deg2rad(15.0), w=[0.1, 0.2, -0.1], Om=[50.0, 0.0, -30.0])
    hz0 = m.momentum_vertical(x)
    dt = 2e-4
    rng = np.random.default_rng(0)
    for k in range(int(3.0 / dt)):
        tau = 0.3 * np.sin(2.0 * np.pi * np.array([1.0, 2.0, 3.0]) * k * dt)
        tau = tau + 0.05 * rng.standard_normal(3)
        x = rk4_step(m, x, tau, dt)
    assert abs(m.momentum_vertical(x) - hz0) < 1e-9


def test_pivot_friction_does_dissipate_vertical_momentum():
    """Sanity of the converse: only external torques can drain the budget."""
    m = Cubli3D(NOMINAL.__class__(b_pivot=5e-2), coulomb=False)
    x = _tilted_state(m, np.deg2rad(5.0), w=[0.0, 0.0, 0.5])
    hz0 = m.momentum_vertical(x)
    dt = 5e-4
    for _ in range(int(2.0 / dt)):
        x = rk4_step(m, x, np.zeros(3), dt)
    assert abs(m.momentum_vertical(x)) < abs(hz0)


@pytest.mark.parametrize("Om0", [np.zeros(3), np.array([400.0, -200.0, 100.0])])
def test_analytic_linearization_matches_numerical_jacobian(Om0):
    p = NOMINAL.__class__(tau_coulomb=0.0)
    m = Cubli3D(p, coulomb=False)
    h0 = m.Jw @ Om0
    A, B = m.linearize(h0)

    def err_dot(dth, w, h, tau):
        q = quat_mul(m.equilibrium_quat(),
                     quat_normalize(np.concatenate([[1.0], 0.5 * dth])))
        x = m.state(q=q, w=w, Om=m.Jw_inv @ h)
        dx = m.f(x, tau)
        # d(theta_e)/dt = w to first order; hdot = Jw Omdot
        return np.concatenate([w, dx[4:7], m.Jw @ dx[7:10]])

    z0 = np.concatenate([np.zeros(3), np.zeros(3), h0])
    u0 = np.zeros(3)
    eps = 1e-6

    An = np.zeros((9, 9))
    for i in range(9):
        dz = np.zeros(9)
        dz[i] = eps
        zp, zm = z0 + dz, z0 - dz
        fp = err_dot(zp[0:3], zp[3:6], zp[6:9], u0)
        fm = err_dot(zm[0:3], zm[3:6], zm[6:9], u0)
        An[:, i] = (fp - fm) / (2 * eps)

    Bn = np.zeros((9, 3))
    for i in range(3):
        du = np.zeros(3)
        du[i] = eps
        fp = err_dot(z0[0:3], z0[3:6], z0[6:9], u0 + du)
        fm = err_dot(z0[0:3], z0[3:6], z0[6:9], u0 - du)
        Bn[:, i] = (fp - fm) / (2 * eps)

    assert np.allclose(A, An, atol=1e-5, rtol=1e-4)
    assert np.allclose(B, Bn, atol=1e-5, rtol=1e-4)


def test_open_loop_is_unstable_in_exactly_two_attitude_directions():
    """Gravity destabilises the two directions orthogonal to the diagonal;
    rotation about the vertical is neutrally stable -- that is the direction
    where the momentum budget, not gravity, decides the outcome."""
    m = Cubli3D(coulomb=False)
    A, _ = m.linearize()
    ev = np.linalg.eigvals(A)
    n_unstable = int(np.sum(ev.real > 1e-6))
    assert n_unstable == 2


def test_the_plant_is_stabilisable_but_not_controllable():
    """Proposition 1 seen through the linear algebra.

    With an ideal pivot exactly one mode is uncontrollable, and its left
    annihilator is the vertical component of the total angular momentum -- the
    quantity the wheels provably cannot change.
    """
    from cubli.lqr_utils import controllable_subspace, momentum_annihilator

    p = NOMINAL.__class__(b_pivot=0.0, b_wheel=0.0, tau_coulomb=0.0)
    m = Cubli3D(p, coulomb=False)
    A, B = m.linearize()
    _, Tu, rank = controllable_subspace(A, B)
    assert rank == 8
    assert Tu.shape[1] == 1

    v = momentum_annihilator(m)
    v = v / np.linalg.norm(v)
    # the uncontrollable direction is spanned by the momentum invariant
    assert abs(abs(float(v @ Tu[:, 0])) - 1.0) < 1e-9
    # no wheel torque reaches it, and gravity does not either
    assert np.allclose(v @ B, 0.0, atol=1e-12)
    assert np.allclose(v @ A, 0.0, atol=1e-12)


def test_the_invariant_stays_numerically_uncontrollable_with_friction():
    """Pivot friction does not buy back authority over the stored momentum.

    In exact arithmetic a dissipative pivot makes the mode controllable, but
    the corresponding controllability singular value stays at the level of
    numerical noise for any physically plausible friction coefficient: the
    authority is not usable.  This is why the LQR is synthesised on the
    controllable subspace instead of on the full model, where the Riccati
    equation has no finite solution.
    """
    from cubli.lqr_utils import controllable_subspace

    for b in (1e-2, 1e-3, 1e-4, 0.0):
        m = Cubli3D(NOMINAL.__class__(b_pivot=b, tau_coulomb=0.0), coulomb=False)
        A, B = m.linearize()
        C = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(9)])
        sv = np.linalg.svd(C, compute_uv=False)
        assert sv[-1] / sv[0] < 1e-12
        assert controllable_subspace(A, B)[2] == 8


def test_reduction_to_the_two_dimensional_edge_case():
    """Rotation confined to one body axis reproduces the planar model."""
    from cubli.model2d import Cubli2D

    p = NOMINAL.__class__(b_pivot=0.0, b_wheel=0.0, tau_coulomb=0.0)
    m3 = Cubli3D(p, coulomb=False)
    m2 = Cubli2D(p, coulomb=False)
    # gravity torque magnitude at the same tilt must use the same lever arm law
    assert m3.Mgl == pytest.approx(p.M * 9.81 * p.corner["l"])
    assert m2.Mgl == pytest.approx(p.edge["m_gl"])
    # the planar inertia is the corner inertia projected on the edge axis
    axis = p.edge["axis"]
    assert float(axis @ p.corner["J"] @ axis) == pytest.approx(p.edge["I"], rel=1e-9)
