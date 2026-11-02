"""Full nonlinear dynamics of the 3-D Cubli balancing on a corner.

Rigid cube pivoting about a fixed vertex (ideal spherical joint), actuated by
three orthogonal reaction wheels.  No small-angle assumption anywhere: the
attitude is carried by a unit quaternion and the equations are the exact
Euler equations of a rigid body with internal momentum wheels.

State (10)
----------
``x = [q (4), w (3), Om (3)]``

* ``q``  -- unit quaternion, scalar first, rotating **body -> inertial**.
* ``w``  -- body angular velocity, body frame [rad/s].
* ``Om`` -- wheel speeds **relative to the body**, along the body axes [rad/s].

Equations
---------
Total angular momentum about the pivot, in body frame::

    h = J w + Jw Om

External torques about the pivot are gravity and pivot friction only (the
wheel torques are *internal*), hence::

    J wdot + Jw Omdot = tau_g - w x h - b_pivot w                      (1)
    Jw (Omdot + wdot)  = tau - b_wheel Om - tau_c sign(Om)             (2)

Eliminating ``Omdot`` gives the form actually integrated::

    Jb wdot = tau_g - w x h - b_pivot w - tau + b_wheel Om + tau_c sign(Om)
    Omdot   = Jw^-1 (tau - b_wheel Om - tau_c sign(Om)) - wdot

with ``Jb = J - Jw``.  Setting ``tau = 0`` and all friction to zero makes the
system conservative, which is the first unit test.

Proposition 1 (momentum budget)
-------------------------------
Gravity acts through the center of mass, so its torque about the pivot,
``r_c x (M g)``, is always **horizontal**.  With a frictionless pivot the
vertical component of the inertial angular momentum is therefore conserved::

    d/dt ( (R h) . e_z ) = 0.

The wheels are internal actuators: they can only *move* that momentum into
themselves, never remove it.  With ``|Om_i| <= Om_max`` this yields a hard
bound on the set of recoverable initial conditions -- the analytical backbone
of the momentum-budget benchmark.  ``momentum_vertical`` measures it and
``h_budget_margin`` reports how much of the budget is left.
"""

from __future__ import annotations

import numpy as np

from math import tanh as _tanh

from .params import G, CubliParams, NOMINAL, _skew

# --------------------------------------------------------------------------
# quaternion helpers (scalar first, body -> inertial)
# --------------------------------------------------------------------------


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrix ``R`` with ``v_inertial = R @ v_body``."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion rotating unit vector ``a`` onto unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = float(a @ b)
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:  # antiparallel: any orthogonal axis
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return quat_from_axis_angle(axis, np.pi)
    axis = np.cross(a, b)
    return quat_from_axis_angle(axis, np.arccos(np.clip(c, -1.0, 1.0)))


# --------------------------------------------------------------------------


class Cubli3D:
    """Nonlinear corner-balancing model."""

    NQ = 4
    NX = 10
    NU = 3

    def __init__(self, p: CubliParams = NOMINAL, coulomb: bool = True,
                 smooth_coulomb: float = 1.0):
        self.p = p
        self.J = p.corner["J"]
        self.Jb = p.corner["Jb"]
        self.Jw = p.corner["Jw"]
        self.Jb_inv = np.linalg.inv(self.Jb)
        self.Jw_inv = np.linalg.inv(self.Jw)
        r_c = np.array(p.corner["r_c"], dtype=float)
        if p.com_offset != 0.0:
            # A real cube is never perfectly symmetric.  The offset moves the
            # true equilibrium away from the body diagonal, so a controller
            # built on the nominal geometry must hold a permanent torque --
            # and a permanent torque means a permanently accelerating wheel.
            r_c = r_c + p.com_offset * np.eye(3)[p.com_offset_dir % 3]
        self.r_c = r_c
        self.n_eq = p.corner["n_eq"]
        self.l = p.corner["l"]
        self.Mgl = p.M * G * p.corner["l"]
        self.coulomb = coulomb
        # tanh smoothing width [rad/s] -- a stiff sign() would dominate the
        # integration error budget and is not what a real bearing does anyway.
        self.smooth = smooth_coulomb
        self.g_I = np.array([0.0, 0.0, -G])
        self._cache_scalars()

    # -- kinematics ------------------------------------------------------
    @staticmethod
    def unpack(x: np.ndarray):
        return x[0:4], x[4:7], x[7:10]

    def equilibrium_quat(self) -> np.ndarray:
        """Attitude for which the center of mass sits straight above the pivot."""
        return quat_between(self.n_eq, np.array([0.0, 0.0, 1.0]))

    def state(self, q=None, w=None, Om=None) -> np.ndarray:
        q = self.equilibrium_quat() if q is None else quat_normalize(np.asarray(q, float))
        w = np.zeros(3) if w is None else np.asarray(w, float)
        Om = np.zeros(3) if Om is None else np.asarray(Om, float)
        return np.concatenate([q, w, Om])

    def tilt_angle(self, x: np.ndarray) -> float:
        """Angle between the balancing diagonal and the vertical [rad].

        Only the vertical component of the rotated diagonal is needed, so the
        full rotation matrix is never formed -- this is called once per
        integration step.
        """
        q0, q1, q2, q3 = x[0], x[1], x[2], x[3]
        n1, n2, n3 = self._n
        z = (2.0 * (q1 * q3 - q2 * q0) * n1
             + 2.0 * (q2 * q3 + q1 * q0) * n2
             + (1.0 - 2.0 * (q1 * q1 + q2 * q2)) * n3)
        return float(np.arccos(z if -1.0 <= z <= 1.0 else (1.0 if z > 0 else -1.0)))

    def attitude_error(self, x: np.ndarray) -> np.ndarray:
        """Small-rotation error vector from the equilibrium, in body frame.

        ``2 * vec(q_err)`` with ``q_err = q_eq^-1 * q``; exact to first order and
        continuous through the shortest-arc sign convention.
        """
        q, _, _ = self.unpack(x)
        q_err = quat_mul(quat_conj(self.equilibrium_quat()), q)
        if q_err[0] < 0.0:
            q_err = -q_err
        return 2.0 * q_err[1:]

    # -- friction --------------------------------------------------------
    def _wheel_friction(self, Om: np.ndarray) -> np.ndarray:
        tau_f = self.p.b_wheel * Om
        if self.coulomb:
            tau_f = tau_f + self.p.tau_coulomb * np.tanh(Om / self.smooth)
        return tau_f

    # -- dynamics --------------------------------------------------------
    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """Gravity torque about the pivot, body frame [N.m]."""
        g_b = quat_to_rot(q).T @ self.g_I
        return self.p.M * np.cross(self.r_c, g_b)

    def f(self, x: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """State derivative for wheel torques ``tau`` (already saturated)."""
        q, w, Om = self.unpack(x)
        q = quat_normalize(q)

        h = self.J @ w + self.Jw @ Om
        tau_g = self.gravity_torque(q)
        tau_f = self._wheel_friction(Om)

        wdot = self.Jb_inv @ (
            tau_g - np.cross(w, h) - self.p.b_pivot * w - tau + tau_f
        )
        Omdot = self.Jw_inv @ (tau - tau_f) - wdot

        qdot = 0.5 * quat_mul(q, np.concatenate([[0.0], w]))
        return np.concatenate([qdot, wdot, Omdot])

    # -- fast path -------------------------------------------------------
    # ``f`` above is the reference implementation: it reads like the equations
    # and is what the derivation in the paper says.  It is also ~30x too slow
    # for 10^4 Monte-Carlo runs, because every call allocates a dozen small
    # numpy arrays.  ``f_fast`` is the same equations written on scalars, with
    # the constant matrix entries hoisted into the constructor.  A unit test
    # asserts the two agree to 1e-12 on random states, so the speed does not
    # cost correctness.
    def _cache_scalars(self) -> None:
        Ji = self.Jb_inv
        self._ji = (float(Ji[0, 0]), float(Ji[0, 1]), float(Ji[0, 2]),
                    float(Ji[1, 1]), float(Ji[1, 2]), float(Ji[2, 2]))
        J = self.J
        self._j = (float(J[0, 0]), float(J[0, 1]), float(J[0, 2]),
                   float(J[1, 1]), float(J[1, 2]), float(J[2, 2]))
        self._ja = float(self.Jw[0, 0])
        self._rc = (float(self.r_c[0]), float(self.r_c[1]), float(self.r_c[2]))
        self._mg = float(self.p.M * G)
        self._n = (float(self.n_eq[0]), float(self.n_eq[1]), float(self.n_eq[2]))
        self._bp = float(self.p.b_pivot)
        self._bw = float(self.p.b_wheel)
        self._tc = float(self.p.tau_coulomb) if self.coulomb else 0.0
        self._sm = float(self.smooth)

    def f_fast(self, x, tau, out=None):
        """Scalar rewrite of :meth:`f` (identical equations, no allocations)."""
        q0, q1, q2, q3 = x[0], x[1], x[2], x[3]
        w1, w2, w3 = x[4], x[5], x[6]
        O1, O2, O3 = x[7], x[8], x[9]
        t1, t2, t3 = tau[0], tau[1], tau[2]
        j00, j01, j02, j11, j12, j22 = self._j
        i00, i01, i02, i11, i12, i22 = self._ji
        ja, bp, bw, tc, sm = self._ja, self._bp, self._bw, self._tc, self._sm
        rc1, rc2, rc3 = self._rc

        # gravity in the body frame: only the third row of R is needed
        r20 = 2.0 * (q1 * q3 - q2 * q0)
        r21 = 2.0 * (q2 * q3 + q1 * q0)
        r22 = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
        mg = self._mg
        g1, g2, g3 = -mg * r20, -mg * r21, -mg * r22      # M * g in body frame
        tg1 = rc2 * g3 - rc3 * g2
        tg2 = rc3 * g1 - rc1 * g3
        tg3 = rc1 * g2 - rc2 * g1

        # total angular momentum about the pivot
        h1 = j00 * w1 + j01 * w2 + j02 * w3 + ja * O1
        h2 = j01 * w1 + j11 * w2 + j12 * w3 + ja * O2
        h3 = j02 * w1 + j12 * w2 + j22 * w3 + ja * O3
        c1 = w2 * h3 - w3 * h2
        c2 = w3 * h1 - w1 * h3
        c3 = w1 * h2 - w2 * h1

        f1 = bw * O1 + (tc * _tanh(O1 / sm) if tc else 0.0)
        f2 = bw * O2 + (tc * _tanh(O2 / sm) if tc else 0.0)
        f3 = bw * O3 + (tc * _tanh(O3 / sm) if tc else 0.0)

        b1 = tg1 - c1 - bp * w1 - t1 + f1
        b2 = tg2 - c2 - bp * w2 - t2 + f2
        b3 = tg3 - c3 - bp * w3 - t3 + f3

        wd1 = i00 * b1 + i01 * b2 + i02 * b3
        wd2 = i01 * b1 + i11 * b2 + i12 * b3
        wd3 = i02 * b1 + i12 * b2 + i22 * b3

        Od1 = (t1 - f1) / ja - wd1
        Od2 = (t2 - f2) / ja - wd2
        Od3 = (t3 - f3) / ja - wd3

        out = np.empty(10) if out is None else out
        out[0] = 0.5 * (-q1 * w1 - q2 * w2 - q3 * w3)
        out[1] = 0.5 * (q0 * w1 + q2 * w3 - q3 * w2)
        out[2] = 0.5 * (q0 * w2 - q1 * w3 + q3 * w1)
        out[3] = 0.5 * (q0 * w3 + q1 * w2 - q2 * w1)
        out[4], out[5], out[6] = wd1, wd2, wd3
        out[7], out[8], out[9] = Od1, Od2, Od3
        return out

    # -- invariants ------------------------------------------------------
    def momentum_body(self, x: np.ndarray) -> np.ndarray:
        _, w, Om = self.unpack(x)
        return self.J @ w + self.Jw @ Om

    def momentum_inertial(self, x: np.ndarray) -> np.ndarray:
        q, _, _ = self.unpack(x)
        return quat_to_rot(q) @ self.momentum_body(x)

    def momentum_vertical(self, x: np.ndarray) -> float:
        """Conserved quantity of Proposition 1 (ideal pivot) [N.m.s]."""
        return float(self.momentum_inertial(x)[2])

    def wheel_momentum(self, x: np.ndarray) -> np.ndarray:
        _, _, Om = self.unpack(x)
        return self.Jw @ Om

    def h_budget_margin(self, x: np.ndarray) -> float:
        """Fraction of the wheel momentum budget still available in ``[0, 1]``."""
        _, _, Om = self.unpack(x)
        return float(1.0 - np.max(np.abs(Om)) / self.p.omega_max)

    def energy(self, x: np.ndarray) -> float:
        """Total mechanical energy (kinetic + potential) [J]."""
        q, w, Om = self.unpack(x)
        T = 0.5 * w @ self.J @ w + Om @ self.Jw @ w + 0.5 * Om @ self.Jw @ Om
        V = self.p.M * G * float((quat_to_rot(q) @ self.r_c)[2])
        return float(T + V)

    # -- linearization ---------------------------------------------------
    def linearize(self, h0: np.ndarray | None = None):
        """Analytic linearization about the corner equilibrium.

        States ``[theta_e (3), w (3), h_w (3)]``, input ``tau (3)``, where
        ``h_w = Jw Om`` is the **stored wheel momentum**.  Using momentum
        rather than wheel speed is both the natural variable of this study and
        a numerical necessity: ``Jw^-1`` is of order 1e4, and a state vector
        mixing radians with kilorad/s makes the Riccati equation ill-scaled.

        ``h0`` is the momentum already stored at the operating point.  It
        enters through the gyroscopic term ``[h0]x``, so the linear model --
        and therefore the LQR gain -- depends on how full the budget is.
        """
        h0 = np.zeros(3) if h0 is None else np.asarray(h0, float)
        n = self.n_eq
        P = np.eye(3) - np.outer(n, n)          # gravity acts only off-diagonal
        Kg = self.Mgl * P                        # d(tau_g)/d(theta_e)
        Gyro = _skew(h0)                         # -w x h0 = [h0]x w
        Bp = self.p.b_pivot * np.eye(3)
        Bw = self.p.b_wheel * np.eye(3)
        BwJ = Bw @ self.Jw_inv                   # wheel friction in momentum units

        A = np.zeros((9, 9))
        B = np.zeros((9, 3))

        A[0:3, 3:6] = np.eye(3)

        Aw_th = self.Jb_inv @ Kg
        Aw_w = self.Jb_inv @ (Gyro - Bp)
        Aw_h = self.Jb_inv @ BwJ
        Bw_tau = -self.Jb_inv

        A[3:6, 0:3] = Aw_th
        A[3:6, 3:6] = Aw_w
        A[3:6, 6:9] = Aw_h
        B[3:6, :] = Bw_tau

        # hdot = tau - Bw Jw^-1 h - Jw wdot
        A[6:9, 0:3] = -self.Jw @ Aw_th
        A[6:9, 3:6] = -self.Jw @ Aw_w
        A[6:9, 6:9] = -BwJ - self.Jw @ Aw_h
        B[6:9, :] = np.eye(3) - self.Jw @ Bw_tau

        return A, B

    def error_state(self, x: np.ndarray) -> np.ndarray:
        """``[theta_e, w, h_w]`` -- the 9-vector the linear controllers act on."""
        _, w, Om = self.unpack(x)
        return np.concatenate([self.attitude_error(x), w, self.Jw @ Om])


# --------------------------------------------------------------------------


def rk4_step(model: Cubli3D, x: np.ndarray, tau: np.ndarray, dt: float) -> np.ndarray:
    """One classical RK4 step, with the quaternion renormalized afterwards."""
    ff = model.f_fast
    k1 = ff(x, tau)
    k2 = ff(x + 0.5 * dt * k1, tau)
    k3 = ff(x + 0.5 * dt * k2, tau)
    k4 = ff(x + dt * k3, tau)
    x_new = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    n = (x_new[0] * x_new[0] + x_new[1] * x_new[1]
         + x_new[2] * x_new[2] + x_new[3] * x_new[3]) ** 0.5
    x_new[0:4] /= n
    return x_new
