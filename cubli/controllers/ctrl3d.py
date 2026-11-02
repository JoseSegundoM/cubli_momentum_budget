"""The four stabilising controllers (plus a PID baseline) for corner balancing.

All of them act on the **full nonlinear plant**; none of them sees a
linearized system except the LQR, which is the declared linear-class design.
The three nonlinear laws share the same cancellation term

    Phi(x) = tau_g(q) - w x h - b_pivot w + tau_friction(Om),

so that the plant reads ``Jb wdot = Phi(x) - tau`` and the differences between
the designs are the *feedback shaping*, not the model knowledge.  That is
deliberate: it keeps the comparison about the control law rather than about
who was allowed to cancel more physics.

Every controller accepts ``momentum_aware``; see :mod:`cubli.controllers.base`.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from ..lqr_utils import lqr_stabilizable
from ..model3d import Cubli3D
from .base import Controller, MomentumManager


def _sat(v: np.ndarray) -> np.ndarray:
    return np.clip(v, -1.0, 1.0)


class _Base3D(Controller):
    """Shared plumbing: model knowledge, integral action, momentum manager.

    ``k_i`` exists to answer the obvious objection to this study: a persistent
    disturbance is textbook integral-action territory, so a comparison in which
    the nonlinear laws have no integrator looks rigged.  They can have one.
    What the experiments then show is that integral action does not help here,
    and for a reason specific to momentum exchange: the integrator responds to
    a standing attitude error by commanding *more* torque, and on this plant
    torque is precisely what fills the wheels.  It reaches the same limit
    sooner.
    """

    def __init__(self, model: Cubli3D, k_h: float = 0.0,
                 tilt_max: float = np.deg2rad(3.0), k_i: float = 0.0,
                 i_max: float = 0.15, dt: float = 2e-3):
        self.m = model
        n = model.n_eq
        self.P = np.eye(3) - np.outer(n, n)
        self.mm = MomentumManager(k_h, model.Mgl, tilt_max, self.P)
        self.k_i = float(k_i)
        self.i_max = float(i_max)
        self.dt = float(dt)
        self._int = np.zeros(3)

    def reset(self) -> None:
        self._int = np.zeros(3)

    @property
    def integral_action(self) -> bool:
        return self.k_i != 0.0

    def _integral(self, e: np.ndarray) -> np.ndarray:
        """Clamped integral of the attitude error, in torque units."""
        if self.k_i == 0.0:
            return np.zeros(3)
        self._int = np.clip(self._int + e * self.dt, -self.i_max, self.i_max)
        return self.k_i * self._int

    # -- pieces every law needs ------------------------------------------
    def _error(self, x: np.ndarray):
        e_att = self.m.attitude_error(x)
        _, w, Om = self.m.unpack(x)
        ref = self.mm.reference(self.m.Jw @ Om)
        return e_att - ref, w, Om

    def _phi(self, x: np.ndarray) -> np.ndarray:
        """Model-based cancellation term, body frame."""
        q, w, Om = self.m.unpack(x)
        h = self.m.J @ w + self.m.Jw @ Om
        return (self.m.gravity_torque(q) - np.cross(w, h)
                - self.m.p.b_pivot * w + self.m._wheel_friction(Om))

    @property
    def momentum_aware(self) -> bool:
        return self.mm.enabled


class LQR3D(_Base3D):
    """Infinite-horizon LQR on the linearization about the corner equilibrium.

    The weight on the wheel-speed states is what makes this design implicitly
    momentum-aware: the optimal gain trades attitude accuracy for keeping the
    wheels near zero speed, which is exactly the tilt-to-unload behavior the
    nonlinear laws have to be given explicitly.
    """

    name = "lqr"
    n_gains = 4

    def __init__(self, model: Cubli3D, q_att: float = 3000.0, q_rate: float = 60.0,
                 q_mom: float = 200.0, r: float = 1.0, h0=None, **kw):
        super().__init__(model, **kw)
        self._w = np.array([q_att, q_rate, q_mom, r], dtype=float)
        self.h0 = np.zeros(3) if h0 is None else np.asarray(h0, float)
        self._design()

    def _design(self) -> None:
        q_att, q_rate, q_mom, r = self._w
        A, B = self.m.linearize(self.h0)
        Q = np.diag(np.concatenate([q_att * np.ones(3), q_rate * np.ones(3),
                                    q_mom * np.ones(3)]))
        R = r * np.eye(3)
        # (A, B) is stabilisable but not controllable: the vertical momentum is
        # invariant (Proposition 1).  Design on the controllable subspace.
        self.K, self.info = lqr_stabilizable(A, B, Q, R)
        self.S = self.info["S"]

    def get_gains(self) -> np.ndarray:
        return self._w.copy()

    def set_gains(self, g: np.ndarray) -> None:
        self._w = np.abs(np.asarray(g, dtype=float))
        self._design()

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        e, w, Om = self._error(x)
        return (-self.K @ np.concatenate([e, w, self.m.Jw @ Om])
                + self._integral(e))


class FBL3D(_Base3D):
    """Exact feedback linearization with an LQR outer loop.

    Feedback linearization is a *transformation*, not a controller: canceling
    ``Phi(x)`` leaves ``wdot = v`` and something still has to choose ``v``.
    Because the cancellation is exact, the transformed plant is three decoupled
    double integrators ``(theta_e, w)``, and the outer loop is the LQR for that
    double integrator with weights ``(q_att, q_rate, r)``:

        v = -(k1 e + k2 w),   K = lqr([[0,1],[0,0]], [0;1], diag(q_att,q_rate), r)
        tau = Phi(x) + Jb (k1 e + k2 w).

    Naming the pair matters: an LQR outer loop and a hand-tuned PD have the same
    structure here, but only one of them states where its gains came from.

    The price of the design is in the zero dynamics: the wheel speeds obey
    ``Omdot = Jw^-1 (tau - tau_f) - wdot`` and are left unregulated.  On a plant
    whose actuators have a finite momentum budget that is not a technicality --
    it is the failure mode.
    """

    name = "fbl"
    n_gains = 3

    def __init__(self, model: Cubli3D, q_att: float = 8000.0, q_rate: float = 300.0,
                 r: float = 1.0, **kw):
        super().__init__(model, **kw)
        self._w = np.array([q_att, q_rate, r], dtype=float)
        self._design()

    def _design(self) -> None:
        q_att, q_rate, r = self._w
        # the exactly linearized axis: a double integrator
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        S = solve_continuous_are(A, B, np.diag([q_att, q_rate]), np.array([[r]]))
        K = (B.T @ S) / r
        self.kp, self.kd = float(K[0, 0]), float(K[0, 1])

    def get_gains(self) -> np.ndarray:
        return self._w.copy()

    def set_gains(self, g: np.ndarray) -> None:
        self._w = np.abs(np.asarray(g, dtype=float))
        self._design()

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        e, w, _ = self._error(x)
        return (self._phi(x) + self.m.Jb @ (self.kp * e + self.kd * w)
                + self._integral(e))


class SMC3D(_Base3D):
    """Sliding-mode control with a quasi-sliding boundary layer.

    Surface ``s = w + Lambda e``; reaching law ``sdot = -K sat(s/phi)``.  The
    boundary layer is not optional here: a pure ``sign`` on a 500 Hz digital
    loop with a quantized encoder chatters hard enough to inject the very
    wheel momentum the system cannot dump.
    """

    name = "smc"
    n_gains = 3

    def __init__(self, model: Cubli3D, lam: float = 9.0, k_reach: float = 60.0,
                 phi: float = 0.35, **kw):
        super().__init__(model, **kw)
        self.lam, self.k_reach, self.phi = float(lam), float(k_reach), float(phi)

    def get_gains(self) -> np.ndarray:
        return np.array([self.lam, self.k_reach, self.phi])

    def set_gains(self, g: np.ndarray) -> None:
        self.lam, self.k_reach = float(abs(g[0])), float(abs(g[1]))
        self.phi = float(max(abs(g[2]), 1e-3))

    def surface(self, x: np.ndarray) -> np.ndarray:
        e, w, _ = self._error(x)
        return w + self.lam * e

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        e, w, _ = self._error(x)
        s = w + self.lam * e
        return (self._phi(x)
                + self.m.Jb @ (self.lam * w + self.k_reach * _sat(s / self.phi))
                + self._integral(e))


class Backstepping3D(_Base3D):
    """Two-step Lyapunov backstepping.

    ``z1 = e``, virtual law ``w_d = -k1 z1``, ``z2 = w - w_d``, and
    ``V = 1/2 |z1|^2 + 1/2 z2' Jb z2`` gives
    ``Vdot = -k1|z1|^2 - k2|z2|^2 <= 0`` with

        tau = Phi + k1 Jb w + k2 z2 + z1.
    """

    name = "backstepping"
    n_gains = 2

    def __init__(self, model: Cubli3D, k1: float = 12.0, k2: float = 2.5, **kw):
        super().__init__(model, **kw)
        self.k1, self.k2 = float(k1), float(k2)

    def get_gains(self) -> np.ndarray:
        return np.array([self.k1, self.k2])

    def set_gains(self, g: np.ndarray) -> None:
        self.k1, self.k2 = float(abs(g[0])), float(abs(g[1]))

    def lyapunov(self, x: np.ndarray) -> float:
        e, w, _ = self._error(x)
        z2 = w + self.k1 * e
        return float(0.5 * e @ e + 0.5 * z2 @ self.m.Jb @ z2)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        e, w, _ = self._error(x)
        z2 = w + self.k1 * e
        return (self._phi(x) + self.k1 * self.m.Jb @ w + self.k2 * z2 + e
                + self._integral(e))


class PID3D(_Base3D):
    """Declared linear-class baseline: PID on attitude, rate damping on w."""

    name = "pid"
    n_gains = 3

    def __init__(self, model: Cubli3D, kp: float = 3.0, ki: float = 2.0,
                 kd: float = 0.35, i_max: float = 0.2, dt: float = 2e-3, **kw):
        super().__init__(model, **kw)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.i_max, self.dt = float(i_max), float(dt)
        self.reset()

    def reset(self) -> None:
        self._int = np.zeros(3)

    def get_gains(self) -> np.ndarray:
        return np.array([self.kp, self.ki, self.kd])

    def set_gains(self, g: np.ndarray) -> None:
        self.kp, self.ki, self.kd = (float(abs(g[0])), float(abs(g[1])), float(abs(g[2])))

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        e, w, _ = self._error(x)
        self._int = np.clip(self._int + e * self.dt, -self.i_max, self.i_max)
        return self.kp * e + self.ki * self._int + self.kd * w


from .mpc import MPC3D  # noqa: E402  (import here to avoid a cycle)

_REGISTRY = {
    "lqr": LQR3D,
    "mpc": MPC3D,
    "fbl": FBL3D,
    "smc": SMC3D,
    "backstepping": Backstepping3D,
    "pid": PID3D,
}


def make_controller_3d(name: str, model: Cubli3D, **kw) -> Controller:
    if name not in _REGISTRY:
        raise KeyError(f"unknown controller {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](model, **kw)
