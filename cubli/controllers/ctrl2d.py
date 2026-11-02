"""Planar versions of the same five laws, for the edge-balancing case.

Plant: ``Ib theta_ddot = phi(x) - tau`` with
``phi = M g l sin(theta) - b theta_dot + tau_friction(Omega)``.

One structural difference from the 3-D case is worth stating, because the
paper turns on it: in the plane, gravity provides an external torque about the
pivot axis, so **all** stored wheel momentum can eventually be dumped by
leaning.  In the 3-D corner case the component of momentum along the vertical
cannot (Proposition 1).  The planar system is therefore the optimistic case,
and the 3-D one is where the momentum budget genuinely binds.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are

from ..model2d import Cubli2D
from .base import Controller, MomentumManager


def _sat(v: float) -> float:
    return float(np.clip(v, -1.0, 1.0))


class _Base2D(Controller):
    def __init__(self, model: Cubli2D, k_h: float = 0.0,
                 tilt_max: float = np.deg2rad(3.0)):
        self.m = model
        self.mm = MomentumManager(k_h, model.Mgl, tilt_max, None)

    def _error(self, x: np.ndarray):
        theta, theta_dot, Omega = x
        ref = float(self.mm.reference(self.m.Ja * Omega)[0])
        return theta - ref, theta_dot, Omega

    def _phi(self, x: np.ndarray) -> float:
        theta, theta_dot, Omega = x
        return float(self.m.Mgl * np.sin(theta) - self.m.p.b_pivot * theta_dot
                     + self.m._wheel_friction(Omega))

    @property
    def momentum_aware(self) -> bool:
        return self.mm.enabled


class LQR2D(_Base2D):
    name = "lqr"
    n_gains = 4

    def __init__(self, model: Cubli2D, q_att: float = 3000.0, q_rate: float = 60.0,
                 q_mom: float = 200.0, r: float = 1.0, **kw):
        super().__init__(model, **kw)
        self._w = np.array([q_att, q_rate, q_mom, r], dtype=float)
        self._design()

    def _design(self) -> None:
        q_att, q_rate, q_mom, r = self._w
        A, B = self.m.linearize(0.0)
        Q = np.diag([q_att, q_rate, q_mom])
        R = np.array([[r]])
        S = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ S).ravel()
        self.S = S

    def get_gains(self) -> np.ndarray:
        return self._w.copy()

    def set_gains(self, g: np.ndarray) -> None:
        self._w = np.abs(np.asarray(g, dtype=float))
        self._design()

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        e, ed, Om = self._error(x)
        return np.array([-float(self.K @ np.array([e, ed, self.m.Ja * Om]))])


class FBL2D(_Base2D):
    """Exact feedback linearization with an LQR outer loop (planar case).

    Canceling ``phi(x)`` leaves the double integrator ``theta_ddot = v``; the
    outer loop is the LQR for it.  Feedback linearization on its own does not
    define a controller, so the pair is named as a pair.
    """

    name = "fbl"
    n_gains = 3

    def __init__(self, model: Cubli2D, q_att: float = 8000.0, q_rate: float = 300.0,
                 r: float = 1.0, **kw):
        super().__init__(model, **kw)
        self._w = np.array([q_att, q_rate, r], dtype=float)
        self._design()

    def _design(self) -> None:
        q_att, q_rate, r = self._w
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
        e, ed, _ = self._error(x)
        return np.array([self._phi(x) + self.m.Ib * (self.kp * e + self.kd * ed)])


class SMC2D(_Base2D):
    name = "smc"
    n_gains = 3

    def __init__(self, model: Cubli2D, lam: float = 9.0, k_reach: float = 60.0,
                 phi: float = 0.35, **kw):
        super().__init__(model, **kw)
        self.lam, self.k_reach, self.phi = float(lam), float(k_reach), float(phi)

    def get_gains(self) -> np.ndarray:
        return np.array([self.lam, self.k_reach, self.phi])

    def set_gains(self, g: np.ndarray) -> None:
        self.lam, self.k_reach = float(abs(g[0])), float(abs(g[1]))
        self.phi = float(max(abs(g[2]), 1e-3))

    def surface(self, x: np.ndarray) -> float:
        e, ed, _ = self._error(x)
        return ed + self.lam * e

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        e, ed, _ = self._error(x)
        s = ed + self.lam * e
        return np.array([self._phi(x)
                         + self.m.Ib * (self.lam * ed + self.k_reach * _sat(s / self.phi))])


class Backstepping2D(_Base2D):
    name = "backstepping"
    n_gains = 2

    def __init__(self, model: Cubli2D, k1: float = 12.0, k2: float = 2.5, **kw):
        super().__init__(model, **kw)
        self.k1, self.k2 = float(k1), float(k2)

    def get_gains(self) -> np.ndarray:
        return np.array([self.k1, self.k2])

    def set_gains(self, g: np.ndarray) -> None:
        self.k1, self.k2 = float(abs(g[0])), float(abs(g[1]))

    def lyapunov(self, x: np.ndarray) -> float:
        e, ed, _ = self._error(x)
        z2 = ed + self.k1 * e
        return float(0.5 * e**2 + 0.5 * self.m.Ib * z2**2)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        e, ed, _ = self._error(x)
        z2 = ed + self.k1 * e
        return np.array([self._phi(x) + self.k1 * self.m.Ib * ed + self.k2 * z2 + e])


class PID2D(_Base2D):
    name = "pid"
    n_gains = 3

    def __init__(self, model: Cubli2D, kp: float = 3.0, ki: float = 2.0,
                 kd: float = 0.35, i_max: float = 0.2, dt: float = 2e-3, **kw):
        super().__init__(model, **kw)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.i_max, self.dt = float(i_max), float(dt)
        self.reset()

    def reset(self) -> None:
        self._int = 0.0

    def get_gains(self) -> np.ndarray:
        return np.array([self.kp, self.ki, self.kd])

    def set_gains(self, g: np.ndarray) -> None:
        self.kp, self.ki, self.kd = (float(abs(g[0])), float(abs(g[1])), float(abs(g[2])))

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        e, ed, _ = self._error(x)
        self._int = float(np.clip(self._int + e * self.dt, -self.i_max, self.i_max))
        return np.array([self.kp * e + self.ki * self._int + self.kd * ed])


_REGISTRY = {
    "lqr": LQR2D,
    "fbl": FBL2D,
    "smc": SMC2D,
    "backstepping": Backstepping2D,
    "pid": PID2D,
}


def make_controller_2d(name: str, model: Cubli2D, **kw) -> Controller:
    if name not in _REGISTRY:
        raise KeyError(f"unknown controller {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](model, **kw)
