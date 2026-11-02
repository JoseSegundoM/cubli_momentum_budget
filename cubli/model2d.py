"""Planar (edge-balancing) Cubli: the reduced model.

Same physical cube as :mod:`cubli.model3d`, now pivoting about one of its
edges, driven by the single wheel whose spin axis is parallel to that edge.
It is the canonical reaction-wheel pendulum, and it is used here for the
swing-up study and for the dense region-of-attraction maps that would be too
expensive in the full 3-D state space.

State (3)
---------
``x = [theta, theta_dot, Omega]``

* ``theta``  -- tilt of the center of mass away from the upward vertical
  [rad]; ``theta = 0`` is the unstable upright equilibrium, ``theta = pi`` is
  the hanging rest position.
* ``Omega``  -- wheel speed relative to the body [rad/s].

Equations
---------
Angular momentum about the edge, ``h = I theta_dot + Ja Omega``, changes only
through gravity and pivot friction::

    I theta_ddot + Ja Omega_dot = M g l sin(theta) - b theta_dot
    Ja (Omega_dot + theta_ddot) = tau - b_w Omega - tau_c sign(Omega)

so, with ``Ib = I - Ja``::

    Ib theta_ddot = M g l sin(theta) - b theta_dot - tau + tau_f
    Omega_dot     = (tau - tau_f) / Ja - theta_ddot
"""

from __future__ import annotations

import numpy as np

from .params import G, CubliParams, NOMINAL


class Cubli2D:
    """Nonlinear edge-balancing model."""

    NX = 3
    NU = 1

    def __init__(self, p: CubliParams = NOMINAL, coulomb: bool = True,
                 smooth_coulomb: float = 1.0):
        self.p = p
        self.I = p.edge["I"]
        self.Ib = p.edge["Ib"]
        self.Ja = p.edge["Ja"]
        self.l = p.edge["l"]
        self.Mgl = p.edge["m_gl"]
        # planar equivalent of the center-of-mass offset: it tilts the true
        # equilibrium away from theta = 0
        self.theta_bias = float(np.arctan2(p.com_offset, p.edge["l"]))
        self.coulomb = coulomb
        self.smooth = smooth_coulomb

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def state(theta: float = 0.0, theta_dot: float = 0.0, Omega: float = 0.0) -> np.ndarray:
        return np.array([theta, theta_dot, Omega], dtype=float)

    def _wheel_friction(self, Omega: float) -> float:
        tau_f = self.p.b_wheel * Omega
        if self.coulomb:
            tau_f = tau_f + self.p.tau_coulomb * np.tanh(Omega / self.smooth)
        return tau_f

    # -- dynamics --------------------------------------------------------
    def f(self, x: np.ndarray, tau: float) -> np.ndarray:
        theta, theta_dot, Omega = x
        tau_f = self._wheel_friction(Omega)
        theta_ddot = (self.Mgl * np.sin(theta + self.theta_bias) - self.p.b_pivot * theta_dot
                      - tau + tau_f) / self.Ib
        Omega_dot = (tau - tau_f) / self.Ja - theta_ddot
        return np.array([theta_dot, theta_ddot, Omega_dot])

    # -- invariants ------------------------------------------------------
    def energy(self, x: np.ndarray) -> float:
        """Mechanical energy, zero at the upright equilibrium at rest."""
        theta, theta_dot, Omega = x
        T = (0.5 * self.I * theta_dot**2 + self.Ja * Omega * theta_dot
             + 0.5 * self.Ja * Omega**2)
        V = self.Mgl * (np.cos(theta) - 1.0)
        return float(T + V)

    def body_energy(self, x: np.ndarray) -> float:
        """Energy of the *body* alone -- the quantity the swing-up law shapes.

        The wheel's own kinetic energy is not part of the pendulum's swing, so
        the energy-based controller must not try to regulate it.
        """
        theta, theta_dot, _ = x
        return float(0.5 * self.I * theta_dot**2 + self.Mgl * (np.cos(theta) - 1.0))

    def momentum(self, x: np.ndarray) -> float:
        """Angular momentum about the edge [N.m.s]."""
        _, theta_dot, Omega = x
        return float(self.I * theta_dot + self.Ja * Omega)

    def wheel_momentum(self, x: np.ndarray) -> float:
        return float(self.Ja * x[2])

    # -- linearization ---------------------------------------------------
    def linearize(self, theta0: float = 0.0):
        """Linearization about ``[theta0, 0, .]``.

        States ``[theta, theta_dot, h_w]`` with ``h_w = Ja Omega`` the stored
        wheel momentum -- same change of variable as in the 3-D model, and for
        the same reason (scaling, and it is the variable the study is about).
        """
        c = self.Mgl * np.cos(theta0)
        b, bw, Ib, Ja = self.p.b_pivot, self.p.b_wheel, self.Ib, self.Ja
        A = np.array([
            [0.0, 1.0, 0.0],
            [c / Ib, -b / Ib, bw / (Ja * Ib)],
            [-Ja * c / Ib, Ja * b / Ib, -bw / Ja - bw / Ib],
        ])
        B = np.array([[0.0], [-1.0 / Ib], [1.0 + Ja / Ib]])
        return A, B


def rk4_step(model: Cubli2D, x: np.ndarray, tau: float, dt: float) -> np.ndarray:
    k1 = model.f(x, tau)
    k2 = model.f(x + 0.5 * dt * k1, tau)
    k3 = model.f(x + 0.5 * dt * k2, tau)
    k4 = model.f(x + dt * k3, tau)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
