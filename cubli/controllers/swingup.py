"""Energy-based swing-up and the region-of-attraction-aware handover.

Swing-up
--------
Let ``E`` be the *body* energy referred to the upright equilibrium,

    E = 1/2 I theta_dot^2 + M g l (cos theta - 1),

which is zero exactly at the upright rest state and negative below it.
Neglecting friction and the (small) wheel inertia, ``Edot = -theta_dot tau``,
so the choice

    tau = k_E * E * theta_dot

gives ``d/dt (E^2/2) = -k_E E^2 theta_dot^2 <= 0``: the energy error decreases
whenever the body is moving.  The law stalls only at ``theta_dot = 0``, which
is handled by a small kick when the pendulum is detected at rest away from
upright.

Handover
--------
Most implementations switch to the balancing controller inside a hand-tuned
angle box.  That is exactly the decision the region-of-attraction study of
this paper is about, so instead the handover uses the LQR value function as a
certificate: switch when

    V(z) = z' S z <= gamma,

with ``S`` from the algebraic Riccati equation and ``gamma`` calibrated from
the measured region of attraction.  The wheel speed is part of ``z``, so a
swing-up that arrives upright with the wheel already saturated is correctly
refused -- the state is inside the angle box but outside the validated set.
"""

from __future__ import annotations

import numpy as np

from ..model2d import Cubli2D
from .base import Controller


class EnergySwingUp(Controller):
    """Energy-pumping law for the planar Cubli."""

    name = "swingup"
    n_gains = 2

    def __init__(self, model: Cubli2D, k_E: float = 8.0, kick: float = 0.05,
                 omega_guard: float = 0.85, k_unload: float = 0.0):
        self.m = model
        self.k_E = float(k_E)
        self.kick = float(kick)
        self.omega_guard = float(omega_guard)
        # Pumping alone is momentum-blind: it delivers the body to the upright
        # position with whatever wheel speed the pumping happened to leave
        # behind, and a handover test that weights stored momentum will reject
        # that state.  The unloading term bleeds the wheel while the body is
        # still swinging -- which is exactly when gravity torque is available
        # to absorb the momentum.
        self.k_unload = float(k_unload)

    def get_gains(self) -> np.ndarray:
        return np.array([self.k_E, self.kick])

    def set_gains(self, g: np.ndarray) -> None:
        self.k_E, self.kick = float(abs(g[0])), float(abs(g[1]))

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        theta, theta_dot, Omega = x
        E = self.m.body_energy(x)
        tau = self.k_E * E * theta_dot - self.k_unload * self.m.Ja * Omega

        # Break the stall at theta_dot ~ 0 away from upright.
        if abs(theta_dot) < 1e-2 and abs(np.cos(theta) - 1.0) > 1e-3:
            tau = float(np.sign(np.sin(theta)) or 1.0) * self.kick

        # Do not spend the last of the momentum budget on pumping: past the
        # guard the wheel can no longer be accelerated anyway, and asking for
        # it only heats the motor.
        if abs(Omega) > self.omega_guard * self.m.p.omega_max and np.sign(tau) == np.sign(Omega):
            tau = 0.0
        return np.array([tau])


class HybridSwingUpBalance(Controller):
    """Swing-up, then validated handover to a balancing controller."""

    name = "hybrid"

    def __init__(self, swingup: EnergySwingUp, balance: Controller,
                 S: np.ndarray, gamma: float, hysteresis: float = 1.5,
                 Ja: float | None = None):
        self.swingup = swingup
        self.balance = balance
        self.S = np.asarray(S, dtype=float)
        self.gamma = float(gamma)
        self.hysteresis = float(hysteresis)
        # ``S`` comes from the Riccati equation in the (theta, theta_dot, h_w)
        # coordinates, so the wheel state must be converted from speed to
        # momentum before the value function is evaluated.  Feeding it a raw
        # wheel speed inflates V by four orders of magnitude and the handover
        # never fires.
        self.Ja = float(swingup.m.Ja if Ja is None else Ja)
        self.reset()

    def reset(self) -> None:
        self.balancing = False
        self.t_switch = np.nan
        self.swingup.reset()
        self.balance.reset()

    def value(self, x: np.ndarray) -> float:
        z = np.array([np.arctan2(np.sin(x[0]), np.cos(x[0])), x[1],
                      self.Ja * x[2]])
        return float(z @ self.S @ z)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        V = self.value(x)
        if not self.balancing and V <= self.gamma:
            self.balancing = True
            self.t_switch = t
        elif self.balancing and V > self.hysteresis * self.gamma:
            # fell out of the validated set: go back to pumping
            self.balancing = False
            self.balance.reset()
        return self.balance(x, t) if self.balancing else self.swingup(x, t)
