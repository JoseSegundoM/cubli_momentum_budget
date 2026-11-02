"""Reaction-wheel actuator model: where the momentum budget becomes physical.

Three limits act on the commanded torque, and they are *not* interchangeable:

1. **Torque limit** ``|tau| <= tau_max`` -- the one every comparative study
   already models.
2. **Torque-speed envelope** -- a brushless motor cannot deliver its peak
   torque at its peak speed.  Accelerating a wheel that already spins near
   ``omega_max`` is impossible; *braking* it is always possible.  This is the
   limit that actually ends a balancing run, and it is what makes the wheel
   momentum a finite, consumable budget.
3. **Slew-rate limit** -- current cannot change instantaneously.

``mode="envelope"`` (default) uses a linear torque-speed curve, the standard
first-order description of a BLDC drive.  ``mode="hard"`` is the naive clamp
used in most of the literature and is kept so the paper can quantify how
optimistic that assumption is.
"""

from __future__ import annotations

import numpy as np

from .params import CubliParams


class WheelActuator:
    """Maps commanded torque to delivered torque, given the wheel speed."""

    def __init__(self, p: CubliParams, mode: str = "envelope",
                 tau_max: float | None = None, omega_max: float | None = None,
                 rate_limit: bool = True):
        if mode not in ("envelope", "hard", "ideal"):
            raise ValueError(f"unknown actuator mode {mode!r}")
        self.p = p
        self.mode = mode
        self.tau_max = p.tau_max if tau_max is None else float(tau_max)
        self.omega_max = p.omega_max if omega_max is None else float(omega_max)
        self.rate_limit = rate_limit
        self.reset()

    def reset(self) -> None:
        self._tau_prev = None
        self.saturated_torque = False
        self.saturated_speed = False

    # ------------------------------------------------------------------
    def available_torque(self, Omega: np.ndarray | float) -> np.ndarray:
        """Peak torque available *in the accelerating direction* at this speed."""
        Om = np.atleast_1d(np.asarray(Omega, dtype=float))
        if self.mode == "ideal":
            return np.full_like(Om, np.inf)
        if self.mode == "hard":
            return np.where(np.abs(Om) >= self.omega_max, 0.0, self.tau_max)
        frac = np.clip(1.0 - np.abs(Om) / self.omega_max, 0.0, 1.0)
        return self.tau_max * frac

    def __call__(self, tau_cmd: np.ndarray | float, Omega: np.ndarray | float,
                 dt: float) -> np.ndarray:
        tau = np.atleast_1d(np.asarray(tau_cmd, dtype=float)).copy()
        Om = np.atleast_1d(np.asarray(Omega, dtype=float))

        if self.mode == "ideal":
            self._tau_prev = tau
            return tau

        # symmetric peak-torque clamp
        clamped = np.clip(tau, -self.tau_max, self.tau_max)
        self.saturated_torque = bool(np.any(np.abs(tau) > self.tau_max + 1e-12))
        tau = clamped

        # torque-speed envelope: only the accelerating direction is limited
        tau_avail = self.available_torque(Om)
        accelerating = np.sign(tau) == np.sign(Om)
        at_rest = np.abs(Om) < 1e-9
        limit = np.where(accelerating & ~at_rest, tau_avail, self.tau_max)
        limited = np.clip(tau, -limit, limit)
        self.saturated_speed = bool(np.any(np.abs(limited - tau) > 1e-12))
        tau = limited

        # slew rate
        if self.rate_limit and self._tau_prev is not None:
            d_max = self.p.tau_rate_max * dt
            tau = np.clip(tau, self._tau_prev - d_max, self._tau_prev + d_max)

        self._tau_prev = tau
        return tau


class Sensors:
    """Encoder quantization, gyro noise and bias -- the implementation layer.

    The controllers never see the true state; they see this.  Keeping it in
    the loop from the start is what makes the study portable to hardware
    instead of a purely numerical exercise.
    """

    def __init__(self, p: CubliParams, rng: np.random.Generator | None = None,
                 enabled: bool = True):
        self.p = p
        self.rng = np.random.default_rng() if rng is None else rng
        self.enabled = enabled
        self.gyro_bias = (self.rng.normal(0.0, p.gyro_bias_std, 3)
                          if enabled else np.zeros(3))

    def measure_angle(self, theta: np.ndarray | float) -> np.ndarray:
        th = np.atleast_1d(np.asarray(theta, dtype=float))
        if not self.enabled:
            return th
        return th + self.rng.normal(0.0, self.p.angle_noise_std, th.shape)

    def measure_rate(self, w: np.ndarray | float) -> np.ndarray:
        ww = np.atleast_1d(np.asarray(w, dtype=float))
        if not self.enabled:
            return ww
        return (ww + self.gyro_bias[: ww.size]
                + self.rng.normal(0.0, self.p.gyro_noise_std, ww.shape))

    def measure_wheel(self, Omega: np.ndarray | float) -> np.ndarray:
        """Wheel speed from finite-difference of a quantized encoder."""
        Om = np.atleast_1d(np.asarray(Omega, dtype=float))
        if not self.enabled:
            return Om
        lsb_rate = self.p.encoder_lsb * self.p.f_s
        return np.round(Om / lsb_rate) * lsb_rate
