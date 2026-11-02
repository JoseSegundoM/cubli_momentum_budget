"""Simulation harness: digital control loop, not a continuous idealisation.

The plant is integrated with RK4 at a fine step while the controller runs at
``f_s`` with zero-order hold, a computation delay of ``n_delay`` samples, and
quantized/noisy measurements.  Torque passes through
:class:`~cubli.actuator.WheelActuator`, so both the peak-torque limit and the
torque-speed envelope are always in the loop.

This is what makes the results portable to hardware: nothing here assumes
continuous-time full-state feedback.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .actuator import Sensors, WheelActuator
from .model2d import Cubli2D, rk4_step as rk4_2d
from .model3d import (
    Cubli3D,
    quat_mul,
    quat_normalize,
    rk4_step as rk4_3d,
)
from .params import CubliParams, NOMINAL


@dataclass
class SimConfig:
    t_end: float = 5.0
    # 1 ms: a step-size study (dt in {2e-4 ... 2e-3}, worst-case run) puts the
    # difference against dt = 2e-4 at 0.67 % on IAE, with the wheel-saturation
    # metric identical.
    dt_sim: float = 1e-3          # integration step [s]
    f_s: float = 500.0            # controller rate [Hz]
    # 0 = none.  Every result in the paper was produced with the delay path
    # inactive (a buffer bug made n_delay=1 behave as 0); the default records
    # what was actually simulated rather than silently changing the numbers.
    n_delay: int = 0              # computation delay [samples]
    sensors: bool = True
    actuator_mode: str = "envelope"
    tau_max: float | None = None
    omega_max: float | None = None
    fall_angle: float = np.deg2rad(45.0)
    settle_tol: float = np.deg2rad(2.0)
    log_every: int = 1
    seed: int | None = None


@dataclass
class SimResult:
    t: np.ndarray
    X: np.ndarray
    U: np.ndarray
    tilt: np.ndarray
    success: bool
    metrics: dict = field(default_factory=dict)

    def __getitem__(self, k):  # convenience
        return self.metrics[k]


# --------------------------------------------------------------------------


def _metrics(t, tilt, U, Om, cfg, p, success, t_fall, cpu_us):
    """Common metric set for both plants."""
    dt = np.gradient(t) if t.size > 1 else np.array([0.0])
    tau_norm = np.linalg.norm(U, axis=1) if U.ndim > 1 else np.abs(U)
    Om_norm = np.max(np.abs(Om), axis=1) if Om.ndim > 1 else np.abs(Om)

    # settling: last instant the tilt leaves the tolerance band and never
    # returns.  If the run ends outside the band it never settled.
    inside = tilt <= cfg.settle_tol
    bad = np.where(~inside)[0]
    if not success or not inside.any() or (bad.size and bad[-1] == t.size - 1):
        t_settle = np.nan
    else:
        t_settle = float(t[bad[-1] + 1]) if bad.size else float(t[0])

    # overshoot: how far the tilt comes back out *after* first reaching the band
    first_in = np.argmax(inside) if inside.any() else None
    if first_in is not None and first_in + 1 < tilt.size:
        overshoot = float(np.rad2deg(tilt[first_in + 1:].max()))
    else:
        overshoot = float("nan")

    return {
        "success": bool(success),
        "t_fall": float(t_fall),
        "t_settle": t_settle,
        "peak_tilt_deg": float(np.rad2deg(tilt.max())),
        "overshoot_deg": overshoot,
        "iae_deg_s": float(np.sum(np.rad2deg(tilt) * dt)),
        "effort_j": float(np.sum(tau_norm**2 * dt)),
        "tau_peak": float(tau_norm.max()),
        "tau_rms": float(np.sqrt(np.mean(tau_norm**2))),
        # NOT "omega_max": that name belongs to the actuator *limit* in the run
        # specification, and a metric of the same name silently overwrote it in
        # the results table, plotting sweeps against the achieved speed instead
        # of the swept limit.
        "omega_peak": float(Om_norm.max()),
        # eta_Omega = max_i |Omega_i| / Omega_max.  The actuator constraint is
        # component-wise (a box in wheel-speed space), so the budget indicator
        # is an infinity norm on wheel speed -- NOT ||h_w||_2 / h_max, which
        # would allow up to sqrt(3) h_max before any single wheel saturates.
        "budget_used": float(Om_norm.max() / p.omega_max),
        "h_wheel_max": float(p.Ja * Om_norm.max()),
        "sat_fraction": float(np.mean(tau_norm >= 0.999 * (cfg.tau_max or p.tau_max))),
        "final_tilt_deg": float(np.rad2deg(tilt[-1])),
        "cpu_us_per_step": float(cpu_us),
    }


def simulate_3d(model: Cubli3D, controller, x0: np.ndarray,
                cfg: SimConfig = SimConfig(), rng: np.random.Generator | None = None) -> SimResult:
    p: CubliParams = model.p
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    act = WheelActuator(p, cfg.actuator_mode, cfg.tau_max, cfg.omega_max)
    sen = Sensors(p, rng, cfg.sensors)
    controller.reset()

    dt = cfg.dt_sim
    n = int(round(cfg.t_end / dt))
    ctrl_every = max(1, int(round((1.0 / cfg.f_s) / dt)))
    # maxlen must be n_delay+1: with maxlen==n_delay the value appended this
    # step is already at index 0, so nothing was ever delayed.
    delay = deque([np.zeros(3)] * (cfg.n_delay + 1), maxlen=cfg.n_delay + 1)

    x = np.array(x0, dtype=float)
    tau = np.zeros(3)
    T, X, U, TILT = [], [], [], []
    t_fall, success, cpu = np.nan, True, 0.0
    n_calls = 0

    for k in range(n + 1):
        t = k * dt
        if k % ctrl_every == 0:
            q, w, Om = model.unpack(x)
            if cfg.sensors:
                dth = rng.normal(0.0, p.angle_noise_std, 3)
                q_m = quat_normalize(quat_mul(q, np.concatenate([[1.0], 0.5 * dth])))
                x_meas = np.concatenate([q_m, sen.measure_rate(w), sen.measure_wheel(Om)])
            else:
                x_meas = x
            t0 = time.perf_counter()
            cmd = np.asarray(controller(x_meas, t), dtype=float).ravel()
            cpu += time.perf_counter() - t0
            n_calls += 1
            delay.append(cmd)
            tau = act(delay[0], model.unpack(x)[2], 1.0 / cfg.f_s)

        tilt = model.tilt_angle(x)
        if k % cfg.log_every == 0:
            T.append(t); X.append(x.copy()); U.append(tau.copy()); TILT.append(tilt)

        if tilt > cfg.fall_angle:
            success, t_fall = False, t
            break
        x = rk4_3d(model, x, tau, dt)

    T = np.asarray(T); X = np.asarray(X); U = np.asarray(U); TILT = np.asarray(TILT)
    cpu_us = 1e6 * cpu / max(n_calls, 1)
    m = _metrics(T, TILT, U, X[:, 7:10], cfg, p, success, t_fall, cpu_us)
    m["hz_drift"] = float(model.momentum_vertical(X[-1]) - model.momentum_vertical(X[0]))
    return SimResult(T, X, U, TILT, success, m)


def simulate_2d(model: Cubli2D, controller, x0: np.ndarray,
                cfg: SimConfig = SimConfig(), rng: np.random.Generator | None = None) -> SimResult:
    p: CubliParams = model.p
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    act = WheelActuator(p, cfg.actuator_mode, cfg.tau_max, cfg.omega_max)
    sen = Sensors(p, rng, cfg.sensors)
    controller.reset()

    dt = cfg.dt_sim
    n = int(round(cfg.t_end / dt))
    ctrl_every = max(1, int(round((1.0 / cfg.f_s) / dt)))
    delay = deque([np.zeros(1)] * (cfg.n_delay + 1), maxlen=cfg.n_delay + 1)

    x = np.array(x0, dtype=float)
    tau = np.zeros(1)
    T, X, U = [], [], []
    t_fall, success, cpu = np.nan, True, 0.0
    n_calls = 0

    for k in range(n + 1):
        t = k * dt
        if k % ctrl_every == 0:
            if cfg.sensors:
                x_meas = np.array([
                    float(sen.measure_angle(x[0])[0]),
                    float(sen.measure_rate(x[1])[0]),
                    float(sen.measure_wheel(x[2])[0]),
                ])
            else:
                x_meas = x
            t0 = time.perf_counter()
            cmd = np.asarray(controller(x_meas, t), dtype=float).ravel()
            cpu += time.perf_counter() - t0
            n_calls += 1
            delay.append(cmd)
            tau = act(delay[0], x[2], 1.0 / cfg.f_s)

        if k % cfg.log_every == 0:
            T.append(t); X.append(x.copy()); U.append(tau.copy())

        if abs(np.arctan2(np.sin(x[0]), np.cos(x[0]))) > cfg.fall_angle:
            success, t_fall = False, t
            break
        x = rk4_2d(model, x, float(tau[0]), dt)

    T = np.asarray(T); X = np.asarray(X); U = np.asarray(U)
    tilt = np.abs(np.arctan2(np.sin(X[:, 0]), np.cos(X[:, 0])))
    cpu_us = 1e6 * cpu / max(n_calls, 1)
    m = _metrics(T, tilt, U, X[:, 2:3], cfg, p, success, t_fall, cpu_us)
    return SimResult(T, X, U, tilt, success, m)


def default_initial_3d(model: Cubli3D, tilt_deg: float, axis=(1.0, -1.0, 0.0),
                       Om0=None) -> np.ndarray:
    """Equilibrium attitude perturbed by ``tilt_deg`` about ``axis``."""
    from .model3d import quat_from_axis_angle
    q = quat_mul(model.equilibrium_quat(),
                 quat_from_axis_angle(np.asarray(axis, float), np.deg2rad(tilt_deg)))
    return model.state(q=q, Om=Om0)


__all__ = ["SimConfig", "SimResult", "simulate_2d", "simulate_3d", "default_initial_3d",
           "NOMINAL"]
