"""Experiment engine: one specification type, one worker, every study.

Every result in the paper -- nominal comparison, actuator sweeps, region of
attraction, Monte-Carlo robustness -- is produced by :func:`run_spec` from a
:class:`RunSpec`.  A single code path means a single set of assumptions, and
the CSV that reaches the figures carries the full specification of the run
that produced each row.

The plant and the controller hold **different** parameter objects: the plant
may be perturbed (Monte Carlo) or asymmetric (center-of-mass offset), while
the controller always designs on the nominal, symmetric cube.  That asymmetry
is deliberate -- a controller that knows the true plant exactly is not a
controller, it is an oracle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np

from .controllers import make_controller_3d
from .model3d import Cubli3D, quat_from_axis_angle, quat_mul
from .params import NOMINAL, CubliParams
from .sim import SimConfig, simulate_3d

CONTROLLERS = ["lqr", "mpc", "fbl", "smc", "backstepping", "pid"]
# MPC enforces the wheel-speed constraint inside the optimisation, so the
# tilt-to-unload augmentation and the integral variant do not apply to it.
SINGLE_VARIANT = {"mpc"}


@dataclass(frozen=True)
class RunSpec:
    """Everything needed to reproduce one simulation."""

    controller: str = "lqr"
    k_h: float = 0.0              # momentum-management gain (0 = classical design)
    k_i: float = 0.0              # integral action on attitude error (0 = off)
    tilt0_deg: float = 3.0        # initial tilt away from the corner equilibrium
    h0_frac: float = 0.0          # initial stored momentum, as a fraction of the budget
    h0_dir: str = "perp"          # "perp" (dumpable) or "vertical" (invariant)
    com_offset: float = 0.0       # plant asymmetry [m]
    omega_max: float | None = None
    tau_max: float | None = None
    b_pivot: float | None = None  # pivot friction override (Theorem 1 study)
    param_rel: float = 0.0        # Monte-Carlo parameter spread (0 = nominal)
    sensors: bool = True
    actuator_mode: str = "envelope"
    t_end: float = 20.0
    seed: int = 0
    tag: str = ""

    def key(self) -> dict:
        return asdict(self)


def _plant_params(spec: RunSpec) -> CubliParams:
    p = replace(NOMINAL, com_offset=spec.com_offset)
    if spec.param_rel > 0.0:
        # ``perturbed`` already carries com_offset through and adds its own
        # dispersion, so adding spec.com_offset again here doubled the nominal
        # asymmetry of every Monte-Carlo plant.  It did, until this line went.
        p = p.perturbed(np.random.default_rng(spec.seed + 10_000), spec.param_rel)
    if spec.omega_max is not None:
        p = replace(p, omega_max=spec.omega_max)
    if spec.tau_max is not None:
        p = replace(p, tau_max=spec.tau_max)
    if spec.b_pivot is not None:
        p = replace(p, b_pivot=spec.b_pivot)
    return p


def initial_state(model: Cubli3D, tilt_deg: float, h0_frac: float, h0_dir: str,
                  omega_max: float) -> np.ndarray:
    """Initial condition: tilt away from equilibrium plus pre-stored momentum.

    ``h0_dir="vertical"`` puts the stored momentum along the balancing axis,
    i.e. along the invariant of Proposition 1, where no controller can remove
    it.  ``"perp"`` puts it in the plane where gravity can be used to dump it.
    The difference between the two region-of-attraction maps is the clearest
    experimental signature of the invariant.
    """
    n = model.n_eq
    if h0_dir == "vertical":
        direction = n
    else:
        d = np.cross(n, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(d) < 1e-9:
            d = np.cross(n, np.array([0.0, 1.0, 0.0]))
        direction = d / np.linalg.norm(d)
    Om0 = h0_frac * omega_max * direction

    # tilt about an axis orthogonal to the balancing diagonal
    axis = np.cross(n, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, -1.0, 0.0])
    q = quat_mul(model.equilibrium_quat(),
                 quat_from_axis_angle(axis, np.deg2rad(tilt_deg)))
    return model.state(q=q, Om=Om0)


def run_spec(spec: RunSpec, gains: dict | None = None) -> dict:
    """Run one simulation and return its metrics, tagged with the specification."""
    p_plant = _plant_params(spec)
    plant = Cubli3D(p_plant)
    nominal = Cubli3D(NOMINAL)

    kw = dict(gains.get(spec.controller, {})) if gains else {}
    if spec.controller in SINGLE_VARIANT:
        ctrl = make_controller_3d(spec.controller, nominal, **kw)
    else:
        ctrl = make_controller_3d(spec.controller, nominal, k_h=spec.k_h,
                                  k_i=spec.k_i, **kw)

    cfg = SimConfig(t_end=spec.t_end, seed=spec.seed, sensors=spec.sensors,
                    actuator_mode=spec.actuator_mode,
                    omega_max=p_plant.omega_max, tau_max=p_plant.tau_max,
                    log_every=20)
    x0 = initial_state(plant, spec.tilt0_deg, spec.h0_frac, spec.h0_dir,
                       p_plant.omega_max)
    res = simulate_3d(plant, ctrl, x0, cfg)
    out = spec.key()
    clash = set(out) & set(res.metrics)
    if clash:   # a metric must never shadow the specification that produced it
        raise RuntimeError(f"metric/spec name collision: {sorted(clash)}")
    out.update(res.metrics)
    return out


def _worker(args):
    spec, gains = args
    try:
        return run_spec(spec, gains)
    except Exception as exc:  # a failed run is data, not a crash
        out = spec.key()
        out.update({"success": False, "error": repr(exc)})
        return out


def run_many(specs: Iterable[RunSpec], gains: dict | None = None,
             workers: int | None = None, progress: bool = True) -> "list[dict]":
    """Run a batch, in parallel when it is worth it."""
    specs = list(specs)
    if workers == 1 or len(specs) < 4:
        return [_worker((s, gains)) for s in specs]

    import concurrent.futures as cf
    import os

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(_worker, [(s, gains) for s in specs],
                                       chunksize=4)):
            rows.append(row)
            if progress and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(specs)} runs", flush=True)
    return rows
