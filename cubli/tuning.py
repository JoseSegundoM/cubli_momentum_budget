"""Fair-tuning protocol.

A comparative study is only worth reading if the baselines were given the same
chance as the proposed method.  Every controller here is therefore tuned by
the *same* procedure:

* the same cost functional,
* the same set of scenarios,
* the same optimiser (Nelder-Mead on log-gains, so gains stay positive),
* the same iteration budget and the same initial simplex scale.

The cost is

    Jc = mean over scenarios of  [ w_e * IAE + w_u * effort + fall penalty ]

with the fall penalty dominating everything else, because a controller that
drops the cube cannot be rescued by a good transient.  Gains are optimised on
the log scale; the resulting values are reported in the paper so the tuning is
auditable rather than declared.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .experiments import RunSpec, run_spec

# The scenario set every controller is tuned on.  It deliberately mixes an easy
# transient, a large transient, a plant asymmetry and a pre-loaded wheel, so no
# design can win by specialising on one regime.
# Two properties this set has to have, and the first version of it had neither.
#
# (1) INFORMATIVE. If every candidate gain set falls on some scenario, the cost
#     is dominated by the fall penalty and the optimiser wanders on a plateau:
#     the tuned "gains" are then noise, and a comparison built on them is the
#     straw man a reviewer will accuse it of being. Measured on the first
#     version: LQR converged at cost 4.96 while the three nonlinear designs sat
#     at 152-160, i.e. falling every time. The stressor (2 mm offset over 20 s)
#     is precisely the condition the paper claims they cannot survive, so it
#     does not belong in their tuning set.
# (2) DISCRIMINATING. The disturbance scenario has to be strong enough, and run
#     long enough, that stored momentum actually matters -- otherwise the
#     optimiser drives the LQR's momentum weight to zero (measured: q_mom -> 2e-3
#     when tuning at 0.8 mm over 25 s) and the "momentum-aware" design in the
#     comparison is quietly momentum-blind. At 1.5 mm over 30 s a momentum-blind
#     design cannot survive the horizon, so awareness is rewarded where it helps
#     and survival time grades the designs that cannot use it.
# (3) DISJOINT FROM THE EVALUATION. Tuning at 2 mm and then reporting failure
#     times at 2 mm is an in-sample result. Tuning uses 1.5 mm; every reported
#     stressed evaluation point is unseen.
#
# The horizon still has to exceed the momentum time scale, or the optimiser
# assigns no value to momentum management (it tuned k_h -> 0.04 at 8 s).
TUNING_SCENARIOS = [
    RunSpec(tilt0_deg=4.0, t_end=12.0, seed=11),
    RunSpec(tilt0_deg=9.0, t_end=12.0, seed=12),
    RunSpec(tilt0_deg=4.0, com_offset=1.5e-3, t_end=30.0, seed=13),
    RunSpec(tilt0_deg=6.0, h0_frac=0.45, h0_dir="perp", t_end=25.0, seed=14),
]

W_ERROR = 1.0        # per deg.s of integrated tilt
W_EFFORT = 20.0      # per J of integrated squared torque
FALL_PENALTY = 1.0e3

DEFAULTS = {
    "lqr": dict(q_att=3000.0, q_rate=60.0, q_mom=200.0, r=1.0),
    "fbl": dict(q_att=8000.0, q_rate=300.0, r=1.0),
    "smc": dict(lam=9.0, k_reach=60.0, phi=0.35),
    "backstepping": dict(k1=12.0, k2=2.5),
    "pid": dict(kp=3.0, ki=2.0, kd=0.35),
    "mpc": dict(q_att=3000.0, q_rate=60.0, q_mom=200.0, r=1.0),
}


def cost(name: str, gains: dict, k_h: float, k_i: float = 0.0) -> float:
    """Mean scenario cost for one candidate gain set.

    A run that falls is scored ONLY by how long it lasted.  Adding its
    integrated error on top would reward falling early, because a truncated run
    accumulates less error -- the same censoring trap that makes unconditional
    error comparisons meaningless, here hiding inside the objective itself.
    Surviving runs are scored by accuracy and effort.

    The two regimes stay commensurate: any fall before the horizon costs more
    than any achievable accuracy term, so survival is always preferred, while
    the optimiser still has a usable gradient (survival time) inside the
    failing region.
    """
    total = 0.0
    for base in TUNING_SCENARIOS:
        spec = replace(base, controller=name, k_h=k_h, k_i=k_i)
        m = run_spec(spec, {name: gains})
        if not m.get("success", False):
            t_fall = m.get("t_fall", 0.0)
            t_fall = 0.0 if not np.isfinite(t_fall) else t_fall
            total += FALL_PENALTY * (1.0 - t_fall / spec.t_end)
        else:
            total += (W_ERROR * m.get("iae_deg_s", 0.0)
                      + W_EFFORT * m.get("effort_j", 0.0))
    return total / len(TUNING_SCENARIOS)


def tune(name: str, variant: str = "classical", maxiter: int = 45,
         base_gains: dict | None = None, verbose: bool = True) -> dict:
    """Optimise one controller under the shared protocol, in two stages.

    Stage 1 (``variant="classical"``) searches the controller's own gains.
    Stage 2 (``"momentum_aware"`` / ``"integral"``) freezes those gains and
    searches only the augmentation gain, ``k_h`` or ``k_i``.

    Freezing the base gains is deliberate and it is the fairer design: the
    three variants of a controller then differ *only* by the augmentation, so a
    difference between them measures the augmentation rather than a difference
    in how well two separate multi-dimensional searches happened to converge.
    It also makes the second stage a one-dimensional problem, which converges
    to a tolerance the multi-dimensional search cannot reach in the same
    budget.

    Every controller gets the same cost, the same scenarios, the same optimiser
    and the same per-stage budget.
    """
    from scipy.optimize import minimize_scalar

    if variant == "classical":
        keys = list(DEFAULTS[name])
        x0 = np.log(np.array([DEFAULTS[name][k] for k in keys], dtype=float))

        def obj(x):
            return cost(name, dict(zip(keys, np.exp(x))), 0.0, 0.0)

        res = minimize(obj, x0, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, maxfev=4 * maxiter,
                                    xatol=5e-3, fatol=5e-3, adaptive=True))
        gains = dict(zip(keys, np.exp(res.x)))
        out = {"gains": gains, "k_h": 0.0, "k_i": 0.0, "cost": float(res.fun),
               "n_eval": int(res.nfev), "hit_cap": bool(res.nfev >= 4 * maxiter)}
    else:
        gains = dict(base_gains or DEFAULTS[name])
        key = "k_h" if variant == "momentum_aware" else "k_i"

        def obj1(u):
            g = float(np.exp(u))
            return cost(name, gains, g if key == "k_h" else 0.0,
                        g if key == "k_i" else 0.0)

        r = minimize_scalar(obj1, bounds=(np.log(1e-3), np.log(50.0)),
                            method="bounded", options={"xatol": 2e-2})
        g = float(np.exp(r.x))
        out = {"gains": gains, "k_h": g if key == "k_h" else 0.0,
               "k_i": g if key == "k_i" else 0.0, "cost": float(r.fun),
               "n_eval": int(r.nfev), "hit_cap": False}

    if verbose:
        print(f"  {name:<14} {variant:<15} cost {out['cost']:9.2f}  "
              + "  ".join(f"{k}={v:.3g}" for k, v in out["gains"].items()),
              flush=True)
    return out


def save(results: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2), encoding="utf-8")


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
