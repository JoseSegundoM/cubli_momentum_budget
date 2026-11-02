"""Swing-up study on the planar (edge) Cubli, with a validated handover.

Two things are measured here that the usual swing-up demonstration skips:

1. the level set ``V(z) = z' S z <= gamma`` that the balancing controller
   actually recovers from, estimated from simulation rather than assumed, and
   used as the switching condition;
2. how much of the wheel's momentum budget the swing-up itself consumes, which
   is what determines whether the balancing controller inherits a usable state
   or a saturated one.

Writes ``results/E7_swingup.csv`` and ``figures/fig_swingup.pdf``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cubli.controllers.ctrl2d import make_controller_2d
from cubli.controllers.swingup import EnergySwingUp, HybridSwingUpBalance
from cubli.figstyle import COL, LABEL, PALETTE, WIDE, save, use_paper_style
from cubli.model2d import Cubli2D
from cubli.params import NOMINAL
from cubli.sim import SimConfig, simulate_2d

use_paper_style()
m = Cubli2D(NOMINAL)
CTRL = ["lqr", "fbl", "smc", "backstepping", "pid"]

# Wheel-unloading gain of the swing-up law.  Swept below: without it the
# handover never fires, and too much of it destroys the pumping.
K_UNLOAD = 20.0

try:
    from cubli.tuning import load as load_gains
    TUNED = load_gains("results/tuned_gains.json")
except Exception:
    TUNED = None


def balance_controller(name, momentum_aware=True):
    """Balancing law with the gains from the fair-tuning stage."""
    variant = "momentum_aware" if momentum_aware else "classical"
    if TUNED is None:
        return make_controller_2d(name, m, k_h=0.5 if momentum_aware else 0.0)
    blk = TUNED[variant][name]
    return make_controller_2d(name, m, k_h=blk["k_h"], **blk["gains"])


# ---------------------------------------------------------------------------
# 1. empirical level set of the LQR value function
# ---------------------------------------------------------------------------
def estimate_gamma(n_theta=15, n_rate=15, n_h=4, t_end=5.0, q=0.95):
    """Sub-level sets of the LQR value function, estimated from simulation.

    Two numbers are reported, because they answer different questions:

    ``gamma_strict``
        the largest level containing **no** failing sample -- the strictly
        verified inner estimate.  It is fragile: one unlucky grid point near
        the origin collapses it.
    ``gamma_q``
        the largest level in which at least a fraction ``q`` of the sampled
        states recover.  This is what the handover uses, and it is reported as
        an empirical estimate rather than a guarantee.

    Neither is a certificate in the formal sense; a Lyapunov or SOS
    certificate under saturation is left to future work, and the paper says so.
    """
    lqr = balance_controller("lqr")
    S = lqr.S
    cfg = SimConfig(t_end=t_end, seed=0, fall_angle=np.deg2rad(60.0))
    rows = []
    for th in np.linspace(-0.9, 0.9, n_theta):
        for thd in np.linspace(-6.0, 6.0, n_rate):
            for hf in np.linspace(0.0, 0.8, n_h):
                Om = hf * NOMINAL.omega_max
                z = np.array([th, thd, m.Ja * Om])
                V = float(z @ S @ z)
                r = simulate_2d(m, balance_controller("lqr"),
                                np.array([th, thd, Om]), cfg)
                rows.append(dict(theta=th, theta_dot=thd, h_frac=hf, V=V,
                                 success=bool(r.success)))
    df = pd.DataFrame(rows).sort_values("V").reset_index(drop=True)

    failed = df[~df.success]["V"]
    gamma_strict = float(failed.min() * 0.95) if len(failed) else float(df["V"].max())

    # largest level whose enclosed samples still recover at rate >= q
    rate = df["success"].expanding().mean().to_numpy()
    ok = np.where(rate >= q)[0]
    gamma_q = float(df["V"].to_numpy()[ok[-1]]) if len(ok) else gamma_strict
    return gamma_q, gamma_strict, S, df


# ---------------------------------------------------------------------------
# 2. swing-up + handover, for each balancing controller
# ---------------------------------------------------------------------------
def run_all(gamma, S, t_end=25.0):
    cfg = SimConfig(t_end=t_end, seed=0, fall_angle=np.pi)   # never "falls": it swings
    x0 = np.array([np.pi, 0.0, 0.0])                          # hanging at rest
    out, traces = [], {}
    for name in CTRL:
        bal = balance_controller(name)
        hyb = HybridSwingUpBalance(EnergySwingUp(m, k_E=8.0, k_unload=K_UNLOAD),
                                   bal, S, gamma)
        r = simulate_2d(m, hyb, x0, cfg)
        wrapped = np.abs(np.arctan2(np.sin(r.X[:, 0]), np.cos(r.X[:, 0])))

        t_sw = hyb.t_switch
        if np.isfinite(t_sw):
            k = int(np.searchsorted(r.t, t_sw))
            budget_sw = float(abs(r.X[min(k, len(r.t) - 1), 2]) / NOMINAL.omega_max)
            # balanced if it is still upright at the end of the run
            held = bool(wrapped[-1] < np.deg2rad(2.0))
            t_up = float(r.t[np.argmax(wrapped < np.deg2rad(2.0))])
        else:
            budget_sw, held, t_up = np.nan, False, np.nan

        out.append(dict(controller=name, t_switch=t_sw, t_upright=t_up,
                        balanced=held, budget_at_switch=budget_sw,
                        budget_peak=float(np.max(np.abs(r.X[:, 2]))
                                          / NOMINAL.omega_max),
                        effort=r.metrics["effort_j"],
                        peak_torque=r.metrics["tau_peak"]))
        traces[name] = r
    return pd.DataFrame(out), traces


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)
    print("estimating the validated handover set ...", flush=True)
    gamma, gamma_strict, S, grid = estimate_gamma()
    grid.to_csv("results/E7_roa2d.csv", index=False)
    print(f"  gamma(95%) = {gamma:.4g}   gamma(strict) = {gamma_strict:.4g}   "
          f"(grid success {grid.success.mean():.1%}, {len(grid)} states)")

    # trade-off of the unloading term, for the record
    sweep = []
    for ku in [0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0]:
        hyb = HybridSwingUpBalance(EnergySwingUp(m, k_E=8.0, k_unload=ku),
                                   balance_controller("lqr"), S, gamma)
        r = simulate_2d(m, hyb, np.array([np.pi, 0.0, 0.0]),
                        SimConfig(t_end=30.0, seed=0, fall_angle=np.pi))
        w = np.abs(np.arctan2(np.sin(r.X[:, 0]), np.cos(r.X[:, 0])))
        sweep.append(dict(k_unload=ku, t_switch=hyb.t_switch,
                          final_tilt_deg=float(np.rad2deg(w[-1])),
                          final_omega=float(r.X[-1, 2]),
                          balanced=bool(w[-1] < np.deg2rad(2.0))))
    pd.DataFrame(sweep).to_csv("results/E7_unload_sweep.csv", index=False)
    print(pd.DataFrame(sweep).to_string(index=False,
                                        float_format=lambda v: f"{v:.3g}"))

    df, traces = run_all(gamma, S)
    df["gamma"] = gamma
    df["gamma_strict"] = gamma_strict
    df.to_csv("results/E7_swingup.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 1.95))
    ref = traces["lqr"]
    th = np.arctan2(np.sin(ref.X[:, 0]), np.cos(ref.X[:, 0]))

    ax = axes[0]
    ax.plot(np.rad2deg(th), ref.X[:, 1], color=PALETTE["swingup"], lw=0.8)
    # validated handover set, sliced at zero stored momentum
    ang = np.linspace(0, 2 * np.pi, 200)
    Sxy = np.linalg.inv(S[:2, :2])
    evals, evecs = np.linalg.eigh(Sxy)
    ell = (evecs @ (np.sqrt(gamma * evals)[:, None] * np.array([np.cos(ang), np.sin(ang)])))
    ax.plot(np.rad2deg(ell[0]), ell[1], color="k", lw=1.0)
    ax.set_xlabel(r"$\theta$ [deg]"); ax.set_ylabel(r"$\dot\theta$ [rad/s]")
    ax.set_title("(a) swing-up trajectory")
    ax.set_xlim(-190, 190)

    ax = axes[1]
    ax.plot(ref.t, [m.body_energy(x) for x in ref.X], color=PALETTE["swingup"])
    ax.axhline(0.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("time [s]"); ax.set_ylabel("body energy [J]")
    ax.set_title("(b) energy shaping")

    ax = axes[2]
    for name in CTRL:
        r = traces[name]
        ax.plot(r.t, np.abs(r.X[:, 2]) / NOMINAL.omega_max, color=PALETTE[name],
                label=LABEL[name], lw=0.9)
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("time [s]"); ax.set_ylabel("budget used [-]")
    ax.set_title("(c) momentum cost of the swing-up")
    ax.legend(ncol=1, loc="upper right")
    save(fig, "fig_swingup")
