"""Figure 2: how a momentum-blind controller fails.

Three runs on the same asymmetric plant: LQR (implicitly momentum-aware),
feedback linearization as classically written, and the same feedback
linearization with the momentum-management augmentation.  The attitude trace
of the classical design is excellent right up to the instant the wheel reaches
its speed limit -- which is the point of the figure.

Layout: short axis labels (the caption defines eta), the friction-envelope
limit eta_lim drawn next to the per-wheel limit, and the tilt axis zoomed so
the managed lean is visible; the fall is marked rather than plotted to 45 deg.
"""

import matplotlib.pyplot as plt
import numpy as np

from cubli.controllers import make_controller_3d
from cubli.experiments import initial_state
from cubli.figstyle import COL, PALETTE, save, use_paper_style
from cubli.model3d import Cubli3D
from cubli.params import NOMINAL
from cubli.sim import SimConfig, simulate_3d
from cubli.tuning import load as load_gains

use_paper_style()

OFFSET = 2.0e-3
try:
    tuned = load_gains("results/tuned_gains.json")
except Exception:
    tuned = None

plant = Cubli3D(NOMINAL.__class__(com_offset=OFFSET))
nominal = Cubli3D(NOMINAL)
cfg = SimConfig(t_end=20.0, seed=3, log_every=10)
x0 = initial_state(plant, 3.0, 0.0, "perp", NOMINAL.omega_max)
P = NOMINAL
# friction-envelope terminal speed: tau_max (1 - W/W_max) = b_w W + tau_c
eta_lim = (P.tau_max - P.tau_coulomb) / (P.b_wheel + P.tau_max / P.omega_max) / P.omega_max


def run(name, variant):
    g = tuned[variant][name]["gains"] if tuned else {}
    k_h = tuned[variant][name]["k_h"] if tuned else (0.0 if variant == "classical" else 0.5)
    c = make_controller_3d(name, nominal, k_h=k_h, **g)
    return simulate_3d(plant, c, x0, cfg)


cases = [
    ("lqr", "classical", PALETTE["lqr"], "-", "LQR"),
    ("fbl", "classical", PALETTE["fbl"], "-", "FBL+LQR, classical"),
    ("fbl", "momentum_aware", PALETTE["smc"], "--", "FBL+LQR + MM"),
]

TILT_MAX = 4.0
# drawn at its printed size (0.72 column) so the text is 6-7 pt on paper
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6})
fig, axes = plt.subplots(3, 1, figsize=(0.72 * COL, 1.85), sharex=True)
for name, variant, col, ls, lab in cases:
    r = run(name, variant)
    eta = np.max(np.abs(r.X[:, 7:10]), axis=1) / P.omega_max
    tilt = np.rad2deg(r.tilt)
    axes[0].plot(r.t, np.minimum(tilt, TILT_MAX + 1), color=col, ls=ls, label=lab)
    axes[1].plot(r.t, eta, color=col, ls=ls)
    axes[2].plot(r.t, np.linalg.norm(r.U, axis=1), color=col, ls=ls, lw=0.7)
    if not r.success:
        tf = r.t[-1]
        for a in axes:
            a.axvline(tf, color=col, lw=0.6, ls=":")
        axes[0].text(tf - 0.9, TILT_MAX * 0.62, f"falls at {tf:.1f} s", fontsize=6,
                     color=col, ha="right", va="center")

axes[0].set_ylabel("tilt [deg]")
axes[0].set_ylim(0, TILT_MAX)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=5.8,
           handlelength=1.6, columnspacing=1.0)
axes[1].axhline(1.0, color="k", lw=0.7, ls=":")
axes[1].axhline(eta_lim, color="k", lw=0.7, ls="--")
axes[1].text(0.4, 1.0, "per-wheel limit", fontsize=5.8, ha="left", va="bottom")
axes[1].text(19.8, eta_lim - 0.03, rf"$\eta_{{\lim}}={eta_lim:.2f}$", fontsize=5.8, ha="right", va="top")
axes[1].set_ylabel(r"$\eta_\Omega$")
axes[1].set_ylim(0, 1.12)
axes[2].set_ylabel(r"$\|\tau\|$ [N m]")
axes[2].set_xlabel("time [s]")
axes[2].set_xlim(0, cfg.t_end)
save(fig, "fig_timeseries")
