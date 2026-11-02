"""Canonical control-theory views of the plant.

These are not needed for the six-page conference paper, but they are exactly
what a journal version (or a thesis chapter) asks for, and they are cheap once
the model exists:

fig_poles      -- open-loop and closed-loop spectra, and the transmission zero
fig_bode       -- frequency response of the planar plant and of each loop
fig_phase      -- planar phase portrait with the separatrix, the certified
                  handover set and a swing-up trajectory
fig_block      -- the control architecture actually implemented

Everything is computed from the same model the simulations integrate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.signal import bode, ss2tf, TransferFunction

from cubli.controllers.ctrl2d import make_controller_2d
from cubli.controllers.swingup import EnergySwingUp, HybridSwingUpBalance
from cubli.figstyle import COL, LABEL, PALETTE, WIDE, save, use_paper_style
from cubli.model2d import Cubli2D
from cubli.model3d import Cubli3D
from cubli.params import NOMINAL
from cubli.sim import SimConfig, simulate_2d

use_paper_style()
CTRL = ["lqr", "fbl", "smc", "backstepping", "pid"]

m2 = Cubli2D(NOMINAL)
m3 = Cubli3D(NOMINAL)

try:
    from cubli.tuning import load as load_gains
    TUNED = load_gains("results/tuned_gains.json")
except Exception:
    TUNED = None


def gains(name, variant="classical"):
    return TUNED[variant][name]["gains"] if TUNED else {}


# ---------------------------------------------------------------- fig_poles
def fig_poles():
    """Where the plant sits, and where each design puts it.

    The nonlinear laws are linearized about the equilibrium for this plot only:
    feedback linearization, sliding mode (inside the boundary layer) and
    backstepping all reduce to a linear state feedback there, which is what
    makes a spectrum comparison meaningful.
    """
    A3, B3 = m3.linearize()
    A2, B2 = m2.linearize()

    fig, axes = plt.subplots(1, 2, figsize=(WIDE * 0.72, 2.15))

    # (a) open-loop spectrum of the 3-D plant
    ax = axes[0]
    ev = np.linalg.eigvals(A3)
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.plot(ev.real, ev.imag, "x", color=PALETTE["grey"], ms=7, mew=1.4,
            label="open loop")
    for e in ev[ev.real > 1e-6]:
        ax.annotate(f"{e.real:+.2f}", (e.real, e.imag), textcoords="offset points",
                    xytext=(4, 5), fontsize=6, color=PALETTE["fbl"])
    ax.set_xlabel(r"$\Re\{\lambda\}$ [1/s]")
    ax.set_ylabel(r"$\Im\{\lambda\}$ [1/s]")
    ax.set_title("(a) open loop: two unstable, one invariant")
    ax.set_xlim(-25, 25)

    # (b) closed-loop spectra, planar plant
    ax = axes[1]
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.6)
    for c in CTRL:
        K = _linear_gain_2d(c)
        if K is None:
            continue
        cl = np.linalg.eigvals(A2 - B2 @ K.reshape(1, -1))
        ax.plot(cl.real, cl.imag, "o", color=PALETTE[c], ms=4.5, alpha=0.85,
                label=LABEL[c])
    ax.set_xlabel(r"$\Re\{\lambda\}$ [1/s]")
    ax.set_title("(b) closed loop, planar plant")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    save(fig, "fig_poles")


def _linear_gain_2d(name):
    """State feedback each design reduces to near the equilibrium."""
    Ib, Ja, Mgl = m2.Ib, m2.Ja, m2.Mgl
    g = gains(name)
    if name == "lqr":
        return make_controller_2d("lqr", m2, **g).K
    if name == "fbl":
        # the outer LQR supplies (kp, kd); tau = phi + Ib(kp e + kd edot), and
        # phi contributes Mgl e near theta = 0
        c = make_controller_2d("fbl", m2, **g)
        return np.array([Mgl + Ib * c.kp, Ib * c.kd, 0.0])
    if name == "smc":
        lam, k = g.get("lam", 9.0), g.get("k_reach", 60.0)
        phi = g.get("phi", 0.35)
        return np.array([Mgl + Ib * k * lam / phi, Ib * (lam + k / phi), 0.0])
    if name == "backstepping":
        k1, k2 = g.get("k1", 12.0), g.get("k2", 2.5)
        return np.array([Mgl + k2 * k1 + 1.0, k1 * Ib + k2, 0.0])
    if name == "pid":
        kp, kd = g.get("kp", 3.0), g.get("kd", 0.35)
        return np.array([kp, kd, 0.0])
    return None


# ----------------------------------------------------------------- fig_bode
def fig_bode():
    """Frequency response of the planar plant, torque to tilt."""
    A, B = m2.linearize()
    C = np.array([[1.0, 0.0, 0.0]])       # output: tilt
    num, den = ss2tf(A, B, C, np.zeros((1, 1)))
    sys = TransferFunction(num[0], den)
    w = np.logspace(-1, 3, 600)
    _, mag, phase = bode(sys, w=w)

    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.6), sharex=True)
    axes[0].semilogx(w, mag, color=PALETTE["lqr"])
    axes[0].set_ylabel("magnitude [dB]")
    axes[0].set_title(r"planar plant $\tau \rightarrow \theta$")
    axes[1].semilogx(w, phase, color=PALETTE["lqr"])
    axes[1].set_ylabel("phase [deg]")
    axes[1].set_xlabel("frequency [rad/s]")

    z, p = sys.zeros, sys.poles
    txt = (f"poles: {', '.join(f'{v.real:+.2f}' for v in p)}\n"
           f"zeros: {', '.join(f'{v.real:+.2f}' for v in z) if len(z) else 'none'}")
    axes[0].text(0.02, 0.06, txt, transform=axes[0].transAxes, fontsize=6,
                 va="bottom", family="monospace")
    save(fig, "fig_bode")
    return sys


# ---------------------------------------------------------------- fig_phase
def fig_phase():
    """Phase portrait of the planar plant with the separatrix and the handover set."""
    fig, ax = plt.subplots(figsize=(COL, 2.3))

    th = np.linspace(-np.pi, np.pi, 41)
    thd = np.linspace(-9.0, 9.0, 31)
    TH, THD = np.meshgrid(th, thd)
    U = THD
    V = (m2.Mgl * np.sin(TH) - m2.p.b_pivot * THD) / m2.Ib
    ax.streamplot(TH, THD, U, V, color="#C9D2DC", linewidth=0.5, density=0.9,
                  arrowsize=0.6)

    # separatrix: the zero-energy orbit through the upright equilibrium
    thf = np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 400)
    sep = np.sqrt(np.maximum(2.0 * m2.Mgl * (1.0 - np.cos(thf)) / m2.I, 0.0))
    ax.plot(thf, sep, color=PALETTE["fbl"], lw=1.0, label="separatrix ($E=0$)")
    ax.plot(thf, -sep, color=PALETTE["fbl"], lw=1.0)

    # validated handover set, sliced at zero stored momentum
    lqr = make_controller_2d("lqr", m2, **gains("lqr", "momentum_aware"))
    S = lqr.S
    try:
        import pandas as pd
        gamma = float(pd.read_csv("results/E7_swingup.csv")["gamma"].iloc[0])
    except Exception:
        gamma = 0.18
    ang = np.linspace(0, 2 * np.pi, 240)
    Sxy = np.linalg.inv(S[:2, :2])
    ev, evec = np.linalg.eigh(Sxy)
    ell = evec @ (np.sqrt(gamma * ev)[:, None] * np.array([np.cos(ang), np.sin(ang)]))
    ax.plot(ell[0], ell[1], color="k", lw=1.2, label="validated handover set")

    # a swing-up trajectory
    hyb = HybridSwingUpBalance(EnergySwingUp(m2, k_E=8.0, k_unload=20.0),
                               make_controller_2d("lqr", m2,
                                                  **gains("lqr", "momentum_aware")),
                               S, gamma)
    r = simulate_2d(m2, hyb, np.array([np.pi, 0.0, 0.0]),
                    SimConfig(t_end=12.0, seed=0, fall_angle=np.pi))
    wrap = np.arctan2(np.sin(r.X[:, 0]), np.cos(r.X[:, 0]))
    jump = np.abs(np.diff(wrap)) > np.pi / 2
    wrap_plot = wrap.copy()
    wrap_plot[1:][jump] = np.nan
    ax.plot(wrap_plot, r.X[:, 1], color=PALETTE["smc"], lw=0.9, label="swing-up")

    ax.plot([0], [0], "o", color="k", ms=4)
    ax.set_xlabel(r"$\theta$ [rad]")
    ax.set_ylabel(r"$\dot\theta$ [rad/s]")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-9, 9)
    ax.legend(fontsize=6, loc="upper right")
    save(fig, "fig_phase")


# ---------------------------------------------------------------- fig_block
def fig_block():
    """The architecture that is actually implemented, drawn to scale with the code."""
    fig, ax = plt.subplots(figsize=(WIDE * 0.78, 1.85))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 3.4)
    ax.axis("off")

    def block(x, y, w, h, label, color, fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.10",
                                    linewidth=0.9, edgecolor=color,
                                    facecolor=color + "1A"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, label=None, dy=0.16):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, color="#333333",
                                     shrinkA=0, shrinkB=0))
        if label:
            ax.text((x0 + x1) / 2, max(y0, y1) + dy, label, ha="center",
                    fontsize=6.4)

    block(0.15, 1.95, 1.55, 0.75, "momentum\nmanager\n" + r"$\theta_r$", "#0072B2")
    block(2.15, 1.95, 1.75, 0.75, "balancing law\nLQR / FBL / SMC /\nBS / PID", "#009E73", 6.6)
    block(4.40, 1.95, 1.55, 0.75, "actuator\n" + r"$\tau$ sat, $\tau$-$\Omega$" + "\nenvelope", "#E69F00", 6.6)
    block(6.65, 1.95, 1.65, 0.75, "nonlinear\nplant\n(3 wheels)", "#D55E00", 6.8)
    block(4.40, 0.35, 1.55, 0.70, "sensors\ngyro + encoder", "#CC79A7", 6.8)
    block(2.15, 0.35, 1.75, 0.70, "ZOH 500 Hz\n+ 1-sample delay", "#666666", 6.8)

    arrow(1.70, 2.33, 2.15, 2.33, r"$\theta_r$")
    arrow(3.90, 2.33, 4.40, 2.33, r"$\tau_{cmd}$")
    arrow(5.95, 2.33, 6.65, 2.33, r"$\tau$")
    arrow(8.30, 2.33, 9.30, 2.33)
    ax.text(9.45, 2.33, r"$q,\ \omega,\ \Omega$", fontsize=7, va="center")

    # feedback path
    ax.plot([9.30, 9.30, 5.95], [2.33, 0.70, 0.70], color="#333333", lw=0.8)
    arrow(5.95, 0.70, 5.95, 0.70)
    ax.add_patch(FancyArrowPatch((5.98, 0.70), (5.95, 0.70), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color="#333333"))
    arrow(4.40, 0.70, 3.90, 0.70)
    ax.plot([2.15, 0.92, 0.92], [0.70, 0.70, 1.95], color="#333333", lw=0.8)
    ax.add_patch(FancyArrowPatch((0.92, 1.90), (0.92, 1.95), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color="#333333"))
    ax.text(1.35, 0.86, r"$h_w = J_w\Omega$", fontsize=6.4)
    ax.plot([2.95, 2.95], [1.05, 1.95], color="#333333", lw=0.8, ls=":")
    ax.text(3.05, 1.45, "measured state", fontsize=6.2)

    ax.text(0.15, 3.05, "Momentum management closes an outer loop on the stored "
            "wheel momentum; everything else is the usual cascade.",
            fontsize=6.6, color="#444444")
    save(fig, "fig_block")


if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    fig_poles()
    sys = fig_bode()
    print(f"  planar plant poles: {sys.poles}")
    print(f"  planar plant zeros: {sys.zeros}")
    fig_phase()
    fig_block()
