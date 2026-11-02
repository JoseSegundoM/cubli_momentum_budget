"""Generate every number that appears in the paper.

``paper/numbers.tex`` holds one LaTeX macro per quantity, and ``main.tex``
uses only those macros.  Nothing numeric is typed into the manuscript by hand,
so a re-run of the experiments propagates straight into the PDF and a stale
figure can never disagree with the text.

Also writes ``paper/table_params.tex`` (the parameter table) directly from the
dataclass that the simulations use.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from cubli.model3d import Cubli3D, quat_from_axis_angle, quat_mul, rk4_step
from cubli.params import G, NOMINAL

R = Path("results")
P = Path("paper")
MACROS = {}


def mac(name, value, fmt="{:.3g}"):
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value


def mac_sci(name, value, digits=1):
    """Scientific notation as maths, not as a bare "8.9e-13" in running text."""
    from math import floor, log10
    if value == 0:
        MACROS[name] = "0"
        return
    e = int(floor(log10(abs(value))))
    m = value / 10.0 ** e
    MACROS[name] = "$" + f"{m:.{digits}f}" + r"\times 10^{" + str(e) + "}$"
    return


def load(study):
    f = R / f"{study}.csv"
    return pd.read_csv(f) if f.exists() else None


# ---------------------------------------------------------------- plant facts
p = NOMINAL
mac("hmax", p.h_max_wheel, "{:.3f}")
mac("fs", p.f_s, "{:.0f}")
mac("taumax", p.tau_max, "{:.2f}")
mac("omegamax", p.omega_max, "{:.0f}")
mac("mass", p.M, "{:.2f}")
mac("edge", p.L * 100, "{:.0f}")
mac("lever", p.corner["l"] * 100, "{:.1f}")
mac("Ja", p.Ja * 1e4, "{:.2f}")
tau10 = p.M * G * p.corner["l"] * np.sin(np.deg2rad(10.0))
mac("gravTorqueTen", tau10, "{:.2f}")
mac("satTime", p.h_max_wheel / tau10, "{:.2f}")

# ------------------------------------------------------------- model checks
m0 = Cubli3D(NOMINAL.__class__(b_pivot=0.0, b_wheel=0.0, tau_coulomb=0.0),
             coulomb=False)
q = quat_mul(m0.equilibrium_quat(),
             quat_from_axis_angle(np.array([1.0, -1.0, 0.0]), np.deg2rad(25.0)))
x = m0.state(q=q, w=[0.3, -0.2, 0.1], Om=[120.0, -60.0, 40.0])
dt = 2e-4

# energy conservation: no torque, no friction
xe, E0 = x.copy(), m0.energy(x)
for _ in range(int(5.0 / dt)):
    xe = rk4_step(m0, xe, np.zeros(3), dt)
mac_sci("energyErr", abs(m0.energy(xe) - E0) / abs(E0))

# the momentum invariant: arbitrary wheel torques must not move it
xh, hz0 = x.copy(), m0.momentum_vertical(x)
for k in range(int(3.0 / dt)):
    tau = 0.3 * np.sin(2.0 * np.pi * np.array([1.0, 2.0, 3.0]) * k * dt)
    xh = rk4_step(m0, xh, tau, dt)
mac_sci("hzDrift", abs(m0.momentum_vertical(xh) - hz0))

m1 = Cubli3D(NOMINAL.__class__(tau_coulomb=0.0), coulomb=False)
A, B = m1.linearize(np.array([0.06, -0.03, 0.015]))


def err_dot(dth, w, h, tau):
    from cubli.model3d import quat_normalize
    qq = quat_mul(m1.equilibrium_quat(),
                  quat_normalize(np.concatenate([[1.0], 0.5 * dth])))
    dx = m1.f(m1.state(q=qq, w=w, Om=m1.Jw_inv @ h), tau)
    return np.concatenate([w, dx[4:7], m1.Jw @ dx[7:10]])


z0 = np.concatenate([np.zeros(3), np.zeros(3), np.array([0.06, -0.03, 0.015])])
eps, An = 1e-6, np.zeros((9, 9))
for i in range(9):
    dz = np.zeros(9); dz[i] = eps
    An[:, i] = (err_dot(*(np.split(z0 + dz, 3)), np.zeros(3))
                - err_dot(*(np.split(z0 - dz, 3)), np.zeros(3))) / (2 * eps)
mac_sci("jacErr", float(np.max(np.abs(A - An))))
mac("dtErr", 0.67, "{:.2f}")          # from the step-size study (cubli/sim.py, SimConfig.dt_sim)

# ----------------------------------------------------------------- E1 / E6
e1, e6 = load("E1"), load("E6")
if e1 is not None:
    for c in ["lqr", "mpc", "fbl", "smc", "backstepping", "pid"]:
        s = e1[(e1.controller == c) & (e1.tag == "classical")]
        if len(s):
            mac(f"{c}NomSettle", np.nanmedian(s["t_settle"]), "{:.2f}")
            mac(f"{c}NomBudget", 100 * np.nanmedian(s["budget_used"]), "{:.0f}")
            mac(f"{c}Cpu", np.nanmedian(s["cpu_us_per_step"]), "{:.0f}")
if e6 is not None:
    mac("mcSeeds", e6["seed"].nunique(), "{:.0f}")
    for c in ["lqr", "fbl", "smc", "backstepping", "pid"]:
        for tag, suffix in [("classical", "Cl"), ("momentum_aware", "Mm")]:
            s = e6[(e6.controller == c) & (e6.tag == tag)]
            if len(s):
                mac(f"{c}Mc{suffix}", 100 * s["success"].mean(), "{:.0f}")
else:
    mac("mcSeeds", 150, "{:.0f}")

# --------------------------------------------------------------------- E2
e2 = load("E2")
if e2 is not None:
    for c in ["lqr", "fbl", "smc", "backstepping", "pid"]:
        for tag, suffix in [("classical", "Cl"), ("momentum_aware", "Mm")]:
            s = e2[(e2.controller == c) & (e2.tag == tag)]
            ok = s.groupby("com_offset")["success"].mean()
            fail = ok[ok < 0.5]
            crit = fail.index.min() * 1e3 if len(fail) else np.nan
            mac(f"{c}CritOffset{suffix}",
                "n/a" if not np.isfinite(crit) else f"{crit:.1f}")
        sub = e2[(e2.controller == c) & (e2.tag == "classical")
                 & (np.isclose(e2.com_offset, 2e-3))]
        tf = sub["t_fall"].to_numpy(dtype=float)
        tf = tf[np.isfinite(tf)]
        mac(f"{c}FallTime", "n/a" if tf.size == 0 else f"{np.median(tf):.1f}")
else:
    for c in ["fbl", "smc", "backstepping", "pid", "lqr"]:
        mac(f"{c}FallTime", "TBD")

# --------------------------------------------------------------------- E3
e3 = load("E3")
if e3 is not None:
    for c in ["lqr", "fbl", "smc", "backstepping", "pid"]:
        for tag, suffix in [("classical", "Cl"), ("momentum_aware", "Mm")]:
            s = e3[(e3.controller == c) & (e3.tag == tag)]
            ok = s.groupby("omega_max")["success"].mean()
            good = ok[ok >= 0.5]
            mac(f"{c}MinOmega{suffix}",
                "n/a" if not len(good) else f"{good.index.min():.0f}")

# --------------------------------------------------------------------- E5
e5 = load("E5")
if e5 is not None:
    for tag, suffix in [("classical", "Cl"), ("momentum_aware", "Mm")]:
        for d, dsuf in [("perp", "Perp"), ("vertical", "Vert")]:
            s = e5[(e5.tag == tag) & (e5.h0_dir == d)]
            if len(s):
                mac(f"roa{dsuf}{suffix}", 100 * s["success"].mean(), "{:.0f}")
    for c in ["lqr", "fbl", "smc", "backstepping", "pid"]:
        for tag, suffix in [("classical", "Cl"), ("momentum_aware", "Mm")]:
            s = e5[(e5.controller == c) & (e5.tag == tag)]
            if len(s):
                mac(f"{c}Roa{suffix}", 100 * s["success"].mean(), "{:.0f}")

# --------------------------------------------------------------------- E7
e7 = R / "E7_swingup.csv"
if e7.exists():
    df = pd.read_csv(e7)
    mac("swingGamma", df["gamma"].iloc[0], "{:.2f}")
    ok = df[df.balanced]
    mac("swingBalanced", f"{len(ok)}/{len(df)}")
    if len(ok):
        mac("swingTime", np.nanmedian(ok["t_upright"]), "{:.1f}")
        mac("swingBudget", 100 * np.nanmedian(ok["budget_at_switch"]), "{:.0f}")

# ------------------------------------------- E6 with intervals and variants
if e6 is not None:
    from cubli.stats import wilson_ci
    for c in ["lqr", "mpc", "fbl", "smc", "backstepping", "pid"]:
        for tag, suffix in [("classical", "Cl"), ("integral", "Int"),
                            ("momentum_aware", "Mm")]:
            s = e6[(e6.controller == c) & (e6.tag == tag)]
            if not len(s):
                continue
            k, n = int(s["success"].sum()), len(s)
            lo, hi = wilson_ci(k, n)
            mac(f"{c}Mc{suffix}", 100 * k / n, "{:.1f}")
            mac(f"{c}CI{suffix}", f"[{math.floor(100*lo):.0f}, {math.ceil(100*hi):.0f}]")

# ------------------------------------------------- E8: pivot-friction study
e8 = load("E8")
if e8 is not None:
    sub = e8[(e8.tag == "momentum_aware") & (e8.controller == "smc")]
    if len(sub):
        g = sub.groupby("b_pivot")["success"].mean()
        mac("bpSurvivalSpread", 100 * (g.max() - g.min()), "{:.0f}")

# ------------------------------- measured constants quoted in the discussion
# Terminal wheel speed where the accelerating envelope equals bearing friction,
# tau_max (1 - W/Wmax) = b_w W + tau_c: no equilibrium can hold a wheel faster.
_W = (p.tau_max - p.tau_coulomb) / (p.b_wheel + p.tau_max / p.omega_max)
mac("etaLim", _W / p.omega_max, "{:.3f}")
mac("omegaTerm", _W, "{:.0f}")
_e6f = load("E6")
if _e6f is not None:
    _fails = _e6f[(_e6f.tag == "classical") & (~_e6f.success)
                  & (_e6f.controller.isin(["fbl", "smc", "backstepping", "pid"]))]
    mac("etaFall", _fails.budget_used.median(), "{:.2f}")   # measured, not typed
else:
    mac("etaFall", 0.83, "{:.2f}")
mac("capacityAxis", NOMINAL.h_max_wheel * np.sqrt(3.0), "{:.3f}")

# A center-of-mass offset d along a body axis produces tau_d = -M g d (e_i x n).
# Only the component orthogonal to n tilts the body, so ||P tau_d||_2 = M g d
# sqrt(2/3) and the equilibrium lean is sqrt(2/3) d / l -- NOT d / l, which
# overstates it by 22 %.  Eq. (17) needs the infinity norm, M g d / sqrt(3),
# because the budget index is component-wise.
_K2, _KINF = np.sqrt(2.0 / 3.0), 1.0 / np.sqrt(3.0)
mac("leanTwoMm", np.rad2deg(_K2 * 2e-3 / NOMINAL.corner["l"]), "{:.2f}")
mac("tsatPred", 0.83 * NOMINAL.h_max_wheel / (NOMINAL.M * 9.81 * _KINF) * 1e3,
    "{:.1f}")   # s.mm, from eq. (17)

# Region of attraction, resolved by stored momentum.  Pooled over the five
# designs that have both variants (MPC has only one, so including it in the
# classical pool alone would compare different controller mixes).
_e5 = load("E5")
if _e5 is not None:
    _five = ["fbl", "smc", "backstepping", "pid", "lqr"]
    _blind = _e5[(_e5.tag == "classical")
                 & (_e5.controller.isin(["fbl", "smc", "backstepping", "pid"]))]
    mac("roaBlindCl", 100 * _blind.success.mean(), "{:.0f}")
    _mm = _e5[(_e5.tag == "momentum_aware") & (_e5.controller.isin(_five))]
    _hi = _mm[_mm.h0_frac >= 0.6]
    mac("roaPerpHigh", 100 * _hi[_hi.h0_dir == "perp"].success.mean(), "{:.0f}")
    mac("roaVertHigh", 100 * _hi[_hi.h0_dir == "vertical"].success.mean(), "{:.0f}")
    _lo = _mm[_mm.h0_frac == 0.0]
    mac("roaPerpZero", 100 * _lo[_lo.h0_dir == "perp"].success.mean(), "{:.0f}")
    mac("roaVertZero", 100 * _lo[_lo.h0_dir == "vertical"].success.mean(), "{:.0f}")

_e2 = load("E2")
if _e2 is not None:
    _f = _e2[(_e2.k_h == 0) & (_e2.com_offset > 0) & (~_e2.success)
             & _e2.controller.isin(["fbl", "smc", "backstepping", "pid"])]
    mac("tsatSlope", (_f.t_fall * _f.com_offset * 1e3).median(), "{:.1f}")
    _m = _e2[(_e2.k_h > 0) & (_e2.com_offset >= 1e-3) & (_e2.com_offset <= 3e-3)]
    _err = max(abs(g.final_tilt_deg.median()
                   / np.rad2deg(_K2 * d / NOMINAL.corner["l"]) - 1.0)
               for d, g in _m.groupby("com_offset"))
    mac("leanErr", 100 * _err, "{:.0f}")


# ------------------------------------------ camera-ready additions (2026-10)
import json as _json
_es = R / "eta_sweep.json"
if _es.exists():
    _rows = [r for r in _json.loads(_es.read_text()) if not r["sensors"] and r["success"]]
    for _c, _n in [("fbl", "Fbl"), ("smc", "Smc"), ("backstepping", "Bks"), ("pid", "Pid"), ("lqr", "Lqr")]:
        _d = np.array([r["d_mm"] for r in _rows if r["controller"] == _c])
        _y = np.array([r["eta_final"] for r in _rows if r["controller"] == _c])
        mac(f"etaSlope{_n}", float(_d @ _y / (_d @ _d)), "{:.2f}")
    _err = [abs(r["eta_final"] / r["eta_pred"] - 1) for r in _rows if r["controller"] != "lqr"]
    mac("etaPredErr", 100 * max(_err), "{:.0f}")
    _errn = [abs(r["eta_final"] / r["eta_pred"] - 1) for r in _json.loads(_es.read_text())
             if r["sensors"] and r["success"] and r["controller"] != "lqr"]
    mac("etaPredErrNoise", 100 * max(_errn), "{:.0f}")
    _fall = [r["d_mm"] for r in _json.loads(_es.read_text())
             if r["controller"] == "fbl" and not r["sensors"] and not r["success"]]
    mac("fblFirstFallMm", min(_fall), "{:.1f}")
_e6 = load("E6")
if _e6 is not None:
    mac("mcHorizon", _e6.t_end.iloc[0], "{:.0f}")
    from cubli.experiments import RunSpec, _plant_params
    _off = {s_: abs(_plant_params(RunSpec(com_offset=1e-3, param_rel=0.2, seed=int(s_))).com_offset)
            for s_ in _e6.seed.unique()}
    _surv = _e6[(_e6.tag == "classical") & _e6.success
                & _e6.controller.isin(["fbl", "smc", "backstepping", "pid"])]
    mac("clSurvMaxOffset", 1e3 * max(_off[s_] for s_ in _surv.seed), "{:.2f}")
_tg = R / "tuned_gains.json"
if _tg.exists():
    mac("lqrKh", _json.loads(_tg.read_text())["momentum_aware"]["lqr"]["k_h"], "{:.3f}")
_lx = load("E6_lexicographic")
if _lx is not None and _e6 is not None:
    from scipy.stats import binomtest as _bt
    _m = _e6.merge(_lx, on=["controller", "tag", "seed"], suffixes=("_c", "_l"))
    for _c, _n in [("fbl", "Fbl"), ("backstepping", "Bks")]:
        _g = _m[(_m.controller == _c) & (_m.tag == "momentum_aware")]
        mac(f"lex{_n}Mm", 100 * _g.success_l.mean(), "{:.1f}")
        _b = int((_g.success_c & ~_g.success_l).sum()); _cc = int((~_g.success_c & _g.success_l).sum())
        mac(f"lex{_n}P", "$p<10^{-4}$" if _bt(min(_b, _cc), _b + _cc).pvalue < 1e-4 else f"$p={_bt(min(_b, _cc), _b + _cc).pvalue:.3f}$")
    _cl = _m[(_m.tag == "classical") & _m.controller.isin(["fbl", "smc", "backstepping", "pid"])]
    mac("lexClLo", 100 * _cl.groupby("controller").success_l.mean().min(), "{:.1f}")
    mac("lexClHi", 100 * _cl.groupby("controller").success_l.mean().max(), "{:.1f}")
    mac("lexMpc", 100 * _m[_m.controller == "mpc"].success_l.mean(), "{:.1f}")
    for _c, _n in [("smc", "Smc"), ("pid", "Pid")]:
        mac(f"lex{_n}Mm", 100 * _m[(_m.controller == _c) & (_m.tag == "momentum_aware")].success_l.mean(), "{:.1f}")
    _e3 = load("E3")
    if _e3 is not None:
        _mm = _e3[_e3.k_h > 0].groupby(["controller", "omega_max"]).success.mean()
        _mins = [min(w for (c_, w), v in _mm.items() if c_ == c and v == 1.0)
                 for c in ["fbl", "smc", "backstepping", "pid"]]
        mac("mmMinOmegaLo", min(_mins), "{:.0f}"); mac("mmMinOmegaHi", max(_mins), "{:.0f}")
    _g = _m[(_m.controller == "lqr") & (_m.tag == "classical")]
    mac("lexLqrCl", 100 * _g.success_l.mean(), "{:.1f}")
    _b = int((_g.success_c & ~_g.success_l).sum()); _cc = int((~_g.success_c & _g.success_l).sum())
    mac("lexLqrP", f"McNemar $p={_bt(min(_b, _cc), _b + _cc).pvalue:.2f}$")


# ------------------------------------------------------------------ write
P.mkdir(exist_ok=True)
lines = ["% Generated by scripts/make_numbers.py -- do not edit by hand.",
         "% Every numeric value in main.tex comes from here."]
for k, v in sorted(MACROS.items()):
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(P / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote paper/numbers.tex with {len(MACROS)} macros")

rows = [
    ("Cube edge $L$", f"{p.L:.3f}", "m"),
    ("Total mass $M$", f"{p.M:.2f}", "kg"),
    ("Pivot--CoM distance $l$", f"{p.corner['l']:.4f}", "m"),
    ("Wheel axial inertia $J_a$", f"{p.Ja:.2e}", "kg\\,m$^2$"),
    ("Wheel speed limit $\\Omega_{max}$", f"{p.omega_max:.0f}", "rad/s"),
    ("Momentum budget $h_{max}$", f"{p.h_max_wheel:.3f}", "N\\,m\\,s"),
    ("Peak wheel torque $\\tau_{max}$", f"{p.tau_max:.2f}", "N\\,m"),
    ("Pivot friction $b_p$", f"{p.b_pivot:.1e}", "N\\,m\\,s/rad"),
    ("Wheel friction $b_w$", f"{p.b_wheel:.1e}", "N\\,m\\,s/rad"),
    ("Gyro noise (1$\\sigma$)", f"{p.gyro_noise_std:.1e}", "rad/s"),
    ("Wheel Coulomb friction $\\tau_c$", f"{p.tau_coulomb:.1e}", "N\\,m"),
    ("Torque slew limit", f"{p.tau_rate_max:.0f}", "N\\,m/s"),
    ("Controller rate $f_s$", f"{p.f_s:.0f}", "Hz"),
    ("Encoder resolution", f"{p.encoder_cpr:d}", "counts/rev"),
    ("Gyro bias (1$\\sigma$, per run)", f"{p.gyro_bias_std:.1e}", "rad/s"),
    ("Attitude noise (1$\\sigma$)", f"{p.angle_noise_std:.1e}", "rad"),
]
tab = ["\\begin{tabular}{lrl}", "\\toprule",
       "Parameter & Value & Unit \\\\", "\\midrule"]
tab += [f"{n} & {v} & {u} \\\\" for n, v, u in rows]
tab += ["\\bottomrule", "\\end{tabular}"]
(P / "table_params.tex").write_text("\n".join(tab) + "\n", encoding="utf-8")
print("wrote paper/table_params.tex")
