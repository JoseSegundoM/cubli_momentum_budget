"""Result figures and tables, all read from ``results/*.csv``.

fig_roa      -- empirical recovery region over initial tilt and pre-stored
                wheel momentum, for momentum stored perpendicular to the
                balancing axis and along it (the invariant direction)
fig_sweeps   -- survival against the center-of-mass asymmetry, the wheel-speed
                limit and the peak torque
fig_mc       -- Monte-Carlo survival and budget, three variants
table_main   -- LaTeX and Markdown tables of the nominal and Monte-Carlo metrics
mc_stats     -- paired statistics: McNemar on survival, a hierarchical win
                ratio over all plants, Wilcoxon on the survivors, Holm-corrected
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cubli.experiments import CONTROLLERS as CTRL, SINGLE_VARIANT
from cubli.figstyle import COL, LABEL, MARKER, PALETTE, WIDE, save, use_paper_style
from cubli.stats import (holm, mcnemar_exact, paired_test, rank_biserial,
                         summarise, wilson_ci, win_ratio)

use_paper_style()
R = Path("results")
VAR = ["classical", "integral", "momentum_aware"]
VAR_LABEL = {"classical": "cl.", "integral": "int.", "momentum_aware": "MM"}


def load(study):
    p = R / f"{study}.csv"
    return pd.read_csv(p) if p.exists() else None


def applicable(c, variant):
    return not (c in SINGLE_VARIANT and variant != "classical")


# ---------------------------------------------------------------- fig_roa
def fig_roa():
    """What momentum management buys, cell by cell, for every design compared.

    red = neither variant recovers, amber = only the momentum-aware one,
    blue = only the classical one, green = both.  Each design gets a pair of
    rows: momentum stored perpendicular to the balancing axis, where gravity
    can dump it, above momentum stored along the axis, where nothing can.
    MPC has a single variant, so its panels resolve to green/red.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    df = load("E5")
    if df is None:
        return
    cols = [c for c in CTRL if c in set(df.controller)]
    # Colour-blind-safe and distinct in greyscale (light grey / orange /
    # purple / dark blue) instead of red against green
    cmap = ListedColormap(["#E0E0E0", "#E69F00", "#CC79A7", "#0B4F8A"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    # Two blocks of three designs, each block a (dumpable, invariant) pair of
    # rows, so all six fit a single column without shrinking the panels.
    ncol = 3
    blocks = [cols[i:i + ncol] for i in range(0, len(cols), ncol)]
    nrow = 2 * len(blocks)
    fig, axes = plt.subplots(nrow, ncol, figsize=(COL, 2.28),
                             sharex=True, sharey=True)

    for b, block in enumerate(blocks):
        for k, direction in enumerate(["perp", "vertical"]):
            row = 2 * b + k
            for col in range(ncol):
                ax = axes[row, col]
                if col >= len(block):
                    ax.axis("off")
                    continue
                c = block[col]
                sub = df[(df.controller == c) & (df.h0_dir == direction)]
                cl = sub[sub.tag == "classical"].pivot_table(
                    index="h0_frac", columns="tilt0_deg", values="success")
                mm = sub[sub.tag == "momentum_aware"].pivot_table(
                    index="h0_frac", columns="tilt0_deg", values="success")
                if cl.empty:
                    continue
                if mm.empty:      # single-variant design: mm coincides with cl
                    mm = cl
                code = ((cl.values >= 0.5).astype(int) * 2
                        + (mm.values >= 0.5).astype(int))
                ax.pcolormesh(cl.columns.values, cl.index.values * 100.0, code,
                              cmap=cmap, norm=norm, shading="nearest",
                              rasterized=True)
                if k == 0:
                    ax.set_title(LABEL[c], pad=2.0)
                if row == nrow - 1:
                    ax.set_xlabel("initial tilt [deg]")
                ax.grid(False)
            tag = r"$h_0\perp n$" if direction == "perp" else r"$h_0\parallel n$"
            axes[row, 0].set_ylabel(tag + chr(10) + "[% budget]", fontsize=6)

    handles = [Patch(facecolor="#0B4F8A", label="both recover"),
               Patch(facecolor="#E69F00", label="only with momentum mgmt."),
               Patch(facecolor="#CC79A7", label="only classical"),
               Patch(facecolor="#E0E0E0", label="neither")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7,
               bbox_to_anchor=(0.5, -0.17), frameon=False)
    save(fig, "fig_roa")


# ------------------------------------------------------------- fig_sweeps
def _survival(df, xcol, scale=1.0):
    out = {}
    for variant in VAR:
        for c in CTRL:
            sub = df[(df.controller == c) & (df.tag == variant)]
            if not len(sub):
                continue
            g = sub.groupby(xcol)["success"].mean()
            out[(variant, c)] = (g.index.values * scale, g.values)
    return out


def fig_sweeps():
    """Balanced runs against plant asymmetry and momentum budget.

    One row per design and variant, one column per swept value; each cell
    shows how many of the runs at that point balanced (3 sensor-noise seeds,
    20 s).  A matrix replaces the earlier overlaid curves, which had to be
    offset vertically to stay visible.
    """
    from matplotlib.colors import ListedColormap

    e2, e3 = load("E2"), load("E3")
    if e2 is None or e3 is None:
        return
    rows = []
    for c in CTRL:
        for variant in ("classical", "momentum_aware"):
            if ((e2.controller == c) & (e2.tag == variant)).any():
                rows.append((c, variant))
    labels = [LABEL[c] + (" + MM" if v == "momentum_aware" else "") for c, v in rows]
    panels = [(e2, "com_offset", 1e3, "CoM offset [mm]", "(a) plant asymmetry",
               lambda x: f"{x:g}"),
              (e3, "omega_max", 1.0, r"$\Omega_{\max}$ [rad/s]",
               "(b) momentum budget", lambda x: f"{x:.0f}")]
    widths = [len(df[xc].unique()) for df, xc, *_ in panels]
    fig, axes = plt.subplots(1, 2, figsize=(0.80 * COL, 1.85), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    cmap = ListedColormap(["#FFFFFF", "#C6DBEF", "#6BAED6", "#08519C"])
    for ax, (df, xcol, sc, xlab, title, fmt) in zip(axes, panels):
        xs = sorted(df[xcol].unique())
        n = int(df.groupby(["controller", "tag", xcol]).size().max())
        M = np.full((len(rows), len(xs)), np.nan)
        for r, (c, v) in enumerate(rows):
            sub = df[(df.controller == c) & (df.tag == v)]
            k = sub.groupby(xcol)["success"].sum()
            for j, x in enumerate(xs):
                if x in k.index:
                    M[r, j] = k[x]
        ax.imshow(M, cmap=cmap, vmin=-0.5, vmax=n + 0.5, aspect="auto")
        for r in range(len(rows)):
            for j in range(len(xs)):
                if not np.isnan(M[r, j]):
                    ax.text(j, r, f"{int(M[r, j])}", ha="center", va="center", fontsize=6,
                            color="white" if M[r, j] >= n - 0.5 else "black")
        ax.set_xticks(range(len(xs)), [fmt(x * sc) for x in xs], rotation=90, fontsize=6)
        ax.set_yticks(range(len(rows)), labels, fontsize=6)
        ax.set_xlabel(xlab, fontsize=7)
        ax.set_title(title, fontsize=7.5)
        ax.grid(False)
        ax.tick_params(length=0)
        for r in range(1, len(rows)):
            if rows[r][0] != rows[r - 1][0]:
                ax.axhline(r - 0.5, color="k", lw=0.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    save(fig, "fig_sweeps")


# ----------------------------------------------------------------- fig_mc
def fig_mc():
    """Monte-Carlo survival and momentum use.

    Survival is the primary endpoint and is shown for all three variants.  The
    budget panel is conditioned on the runs that balanced, because a fallen run
    truncates its own record.
    """
    df = load("E6")
    if df is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.15))

    ax = axes[0]
    w, xs = 0.27, np.arange(len(CTRL))
    for j, (variant, alpha) in enumerate(zip(VAR, [0.32, 0.6, 1.0])):
        vals = []
        for c in CTRL:
            sub = df[(df.controller == c) & (df.tag == variant)]
            vals.append(100 * sub["success"].mean() if len(sub) else np.nan)
        ax.bar(xs + (j - 1) * w, vals, w, color=[PALETTE[c] for c in CTRL],
               alpha=alpha, edgecolor="none")
    ax.set_xticks(xs)
    ax.set_xticklabels([])
    ax.set_ylabel("balanced [%]")
    ax.set_ylim(0, 105)
    ax.set_title("(a) survival: pale classical, mid integral, solid momentum mgmt.")

    ax = axes[1]
    data, colours = [], []
    for c in CTRL:
        for variant in ("classical", "momentum_aware"):
            sub = df[(df.controller == c) & (df.tag == variant) & df.success]
            v = sub["budget_used"].to_numpy(dtype=float) if len(sub) else np.array([])
            data.append(v[np.isfinite(v)] if v.size else np.array([np.nan]))
            colours.append(PALETTE[c])
    bp = ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=False,
                    medianprops=dict(color="k", lw=0.9))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colours[i])
        patch.set_alpha(0.45 if i % 2 == 0 else 0.9)
        patch.set_edgecolor(colours[i])
    ax.set_xticks(np.arange(1, len(data) + 1, 2) + 0.5)
    ax.set_xticklabels([LABEL[c] for c in CTRL], rotation=25, ha="right",
                       fontsize=6)
    ax.set_ylabel(r"$\eta_\Omega$ used [-]")
    ax.set_title("(b) momentum budget, survivors only")
    save(fig, "fig_mc")


# ------------------------------------------------------------- table_main
def table_main():
    e1, e6 = load("E1"), load("E6")
    if e1 is None or e6 is None:
        return
    rows = []
    # the integral variant is reported in the text: it moves nothing, and the
    # table has to fit one column
    for c in CTRL:
        for variant in ("classical", "momentum_aware"):
            if not applicable(c, variant):
                continue
            n = e1[(e1.controller == c) & (e1.tag == variant)]
            m = e6[(e6.controller == c) & (e6.tag == variant)]
            if not len(m):
                continue
            ok = m[m.success]
            k, N = int(m["success"].sum()), len(m)
            lo, hi = wilson_ci(k, N)
            rows.append(dict(
                controller=LABEL[c], variant=VAR_LABEL[variant],
                nom_settle=np.nanmedian(n["t_settle"]) if len(n) else np.nan,
                nom_effort=np.nanmedian(n["effort_j"]) if len(n) else np.nan,
                cpu=np.nanmedian(n["cpu_us_per_step"]) if len(n) else np.nan,
                mc_k=k, mc_n=N, mc_success=100.0 * k / N,
                mc_lo=100 * lo, mc_hi=100 * hi,
                mc_iae=summarise(ok["iae_deg_s"])["median"],
                mc_budget=np.nanmedian(ok["budget_used"]) if len(ok) else np.nan,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(R / "table_main.csv", index=False)

    head = (r"Controller & Var. & $t_s$ [s] & $\int\|\tau\|^2$ & "
            r"bal.\ [\%] & 95\,\% CI & IAE$^\dagger$ \\")
    lines = [r"\setlength{\tabcolsep}{3.0pt}",
             r"\begin{tabular}{@{}llrrrcr@{}}", r"\toprule",
             r"& & \multicolumn{2}{c}{Nominal} & \multicolumn{3}{c}{Monte-Carlo} \\",
             r"\cmidrule(lr){3-4}\cmidrule(l){5-7}", head, r"\midrule"]
    for _, r_ in df.iterrows():
        lines.append(
            f"{r_.controller} & {r_.variant} & {r_.nom_settle:.2f} & "
            f"{r_.nom_effort:.3f} & {r_.mc_success:.1f} & "
            f"[{math.floor(r_.mc_lo):.0f}, {math.ceil(r_.mc_hi):.0f}] & {r_.mc_iae:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path("paper").mkdir(exist_ok=True)
    (Path("paper") / "table_main.tex").write_text(chr(10).join(lines), encoding="utf-8")
    print("  wrote paper/table_main.tex")

    md = ["| Controller | Variant | t_s [s] | effort | MC balanced [%] | 95% CI | MC IAE |",
          "|:--|:--|--:|--:|--:|:--:|--:|"]
    for _, r_ in df.iterrows():
        md.append(f"| {r_.controller} | {r_.variant} | {r_.nom_settle:.2f} | "
                  f"{r_.nom_effort:.3f} | {r_.mc_success:.1f} | "
                  f"[{math.floor(r_.mc_lo):.0f}, {math.ceil(r_.mc_hi):.0f}] | {r_.mc_iae:.1f} |")
    (R / "table_main.md").write_text(chr(10).join(md) + chr(10), encoding="utf-8")
    print("  wrote results/table_main.md")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


# ------------------------------------------------------------ mc statistics
def mc_stats():
    """Paired analysis, done the way the design demands.

    Both variants run on the *same* plant per seed, so survival is compared
    with McNemar rather than with two independent proportions.  The primary
    endpoint is a hierarchical win ratio over all plants: survival decides a
    pair, then accuracy, then time-to-fall.  Conditioning accuracy on "both
    survived" selects the plants the baseline could already handle, so that
    comparison is reported as a labelled secondary.  Holm-corrected over the
    declared family.
    """
    df = load("E6")
    if df is None:
        return
    rows, praw = [], {}
    for c in CTRL:
        a = df[(df.controller == c) & (df.tag == "classical")].sort_values("seed")
        b = df[(df.controller == c) & (df.tag == "momentum_aware")].sort_values("seed")
        if not len(b):
            continue
        n = min(len(a), len(b))
        a, b = a.iloc[:n], b.iloc[:n]

        mc = mcnemar_exact(a["success"].to_numpy(), b["success"].to_numpy())
        wr = win_ratio(
            [dict(success=bool(s), score=float(i), t_fall=float(t))
             for s, i, t in zip(a.success, a.iae_deg_s, a.t_fall)],
            [dict(success=bool(s), score=float(i), t_fall=float(t))
             for s, i, t in zip(b.success, b.iae_deg_s, b.t_fall)])
        both = a["success"].to_numpy() & b["success"].to_numpy()
        cond = paired_test(a["iae_deg_s"].to_numpy()[both],
                           b["iae_deg_s"].to_numpy()[both])
        rb = rank_biserial(a["iae_deg_s"].to_numpy()[both],
                           b["iae_deg_s"].to_numpy()[both])
        praw[f"survival::{c}"] = mc["p"]
        praw[f"accuracy::{c}"] = cond["p"]
        praw[f"win::{c}"] = wr["p"]      # the paper reports this corrected too
        rows.append(dict(controller=c, n=n,
                         surv_cl=100 * a["success"].mean(),
                         surv_mm=100 * b["success"].mean(),
                         mcnemar_p=mc["p"], only_cl=mc["only_a"], only_mm=mc["only_b"],
                         win_ratio=wr["ratio"], win_p=wr["p"],
                         wins_mm=wr["wins_b"], wins_cl=wr["wins_a"],
                         cond_n=int(both.sum()), cond_p=cond["p"],
                         rank_biserial=rb))
    adj = holm(praw)
    for r_ in rows:
        r_["mcnemar_p_holm"] = adj[f"survival::{r_['controller']}"]
        r_["cond_p_holm"] = adj[f"accuracy::{r_['controller']}"]
        r_["win_p_holm"] = adj[f"win::{r_['controller']}"]
    out = pd.DataFrame(rows)
    out.to_csv(R / "mc_stats.csv", index=False)

    print(chr(10) + "Monte-Carlo, classical vs momentum-aware "
          f"(paired, Holm over {len(praw)} tests):")
    for _, r_ in out.iterrows():
        print(f"  {LABEL[r_.controller]:<14} survival {r_.surv_cl:5.1f} -> "
              f"{r_.surv_mm:5.1f} %  McNemar p={r_.mcnemar_p_holm:.1e}"
              f"  | win ratio {r_.win_ratio:5.2f} "
              f"({r_.wins_mm:.0f}/{r_.wins_cl:.0f}) p={r_.win_p:.1e}"
              f"  | survivors n={r_.cond_n:.0f} r_rb={r_.rank_biserial:+.2f}")


if __name__ == "__main__":
    fig_roa()
    fig_sweeps()
    fig_mc()
    table_main()
    mc_stats()
