"""Shared figure style: IEEE two-column geometry, colour-blind-safe palette.

Figures are produced at their final printed size (no rescaling in LaTeX), with
8 pt type matching the caption size, so the text in a figure is legible in the
column rather than shrunk to nothing.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# IEEE conference template, in inches
COL = 3.45          # single column
WIDE = 7.16         # full page width

# Okabe-Ito: distinguishable under the common forms of colour blindness and in
# greyscale print, which a conference proceedings may well be.
PALETTE = {
    "lqr": "#0072B2",           # blue
    "fbl": "#D55E00",           # vermillion
    "smc": "#009E73",           # bluish green
    "backstepping": "#CC79A7",  # reddish purple
    "pid": "#E69F00",           # orange
    "mpc": "#8C6BB1",           # purple
    "swingup": "#56B4E9",       # sky blue
    "grey": "#666666",
    "bad": "#D55E00",
    "good": "#009E73",
}

LABEL = {
    "lqr": "LQR",
    "mpc": "MPC",
    "fbl": "FBL+LQR",
    "smc": "SMC",
    "backstepping": "Backstepping",
    "pid": "PID",
    "swingup": "Swing-up",
}

MARKER = {"lqr": "o", "mpc": "P", "fbl": "s", "smc": "^", "backstepping": "D", "pid": "v"}


def use_paper_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed TrueType: IEEE PDF eXpress requires it
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",          # Times-like maths, as in the text
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "grid.alpha": 0.30,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def save(fig, name: str, outdir: str = "figures", tight: bool = True) -> str:
    from pathlib import Path

    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = str(Path(outdir) / f"{name}.pdf")
    kw = {} if tight else {"bbox_inches": None}
    fig.savefig(path, **kw)
    fig.savefig(path.replace(".pdf", ".png"), **kw)
    plt.close(fig)
    print(f"  wrote {path}")
    return path
