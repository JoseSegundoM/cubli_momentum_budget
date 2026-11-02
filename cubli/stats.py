"""Non-parametric statistics for the Monte-Carlo comparison.

Simulation metrics are bounded, skewed and full of censored runs (a fall
truncates the record), so means and t-tests would be the wrong tools.  The
paper reports medians with interquartile ranges, the Wilcoxon signed-rank test
on paired seeds, and Cliff's delta as a distribution-free effect size.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def cliffs_delta(a, b) -> float:
    """Cliff's delta in ``[-1, 1]``: P(a > b) - P(a < b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = np.sign(a[:, None] - b[None, :])
    return float(diff.mean())


def magnitude(delta: float) -> str:
    """Romano et al. thresholds, for reporting rather than for deciding."""
    d = abs(delta)
    if not np.isfinite(d):
        return "n/a"
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def paired_test(a, b) -> dict:
    """Wilcoxon signed-rank on paired samples, with the effect size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    out = {"n": int(a.size), "median_a": float(np.median(a)) if a.size else np.nan,
           "median_b": float(np.median(b)) if b.size else np.nan,
           "cliffs_delta": cliffs_delta(a, b)}
    out["magnitude"] = magnitude(out["cliffs_delta"])
    if a.size >= 6 and np.any(a != b):
        try:
            stat, p = wilcoxon(a, b)
            out.update({"stat": float(stat), "p": float(p)})
        except ValueError:
            out.update({"stat": np.nan, "p": np.nan})
    else:
        out.update({"stat": np.nan, "p": np.nan})
    return out


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple:
    """Wilson score interval for a proportion.

    Reporting a bare "100 %" from 150 trials overstates what the data support;
    the Wilson interval degrades gracefully at the boundaries, unlike the normal
    approximation, which returns a zero-width interval there.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni step-down adjustment over a family of tests."""
    items = [(k, v) for k, v in pvalues.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)      # enforce monotonicity
        out[k] = running
    for k, v in pvalues.items():
        out.setdefault(k, float("nan"))
    return out


def mcnemar_exact(a_success, b_success) -> dict:
    """Exact McNemar test on paired binary outcomes.

    Survival is measured on the *same* plant for both variants, so two
    independent proportions are the wrong summary of the difference: the
    correct test conditions on the discordant pairs.
    """
    a = np.asarray(a_success, dtype=bool)
    b = np.asarray(b_success, dtype=bool)
    n01 = int(np.sum(~a & b))       # only b survives
    n10 = int(np.sum(a & ~b))       # only a survives
    n = a.size
    from scipy.stats import binomtest
    p = (binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
         if (n01 + n10) > 0 else 1.0)
    diff = (b.sum() - a.sum()) / n if n else np.nan
    return {"n": n, "only_a": n10, "only_b": n01, "diff": float(diff),
            "p": float(p)}


def win_ratio(a, b) -> dict:
    """Hierarchical paired comparison (Finkelstein-Schoenfeld / win ratio).

    Conditioning the accuracy comparison on "both variants survived" selects
    the plants the *baseline* could handle, which are precisely the benign
    ones -- a collider that makes the augmentation look worse than it is.  This
    uses every pair instead, with the engineering priority stated explicitly:

    1. if exactly one variant survives, it wins the pair;
    2. if both survive, the smaller error wins;
    3. if both fall, the later fall wins.

    ``a`` and ``b`` are lists of dicts with keys ``success``, ``score`` (lower
    is better) and ``t_fall``.
    """
    from scipy.stats import binomtest
    wins_b = wins_a = ties = 0
    for ra, rb in zip(a, b):
        if ra["success"] != rb["success"]:
            wins_b += int(rb["success"]); wins_a += int(ra["success"])
        elif ra["success"]:
            if rb["score"] < ra["score"]:
                wins_b += 1
            elif ra["score"] < rb["score"]:
                wins_a += 1
            else:
                ties += 1
        else:
            ta = ra.get("t_fall", np.nan); tb = rb.get("t_fall", np.nan)
            ta = -np.inf if not np.isfinite(ta) else ta
            tb = -np.inf if not np.isfinite(tb) else tb
            if tb > ta:
                wins_b += 1
            elif ta > tb:
                wins_a += 1
            else:
                ties += 1
    total = wins_a + wins_b
    p = binomtest(wins_b, total, 0.5).pvalue if total else 1.0
    return {"wins_b": wins_b, "wins_a": wins_a, "ties": ties,
            "ratio": (wins_b / wins_a) if wins_a else np.inf, "p": float(p)}


def rank_biserial(a, b) -> float:
    """Matched-pairs rank-biserial correlation, the effect size that matches
    a paired Wilcoxon.  Cliff's delta is an unpaired dominance measure and
    averages the within-pair effect away."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    d = d[d != 0]
    if d.size == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def summarise(x) -> dict:
    """Median and interquartile range, ignoring non-finite entries."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"median": np.nan, "q1": np.nan, "q3": np.nan, "n": 0}
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "n": int(x.size)}
