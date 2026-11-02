"""Run the full experiment battery and write one CSV per study to ``results/``.

Studies
-------
E1  nominal    -- corner balancing from a 3 deg tilt, all controllers, both variants
E2  offset     -- sweep of the center-of-mass asymmetry that drains the budget
E3  omega_max  -- sweep of the wheel speed limit: the momentum budget itself
E4  tau_max    -- sweep of the peak torque: the limit the literature does model
E5  roa        -- region of attraction in (initial tilt, stored momentum),
                  separately for momentum stored along the invariant direction
                  and perpendicular to it
E6  mc         -- Monte-Carlo robustness, +/-20 % on every physical parameter

Usage: ``python scripts/run_experiments.py [study ...]`` (default: all).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cubli.experiments import CONTROLLERS, SINGLE_VARIANT, RunSpec, run_many
from cubli.tuning import load as load_gains

RESULTS = Path("results")
VARIANTS = ["classical", "momentum_aware", "integral"]


def gains_for(tuned, variant, controller):
    """(k_h, k_i) for this controller/variant, from the fair-tuning stage."""
    if tuned and controller in tuned.get(variant, {}):
        blk = tuned[variant][controller]
        return blk.get("k_h", 0.0), blk.get("k_i", 0.0)
    return (0.0 if variant != "momentum_aware" else 0.5,
            0.0 if variant != "integral" else 0.5)


def build(study, tuned):
    specs = []
    for variant in VARIANTS:
        for c in CONTROLLERS:
            if c in SINGLE_VARIANT and variant != "classical":
                continue
            kh, ki = gains_for(tuned, variant, c)
            if study == "E1":
                for seed in range(10):
                    specs.append(RunSpec(controller=c, k_h=kh, k_i=ki, tilt0_deg=3.0,
                                         t_end=20.0, seed=seed, tag=variant))
            elif study == "E2":
                for off in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
                    for seed in range(3):
                        specs.append(RunSpec(controller=c, k_h=kh, k_i=ki, tilt0_deg=3.0,
                                             com_offset=off * 1e-3, t_end=20.0,
                                             seed=seed, tag=variant))
            elif study == "E3":
                for om in [250.0, 400.0, 600.0, 800.0, 1000.0, 1400.0, 2000.0]:
                    for seed in range(3):
                        specs.append(RunSpec(controller=c, k_h=kh, k_i=ki, tilt0_deg=3.0,
                                             com_offset=2e-3, omega_max=om,
                                             t_end=20.0, seed=seed, tag=variant))
            elif study == "E4":
                for tm in [0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]:
                    for seed in range(3):
                        specs.append(RunSpec(controller=c, k_h=kh, k_i=ki, tilt0_deg=6.0,
                                             com_offset=2e-3, tau_max=tm,
                                             t_end=20.0, seed=seed, tag=variant))
            elif study == "E5":
                for tilt in np.linspace(1.0, 24.0, 12):
                    for frac in np.linspace(0.0, 0.8, 9):
                        for d in ("perp", "vertical"):
                            specs.append(RunSpec(controller=c, k_h=kh, k_i=ki,
                                                 tilt0_deg=float(tilt),
                                                 h0_frac=float(frac), h0_dir=d,
                                                 com_offset=1e-3, t_end=25.0,
                                                 seed=7, tag=variant))
            elif study == "E8":
                # Theorem 1 assumes an ideal pivot; every other study uses
                # b_p = 1e-3.  Sweeping it separates the proposed gravity-based
                # unloading from passive dissipation through the pivot.
                for bp in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
                    for seed in range(3):
                        specs.append(RunSpec(controller=c, k_h=kh, k_i=ki,
                                             tilt0_deg=3.0, com_offset=2e-3,
                                             b_pivot=bp, t_end=20.0, seed=seed,
                                             tag=variant))
            elif study == "E6":
                for seed in range(150):
                    specs.append(RunSpec(controller=c, k_h=kh, k_i=ki, tilt0_deg=3.0,
                                         com_offset=1e-3, param_rel=0.20,
                                         t_end=15.0, seed=seed, tag=variant))
    return specs


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    tuned = None
    gp = RESULTS / "tuned_gains.json"
    if gp.exists():
        tuned = load_gains(gp)
        print(f"using tuned gains from {gp}")
    else:
        print("WARNING: no tuned gains found, using library defaults")

    studies = sys.argv[1:] or ["E1", "E2", "E3", "E4", "E5", "E6", "E8"]
    for study in studies:
        specs = build(study, tuned)
        print(f"\n== {study}: {len(specs)} runs ==", flush=True)
        t0 = time.time()
        rows = []
        for variant in VARIANTS:
            sub = [s for s in specs if s.tag == variant]
            g = ({c: tuned[variant][c]["gains"] for c in tuned.get(variant, {})}
                 if tuned else None)
            rows += run_many(sub, g)
        df = pd.DataFrame(rows)
        out = RESULTS / f"{study}.csv"
        df.to_csv(out, index=False)
        print(f"   -> {out}  ({len(df)} rows, {time.time() - t0:.0f} s, "
              f"success {df['success'].mean():.1%})", flush=True)
