"""The core experiment: a 2 mm center-of-mass offset drains the momentum budget.

The plant has a small manufacturing asymmetry; the controller does not know
about it.  Holding the cube upright therefore requires a permanent torque,
which means a permanently accelerating wheel.  Nothing in the attitude loop
notices until the wheel hits its speed limit -- and then the cube falls.
"""

import sys

import numpy as np

from cubli.controllers import make_controller_3d
from cubli.model3d import Cubli3D
from cubli.params import NOMINAL
from cubli.sim import SimConfig, default_initial_3d, simulate_3d

offset = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0e-3
k_hs = [0.0, 0.2, 0.5, 1.0, 2.0]

plant = Cubli3D(NOMINAL.__class__(com_offset=offset))   # true, asymmetric cube
nominal = Cubli3D(NOMINAL)                              # what the controller believes

cfg = SimConfig(t_end=30.0, seed=1, log_every=20)
x0 = default_initial_3d(plant, tilt_deg=3.0)

print(f"CoM offset = {offset*1e3:.1f} mm   |   run = {cfg.t_end:.0f} s   "
      f"|   wheel budget = {NOMINAL.h_max_wheel:.3f} N.m.s")
hdr = f"{'ctrl':<14}{'k_h':>5}{'ok':>6}{'t_fall':>8}{'peak':>7}{'IAE':>8}{'budget':>8}{'eff':>8}"
print(hdr); print("-" * len(hdr))

for name in ["lqr", "fbl", "smc", "backstepping", "pid"]:
    for k_h in k_hs:
        c = make_controller_3d(name, nominal, k_h=k_h)
        r = simulate_3d(plant, c, x0, cfg)
        d = r.metrics
        tf = "-" if np.isnan(d["t_fall"]) else f"{d['t_fall']:.1f}"
        print(f"{name:<14}{k_h:>5.1f}{('OK' if d['success'] else 'FALL'):>6}{tf:>8}"
              f"{d['peak_tilt_deg']:>7.2f}{d['iae_deg_s']:>8.2f}"
              f"{d['budget_used']:>8.2f}{d['effort_j']:>8.3f}")
