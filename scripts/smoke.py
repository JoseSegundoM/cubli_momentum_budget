"""Smoke run: all five laws, corner balancing, with and without momentum awareness."""

import numpy as np

from cubli.controllers import make_controller_3d
from cubli.model3d import Cubli3D
from cubli.sim import SimConfig, default_initial_3d, simulate_3d

m = Cubli3D()
cfg = SimConfig(t_end=8.0, seed=0)
x0 = default_initial_3d(m, tilt_deg=5.0)

hdr = f"{'ctrl':<14}{'mm':<5}{'ok':<5}{'t_set':>8}{'over':>8}{'eff':>9}{'|Om|max':>10}{'budget':>8}{'us':>7}"
print(hdr)
print("-" * len(hdr))
for name in ["lqr", "fbl", "smc", "backstepping", "pid"]:
    for k_h in (0.0, 40.0):
        c = make_controller_3d(name, m, k_h=k_h)
        r = simulate_3d(m, c, x0, cfg)
        d = r.metrics
        print(f"{name:<14}{'on' if k_h else 'off':<5}{str(d['success']):<5}"
              f"{d['t_settle']:>8.2f}{d['overshoot_deg']:>8.2f}{d['effort_j']:>9.4f}"
              f"{d['omega_peak']:>10.1f}{d['budget_used']:>8.2f}{d['cpu_us_per_step']:>7.1f}")
