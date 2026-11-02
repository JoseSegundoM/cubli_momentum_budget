"""Figure 1: the testbed, rendered from the simulation model.

Both panels come from the same renderer and the same parameter set the
simulations integrate, so the picture and the physics cannot disagree.
"""

import matplotlib.pyplot as plt
import numpy as np

from cubli.figstyle import COL, save, use_paper_style
from cubli.model3d import Cubli3D, quat_from_axis_angle, quat_mul
from cubli.render import default_camera, frame_limits, render

use_paper_style()
plt.rcParams["figure.constrained_layout.use"] = False

m = Cubli3D()
cam = default_camera(m.p, azim=38.0, elev=15.0)

fig = plt.figure(figsize=(COL, 1.38), dpi=200)
ax1 = fig.add_axes([0.005, 0.13, 0.49, 0.85])
ax2 = fig.add_axes([0.505, 0.13, 0.49, 0.85])

# (a) corner equilibrium, wheels at rest
render(ax1, m, m.state(), cam, spin_phase=np.array([0.0, 0.5, 1.0]))
frame_limits(ax1, m.p, cam)

# (b) tilted, wheel 1 close to its momentum limit
axis = np.cross(m.n_eq, np.array([0.0, 0.0, 1.0]))
q = quat_mul(m.equilibrium_quat(), quat_from_axis_angle(axis, np.deg2rad(13.0)))
render(ax2, m, m.state(q=q, Om=np.array([950.0, 260.0, -180.0])), cam,
       spin_phase=np.array([1.7, 0.4, 2.6]))
frame_limits(ax2, m.p, cam)

# frame_limits reserves a square view with headroom to 1.62 L, which leaves a
# dead band above the machine.  Crop the view to the ink and shrink the figure
# by the same factor so the aspect ratio is untouched.
SHRINK = 0.86
for ax in (ax1, ax2):
    y0, y1 = ax.get_ylim()
    span = (y1 - y0) * SHRINK
    ax.set_ylim(y0, y0 + span)          # keep the ground, drop the headroom
    x0, x1 = ax.get_xlim()
    cx = 0.5 * (x0 + x1)
    ax.set_xlim(cx - 0.5 * (x1 - x0), cx + 0.5 * (x1 - x0))

for ax, txt in ((ax1, "(a) corner equilibrium"), (ax2, "(b) wheel 1 near its limit")):
    ax.text(0.5, -0.01, txt, transform=ax.transAxes, ha="center", va="top",
            fontsize=7)

save(fig, "fig_system", tight=False)
