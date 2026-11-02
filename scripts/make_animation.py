"""Side-by-side animation: the same controller, with and without momentum
management, on the same asymmetric plant.

Left panel spends its momentum budget and falls; right panel leans slightly,
lets gravity unload the wheels, and keeps balancing.  The wheels spin at the
simulated rate and their rims are coloured by the fraction of the momentum
budget consumed, so the failure is visible before it happens.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from cubli.controllers import make_controller_3d
from cubli.experiments import initial_state
from cubli.figstyle import use_paper_style
from cubli.model3d import Cubli3D
from cubli.params import NOMINAL
from cubli.render import default_camera, frame_limits, render
from cubli.sim import SimConfig, simulate_3d

use_paper_style()
plt.rcParams["axes.grid"] = False
plt.rcParams["figure.constrained_layout.use"] = False
plt.rcParams["savefig.dpi"] = 150

# matplotlib needs an ffmpeg binary to write MP4.  imageio-ffmpeg ships one, so
# the animation does not depend on a system install; without it the writer
# falls back to GIF, which is an order of magnitude larger for the same clip.
try:
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

CTRL = "fbl"
OFFSET = 2.0e-3
T_END = 14.0
STRIDE = 2

try:
    from cubli.tuning import load as load_gains
    tuned = load_gains("results/tuned_gains.json")
except Exception:
    tuned = None

plant = Cubli3D(NOMINAL.__class__(com_offset=OFFSET))
nominal = Cubli3D(NOMINAL)
cfg = SimConfig(t_end=T_END, seed=3, log_every=20)
x0 = initial_state(plant, 3.0, 0.0, "perp", NOMINAL.omega_max)

runs, spins = {}, {}
for variant in ("classical", "momentum_aware"):
    g = tuned[variant][CTRL]["gains"] if tuned else {}
    k_h = tuned[variant][CTRL]["k_h"] if tuned else (0.0 if variant == "classical" else 0.5)
    c = make_controller_3d(CTRL, nominal, k_h=k_h, **g)
    r = simulate_3d(plant, c, x0, cfg)
    runs[variant] = r
    # integrate the wheel angle so the rendered wheels actually turn (scaled
    # down: at 1000 rad/s the true rate would just alias between frames)
    dt = np.diff(r.t, prepend=r.t[0])[:, None]
    spins[variant] = np.cumsum(r.X[:, 7:10] * dt, axis=0) * 0.06
    print(f"{variant:<16} success={r.success}  peak tilt="
          f"{r.metrics['peak_tilt_deg']:.1f} deg  budget={r.metrics['budget_used']:.2f}")

n_frames = max(len(r.t) for r in runs.values()) // STRIDE
cam = default_camera(plant.p, azim=38.0, elev=15.0)

fig = plt.figure(figsize=(6.0, 3.1), dpi=150)
axes = [fig.add_axes([0.01 + 0.5 * i, 0.16, 0.48, 0.80]) for i in range(2)]
titles = ["classical design", "with momentum management"]
labels = [fig.text(0.045 + 0.5 * i, 0.12, "", fontsize=7.5, family="monospace",
                   va="top") for i in range(2)]


def frame(k):
    for ax, lab, variant, title in zip(axes, labels, runs, titles):
        r = runs[variant]
        ax.clear()
        i = min(k * STRIDE, len(r.t) - 1)
        fallen = (not r.success) and i == len(r.t) - 1
        render(ax, plant, r.X[i], cam, spin_phase=spins[variant][i])
        frame_limits(ax, plant.p, cam)
        budget = float(np.max(np.abs(r.X[i, 7:10])) / plant.p.omega_max)
        state = "FALLEN" if fallen else "balancing"
        lab.set_text(f"{title}\n"
                     f"t {r.t[i]:5.2f} s   tilt {np.rad2deg(r.tilt[i]):5.1f} deg\n"
                     f"budget {budget:4.0%}   {state}")
        lab.set_color("#B2182B" if fallen else "#222222")
    return ()


anim = animation.FuncAnimation(fig, frame, frames=n_frames, interval=40)
out = "figures/cubli_momentum.mp4"
try:
    anim.save(out, writer=animation.FFMpegWriter(
        fps=25, bitrate=2600,
        extra_args=["-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "24"]))
except Exception as exc:
    print(f"  ffmpeg unavailable ({exc.__class__.__name__}); writing GIF instead")
    out = "figures/cubli_momentum.gif"
    anim.save(out, writer=animation.PillowWriter(fps=18))
plt.close(fig)
print(f"wrote {out}")
