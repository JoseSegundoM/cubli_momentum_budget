"""Photoreal-ish renderer for the Cubli.

``mpl_toolkits.mplot3d`` draws wireframes and flat patches with no lighting and
no reliable depth ordering; it is fine for a scatter plot and wrong for a
figure that is meant to show a machine.  This module is a small painter's
renderer instead: perspective camera, real geometry (a frame of square-section
beams, extruded flywheels with rim, spokes and hub, motor housings), Lambertian
plus specular shading from a key and a fill light, back-face culling, depth
sorting, and a projected ground shadow.

It draws onto an ordinary 2-D matplotlib axes, so the output is a vector PDF
that drops straight into the paper, with no extra dependency beyond matplotlib
and numpy.

The geometry is generated from the same :class:`~cubli.params.CubliParams`
that the simulation integrates, and the attitude comes from the simulated
quaternion, so the picture cannot drift from the physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.collections import PolyCollection

from .model3d import Cubli3D, quat_to_rot
from .params import CubliParams

# ---------------------------------------------------------------- materials

@dataclass
class Material:
    """Base colour plus how the surface responds to light."""
    color: tuple
    ambient: float = 0.34
    diffuse: float = 0.68
    specular: float = 0.22
    shininess: float = 18.0
    edge: float = 0.55        # darkening applied to the polygon outline
    alpha: float = 1.0


ALUMINIUM = Material((0.82, 0.85, 0.88), ambient=0.44, diffuse=0.62,
                     specular=0.32, shininess=28)
ALUMINIUM_DARK = Material((0.68, 0.71, 0.75), ambient=0.40, diffuse=0.60,
                          specular=0.28, shininess=24)
GRAPHITE = Material((0.24, 0.26, 0.29), ambient=0.30, specular=0.34, shininess=40)
STEEL = Material((0.62, 0.64, 0.67), specular=0.42, shininess=48)
MOTOR = Material((0.34, 0.36, 0.40), specular=0.30, shininess=30)


def budget_material(frac: float) -> Material:
    """Rim colour as a function of the momentum budget spent (green->red)."""
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac < 0.5:
        t = frac / 0.5
        c = (0.10 + 0.80 * t, 0.62 + 0.06 * t, 0.42 - 0.42 * t)     # green -> amber
    else:
        t = (frac - 0.5) / 0.5
        c = (0.90 - 0.06 * t, 0.68 - 0.60 * t, 0.00 + 0.06 * t)     # amber -> red
    return Material(c, ambient=0.42, diffuse=0.60, specular=0.30, shininess=24)


# ------------------------------------------------------------------- camera

@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    fov_deg: float = 26.0

    def __post_init__(self):
        self.eye = np.asarray(self.eye, float)
        self.target = np.asarray(self.target, float)
        f = self.target - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, self.up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.basis = np.stack([r, u, -f])          # world -> camera
        self.focal = 1.0 / np.tan(np.deg2rad(self.fov_deg) / 2.0)

    def project(self, P: np.ndarray):
        """World points ``(N,3)`` -> image points ``(N,2)`` and camera depth."""
        cam = (self.basis @ (np.asarray(P, float) - self.eye).T).T
        depth = -cam[:, 2]
        d = np.maximum(depth, 1e-6)
        return np.stack([self.focal * cam[:, 0] / d,
                         self.focal * cam[:, 1] / d], axis=1), depth


# ----------------------------------------------------------------- geometry

def _quad_strip(a, b):
    """Quads joining two closed rings of equal length."""
    n = len(a)
    return [np.array([a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]]) for i in range(n)]


def box(centre, half, R=np.eye(3)):
    """Axis-aligned box, rotated by ``R`` about ``centre``.  Returns 6 quads."""
    c = np.asarray(centre, float)
    hx, hy, hz = half
    v = np.array([[sx * hx, sy * hy, sz * hz]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    v = (R @ v.T).T + c
    idx = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
           [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return [v[i] for i in idx]


def ring(centre, axis, radius, n=48, phase=0.0):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(ref @ axis) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return (np.asarray(centre, float)[None, :]
            + radius * (np.cos(th)[:, None] * u[None, :]
                        + np.sin(th)[:, None] * v[None, :]))


def flywheel(centre, axis, r_out, r_in, thickness, spin=0.0, n=44,
             n_spokes=6, hub_r=0.010):
    """A wheel that looks like a wheel: rim band, two annular faces, spokes, hub.

    ``spin`` rotates the spokes, so a rendered sequence shows the wheel turning.
    """
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = 0.5 * thickness * axis
    c = np.asarray(centre, float)
    faces = []

    o_f, o_b = ring(c + h, axis, r_out, n, spin), ring(c - h, axis, r_out, n, spin)
    i_f, i_b = ring(c + h, axis, r_in, n, spin), ring(c - h, axis, r_in, n, spin)
    faces += [(q, "rim") for q in _quad_strip(o_f, o_b)]        # outer band
    faces += [(q, "face") for q in _quad_strip(i_f, o_f)]       # front annulus
    faces += [(q, "face") for q in _quad_strip(o_b, i_b)]       # back annulus
    faces += [(q, "inner") for q in _quad_strip(i_b, i_f)]      # inner band

    # spokes: flat bars from hub to inner rim
    ref = np.array([1.0, 0.0, 0.0])
    if abs(ref @ axis) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    w = 0.006
    hs = 0.30 * thickness * axis
    for k in range(n_spokes):
        a = spin + 2 * np.pi * k / n_spokes
        d = np.cos(a) * u + np.sin(a) * v
        p = np.cross(axis, d)
        for s in (+1, -1):
            quad = np.array([c + hub_r * d + w * p + s * hs,
                             c + r_in * d + w * p + s * hs,
                             c + r_in * d - w * p + s * hs,
                             c + hub_r * d - w * p + s * hs])
            faces.append((quad, "spoke"))

    hub_f, hub_b = ring(c + 1.25 * h, axis, hub_r, 20), ring(c - 1.25 * h, axis, hub_r, 20)
    faces += [(q, "hub") for q in _quad_strip(hub_f, hub_b)]
    faces.append((hub_f, "hub"))
    faces.append((hub_b[::-1], "hub"))
    return faces


def cubli_geometry(p: CubliParams, R: np.ndarray, Om: np.ndarray | None = None,
                   spin_phase: np.ndarray | None = None, beam: float = 0.0075):
    """All polygons of the machine, in world coordinates, pivot at the origin.

    The frame is modeled the way the hardware is built --- twelve
    square-section beams --- rather than as a solid block, so the wheels are
    visible inside it.
    """
    L = p.L
    pivot_b = p.corner["pivot"]
    Om = np.zeros(3) if Om is None else np.asarray(Om, float)
    spin_phase = np.zeros(3) if spin_phase is None else np.asarray(spin_phase, float)

    def w(pts):
        return (R @ (np.atleast_2d(pts) - pivot_b).T).T

    polys = []   # (verts_world, Material)

    # --- frame: 12 beams along the cube edges ---------------------------
    h = 0.5 * L
    for axis in range(3):
        for s1 in (-h, h):
            for s2 in (-h, h):
                c = np.zeros(3)
                other = [i for i in range(3) if i != axis]
                c[other[0]], c[other[1]] = s1, s2
                half = [beam, beam, beam]
                half[axis] = h + beam
                mat = ALUMINIUM if (s1 > 0 or s2 > 0) else ALUMINIUM_DARK
                for q in box(c, half):
                    polys.append((w(q), mat))

    # --- wheels and motors ----------------------------------------------
    d_seat = 0.5 * L - p.h_mount
    for i in range(3):
        axis_b = np.eye(3)[i]
        centre = d_seat * axis_b
        frac = min(abs(Om[i]) / p.omega_max, 1.0)
        rim_mat = budget_material(frac)
        for quad, kind in flywheel(centre, axis_b, p.r_wheel, 0.74 * p.r_wheel,
                                   0.016, spin=float(spin_phase[i])):
            mat = {"rim": rim_mat, "face": GRAPHITE, "inner": GRAPHITE,
                   "spoke": GRAPHITE, "hub": STEEL}[kind]
            polys.append((w(quad), mat))

        # motor housing behind the wheel, toward the cube centre
        m_c = centre - 0.030 * axis_b
        m_f, m_b = ring(m_c + 0.014 * axis_b, axis_b, 0.021, 24), \
            ring(m_c - 0.014 * axis_b, axis_b, 0.021, 24)
        for q in _quad_strip(m_f, m_b):
            polys.append((w(q), MOTOR))
        polys.append((w(m_b[::-1]), MOTOR))

    return polys


# ------------------------------------------------------------------ shading

def _shade(mat: Material, normal, view_dir, lights):
    n = normal / max(np.linalg.norm(normal), 1e-12)
    col = np.array(mat.color, float) * mat.ambient
    spec = 0.0
    for ldir, intensity in lights:
        l = ldir / np.linalg.norm(ldir)
        lam = max(float(n @ l), 0.0)
        col = col + np.array(mat.color) * mat.diffuse * lam * intensity
        if lam > 0:
            hvec = l + view_dir
            hvec /= max(np.linalg.norm(hvec), 1e-12)
            spec += intensity * mat.specular * max(float(n @ hvec), 0.0) ** mat.shininess
    return np.clip(col + spec, 0.0, 1.0)


# Key light high and slightly behind the camera, so the cast shadow falls
# under the machine instead of streaking away from it.
KEY = (np.array([-0.32, -0.48, 2.30]), 0.88)
FILL = (np.array([0.90, -0.15, 0.45]), 0.28)


def render(ax, model: Cubli3D, x: np.ndarray, camera: Camera | None = None,
           spin_phase=None, shadow: bool = True, ground: bool = False,
           lights=(KEY, FILL), linewidth: float = 0.16, dark: bool = False):
    """Draw one attitude onto ``ax`` (a plain 2-D axes).

    ``dark=True`` lifts the ambient term and darkens the cast shadow so the
    machine reads against a dark background (used by the repository banner);
    the geometry and the lighting directions are unchanged.
    """
    p = model.p
    R = quat_to_rot(x[0:4])
    Om = x[7:10]
    cam = camera or default_camera(p)

    polys = cubli_geometry(p, R, Om, spin_phase)

    verts2d, facecolors, edgecolors, depths = [], [], [], []

    if ground:
        # Off by default: a white field with only the cast shadow reads as a
        # product render, while a finite ground quad puts a visible horizon
        # edge across the figure.
        s = 22.0 * p.L
        g = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], float)
        gp, gd = cam.project(g)
        verts2d.append(gp); facecolors.append((0.955, 0.958, 0.965))
        edgecolors.append("none"); depths.append(gd.mean() + 1e3)

    if shadow:
        # One soft blob (convex hull of the flattened geometry) plus a tighter
        # contact shadow at the pivot.  Projecting every polygon separately
        # reproduces the frame's holes and reads as visual noise.
        l = lights[0][0] / np.linalg.norm(lights[0][0])
        flat_pts = []
        for verts, _ in polys:
            t = verts[:, 2] / max(l[2], 1e-3)
            flat_pts.append(verts - t[:, None] * l[None, :])
        flat = np.vstack(flat_pts)
        flat[:, 2] = 0.0012
        try:
            from scipy.spatial import ConvexHull
            hull = flat[ConvexHull(flat[:, :2]).vertices]
        except Exception:
            hull = flat
        hp, hd = cam.project(hull)
        verts2d.append(hp)
        facecolors.append((0.09, 0.10, 0.12) if dark else (0.72, 0.735, 0.76))
        edgecolors.append("none"); depths.append(hd.mean() + 400.0)

        contact = ring(np.array([0.0, 0.0, 0.0008]), np.array([0.0, 0.0, 1.0]),
                       0.10 * p.L, 28)
        cp, cd = cam.project(contact)
        verts2d.append(cp)
        facecolors.append((0.04, 0.05, 0.06) if dark else (0.36, 0.38, 0.41))
        edgecolors.append("none"); depths.append(cd.mean() + 300.0)

    for verts, mat in polys:
        normal = np.cross(verts[1] - verts[0], verts[2] - verts[0])
        centre = verts.mean(axis=0)
        view = cam.eye - centre
        if float(normal @ view) <= 0.0:          # back-face culling
            continue
        col = _shade(mat, normal, view / np.linalg.norm(view), lights)
        if dark:      # lift the darkest faces so the silhouette does not vanish
            col = np.clip(0.22 + 0.86 * col, 0.0, 1.0)
        pts, d = cam.project(verts)
        verts2d.append(pts)
        facecolors.append(tuple(col))
        edgecolors.append(tuple(np.clip(col * mat.edge, 0, 1)))
        depths.append(d.mean())

    order = np.argsort(-np.asarray(depths))       # far to near
    coll = PolyCollection([verts2d[i] for i in order],
                          facecolors=[facecolors[i] for i in order],
                          edgecolors=[edgecolors[i] for i in order],
                          linewidths=linewidth, antialiased=True)
    ax.add_collection(coll)

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    return coll


def default_camera(p: CubliParams, azim: float = 38.0, elev: float = 15.0,
                   distance: float = 8.2) -> Camera:
    """A camera at eye height, far enough that perspective is subtle."""
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    d = distance * p.L
    eye = np.array([d * np.cos(e) * np.cos(a), d * np.cos(e) * np.sin(a),
                    d * np.sin(e) + 0.55 * p.L])
    return Camera(eye=eye, target=np.array([0.0, 0.0, 0.62 * p.L]))


def frame_limits(ax, p: CubliParams, camera: Camera, pad: float = 0.06):
    """Fit the view to the machine plus a margin, in projected coordinates."""
    L = p.L
    box_pts = np.array([[sx, sy, sz] for sx in (-0.92 * L, 0.92 * L)
                        for sy in (-0.92 * L, 0.92 * L)
                        for sz in (-0.05 * L, 1.62 * L)])
    q, _ = camera.project(box_pts)
    x0, y0 = q.min(axis=0) - pad * L
    x1, y1 = q.max(axis=0) + pad * L
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    half = 0.5 * max(x1 - x0, y1 - y0)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
