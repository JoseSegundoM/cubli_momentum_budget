"""Figure 1: vector schematic of the corner-balancing cube, written as TikZ.

Orthographic side view of a unit cube standing on a vertex.  Body axes e_i run
along the three edges leaving the pivot; the balancing axis n is the body
diagonal.  The whole body is drawn tilted by an exaggerated lean so that the
gravity-balanced equilibrium of Proposition 2 can be labelled; geometry is the
idealised design point (no wheel thickness).  Output: paper/fig_schematic.tex.
"""
import numpy as np

LEAN = np.deg2rad(7.0)      # exaggerated for legibility; the real lean is ~0.6 deg
VIEW = np.deg2rad(42.0)     # azimuth of the viewer

# cube with the diagonal (1,1,1) vertical: rotate (1,1,1)/sqrt3 onto z
n = np.ones(3) / np.sqrt(3)
z = np.array([0, 0, 1.0])
v = np.cross(n, z); s = np.linalg.norm(v); c = n @ z
K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
R0 = np.eye(3) + K + K @ K * ((1 - c) / s**2)
Ry = np.array([[np.cos(LEAN), 0, np.sin(LEAN)], [0, 1, 0], [-np.sin(LEAN), 0, np.cos(LEAN)]])
Rv = np.array([[np.cos(VIEW), -np.sin(VIEW), 0], [np.sin(VIEW), np.cos(VIEW), 0], [0, 0, 1]])
R = Rv @ Ry @ R0
S = 1.0   # cm per cube edge


def P(p):
    q = R @ np.asarray(p, float)
    return q[0] * S, q[2] * S, q[1]          # screen x, screen y, depth (+ = away)


verts = {(i, j, k): P((i, j, k)) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
edges = [(a, b) for a in verts for b in verts if sum(abs(x - y) for x, y in zip(a, b)) == 1 and a < b]
out = [r"\begin{tikzpicture}[>=stealth, line cap=round, font=\scriptsize]"]
# ground and pivot
out.append(r"\fill[pattern=north east lines, pattern color=black!45] (-1.25,-0.12) rectangle (1.25,0);")
out.append(r"\draw[thick] (-1.25,0) -- (1.25,0);")
# visible faces, shaded by orientation (light from upper left), drawn first
for axis in range(3):
    for val in (0, 1):
        nrm = np.zeros(3); nrm[axis] = 1 if val else -1
        rn = R @ nrm
        if rn[1] >= 0:            # faces pointing away from the viewer
            continue
        corners = [k for k in verts if k[axis] == val]
        corners = sorted(corners, key=lambda k: np.arctan2(
            verts[k][1] - np.mean([verts[q][1] for q in corners]),
            verts[k][0] - np.mean([verts[q][0] for q in corners])))
        shade = int(6 + 26 * (1 - max(0.0, -rn[0] * 0.55 + rn[2] * 0.85)))
        out.append(rf"\fill[black!{shade}] " + " -- ".join(
            f"({verts[k][0]:.3f},{verts[k][1]:.3f})" for k in corners) + " -- cycle;")
far = max(verts, key=lambda k: verts[k][2])     # the 3 edges at the farthest vertex are hidden
for a, b in edges:
    (xa, ya, da), (xb, yb, db) = verts[a], verts[b]
    hidden = far in (a, b)
    style = "black!35, dashed" if hidden else "black, line width=0.6pt"
    out.append(rf"\draw[{style}] ({xa:.3f},{ya:.3f}) -- ({xb:.3f},{yb:.3f});")
# wheels: discs at the centres of the three faces through the pivot, drawn as
# projected circles (radius 0.38 edge) with their body axis e_i
for i, name in enumerate(["r_1", "r_2", "r_3"]):
    e = np.zeros(3); e[i] = 1
    u, w = [np.eye(3)[k] for k in range(3) if k != i]
    ctr = 0.5 * (u + w)
    pts = [P(ctr + 0.38 * (np.cos(t) * u + np.sin(t) * w)) for t in np.linspace(0, 2 * np.pi, 41)]
    if (R @ (-e))[1] > 0:          # outward normal of the face x_i = 0 points away
        continue
    sty = "blue!45!black, fill=blue!7, line width=0.5pt"
    out.append(rf"\draw[{sty}] " + " -- ".join(f"({x:.3f},{y:.3f})" for x, y, _ in pts) + " -- cycle;")
    x0, y0, _ = P((0, 0, 0)); x1, y1, _ = P(0.62 * e)
for i, name in enumerate(["r_1", "r_2", "r_3"]):
    e = np.zeros(3); e[i] = 1
    x1, y1, _ = P(0.55 * e); xl, yl, _ = P(0.72 * e)
    if i == 2:
        xl, yl = x1 + 0.17, y1 - 0.02
    out.append(rf"\draw[->, thick, blue!60!black] (0,0) -- ({x1:.3f},{y1:.3f});")
    out.append(rf"\node[blue!60!black, fill=white, inner sep=0.5pt] at ({xl:.3f},{yl:.3f}) {{$\boldsymbol {name}$}};")
# balancing axis n, vertical, centre of mass, gravity, lean
xt, yt, _ = P((1, 1, 1)); xc, yc, _ = P((0.5, 0.5, 0.5))
out.append(rf"\draw[densely dashed, red!70!black] (0,0) -- ({1.18*xt:.3f},{1.18*yt:.3f}) node[anchor=south west, inner sep=1pt] {{$\boldsymbol n$}};")
out.append(rf"\draw[densely dotted] (0,0) -- (0,{1.18*yt:.3f}) node[anchor=south, inner sep=1pt] {{$\boldsymbol e_z$}};")
out.append(rf"\draw[->] (0,{1.1*yt:.3f}) arc[start angle=90, delta angle={-np.rad2deg(LEAN):.2f}, radius={1.1*yt:.3f}];")
out.append(rf"\node[anchor=east, fill=white, inner sep=0.6pt] at (-0.04,{1.03*yt:.3f}) {{$\theta^\star$}};")
# with a center-of-mass offset the TRUE center of mass sits on the vertical
# through the pivot at equilibrium; the geometric centre (open) leans with n
hc = np.hypot(xc, yc)
out.append(rf"\draw[black!60] ({xc:.3f},{yc:.3f}) circle (1.3pt);")
out.append(rf"\fill (0,{hc:.3f}) circle (1.3pt);")
out.append(rf"\draw[black!60, line width=0.3pt] (0,{hc:.3f}) -- (-0.95,{hc+0.25:.3f}) node[anchor=east, inner sep=1pt, text=black] {{CoM}};")
out.append(rf"\draw[->, thick] (0,{hc:.3f}) -- (0,{hc-0.38:.3f}) node[anchor=west, pos=0.6, fill=white, inner sep=0.6pt] {{$M\boldsymbol g$}};")
out.append(r"\fill (0,0) circle (1.4pt);")
out.append(r"\node[anchor=north east, inner sep=2pt] at (-0.05,-0.12) {pivot};")
out.append(r"\end{tikzpicture}")
open("paper/fig_schematic.tex", "w").write("\n".join(out) + "\n")
print("wrote paper/fig_schematic.tex")
