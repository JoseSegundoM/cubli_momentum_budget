"""Physical parameters of the Cubli testbed.

One physical object, two equilibria:

* **Corner balancing (3-D)** -- the cube pivots about one vertex; all three
  orthogonal reaction wheels are used.
* **Edge balancing (2-D)** -- the cube pivots about one edge; only the wheel
  whose spin axis is parallel to that edge is used.

Both cases are derived from the *same* mass/geometry description so that the
2-D reduction is a genuine special case of the 3-D model and not a separate,
hand-tuned toy.  This is what makes the reduction test in ``tests/`` meaningful.

Body frame :math:`\\mathcal{B}`: origin at the geometric centre of the cube,
axes :math:`e_1,e_2,e_3` aligned with the cube edges.

Conventions
-----------
* ``J``      -- inertia tensor of the **whole** assembly about the pivot,
  wheels included as if rigidly attached (spacecraft convention).
* ``Ja``     -- axial inertia of one wheel about its own spin axis.
* ``Jb``     -- ``J - diag(Ja)``: the inertia that multiplies the body angular
  acceleration once the wheel momenta are written explicitly.
* ``r_c``    -- vector from the pivot to the assembly center of mass, in
  :math:`\\mathcal{B}`.

All SI units.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

G = 9.81  # m/s^2


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that ``_skew(a) @ b == np.cross(a, b)``."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _parallel_axis(mass: float, d: np.ndarray) -> np.ndarray:
    """Inertia contribution of a point mass ``mass`` at offset ``d``."""
    return mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


@dataclass(frozen=True)
class CubliParams:
    """Design-level description of the testbed.

    The nominal values describe a 15 cm class cube in the spirit of the ETH
    Cubli, sized around a Maxon EC-45-flat class BLDC motor.  They are a
    *declared design point*, not identified hardware parameters; every number
    used in the paper comes from this table.
    """

    # --- geometry / masses -------------------------------------------------
    L: float = 0.15          # cube edge [m]
    m_frame: float = 0.60    # structural frame (modeled as a hollow cube shell) [kg]
    m_wheel: float = 0.12    # rotating wheel assembly, one wheel [kg]
    m_motor: float = 0.15    # non-rotating motor/electronics lumped at the wheel seat [kg]
    r_wheel: float = 0.050   # wheel radius (uniform disc) [m]
    h_mount: float = 0.025   # inward offset of the wheel plane from the face [m]

    # --- dissipation -------------------------------------------------------
    com_offset: float = 0.0  # center-of-mass offset from the nominal diagonal [m]
    com_offset_dir: int = 0  # which body axis the offset is applied along
    b_pivot: float = 1.0e-3  # viscous friction at the pivot [N.m.s/rad]
    b_wheel: float = 1.0e-4  # viscous friction of one wheel [N.m.s/rad]
    tau_coulomb: float = 1.5e-3  # Coulomb friction at the wheel bearing [N.m]

    # --- actuator limits (the heart of the study) --------------------------
    tau_max: float = 0.50    # peak wheel torque [N.m]
    tau_cont: float = 0.15   # continuous (thermal) wheel torque [N.m]
    omega_max: float = 1000.0  # wheel speed limit [rad/s]  (~9550 rpm)
    tau_rate_max: float = 50.0  # torque slew limit [N.m/s]

    # --- digital implementation (this model is meant to be portable to hardware)
    f_s: float = 500.0       # controller sampling rate [Hz]
    n_delay: int = 0         # computation delay, in samples (see SimConfig)
    encoder_cpr: int = 4096  # encoder counts per revolution (quadrature)
    gyro_noise_std: float = 2.0e-3   # gyro white noise [rad/s]
    gyro_bias_std: float = 5.0e-4    # gyro bias, drawn once per run [rad/s]
    angle_noise_std: float = 5.0e-4  # attitude estimate noise [rad]

    # --- derived (filled in __post_init__) ---------------------------------
    M: float = field(init=False)
    Ja: float = field(init=False)
    corner: dict = field(init=False, repr=False)
    edge: dict = field(init=False, repr=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        L, rw = self.L, self.r_wheel
        m_s, m_w, m_m = self.m_frame, self.m_wheel, self.m_motor

        M = m_s + 3.0 * (m_w + m_m)
        object.__setattr__(self, "M", M)

        # Wheel: uniform disc.  Axial about its spin axis, transverse otherwise.
        Ja = 0.5 * m_w * rw**2
        Jt = 0.25 * m_w * rw**2
        object.__setattr__(self, "Ja", Ja)

        # Wheel/motor seats: centre of the three faces meeting at the vertex
        # opposite the balancing corner, pulled inward by ``h_mount``.
        d_seat = 0.5 * L - self.h_mount
        seats = [d_seat * np.eye(3)[i] for i in range(3)]

        # Center of mass of the assembly, in body frame (frame shell is centred).
        r_cm = sum((m_w + m_m) * p for p in seats) / M

        # --- inertia about the body centre ---------------------------------
        # Hollow cube shell (6 square plates): I = 5/18 m L^2, isotropic.
        J_cm = (5.0 / 18.0) * m_s * L**2 * np.eye(3)
        for i, p in enumerate(seats):
            axis = np.eye(3)[i]
            # wheel: axial Ja along its own axis, transverse Jt elsewhere
            J_cm = J_cm + Jt * np.eye(3) + (Ja - Jt) * np.outer(axis, axis)
            J_cm = J_cm + _parallel_axis(m_w, p)
            # motor stator: lumped point mass at the same seat
            J_cm = J_cm + _parallel_axis(m_m, p)

        # ---------------- corner balancing (3-D) ---------------------------
        p_corner = -0.5 * L * np.ones(3)
        r_c = r_cm - p_corner                       # pivot -> CoM
        J_corner = J_cm + _parallel_axis(M, r_c)    # whole assembly about pivot
        Jb_corner = J_corner - np.diag([Ja, Ja, Ja])

        corner = {
            "pivot": p_corner,
            "r_c": r_c,
            "J": J_corner,
            "Jb": Jb_corner,
            "Jw": np.diag([Ja, Ja, Ja]),
            # equilibrium: CoM straight above the pivot, i.e. the body diagonal
            # is vertical.  ``n_eq`` is that direction expressed in the body frame.
            "n_eq": r_c / np.linalg.norm(r_c),
            "l": float(np.linalg.norm(r_c)),
        }
        object.__setattr__(self, "corner", corner)

        # ---------------- edge balancing (2-D) -----------------------------
        # Rotation axis: the cube edge parallel to e1 at (y, z) = (-L/2, -L/2).
        axis = np.array([1.0, 0.0, 0.0])
        p_edge = np.array([0.0, -0.5 * L, -0.5 * L])
        r_e = r_cm - p_edge
        J_edge_full = J_cm + _parallel_axis(M, r_e)
        I_edge = float(axis @ J_edge_full @ axis)   # about the edge axis
        # perpendicular distance from the edge axis to the CoM
        r_perp = r_e - float(r_e @ axis) * axis
        l_edge = float(np.linalg.norm(r_perp))

        edge = {
            "pivot": p_edge,
            "axis": axis,
            "I": I_edge,          # total inertia about the edge, wheel included
            "Ib": I_edge - Ja,    # body-only inertia (wheel momentum written apart)
            "Ja": Ja,
            "l": l_edge,          # pivot -> CoM lever arm
            "m_gl": M * G * l_edge,
        }
        object.__setattr__(self, "edge", edge)

    # ------------------------------------------------------------------
    @property
    def dt_ctrl(self) -> float:
        """Controller period [s]."""
        return 1.0 / self.f_s

    @property
    def h_max_wheel(self) -> float:
        """Angular momentum a single wheel can store [N.m.s]."""
        return self.Ja * self.omega_max

    @property
    def encoder_lsb(self) -> float:
        """Angular resolution of the wheel encoder [rad]."""
        return 2.0 * np.pi / (4.0 * self.encoder_cpr)

    def perturbed(self, rng: np.random.Generator, rel: float = 0.20,
                  com_sigma: float = 1.5e-3, com_clip: float = 3.0) -> "CubliParams":
        """Draw a parameter sample for the Monte-Carlo robustness study.

        Five geometry/mass parameters and three dissipation parameters get
        independent multiplicative factors from ``U(1-rel, 1+rel)``.  The
        center-of-mass offset gets an additive draw from a truncated
        ``N(0, com_sigma)`` about its nominal value --- additive because an
        asymmetry has no meaningful nominal to scale, truncated at ``com_clip``
        sigma so the ensemble contains no absurd plants.

        Two things this does **not** do, both of which the paper must state:

        * The wheel radius is drawn as a *ratio* of the mounted face, and the
          mount depth scales with the edge, so every sampled cube is
          geometrically realisable.  Drawing ``r_wheel`` and ``L``
          independently puts the wheel through the frame in a third of the
          plants.
        * Actuator limits (``tau_max``, ``omega_max``) are held fixed, but the
          momentum budget ``h_max = Ja * omega_max`` is **not** fixed: ``Ja``
          follows the drawn wheel mass and radius, so the budget itself
          inherits roughly a factor-of-three spread across the ensemble.  That
          is physically right and has to be reported.
        """
        def f() -> float:
            return float(rng.uniform(1.0 - rel, 1.0 + rel))

        L_new = self.L * f()
        # keep the wheel inside the frame: sample the fill ratio, not the radius
        ratio = (self.r_wheel / (0.5 * self.L - self.h_mount)) * f()
        h_mount_new = self.h_mount * (L_new / self.L)
        dcom = float(np.clip(rng.normal(0.0, com_sigma),
                             -com_clip * com_sigma, com_clip * com_sigma))

        return replace(
            self,
            m_frame=self.m_frame * f(),
            m_wheel=self.m_wheel * f(),
            m_motor=self.m_motor * f(),
            L=L_new,
            h_mount=h_mount_new,
            r_wheel=min(ratio, 0.95) * (0.5 * L_new - h_mount_new),
            com_offset=self.com_offset + dcom,
            b_pivot=self.b_pivot * f(),
            b_wheel=self.b_wheel * f(),
            tau_coulomb=self.tau_coulomb * f(),
        )


NOMINAL = CubliParams()


if __name__ == "__main__":  # pragma: no cover - quick inspection
    p = NOMINAL
    np.set_printoptions(precision=6, suppress=True)
    print(f"total mass      M = {p.M:.4f} kg")
    print(f"wheel axial    Ja = {p.Ja:.6e} kg.m^2")
    print(f"wheel momentum h_max = {p.h_max_wheel:.4f} N.m.s per wheel")
    print("\n-- corner balancing --")
    print("J  =\n", p.corner["J"])
    print("Jb =\n", p.corner["Jb"])
    print("r_c =", p.corner["r_c"], " |r_c| =", p.corner["l"])
    print("n_eq =", p.corner["n_eq"])
    print("gravity torque @10 deg =", p.M * G * p.corner["l"] * np.sin(np.deg2rad(10)))
    print("\n-- edge balancing --")
    for k, v in p.edge.items():
        print(f"{k:6s} = {v}")
    print("time to saturate a wheel at 10 deg tilt [s] =",
          p.h_max_wheel / (p.M * G * p.corner["l"] * np.sin(np.deg2rad(10))))
