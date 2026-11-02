"""Controller interface and the momentum-management augmentation.

The augmentation deserves a word, because it is the constructive half of the
paper.  Proposition 1 says the wheels cannot destroy angular momentum -- only
an *external* torque can, and on this system the only usable external torque
is gravity.  Gravity torque about the pivot is, to first order,

    tau_g = M g l (I - n n^T) theta_e,

i.e. it is produced by *tilting*.  So to bleed a stored wheel momentum
``h_w`` the controller must hold a small steady tilt

    theta_ref = -(k_h / (M g l)) (I - n n^T) h_w,

which drives exactly the torque that decelerates the wheel.  The projector
``(I - n n^T)`` is not cosmetic: it removes the component along the balancing
axis, which is precisely the component Proposition 1 forbids removing.

A classical LQR does this implicitly through its state weight on the wheel
speed.  Feedback linearization, sliding mode and backstepping, as they are
normally written for this plant, do not -- they drive the attitude error to
zero and let the wheel run away.  Every controller here therefore exists in
two variants, ``momentum_aware`` on and off, so the effect is measured rather
than asserted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """Common interface: ``reset()`` then ``__call__(x_meas, t) -> tau``."""

    name: str = "base"
    n_gains: int = 0

    @abstractmethod
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        ...

    def reset(self) -> None:
        return None

    # -- fair-tuning hooks ------------------------------------------------
    def get_gains(self) -> np.ndarray:
        raise NotImplementedError

    def set_gains(self, g: np.ndarray) -> None:
        raise NotImplementedError


class MomentumManager:
    """Turns stored wheel momentum into a small attitude reference.

    Parameters
    ----------
    k_h : float
        Unloading rate.  ``0`` disables the augmentation.
    tilt_max : float
        Hard cap on the commanded tilt [rad].  Without it the manager would
        happily command a lean the balancing controller cannot hold.
    Mgl : float
        Gravity stiffness of the plant, used to convert a desired unloading
        torque into a tilt.
    projector : ndarray or None
        ``I - n n^T`` for the 3-D case; ``None`` in the planar case.
    """

    def __init__(self, k_h: float, Mgl: float, tilt_max: float = np.deg2rad(3.0),
                 projector: np.ndarray | None = None):
        self.k_h = float(k_h)
        self.Mgl = float(Mgl)
        self.tilt_max = float(tilt_max)
        self.P = projector

    @property
    def enabled(self) -> bool:
        return self.k_h != 0.0

    def reference(self, h_wheel: np.ndarray | float) -> np.ndarray:
        """Attitude reference that unloads ``h_wheel`` through gravity."""
        h = np.atleast_1d(np.asarray(h_wheel, dtype=float))
        if not self.enabled:
            return np.zeros_like(h)
        if self.P is not None:
            h = self.P @ h
        ref = -(self.k_h / self.Mgl) * h
        n = np.linalg.norm(ref)
        if n > self.tilt_max:
            ref = ref * (self.tilt_max / n)
        return ref
