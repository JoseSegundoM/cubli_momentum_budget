"""Controller library for the Cubli benchmark."""

from .base import Controller, MomentumManager
from .ctrl3d import LQR3D, FBL3D, SMC3D, Backstepping3D, PID3D, make_controller_3d
from .mpc import MPC3D
from .ctrl2d import LQR2D, FBL2D, SMC2D, Backstepping2D, PID2D, make_controller_2d
from .swingup import EnergySwingUp, HybridSwingUpBalance

__all__ = [
    "Controller", "MomentumManager",
    "LQR3D", "MPC3D", "FBL3D", "SMC3D", "Backstepping3D", "PID3D", "make_controller_3d",
    "LQR2D", "FBL2D", "SMC2D", "Backstepping2D", "PID2D", "make_controller_2d",
    "EnergySwingUp", "HybridSwingUpBalance",
]
