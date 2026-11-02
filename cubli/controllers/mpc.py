"""Model predictive control with the momentum constraint written down explicitly.

This is the benchmark the rest of the study invites. The paper's argument is
that the binding actuator limit is the stored wheel momentum, so the natural
question a reviewer asks is: what does a controller that *knows* the constraint
do?  MPC answers it. Where the other designs meet the wheel-speed limit through
saturation --- that is, by being clipped after the fact --- MPC has

    |h_{w,i}(k)| <= J_a Omega_max,     i = 1,2,3,   k = 1..N

as a constraint of the optimisation, so it trades attitude accuracy for
momentum headroom on its own, without an added unloading loop.

Formulation
-----------
Linear MPC on the linearisation about the corner equilibrium, in the same
states the rest of the code uses, ``z = (theta_e, w, h_w)``:

    min  sum_k  z_k' Q z_k + u_k' R u_k  +  z_N' P z_N
    s.t. z_{k+1} = A_d z_k + B_d u_k
         |u_{k,i}| <= tau_max
         |h_{w,i}(k)| <= h_max
         z_0 = z(t)

with ``P`` the discrete algebraic Riccati solution on the controllable
subspace, so the terminal cost is the unconstrained optimal cost-to-go.  The
sparse problem is handed to OSQP and warm-started between solves; only the
bounds and the initial-state equality change from step to step, so the KKT
factorisation is reused.

The prediction model is linear while the plant is not.  That is deliberate: it
is the cheapest MPC a practitioner would actually deploy on this hardware, and
it keeps the comparison about the constraint rather than about who has the
better model.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from ..lqr_utils import controllable_subspace
from ..model3d import Cubli3D
from .base import Controller


class MPC3D(Controller):
    """Constrained MPC for corner balancing."""

    name = "mpc"
    n_gains = 4

    def __init__(self, model: Cubli3D, q_att: float = 3000.0, q_rate: float = 60.0,
                 q_mom: float = 200.0, r: float = 1.0, horizon: int = 25,
                 dt_mpc: float = 0.04, solve_every: float = 0.02,
                 tau_max: float | None = None, omega_max: float | None = None,
                 **kw):
        # MPC handles the momentum constraint by construction, so the
        # tilt-to-unload augmentation is not applied to it.
        kw.pop("k_h", None)
        self.m = model
        self.N = int(horizon)
        self.dt_mpc = float(dt_mpc)
        self.solve_every = float(solve_every)
        self.tau_max = float(model.p.tau_max if tau_max is None else tau_max)
        self.h_max = float(model.p.Ja
                           * (model.p.omega_max if omega_max is None else omega_max))
        self._w = np.array([q_att, q_rate, q_mom, r], dtype=float)
        self._build()
        self.reset()

    # ------------------------------------------------------------------
    @property
    def momentum_aware(self) -> bool:
        return True          # by constraint, not by weight

    def _build(self) -> None:
        import osqp

        q_att, q_rate, q_mom, r = self._w
        A, B = self.m.linearize()
        nz, nu = A.shape[0], B.shape[1]
        Ad, Bd, *_ = cont2discrete((A, B, np.eye(nz), np.zeros((nz, nu))), self.dt_mpc)

        Q = np.diag(np.concatenate([q_att * np.ones(3), q_rate * np.ones(3),
                                    q_mom * np.ones(3)]))
        R = r * np.eye(nu)

        # Terminal cost from the discrete Riccati equation on the controllable
        # subspace -- the pair is not controllable (the momentum invariant), so
        # the full-space DARE has no solution.
        Tc, _, rank = controllable_subspace(A, B)
        A11, B1 = Tc.T @ Ad @ Tc, Tc.T @ Bd
        P11 = solve_discrete_are(A11, B1, Tc.T @ Q @ Tc, R)
        P = Tc @ P11 @ Tc.T
        P = 0.5 * (P + P.T)

        N = self.N
        self.nz, self.nu = nz, nu
        nvar = nz * (N + 1) + nu * N

        # cost
        Pblk = sp.block_diag([sp.kron(sp.eye(N), sp.csc_matrix(Q)), sp.csc_matrix(P),
                              sp.kron(sp.eye(N), sp.csc_matrix(R))], format="csc")
        q_vec = np.zeros(nvar)

        # dynamics equalities
        Ax = sp.kron(sp.eye(N + 1), -sp.eye(nz)) + sp.kron(
            sp.eye(N + 1, k=-1), sp.csc_matrix(Ad))
        Bu = sp.kron(sp.vstack([sp.csc_matrix((1, N)), sp.eye(N)]), sp.csc_matrix(Bd))
        Aeq = sp.hstack([Ax, Bu])
        leq = np.zeros(nz * (N + 1))          # first block overwritten with -z0
        ueq = leq.copy()

        # input bounds and the momentum box on every predicted step
        Ain_u = sp.hstack([sp.csc_matrix((nu * N, nz * (N + 1))), sp.eye(nu * N)])
        sel_h = sp.csc_matrix(np.hstack([np.zeros((3, 6)), np.eye(3)]))   # h_w rows
        Ain_h = sp.hstack([sp.kron(sp.eye(N + 1), sel_h),
                           sp.csc_matrix((3 * (N + 1), nu * N))])
        Ain = sp.vstack([Ain_u, Ain_h], format="csc")
        lin = np.concatenate([-self.tau_max * np.ones(nu * N),
                              -self.h_max * np.ones(3 * (N + 1))])
        uin = -lin

        self.A_osqp = sp.vstack([Aeq, Ain], format="csc")
        self.l = np.concatenate([leq, lin])
        self.u = np.concatenate([ueq, uin])

        self.prob = osqp.OSQP()
        self.prob.setup(P=Pblk, q=q_vec, A=self.A_osqp, l=self.l, u=self.u,
                        verbose=False, warm_starting=True, polishing=False,
                        eps_abs=1e-4, eps_rel=1e-4, max_iter=400)
        self._u_slice = slice(nz * (N + 1), nz * (N + 1) + nu)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._t_last = -np.inf
        self._tau = np.zeros(3)
        self._fallbacks = 0

    def get_gains(self) -> np.ndarray:
        return self._w.copy()

    def set_gains(self, g: np.ndarray) -> None:
        self._w = np.abs(np.asarray(g, dtype=float))
        self._build()

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        # solve at its own rate; hold the previous torque in between, exactly as
        # a real implementation would
        if t - self._t_last < self.solve_every:
            return self._tau
        self._t_last = t

        z0 = self.m.error_state(x)
        nz = self.nz
        self.l[:nz] = self.u[:nz] = -z0
        self.prob.update(l=self.l, u=self.u)
        res = self.prob.solve()

        if res.info.status_val in (1, 2) and res.x is not None and np.all(np.isfinite(res.x)):
            self._tau = np.asarray(res.x[self._u_slice], dtype=float)
        else:
            # infeasible or failed: hold the last torque.  Counted, not hidden --
            # the failure rate is reported with the results.
            self._fallbacks += 1
        return self._tau
