"""LQR synthesis on a plant that is stabilisable but **not** controllable.

Corner balancing with three reaction wheels has an uncontrollable mode, and it
is not a modeling artefact: by Proposition 1 the vertical component of the
total angular momentum cannot be changed by any internal torque.  In the
linearization this shows up as the left annihilator

    v = [0, (J n)^T, n^T],      v A = 0 (frictionless),   v B = 0,

with ``n`` the balancing axis in body coordinates.  The pair ``(A, B)`` has
rank ``n_x - 1``, and the missing mode is only *marginally* stable -- it decays
solely through pivot friction.  Feeding that straight to a Riccati solver
fails, which is exactly what a naive comparative study would hit.

The fix is the textbook one: Kalman decomposition, design on the controllable
subspace, and leave the invariant alone because no control law can touch it.
``lqr_stabilizable`` returns the gain together with the decomposition so the
paper can report the uncontrollable direction explicitly.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


def controllable_subspace(A: np.ndarray, B: np.ndarray, tol: float = 1e-9):
    """Orthonormal basis ``(Tc, Tu)`` of the controllable subspace and its complement."""
    n = A.shape[0]
    C = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    U, s, _ = np.linalg.svd(C)
    r = int(np.sum(s > tol * max(1.0, s[0])))
    return U[:, :r], U[:, r:], r


def lqr_stabilizable(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                     tol: float = 1e-9):
    """LQR gain designed on the controllable subspace.

    Returns
    -------
    K : ndarray
        Gain in the original coordinates; identically zero along the
        uncontrollable directions.
    info : dict
        ``rank``, the basis ``Tc``/``Tu``, the Riccati solution lifted back to
        the full space, and the uncontrollable eigenvalues.
    """
    Tc, Tu, r = controllable_subspace(A, B, tol)
    n = A.shape[0]

    if r == n:
        S = solve_continuous_are(A, B, Q, R)
        return np.linalg.solve(R, B.T @ S), {"rank": r, "Tc": Tc, "Tu": Tu,
                                             "S": S, "uncontrollable_eig": np.array([])}

    A11 = Tc.T @ A @ Tc
    B1 = Tc.T @ B
    Q11 = Tc.T @ Q @ Tc
    Q11 = 0.5 * (Q11 + Q11.T)
    S11 = solve_continuous_are(A11, B1, Q11, R)
    K1 = np.linalg.solve(R, B1.T @ S11)

    K = K1 @ Tc.T
    S = Tc @ S11 @ Tc.T
    A22 = Tu.T @ A @ Tu
    return K, {"rank": r, "Tc": Tc, "Tu": Tu, "S": S,
               "uncontrollable_eig": np.linalg.eigvals(A22)}


def momentum_annihilator(model) -> np.ndarray:
    """The invariant direction ``v`` of Proposition 1, in linearized coordinates.

    ``v z = n . (J w + h_w)`` is the vertical component of the total angular
    momentum, which no wheel torque can alter.
    """
    n = model.n_eq
    return np.concatenate([np.zeros(3), model.J @ n, n])
