"""Blanchard-Kahn solver for linear rational expectations models.

Solves models of the form:

    A · E_t[s_{t+1}] = B · s_t + C · ε_t

Where s_t = [x_t, y_t]' contains:
    - x_t: predetermined (backward-looking) variables
    - y_t: non-predetermined (forward-looking) variables

The solution takes the form:

    x_{t+1} = P · x_t + Q · ε_{t+1}
    y_t = R · x_t + S · ε_t

Reference: Blanchard & Kahn (1980), Klein (2000)
"""

from dataclasses import dataclass

import numpy as np
from scipy import linalg


class IndeterminacyError(Exception):
    """Model has multiple solutions (too few explosive eigenvalues)."""



class NoSolutionError(Exception):
    """Model has no stable solution (too many explosive eigenvalues)."""



@dataclass
class BKSolution:
    """Solution to a rational expectations model.

    Attributes:
        P: Transition matrix for predetermined variables (n_x × n_x)
        Q: Shock impact on predetermined variables (n_x × n_shocks)
        R: Policy function mapping predetermined to forward-looking (n_y × n_x)
        S: Shock impact on forward-looking variables (n_y × n_shocks)
        eigenvalues: Generalized eigenvalues of the system
        n_stable: Number of stable eigenvalues (|λ| < 1)
        n_unstable: Number of unstable eigenvalues (|λ| ≥ 1)

    """

    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    S: np.ndarray
    eigenvalues: np.ndarray
    n_stable: int
    n_unstable: int


def solve_blanchard_kahn(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    n_predetermined: int,
    tol: float = 1e-10,
) -> BKSolution:
    """Solve a linear rational expectations model using Blanchard-Kahn method.

    The model is:
        A · E_t[s_{t+1}] = B · s_t + C · ε_t

    where s_t has the first n_predetermined variables as predetermined (x_t)
    and the remaining as non-predetermined (y_t).

    Args:
        A: Coefficient matrix on E_t[s_{t+1}], shape (n, n)
        B: Coefficient matrix on s_t, shape (n, n)
        C: Coefficient matrix on shocks ε_t, shape (n, n_shocks)
        n_predetermined: Number of predetermined (backward-looking) variables
        tol: Tolerance for unit circle boundary

    Returns:
        BKSolution with policy function matrices

    Raises:
        IndeterminacyError: If too few explosive eigenvalues (multiple solutions)
        NoSolutionError: If too many explosive eigenvalues (no stable solution)

    """
    n = A.shape[0]
    n_forward = n - n_predetermined
    n_shocks = C.shape[1]

    # Generalized eigenvalue decomposition: A @ v = λ @ B @ v
    # We solve B⁻¹ @ A @ v = λ @ v, or equivalently use QZ decomposition
    # QZ is more stable: A = Q @ S @ Z', B = Q @ T @ Z'
    S_mat, T_mat, alpha, beta, Q_mat, Z_mat = linalg.ordqz(
        A, B, sort="ouc"  # Order: outside unit circle last (stable first)
    )

    # Compute eigenvalues (handling division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        eigenvalues = alpha / beta
        # Infinite eigenvalues (beta=0) are explosive
        eigenvalues = np.where(np.abs(beta) < tol, np.inf, eigenvalues)

    # Count stable (|λ| < 1) and unstable (|λ| ≥ 1) eigenvalues
    n_stable = np.sum(np.abs(eigenvalues) < 1 - tol)
    n_unstable = n - n_stable

    # Blanchard-Kahn condition: n_unstable must equal n_forward
    if n_unstable < n_forward:
        raise IndeterminacyError(
            f"Indeterminacy: {n_unstable} explosive eigenvalues but "
            f"{n_forward} forward-looking variables. Model has multiple solutions."
        )
    if n_unstable > n_forward:
        raise NoSolutionError(
            f"No solution: {n_unstable} explosive eigenvalues but only "
            f"{n_forward} forward-looking variables. No stable equilibrium."
        )

    # Partition Z matrix (note: ordqz gives Z', so we work with Z_mat.T)
    # Z' @ s = [s_stable, s_unstable]
    Z = Z_mat.conj().T  # Z matrix (n × n)

    # Partition: Z = [[Z11, Z12], [Z21, Z22]]
    # where Z11 is n_stable × n_predetermined, etc.
    Z11 = Z[:n_stable, :n_predetermined]
    Z12 = Z[:n_stable, n_predetermined:]
    Z21 = Z[n_stable:, :n_predetermined]
    Z22 = Z[n_stable:, n_predetermined:]

    # Check Z22 is invertible
    if np.abs(linalg.det(Z22)) < tol:
        raise NoSolutionError("Z22 matrix is singular - no unique solution.")

    # Solve for R: y_t = R @ x_t
    # From the requirement that unstable part = 0: Z21 @ x + Z22 @ y = 0
    # => y = -Z22⁻¹ @ Z21 @ x = R @ x
    Z22_inv = linalg.inv(Z22)
    R = -Z22_inv @ Z21

    # Partition S and T matrices similarly
    S11 = S_mat[:n_stable, :n_stable]
    T11 = T_mat[:n_stable, :n_stable]

    # Solve for P: x_{t+1} = P @ x_t
    # From the stable block dynamics
    # Need to transform back from QZ coordinates
    Z_x = Z11 + Z12 @ R  # Maps x_t to stable states

    if np.abs(linalg.det(Z_x)) < tol:
        raise NoSolutionError("Cannot solve for transition matrix P.")

    Z_x_inv = linalg.inv(Z_x)
    P = Z_x_inv @ linalg.solve(S11, T11) @ Z_x

    # Solve for shock impact matrices Q and S_shock
    # The full solution is:
    #   s_{t+1} = [[P], [R @ P]] @ x_t + shock_impact @ ε_{t+1}
    #
    # From: A @ s_{t+1} = B @ s_t + C @ ε_t
    # At t+1: shock_impact = A⁻¹ @ C (simplified for the stable part)

    # Compute shock impacts in the original coordinates
    # This requires solving the system for the shock response
    M = np.block([[np.eye(n_predetermined), np.zeros((n_predetermined, n_forward))],
                  [R, np.eye(n_forward)]])  # Maps [x, 0] + [0, y] structure

    # Impact on next period: need A⁻¹ @ C projected onto stable dynamics
    try:
        A_inv = linalg.inv(A)
        shock_full = A_inv @ C

        # Q: impact on predetermined variables
        Q = shock_full[:n_predetermined, :]

        # S: impact on forward-looking variables
        S = shock_full[n_predetermined:, :]
    except linalg.LinAlgError:
        # A is singular, use pseudoinverse
        A_pinv = linalg.pinv(A)
        shock_full = A_pinv @ C
        Q = shock_full[:n_predetermined, :]
        S = shock_full[n_predetermined:, :]

    return BKSolution(
        P=np.real(P),
        Q=np.real(Q),
        R=np.real(R),
        S=np.real(S),
        eigenvalues=eigenvalues,
        n_stable=n_stable,
        n_unstable=n_unstable,
    )


def check_determinacy(
    A: np.ndarray,
    B: np.ndarray,
    n_predetermined: int,
    tol: float = 1e-10,
) -> tuple[bool, int, int]:
    """Check if a model satisfies the Blanchard-Kahn conditions.

    Args:
        A: Coefficient matrix on E_t[s_{t+1}]
        B: Coefficient matrix on s_t
        n_predetermined: Number of predetermined variables
        tol: Tolerance for unit circle boundary

    Returns:
        Tuple of (is_determinate, n_stable, n_unstable)

    """
    n = A.shape[0]
    n_forward = n - n_predetermined

    # QZ decomposition just for eigenvalues
    _, _, alpha, beta, _, _ = linalg.ordqz(A, B)

    with np.errstate(divide="ignore", invalid="ignore"):
        eigenvalues = np.where(np.abs(beta) < tol, np.inf, alpha / beta)

    n_stable = np.sum(np.abs(eigenvalues) < 1 - tol)
    n_unstable = n - n_stable

    is_determinate = n_unstable == n_forward

    return is_determinate, n_stable, n_unstable


def compute_irfs(
    solution: BKSolution,
    shock_index: int,
    periods: int = 40,
    shock_size: float = 1.0,
) -> np.ndarray:
    """Compute impulse response functions for a given shock.

    Args:
        solution: BKSolution from solve_blanchard_kahn
        shock_index: Index of the shock (column of Q/S matrices)
        periods: Number of periods to compute
        shock_size: Size of the shock (in standard deviations)

    Returns:
        Array of shape (periods, n_vars) with IRFs for all variables
        First n_predetermined columns are x variables, rest are y variables

    """
    n_x = solution.P.shape[0]
    n_y = solution.R.shape[0]
    n_vars = n_x + n_y

    irfs = np.zeros((periods, n_vars))

    # Initial shock impact
    x = solution.Q[:, shock_index] * shock_size
    y = solution.S[:, shock_index] * shock_size + solution.R @ x

    irfs[0, :n_x] = x
    irfs[0, n_x:] = y

    # Propagate forward
    for t in range(1, periods):
        x = solution.P @ x
        y = solution.R @ x

        irfs[t, :n_x] = x
        irfs[t, n_x:] = y

    return irfs
