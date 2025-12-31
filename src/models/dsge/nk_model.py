"""New Keynesian DSGE Model with Wages and Interest Rate Smoothing.

A 4-equation NK model:

1. IS Curve (Euler equation):
   ŷ_t = E_t[ŷ_{t+1}] + σ·(E_t[π_{t+1}] - i_t) + ε_demand

2. Price Phillips Curve:
   π_t = β·E_t[π_{t+1}] + κ_p·ŷ_t + ε_supply

3. Wage Phillips Curve:
   π_w,t = β·E_t[π_w,{t+1}] + κ_w·ŷ_t + ε_wage

4. Taylor Rule with Smoothing:
   i_t = ρ_i·i_{t-1} + (1-ρ_i)·(φ_π·π_t + φ_y·ŷ_t) + ε_monetary

Solution approach:
- Interest rate smoothing makes i_t a state variable
- States: [ε_d, ε_s, ε_w, i] (4 predetermined) - monetary shock is iid
- Controls: [ŷ, π, π_w] (3 forward-looking)
- Solve using Blanchard-Kahn method
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg


class IndeterminacyError(Exception):
    """Model has multiple solutions (eigenvalues inside unit circle)."""

    pass


class NoSolutionError(Exception):
    """Model has no stable solution (eigenvalues outside unit circle)."""

    pass


@dataclass
class NKParameters:
    """Parameters for the NK DSGE model with wages and interest rate smoothing."""

    # Structural
    sigma: float = 1.0  # IES (IS curve slope)
    beta: float = 0.99  # Discount factor
    kappa_p: float = 0.1  # Price Phillips curve slope
    kappa_w: float = 0.1  # Wage Phillips curve slope

    # Okun's law: u_gap = -omega * y_gap (unemployment gap vs output gap)
    omega: float = 0.5  # Okun coefficient (typically 0.3-0.5)

    # Taylor rule
    phi_pi: float = 1.5  # Response to inflation (must be > 1)
    phi_y: float = 0.5  # Response to output gap
    rho_i: float = 0.8  # Interest rate smoothing

    # Shock persistence (demand, supply, wage have AR(1), monetary is iid)
    rho_demand: float = 0.8
    rho_supply: float = 0.5
    rho_wage: float = 0.5

    # Shock volatilities
    sigma_demand: float = 0.5
    sigma_supply: float = 0.3
    sigma_wage: float = 0.3
    sigma_monetary: float = 0.25

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "sigma": self.sigma,
            "beta": self.beta,
            "kappa_p": self.kappa_p,
            "kappa_w": self.kappa_w,
            "omega": self.omega,
            "phi_pi": self.phi_pi,
            "phi_y": self.phi_y,
            "rho_i": self.rho_i,
            "rho_demand": self.rho_demand,
            "rho_supply": self.rho_supply,
            "rho_wage": self.rho_wage,
            "sigma_demand": self.sigma_demand,
            "sigma_supply": self.sigma_supply,
            "sigma_wage": self.sigma_wage,
            "sigma_monetary": self.sigma_monetary,
        }


@dataclass
class NKSolution:
    """Solution to the NK model with interest rate smoothing.

    State vector: s_t = [ε_d, ε_s, ε_w, i]
    Control vector: y_t = [ŷ, π, π_w]

    Attributes:
        P: State transition matrix (4 × 4): s_{t+1} = P @ s_t + Q @ η_{t+1}
        Q: Shock impact on states (4 × 4)
        R: Policy function (3 × 4): y_t = R @ s_t
        eigenvalues: Eigenvalues for determinacy check
        Sigma: Shock volatility matrix (4 × 4 diagonal)

    """

    P: np.ndarray  # 4×4 state transition
    Q: np.ndarray  # 4×4 shock impact on states
    R: np.ndarray  # 3×4 policy function
    eigenvalues: np.ndarray
    Sigma: np.ndarray  # 4×4 shock volatilities


@dataclass
class NKModel:
    """4-Equation NK DSGE Model with Wages and Interest Rate Smoothing.

    With interest rate smoothing, i_t depends on i_{t-1}, making it a state.
    The model has 4 predetermined states and 3 forward-looking controls.

    """

    params: NKParameters = field(default_factory=NKParameters)

    # Dimensions
    n_states: int = 4  # ε_d, ε_s, ε_w, i
    n_shocks: int = 4  # η_d, η_s, η_w, η_m (innovations)
    n_forward: int = 3  # ŷ, π, π_w

    # Variable names
    state_names: list[str] = field(
        default_factory=lambda: ["eps_demand", "eps_supply", "eps_wage", "i"]
    )
    shock_names: list[str] = field(
        default_factory=lambda: ["eta_demand", "eta_supply", "eta_wage", "eta_monetary"]
    )
    forward_names: list[str] = field(default_factory=lambda: ["y", "pi", "pi_w"])

    def _build_system_matrices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the full system matrices for Blanchard-Kahn.

        The system: A @ E_t[z_{t+1}] = B @ z_t + C @ η_{t+1}
        where z = [ε_d, ε_s, ε_w, i, ŷ, π, π_w]' (7 variables)

        States: [ε_d, ε_s, ε_w, i] (4 predetermined)
        Controls: [ŷ, π, π_w] (3 forward-looking)

        Equations:
        1-3. Shock AR(1) dynamics for demand, supply, wage
        4. Interest rate smoothing (led by 1 period)
        5. IS curve
        6. Price Phillips curve
        7. Wage Phillips curve

        Returns:
            (A, B, C) matrices

        """
        p = self.params
        n = self.n_states + self.n_forward  # 7 total variables

        A = np.zeros((n, n))
        B = np.zeros((n, n))
        C = np.zeros((n, self.n_shocks))

        # Variable indices
        # States: eps_d=0, eps_s=1, eps_w=2, i=3
        # Controls: y=4, pi=5, pi_w=6

        # 1. ε_d dynamics: E[ε_d'] = ρ_d·ε_d
        A[0, 0] = 1.0
        B[0, 0] = p.rho_demand
        C[0, 0] = p.sigma_demand

        # 2. ε_s dynamics: E[ε_s'] = ρ_s·ε_s
        A[1, 1] = 1.0
        B[1, 1] = p.rho_supply
        C[1, 1] = p.sigma_supply

        # 3. ε_w dynamics: E[ε_w'] = ρ_w·ε_w
        A[2, 2] = 1.0
        B[2, 2] = p.rho_wage
        C[2, 2] = p.sigma_wage

        # 4. Lagged interest rate dynamics:
        # i_t = ρ_i·i_{t-1} + (1-ρ_i)·(φ_π·π_t + φ_y·ŷ_t) + ε_m,t
        # Define ĩ_t = i_{t-1} (lagged rate as state), so:
        # ĩ_{t+1} = i_t = ρ_i·ĩ_t + (1-ρ_i)·(φ_π·π_t + φ_y·ŷ_t) + ε_m,t
        # Note: RHS has current controls (π_t, ŷ_t), not expectations
        A[3, 3] = 1.0  # E[ĩ'] = ĩ_{t+1}
        B[3, 3] = p.rho_i  # ĩ
        B[3, 4] = (1 - p.rho_i) * p.phi_y  # ŷ
        B[3, 5] = (1 - p.rho_i) * p.phi_pi  # π
        C[3, 3] = p.sigma_monetary  # monetary shock is iid

        # 5. IS curve: ŷ = E[ŷ'] + σ·(E[π'] - i) + ε_d
        # where i = ρ_i·ĩ + (1-ρ_i)·(φ_π·π + φ_y·ŷ) + ε_m
        # Substituting and rearranging (ignoring ε_m for A/B):
        # E[ŷ'] + σ·E[π'] = (1 + σ(1-ρ_i)φ_y)ŷ + σ(1-ρ_i)φ_π·π + σρ_i·ĩ - ε_d
        A[4, 4] = 1.0  # E[ŷ']
        A[4, 5] = p.sigma  # E[π']
        B[4, 4] = 1.0 + p.sigma * (1 - p.rho_i) * p.phi_y  # ŷ
        B[4, 5] = p.sigma * (1 - p.rho_i) * p.phi_pi  # π
        B[4, 3] = p.sigma * p.rho_i  # ĩ (lagged interest rate)
        B[4, 0] = -1.0  # ε_d

        # 6. Price Phillips: π = β·E[π'] + κ_p·ŷ + ε_s
        # Rearranging: β·E[π'] = π - κ_p·ŷ - ε_s
        A[5, 5] = p.beta  # E[π']
        B[5, 5] = 1.0  # π
        B[5, 4] = -p.kappa_p  # ŷ
        B[5, 1] = -1.0  # ε_s

        # 7. Wage Phillips: π_w = β·E[π_w'] + κ_w·ŷ + ε_w
        # Rearranging: β·E[π_w'] = π_w - κ_w·ŷ - ε_w
        A[6, 6] = p.beta  # E[π_w']
        B[6, 6] = 1.0  # π_w
        B[6, 4] = -p.kappa_w  # ŷ
        B[6, 2] = -1.0  # ε_w

        return A, B, C

    def check_determinacy(self) -> tuple[bool, np.ndarray]:
        """Check Blanchard-Kahn conditions.

        For 3 forward-looking variables, we need exactly 3 eigenvalues
        of the transition matrix M = A^{-1}B outside the unit circle.

        Returns:
            (is_determinate, eigenvalues of transition matrix)

        """
        A, B, _ = self._build_system_matrices()

        # For M = A^{-1}B, eigenvalues μ satisfy B @ v = μ @ A @ v
        # Use ordqz(B, A) to get eigenvalues α/β where B @ v = (α/β) @ A @ v
        # Sort: stable (|μ| < 1) first, unstable (|μ| > 1) last
        _, _, alpha, beta_eig, _, _ = linalg.ordqz(B, A, sort="iuc")

        # Compute eigenvalues of transition matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            eigenvalues = np.where(np.abs(beta_eig) < 1e-10, np.inf, alpha / beta_eig)

        # Count unstable (outside unit circle)
        n_unstable = np.sum(np.abs(eigenvalues) > 1.0 + 1e-10)

        is_determinate = n_unstable == self.n_forward

        return is_determinate, eigenvalues

    def solve(self) -> NKSolution:
        """Solve the model using Blanchard-Kahn method.

        Returns:
            NKSolution with state transition and policy function matrices

        """
        A, B, C = self._build_system_matrices()
        p = self.params
        n = self.n_states + self.n_forward

        # Check determinacy
        is_det, eigenvalues = self.check_determinacy()
        if not is_det:
            n_unstable = np.sum(np.abs(eigenvalues) > 1.0 + 1e-10)
            if n_unstable < self.n_forward:
                raise IndeterminacyError(
                    f"Indeterminacy: {n_unstable} eigenvalues outside unit circle, "
                    f"need {self.n_forward}. Model has multiple solutions."
                )
            else:
                raise NoSolutionError(
                    f"No solution: {n_unstable} eigenvalues outside unit circle, "
                    f"need exactly {self.n_forward}."
                )

        # QZ decomposition for B @ v = μ @ A @ v (eigenvalues of M = A^{-1}B)
        # Stable eigenvalues (|μ| < 1) first, unstable (|μ| > 1) last
        S, T, alpha, beta_eig, Q, Z = linalg.ordqz(B, A, sort="iuc")

        # Compute eigenvalues for reference
        with np.errstate(divide="ignore", invalid="ignore"):
            eigenvalues = np.where(np.abs(beta_eig) < 1e-10, np.inf, alpha / beta_eig)

        n_stable = n - self.n_forward  # 4 stable eigenvalues

        # Partition Z matrix
        Z_full = Z.conj().T
        Z11 = Z_full[:n_stable, :self.n_states]  # 4×4
        Z12 = Z_full[:n_stable, self.n_states:]  # 4×3
        Z21 = Z_full[n_stable:, :self.n_states]  # 3×4
        Z22 = Z_full[n_stable:, self.n_states:]  # 3×3

        # Check Z22 invertibility
        if np.abs(linalg.det(Z22)) < 1e-10:
            raise NoSolutionError("Z22 matrix is singular - no unique solution.")

        # Policy function R: controls = R @ states
        # From BK: y_t = -Z22^{-1} @ Z21 @ s_t
        Z22_inv = linalg.inv(Z22)
        R = -Z22_inv @ Z21  # 3×4

        # State transition P
        S11 = S[:n_stable, :n_stable]
        T11 = T[:n_stable, :n_stable]

        # Z_s maps states to stable block
        Z_s = Z11 + Z12 @ R  # 4×4

        if np.abs(linalg.det(Z_s)) < 1e-10:
            raise NoSolutionError("Cannot solve for state transition matrix P.")

        Z_s_inv = linalg.inv(Z_s)
        # Transition matrix: eigenvalues are S_ii/T_ii = μ (stable)
        # So we need T11^{-1} @ S11, which is linalg.solve(T11, S11)
        P = np.real(Z_s_inv @ linalg.solve(T11, S11) @ Z_s)  # 4×4

        # Shock impact matrix Q_mat
        # From A @ z' = B @ z + C @ η, the shock impacts come from C
        try:
            A_inv = linalg.inv(A)
            shock_full = A_inv @ C  # n×4
        except linalg.LinAlgError:
            A_pinv = linalg.pinv(A)
            shock_full = A_pinv @ C

        # Q_mat: shock impact on states (first 4 components)
        Q_mat = np.real(shock_full[:self.n_states, :])  # 4×4

        # Shock volatilities
        Sigma = np.diag([p.sigma_demand, p.sigma_supply, p.sigma_wage, p.sigma_monetary])

        return NKSolution(
            P=P,
            Q=Q_mat,
            R=np.real(R),
            eigenvalues=eigenvalues,
            Sigma=Sigma,
        )

    def state_space_matrices(
        self,
        solution: NKSolution | None = None,
        n_observables: int = 2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Get state-space matrices for Kalman filter.

        State: s_t = [ε_d, ε_s, ε_w, i] (4 states)
        Transition: s_{t+1} = P @ s_t + Q @ η_{t+1}
        Observation: y_t = Z @ s_t + measurement_error

        Args:
            solution: Pre-computed solution (optional)
            n_observables: Number of observables
                2: [ŷ, π]
                3: [ŷ, π, i]
                4: [ŷ, π, i, π_w]
                5: [ŷ, π, i, π_w, u_gap] (adds unemployment gap via Okun's law)

        Returns:
            (T, R, Z, Q, H) for Kalman filter
            H is measurement error covariance (None if no measurement error)

        """
        if solution is None:
            solution = self.solve()

        T = solution.P  # 4×4 state transition
        R = solution.Q  # 4×4 shock impact on states
        Q_cov = np.eye(self.n_shocks)  # Standard normal innovations
        H = None  # Measurement error covariance

        # Control variables from policy function
        # y_t = [ŷ, π, π_w] = R_policy @ s_t
        R_policy = solution.R  # 3×4

        if n_observables == 2:
            # Observables: [ŷ, π]
            Z = R_policy[:2, :]  # 2×4
        elif n_observables == 3:
            # Observables: [ŷ, π, i]
            # i is the 4th state (index 3)
            i_row = np.zeros((1, self.n_states))
            i_row[0, 3] = 1.0
            Z = np.vstack([R_policy[:2, :], i_row])  # 3×4
        elif n_observables == 4:
            # Observables: [ŷ, π, i, π_w]
            i_row = np.zeros((1, self.n_states))
            i_row[0, 3] = 1.0
            Z = np.vstack([R_policy[:2, :], i_row, R_policy[2:3, :]])  # 4×4
        elif n_observables == 5:
            # Observables: [ŷ, π, i, π_w, u_gap]
            # u_gap = -omega * ŷ + measurement_error (Okun's law with error)
            # ŷ is in R_policy[0, :], so u_gap row = -omega * R_policy[0, :]
            i_row = np.zeros((1, self.n_states))
            i_row[0, 3] = 1.0
            u_gap_row = -self.params.omega * R_policy[0:1, :]  # 1×4
            Z = np.vstack([R_policy[:2, :], i_row, R_policy[2:3, :], u_gap_row])  # 5×4

            # Add measurement error on u_gap to avoid singular covariance
            # (Okun's law is approximate, not exact)
            H = np.zeros((5, 5))
            H[4, 4] = 0.1**2  # Small measurement error on u_gap (0.1 pp std dev)
        else:
            raise ValueError(f"n_observables must be 2, 3, 4, or 5, got {n_observables}")

        return T, R, Z, Q_cov, H

    def compute_impulse_responses(
        self,
        shock_name: str,
        periods: int = 40,
        solution: NKSolution | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute impulse response functions for a given shock.

        Args:
            shock_name: One of 'demand', 'supply', 'wage', 'monetary'
            periods: Number of periods
            solution: Pre-computed solution (optional)

        Returns:
            Dict mapping variable names to IRF arrays

        """
        if solution is None:
            solution = self.solve()

        shock_map = {"demand": 0, "supply": 1, "wage": 2, "monetary": 3}
        shock_idx = shock_map[shock_name]

        # Initial shock (1 standard deviation)
        eta = np.zeros(self.n_shocks)
        eta[shock_idx] = solution.Sigma[shock_idx, shock_idx]

        # Initial state impact
        state = solution.Q @ eta

        # Storage
        irfs_state = np.zeros((periods, self.n_states))
        irfs_control = np.zeros((periods, self.n_forward))

        for t in range(periods):
            irfs_state[t, :] = state
            irfs_control[t, :] = solution.R @ state

            # Propagate state forward
            state = solution.P @ state

        return {
            "eps_demand": irfs_state[:, 0],
            "eps_supply": irfs_state[:, 1],
            "eps_wage": irfs_state[:, 2],
            "i": irfs_state[:, 3],
            "y": irfs_control[:, 0],
            "pi": irfs_control[:, 1],
            "pi_w": irfs_control[:, 2],
        }


def simulate_model(
    params: NKParameters,
    n_periods: int = 200,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate the NK model.

    Args:
        params: Model parameters
        n_periods: Number of periods
        seed: Random seed

    Returns:
        Dict with states, shocks, and observables

    """
    np.random.seed(seed)

    model = NKModel(params=params)
    solution = model.solve()

    # Simulate state
    state = np.zeros(model.n_states)
    states = np.zeros((n_periods, model.n_states))
    innovations = np.random.randn(n_periods, model.n_shocks)

    for t in range(n_periods):
        eta = innovations[t, :]
        state = solution.P @ state + solution.Q @ eta
        states[t, :] = state

    # Compute control variables
    controls = states @ solution.R.T

    return {
        "states": states,
        "innovations": innovations,
        "controls": controls,
        "state_names": model.state_names,
        "shock_names": model.shock_names,
        "control_names": ["output_gap", "inflation", "wage_inflation"],
    }


def test_model():
    """Test that the model solves correctly."""
    params = NKParameters()
    model = NKModel(params=params)

    print("=" * 60)
    print("NK MODEL WITH INTEREST RATE SMOOTHING")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Interest rate smoothing (ρ_i): {params.rho_i}")
    print(f"  Taylor rule: φ_π = {params.phi_pi}, φ_y = {params.phi_y}")

    # Check determinacy
    is_det, eigenvalues = model.check_determinacy()
    print(f"\nDeterminacy: {'PASS' if is_det else 'FAIL'}")
    print(f"  Eigenvalues (magnitude): {np.round(np.abs(eigenvalues), 4)}")
    n_unstable = np.sum(np.abs(eigenvalues) > 1.0 + 1e-10)
    print(f"  Unstable: {n_unstable}, Forward-looking: {model.n_forward}")

    if is_det:
        solution = model.solve()
        print(f"\nState transition matrix P (4×4):")
        print(f"  Eigenvalues: {np.round(np.abs(linalg.eigvals(solution.P)), 4)}")

        print(f"\nPolicy function R (maps states to [ŷ, π, π_w]):")
        print(f"             ε_demand   ε_supply   ε_wage        i")
        print(f"  ŷ:       {solution.R[0, 0]:9.4f}  {solution.R[0, 1]:9.4f}  {solution.R[0, 2]:9.4f}  {solution.R[0, 3]:9.4f}")
        print(f"  π:       {solution.R[1, 0]:9.4f}  {solution.R[1, 1]:9.4f}  {solution.R[1, 2]:9.4f}  {solution.R[1, 3]:9.4f}")
        print(f"  π_w:     {solution.R[2, 0]:9.4f}  {solution.R[2, 1]:9.4f}  {solution.R[2, 2]:9.4f}  {solution.R[2, 3]:9.4f}")

        # Test IRFs
        for shock in ["demand", "monetary"]:
            print(f"\n--- IRF to {shock} shock ---")
            irfs = model.compute_impulse_responses(shock, periods=10)
            print("Quarter |  Output |  Price π |  Wage π  |   i")
            for t in range(6):
                print(f"   {t:2d}    | {irfs['y'][t]:7.3f} | {irfs['pi'][t]:8.3f} | {irfs['pi_w'][t]:8.3f} | {irfs['i'][t]:7.3f}")

    return is_det


if __name__ == "__main__":
    test_model()
