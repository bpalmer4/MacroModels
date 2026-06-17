"""Financial-accelerator NK DSGE: the PROPER two-r* model (path 1).

This replaces the reduced-form wedge of nk_twostar_model.py (the −σ_k·ω graft)
with an ENDOGENOUS external-finance premium (EFP). Two natural rates now emerge
from optimisation rather than being labelled by hand:

  safe rate   = R − E[π']         (household consumption Euler)
  return on K = r^k               (firms' capital, via Tobin's Q)
  wedge       = r^k − safe rate   = χ·leverage + ω   (the EFP, equation R8)

The wedge is the great divergence as an equilibrium object: it widens when firm
leverage (q + k − n) rises or a financial shock ω hits, and it transmits to the
real economy by depressing investment through Q. A Bernanke-Gertler-Gilchrist
financial accelerator, kept lean (mc = ξ·y closure, no separate labour block).

Log-linearised system (deviations from steady state), with the static
substitutions inv = k + q/ψ,  y = c_y·c + i_y·inv,  mc = ξ·y,
mpk = (1+ξ)y − k  folded in:

  Euler (safe):   c = E c' − σ(R − E π') + ε_d
  Capital:        k' = (1−δ)k + δ·inv = k + (δ/ψ)q
  Q / finance:    a1·E mpk' + β(1−δ)E q' − q − (R − E π') = χ(q + k' − n') + ω
  Net worth:      n' = κ_n·n + (1−κ_n)·levK·(r^k − (R₋₁ − π))
  Phillips:       π = β E π' + κ_p·ξ·y + ε_s
  Taylor:         R = ρ_i·R₋₁ + (1−ρ_i)(φ_π·π + φ_y·y) + ε_m
  r^k (realised): a1·mpk + β(1−δ)q − q₋₁      (a1 = 1−β(1−δ))

Canonical form  A·E_t[z_{t+1}] = B·z_t + C·η_{t+1}, solved by Blanchard-Kahn
(same QZ machinery as nk_model.py).

z = [ε_d, ε_s, ω, R₋₁, k, n, q₋₁,  c, q, π]   (7 states, 3 forward)
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from src.models.dsge.nk_model import IndeterminacyError, NoSolutionError


@dataclass
class FANKParameters:
    """Parameters / calibration for the financial-accelerator NK DSGE."""

    # Preferences / technology
    beta: float = 0.99      # discount factor
    delta: float = 0.025    # depreciation
    sigma: float = 1.0      # IES (consumption Euler)
    xi: float = 1.0         # real marginal cost elasticity to output (leaner mc = ξ·y)
    psi: float = 4.0        # capital adjustment cost curvature (q = ψ(inv − k))
    alpha: float = 0.33     # capital share (labour-block production: y = αk + (1−α)l)
    phi_l: float = 1.0      # inverse Frisch elasticity (labour-block supply: w = c/σ + φ_l·l)

    # Great ratios (steady state)
    c_y: float = 0.8        # consumption share of output
    i_y: float = 0.2        # investment share of output
    lev_k: float = 2.0      # steady-state leverage K/N

    # Financial accelerator
    chi: float = 0.05       # EFP elasticity to leverage (the wedge slope)
    kappa_n: float = 0.95   # net-worth survival rate

    # Nominal
    kappa_p: float = 0.10   # Phillips slope (on marginal cost)
    kappa_w: float = 0.05   # wage Phillips slope (Calvo wages; Tier-2 model only)
    rho_wage: float = 0.5   # wage-markup shock persistence (Tier-2 model only)
    sigma_wage: float = 0.30  # wage-markup shock volatility (Tier-2 model only)

    # Taylor rule
    phi_pi: float = 1.5
    phi_y: float = 0.125
    rho_i: float = 0.8

    # Exogenous shock persistence
    rho_demand: float = 0.8
    rho_supply: float = 0.5
    rho_omega: float = 0.9   # financial-spread shock

    # Shock volatilities
    sigma_demand: float = 0.5
    sigma_supply: float = 0.3
    sigma_omega: float = 0.3
    sigma_monetary: float = 0.25

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class FANKSolution:
    P: np.ndarray  # 7×7 state transition
    Q: np.ndarray  # 7×4 shock impact on states
    R: np.ndarray  # 3×7 policy function (jumps = R·states)
    eigenvalues: np.ndarray
    Sigma: np.ndarray


@dataclass
class FANKModel:
    """Financial-accelerator NK DSGE (7 states, 3 forward-looking controls)."""

    params: FANKParameters = field(default_factory=FANKParameters)

    n_states: int = 7   # eps_d, eps_s, omega, Rlag, k, n, qlag
    n_shocks: int = 4    # eta_d, eta_s, eta_omega, eta_m
    n_forward: int = 3   # c, q, pi

    # Tier-1 labour block: replace mc = ξ·y with structural mc = w + l − y (flexible
    # wages → l, w are static functions of (c, k, q), so no new states).
    labour_block: bool = False

    state_names: list[str] = field(
        default_factory=lambda: ["eps_demand", "eps_supply", "omega", "R_lag", "k", "n", "q_lag"]
    )
    forward_names: list[str] = field(default_factory=lambda: ["c", "q", "pi"])

    # z-vector indices
    I_EPSD, I_EPSS, I_OMEGA, I_RLAG, I_K, I_N, I_QLAG, I_C, I_Q, I_PI = range(10)

    def _static_coeffs(self) -> dict:
        """Coefficients (on c, k, q) for output y, marginal cost mc, and the marginal
        product of capital mpk. Leaner closure mc = ξ·y, unless `labour_block`, where
        a flexible-wage labour block gives mc = w + l − y (l, w static, no new states):
            l = (y − α·k)/(1−α),  w = c/σ + φ_l·l,  mc = w + l − y.
        """
        p = self.params
        Y_C, Y_K, Y_Q = p.c_y, p.i_y, p.i_y / p.psi
        if self.labour_block:
            a = 1.0 - p.alpha
            l_C, l_K, l_Q = Y_C / a, Y_K / a - p.alpha / a, Y_Q / a
            w_C = 1.0 / p.sigma + p.phi_l * l_C
            w_K = p.phi_l * l_K
            w_Q = p.phi_l * l_Q
            mc_C, mc_K, mc_Q = w_C + l_C - Y_C, w_K + l_K - Y_K, w_Q + l_Q - Y_Q
        else:
            mc_C, mc_K, mc_Q = p.xi * Y_C, p.xi * Y_K, p.xi * Y_Q
        mpk_C, mpk_K, mpk_Q = mc_C + Y_C, mc_K + Y_K - 1.0, mc_Q + Y_Q  # mpk = mc + y − k
        return {"Y": (Y_C, Y_K, Y_Q), "mc": (mc_C, mc_K, mc_Q), "mpk": (mpk_C, mpk_K, mpk_Q)}

    def _build_system_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.params
        n = self.n_states + self.n_forward  # 10
        A = np.zeros((n, n)); B = np.zeros((n, n)); C = np.zeros((n, self.n_shocks))

        (EPSD, EPSS, OMEGA, RLAG, K, N, QLAG, Cc, Q, PI) = range(10)
        a1 = 1.0 - p.beta * (1.0 - p.delta)            # weight on mpk in r^k
        bd = p.beta * (1.0 - p.delta)                  # weight on capital gain in r^k
        cy, iy, psi = p.c_y, p.i_y, p.psi
        sc = self._static_coeffs()
        Y_C, Y_K, Y_Q = sc["Y"]          # y = cy·c + iy·k + (iy/psi)·q
        MPK_C, MPK_K, MPK_Q = sc["mpk"]
        MC_C, MC_K, MC_Q = sc["mc"]
        # realised r^k = a1·mpk + bd·q − qlag
        RK_C, RK_K = a1 * MPK_C, a1 * MPK_K
        RK_Q = a1 * MPK_Q + bd
        RK_QLAG = -1.0

        # R0-R2: exogenous AR(1) shocks
        A[EPSD, EPSD] = 1.0; B[EPSD, EPSD] = p.rho_demand; C[EPSD, 0] = p.sigma_demand
        A[EPSS, EPSS] = 1.0; B[EPSS, EPSS] = p.rho_supply; C[EPSS, 1] = p.sigma_supply
        A[OMEGA, OMEGA] = 1.0; B[OMEGA, OMEGA] = p.rho_omega; C[OMEGA, 2] = p.sigma_omega

        # R3: Taylor rule defines R_lag' = R_t = ρ_i·R₋₁ + (1−ρ_i)(φ_π·π + φ_y·y) + ε_m
        A[RLAG, RLAG] = 1.0
        B[RLAG, RLAG] = p.rho_i
        B[RLAG, PI] = (1 - p.rho_i) * p.phi_pi
        B[RLAG, Cc] = (1 - p.rho_i) * p.phi_y * Y_C
        B[RLAG, K] = (1 - p.rho_i) * p.phi_y * Y_K
        B[RLAG, Q] = (1 - p.rho_i) * p.phi_y * Y_Q
        C[RLAG, 3] = p.sigma_monetary

        # R4: capital law  k' = k + (δ/ψ)·q
        A[K, K] = 1.0
        B[K, K] = 1.0
        B[K, Q] = p.delta / psi

        # R5: net worth  n' = κ_n·n + (1−κ_n)·levK·(r^k − R₋₁ + π)
        g = (1 - p.kappa_n) * p.lev_k
        A[N, N] = 1.0
        B[N, N] = p.kappa_n
        B[N, Cc] = g * RK_C
        B[N, K] = g * RK_K
        B[N, Q] = g * RK_Q
        B[N, QLAG] = g * RK_QLAG
        B[N, RLAG] = g * (-1.0)
        B[N, PI] = g * (1.0)

        # R6: qlag' = q
        A[QLAG, QLAG] = 1.0
        B[QLAG, Q] = 1.0

        # R7: consumption Euler  E c' + σ E π' = c + σ·R(systematic) − ε_d
        A[Cc, Cc] = 1.0
        A[Cc, PI] = p.sigma
        B[Cc, Cc] = 1.0 + p.sigma * (1 - p.rho_i) * p.phi_y * Y_C
        B[Cc, RLAG] = p.sigma * p.rho_i
        B[Cc, PI] = p.sigma * (1 - p.rho_i) * p.phi_pi
        B[Cc, K] = p.sigma * (1 - p.rho_i) * p.phi_y * Y_K
        B[Cc, Q] = p.sigma * (1 - p.rho_i) * p.phi_y * Y_Q
        B[Cc, EPSD] = -1.0

        # R8: Q / finance Euler (return on capital = safe rate + EFP)
        #   LHS (E[z']): a1·E mpk' + bd·E q' + E π' − χ·k' + χ·n'
        #   RHS (z):     (1+χ)q + R(systematic) + ω
        A[Q, Cc] = a1 * MPK_C
        A[Q, Q] = a1 * MPK_Q + bd
        A[Q, PI] = 1.0
        A[Q, K] = a1 * MPK_K - p.chi
        A[Q, N] = p.chi
        B[Q, Q] = (1.0 + p.chi) + (1 - p.rho_i) * p.phi_y * Y_Q
        B[Q, RLAG] = p.rho_i
        B[Q, PI] = (1 - p.rho_i) * p.phi_pi
        B[Q, Cc] = (1 - p.rho_i) * p.phi_y * Y_C
        B[Q, K] = (1 - p.rho_i) * p.phi_y * Y_K
        B[Q, OMEGA] = 1.0

        # R9: Phillips  β E π' = π − κ_p·mc − ε_s   (mc = ξ·y, or structural if labour_block)
        A[PI, PI] = p.beta
        B[PI, PI] = 1.0
        B[PI, Cc] = -p.kappa_p * MC_C
        B[PI, K] = -p.kappa_p * MC_K
        B[PI, Q] = -p.kappa_p * MC_Q
        B[PI, EPSS] = -1.0

        return A, B, C

    def check_determinacy(self) -> tuple[bool, np.ndarray]:
        A, B, _ = self._build_system_matrices()
        _, _, alpha, beta_eig, _, _ = linalg.ordqz(B, A, sort="iuc")
        with np.errstate(divide="ignore", invalid="ignore"):
            eig = np.where(np.abs(beta_eig) < 1e-10, np.inf, alpha / beta_eig)
        n_unstable = int(np.sum(np.abs(eig) > 1.0 + 1e-10))
        return n_unstable == self.n_forward, eig

    def solve(self) -> FANKSolution:
        A, B, C = self._build_system_matrices()
        p = self.params
        n = self.n_states + self.n_forward

        is_det, eig = self.check_determinacy()
        if not is_det:
            n_unstable = int(np.sum(np.abs(eig) > 1.0 + 1e-10))
            if n_unstable < self.n_forward:
                raise IndeterminacyError(
                    f"Indeterminacy: {n_unstable} unstable eigenvalues, need {self.n_forward}."
                )
            raise NoSolutionError(
                f"No solution: {n_unstable} unstable eigenvalues, need exactly {self.n_forward}."
            )

        S, T, alpha, beta_eig, Q, Z = linalg.ordqz(B, A, sort="iuc")
        with np.errstate(divide="ignore", invalid="ignore"):
            eig = np.where(np.abs(beta_eig) < 1e-10, np.inf, alpha / beta_eig)

        n_stable = n - self.n_forward  # 7
        Zt = Z.conj().T
        Z11 = Zt[:n_stable, :self.n_states]
        Z12 = Zt[:n_stable, self.n_states:]
        Z21 = Zt[n_stable:, :self.n_states]
        Z22 = Zt[n_stable:, self.n_states:]
        if np.abs(linalg.det(Z22)) < 1e-10:
            raise NoSolutionError("Z22 singular.")
        R = -linalg.inv(Z22) @ Z21  # 3×7

        S11 = S[:n_stable, :n_stable]
        T11 = T[:n_stable, :n_stable]
        Z_s = Z11 + Z12 @ R
        if np.abs(linalg.det(Z_s)) < 1e-10:
            raise NoSolutionError("Cannot solve for P.")
        P = np.real(linalg.inv(Z_s) @ linalg.solve(T11, S11) @ Z_s)  # 7×7

        try:
            shock_full = linalg.inv(A) @ C
        except linalg.LinAlgError:
            shock_full = linalg.pinv(A) @ C
        Q_mat = np.real(shock_full[:self.n_states, :])  # 7×4

        Sigma = np.diag([p.sigma_demand, p.sigma_supply, p.sigma_omega, p.sigma_monetary])
        return FANKSolution(P=P, Q=Q_mat, R=np.real(R), eigenvalues=eig, Sigma=Sigma)

    # ---- derived observables from a (states, jumps) path ----

    def derive(self, states: np.ndarray, jumps: np.ndarray) -> dict[str, np.ndarray]:
        """Reconstruct y, inv, r^k, safe rate, spread from state/jump paths."""
        p = self.params
        a1 = 1.0 - p.beta * (1.0 - p.delta)
        bd = p.beta * (1.0 - p.delta)
        eps_d, eps_s, omega, Rlag, k, n, qlag = (states[:, i] for i in range(7))
        c, q, pi = jumps[:, 0], jumps[:, 1], jumps[:, 2]

        inv = k + q / p.psi
        y = p.c_y * c + p.i_y * inv
        MPK_C, MPK_K, MPK_Q = self._static_coeffs()["mpk"]
        mpk = MPK_C * c + MPK_K * k + MPK_Q * q
        rk = a1 * mpk + bd * q - qlag                      # realised return on capital
        R = p.rho_i * Rlag + (1 - p.rho_i) * (p.phi_pi * pi + p.phi_y * y)
        # ex-post spread proxy: r^k − real safe rate
        spread = rk - (R - pi)
        return {"y": y, "inv": inv, "rk": rk, "R": R, "spread": spread, "c": c, "q": q, "pi": pi,
                "k": k, "n": n, "omega": omega}

    def compute_irf(self, shock_name: str, periods: int = 24) -> dict[str, np.ndarray]:
        sol = self.solve()
        shock_map = {"demand": 0, "supply": 1, "omega": 2, "monetary": 3}
        eta = np.zeros(self.n_shocks)
        eta[shock_map[shock_name]] = sol.Sigma[shock_map[shock_name], shock_map[shock_name]]

        state = sol.Q @ eta
        states = np.zeros((periods, self.n_states))
        jumps = np.zeros((periods, self.n_forward))
        for t in range(periods):
            states[t] = state
            jumps[t] = sol.R @ state
            state = sol.P @ state
        return self.derive(states, jumps)


# =============================================================================
# Observation equation, likelihood, data, estimation
# =============================================================================

import pandas as pd  # noqa: E402

COVID_START = pd.Period("2020Q1", freq="Q")
COVID_END = pd.Period("2021Q4", freq="Q")

# Observables: [output_gap, inflation, cash_rate, credit_spread] — 4 obs, 4 shocks.
N_OBS = 4


def _observation_matrix(model: "FANKModel", solution: FANKSolution) -> np.ndarray:
    """Build the 4×7 observation matrix Z mapping states to observables.

    Observables (all model deviations): output gap y, inflation π, cash rate R,
    credit spread = ex-ante EFP. Jumps are recovered from states via the policy
    function (jumps = R_policy · s), so every observable is a linear function of
    the 7-state vector.
    """
    p = model.params
    a1 = 1.0 - p.beta * (1.0 - p.delta)
    bd = p.beta * (1.0 - p.delta)
    cy, iy, psi = p.c_y, p.i_y, p.psi
    sc = model._static_coeffs()  # noqa: SLF001 — same class family
    Y_C, Y_K, Y_Q = sc["Y"]
    MPK_C, MPK_K, MPK_Q = sc["mpk"]
    Rp = solution.R  # 3×7 : rows c, q, pi
    ns = model.n_states
    EPSD, EPSS, OMEGA, RLAG, K, N, QLAG = range(7)

    def e(i: int) -> np.ndarray:
        v = np.zeros(ns); v[i] = 1.0; return v

    c_row, q_row, pi_row = Rp[0], Rp[1], Rp[2]
    y_row = Y_C * c_row + Y_K * e(K) + Y_Q * q_row              # output gap
    R_row = p.rho_i * e(RLAG) + (1 - p.rho_i) * (p.phi_pi * pi_row + p.phi_y * y_row)  # cash rate

    # ex-ante EFP = χ·(q + k' − n') + ω
    knext = e(K) + (p.delta / psi) * q_row
    mpk_row = MPK_C * c_row + MPK_K * e(K) + MPK_Q * q_row
    rk_row = a1 * mpk_row + bd * q_row - e(QLAG)
    nnext = p.kappa_n * e(N) + (1 - p.kappa_n) * p.lev_k * (rk_row - e(RLAG) + pi_row)
    efp_row = p.chi * (q_row + knext - nnext) + e(OMEGA)

    return np.vstack([y_row, pi_row, R_row, efp_row])  # 4×7


def _fa_state_space(model: "FANKModel", solution: FANKSolution | None = None):
    if solution is None:
        solution = model.solve()
    T = solution.P
    R = solution.Q
    Q_cov = np.eye(model.n_shocks)
    Z = _observation_matrix(model, solution)
    H = np.diag([0.05 ** 2] * N_OBS)  # small measurement error for numerical stability
    return T, R, Z, Q_cov, H


def compute_fa_nk_log_likelihood(y_obs: np.ndarray, params: FANKParameters, labour_block: bool = False) -> float:
    from src.models.dsge.kalman import kalman_filter
    try:
        model = FANKModel(params=params, labour_block=labour_block)
        T, R, Z, Q, H = _fa_state_space(model)
        return kalman_filter(y_obs, T, R, Z, Q, H).log_likelihood
    except (IndeterminacyError, NoSolutionError):
        return -1e10
    except Exception:
        return -1e10


def _fa_nk_likelihood(params: FANKParameters, data: dict) -> float:
    return compute_fa_nk_log_likelihood(data["y"], params)


def _fa_nk_labour_likelihood(params: FANKParameters, data: dict) -> float:
    return compute_fa_nk_log_likelihood(data["y"], params, labour_block=True)


def fa_nk_extract_states(params: FANKParameters, data: dict, labour_block: bool = False) -> dict:
    from src.models.dsge.kalman import kalman_smoother
    try:
        model = FANKModel(params=params, labour_block=labour_block)
        solution = model.solve()
        T, R, Z, Q, H = _fa_state_space(model, solution)
        res = kalman_smoother(data["y"], T, R, Z, Q, H)
        s = res.smoothed_states
        jumps = s @ solution.R.T
        d = model.derive(s, jumps)
        # ex-ante EFP path
        kn, nn = s[:, 4], s[:, 5]
        q, om = jumps[:, 1], s[:, 2]
        efp = np.full(len(s), np.nan)
        efp[:-1] = params.chi * (q[:-1] + kn[1:] - nn[1:]) + om[:-1]
        states_df = pd.DataFrame({
            "output_gap": d["y"], "inflation": d["pi"], "cash_rate": d["R"],
            "investment": d["inv"], "q": d["q"], "net_worth": d["n"],
            "r_safe": d["R"] - d["pi"], "r_capital": d["rk"],
            "wedge_efp": efp, "omega": d["omega"],
        }, index=data["dates"])
        return {"states": states_df, "log_likelihood": res.log_likelihood}
    except (IndeterminacyError, NoSolutionError):
        return {"states": pd.DataFrame(index=data["dates"]), "log_likelihood": -1e10}


def load_fa_nk_data(start: str = "1993Q1", end: str | None = None, exclude_covid: bool = True) -> dict:
    """Observables [output_gap, inflation, cash_rate(dev), credit_spread(dev)].

    Full sample from 1993Q1 (the working assumption). The corporate spread only
    exists from 2005Q1, so it is left as NaN before then — the Kalman filter uses
    4 observables from 2005 and 3 before, rather than truncating the whole sample
    to the spread's start. The other three observables are complete-case from
    1993. Excludes COVID (2020Q1-2021Q4); keeps the GFC (its spread blowout
    identifies the financial block).
    """
    from src.data.bonds import get_corporate_spread
    from src.models.dsge.data_loader import load_estimation_data

    # 3 core observables, complete-case from `start`
    base = load_estimation_data(start=start, end=end, n_observables=3, anchor_inflation=True)
    spread = get_corporate_spread().data
    if not isinstance(spread.index, pd.PeriodIndex):
        spread.index = pd.PeriodIndex(spread.index, freq="Q")

    # Attach the spread WITHOUT dropping rows where it is missing (pre-2005 → NaN)
    df = base.copy()
    df["credit_spread"] = spread.reindex(base.index)
    if exclude_covid:
        df = df[~((df.index >= COVID_START) & (df.index <= COVID_END))]

    # Deviation space: re-center cash rate over the full sample, spread over its
    # available (2005+) values (mean() skips NaN; NaN − mean stays NaN).
    df["interest_rate"] = df["interest_rate"] - df["interest_rate"].mean()
    df["credit_spread"] = df["credit_spread"] - df["credit_spread"].mean()

    cols = ["output_gap", "inflation", "interest_rate", "credit_spread"]
    return {"y": df[cols].to_numpy(), "dates": df.index, "n_observables": N_OBS}


FA_NK_PARAM_BOUNDS = {
    "chi": (0.001, 0.5),       # EFP / wedge slope — the parameter of interest
    "kappa_p": (0.01, 5.0),    # widened (settles ~2.2, identified)
    "phi_pi": (1.01, 3.0),     # economically-reasonable CAP (Taylor block weakly identified, runs to 9+ if freed)
    "phi_y": (0.0, 1.0),       # economically-reasonable CAP (runs away if freed)
    "rho_i": (0.5, 0.95),
    "rho_demand": (0.3, 0.95),
    "rho_supply": (0.1, 0.9),
    "rho_omega": (0.5, 0.99),
    "sigma_demand": (0.05, 3.0),
    "sigma_supply": (0.05, 3.0),
    "sigma_omega": (0.05, 3.0),
    "sigma_monetary": (0.05, 3.0),
}

from src.models.dsge.estimation import ModelSpec  # noqa: E402

FA_NK_SPEC = ModelSpec(
    name="FA-NK",
    description="Financial-accelerator NK DSGE with endogenous EFP wedge",
    param_class=FANKParameters,
    param_bounds=FA_NK_PARAM_BOUNDS,
    estimate_params=[
        "chi", "kappa_p", "phi_pi", "phi_y", "rho_i",
        "rho_demand", "rho_supply", "rho_omega",
        "sigma_demand", "sigma_supply", "sigma_omega", "sigma_monetary",
    ],
    fixed_params={
        "beta": 0.99, "delta": 0.025, "sigma": 1.0, "xi": 1.0,
        "psi": 4.0, "c_y": 0.8, "i_y": 0.2, "lev_k": 2.0, "kappa_n": 0.95,
    },
    likelihood_fn=_fa_nk_likelihood,
    state_extractor_fn=fa_nk_extract_states,
)


def run_fa_nk(start: str = "1993Q1", end: str | None = None, verbose: bool = True) -> dict:
    """Full-sample (1993+, ex-COVID, GFC kept) estimation of the FA-NK DSGE.

    The credit spread is missing before 2005 (filter handles it); the other three
    observables run from 1993.
    """
    from src.models.dsge.estimation import estimate_model
    data = load_fa_nk_data(start=start, end=end)
    if verbose:
        print(f"FA-NK estimation: {data['dates'][0]} to {data['dates'][-1]} (n={len(data['dates'])})")
    est = estimate_model(FA_NK_SPEC, data, verbose=verbose)
    states = fa_nk_extract_states(est.params, data)["states"]
    return {"params": est.params, "estimation_result": est, "states": states,
            "dates": data["dates"], "data": data}


def _fa_nk_labour_extract(params: FANKParameters, data: dict) -> dict:
    return fa_nk_extract_states(params, data, labour_block=True)


# Tier-1 labour-block spec: structural mc = w + l − y instead of ξ·y. α and φ_l
# fixed (no labour-market observable to identify them); same estimated set otherwise.
FA_NK_LABOUR_SPEC = ModelSpec(
    name="FA-NK-labour",
    description="FA-NK with Tier-1 labour block (structural marginal cost)",
    param_class=FANKParameters,
    param_bounds=FA_NK_PARAM_BOUNDS,
    estimate_params=list(FA_NK_SPEC.estimate_params),
    fixed_params={**FA_NK_SPEC.fixed_params, "alpha": 0.33, "phi_l": 1.0},
    likelihood_fn=_fa_nk_labour_likelihood,
    state_extractor_fn=_fa_nk_labour_extract,
)


def run_fa_nk_labour(start: str = "1993Q1", end: str | None = None, verbose: bool = True) -> dict:
    """Estimate the FA-NK with the Tier-1 labour block (structural marginal cost)."""
    from src.models.dsge.estimation import estimate_model
    data = load_fa_nk_data(start=start, end=end)
    if verbose:
        print(f"FA-NK (labour block) estimation: {data['dates'][0]} to {data['dates'][-1]} (n={len(data['dates'])})")
    est = estimate_model(FA_NK_LABOUR_SPEC, data, verbose=verbose)
    states = fa_nk_extract_states(est.params, data, labour_block=True)["states"]
    return {"params": est.params, "estimation_result": est, "states": states,
            "dates": data["dates"], "data": data}


def _mean_real_rate(dates: pd.PeriodIndex) -> float:
    """Empirical mean real cash rate (cash rate − annualised trimmed-mean inflation)
    over the estimation dates — used to re-centre deviation-space rates as levels."""
    from src.data.abs_loader import load_series
    from src.data.cash_rate import get_cash_rate_qrtly
    from src.data.series_specs import CPI_TRIMMED_MEAN_QUARTERLY

    cash = get_cash_rate_qrtly().data.copy()
    inf = load_series(CPI_TRIMMED_MEAN_QUARTERLY).data.copy()
    for s in (cash, inf):
        if not isinstance(s.index, pd.PeriodIndex):
            s.index = pd.PeriodIndex(s.index, freq="Q")
    inf_annual = ((1 + inf / 100) ** 4 - 1) * 100
    real = (cash - inf_annual).reindex(dates).dropna()
    return float(real.mean())


def _historical_decomposition(model: "FANKModel", solution: FANKSolution, data: dict):
    """Decompose smoothed variables into the 4 structural shocks.

    Returns (Z, smoothed_states S, per-shock state contributions dict). Each
    contribution is the path the states would have taken under that shock alone,
    so for any observation row Z[i], `contrib[j] @ Z[i]` is shock j's contribution
    to that observable and the four sum to the smoothed total.
    """
    from src.models.dsge.kalman import kalman_smoother

    Tm, Rm, Z, Qc, H = _fa_state_space(model, solution)
    res = kalman_smoother(data["y"], Tm, Rm, Z, Qc, H)
    S = res.smoothed_states  # (T, 7)
    P, Qm = solution.P, solution.Q  # 7×7, 7×4
    Qpinv = np.linalg.pinv(Qm)      # 4×7
    Tn = len(S)

    eta = np.zeros((Tn, 4))
    eta[0] = Qpinv @ S[0]
    for t in range(1, Tn):
        eta[t] = Qpinv @ (S[t] - P @ S[t - 1])

    contrib = {}
    for j in range(4):
        Sj = np.zeros((Tn, 7))
        sj = Qm[:, j] * eta[0, j]
        Sj[0] = sj
        for t in range(1, Tn):
            sj = P @ sj + Qm[:, j] * eta[t, j]
            Sj[t] = sj
        contrib[j] = Sj
    return Z, S, contrib, res


def produce_fa_nk_outputs(out: dict | None = None, start: str = "2005Q1", end: str | None = None) -> tuple:
    """Estimate (if needed), save smoothed states + params, and render charts.

    Charts (charts/dsge-fa-nk/): actual safe rate vs cost of capital (levels) with
    the natural rate; EFP wedge; shock decompositions of the wedge and the output
    gap; financial-block internals; IRFs to financial and monetary shocks.
    """
    from pathlib import Path

    import mgplot as mg

    root = Path(__file__).parent.parent.parent.parent
    chart_dir = root / "charts" / "dsge-fa-nk"
    output_dir = root / "model_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if out is None:
        out = run_fa_nk(start=start, end=end, verbose=False)
    p, est, st, data = out["params"], out["estimation_result"], out["states"], out["data"]

    # Natural rate of interest (approx): the demand/preference-shock component, r^n ≈ ε_d/σ.
    model = FANKModel(params=p)
    sol = model.solve()
    Z, S, contrib, smooth = _historical_decomposition(model, sol, data)
    st = st.copy()
    st["r_natural"] = S[:, 0] / p.sigma  # ε_d is state 0
    # smoothed s.d. of r* (= s.d. of ε_d / σ) for the uncertainty band
    rstar_sd = np.sqrt(np.maximum(smooth.smoothed_covs[:, 0, 0], 0.0)) / p.sigma

    # --- persisted data ---
    st.to_csv(output_dir / "fa_nk_states.csv")
    lines = [
        "FA-NK DSGE (financial-accelerator NK) — estimated parameters",
        f"sample: {st.index.min()} to {st.index.max()} (n={len(st)}), COVID excluded",
        f"log-likelihood: {est.log_likelihood:.2f}",
        f"at bounds: {est.params_at_bounds}",
        "",
    ]
    lines += [f"  {k:15s} = {v:.4f}" for k, v in p.to_dict().items()]
    (output_dir / "fa_nk_params.txt").write_text("\n".join(lines))

    # continuous quarterly index so the COVID gap renders as a break
    full = pd.period_range(st.index.min(), st.index.max(), freq="Q")
    stp = st.reindex(full)

    mg.set_chart_dir(str(chart_dir))
    mg.clear_chart_dir()
    RF = "FA-NK DSGE"

    # 1. Actual safe rate vs cost of capital (RE-LEVELLED) + natural rate
    mrr = _mean_real_rate(st.index)
    levels = pd.DataFrame({
        "Actual safe real rate (R-π)": stp["r_safe"] + mrr,
        "Cost of capital (= safe + EFP)": stp["r_safe"] + stp["wedge_efp"] + mrr,
        "Natural rate r* (ε_d-implied, approx)": stp["r_natural"] + mrr,
    })
    mg.line_plot_finalise(
        levels, width=[2, 2, 1.5], style=["-", "-", "--"],
        color=["navy", "firebrick", "darkgreen"], dropna=False,
        title="FA-NK safe rate, cost of capital and natural rate",
        ylabel="Per cent (annual)",
        legend={"loc": "best"}, rfooter=RF,
        lfooter=f"Australia. COVID excluded. Levels = model deviation + sample-mean real rate ({mrr:.1f}%). ",
    )

    # 1b. STANDALONE natural rate r* in levels, with ±2 s.d. band
    rstar_sd_s = pd.Series(rstar_sd, index=st.index)
    rstar_chart = pd.DataFrame({
        "r* (natural rate, model)": stp["r_natural"] + mrr,
        "−2 s.d.": (st["r_natural"] - 2 * rstar_sd_s + mrr).reindex(full),
        "+2 s.d.": (st["r_natural"] + 2 * rstar_sd_s + mrr).reindex(full),
    })
    mg.line_plot_finalise(
        rstar_chart, width=[2.5, 1, 1], style=["-", "--", "--"],
        color=["navy", "grey", "grey"], dropna=False, annotate=False,
        title="FA-NK natural rate of interest r-star", ylabel="Per cent (annual)",
        legend={"loc": "best"}, rfooter=RF,
        lfooter="Australia. Approx r*. Band = state s.d. only (excludes parameter uncertainty). ",
    )

    # 2. Endogenous EFP wedge (deviation)
    mg.line_plot_finalise(
        stp["wedge_efp"].rename("EFP wedge"), width=2, color=["darkgreen"], dropna=False,
        title="FA-NK external finance premium wedge", ylabel="Deviation (ppt)",
        y0=True, rfooter=RF, lfooter="Australia. COVID excluded. Deviation from steady state. ",
    )

    # 3 & 4. Historical shock decomposition (EFP wedge and output gap)
    shock_lbl = ["Demand", "Supply (cost-push)", "Financial (ω)", "Monetary"]
    dec_color = ["navy", "darkorange", "firebrick", "purple"]

    def decomp_df(zrow: np.ndarray, total: pd.Series) -> pd.DataFrame:
        cols = {"Total (smoothed)": total}
        for j in range(4):
            cols[shock_lbl[j]] = pd.Series(contrib[j] @ zrow, index=st.index)
        return pd.DataFrame(cols).reindex(full)

    efp_dec = decomp_df(Z[3], stp["wedge_efp"])
    mg.line_plot_finalise(
        efp_dec, width=[2.5, 1.5, 1.5, 1.5, 1.5], dropna=False,
        color=["black", *dec_color],
        title="FA-NK EFP wedge: shock decomposition", ylabel="Contribution (ppt deviation)",
        y0=True, legend={"loc": "best"}, rfooter=RF,
        lfooter="Australia. COVID excluded. Wedge decomposed into structural shocks. ",
    )

    ygap_dec = decomp_df(Z[0], stp["output_gap"])
    mg.line_plot_finalise(
        ygap_dec, width=[2.5, 1.5, 1.5, 1.5, 1.5], dropna=False,
        color=["black", *dec_color],
        title="FA-NK output gap: shock decomposition", ylabel="Contribution (ppt deviation)",
        y0=True, legend={"loc": "best"}, rfooter=RF,
        lfooter="Australia. COVID excluded. Output gap decomposed into structural shocks. ",
    )

    # 5. Financial-block internals (deviations). k is not in states_df; take it from S[:,4].
    k_series = pd.Series(S[:, 4], index=st.index).reindex(full)
    internals = pd.DataFrame({
        "Tobin's Q": stp["q"],
        "Net worth": stp["net_worth"],
        "Investment": stp["investment"],
        "Leverage (q+k-n)": stp["q"] + k_series - stp["net_worth"],
    })
    mg.line_plot_finalise(
        internals, width=2, color=["navy", "firebrick", "darkorange", "purple"], dropna=False,
        title="FA-NK financial block internals", ylabel="Deviation from steady state",
        y0=True, legend={"loc": "best"}, rfooter=RF,
        lfooter="Australia. COVID excluded. ", annotate=False,
    )

    # 6 & 7. Impulse responses at the ESTIMATED parameters
    def irf(shock: str, T: int = 20) -> pd.DataFrame:
        idx = {"demand": 0, "supply": 1, "omega": 2, "monetary": 3}[shock]
        eta = np.zeros(4); eta[idx] = sol.Sigma[idx, idx]
        s = sol.Q @ eta
        Sx = np.zeros((T + 1, 7)); Jx = np.zeros((T + 1, 3))
        for t in range(T + 1):
            Sx[t] = s; Jx[t] = sol.R @ s; s = sol.P @ s
        d = model.derive(Sx, Jx)
        efp = p.chi * (Jx[:, 1] + np.r_[Sx[1:, 4], np.nan] - np.r_[Sx[1:, 5], np.nan]) + Sx[:, 2]
        return pd.DataFrame({
            "Output": d["y"], "Investment": d["inv"], "Inflation": d["pi"],
            "Cash rate": d["R"], "EFP wedge": efp,
        }).iloc[:T]

    irf_colors = ["navy", "firebrick", "darkgreen", "darkorange", "purple"]
    mg.line_plot_finalise(
        irf("omega"), width=2, color=irf_colors, annotate=False,
        title="FA-NK IRF to a financial (spread) shock", xlabel="Quarters after shock",
        ylabel="Response (ppt deviation)", y0=True, legend={"loc": "best"},
        rfooter=RF, lfooter="Australia. One standard-deviation shock. ",
    )
    mg.line_plot_finalise(
        irf("monetary"), width=2, color=irf_colors, annotate=False,
        title="FA-NK IRF to a monetary tightening", xlabel="Quarters after shock",
        ylabel="Response (ppt deviation)", y0=True, legend={"loc": "best"},
        rfooter=RF, lfooter="Australia. One standard-deviation shock. ",
    )

    print(f"\n[4] Outputs written:\n      {output_dir}/fa_nk_states.csv"
          f"\n      {output_dir}/fa_nk_params.txt\n      {chart_dir}/  (7 charts)")
    return chart_dir, output_dir


if __name__ == "__main__":
    print("=" * 64)
    print("FINANCIAL-ACCELERATOR NK DSGE  (path 1)")
    print("=" * 64)

    # 1. Determinacy at calibrated defaults
    m = FANKModel()
    det, eig = m.check_determinacy()
    n_unstable = int(np.sum(np.abs(eig) > 1.0 + 1e-10))
    print(f"\n[1] Calibrated solve: determinate={det}, "
          f"{n_unstable} unstable eigenvalues (need {m.n_forward})")

    # 2. Estimate on AU data (2005Q1+, GFC kept, COVID excluded)
    print("\n[2] Estimation")
    out = run_fa_nk(verbose=True)
    p, est, st = out["params"], out["estimation_result"], out["states"]
    print("\n    parameter estimates:")
    for k in FA_NK_PARAM_BOUNDS:
        v = getattr(p, k); lo, hi = FA_NK_PARAM_BOUNDS[k]
        at = " <-- LOWER" if abs(v - lo) < 0.01 * (hi - lo) + 1e-6 else (
             " <-- UPPER" if abs(v - hi) < 0.01 * (hi - lo) + 1e-6 else "")
        print(f"      {k:15s} = {v:7.3f}   [{lo}, {hi}]{at}")
    print(f"\n    log-likelihood: {est.log_likelihood:.2f}")
    print(f"    at bounds: {est.params_at_bounds}")

    # 3. Smoothed two natural rates + endogenous wedge
    print("\n[3] Smoothed r*_safe / r*_capital / EFP wedge (deviations, key dates):")
    for yr in ["2007Q2", "2009Q1", "2015Q2", "2019Q4", "2025Q4"]:
        try:
            r = st.loc[pd.Period(yr, "Q")]
            print(f"      {yr}: r_safe={r['r_safe']:+.2f}  "
                  f"r_capital={r['r_capital']:+.2f}  EFP={r['wedge_efp']:+.2f}")
        except KeyError:
            pass

    # 4. Save states/params and render charts
    produce_fa_nk_outputs(out=out)
