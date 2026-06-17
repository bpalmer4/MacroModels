"""New Keynesian DSGE with TWO natural rates (goods-market r* and a capital wedge).

This is the "two r*s, stop forcing one" experiment. It keeps the closed,
forward-looking NK DSGE of nk_model.py (Taylor rule, Blanchard-Kahn solve) but
stops assuming a single r*. The thesis (the post-GFC "great divergence") is that
post-GFC the goods-market safe rate and the return on capital decoupled and
stayed decoupled - the "great divergence" - so the model should carry both.

Two natural rates, both pinned by OBSERVED anchors (no free random-walk r*,
which is what left the standalone hlw_model.py under-identified):

  r*_goods  - the low, safe, goods-market-clearing rate the IS/Euler margin and
              the Taylor rule actually face. Anchored to the real (indexed) 10y
              bond yield.   [src.data.bonds.get_indexed_yield_filled]
  r*_capital- the high return on capital / hurdle rate. Anchored to trend GDP
              growth.        [trend of YoY log-GDP growth]
  wedge ω   = r*_capital - r*_goods. The great divergence, as a state.

Changes vs nk_model.py (the three-equation graft we agreed to try linearly):

1. Taylor rule neutral anchor is the time-varying r*_goods (not a constant):
       i_t = ρ_i·i_{t-1} + (1-ρ_i)·( r*_goods,t + φ_π·π_t + φ_y·ŷ_t ) + ε_m
   This builds Bullock's "neutral rate shifted in 16 months" into the rule and
   is our candidate fix for why φ_π, φ_y hit their bounds in nk_model.py: there
   the rule had to contort to track a falling neutral rate it assumed constant.

2. IS curve neutral rate is also r*_goods (the consumption/safe margin):
       ŷ_t = E[ŷ'] - σ·(i_t - E[π'] - r*_goods,t) - σ_k·ω_t + ε_d

3. The wedge enters the IS curve linearly as the broken investment channel
   (-σ_k·ω_t): a wide hurdle-rate-vs-safe-rate gap suppresses demand
   independently of the policy rate. σ_k > 0 is the transmission-failure signal.

Determinacy is untouched: r*_goods and ω are exogenous AR(1) forcing states, not
feedback, so the Taylor principle (φ_π > 1) still governs the Blanchard-Kahn
eigenvalue count. The model stays a determinate DSGE.

Everything is in deviation-from-steady-state space (as in nk_model.py), so the
two anchors enter as deviations from their sample means - they capture the
*time variation* (the divergence), not the level.

State vector  s_t = [ε_d, ε_s, ε_w, i, r_goods, ω]   (6 predetermined)
Controls      y_t = [ŷ, π, π_w]                       (3 forward-looking)
Observables   [ŷ, π, i, π_w, u_gap, r_goods, ω]       (7)
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import linalg

from src.models.dsge.nk_model import IndeterminacyError, NoSolutionError


@dataclass
class NKTwoStarParameters:
    """Parameters for the two-r* NK DSGE."""

    # Structural
    sigma: float = 1.0  # IES (IS curve slope)
    beta: float = 0.99  # Discount factor
    kappa_p: float = 0.1  # Price Phillips curve slope
    kappa_w: float = 0.1  # Wage Phillips curve slope
    sigma_k: float = 0.1  # Wedge effect on demand (the broken investment channel)

    # Okun's law: u_gap = -omega * y_gap
    omega: float = 0.5

    # Taylor rule
    phi_pi: float = 1.5  # Response to inflation (must be > 1 for determinacy)
    phi_y: float = 0.5  # Response to output gap
    rho_i: float = 0.8  # Interest rate smoothing

    # Exogenous-shock persistence
    rho_demand: float = 0.8
    rho_supply: float = 0.5
    rho_wage: float = 0.5
    rho_rg: float = 0.92  # r*_goods persistence (data-pinned; high)
    rho_omega: float = 0.92  # wedge persistence (data-pinned; high)

    # Shock volatilities
    sigma_demand: float = 0.5
    sigma_supply: float = 0.3
    sigma_wage: float = 0.3
    sigma_monetary: float = 0.25
    sigma_rg: float = 0.3  # r*_goods innovation
    sigma_omega: float = 0.3  # wedge innovation

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class NKTwoStarSolution:
    """Solution to the two-r* NK model.

    State:   s_t = [ε_d, ε_s, ε_w, i, r_goods, ω]
    Control: y_t = [ŷ, π, π_w]
    """

    P: np.ndarray  # 6×6 state transition
    Q: np.ndarray  # 6×6 shock impact on states
    R: np.ndarray  # 3×6 policy function
    eigenvalues: np.ndarray
    Sigma: np.ndarray  # 6×6 shock volatilities


@dataclass
class NKTwoStarModel:
    """Two-r* NK DSGE with goods-market r* and a capital wedge.

    6 predetermined states, 3 forward-looking controls, solved via Blanchard-Kahn.
    """

    params: NKTwoStarParameters = field(default_factory=NKTwoStarParameters)

    n_states: int = 6  # ε_d, ε_s, ε_w, i, r_goods, ω
    n_shocks: int = 6  # η_d, η_s, η_w, η_m, η_rg, η_ω
    n_forward: int = 3  # ŷ, π, π_w

    state_names: list[str] = field(
        default_factory=lambda: ["eps_demand", "eps_supply", "eps_wage", "i", "r_goods", "wedge"]
    )
    shock_names: list[str] = field(
        default_factory=lambda: [
            "eta_demand", "eta_supply", "eta_wage", "eta_monetary", "eta_rg", "eta_omega",
        ]
    )
    forward_names: list[str] = field(default_factory=lambda: ["y", "pi", "pi_w"])

    def _build_system_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build A, B, C for  A·E_t[z_{t+1}] = B·z_t + C·η_{t+1}.

        z = [ε_d, ε_s, ε_w, i, r_goods, ω, ŷ, π, π_w]  (9 variables)
        indices: eps_d=0, eps_s=1, eps_w=2, i=3, r_goods=4, wedge=5, y=6, pi=7, pi_w=8
        """
        p = self.params
        n = self.n_states + self.n_forward  # 9

        A = np.zeros((n, n))
        B = np.zeros((n, n))
        C = np.zeros((n, self.n_shocks))

        # 1-3. Exogenous AR(1) shocks
        A[0, 0] = 1.0; B[0, 0] = p.rho_demand; C[0, 0] = p.sigma_demand
        A[1, 1] = 1.0; B[1, 1] = p.rho_supply; C[1, 1] = p.sigma_supply
        A[2, 2] = 1.0; B[2, 2] = p.rho_wage; C[2, 2] = p.sigma_wage

        # 4. Taylor rule with smoothing; neutral anchor is r*_goods (state 4):
        #    ĩ_{t+1} = i_t = ρ_i·ĩ_t + (1-ρ_i)·(r_goods_t + φ_π·π_t + φ_y·ŷ_t) + ε_m
        A[3, 3] = 1.0
        B[3, 3] = p.rho_i              # ĩ
        B[3, 4] = (1 - p.rho_i) * 1.0  # r_goods  (neutral rate enters the rule)
        B[3, 6] = (1 - p.rho_i) * p.phi_y   # ŷ
        B[3, 7] = (1 - p.rho_i) * p.phi_pi  # π
        C[3, 3] = p.sigma_monetary

        # 5. r*_goods dynamics (exogenous AR(1), pinned by the bond-yield observation)
        A[4, 4] = 1.0; B[4, 4] = p.rho_rg; C[4, 4] = p.sigma_rg

        # 6. wedge ω dynamics (exogenous AR(1), pinned by the (g - bond) observation)
        A[5, 5] = 1.0; B[5, 5] = p.rho_omega; C[5, 5] = p.sigma_omega

        # 7. IS curve:  ŷ = E[ŷ'] - σ(i - E[π'] - r_goods) - σ_k·ω + ε_d
        #    Substitute i = ρ_i·ĩ + (1-ρ_i)(r_goods + φ_π·π + φ_y·ŷ) + ε_m  (drop ε_m
        #    for A/B, as nk_model.py does; the monetary shock enters via the i state).
        #    => E[ŷ'] + σ·E[π'] = (1+σ(1-ρ_i)φ_y)·ŷ + σ(1-ρ_i)φ_π·π
        #                         + σρ_i·ĩ - σρ_i·r_goods + σ_k·ω - ε_d
        A[6, 6] = 1.0          # E[ŷ']
        A[6, 7] = p.sigma      # E[π']
        B[6, 6] = 1.0 + p.sigma * (1 - p.rho_i) * p.phi_y   # ŷ
        B[6, 7] = p.sigma * (1 - p.rho_i) * p.phi_pi        # π
        B[6, 3] = p.sigma * p.rho_i        # ĩ
        B[6, 4] = -p.sigma * p.rho_i       # r_goods  (net of the +σ·r_goods term)
        B[6, 5] = p.sigma_k                # ω  (the wedge / broken investment channel)
        B[6, 0] = -1.0                     # ε_d

        # 8. Price Phillips: π = β·E[π'] + κ_p·ŷ + ε_s
        A[7, 7] = p.beta; B[7, 7] = 1.0; B[7, 6] = -p.kappa_p; B[7, 1] = -1.0

        # 9. Wage Phillips: π_w = β·E[π_w'] + κ_w·ŷ + ε_w
        A[8, 8] = p.beta; B[8, 8] = 1.0; B[8, 6] = -p.kappa_w; B[8, 2] = -1.0

        return A, B, C

    def check_determinacy(self) -> tuple[bool, np.ndarray]:
        """Blanchard-Kahn: need exactly n_forward eigenvalues outside the unit circle."""
        A, B, _ = self._build_system_matrices()
        _, _, alpha, beta_eig, _, _ = linalg.ordqz(B, A, sort="iuc")
        with np.errstate(divide="ignore", invalid="ignore"):
            eigenvalues = np.where(np.abs(beta_eig) < 1e-10, np.inf, alpha / beta_eig)
        n_unstable = np.sum(np.abs(eigenvalues) > 1.0 + 1e-10)
        return n_unstable == self.n_forward, eigenvalues

    def solve(self) -> NKTwoStarSolution:
        """Solve via Blanchard-Kahn (same machinery as nk_model.py, generalised dims)."""
        A, B, C = self._build_system_matrices()
        p = self.params
        n = self.n_states + self.n_forward

        is_det, eigenvalues = self.check_determinacy()
        if not is_det:
            n_unstable = np.sum(np.abs(eigenvalues) > 1.0 + 1e-10)
            if n_unstable < self.n_forward:
                raise IndeterminacyError(
                    f"Indeterminacy: {n_unstable} unstable eigenvalues, need {self.n_forward}."
                )
            raise NoSolutionError(
                f"No solution: {n_unstable} unstable eigenvalues, need exactly {self.n_forward}."
            )

        S, T, alpha, beta_eig, Q, Z = linalg.ordqz(B, A, sort="iuc")
        with np.errstate(divide="ignore", invalid="ignore"):
            eigenvalues = np.where(np.abs(beta_eig) < 1e-10, np.inf, alpha / beta_eig)

        n_stable = n - self.n_forward  # 6

        Z_full = Z.conj().T
        Z11 = Z_full[:n_stable, :self.n_states]
        Z12 = Z_full[:n_stable, self.n_states:]
        Z21 = Z_full[n_stable:, :self.n_states]
        Z22 = Z_full[n_stable:, self.n_states:]

        if np.abs(linalg.det(Z22)) < 1e-10:
            raise NoSolutionError("Z22 singular - no unique solution.")

        R = -linalg.inv(Z22) @ Z21  # 3×6 policy function

        S11 = S[:n_stable, :n_stable]
        T11 = T[:n_stable, :n_stable]
        Z_s = Z11 + Z12 @ R  # 6×6
        if np.abs(linalg.det(Z_s)) < 1e-10:
            raise NoSolutionError("Cannot solve for state transition P.")

        P = np.real(linalg.inv(Z_s) @ linalg.solve(T11, S11) @ Z_s)  # 6×6

        try:
            shock_full = linalg.inv(A) @ C
        except linalg.LinAlgError:
            shock_full = linalg.pinv(A) @ C
        Q_mat = np.real(shock_full[:self.n_states, :])  # 6×6

        Sigma = np.diag([
            p.sigma_demand, p.sigma_supply, p.sigma_wage,
            p.sigma_monetary, p.sigma_rg, p.sigma_omega,
        ])

        return NKTwoStarSolution(
            P=P, Q=Q_mat, R=np.real(R), eigenvalues=eigenvalues, Sigma=Sigma,
        )

    def state_space_matrices(
        self, solution: NKTwoStarSolution | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """State-space matrices for the Kalman filter (fixed 7 observables).

        Observables: [ŷ, π, i, π_w, u_gap, r_goods, ω]
        """
        if solution is None:
            solution = self.solve()

        T = solution.P
        R = solution.Q
        Q_cov = np.eye(self.n_shocks)
        Rp = solution.R  # 3×6 policy: [ŷ, π, π_w] = Rp @ s

        def unit_row(idx: int) -> np.ndarray:
            row = np.zeros((1, self.n_states))
            row[0, idx] = 1.0
            return row

        y_row = Rp[0:1, :]
        pi_row = Rp[1:2, :]
        i_row = unit_row(3)
        pi_w_row = Rp[2:3, :]
        u_gap_row = -self.params.omega * Rp[0:1, :]  # Okun's law on the output gap
        rg_row = unit_row(4)
        wedge_row = unit_row(5)

        Z = np.vstack([y_row, pi_row, i_row, pi_w_row, u_gap_row, rg_row, wedge_row])  # 7×6

        # Measurement error: u_gap (Okun is approximate) and the two anchors
        # (bond yield / trend growth are proxies for r*, not r* itself).
        H = np.zeros((7, 7))
        H[4, 4] = 0.1 ** 2   # u_gap
        H[5, 5] = 0.2 ** 2   # r_goods proxy error
        H[6, 6] = 0.2 ** 2   # wedge proxy error

        return T, R, Z, Q_cov, H

    def compute_impulse_responses(
        self, shock_name: str, periods: int = 40, solution: NKTwoStarSolution | None = None,
    ) -> dict[str, np.ndarray]:
        """Impulse responses to a one-SD shock."""
        if solution is None:
            solution = self.solve()
        shock_map = {
            "demand": 0, "supply": 1, "wage": 2, "monetary": 3, "rg": 4, "omega": 5,
        }
        eta = np.zeros(self.n_shocks)
        eta[shock_map[shock_name]] = solution.Sigma[shock_map[shock_name], shock_map[shock_name]]

        state = solution.Q @ eta
        irfs_state = np.zeros((periods, self.n_states))
        irfs_control = np.zeros((periods, self.n_forward))
        for t in range(periods):
            irfs_state[t, :] = state
            irfs_control[t, :] = solution.R @ state
            state = solution.P @ state

        return {
            **{name: irfs_state[:, j] for j, name in enumerate(self.state_names)},
            "y": irfs_control[:, 0], "pi": irfs_control[:, 1], "pi_w": irfs_control[:, 2],
        }


# =============================================================================
# Likelihood and state extraction
# =============================================================================


def compute_nk_twostar_log_likelihood(y_obs: np.ndarray, params: NKTwoStarParameters) -> float:
    """Log-likelihood via Kalman filter (returns -1e10 on solve failure)."""
    from src.models.dsge.kalman import kalman_filter

    try:
        model = NKTwoStarModel(params=params)
        T, R, Z, Q, H = model.state_space_matrices()
        result = kalman_filter(y_obs, T, R, Z, Q, H)
        return result.log_likelihood
    except (IndeterminacyError, NoSolutionError):
        return -1e10
    except Exception:
        return -1e10


def _nk_twostar_likelihood(params: NKTwoStarParameters, data: dict) -> float:
    return compute_nk_twostar_log_likelihood(y_obs=data["y"], params=params)


def nk_twostar_extract_states(params: NKTwoStarParameters, data: dict) -> dict:
    """Kalman-smoothed states + implied controls and the two natural rates."""
    from src.models.dsge.kalman import kalman_smoother

    try:
        model = NKTwoStarModel(params=params)
        solution = model.solve()
        T, R, Z, Q, H = model.state_space_matrices(solution)
        result = kalman_smoother(data["y"], T, R, Z, Q, H)
        s = result.smoothed_states
        controls = s @ solution.R.T

        states_df = pd.DataFrame({
            "eps_demand": s[:, 0],
            "eps_supply": s[:, 1],
            "eps_wage": s[:, 2],
            "interest_rate": s[:, 3],
            "r_goods": s[:, 4],
            "wedge": s[:, 5],
            "r_capital": s[:, 4] + s[:, 5],  # r*_capital = r*_goods + ω
            "output_gap": controls[:, 0],
            "inflation": controls[:, 1],
            "wage_inflation": controls[:, 2],
        }, index=data["dates"])

        return {"states": states_df, "log_likelihood": result.log_likelihood}
    except (IndeterminacyError, NoSolutionError):
        return {"states": pd.DataFrame(index=data["dates"]), "log_likelihood": -1e10}


# =============================================================================
# Data loading
# =============================================================================


COVID_START = pd.Period("2020Q1", freq="Q")
COVID_END = pd.Period("2021Q4", freq="Q")


def load_nk_twostar_data(
    start: str = "1993Q1", end: str | None = None, exclude_covid: bool = True,
) -> dict:
    """Load 7 observables: [ŷ, π, i, π_w, u_gap, r_goods, ω].

    r_goods : real (indexed) 10y bond yield, demeaned (the low yield anchor).
    ω       : trend GDP growth − real bond yield, demeaned (the great divergence).
    Both are deviations from their sample means (deviation-space model).

    exclude_covid : drop 2020Q1-2021Q4. The COVID GDP collapse/rebound otherwise
        wrecks the HP-trend growth that anchors r*_capital (and hence ω). Removed
        from both the trend filter and the estimation sample. Note: this leaves a
        gap in the otherwise-contiguous quarterly index, which the Kalman filter
        treats as a single transition step (same crude handling the other DSGE
        models use for their crisis exclusion).
    """
    from src.data.bonds import get_indexed_yield_filled
    from src.data.gdp import get_log_gdp
    from src.models.dsge.data_loader import hp_filter, load_estimation_data

    # Core 5 observables (output_gap, inflation [anchor-adjusted], i [demeaned], wage, u_gap)
    base = load_estimation_data(start=start, end=end, n_observables=5, anchor_inflation=True)

    # Yield anchor: real indexed 10y bond yield (annual %)
    indexed_10y = get_indexed_yield_filled().data
    if not isinstance(indexed_10y.index, pd.PeriodIndex):
        indexed_10y.index = pd.PeriodIndex(indexed_10y.index, freq="Q")

    # Growth anchor: trend of YoY GDP growth (annual %). Drop COVID quarters
    # BEFORE the HP filter so the collapse/rebound spike does not bend the trend.
    # NB: get_log_gdp() returns the GDP *level* (~1130) despite its name, so use
    # pct_change, not diff, for the growth rate.
    gdp_level = get_log_gdp().data
    if not isinstance(gdp_level.index, pd.PeriodIndex):
        gdp_level.index = pd.PeriodIndex(gdp_level.index, freq="Q")
    yoy_growth = (gdp_level.pct_change(4) * 100).dropna()
    if exclude_covid:
        covid = (yoy_growth.index >= COVID_START) & (yoy_growth.index <= COVID_END)
        yoy_growth = yoy_growth[~covid]
    g_trend_vals, _ = hp_filter(yoy_growth.to_numpy(), lamb=1600)
    trend_growth = pd.Series(g_trend_vals, index=yoy_growth.index, name="trend_growth")

    # Assemble on the base sample
    extra = pd.DataFrame({
        "r_goods": indexed_10y.reindex(base.index),
        "trend_growth": trend_growth.reindex(base.index),
    })
    df = base.join(extra).dropna()

    if exclude_covid:
        covid = (df.index >= COVID_START) & (df.index <= COVID_END)
        df = df[~covid]

    df["wedge"] = df["trend_growth"] - df["r_goods"]
    # Deviation-space: demean the anchors AND wage inflation over the estimation
    # sample. The model's π_w control is a zero-mean deviation; raw wage inflation
    # has mean ~5.9, so without demeaning the filter inflates the wage shock to
    # swallow the constant offset (σ_wage pegged at its ceiling).
    df["r_goods"] = df["r_goods"] - df["r_goods"].mean()
    df["wedge"] = df["wedge"] - df["wedge"].mean()
    df["wage_inflation"] = df["wage_inflation"] - df["wage_inflation"].mean()

    cols = ["output_gap", "inflation", "interest_rate", "wage_inflation", "u_gap", "r_goods", "wedge"]
    return {"y": df[cols].to_numpy(), "dates": df.index, "n_observables": 7}


# =============================================================================
# Parameter bounds and spec
# =============================================================================

NK_TWOSTAR_PARAM_BOUNDS = {
    "sigma": (0.5, 3.0),
    "kappa_p": (0.01, 1.5),
    "kappa_w": (0.01, 1.5),
    "sigma_k": (0.0, 2.0),       # wedge effect; allowed to be ~0 (collapses to one-r*)
    "omega": (0.2, 0.8),
    "phi_pi": (1.01, 3.0),       # > 1 for determinacy
    "phi_y": (0.1, 1.0),
    "rho_i": (0.5, 0.95),
    "rho_rg": (0.5, 0.99),
    "rho_omega": (0.5, 0.99),
    "sigma_demand": (0.1, 2.0),
    "sigma_supply": (0.05, 1.0),
    "sigma_wage": (0.05, 1.0),
    "sigma_monetary": (0.05, 1.0),
    "sigma_rg": (0.05, 3.0),
    "sigma_omega": (0.05, 3.0),
}

from src.models.dsge.estimation import ModelSpec  # noqa: E402

NK_TWOSTAR_SPEC = ModelSpec(
    name="NK-TwoStar",
    description="NK DSGE with goods-market r* + capital wedge (linear), Taylor rule anchored to r*_goods",
    param_class=NKTwoStarParameters,
    param_bounds=NK_TWOSTAR_PARAM_BOUNDS,
    estimate_params=[
        "kappa_p", "kappa_w", "sigma_k", "omega", "phi_pi", "phi_y", "rho_i",
        "sigma_demand", "sigma_supply", "sigma_wage", "sigma_monetary",
        "sigma_rg", "sigma_omega",
    ],
    fixed_params={
        "sigma": 1.0, "beta": 0.99,
        "rho_demand": 0.8, "rho_supply": 0.5, "rho_wage": 0.5,
        "rho_rg": 0.92, "rho_omega": 0.92,
    },
    likelihood_fn=_nk_twostar_likelihood,
    state_extractor_fn=nk_twostar_extract_states,
)


def run_full_sample(start: str = "1993Q1", end: str | None = None, verbose: bool = True) -> dict:
    """Estimate on the FULL sample (no crisis exclusion).

    The wedge's identification lives in the 2008-2020 divergence, so unlike the
    other DSGE models we do NOT exclude the crisis window here.
    """
    from src.models.dsge.estimation import estimate_model

    data = load_nk_twostar_data(start=start, end=end)
    if verbose:
        print(f"NK-TwoStar - full-sample estimation: {data['dates'][0]} to {data['dates'][-1]} "
              f"(n={len(data['dates'])})")
    est = estimate_model(NK_TWOSTAR_SPEC, data, verbose=verbose)
    state_result = nk_twostar_extract_states(est.params, data)
    return {
        "params": est.params,
        "estimation_result": est,
        "states": state_result["states"],
        "dates": data["dates"],
    }


if __name__ == "__main__":
    from src.models.dsge.estimation import print_single_result

    print("NK Two-Star Model - full-sample estimation")
    print("=" * 60)
    out = run_full_sample()
    print_single_result(out["estimation_result"], NK_TWOSTAR_SPEC)
