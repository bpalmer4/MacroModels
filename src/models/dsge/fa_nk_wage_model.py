"""FA-NK DSGE with sticky wages (Tier 2) and Galí (2011) unemployment.

Extends fa_nk_model.py (financial accelerator + capital) with an Erceg-Henderson-
Levin sticky-wage block, which delivers a model-consistent UNEMPLOYMENT rate via
Galí (2011): unemployment is the gap between notional labour supply and actual
employment under wage rigidity, equal to the wage markup scaled by the inverse
Frisch elasticity.

  employment  n = (y − α·k)/(1−α)              (labour demand, from production)
  real mc     mc = w + n − y                    (w is now the STICKY real-wage STATE)
  wage markup μ_w = w − (c/σ + φ_l·n)           (real wage minus the MRS)
  unemployment u = μ_w / φ_l                     (Galí 2011)
  wage PC     π_w = β·E[π_w'] − κ_w·μ_w + ε_w   (Calvo wages)
  real wage   w_t = w_{t−1} + π_w,t − π_t        (real wage is predetermined)

NAIRU U* is then recovered post-hoc as observed unemployment minus the model's
unemployment gap u.

State z = [ε_d, ε_s, ω, ε_w, R₋₁, k, n_worth, q₋₁, w,  c, q, π, π_w]
          (9 predetermined, 4 forward-looking), 5 shocks.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from src.models.dsge.fa_nk_model import FANKParameters
from src.models.dsge.nk_model import IndeterminacyError, NoSolutionError


@dataclass
class FANKWageSolution:
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    eigenvalues: np.ndarray
    Sigma: np.ndarray


@dataclass
class FANKWageModel:
    """FA-NK with sticky wages + Galí unemployment (9 states, 4 forward, 5 shocks)."""

    params: FANKParameters = field(default_factory=FANKParameters)
    n_states: int = 9
    n_shocks: int = 5
    n_forward: int = 4

    # z indices
    EPSD, EPSS, OMEGA, EPSW, RLAG, K, NW, QLAG, W, C, Q, PI, PIW = range(13)

    def _coeffs(self) -> dict:
        """mc, mpk, wage-markup μ_w as linear combos of (c, k, q, w)."""
        p = self.params
        a = 1.0 - p.alpha
        Y_C, Y_K, Y_Q = p.c_y, p.i_y, p.i_y / p.psi
        # employment n = (y − α·k)/(1−α)
        n_C, n_K, n_Q = Y_C / a, Y_K / a - p.alpha / a, Y_Q / a
        # mc = w + n − y
        mc = {"C": n_C - Y_C, "K": n_K - Y_K, "Q": n_Q - Y_Q, "W": 1.0}
        # mpk = mc + y − k  = n − k + w
        mpk = {"C": n_C, "K": n_K - 1.0, "Q": n_Q, "W": 1.0}
        # wage markup μ_w = w − (c/σ + φ_l·n)
        muw = {"C": -1.0 / p.sigma - p.phi_l * n_C, "K": -p.phi_l * n_K,
               "Q": -p.phi_l * n_Q, "W": 1.0}
        return {"Y": (Y_C, Y_K, Y_Q), "n": (n_C, n_K, n_Q), "mc": mc, "mpk": mpk, "muw": muw}

    def _build_system_matrices(self):
        p = self.params
        n = self.n_states + self.n_forward  # 13
        A = np.zeros((n, n)); B = np.zeros((n, n)); C = np.zeros((n, self.n_shocks))
        (EPSD, EPSS, OMEGA, EPSW, RLAG, K, NW, QLAG, W, Cc, Q, PI, PIW) = range(13)

        a1 = 1.0 - p.beta * (1.0 - p.delta)
        bd = p.beta * (1.0 - p.delta)
        psi = p.psi
        sc = self._coeffs()
        Y_C, Y_K, Y_Q = sc["Y"]
        mc, mpk, muw = sc["mc"], sc["mpk"], sc["muw"]
        ri, ppi, py = p.rho_i, p.phi_pi, p.phi_y

        # exogenous AR(1) shocks
        A[EPSD, EPSD] = 1.0; B[EPSD, EPSD] = p.rho_demand; C[EPSD, 0] = p.sigma_demand
        A[EPSS, EPSS] = 1.0; B[EPSS, EPSS] = p.rho_supply; C[EPSS, 1] = p.sigma_supply
        A[OMEGA, OMEGA] = 1.0; B[OMEGA, OMEGA] = p.rho_omega; C[OMEGA, 2] = p.sigma_omega
        A[EPSW, EPSW] = 1.0; B[EPSW, EPSW] = p.rho_wage; C[EPSW, 3] = p.sigma_wage

        # Taylor rule  R₋₁' = ρ_i R₋₁ + (1−ρ_i)(φ_π π + φ_y y) + ε_m
        A[RLAG, RLAG] = 1.0
        B[RLAG, RLAG] = ri
        B[RLAG, PI] = (1 - ri) * ppi
        B[RLAG, Cc] = (1 - ri) * py * Y_C
        B[RLAG, K] += (1 - ri) * py * Y_K
        B[RLAG, Q] = (1 - ri) * py * Y_Q
        C[RLAG, 4] = p.sigma_monetary

        # capital  k' = k + (δ/ψ)q
        A[K, K] = 1.0; B[K, K] = 1.0; B[K, Q] = p.delta / psi

        # net worth  nw' = κ_n nw + (1−κ_n)lev(rk − R₋₁ + π);  rk = a1·mpk + bd·q − qlag
        g = (1 - p.kappa_n) * p.lev_k
        A[NW, NW] = 1.0; B[NW, NW] = p.kappa_n
        B[NW, Cc] = g * a1 * mpk["C"]
        B[NW, K] += g * a1 * mpk["K"]
        B[NW, Q] = g * (a1 * mpk["Q"] + bd)
        B[NW, W] = g * a1 * mpk["W"]
        B[NW, QLAG] = g * (-1.0)
        B[NW, RLAG] += g * (-1.0)
        B[NW, PI] += g * (1.0)

        # qlag' = q
        A[QLAG, QLAG] = 1.0; B[QLAG, Q] = 1.0

        # real wage  w' = w + π_w' − π'   →   E[w'] − E[π_w'] + E[π'] = w
        A[W, W] = 1.0; A[W, PIW] = -1.0; A[W, PI] = 1.0; B[W, W] = 1.0

        # consumption Euler  E c' + σ E π' = c + σ R(systematic) − ε_d
        A[Cc, Cc] = 1.0; A[Cc, PI] = p.sigma
        B[Cc, Cc] = 1.0 + p.sigma * (1 - ri) * py * Y_C
        B[Cc, RLAG] = p.sigma * ri
        B[Cc, PI] = p.sigma * (1 - ri) * ppi
        B[Cc, K] += p.sigma * (1 - ri) * py * Y_K
        B[Cc, Q] = p.sigma * (1 - ri) * py * Y_Q
        B[Cc, EPSD] = -1.0

        # Q / finance Euler (return on capital = safe rate + EFP)
        A[Q, Cc] = a1 * mpk["C"]
        A[Q, Q] = a1 * mpk["Q"] + bd
        A[Q, PI] = 1.0
        A[Q, K] = a1 * mpk["K"] - p.chi
        A[Q, W] = a1 * mpk["W"]
        A[Q, NW] = p.chi
        B[Q, Q] = (1.0 + p.chi) + (1 - ri) * py * Y_Q
        B[Q, RLAG] = ri
        B[Q, PI] = (1 - ri) * ppi
        B[Q, Cc] = (1 - ri) * py * Y_C
        B[Q, K] += (1 - ri) * py * Y_K
        B[Q, OMEGA] = 1.0

        # price Phillips  β E π' = π − κ_p·mc − ε_s
        A[PI, PI] = p.beta; B[PI, PI] = 1.0
        B[PI, Cc] = -p.kappa_p * mc["C"]
        B[PI, K] += -p.kappa_p * mc["K"]
        B[PI, Q] = -p.kappa_p * mc["Q"]
        B[PI, W] = -p.kappa_p * mc["W"]
        B[PI, EPSS] = -1.0

        # wage Phillips  β E π_w' = π_w + κ_w·μ_w − ε_w
        A[PIW, PIW] = p.beta; B[PIW, PIW] = 1.0
        B[PIW, Cc] = p.kappa_w * muw["C"]
        B[PIW, K] += p.kappa_w * muw["K"]
        B[PIW, Q] = p.kappa_w * muw["Q"]
        B[PIW, W] = p.kappa_w * muw["W"]
        B[PIW, EPSW] = -1.0

        return A, B, C

    def check_determinacy(self):
        A, B, _ = self._build_system_matrices()
        _, _, alpha, beta_eig, _, _ = linalg.ordqz(B, A, sort="iuc")
        with np.errstate(divide="ignore", invalid="ignore"):
            eig = np.where(np.abs(beta_eig) < 1e-10, np.inf, alpha / beta_eig)
        return int(np.sum(np.abs(eig) > 1.0 + 1e-10)) == self.n_forward, eig

    def solve(self) -> FANKWageSolution:
        A, B, C = self._build_system_matrices()
        p = self.params
        n = self.n_states + self.n_forward
        det, eig = self.check_determinacy()
        if not det:
            nu = int(np.sum(np.abs(eig) > 1.0 + 1e-10))
            raise (IndeterminacyError if nu < self.n_forward else NoSolutionError)(
                f"{nu} unstable eigenvalues, need {self.n_forward}")
        S, T, alpha, beta_eig, Q, Z = linalg.ordqz(B, A, sort="iuc")
        with np.errstate(divide="ignore", invalid="ignore"):
            eig = np.where(np.abs(beta_eig) < 1e-10, np.inf, alpha / beta_eig)
        ns = n - self.n_forward
        Zt = Z.conj().T
        Z11, Z12 = Zt[:ns, :self.n_states], Zt[:ns, self.n_states:]
        Z21, Z22 = Zt[ns:, :self.n_states], Zt[ns:, self.n_states:]
        if np.abs(linalg.det(Z22)) < 1e-10:
            raise NoSolutionError("Z22 singular.")
        R = -linalg.inv(Z22) @ Z21
        S11, T11 = S[:ns, :ns], T[:ns, :ns]
        Z_s = Z11 + Z12 @ R
        if np.abs(linalg.det(Z_s)) < 1e-10:
            raise NoSolutionError("cannot solve P.")
        P = np.real(linalg.inv(Z_s) @ linalg.solve(T11, S11) @ Z_s)
        try:
            sf = linalg.inv(A) @ C
        except linalg.LinAlgError:
            sf = linalg.pinv(A) @ C
        Q_mat = np.real(sf[:self.n_states, :])
        Sigma = np.diag([p.sigma_demand, p.sigma_supply, p.sigma_omega, p.sigma_wage, p.sigma_monetary])
        return FANKWageSolution(P=P, Q=Q_mat, R=np.real(R), eigenvalues=eig, Sigma=Sigma)

    def derive(self, S: np.ndarray, J: np.ndarray) -> dict:
        """Reconstruct y, employment n, wage markup, unemployment gap, etc."""
        p = self.params
        sc = self._coeffs()
        Y_C, Y_K, Y_Q = sc["Y"]
        c, q, pi, piw = J[:, 0], J[:, 1], J[:, 2], J[:, 3]
        k, w = S[:, 5], S[:, 8]
        y = Y_C * c + Y_K * k + Y_Q * q
        n = sc["n"][0] * c + sc["n"][1] * k + sc["n"][2] * q
        muw = sc["muw"]["C"] * c + sc["muw"]["K"] * k + sc["muw"]["Q"] * q + sc["muw"]["W"] * w
        u_gap = muw / p.phi_l                       # Galí (2011) unemployment gap
        return {"y": y, "n": n, "w": w, "pi": pi, "pi_w": piw, "wage_markup": muw,
                "u_gap": u_gap, "R": p.rho_i * S[:, 4] + (1 - p.rho_i) * (p.phi_pi * pi + p.phi_y * y)}

    def compute_irf(self, shock: str, periods: int = 16) -> dict:
        sol = self.solve()
        idx = {"demand": 0, "supply": 1, "omega": 2, "wage": 3, "monetary": 4}[shock]
        eta = np.zeros(5); eta[idx] = sol.Sigma[idx, idx]
        s = sol.Q @ eta
        S = np.zeros((periods, 9)); J = np.zeros((periods, 4))
        for t in range(periods):
            S[t] = s; J[t] = sol.R @ s; s = sol.P @ s
        return self.derive(S, J)


# =============================================================================
# Observation, likelihood, data, estimation, U* chart
# =============================================================================

import pandas as pd  # noqa: E402

from src.models.dsge.estimation import ModelSpec  # noqa: E402
from src.models.dsge.fa_nk_model import COVID_START, COVID_END, FA_NK_PARAM_BOUNDS  # noqa: E402

N_OBS = 5  # [output_gap, inflation, cash_rate, credit_spread, wage_inflation]


def _obs_matrix(model: FANKWageModel, sol: FANKWageSolution, observe_u: bool = False) -> np.ndarray:
    p = model.params
    sc = model._coeffs()
    Y_C, Y_K, Y_Q = sc["Y"]
    mpk, muw = sc["mpk"], sc["muw"]
    a1 = 1.0 - p.beta * (1.0 - p.delta)
    bd = p.beta * (1.0 - p.delta)
    Rp = sol.R  # 4×9 : c, q, pi, pi_w
    ns = model.n_states
    EPSD, EPSS, OMEGA, EPSW, RLAG, K, NW, QLAG, W = range(9)

    def e(i):
        v = np.zeros(ns); v[i] = 1.0; return v

    c_row, q_row, pi_row, piw_row = Rp[0], Rp[1], Rp[2], Rp[3]
    y_row = Y_C * c_row + Y_K * e(K) + Y_Q * q_row
    R_row = p.rho_i * e(RLAG) + (1 - p.rho_i) * (p.phi_pi * pi_row + p.phi_y * y_row)
    knext = e(K) + (p.delta / p.psi) * q_row
    mpk_row = mpk["C"] * c_row + mpk["K"] * e(K) + mpk["Q"] * q_row + mpk["W"] * e(W)
    rk_row = a1 * mpk_row + bd * q_row - e(QLAG)
    g = (1 - p.kappa_n) * p.lev_k
    nwnext = p.kappa_n * e(NW) + g * (rk_row - e(RLAG) + pi_row)
    efp_row = p.chi * (q_row + knext - nwnext) + e(OMEGA)
    rows = [y_row, pi_row, R_row, efp_row, piw_row]
    if observe_u:
        # unemployment gap u = μ_w / φ_l (Galí 2011)
        muw_row = muw["C"] * c_row + muw["K"] * e(K) + muw["Q"] * q_row + muw["W"] * e(W)
        rows.append(muw_row / p.phi_l)
    return np.vstack(rows)


def _state_space(model, sol=None, observe_u: bool = False):
    if sol is None:
        sol = model.solve()
    Z = _obs_matrix(model, sol, observe_u=observe_u)
    h = [0.05 ** 2] * 5 + ([0.3 ** 2] if observe_u else [])  # looser m.e. on the u mapping
    return sol.P, sol.Q, Z, np.eye(model.n_shocks), np.diag(h)


def compute_wage_ll(y, params, observe_u: bool = False) -> float:
    from src.models.dsge.kalman import kalman_filter
    try:
        m = FANKWageModel(params=params)
        T, R, Z, Q, H = _state_space(m, observe_u=observe_u)
        return kalman_filter(y, T, R, Z, Q, H).log_likelihood
    except (IndeterminacyError, NoSolutionError):
        return -1e10
    except Exception:
        return -1e10


def _wage_likelihood(params, data):
    return compute_wage_ll(data["y"], params, observe_u=data.get("observe_u", False))


def wage_extract_states(params, data, observe_u: bool | None = None) -> dict:
    from src.models.dsge.kalman import kalman_smoother
    if observe_u is None:
        observe_u = data.get("observe_u", False)
    try:
        m = FANKWageModel(params=params)
        sol = m.solve()
        T, R, Z, Q, H = _state_space(m, sol, observe_u=observe_u)
        res = kalman_smoother(data["y"], T, R, Z, Q, H)
        S = res.smoothed_states
        J = S @ sol.R.T
        d = m.derive(S, J)
        df = pd.DataFrame({
            "output_gap": d["y"], "inflation": d["pi"], "wage_inflation": d["pi_w"],
            "real_wage": d["w"], "employment": d["n"], "wage_markup": d["wage_markup"],
            "u_gap": d["u_gap"], "cash_rate": d["R"],
        }, index=data["dates"])
        return {"states": df, "log_likelihood": res.log_likelihood}
    except (IndeterminacyError, NoSolutionError):
        return {"states": pd.DataFrame(index=data["dates"]), "log_likelihood": -1e10}


def load_wage_data(start="1993Q1", end=None, exclude_covid=True, observe_u=False) -> dict:
    """5 (or 6, with unemployment) observables; credit spread NaN before 2005."""
    from src.data.abs_loader import load_series
    from src.data.bonds import get_corporate_spread
    from src.data.series_specs import UNEMPLOYMENT_RATE
    from src.models.dsge.data_loader import load_estimation_data
    base = load_estimation_data(start=start, end=end, n_observables=4, anchor_inflation=True)
    spread = get_corporate_spread().data
    if not isinstance(spread.index, pd.PeriodIndex):
        spread.index = pd.PeriodIndex(spread.index, freq="Q")
    df = base.copy()
    df["credit_spread"] = spread.reindex(base.index)
    cols = ["output_gap", "inflation", "interest_rate", "credit_spread", "wage_inflation"]
    demean = ["interest_rate", "credit_spread", "wage_inflation"]
    if observe_u:
        ur = load_series(UNEMPLOYMENT_RATE).data
        if not isinstance(ur.index, pd.PeriodIndex):
            ur.index = pd.PeriodIndex(ur.index, freq="M")
        ur_q = ur.groupby(ur.index.asfreq("Q")).mean()
        df["unemployment"] = ur_q.reindex(df.index)
        cols.append("unemployment")
        demean.append("unemployment")
    if exclude_covid:
        df = df[~((df.index >= COVID_START) & (df.index <= COVID_END))]
    for col in demean:
        df[col] = df[col] - df[col].mean()
    return {"y": df[cols].to_numpy(), "dates": df.index,
            "n_observables": len(cols), "observe_u": observe_u}


WAGE_PARAM_BOUNDS = {
    **FA_NK_PARAM_BOUNDS,
    "kappa_w": (0.001, 1.0),
    "rho_wage": (0.1, 0.9),
    "sigma_wage": (0.05, 3.0),
}

WAGE_SPEC = ModelSpec(
    name="FA-NK-wage",
    description="FA-NK + sticky wages + Galí unemployment",
    param_class=FANKParameters,
    param_bounds=WAGE_PARAM_BOUNDS,
    estimate_params=[
        "chi", "kappa_p", "kappa_w", "phi_pi", "phi_y", "rho_i",
        "rho_demand", "rho_supply", "rho_omega", "rho_wage",
        "sigma_demand", "sigma_supply", "sigma_omega", "sigma_wage", "sigma_monetary",
    ],
    fixed_params={
        "beta": 0.99, "delta": 0.025, "sigma": 1.0, "xi": 1.0, "psi": 4.0,
        "c_y": 0.8, "i_y": 0.2, "lev_k": 2.0, "kappa_n": 0.95, "alpha": 0.33, "phi_l": 1.0,
    },
    likelihood_fn=_wage_likelihood,
    state_extractor_fn=wage_extract_states,
)


def run_wage(start="1993Q1", end=None, verbose=True, observe_u=False) -> dict:
    from src.models.dsge.estimation import estimate_model
    data = load_wage_data(start=start, end=end, observe_u=observe_u)
    if verbose:
        print(f"FA-NK-wage estimation ({data['n_observables']} obs): "
              f"{data['dates'][0]} to {data['dates'][-1]} (n={len(data['dates'])})")
    est = estimate_model(WAGE_SPEC, data, verbose=verbose)
    states = wage_extract_states(est.params, data)["states"]
    return {"params": est.params, "estimation_result": est, "states": states,
            "dates": data["dates"], "observe_u": observe_u}


def produce_ustar(out=None, observe_u=False) -> None:
    """Estimate (if needed) and chart actual unemployment vs the model NAIRU U*."""
    from pathlib import Path
    import mgplot as mg
    from src.data.abs_loader import load_series
    from src.data.series_specs import UNEMPLOYMENT_RATE

    if out is None:
        out = run_wage(verbose=False, observe_u=observe_u)
    st = out["states"]
    observe_u = out.get("observe_u", observe_u)
    # observed unemployment rate, quarterly
    ur = load_series(UNEMPLOYMENT_RATE).data
    if not isinstance(ur.index, pd.PeriodIndex):
        ur.index = pd.PeriodIndex(ur.index, freq="M")
    ur_q = ur.groupby(ur.index.asfreq("Q")).mean()  # monthly periods → quarterly average
    U = ur_q.reindex(st.index)
    ustar = U - st["u_gap"]   # NAIRU = actual − unemployment gap (u_gap = μ_w/φ)

    full = pd.period_range(st.index.min(), st.index.max(), freq="Q")
    chart = pd.DataFrame({"Unemployment rate": U, "NAIRU U* (FA-NK-wage)": ustar}).reindex(full)

    root = Path(__file__).parent.parent.parent.parent
    mg.set_chart_dir(str(root / "charts" / "dsge-fa-nk"))
    tag = "uobs" if observe_u else "free"
    title = ("FA-NK-wage NAIRU U-star (unemployment observed)" if observe_u
             else "FA-NK-wage unemployment and NAIRU U-star")
    mg.line_plot_finalise(
        chart, width=[2, 2], color=["black", "firebrick"], dropna=False,
        title=title, tag=tag, ylabel="Per cent",
        legend={"loc": "best"}, rfooter="FA-NK-wage DSGE",
        lfooter="Australia. COVID excluded. U* = actual − Gali (2011) unemployment gap. ",
    )
    print(f"  U* chart written; recent U*={ustar.dropna().iloc[-1]:.1f}%  (U={U.dropna().iloc[-1]:.1f}%)")


if __name__ == "__main__":
    m = FANKWageModel()
    det, eig = m.check_determinacy()
    nu = int(np.sum(np.abs(eig) > 1.0 + 1e-10))
    print(f"determinate = {det}   unstable = {nu} (need {m.n_forward})")
    if det:
        print("\nIRF to a wage-markup shock — expect unemployment UP, wage inflation DOWN:")
        irf = m.compute_irf("wage", periods=10)
        for v in ["wage_markup", "u_gap", "pi_w", "pi", "y"]:
            print(f"  {v:12s}: {np.round(irf[v][:8], 4)}")
        print("\nIRF to a monetary tightening — expect u UP, y DOWN, pi DOWN:")
        irf = m.compute_irf("monetary", periods=10)
        for v in ["u_gap", "y", "pi", "pi_w"]:
            print(f"  {v:12s}: {np.round(irf[v][:8], 4)}")
