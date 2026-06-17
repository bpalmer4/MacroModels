"""Bayesian re-estimation of the FA-NK DSGE family (Taylor-block priors).

Motivation
----------
Under MLE (`estimation.py`, L-BFGS-B + hard bounds) the FA-NK Taylor block is
weakly identified: phi_pi pegs at its upper cap in nearly every spec, and only
comes off the bound when unemployment is observed. The bounds are acting as
de-facto priors. This module makes the priors EXPLICIT and samples the posterior
gradient-free, reusing the existing model maths via `ModelSpec.likelihood_fn`.

The headline deliverable is the phi_pi posterior: does it concentrate at an
interior value (the policy block is identified) or merely track its prior (the
AU sample carries no information about it)? Both answers are reportable.

Design
------
- `DSGELogLike` wraps the numpy `spec.likelihood_fn(params, data)` (Blanchard-Kahn
  solve + Kalman filter) as a black-box pytensor Op. Indeterminate draws already
  return -1e10 from the likelihood, which the sampler rejects. No gradient, so
  sampling is gradient-free (DEMetropolis-Z).
- `PRIOR_SPECS` is a name -> (distribution, kwargs) registry shared by both models;
  each model uses the subset matching its `spec.estimate_params`. Priors are bounded
  by construction (Beta in (0,1), HalfNormal>0, TruncatedNormal), so they REPLACE the
  MLE hard bounds rather than stacking on them.

Run:
    uv run python -m src.models.dsge.fa_nk_bayes          # full run, both models
    uv run python -m src.models.dsge.fa_nk_bayes --smoke  # quick wiring check
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytensor.tensor as pt
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op

from src.models.dsge.estimation import ModelSpec

# =============================================================================
# Black-box likelihood Op
# =============================================================================


class DSGELogLike(Op):
    """Wrap a numpy DSGE log-likelihood as a gradient-free pytensor Op.

    Input: theta, a 1-D vector of the estimated parameters in `spec.estimate_params`
    order. Output: scalar log-likelihood. Non-finite / indeterminate draws collapse
    to -1e10 (a finite barrier the sampler rejects).
    """

    __props__ = ()

    def __init__(self, spec: ModelSpec, data: dict) -> None:
        self.spec = spec
        self.data = data
        self.names = list(spec.estimate_params)
        # Full parameter dict: class defaults overlaid with the model's fixed params.
        base = spec.param_class().to_dict()
        base.update(spec.fixed_params)
        self.base = base

    def make_node(self, theta) -> Apply:  # noqa: ANN001
        theta = pt.as_tensor_variable(theta)
        return Apply(self, [theta], [pt.scalar(dtype="float64")])

    def perform(self, node, inputs, outputs) -> None:  # noqa: ANN001
        (theta,) = inputs
        pdict = dict(self.base)
        for name, value in zip(self.names, np.asarray(theta), strict=True):
            pdict[name] = float(value)
        params = self.spec.param_class(**pdict)
        ll = self.spec.likelihood_fn(params, self.data)
        if not np.isfinite(ll):
            ll = -1e10
        outputs[0][0] = np.asarray(float(ll))


# =============================================================================
# Prior registry (shared across both models; subset chosen per spec)
# =============================================================================

# (distribution name on pm, kwargs). phi_pi is the informative Taylor prior; the
# rest are weakly informative and bounded so they replace the MLE hard caps.
PRIOR_SPECS: dict[str, tuple[str, dict]] = {
    # Taylor rule -- the block MLE cannot pin
    "phi_pi": ("TruncatedNormal", {"mu": 1.5, "sigma": 0.25, "lower": 1.0}),
    "phi_y": ("TruncatedNormal", {"mu": 0.25, "sigma": 0.15, "lower": 0.0}),
    "rho_i": ("Beta", {"alpha": 8.0, "beta": 2.0}),          # ~0.8 policy smoothing
    # Financial accelerator (MLE-stable near ~0.015 -- weak prior, should be data-driven)
    "chi": ("HalfNormal", {"sigma": 0.1}),
    # Nominal slopes
    "kappa_p": ("Gamma", {"alpha": 2.0, "beta": 1.0}),       # mean 2, wide
    "kappa_w": ("HalfNormal", {"sigma": 0.3}),
    # Shock persistences (financial shock more persistent)
    "rho_demand": ("Beta", {"alpha": 3.0, "beta": 2.0}),
    "rho_supply": ("Beta", {"alpha": 3.0, "beta": 2.0}),
    "rho_omega": ("Beta", {"alpha": 8.0, "beta": 1.5}),
    "rho_wage": ("Beta", {"alpha": 3.0, "beta": 2.0}),
    # Shock volatilities (deviation space)
    "sigma_demand": ("HalfNormal", {"sigma": 1.0}),
    "sigma_supply": ("HalfNormal", {"sigma": 1.0}),
    "sigma_omega": ("HalfNormal", {"sigma": 1.0}),
    "sigma_monetary": ("HalfNormal", {"sigma": 1.0}),
    "sigma_wage": ("HalfNormal", {"sigma": 1.0}),
}


def _build_prior(name: str):  # noqa: ANN202 -- returns a pymc RV
    import pymc as pm

    if name not in PRIOR_SPECS:
        raise KeyError(f"No prior defined for estimated parameter '{name}'")
    dist_name, kwargs = PRIOR_SPECS[name]
    return getattr(pm, dist_name)(name, **kwargs)


# =============================================================================
# Sampling
# =============================================================================


def run_bayes(
    spec: ModelSpec,
    data: dict,
    draws: int = 5000,
    tune: int = 8000,
    chains: int = 4,
    seed: int = 12345,
    verbose: bool = True,
):  # noqa: ANN201 -- returns arviz.InferenceData
    """Sample the posterior of `spec`'s estimated parameters via DEMetropolis-Z.

    Returns an arviz InferenceData with `posterior` and `prior` groups.
    """
    import pymc as pm

    names = list(spec.estimate_params)
    missing = [n for n in names if n not in PRIOR_SPECS]
    if missing:
        raise KeyError(f"Priors missing for: {missing}")

    if verbose:
        print(f"\n{spec.name} -- Bayesian estimation (DEMetropolis-Z)")
        print(f"  params: {names}")
        print(f"  draws={draws} tune={tune} chains={chains}")

    loglike = DSGELogLike(spec, data)

    with pm.Model() as model:  # noqa: F841
        rvs = [_build_prior(n) for n in names]
        theta = pt.stack(rvs)
        pm.Potential("loglike", loglike(theta))

        # Prior draws (for prior-vs-posterior contrast); cheap, no likelihood eval.
        prior = pm.sample_prior_predictive(draws=2000, random_seed=seed, var_names=names)

        # Gradient-free sampler: cores=1 avoids pickling the black-box Op across
        # processes (chains run sequentially, still gives r_hat across chains).
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            step=pm.DEMetropolisZ(),
            random_seed=seed,
            progressbar=verbose,
            compute_convergence_checks=True,
        )

    idata.extend(prior)
    return idata


# =============================================================================
# Diagnostics and outputs
# =============================================================================


def _posterior_contraction(idata, names: list[str]) -> pd.DataFrame:
    """1 - Var(posterior)/Var(prior) per parameter. ~0 => data uninformative
    (posterior tracks prior); ~1 => sharply identified by the data."""
    post = idata.posterior
    pri = idata.prior
    rows = {}
    for n in names:
        v_post = float(post[n].var())
        v_pri = float(pri[n].var())
        contraction = 1.0 - v_post / v_pri if v_pri > 0 else np.nan
        rows[n] = {
            "post_mean": float(post[n].mean()),
            "post_sd": float(np.sqrt(v_post)),
            "prior_sd": float(np.sqrt(v_pri)),
            "contraction": contraction,
        }
    return pd.DataFrame(rows).T


def produce_bayes_outputs(idata, spec: ModelSpec, tag: str) -> None:
    """Write summary, contraction table, netCDF, and prior-vs-posterior charts.

    The density and forest charts are not time series, so their CONTENT is drawn
    on an Axes and then finalised through mgplot's `finalise_plot` (consistent
    titles / footers / save / close), per the project's charting convention.
    """
    import arviz as az
    import matplotlib.pyplot as plt
    import mgplot as mg
    from scipy.stats import gaussian_kde

    root = Path(__file__).parent.parent.parent.parent
    chart_dir = root / "charts" / "dsge-fa-nk-bayes"
    out_dir = root / "model_outputs"
    chart_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(spec.estimate_params)
    summary = az.summary(idata, var_names=names, hdi_prob=0.94)
    contraction = _posterior_contraction(idata, names)

    # Merge convergence (r_hat, ess) with contraction into one table.
    table = summary.join(contraction[["prior_sd", "contraction"]])
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())

    key = ", ".join(
        f"{p}={contraction.loc[p, 'post_mean']:.3f} (contraction {contraction.loc[p, 'contraction']:.2f})"
        for p in ("phi_pi", "phi_y") if p in contraction.index
    )
    lines = [
        f"FA-NK Bayesian re-estimation -- {spec.name} [{tag}]",
        f"sampler: DEMetropolis-Z; max r_hat={max_rhat:.3f}, min ess_bulk={min_ess:.0f}",
        f"Taylor block: {key}",
        "",
        table.to_string(),
    ]
    txt_path = out_dir / f"fa_nk_bayes_params_{tag}.txt"
    txt_path.write_text("\n".join(lines))

    try:
        idata.to_netcdf(str(out_dir / f"fa_nk_bayes_{tag}.nc"), engine="h5netcdf")
    except Exception as exc:  # noqa: BLE001 -- saving idata is optional
        print(f"  (idata netCDF not saved: {exc})")

    mg.set_chart_dir(str(chart_dir))
    RF = "FA-NK Bayes"
    LF = "Australia. DEMetropolis-Z posterior. "

    # Prior-vs-posterior overlay for the Taylor block -- the chart that answers the
    # identification question. KDE content on an Axes; finalised via mgplot.
    for param in ("phi_pi", "phi_y"):
        if param not in names:
            continue
        pri = idata.prior[param].to_numpy().ravel()
        pos = idata.posterior[param].to_numpy().ravel()
        grid = np.linspace(min(pri.min(), pos.min()), max(pri.max(), pos.max()), 256)
        d_pos = gaussian_kde(pos)(grid)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(grid, gaussian_kde(pri)(grid), color="grey", linestyle="--", linewidth=2, label="Prior")
        ax.plot(grid, d_pos, color="navy", linewidth=2.5, label="Posterior")
        ax.fill_between(grid, d_pos, color="navy", alpha=0.12)
        mg.finalise_plot(
            ax, title=f"{spec.name} {param} prior vs posterior", xlabel=param,
            ylabel="Density", legend={"loc": "best"}, rfooter=RF,
            lfooter=LF + "Posterior shifted off prior => identified by the data. ",
            tag=tag, show=False,
        )

    # Posterior 94% HDI ranges for every parameter (forest-style convergence view).
    # mgplot has no forest primitive, so errorbar content + mgplot finalise.
    y = np.arange(len(names))
    means = summary["mean"].to_numpy()
    lo94, hi94 = summary["hdi_3%"].to_numpy(), summary["hdi_97%"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.42 * len(names))))
    ax.errorbar(means, y, xerr=[means - lo94, hi94 - means], fmt="o", color="navy",
                ecolor="grey", capsize=3, lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=0.6)
    mg.finalise_plot(
        ax, title=f"{spec.name} posterior 94pc HDI", xlabel="Parameter value",
        rfooter=RF, lfooter=LF + f"max r_hat={max_rhat:.3f}. ", tag=tag, show=False,
    )

    print(f"\n[outputs] {spec.name} [{tag}]")
    print(f"  {txt_path}")
    print(f"  {chart_dir}/fa-nk-bayes-{tag}-*.png")
    print(f"  max r_hat={max_rhat:.3f}  min ess_bulk={min_ess:.0f}")
    print(f"  Taylor block -> {key}")


# =============================================================================
# Convenience runners
# =============================================================================


def run_fa_nk_bayes(smoke: bool = False, **kw):  # noqa: ANN201
    from src.models.dsge.fa_nk_model import FA_NK_SPEC, load_fa_nk_data

    data = load_fa_nk_data()
    cfg = {"draws": 500, "tune": 500, "chains": 2} if smoke else {}
    idata = run_bayes(FA_NK_SPEC, data, **{**cfg, **kw})
    produce_bayes_outputs(idata, FA_NK_SPEC, tag="fa_nk")
    return idata


def run_fa_nk_wage_bayes(smoke: bool = False, observe_u: bool = False, **kw):  # noqa: ANN201
    from src.models.dsge.fa_nk_wage_model import WAGE_SPEC, load_wage_data

    data = load_wage_data(observe_u=observe_u)
    cfg = {"draws": 500, "tune": 500, "chains": 2} if smoke else {}
    idata = run_bayes(WAGE_SPEC, data, **{**cfg, **kw})
    produce_bayes_outputs(idata, WAGE_SPEC, tag="wage_uobs" if observe_u else "wage")
    return idata


# =============================================================================
# Posterior state extraction (smoothed r*, EFP etc. WITH parameter uncertainty)
# =============================================================================


def _fa_nk_states_for_params(params, data) -> pd.DataFrame:
    """Smoothed latent series for one FA-NK parameter draw.

    Mirrors `fa_nk_model.fa_nk_extract_states` but also returns the natural rate
    r^n = smoothed eps_d / sigma (state 0). Deviation-space series; rates are
    re-levelled to per cent by the caller.
    """
    from src.models.dsge.fa_nk_model import FANKModel, _fa_state_space
    from src.models.dsge.kalman import kalman_smoother

    model = FANKModel(params=params)
    sol = model.solve()
    T, R, Z, Q, H = _fa_state_space(model, sol)
    res = kalman_smoother(data["y"], T, R, Z, Q, H)
    s = res.smoothed_states
    jumps = s @ sol.R.T
    d = model.derive(s, jumps)
    kn, nn = s[:, 4], s[:, 5]
    q, om = jumps[:, 1], s[:, 2]
    efp = np.full(len(s), np.nan)
    efp[:-1] = params.chi * (q[:-1] + kn[1:] - nn[1:]) + om[:-1]
    return pd.DataFrame(
        {
            "r_safe": d["R"] - d["pi"],
            "r_capital": d["rk"],
            "wedge_efp": efp,
            "r_natural": s[:, 0] / params.sigma,
            "output_gap": d["y"],
        },
        index=data["dates"],
    )


def extract_states_posterior(spec, data, idata, n_draws: int = 400, seed: int = 7):  # noqa: ANN201
    """Run the Kalman smoother over `n_draws` posterior parameter draws.

    Returns a dict: series name -> DataFrame with columns [median, lo, hi, mean]
    (16/84 credible band across the parameter posterior), plus `_n_draws`.
    Indeterminate draws are skipped. This is the band the MLE byproduct charts
    could NOT produce: it folds in PARAMETER uncertainty, not just state-filtering
    uncertainty.
    """
    base = spec.param_class().to_dict()
    base.update(spec.fixed_params)
    names = list(spec.estimate_params)

    post = idata.posterior
    total = post.sizes["chain"] * post.sizes["draw"]
    flat = {n: post[n].to_numpy().reshape(-1) for n in names}
    rng = np.random.default_rng(seed)
    pick = rng.choice(total, size=min(n_draws, total), replace=False)

    dates = data["dates"]
    stacks: dict[str, list[np.ndarray]] = {}
    n_ok = 0
    for i in pick:
        pdict = dict(base)
        for n in names:
            pdict[n] = float(flat[n][i])
        params = spec.param_class(**pdict)
        try:
            df = _fa_nk_states_for_params(params, data)
        except Exception:  # noqa: BLE001 -- skip indeterminate / failed draws
            continue
        n_ok += 1
        for col in df.columns:
            stacks.setdefault(col, []).append(df[col].to_numpy())

    out: dict = {}
    for col, arrs in stacks.items():
        A = np.vstack(arrs)  # (n_ok, T)
        out[col] = pd.DataFrame(
            {
                "median": np.nanmedian(A, axis=0),
                "lo": np.nanpercentile(A, 16, axis=0),
                "hi": np.nanpercentile(A, 84, axis=0),
                "mean": np.nanmean(A, axis=0),
            },
            index=dates,
        )
    out["_n_draws"] = n_ok
    return out


def produce_fa_nk_extractions(idata=None, n_draws: int = 400) -> None:  # noqa: ANN001
    """Charts of smoothed r*, the two rates and the EFP wedge WITH the posterior
    parameter-uncertainty band. Loads the saved FA-NK InferenceData if not given.
    Bands are layered with mgplot (`fill_between_plot` + `line_plot`) and closed
    out with `finalise_plot`.
    """
    import arviz as az
    import mgplot as mg

    from src.models.dsge.fa_nk_model import FA_NK_SPEC, _mean_real_rate, load_fa_nk_data

    root = Path(__file__).parent.parent.parent.parent
    chart_dir = root / "charts" / "dsge-fa-nk-bayes"
    out_dir = root / "model_outputs"

    if idata is None:
        idata = az.from_netcdf(str(out_dir / "fa_nk_bayes_fa_nk.nc"))

    data = load_fa_nk_data()
    bands = extract_states_posterior(FA_NK_SPEC, data, idata, n_draws=n_draws)
    n_ok = bands.pop("_n_draws")
    dates = data["dates"]
    mrr = _mean_real_rate(dates)
    full = pd.period_range(dates.min(), dates.max(), freq="Q")  # render COVID gap as a break

    def relevel(name: str, add: float) -> pd.DataFrame:
        b = bands[name]
        return (pd.DataFrame({"median": b["median"] + add, "lo": b["lo"] + add, "hi": b["hi"] + add})
                .set_index(dates).reindex(full))

    mg.set_chart_dir(str(chart_dir))
    RF = "FA-NK Bayes"
    LF = f"Australia. COVID excluded. Band = 16-84% posterior (n={n_ok} draws, parameter uncertainty). "

    # 1. Natural rate r* with FULL parameter-uncertainty band (the key deliverable)
    rstar = relevel("r_natural", mrr)
    ax = mg.fill_between_plot(rstar[["lo", "hi"]], color="navy", alpha=0.15, label="16-84% (parameter)")
    mg.line_plot(rstar[["median"]].rename(columns={"median": "r* (natural rate)"}),
                 ax=ax, color=["navy"], width=2.5, annotate=False, dropna=False)
    mg.finalise_plot(
        ax, title="FA-NK Bayesian natural rate r-star", ylabel="Per cent (annual)",
        y0=True, legend={"loc": "best"}, rfooter=RF,
        lfooter=LF + f"Levels = deviation + sample-mean real rate ({mrr:.1f}%). ", show=False,
    )

    # 2. Safe rate vs cost of capital (the divergence), with the r_safe band
    safe = relevel("r_safe", mrr)
    cap = (pd.DataFrame({"v": bands["r_safe"]["median"] + bands["wedge_efp"]["median"] + mrr})
           .set_index(dates).reindex(full))
    ax = mg.fill_between_plot(safe[["lo", "hi"]], color="navy", alpha=0.12, label="Safe rate 16-84%")
    mg.line_plot(
        pd.DataFrame({"Safe real rate (R-pi)": safe["median"],
                      "Cost of capital (= safe + EFP)": cap["v"]}),
        ax=ax, color=["navy", "firebrick"], width=[2.5, 2], annotate=False, dropna=False,
    )
    mg.finalise_plot(
        ax, title="FA-NK Bayesian safe rate vs cost of capital", ylabel="Per cent (annual)",
        legend={"loc": "best"}, rfooter=RF, lfooter=LF, show=False,
    )

    # 3. EFP wedge with band (deviation space)
    efp = relevel("wedge_efp", 0.0)
    ax = mg.fill_between_plot(efp[["lo", "hi"]], color="darkgreen", alpha=0.15, label="16-84%")
    mg.line_plot(efp[["median"]].rename(columns={"median": "EFP wedge"}),
                 ax=ax, color=["darkgreen"], width=2.5, annotate=False, dropna=False)
    mg.finalise_plot(
        ax, title="FA-NK Bayesian external finance premium wedge", ylabel="Deviation (ppt)",
        y0=True, legend={"loc": "best"}, rfooter=RF, lfooter=LF, show=False,
    )

    pd.concat({k: bands[k] for k in bands}, axis=1).to_csv(out_dir / "fa_nk_bayes_states.csv")
    print(f"\n[extractions] {n_ok} posterior draws smoothed")
    print(f"  {out_dir}/fa_nk_bayes_states.csv")
    print(f"  {chart_dir}/  (3 banded charts: r-star, two-rates, EFP wedge)")


if __name__ == "__main__":
    import sys

    smoke = "--smoke" in sys.argv
    print("=" * 64)
    print(f"FA-NK BAYESIAN RE-ESTIMATION{'  (smoke test)' if smoke else ''}")
    print("=" * 64)

    if "--extract-only" in sys.argv:
        produce_fa_nk_extractions()
    else:
        run_fa_nk_bayes(smoke=smoke)
        run_fa_nk_wage_bayes(smoke=smoke)
        produce_fa_nk_extractions()
