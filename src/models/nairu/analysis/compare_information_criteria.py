"""LOO / WAIC model comparison for NAIRU variants.

The saved traces do not carry a ``log_likelihood`` group, so this module
rebuilds each model from its saved config + observations and evaluates the
log-likelihood at the existing posterior draws via
``pm.compute_log_likelihood`` (no re-sampling — the comparison is faithful to
the exact traces on disk). The slim log-likelihood group is cached to
``model_outputs/nairu_<variant>_loglik.nc`` so re-runs are fast.

Pooling
-------
Each variant has several observation equations, so the log_likelihood group
holds several arrays (price, Okun, ΔULC, IS, HCOE, ...). For a *joint* LOO/WAIC
every (equation, time) contribution is stacked into one exchangeable
observation vector and the criterion is computed over that pooled vector. The
cached posterior is not retained, so PSIS-LOO's relative-ESS (``reff``) is
computed from the log-likelihood's own ESS and the comparison table (including
the SE of each pairwise difference, ``dse``) is assembled by hand.

Comparability caveat
--------------------
LOO/WAIC are only comparable across models conditioned on the *same* observed
data. The ``simple*`` family shares the same five observation equations over the
same 167 quarters → a pooled comparison is valid. ``complex`` adds five more
likelihood terms and ends one quarter earlier, so its pooled ELPD is NOT
comparable to the simple family. Every variant shares the price Phillips curve
(``observed_price_inflation``) — the equation the excess-expectations term
modifies — so the simple family is also compared on that term alone, and all
variants are reported per-observation on it for reference.

Run:
    uv run python -m src.models.nairu.analysis.compare_information_criteria
"""

import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from src.models.nairu.results import DEFAULT_OUTPUT_DIR, load_results

warnings.filterwarnings("ignore")

# 5-observation-equation models on the target anchor, full sample (n=167):
# joint LOO/WAIC is valid across these.
JOINT_FAMILY = [
    "nairu_simple_target",
    "nairu_simple_regime_target",
    "nairu_simple_excess_target",
    "nairu_simple_excess_regime_target",
    "nairu_simple_excess_studentt_target",
    "nairu_simple_excess_regime_studentt_target",
    "nairu_simple_excess_tprice_target",
    "nairu_simple_excess_regime_tprice_target",
]
# Price-equation reference adds models that are NOT joint-comparable:
# nohcoe has 4 equations; complex has 10 and ends a quarter earlier.
PRICE_REFERENCE = [*JOINT_FAMILY, "nairu_simple_excess_nohcoe_target", "nairu_complex_target"]
PRICE_LL = "observed_price_inflation"
THIN = 10  # 50k draws -> 5k; ample for stable LOO/WAIC, keeps memory sane


def _loglik_dataset(prefix: str) -> xr.Dataset:
    """Return the log_likelihood group for a variant, computing+caching as needed."""
    cache = Path(DEFAULT_OUTPUT_DIR) / f"{prefix}_loglik.nc"
    if cache.exists():
        return xr.open_dataset(cache).isel(draw=slice(None, None, THIN))
    print(f"Computing log-likelihood: {prefix}")
    res = load_results(prefix=prefix, rebuild_model=True)
    with res.model:
        idata = pm.compute_log_likelihood(res.trace, extend_inferencedata=True, progressbar=False)
    idata.log_likelihood.to_netcdf(cache)
    return idata.log_likelihood.isel(draw=slice(None, None, THIN))


def _pool(ll: xr.Dataset, names: list[str] | None = None) -> az.InferenceData:
    """Stack chosen equations' (equation, time) contributions into one obs vector."""
    names = names or list(ll.data_vars)
    parts = []
    for n in names:
        da = ll[n]
        extra = [d for d in da.dims if d not in ("chain", "draw")]
        st = da.stack(__o=extra) if extra else da.expand_dims("__o")
        parts.append(st.reset_index("__o", drop=True))
    cat = xr.concat(parts, dim="__o").rename({"__o": "obs_id"})
    return az.InferenceData(log_likelihood=xr.Dataset({"loglik": cat}))


def _loo(idata: az.InferenceData) -> az.ELPDData:
    """PSIS-LOO with reff derived from the log-likelihood's own ESS."""
    reff = float(az.ess(idata.log_likelihood, var_names=["loglik"], relative=True)["loglik"].mean())
    return az.loo(idata, pointwise=True, reff=reff, scale="log")


def _pareto_k_table(loos: dict[str, az.ELPDData]) -> pd.DataFrame:
    """Summarise PSIS-LOO Pareto-k reliability per model.

    k <= 0.7 is reliable; 0.7 < k <= 1 is unreliable for that point; k > 1 means
    the LOO estimate for that observation cannot be trusted. A handful of points
    above 0.7 in a long pooled vector is usually fine; many (or any k > 1) means
    the elpd/ranking for that model should be treated with caution.
    """
    rows = {}
    for name, e in loos.items():
        k = e.pareto_k.values.ravel()
        n = len(k)
        rows[name] = {
            "n": n,
            "good(<=0.5)": int((k <= 0.5).sum()),
            "ok(0.5-0.7)": int(((k > 0.5) & (k <= 0.7)).sum()),
            "bad(0.7-1)": int(((k > 0.7) & (k <= 1.0)).sum()),
            "vbad(>1)": int((k > 1.0).sum()),
            "%>0.7": round(100 * (k > 0.7).mean(), 1),
            "max_k": round(float(k.max()), 2),
        }
    return pd.DataFrame(rows).T


def _ic_table(idatas: dict[str, az.ELPDData], ic: str) -> pd.DataFrame:
    """Build a ranked comparison table (elpd, p, se, dse vs best) for loo or waic."""
    elpd_attr = f"elpd_{ic}"
    pointwise_attr = f"{ic}_i"
    rows = {}
    for name, e in idatas.items():
        rows[name] = {
            elpd_attr: float(getattr(e, elpd_attr)),
            f"p_{ic}": float(getattr(e, f"p_{ic}")),
            "se": float(e.se),
            "_pw": getattr(e, pointwise_attr).values.ravel(),
        }
    df = pd.DataFrame(rows).T.sort_values(elpd_attr, ascending=False)
    best = df.index[0]
    best_pw = rows[best]["_pw"]
    n = len(best_pw)
    df["delta"] = df[elpd_attr] - df.loc[best, elpd_attr]
    df["dse"] = [0.0 if name == best else np.sqrt(n) * (best_pw - rows[name]["_pw"]).std()
                 for name in df.index]
    return df[[elpd_attr, f"p_{ic}", "se", "delta", "dse"]].round(2)


def main() -> None:
    ll = {name: _loglik_dataset(name) for name in PRICE_REFERENCE}

    # --- 1. Joint (pooled over all equations) — 5-equation family only -----
    print("\n" + "#" * 72)
    print("# JOINT LOO/WAIC — 5-equation family, all equations pooled (valid)")
    print("# higher elpd = better; delta vs best; dse = SE of that difference")
    print("#" * 72)
    joint_loo = {n: _loo(_pool(ll[n])) for n in JOINT_FAMILY}
    print("\n--- LOO (joint) ---")
    print(_ic_table(joint_loo, "loo").to_string())
    print("\n--- WAIC (joint) ---")
    print(_ic_table({n: az.waic(_pool(ll[n]), pointwise=True, scale="log") for n in JOINT_FAMILY},
                    "waic").to_string())
    print("\n--- Pareto-k reliability (joint LOO) ---")
    print(_pareto_k_table(joint_loo).to_string())

    # --- 2. Price equation only — 5-equation family (same n) --------------
    print("\n\n" + "#" * 72)
    print(f"# PRICE-EQUATION LOO/WAIC — 5-equation family ({PRICE_LL})")
    print("#" * 72)
    price_loo = {n: _loo(_pool(ll[n], [PRICE_LL])) for n in JOINT_FAMILY}
    print("\n--- LOO (price eq) ---")
    print(_ic_table(price_loo, "loo").to_string())
    print("\n--- WAIC (price eq) ---")
    print(_ic_table({n: az.waic(_pool(ll[n], [PRICE_LL]), pointwise=True, scale="log")
                     for n in JOINT_FAMILY}, "waic").to_string())
    print("\n--- Pareto-k reliability (price-eq LOO) ---")
    print(_pareto_k_table(price_loo).to_string())

    # --- 3. Price-eq per-observation across ALL variants (reference) ------
    print("\n\n" + "#" * 72)
    print("# PRICE-EQ per-observation elpd_loo — all variants (NOT differenced)")
    print("#" * 72)
    all_loo = {n: _loo(_pool(ll[n], [PRICE_LL])) for n in PRICE_REFERENCE}
    for n, e in all_loo.items():
        nn = len(e.loo_i.values.ravel())
        print(f"  {n:36s} elpd_loo={e.elpd_loo:8.1f}  per-obs={e.elpd_loo / nn:7.4f}  n={nn}")
    print("\n--- Pareto-k reliability (price-eq LOO, all variants) ---")
    print(_pareto_k_table(all_loo).to_string())


if __name__ == "__main__":
    main()
