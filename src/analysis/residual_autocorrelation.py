"""Residual autocorrelation analysis for model validation."""

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox


def _place_model_name(model_name: str, kwargs: dict) -> dict:
    """Place model_name in first available footer/header slot."""
    for slot in ["rfooter", "rheader", "lheader"]:
        if slot not in kwargs:
            return {slot: model_name}
    return {}


def residual_autocorrelation_analysis(
    ppc: az.InferenceData,
    obs_vars: dict[str, np.ndarray],
    obs_index: pd.Index,
    var_labels: dict[str, str] | None = None,
    max_lags: int = 20,
    model_name: str = "Model",
    **kwargs,
) -> None:
    """Analyze residual autocorrelation for model validation."""
    if var_labels is None:
        var_labels = {k: k for k in obs_vars}

    for var_name, observed_data in obs_vars.items():
        ppc_samples = ppc.posterior_predictive[var_name].values
        ppc_flat = ppc_samples.reshape(-1, ppc_samples.shape[-1])
        ppc_mean = ppc_flat.mean(axis=0)

        residuals = observed_data - ppc_mean
        residuals_series = pd.Series(residuals, index=obs_index, name="Residuals")

        std_band = pd.DataFrame(
            {
                "lower": np.full(len(obs_index), -2 * residuals.std()),
                "upper": np.full(len(obs_index), 2 * residuals.std()),
            },
            index=obs_index,
        )
        ax = mg.fill_between_plot(std_band, color="grey", alpha=0.1, label="±2σ")
        mg.line_plot(residuals_series, ax=ax, color="steelblue", width=0.8)

        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        p_value = lb_test["lb_pvalue"].values[0]
        status = "OK" if p_value > 0.05 else "AUTOCORRELATED"

        label = var_labels.get(var_name, var_name)
        defaults = {
            "title": f"Residuals - {label}",
            "ylabel": "Residual",
            "lfooter": f"Ljung-Box test (lag 10): p={p_value:.4f} ({status})",
            "y0": True,
            **_place_model_name(model_name, kwargs),
        }
        for key in list(defaults.keys()):
            if key in kwargs:
                defaults.pop(key)

        mg.finalise_plot(ax, **defaults, **kwargs)

        # ACF plot
        n = len(residuals)
        acf_vals = np.correlate(
            residuals - residuals.mean(), residuals - residuals.mean(), mode="full"
        )
        acf_vals = acf_vals[n - 1 :] / acf_vals[n - 1]
        acf_series = pd.Series(
            acf_vals[: max_lags + 1], index=range(max_lags + 1), name="ACF"
        )

        conf_bound = 1.96 / np.sqrt(n)
        conf_band = pd.DataFrame(
            {
                "lower": np.full(max_lags + 1, -conf_bound),
                "upper": np.full(max_lags + 1, conf_bound),
            },
            index=range(max_lags + 1),
        )

        ax = mg.fill_between_plot(conf_band, color="red", alpha=0.1, label="95% CI")
        mg.line_plot(acf_series, ax=ax, color="steelblue", width=1.5)

        acf_defaults = {
            "title": f"Autocorrelation Function - {label}",
            "ylabel": "ACF",
            "xlabel": "Lag",
            "lfooter": "Red band: 95% confidence bounds for white noise.",
            "y0": True,
            **_place_model_name(model_name, kwargs),
        }
        for key in list(acf_defaults.keys()):
            if key in kwargs:
                acf_defaults.pop(key)

        mg.finalise_plot(ax, **acf_defaults, **kwargs)

    # Only print warnings for autocorrelated residuals
    for var_name, observed_data in obs_vars.items():
        ppc_samples = ppc.posterior_predictive[var_name].values
        ppc_mean = ppc_samples.reshape(-1, ppc_samples.shape[-1]).mean(axis=0)
        residuals = observed_data - ppc_mean
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        p_value = lb_test["lb_pvalue"].values[0]
        if p_value <= 0.05:
            label = var_labels.get(var_name, var_name)
            print(f"*** WARNING: {label} residuals are autocorrelated (Ljung-Box p={p_value:.4f}) ***")
