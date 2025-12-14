"""Plotting utilities for PyMC posterior analysis."""

import math
from collections.abc import Sequence

import arviz as az
import matplotlib.pyplot as plt
import mgplot as mg
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import stats

from src.analysis.extraction import get_scalar_var, get_scalar_var_names, get_vector_var


def _place_model_name(model_name: str, kwargs: dict) -> dict:
    """Place model_name in first available footer/header slot."""
    for slot in ["rfooter", "rheader", "lheader"]:
        if slot not in kwargs:
            return {slot: model_name}
    return {}


def _auto_scale(samples: pd.Series, median: float) -> tuple[pd.Series, int]:
    """Scale samples for better visualization when values are large."""
    threshold = 1.3
    if abs(median) <= threshold:
        return samples, 1
    scale = 10 ** math.floor(math.log10(abs(median * 10)))
    return samples / scale, max(int(scale), 1)


def plot_posteriors_kde(
    trace: az.InferenceData,
    model_name: str = "Model",
    **kwargs,
) -> None:
    """Plot separate Kernel Density Estimates for each coefficient posterior."""
    scalar_vars = get_scalar_var_names(trace)

    for var_name in sorted(scalar_vars):
        samples = get_scalar_var(var_name, trace)

        _, ax = plt.subplots()

        samples.plot.kde(ax=ax, color="steelblue", linewidth=2)

        kde = stats.gaussian_kde(samples)
        x_range = np.linspace(samples.min(), samples.max(), 200)
        kde_values = kde(x_range)
        ax.fill_between(x_range, kde_values, alpha=0.3, color="steelblue")

        ax.axvline(x=0, color="darkred", linestyle="--", linewidth=1.5)

        median_val = samples.quantile(0.5)
        ax.axvline(x=median_val, color="black", linestyle="--", linewidth=1, alpha=0.7)

        max_y = kde_values.max()
        ax.text(
            median_val,
            max_y,
            f"{median_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

        defaults = {
            "title": f"{var_name} Posterior",
            "xlabel": "Coefficient value",
            "lfooter": "Red dashed line marks zero. Black dashed line marks median.",
            **_place_model_name(model_name, kwargs),
        }
        for key in list(defaults.keys()):
            if key in kwargs:
                defaults.pop(key)

        mg.finalise_plot(ax, **defaults, **kwargs)


def plot_posteriors_bar(
    trace: az.InferenceData,
    model_name: str = "Model",
    **kwargs,
) -> None:
    """Plot horizontal bar chart of coefficient posteriors."""
    scalar_vars = get_scalar_var_names(trace)

    posteriors = {}
    labels = {}
    all_significant_99 = True
    all_significant_95 = True

    for var in scalar_vars:
        samples = get_scalar_var(var, trace)
        median = samples.quantile(0.5)

        if median < 0:
            if samples.quantile(0.99) >= 0:
                all_significant_99 = False
            if samples.quantile(0.95) >= 0:
                all_significant_95 = False
        else:
            if samples.quantile(0.01) <= 0:
                all_significant_99 = False
            if samples.quantile(0.05) <= 0:
                all_significant_95 = False

        scaled_samples, scale = _auto_scale(samples, median)
        if scale != 1:
            posteriors[var] = scaled_samples
            labels[var] = f"{var}/{scale}"
        else:
            posteriors[var] = samples
            labels[var] = var

    cuts = [2.5, 16]
    palette = "Blues"
    cmap = plt.get_cmap(palette)
    color_fracs = [0.4, 0.7]

    _, ax = plt.subplots(figsize=(10, len(scalar_vars) * 0.6 + 1))

    y_positions = range(len(scalar_vars))
    bar_height = 0.7

    sorted_vars = sorted(scalar_vars)
    for i, var in enumerate(sorted_vars):
        samples = posteriors[var]

        for j, p in enumerate(cuts):
            quants = (p, 100 - p)
            lower = samples.quantile(quants[0] / 100.0)
            upper = samples.quantile(quants[1] / 100.0)
            height = bar_height * (1 - j * 0.25)

            ax.barh(
                i,
                width=upper - lower,
                left=lower,
                height=height,
                color=cmap(color_fracs[j]),
                alpha=0.7,
                label=f"{quants[1] - quants[0]:.0f}% HDI" if i == 0 else "_",
                zorder=j + 1,
            )

        median = samples.quantile(0.5)
        ax.vlines(
            median,
            i - bar_height / 2,
            i + bar_height / 2,
            color="black",
            linestyle="-",
            linewidth=1,
            zorder=10,
            label="Median" if i == 0 else "_",
        )
        ax.text(
            median,
            i + bar_height / 2 + 0.05,
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    ax.axvline(x=0, color="darkred", linestyle="-", linewidth=1.5, zorder=15)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([labels[var] for var in sorted_vars])
    ax.invert_yaxis()

    lfooter = "Some variables have been scaled (as indicated)."
    if all_significant_99:
        lfooter += " All coefficients are different from zero (>99% probability)."
    elif all_significant_95:
        lfooter += " All coefficients are different from zero (>95% probability)."

    defaults = {
        "title": "Coefficient Posteriors",
        "xlabel": "Coefficient value",
        "legend": {"loc": "best", "fontsize": "x-small"},
        "lfooter": lfooter,
        **_place_model_name(model_name, kwargs),
    }
    for key in list(defaults.keys()):
        if key in kwargs:
            defaults.pop(key)

    mg.finalise_plot(ax, **defaults, **kwargs)


def posterior_predictive_checks(
    trace: az.InferenceData,
    model,
    obs_vars: dict[str, np.ndarray],
    obs_index: pd.Index,
    var_labels: dict[str, str] | None = None,
    model_name: str = "Model",
    **kwargs,
) -> az.InferenceData:
    """Generate and plot posterior predictive samples."""
    import pymc as pm

    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)

    if var_labels is None:
        var_labels = {k: k for k in obs_vars}

    for var_name, observed_data in obs_vars.items():
        ppc_samples = ppc.posterior_predictive[var_name].values
        ppc_flat = ppc_samples.reshape(-1, ppc_samples.shape[-1])

        ppc_mean = ppc_flat.mean(axis=0)
        ppc_05 = np.percentile(ppc_flat, 5, axis=0)
        ppc_95 = np.percentile(ppc_flat, 95, axis=0)
        ppc_16 = np.percentile(ppc_flat, 16, axis=0)
        ppc_84 = np.percentile(ppc_flat, 84, axis=0)

        band_90 = pd.DataFrame({"lower": ppc_05, "upper": ppc_95}, index=obs_index)
        ax = mg.fill_between_plot(band_90, color="steelblue", alpha=0.15, label="90% CI")

        band_68 = pd.DataFrame({"lower": ppc_16, "upper": ppc_84}, index=obs_index)
        ax = mg.fill_between_plot(
            band_68, ax=ax, color="steelblue", alpha=0.25, label="68% CI"
        )

        predicted = pd.Series(ppc_mean, index=obs_index, name="Predicted mean")
        mg.line_plot(predicted, ax=ax, color="steelblue", width=1.5)

        observed = pd.Series(observed_data, index=obs_index, name="Observed")
        mg.line_plot(observed, ax=ax, color="darkred", width=1)

        label = var_labels.get(var_name, var_name)
        defaults = {
            "title": f"Posterior Predictive Check - {label}",
            "ylabel": label,
            "legend": {"loc": "upper right", "fontsize": "x-small"},
            "lfooter": "Blue: model prediction with credible intervals. Red: observed.",
            "y0": True,
            **_place_model_name(model_name, kwargs),
        }
        for key in list(defaults.keys()):
            if key in kwargs:
                defaults.pop(key)

        mg.finalise_plot(ax, **defaults, **kwargs)

    return ppc


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
    from statsmodels.stats.diagnostic import acorr_ljungbox

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


def plot_timeseries(
    trace: az.InferenceData | None = None,
    var: str | None = None,
    index: pd.PeriodIndex | None = None,
    data: pd.DataFrame | None = None,
    legend_stem: str = "Model Estimate",
    color: str = "blue",
    start: pd.Period | None = None,
    cuts: Sequence[float] = (0.005, 0.025, 0.16),
    alphas: Sequence[float] = (0.1, 0.2, 0.3),
) -> Axes | None:
    """Plot time series with credible intervals."""
    if len(cuts) != len(alphas):
        raise ValueError("Cuts and alphas must have the same length")

    if data is not None:
        samples = data
    elif trace is not None and var is not None and index is not None:
        samples = get_vector_var(var, trace)
        samples.index = index
    else:
        raise ValueError("Must provide either (trace, var, index) or data")

    if start is not None:
        samples = samples[samples.index >= start]

    ax: Axes | None = None
    for cut, alpha in zip(cuts, alphas):
        if not (0 < cut < 0.5):
            raise ValueError("Cuts must be between 0 and 0.5")

        lower = samples.quantile(q=cut, axis=1)
        upper = samples.quantile(q=1 - cut, axis=1)
        band = pd.DataFrame({"lower": lower, "upper": upper}, index=samples.index)
        ax = mg.fill_between_plot(
            band,
            ax=ax,
            color=color,
            alpha=alpha,
            label=f"{legend_stem} {int((1 - 2 * cut) * 100)}% Credible Interval",
        )

    median = samples.quantile(q=0.5, axis=1)
    median.name = f"{legend_stem} Median"
    ax = mg.line_plot(median, ax=ax, color=color, width=1, annotate=True)

    return ax
