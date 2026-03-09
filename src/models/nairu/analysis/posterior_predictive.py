"""Posterior predictive checks."""

import arviz as az
import mgplot as mg
import numpy as np
import pandas as pd
import pymc as pm


def posterior_predictive_checks(
    trace: az.InferenceData,
    model: pm.Model,
    obs_vars: dict[str, np.ndarray],
    obs_index: pd.Index,
    *,
    var_labels: dict[str, str] | None = None,
    rfooter: str = "",
    show: bool = False,
) -> az.InferenceData:
    """Generate and plot posterior predictive samples."""
    with model:
        ppc = pm.sample_posterior_predictive(trace, random_seed=42)

    if var_labels is None:
        var_labels = {k: k for k in obs_vars}

    for var_name, observed_data in obs_vars.items():
        ppc_samples = ppc.posterior_predictive[var_name].to_numpy()
        ppc_flat = ppc_samples.reshape(-1, ppc_samples.shape[-1])

        ppc_mean = ppc_flat.mean(axis=0)
        ppc_05 = np.percentile(ppc_flat, 5, axis=0)
        ppc_95 = np.percentile(ppc_flat, 95, axis=0)
        ppc_16 = np.percentile(ppc_flat, 16, axis=0)
        ppc_84 = np.percentile(ppc_flat, 84, axis=0)

        band_90 = pd.DataFrame({"lower": ppc_05, "upper": ppc_95}, index=obs_index)
        ax = mg.fill_between_plot(band_90, color="steelblue", alpha=0.15, label="90% CI")

        band_68 = pd.DataFrame({"lower": ppc_16, "upper": ppc_84}, index=obs_index)
        mg.fill_between_plot(band_68, ax=ax, color="steelblue", alpha=0.25, label="68% CI")

        predicted = pd.Series(ppc_mean, index=obs_index, name="Predicted mean")
        mg.line_plot(predicted, ax=ax, color="steelblue", width=1.5)

        observed = pd.Series(observed_data, index=obs_index, name="Observed")
        mg.line_plot(observed, ax=ax, color="darkred", width=1)

        label = var_labels.get(var_name, var_name)
        mg.finalise_plot(
            ax,
            title=f"Posterior Predictive Check - {label}",
            ylabel=label,
            legend={"loc": "upper right", "fontsize": "x-small"},
            lfooter="Blue: model prediction with credible intervals. Red: observed.",
            rfooter=rfooter,
            y0=True,
            show=show,
        )

    return ppc
