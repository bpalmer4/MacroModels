"""MCMC diagnostics for PyMC models."""


import arviz as az
import numpy as np
import pandas as pd

from src.models.common.extraction import get_scalar_var, get_scalar_var_names


def check_model_diagnostics(trace: az.InferenceData) -> None:
    """Check the inference data for potential problems.

    Diagnostics applied:
    - R-hat (Gelman-Rubin): Compares between-chain and within-chain variance.
      Values > 1.01 suggest chains have not converged to the same distribution.
    - ESS (Effective Sample Size): Estimates independent samples accounting for
      autocorrelation. Low ESS (< 400) indicates high autocorrelation or short chains.
    - MCSE/sd ratio: Monte Carlo standard error relative to posterior sd.
      Ratios > 5% suggest insufficient samples for reliable posterior mean estimates.
    - Divergent transitions: Indicate regions where the sampler struggled with
      posterior geometry. Any divergences may signal biased estimates.
    - Tree depth saturation: High rates at max tree depth suggest the sampler
      is working harder than expected, possibly due to difficult geometry.
    - BFMI (Bayesian Fraction of Missing Information): Measures how well the
      sampler explores the energy distribution. Values < 0.3 suggest poor exploration.
    """

    def warn(w: bool) -> str:
        return "--- THERE BE DRAGONS ---> " if w else ""

    summary = az.summary(trace)

    # check model convergence
    max_r_hat = 1.01
    statistic = summary.r_hat.max()
    print(
        f"{warn(statistic > max_r_hat)}Maximum R-hat convergence diagnostic: {statistic}"
    )

    # check effective sample size
    min_ess = 400
    statistic = summary[["ess_tail", "ess_bulk"]].min().min()
    print(
        f"{warn(statistic < min_ess)}Minimum effective sample size (ESS) estimate: {int(statistic)}"
    )

    # check MCSE ratio (should be < 5% of posterior sd)
    max_mcse_ratio = 0.05
    statistic = (summary["mcse_mean"] / summary["sd"]).max()
    print(
        f"{warn(statistic > max_mcse_ratio)}Maximum MCSE/sd ratio: {statistic:0.3f}"
    )

    # check for divergences (rate-based: < 1 in 10,000 samples)
    max_divergence_rate = 1 / 10_000
    try:
        diverging_count = int(np.sum(trace.sample_stats.diverging))
    except (ValueError, AttributeError):
        diverging_count = 0
    total_samples = trace.posterior.sizes["draw"] * trace.posterior.sizes["chain"]
    divergence_rate = diverging_count / total_samples
    print(
        f"{warn(divergence_rate > max_divergence_rate)}Divergent transitions: "
        f"{diverging_count}/{total_samples} ({divergence_rate:.4%})"
    )

    # check max tree depth saturation
    max_tree_depth_rate = 0.05
    try:
        if hasattr(trace.sample_stats, "reached_max_treedepth"):
            at_max = trace.sample_stats.reached_max_treedepth.to_numpy()
            at_max_rate = float(at_max.mean())
            max_observed = int(trace.sample_stats.tree_depth.to_numpy().max())
            print(
                f"{warn(at_max_rate >= max_tree_depth_rate)}Tree depth at configured max: "
                f"{at_max_rate:.2%} (max observed: {max_observed})"
            )
        else:
            ignore = 10
            tree_depth = trace.sample_stats.tree_depth.to_numpy()
            max_depth = int(tree_depth.max())
            if max_depth < ignore:
                print("Tree depth check skipped (max depth too low).")
            else:
                at_max_rate = (tree_depth == max_depth).mean()
                print(
                    f"{warn(at_max_rate >= max_tree_depth_rate)}Tree depth at max ({max_depth}): "
                    f"{at_max_rate:.2%} (note: comparing to observed max, not configured)"
                )
    except AttributeError:
        pass

    # check BFMI
    min_bfmi = 0.3
    statistic = az.bfmi(trace).min()
    print(
        f"{warn(statistic < min_bfmi)}Minimum Bayesian fraction of missing information: {statistic:0.2f}"
    )


def check_for_zero_coeffs(
    trace: az.InferenceData,
    critical_params: list[str] | None = None,
) -> pd.DataFrame:
    """Check scalar parameters for coefficients indistinguishable from zero.

    Automatically detects scalar variables (excludes vector/time series variables).
    Shows quantiles and flags parameters that may be indistinguishable from zero.

    Args:
        trace: InferenceData from model fitting
        critical_params: List of parameter names that are critical (warn if any
            quantile crosses zero). If None, uses default threshold of 2+ crossings.

    Returns:
        DataFrame with quantiles and significance markers.

    """
    if critical_params is None:
        critical_params = []

    q = [0.01, 0.05, 0.10, 0.25, 0.50]
    q_tail = [1 - x for x in q[:-1]][::-1]
    q = q + q_tail

    scalar_vars = get_scalar_var_names(trace)

    if not scalar_vars:
        return pd.DataFrame()

    quantiles = {
        var_name: get_scalar_var(var_name, trace).quantile(q)
        for var_name in scalar_vars
    }

    df = pd.DataFrame(quantiles).T.sort_index()
    problem_intensity = (
        pd.DataFrame(np.sign(df.T))
        .apply([lambda x: x.lt(0).sum(), lambda x: x.ge(0).sum()])
        .min()
        .astype(int)
    )
    marker = pd.Series(["*"] * len(problem_intensity), index=problem_intensity.index)
    markers = (
        marker.str.repeat(problem_intensity).reindex(problem_intensity.index).fillna("")
    )
    df["Check Significance"] = markers

    for param in df.index:
        if param in problem_intensity:
            stars = problem_intensity[param]
            if (stars > 0 if param in critical_params else stars > 2):
                print(
                    f"*** WARNING: Parameter '{param}' may be indistinguishable from zero "
                    f"({stars} stars). Check model specification! ***"
                )

    return df
