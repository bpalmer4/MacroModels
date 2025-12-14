"""MCMC diagnostics for PyMC models."""

import arviz as az
import numpy as np


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
            at_max = trace.sample_stats.reached_max_treedepth.values
            at_max_rate = float(at_max.mean())
            max_observed = int(trace.sample_stats.tree_depth.values.max())
            print(
                f"{warn(at_max_rate >= max_tree_depth_rate)}Tree depth at configured max: "
                f"{at_max_rate:.2%} (max observed: {max_observed})"
            )
        else:
            ignore = 10
            tree_depth = trace.sample_stats.tree_depth.values
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
