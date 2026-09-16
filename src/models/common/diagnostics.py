"""MCMC diagnostics for PyMC models."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from src.models.common.extraction import get_scalar_var, get_scalar_var_names

# Diagnostic thresholds. At module level so a written report can print the
# rule beside the value, and so callers can reason about them.
MAX_R_HAT = 1.01
MIN_ESS = 400
MAX_MCSE_RATIO = 0.05
MAX_DIVERGENCE_RATE = 1 / 10_000
MAX_TREE_DEPTH_RATE = 0.05
MIN_BFMI = 0.3

# A sampler that never needed a deep tree is not saturating, so there is
# nothing to compare an observed maximum below this against.
TREE_DEPTH_IGNORE_BELOW = 10

DRAGONS = "--- THERE BE DRAGONS ---> "


@dataclass
class Check:
    """One diagnostic: what was measured, the rule, and whether it passed.

    `applicable` is False where the sampler does not produce the statistic at
    all (DEMetropolis-Z has no tree depth and no energy), which is not a
    failure and must not be reported as one.
    """

    name: str
    detail: str
    threshold: str
    passed: bool = True
    issue: str = ""
    applicable: bool = True


def _r_hat_check(summary: pd.DataFrame) -> Check:
    statistic = float(summary.r_hat.max())
    passed = statistic <= MAX_R_HAT
    return Check(
        name="R-hat",
        detail=f"Maximum R-hat convergence diagnostic: {statistic}",
        threshold=f"<= {MAX_R_HAT}",
        passed=passed,
        issue="" if passed else f"R-hat {statistic:.3f}",
    )


def _ess_check(summary: pd.DataFrame) -> Check:
    statistic = summary[["ess_tail", "ess_bulk"]].min().min()
    passed = statistic >= MIN_ESS
    return Check(
        name="ESS",
        detail=f"Minimum effective sample size (ESS) estimate: {int(statistic)}",
        threshold=f">= {MIN_ESS}",
        passed=passed,
        issue="" if passed else f"ESS {int(statistic)}",
    )


def _mcse_check(summary: pd.DataFrame) -> Check:
    statistic = (summary["mcse_mean"] / summary["sd"]).max()
    passed = statistic <= MAX_MCSE_RATIO
    return Check(
        name="MCSE/sd",
        detail=f"Maximum MCSE/sd ratio: {statistic:0.3f}",
        threshold=f"<= {MAX_MCSE_RATIO}",
        passed=passed,
        issue="" if passed else f"MCSE/sd {statistic:0.3f}",
    )


def _divergence_check(trace: az.InferenceData) -> Check:
    """Divergent transitions, where the sampler records them.

    A sampler with no divergence concept (DEMetropolis-Z) must report n/a and
    not a pass: zero divergences out of a statistic that does not exist would
    be a clean bill of health nothing had earned.
    """
    try:
        diverging_count = int(np.sum(trace.sample_stats.diverging))
    except (ValueError, AttributeError):
        return Check(
            name="Divergences",
            detail="Divergences not recorded by this sampler.",
            threshold=f"<= {MAX_DIVERGENCE_RATE:.4%}",
            applicable=False,
        )
    total_samples = trace.posterior.sizes["draw"] * trace.posterior.sizes["chain"]
    divergence_rate = diverging_count / total_samples
    passed = divergence_rate <= MAX_DIVERGENCE_RATE
    return Check(
        name="Divergences",
        detail=(
            f"Divergent transitions: {diverging_count}/{total_samples} "
            f"({divergence_rate:.4%})"
        ),
        threshold=f"<= {MAX_DIVERGENCE_RATE:.4%}",
        passed=passed,
        issue="" if passed else f"{diverging_count} divergences",
    )


def _tree_depth_check(trace: az.InferenceData) -> Check:
    """Tree depth saturation, against the configured max where it is recorded.

    Where it is not, the observed max stands in, which only means anything if
    the sampler went deep enough for saturation to be a plausible reading.
    """
    na = Check(
        name="Tree depth",
        detail="Tree depth not recorded by this sampler.",
        threshold=f"< {MAX_TREE_DEPTH_RATE:.0%} at max",
        applicable=False,
    )
    try:
        sample_stats = trace.sample_stats
    except AttributeError:
        return na

    if hasattr(sample_stats, "reached_max_treedepth"):
        at_max_rate = float(sample_stats.reached_max_treedepth.to_numpy().mean())
        max_observed = int(sample_stats.tree_depth.to_numpy().max())
        passed = at_max_rate < MAX_TREE_DEPTH_RATE
        return Check(
            name="Tree depth",
            detail=(
                f"Tree depth at configured max: {at_max_rate:.2%} "
                f"(max observed: {max_observed})"
            ),
            threshold=f"< {MAX_TREE_DEPTH_RATE:.0%} at max",
            passed=passed,
            issue="" if passed else f"tree depth {at_max_rate:.1%} at max",
        )

    if not hasattr(sample_stats, "tree_depth"):
        return na

    tree_depth = sample_stats.tree_depth.to_numpy()

    # The configured cap, where the run recorded it (see `sample_model`). This
    # is the only way to tell truncation from a trajectory that U-turned on its
    # own: NumPyro reports no `reached_max_treedepth`, so the fallback below can
    # only count draws at the deepest depth observed, and that number is
    # identical whether the sampler was cut off or simply finished there.
    # Measured case: rstar_bonds reported "8.44% at max (10)" and failed, and
    # raising the cap to 12 left the deepest trajectory at 10 and mean depth
    # unchanged, so nothing had been truncated at all.
    configured = sample_stats.attrs.get("max_tree_depth")
    if configured is not None:
        at_max_rate = float((tree_depth >= int(configured)).mean())
        passed = at_max_rate < MAX_TREE_DEPTH_RATE
        return Check(
            name="Tree depth",
            detail=(
                f"Tree depth at configured max ({int(configured)}): {at_max_rate:.2%} "
                f"(max observed: {int(tree_depth.max())})"
            ),
            threshold=f"< {MAX_TREE_DEPTH_RATE:.0%} at max",
            passed=passed,
            issue="" if passed else f"tree depth {at_max_rate:.1%} at max",
        )

    max_depth = int(tree_depth.max())
    if max_depth < TREE_DEPTH_IGNORE_BELOW:
        return Check(
            name="Tree depth",
            detail=f"Tree depth check skipped (max observed {max_depth}, too low to read).",
            threshold=f"< {MAX_TREE_DEPTH_RATE:.0%} at max",
            applicable=False,
        )

    at_max_rate = float((tree_depth == max_depth).mean())
    passed = at_max_rate < MAX_TREE_DEPTH_RATE
    return Check(
        name="Tree depth",
        detail=(
            f"Tree depth at max ({max_depth}): {at_max_rate:.2%} "
            f"(note: comparing to observed max, not configured)"
        ),
        threshold=f"< {MAX_TREE_DEPTH_RATE:.0%} at max",
        passed=passed,
        issue="" if passed else f"tree depth {at_max_rate:.1%} at max",
    )


def _bfmi_check(trace: az.InferenceData) -> Check:
    """BFMI, which needs the energy statistic and so needs a Hamiltonian sampler."""
    try:
        energy = trace.sample_stats.energy
    except AttributeError:
        return Check(
            name="BFMI",
            detail="BFMI not available (no energy statistic; not a Hamiltonian sampler).",
            threshold=f">= {MIN_BFMI}",
            applicable=False,
        )
    del energy
    statistic = float(az.bfmi(trace).min())
    passed = statistic >= MIN_BFMI
    return Check(
        name="BFMI",
        detail=f"Minimum Bayesian fraction of missing information: {statistic:0.2f}",
        threshold=f">= {MIN_BFMI}",
        passed=passed,
        issue="" if passed else f"BFMI {statistic:0.2f}",
    )


def run_checks(trace: az.InferenceData) -> list[Check]:
    """Apply every diagnostic to a trace and return the results in order."""
    summary = az.summary(trace)
    return [
        _r_hat_check(summary),
        _ess_check(summary),
        _mcse_check(summary),
        _divergence_check(trace),
        _tree_depth_check(trace),
        _bfmi_check(trace),
    ]


def check_model_diagnostics(trace: az.InferenceData, *, verbose: bool = True) -> list[str]:
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

    Args:
        trace: InferenceData from model fitting
        verbose: If True, print each diagnostic as it is checked

    Returns:
        Short descriptions of the checks that failed (empty if none did).

    """
    checks = run_checks(trace)
    if verbose:
        for check in checks:
            print(f"{DRAGONS if not check.passed else ''}{check.detail}")
    return [check.issue for check in checks if check.issue]


def _worst_offenders(summary: pd.DataFrame, n_worst: int) -> list[str]:
    """Name the parameters behind the headline numbers.

    The single worst r_hat or ESS says a model failed to mix; it does not say
    which part of it did. Vector elements are indexed in the summary, so this
    points at the quarter as well as the state.
    """
    lines: list[str] = []
    wanted = (
        ("Highest R-hat", "r_hat", False, "{:.4f}"),
        ("Lowest ESS bulk", "ess_bulk", True, "{:.0f}"),
        ("Lowest ESS tail", "ess_tail", True, "{:.0f}"),
    )
    for label, column, ascending, fmt in wanted:
        if column not in summary.columns:
            continue
        ranked = summary[column].dropna().sort_values(ascending=ascending).head(n_worst)
        if ranked.empty:
            continue
        entries = ", ".join(f"{name} {fmt.format(value)}" for name, value in ranked.items())
        lines.append(f"  {label + ':':<18}{entries}")
    return lines


def diagnostics_headline(issues: list[str], max_len: int = 60) -> str:
    """One short line for a chart header. Empty when the sample was clean.

    Kept short deliberately: this lands in a chart's `lheader`, where a long
    string collides with the header on the other side.
    """
    if not issues:
        return ""
    text = f"Sampling issues: {'; '.join(issues)}"
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip(" ;,") + "..."
    return text


def write_diagnostics_report(
    trace: az.InferenceData,
    path: str | Path,
    *,
    title: str,
    notes: list[str] | None = None,
    n_worst: int = 5,
) -> list[str]:
    """Write a short diagnostics file beside a saved trace.

    Args:
        trace: InferenceData from model fitting
        path: File to write (overwritten)
        title: What was estimated, e.g. "ystar (simple_excess)"
        notes: Extra context lines for the header, e.g. the sampler settings
        n_worst: How many parameters to name per worst-offender list

    Returns:
        The same issue list `check_model_diagnostics` returns.

    """
    path = Path(path)
    checks = run_checks(trace)
    issues = [check.issue for check in checks if check.issue]
    summary = az.summary(trace)

    chains = trace.posterior.sizes["chain"]
    draws = trace.posterior.sizes["draw"]

    lines = [
        f"Diagnostics: {title}",
        f"Sampled:     {_sampled_line(trace)}",
        f"Reported:    {datetime.now().astimezone():%Y-%m-%d %H:%M}",
        f"Sample:      {chains} chains x {draws} draws = {chains * draws} samples",
    ]
    lines.extend(f"             {note}" for note in notes or [])

    lines.append("")
    lines.append("CRITICAL ISSUES")
    if issues:
        lines.extend(f"  {issue}" for issue in issues)
    else:
        lines.append("  none")

    lines.append("")
    lines.append("CHECKS")
    for check in checks:
        if not check.applicable:
            marker = "[n/a ]"
        elif check.passed:
            marker = "[PASS]"
        else:
            marker = "[FAIL]"
        lines.append(f"  {marker} {check.name + ':':<13}{check.detail}  ({check.threshold})")

    worst = _worst_offenders(summary, n_worst)
    if worst:
        lines.append("")
        lines.append("WORST PARAMETERS")
        lines.extend(worst)

    path.write_text("\n".join(lines) + "\n")
    return issues


DIAGNOSTICS_STEM = "run-diagnostics"

def _sampled_line(trace: az.InferenceData) -> str:
    """When the trace was sampled, and how long before this report.

    A report is written whenever the analysis is re-run, so its own date can
    be days newer than the sample it describes: `--analyse-only` re-reads a
    trace estimated last week. Read from the trace's own `created_at`, not
    from the file's timestamp, which a copy or a touch would change.
    """
    created = str(getattr(trace, "posterior", None) and trace.posterior.attrs.get("created_at", ""))
    if not created:
        return "unknown (the trace records no creation date)"

    try:
        sampled = datetime.fromisoformat(created).astimezone()
    except ValueError:
        return f"{created} (unrecognised date format)"

    age = (datetime.now().astimezone() - sampled).days
    when = "today" if age < 1 else f"{age} day{'s' if age > 1 else ''} before this report"
    return f"{sampled:%Y-%m-%d %H:%M} ({when})"


def diagnostics_path(chart_dir: str | Path, prefix: str) -> Path:
    """Where a run's diagnostics file lives: in its own chart directory.

    One file per run, named for the run, so it sits with the charts it
    describes rather than among the traces. `mg.clear_chart_dir()` removes
    image files only, so re-charting does not delete it.
    """
    return Path(chart_dir) / f"{DIAGNOSTICS_STEM}-{prefix}.txt"


def save_diagnostics(
    trace: az.InferenceData,
    chart_dir: str | Path,
    prefix: str,
    *,
    model: str,
    notes: list[str] | None = None,
) -> Path:
    """Write this run's diagnostics file into its chart directory, and say so.

    The one line every model's analysis calls, so that a run of any model
    leaves exactly one diagnostics file, beside that run's charts.
    """
    path = diagnostics_path(chart_dir, prefix)
    path.parent.mkdir(parents=True, exist_ok=True)

    # `mg.clear_chart_dir()` removes image files only. Two prefixes charted
    # into one directory would therefore leave the first run's diagnostics
    # behind, describing charts that have just been deleted. Clear any other
    # run's file so what is here always matches the charts that are here.
    for stale in path.parent.glob(f"{DIAGNOSTICS_STEM}-*.txt"):
        if stale != path and stale.is_file():
            stale.unlink()
    issues = write_diagnostics_report(
        trace,
        path,
        title=f"{model} ({prefix})",
        notes=notes,
    )
    flag = f"  {DRAGONS}{'; '.join(issues)}" if issues else ""
    print(f"Saved diagnostics to: {path}{flag}")
    return path


def load_diagnostics_headline(path: str | Path) -> str:
    """Read back the critical issues from a written report, as a chart header.

    Charting runs separately from estimation in most of these models, so the
    chart cannot see the issue list the sampler produced. This recovers it
    from the report written beside the trace.
    """
    path = Path(path)
    if not path.is_file():
        return ""
    issues: list[str] = []
    in_block = False
    for line in path.read_text().splitlines():
        if line.startswith("CRITICAL ISSUES"):
            in_block = True
            continue
        if in_block:
            if not line.strip():
                break
            issue = line.strip()
            if issue == "none":
                return ""
            issues.append(issue)
    return diagnostics_headline(issues)


def diagnostics_header(
    chart_dir: str | Path,
    prefix: str,
    existing: str = "",
) -> str:
    """Build a chart header carrying any sampling issues, ahead of the chart's own text.

    Returns `existing` unchanged when the sample was clean, so a chart that
    already says something useful in its header keeps saying it. When it was
    not clean the warning goes first, because that is the thing that decides
    whether the rest of the chart means anything.
    """
    note = load_diagnostics_headline(diagnostics_path(chart_dir, prefix))
    if not note:
        return existing
    return f"{note} | {existing}" if existing else note


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
            non_critical_threshold = 2  # non-critical params only warn at 3+ stars
            if (stars > 0 if param in critical_params else stars > non_critical_threshold):
                print(
                    f"*** WARNING: Parameter '{param}' may be indistinguishable from zero "
                    f"({stars} stars). Check model specification! ***"
                )

    return df
