"""GDP nowcasting via Bayesian VAR with Minnesota prior.

T-0 only model: uses all quarterly indicators (including monthly indicators
aggregated to quarterly) to nowcast the current quarter's GDP growth, just
before the official GDP release.

Approach:
    1. Build a quarterly panel of GDP growth + ~10 quarterly indicators
    2. Fit a BVAR(2) with Minnesota prior on the historical panel
    3. Use the contemporaneous values of the indicators in the target quarter
       as conditioning information
    4. Compute the conditional mean of GDP given the other indicators using
       the Gaussian conditioning formula

The Minnesota prior is implemented directly via the closed-form normal posterior
(equation by equation, since the prior is diagonal across equations). This is
fast (sub-second), exact (no MCMC sampling noise), and easy to debug.

References:
    - Litterman (1986) "Forecasting with Bayesian Vector Autoregressions"
    - Karlsson (2013) "Forecasting with Bayesian Vector Autoregression"
      (Handbook of Economic Forecasting Vol 2B)
    - Bańbura, Giannone, Reichlin (2010) "Large Bayesian Vector Auto Regressions"

Usage:
    # Live nowcast
    uv run python -m src.models.gdp_nowcast_bvar.model

    # Programmatic use
    from src.models.gdp_nowcast_bvar.model import nowcast
    result = nowcast(target_quarter=pd.Period("2025Q4", "Q-DEC"))
"""

import logging
import warnings
from dataclasses import dataclass

import mgplot as mg
import numpy as np
import pandas as pd

from src.data import (
    get_building_approvals_growth_qrtly,
    get_employment_growth_qrtly,
    get_goods_balance_qrtly,
    get_hours_growth_qrtly,
    get_trimmed_mean_qrtly,
)
from src.data.capex import get_total_capex_growth_qrtly
from src.data.construction import get_total_construction_growth_qrtly
from src.data.gdp import get_gdp
from src.data.surveys import get_nab_business_conditions_qrtly
from src.data.wpi import get_wpi_growth_qrtly

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Constants ---

CHART_DIR = "./charts/GDP-Nowcast-BVAR/"
SHOW = False

# Sample start: same as DFM, avoids pre-inflation-target structural breaks
SAMPLE_START = pd.Period("1990Q1", freq="Q-DEC")

# BVAR hyperparameters
N_LAGS = 2          # VAR(2)
LAMBDA_TIGHT = 0.2  # overall tightness of Minnesota prior (smaller = more shrinkage)
LAMBDA_CROSS = 0.5  # cross-variable tightness (relative to own-variable)
LAMBDA_DECAY = 1.0  # lag decay exponent (1 = harmonic, 2 = quadratic)


# --- BVAR with Minnesota prior ---


@dataclass
class BVARFit:
    """Fitted BVAR with Minnesota prior."""

    coefficients: np.ndarray  # (1 + n_vars*n_lags) x n_vars matrix of coefficients
    sigma: np.ndarray         # n_vars x n_vars residual covariance
    var_names: list[str]
    n_lags: int
    fitted: pd.DataFrame      # in-sample fitted values

    @property
    def n_vars(self) -> int:
        """Number of variables in the VAR."""
        return len(self.var_names)


def _build_lag_matrix(data: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the (Y, X) regression matrices for a VAR(p).

    Y is (T-p) x N (target observations)
    X is (T-p) x (1 + N*p) (constant + lagged regressors)
    """
    T, n = data.shape
    Y = data[n_lags:]
    X_blocks = [np.ones((T - n_lags, 1))]
    for k in range(1, n_lags + 1):
        X_blocks.append(data[n_lags - k : T - k])
    X = np.hstack(X_blocks)
    return Y, X


def _ar1_residual_variance(series: np.ndarray) -> float:
    """Compute residual variance from a univariate AR(1) on a single series.

    Used to set the Minnesota prior's variance scaling, following Litterman.
    """
    s = series[~np.isnan(series)]
    if len(s) < 5:
        return float(np.var(s))
    y = s[1:]
    x = s[:-1]
    beta = float(np.dot(x, y) / np.dot(x, x))
    resid = y - x * beta
    return float(np.var(resid))


def fit_bvar_minnesota(
    data: pd.DataFrame,
    n_lags: int = N_LAGS,
    lambda_tight: float = LAMBDA_TIGHT,
    lambda_cross: float = LAMBDA_CROSS,
    lambda_decay: float = LAMBDA_DECAY,
) -> BVARFit:
    """Fit a VAR(p) with Minnesota prior, equation-by-equation closed form.

    The Minnesota prior encodes:
        - Each variable follows AR(1) with own-lag-1 coefficient = 1 (random walk)
        - Higher lags shrink toward zero with strength k^lambda_decay
        - Cross-variable coefficients shrink harder than own (by factor lambda_cross)
        - Constant has diffuse prior

    Posterior is computed equation-by-equation since the prior is independent
    across equations. Each equation has a normal-normal-conjugate posterior:
        β_post = (X'X / σ² + Λ⁻¹) ⁻¹ (X'Y_i / σ² + Λ⁻¹ β_prior)

    where Λ is the diagonal prior covariance and σ² is the equation's residual
    variance (estimated from a univariate AR(1)).
    """
    var_names = list(data.columns)
    n_vars = len(var_names)

    # Drop rows with any NaN to keep things simple (BVAR needs balanced panel)
    df = data.dropna()
    if len(df) < n_lags + n_vars + 5:
        msg = f"Not enough complete observations: {len(df)} (need at least {n_lags + n_vars + 5})"
        raise ValueError(msg)

    arr = df.to_numpy()
    Y, X = _build_lag_matrix(arr, n_lags)
    n_obs, n_coeffs = X.shape

    # AR(1) residual variances for prior scaling
    sigma_ar1 = np.array([_ar1_residual_variance(arr[:, i]) for i in range(n_vars)])
    sigma_ar1 = np.maximum(sigma_ar1, 1e-8)

    # Prior mean: 1 on own first lag, 0 elsewhere; 0 on constant
    B_prior = np.zeros((n_coeffs, n_vars))
    for i in range(n_vars):
        B_prior[1 + i, i] = 1.0  # constant is column 0, then lag-1 block starts at column 1

    # Prior variance for each coefficient (diagonal Λ)
    # Indexing: column 0 = constant, columns 1..N = lag 1 of vars, columns N+1..2N = lag 2, etc.
    V_prior = np.zeros((n_coeffs, n_vars))
    V_prior[0, :] = 1e6  # diffuse prior on the constant
    for k in range(1, n_lags + 1):
        for j in range(n_vars):  # variable j contributes to regressor block at lag k
            col_idx = 1 + (k - 1) * n_vars + j
            for i in range(n_vars):  # equation for variable i
                if i == j:
                    # Own-variable coefficient
                    V_prior[col_idx, i] = (lambda_tight / (k ** lambda_decay)) ** 2
                else:
                    # Cross-variable coefficient: scaled by sigma ratio
                    V_prior[col_idx, i] = (
                        (lambda_tight * lambda_cross / (k ** lambda_decay)) ** 2
                        * sigma_ar1[i] / sigma_ar1[j]
                    )

    # Posterior, one equation at a time
    B_post = np.zeros((n_coeffs, n_vars))
    for i in range(n_vars):
        Lambda_inv = np.diag(1.0 / V_prior[:, i])
        sig_i = sigma_ar1[i]
        # Posterior precision and posterior mean
        precision = X.T @ X / sig_i + Lambda_inv
        rhs = X.T @ Y[:, i] / sig_i + Lambda_inv @ B_prior[:, i]
        B_post[:, i] = np.linalg.solve(precision, rhs)

    # Posterior residual covariance from in-sample fit
    fitted_arr = X @ B_post
    resid = Y - fitted_arr
    Sigma_post = resid.T @ resid / max(n_obs - n_coeffs, 1)

    fitted_df = pd.DataFrame(
        fitted_arr,
        index=df.index[n_lags:],
        columns=var_names,
    )

    return BVARFit(
        coefficients=B_post,
        sigma=Sigma_post,
        var_names=var_names,
        n_lags=n_lags,
        fitted=fitted_df,
    )


def conditional_nowcast(
    fit: BVARFit,
    history: pd.DataFrame,
    target_quarter: pd.Period,
    target_var: str,
    contemporaneous: pd.Series,
) -> tuple[float, float, list[str]]:
    """Compute the conditional forecast of `target_var` at `target_quarter`,
    given the contemporaneous values of *whichever* other variables are
    available in the panel for that quarter.

    Uses the Gaussian conditioning formula on the observed subset:
        E[y_t | obs, lags] = ŷ_t + Σ_{y,o} Σ_{o,o}⁻¹ (obs - ŷ_other,t)
        Var[y_t | obs, lags] = Σ_{y,y} - Σ_{y,o} Σ_{o,o}⁻¹ Σ_{o,y}

    where Σ_{o,o} and Σ_{y,o} are partitioned to only include the indicators
    that actually have observed values for the target quarter. As more
    indicators publish, the conditioning subset grows and the forecast tightens.
    If no indicators are available, falls back to the unconditional VAR forecast.

    Returns (conditional_mean, conditional_std, used_indicators).
    """
    var_names = fit.var_names
    n_vars = len(var_names)
    n_lags = fit.n_lags

    if target_var not in var_names:
        msg = f"target_var '{target_var}' not in panel: {var_names}"
        raise ValueError(msg)
    target_idx = var_names.index(target_var)

    # Build the lagged regressor row for the target quarter
    hist = history.loc[history.index < target_quarter, var_names]
    hist = hist.dropna()
    if len(hist) < n_lags:
        msg = f"Not enough lagged observations to forecast {target_quarter}"
        raise RuntimeError(msg)

    lagged = hist.iloc[-n_lags:].to_numpy()  # n_lags x n_vars, most recent at the bottom
    x_row = [1.0]
    for k in range(1, n_lags + 1):
        x_row.extend(lagged[-k])  # lag k = the k-th most recent observation
    x_row = np.array(x_row)

    # Unconditional forecast for all variables at target_quarter
    y_uncond = x_row @ fit.coefficients  # shape (n_vars,)

    # Determine which non-target indicators have observed contemporaneous values
    other_idx_full = [i for i in range(n_vars) if i != target_idx]
    available_pairs = []
    for i in other_idx_full:
        name = var_names[i]
        if name in contemporaneous.index and pd.notna(contemporaneous[name]):
            available_pairs.append((i, name))

    if not available_pairs:
        # No contemporaneous information at all — fall back to unconditional forecast
        return (
            float(y_uncond[target_idx]),
            float(np.sqrt(fit.sigma[target_idx, target_idx])),
            [],
        )

    available_idx = [i for i, _ in available_pairs]
    used_names = [n for _, n in available_pairs]
    obs = np.array([contemporaneous[n] for n in used_names])

    Sigma = fit.sigma
    sigma_yy = Sigma[target_idx, target_idx]
    sigma_yo = Sigma[target_idx, available_idx]
    sigma_oo = Sigma[np.ix_(available_idx, available_idx)]

    # Conditional update on the available subset
    diff = obs - y_uncond[available_idx]
    w = np.linalg.solve(sigma_oo, diff)
    cond_mean = float(y_uncond[target_idx] + sigma_yo @ w)
    cond_var = float(sigma_yy - sigma_yo @ np.linalg.solve(sigma_oo, sigma_yo))
    cond_std = float(np.sqrt(max(cond_var, 1e-12)))

    return cond_mean, cond_std, used_names


# --- Data loading ---


def _load_panel() -> pd.DataFrame:
    """Load the quarterly panel of GDP growth + indicators.

    Everything is already in growth-rate / stationary form.
    """
    gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()
    gdp_growth = (np.log(gdp).diff(1) * 100).rename("gdp_growth")

    # 10-variable panel (longest viable history starts 1997Q4, ~112 obs).
    # Tested with 5 variables instead — surprisingly that was strictly worse
    # (every NCstd > every NCstd in the 10-var version). Reducing collinearity
    # removed the natural dampening that came from conflicting indicator
    # surprises in the conditional update step.
    #
    # Excluded entirely (short history):
    #   - retail_growth (5682.0 only from 2012Q4)
    #   - business_profits_growth (5676.0 only from 2001Q2)
    series = {
        "gdp_growth": gdp_growth,
        "building_approvals_growth": get_building_approvals_growth_qrtly().data,
        "employment_growth": get_employment_growth_qrtly().data,
        "hours_growth": get_hours_growth_qrtly().data,
        "goods_balance": get_goods_balance_qrtly().data,
        "nab_conditions": get_nab_business_conditions_qrtly().data,
        "cpi_trimmed_mean": get_trimmed_mean_qrtly().data,
        "wpi_growth": get_wpi_growth_qrtly().data,
        "construction_growth": get_total_construction_growth_qrtly().data,
        "capex_growth": get_total_capex_growth_qrtly().data,
    }
    panel = pd.DataFrame(series)
    if not isinstance(panel.index, pd.PeriodIndex):
        panel.index = panel.index.to_period("Q-DEC")
    panel = panel.sort_index()
    panel = panel.loc[panel.index >= SAMPLE_START]
    return panel


# --- Result types ---


@dataclass
class NowcastResult:
    """BVAR nowcast result for a single target quarter."""

    target_quarter: pd.Period
    gdp_qoq: float
    gdp_tty: float
    gdp_qoq_70: tuple[float, float]
    gdp_qoq_90: tuple[float, float]
    gdp_tty_70: tuple[float, float]
    gdp_tty_90: tuple[float, float]
    n_vars: int
    n_lags: int
    panel: pd.DataFrame
    fit: BVARFit


# --- Helpers ---


def _detect_target_quarter(
    panel: pd.DataFrame,
    min_indicators: int = 1,
) -> tuple[pd.Period, bool, dict[str, list[str]]]:
    """Auto-detect target quarter for a live nowcast.

    Returns (target_quarter, is_hindcast, availability_report).

    Switches to forward-mode nowcasting as soon as ANY indicator is published
    for the next quarter (min_indicators=1 default). conditional_nowcast()
    handles partial panels gracefully — it conditions on whichever subset is
    available and falls back to the unconditional VAR forecast if zero are.

    is_hindcast=True means we're nowcasting the most recent published GDP
    quarter as a sanity check because no later quarter has any indicator
    data at all.

    availability_report is a dict with keys 'available' and 'missing' listing
    the indicators for the *next* quarter after the last published GDP.
    """
    gdp_obs = panel["gdp_growth"].dropna()
    last_gdp = gdp_obs.index[-1]
    next_q = last_gdp + 1

    indicator_cols = [c for c in panel.columns if c != "gdp_growth"]

    # Build the availability report for the next quarter (the one we'd
    # nowcast in forward mode if all indicators were ready)
    if next_q in panel.index:
        row = panel.loc[next_q, indicator_cols]
        available = [c for c in indicator_cols if pd.notna(row[c])]
        missing = [c for c in indicator_cols if pd.isna(row[c])]
    else:
        available = []
        missing = list(indicator_cols)
    report = {"available": available, "missing": missing, "next_quarter": next_q}

    candidates = panel.loc[panel.index > last_gdp]
    for q in candidates.index:
        n_available = candidates.loc[q, indicator_cols].notna().sum()
        if n_available >= min_indicators:
            return q, False, report

    # Fall back to hindcasting the most recent published GDP quarter
    return last_gdp, True, report


def _compute_tty(gdp_qoq: float, gdp: pd.Series, target_quarter: pd.Period) -> float:
    """Compute through-the-year growth from a single Q/Q nowcast."""
    last_gdp_level = gdp.iloc[-1]
    projected = last_gdp_level * np.exp(gdp_qoq / 100)
    q_minus_4 = target_quarter - 4
    gdp_4q_ago = gdp.loc[q_minus_4] if q_minus_4 in gdp.index else gdp.iloc[-4]
    return (projected / gdp_4q_ago - 1) * 100


# --- Core nowcast ---


def nowcast(
    target_quarter: pd.Period | None = None,
    panel: pd.DataFrame | None = None,
    gdp: pd.Series | None = None,
    n_lags: int = N_LAGS,
    lambda_tight: float = LAMBDA_TIGHT,
    lambda_cross: float = LAMBDA_CROSS,
    quiet: bool = False,
) -> NowcastResult:
    """Run the BVAR T-0 GDP nowcast.

    Args:
        target_quarter: The quarter to nowcast. If None, auto-detected.
        panel: Pre-loaded quarterly panel. If None, fetched from ABS.
        gdp: Pre-loaded GDP level series. If None, fetched.
        n_lags: VAR lag order.
        lambda_tight: Minnesota prior overall tightness.
        lambda_cross: Cross-variable shrinkage relative to own-variable.
        quiet: If True, suppress terminal output.

    Returns:
        NowcastResult with the conditional GDP forecast and prediction intervals.
    """
    if panel is None:
        panel = _load_panel()
    if gdp is None:
        gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()

    is_hindcast = False
    if target_quarter is None:
        target_quarter, is_hindcast, report = _detect_target_quarter(panel)
        if is_hindcast and not quiet:
            next_q = report["next_quarter"]
            available = report["available"]
            missing = report["missing"]
            print()
            print(f"  ⓘ Cannot nowcast {next_q} yet — too few indicators published.")
            print(f"    Available for {next_q} ({len(available)}/{len(available)+len(missing)}):")
            if available:
                for ind in available:
                    print(f"      ✓ {ind}")
            else:
                print("      (none)")
            print(f"    Still missing for {next_q}:")
            for ind in missing:
                print(f"      ✗ {ind}")
            print(f"    Falling back to hindcast of last published quarter ({target_quarter}).")
            print()

    # Training data: everything strictly before target_quarter
    train = panel.loc[panel.index < target_quarter]

    fit = fit_bvar_minnesota(
        train,
        n_lags=n_lags,
        lambda_tight=lambda_tight,
        lambda_cross=lambda_cross,
    )

    # Contemporaneous values for the target quarter (everything except GDP).
    # Indicators not yet published are NaN; conditional_nowcast() will use
    # whichever subset is available and report any that are missing.
    if target_quarter in panel.index:
        target_row = panel.loc[target_quarter].drop("gdp_growth")
    else:
        # Build a row of NaNs so the conditional update degrades gracefully
        target_row = pd.Series(
            {c: np.nan for c in panel.columns if c != "gdp_growth"},
            name=target_quarter,
        )

    cond_mean, cond_std, used_indicators = conditional_nowcast(
        fit,
        history=panel,
        target_quarter=target_quarter,
        target_var="gdp_growth",
        contemporaneous=target_row,
    )

    # Report any indicators that are missing for the target quarter
    all_indicators = [c for c in panel.columns if c != "gdp_growth"]
    missing_indicators = [c for c in all_indicators if c not in used_indicators]
    if missing_indicators and not quiet:
        print()
        print(f"  ⚠ {len(missing_indicators)} of {len(all_indicators)} indicators "
              f"missing for {target_quarter}:")
        for c in missing_indicators:
            print(f"      ✗ {c}")
        if used_indicators:
            print(f"    Conditioning on the {len(used_indicators)} available indicators:")
            for c in used_indicators:
                print(f"      ✓ {c}")
        else:
            print("    No contemporaneous indicators — falling back to unconditional VAR forecast.")
        print()

    # Prediction intervals from the conditional Gaussian
    ci_70 = (cond_mean - 1.04 * cond_std, cond_mean + 1.04 * cond_std)
    ci_90 = (cond_mean - 1.645 * cond_std, cond_mean + 1.645 * cond_std)

    # Convert to TTY
    gdp_truncated = gdp.loc[gdp.index < target_quarter]
    tty = _compute_tty(cond_mean, gdp_truncated, target_quarter)
    tty_70 = (
        _compute_tty(ci_70[0], gdp_truncated, target_quarter),
        _compute_tty(ci_70[1], gdp_truncated, target_quarter),
    )
    tty_90 = (
        _compute_tty(ci_90[0], gdp_truncated, target_quarter),
        _compute_tty(ci_90[1], gdp_truncated, target_quarter),
    )

    result = NowcastResult(
        target_quarter=target_quarter,
        gdp_qoq=cond_mean,
        gdp_tty=tty,
        gdp_qoq_70=ci_70,
        gdp_qoq_90=ci_90,
        gdp_tty_70=tty_70,
        gdp_tty_90=tty_90,
        n_vars=fit.n_vars,
        n_lags=n_lags,
        panel=train,
        fit=fit,
    )

    if not quiet:
        _print_summary(result)

    return result


# --- Output ---


def _print_summary(result: NowcastResult) -> None:
    """Print nowcast summary to terminal."""
    print("\n" + "=" * 70)
    print(f"  GDP NOWCAST (BVAR): {result.target_quarter}")
    print("=" * 70)

    print(f"\n  Q/Q growth:   {result.gdp_qoq:+.2f}%")
    print(f"    70% CI:     [{result.gdp_qoq_70[0]:+.2f}%, {result.gdp_qoq_70[1]:+.2f}%]")
    print(f"    90% CI:     [{result.gdp_qoq_90[0]:+.2f}%, {result.gdp_qoq_90[1]:+.2f}%]")

    print(f"\n  TTY growth:   {result.gdp_tty:+.2f}%")
    print(f"    70% CI:     [{result.gdp_tty_70[0]:+.2f}%, {result.gdp_tty_70[1]:+.2f}%]")
    print(f"    90% CI:     [{result.gdp_tty_90[0]:+.2f}%, {result.gdp_tty_90[1]:+.2f}%]")

    print(f"\n  BVAR(p={result.n_lags}) on {result.n_vars} quarterly variables")
    print(f"  Sample: {result.panel.dropna().index[0]} to {result.panel.dropna().index[-1]}")
    print(f"  Training observations: {len(result.panel.dropna())}")
    print(f"  Minnesota prior: λ_tight={LAMBDA_TIGHT}, λ_cross={LAMBDA_CROSS}, λ_decay={LAMBDA_DECAY}")

    print("\n" + "=" * 70)


# --- Plotting ---


def _plot_fan_chart(
    hist: pd.Series,
    nowcast_value: float,
    ci_70: tuple[float, float],
    ci_90: tuple[float, float],
    target_quarter: pd.Period,
    title: str,
    lfooter_value: str,
) -> None:
    """Plot a nowcast fan chart using mgplot layering."""
    # If we're hindcasting, drop the target quarter from history so the
    # nowcast line doesn't create a duplicate index point.
    hist_excluding_target = hist.loc[hist.index < target_quarter]
    recent = hist_excluding_target.iloc[-20:]

    nowcast_idx = pd.PeriodIndex([recent.index[-1], target_quarter], freq="Q-DEC")
    nowcast_line = pd.Series([recent.iloc[-1], nowcast_value], index=nowcast_idx)

    band_90 = pd.DataFrame({
        "lower": [recent.iloc[-1], ci_90[0]],
        "upper": [recent.iloc[-1], ci_90[1]],
    }, index=nowcast_idx)

    band_70 = pd.DataFrame({
        "lower": [recent.iloc[-1], ci_70[0]],
        "upper": [recent.iloc[-1], ci_70[1]],
    }, index=nowcast_idx)

    recent = recent.rename("GDP growth")
    nowcast_line = nowcast_line.rename("Nowcast")

    ax = mg.fill_between_plot(band_90, color="green", alpha=0.1, label="90% CI")
    mg.fill_between_plot(band_70, ax=ax, color="green", alpha=0.2, label="70% CI")
    mg.line_plot(recent, ax=ax, color=["navy"], width=2)
    mg.line_plot(nowcast_line, ax=ax, color=["green"], width=2, style="--", annotate=True, rounding=2)

    mg.finalise_plot(
        ax,
        title=title,
        ylabel="Per cent",
        rfooter="Source: ABS 5206.0",
        lfooter=f"Australia. BVAR nowcast: {lfooter_value}. {pd.Timestamp.now().strftime('%d %b %Y')}. ",
        y0=True,
        legend={"loc": "best", "fontsize": "x-small"},
        show=SHOW,
    )


def _plot_nowcast(result: NowcastResult, gdp_growth_hist: pd.Series) -> None:
    """Plot Q/Q nowcast vs GDP history with fan chart."""
    _plot_fan_chart(
        hist=gdp_growth_hist.dropna(),
        nowcast_value=result.gdp_qoq,
        ci_70=result.gdp_qoq_70,
        ci_90=result.gdp_qoq_90,
        target_quarter=result.target_quarter,
        title=f"GDP Growth BVAR Nowcast (Q/Q): {result.target_quarter}",
        lfooter_value=f"{result.gdp_qoq:+.2f}%",
    )


def _plot_nowcast_tty(result: NowcastResult, gdp: pd.Series) -> None:
    """Plot TTY nowcast vs GDP TTY history with fan chart."""
    tty_hist = ((gdp / gdp.shift(4)) - 1) * 100
    _plot_fan_chart(
        hist=tty_hist.dropna(),
        nowcast_value=result.gdp_tty,
        ci_70=result.gdp_tty_70,
        ci_90=result.gdp_tty_90,
        target_quarter=result.target_quarter,
        title=f"GDP Growth BVAR Nowcast (TTY): {result.target_quarter}",
        lfooter_value=f"{result.gdp_tty:+.2f}%",
    )


def _plot_growth(result: NowcastResult, gdp: pd.Series) -> None:
    """Bar (Q/Q) + line (TTY) chart of GDP growth with nowcast appended."""
    gdp_growth = (np.log(gdp).diff(1) * 100).dropna()
    tty = ((gdp / gdp.shift(4)) - 1) * 100

    # When hindcasting, replace the target quarter's actual with the nowcast
    # rather than appending — otherwise we duplicate it.
    gdp_growth = gdp_growth.loc[gdp_growth.index < result.target_quarter]
    tty = tty.loc[tty.index < result.target_quarter]
    gdp_growth.loc[result.target_quarter] = result.gdp_qoq
    tty.loc[result.target_quarter] = result.gdp_tty

    growth_df = pd.DataFrame({
        "Annual Growth": tty,
        "Quarterly Growth": gdp_growth,
    }).iloc[-20:]

    mg.growth_plot_finalise(
        growth_df,
        title=f"GDP Growth BVAR Nowcast: {result.target_quarter}",
        ylabel="Per cent",
        rfooter="Source: ABS 5206.0",
        lfooter=f"Australia. BVAR Q/Q: {result.gdp_qoq:+.2f}%, TTY: {result.gdp_tty:+.2f}%. "
                f"{pd.Timestamp.now().strftime('%d %b %Y')}. ",
        legend={"loc": "best", "fontsize": "x-small"},
        show=SHOW,
    )


# --- Live entry point ---


def run_nowcast() -> NowcastResult:
    """Run the live BVAR GDP nowcast."""
    result = nowcast()

    gdp = get_gdp(gdp_type="CVM", seasonal="SA").data.dropna()
    gdp_growth = (np.log(gdp).diff(1) * 100)

    mg.set_chart_dir(CHART_DIR)
    mg.clear_chart_dir()
    _plot_nowcast(result, gdp_growth)
    _plot_nowcast_tty(result, gdp)
    _plot_growth(result, gdp)

    return result


if __name__ == "__main__":
    run_nowcast()
