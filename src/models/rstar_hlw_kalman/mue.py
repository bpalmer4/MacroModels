"""Stock-Watson median-unbiased estimation of a signal-to-noise ratio.

WHAT IT IS FOR. Maximum likelihood estimates of the innovation variances of
trend growth and z are biased towards zero and in practice come back at
exactly zero: the pile-up problem, which LW2001 report hitting directly.
Rather than estimate those variances, HLW estimate the RATIO of each to the
variance it competes with, by a route that does not go through the likelihood
at all.

The route is a structural break test. Hold a state constant that should
really move, and its movement is forced into the residual of whatever
equation it belonged to, where it shows up as a break in the mean at an
unknown date. The size of that break maps to the signal-to-noise ratio the
state would have needed. Stock and Watson (1998) tabulate that mapping for
three break statistics; HLW use the exponential Wald of Andrews and Ploberger
(1994).

THE TABLE BELOW IS UNVERIFIED AGAINST THE PAPER. It is transcribed from the
source of the `rStar` R package, cross-read between its stage 1 and stage 2
files, which agree:

    https://rdrr.io/github/JannesRed/rStar/src/R/median.unbiased.estimator.stage1.R
    https://rdrr.io/github/JannesRed/rStar/src/R/median.unbiased.estimator.stage2.R

Stock and Watson (1998) Table 3 itself has not been consulted, and neither
has the Federal Reserve Bank of New York's own replication code at
https://www.newyorkfed.org/research/policy/rstar. Every number that matters
lives in this one module so it can be checked in one place. If it is wrong,
everything downstream of it is wrong.

AND THE PROCEDURE IS DISPUTED. Buncic (arXiv 2002.11583) argues HLW apply
median-unbiased estimation to a misspecified second-stage model and so
inflate lambda_z. This module implements what they do, not what the critique
would prefer, because the point of the package is to follow their process.
"""

import numpy as np
from scipy.special import logsumexp

# Stock and Watson (1998) Table 3, exponential Wald. The i-th entry is the
# critical value for a signal-to-noise ratio of i on a grid running 0 to 30,
# which is scaled to a ratio by dividing by the sample size (see `lambda_from`).
# UNVERIFIED: see the module docstring for provenance.
EW_CRITICAL_VALUES = (
    0.426, 0.476, 0.516, 0.661, 0.826, 1.111, 1.419, 1.762, 2.355, 2.910,
    3.413, 3.868, 4.925, 5.684, 6.670, 7.690, 8.477, 9.191, 10.693, 12.024,
    13.089, 14.440, 16.191, 17.332, 18.699, 20.464, 21.667, 23.851, 25.538,
    26.762, 27.874,
)

# Mean Wald and QLR from the same table, kept so the choice of statistic can
# be swept rather than asserted. HLW use the exponential Wald.
MW_CRITICAL_VALUES = (
    0.689, 0.757, 0.806, 1.015, 1.234, 1.632, 2.018, 2.390, 3.081, 3.699,
    4.222, 4.776, 5.767, 6.586, 7.703, 8.683, 9.467, 10.101, 11.639, 13.039,
    13.900, 15.214, 16.806, 18.330, 19.020, 20.562, 21.837, 24.350, 26.248,
    27.089, 27.758,
)
QLR_CRITICAL_VALUES = (
    3.198, 3.416, 3.594, 4.106, 4.848, 5.689, 6.682, 7.626, 9.160, 10.660,
    11.841, 13.098, 15.451, 17.094, 19.423, 21.682, 23.342, 24.920, 28.174,
    30.736, 33.313, 36.109, 39.673, 41.955, 45.056, 48.647, 50.983, 55.514,
    59.278, 61.311, 64.016,
)

# Andrews' (1993) conventional trimming: break dates in the outer 15% at each
# end are not tested, because a break too near an end is not identified.
DEFAULT_TRIM = 0.15


def break_statistics(
    y: np.ndarray,
    regressors: np.ndarray | None = None,
    trim: float = DEFAULT_TRIM,
) -> np.ndarray:
    """Return the t-statistic on an intercept shift at every candidate date.

    `y` is the series whose mean may break. `regressors` are any variables
    held alongside the intercept, which is how stage 2 tests for a shift in an
    equation's intercept rather than in a raw mean. A column of ones is always
    added.
    """
    n = len(y)
    base = np.ones((n, 1)) if regressors is None else np.column_stack(
        [np.ones(n), np.asarray(regressors, dtype=float).reshape(n, -1)],
    )

    lo = int(np.floor(n * trim))
    hi = int(np.ceil(n * (1.0 - trim)))
    if hi - lo < 1:
        raise ValueError(
            f"a sample of {n} trimmed at {trim:.0%} leaves no candidate break "
            f"dates; shorten the trim or lengthen the sample",
        )

    k = base.shape[1] + 1
    stats = np.empty(hi - lo)
    for j, tau in enumerate(range(lo, hi)):
        shift = np.zeros((n, 1))
        shift[tau:] = 1.0
        x = np.hstack([base, shift])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        sigma2 = float(resid @ resid) / (n - k)
        xtx_inv = np.linalg.pinv(x.T @ x)
        stats[j] = beta[-1] / np.sqrt(sigma2 * xtx_inv[-1, -1])
    return stats


def exponential_wald(stats: np.ndarray) -> float:
    """Andrews-Ploberger exponential Wald, `log(mean(exp(W/2)))` with `W = t^2`.

    Computed through `logsumexp` rather than directly: a t-statistic of 30
    would overflow `exp(450)`, and break statistics do reach that size when a
    break is large.
    """
    return float(logsumexp(np.asarray(stats, dtype=float) ** 2 / 2.0) - np.log(len(stats)))


def mean_wald(stats: np.ndarray) -> float:
    """Andrews-Ploberger mean Wald, the average of `t^2`."""
    return float(np.mean(np.asarray(stats, dtype=float) ** 2))


def quandt_likelihood_ratio(stats: np.ndarray) -> float:
    """Quandt's supremum Wald, the largest `t^2` over candidate dates."""
    return float(np.max(np.asarray(stats, dtype=float) ** 2))


def lambda_from(
    statistic: float,
    n_obs: int,
    table: tuple[float, ...] = EW_CRITICAL_VALUES,
    *,
    denominator_offset: int = 0,
) -> float:
    """Map a break statistic to a median-unbiased signal-to-noise ratio.

    Linear interpolation into `table`, whose entries sit on a grid of 0 to 30,
    then division by the sample size to turn the grid value into a ratio.

    `denominator_offset` is 1 in HLW's stage 1, which divides by T-1, and 0 in
    stage 2, which divides by T. The asymmetry is theirs, not a typo here.

    A statistic at or below the first entry gives exactly zero, which is the
    procedure conceding the state does not move. Above the last entry the grid
    is capped at 30, and the caller should know the table has been exhausted
    rather than read the result as a measurement.
    """
    values = np.asarray(table, dtype=float)
    denominator = n_obs - denominator_offset
    if denominator <= 0:
        raise ValueError(f"sample of {n_obs} is too short for offset {denominator_offset}")

    if statistic <= values[0]:
        return 0.0
    if statistic > values[-1]:
        return float(len(values) - 1) / denominator

    index = int(np.searchsorted(values, statistic, side="left")) - 1
    lower, upper = values[index], values[index + 1]
    grid = index + (statistic - lower) / (upper - lower)
    return float(grid) / denominator
