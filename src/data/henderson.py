"""Henderson Moving Average implementation.

Formulas from:
- ABS (2003), 'A Guide to Interpreting Time Series', page 41
- Mike Doherty (2001), 'The Surrogate Henderson Filters in X-11',
  Aust, NZ J of Stat. 43(4), pp901-999
"""

import numpy as np
import pandas as pd

_hma_cache: dict[int | tuple[int, int], np.ndarray] = {}


def hma_symmetric_weights(n: int) -> np.ndarray:
    """Derive an n-term array of symmetric Henderson Moving Average weights.

    Formula from ABS (2003), 'A Guide to Interpreting Time Series', page 41.

    Args:
        n: Number of terms (must be odd, >= 3)

    Returns:
        numpy array of symmetric Henderson weights indexed from 0 to n-1

    """
    if n in _hma_cache:
        return _hma_cache[n]

    # calculate the constant denominator and terms
    m: int = int((n - 1) // 2)  # the mid point - n must be odd
    m1: int = (m + 1) * (m + 1)
    m2: int = (m + 2) * (m + 2)
    d: float = float(
        8 * (m + 2) * (m2 - 1) * (4 * m2 - 1) * (4 * m2 - 9) * (4 * m2 - 25)
    )
    m3: int = (m + 3) * (m + 3)

    # calculate the weights
    sym_weights = np.repeat(np.nan, n)  # indexed from 0 to n-1
    for j in range(m + 1):
        j2: int = j * j
        a_weight: float = (
            315 * (m1 - j2) * (m2 - j2) * (m3 - j2) * (3 * m2 - 11 * j2 - 16)
        ) / d
        sym_weights[(m + j)] = a_weight
        if j > 0:
            sym_weights[(m - j)] = a_weight

    sym_weights.flags.writeable = False  # make quasi-immutable
    _hma_cache[n] = sym_weights
    return sym_weights


def hma_asymmetric_weights(m: int, sym_weights: np.ndarray) -> np.ndarray:
    """Calculate asymmetric end-weights for Henderson moving average.

    Formula from Mike Doherty (2001), 'The Surrogate Henderson Filters in X-11',
    Aust, NZ J of Stat. 43(4), 2001, pp901-999; see formula (1) on page 903.

    Args:
        m: Number of asymmetric weights sought (m < len(sym_weights))
        sym_weights: Array of symmetrical henderson weights

    Returns:
        numpy array of asymmetrical weights, indexed from 0 to m-1

    """
    n: int = len(sym_weights)
    cache_key = (n, m)
    if cache_key in _hma_cache:
        return _hma_cache[cache_key]

    # Build up Doherty's formula (1) from the top of page 903
    sum_residual: float = sym_weights[range(m, n)].sum() / float(m)

    sum_end: float = 0.0
    for i in range(m + 1, n + 1):
        sum_end += (float(i) - ((m + 1.0) / 2.0)) * sym_weights[i - 1]

    # beta squared / sigma squared - formula at the bottom of page 904
    ic: float = 1.0
    if 13 <= n < 15:
        ic = 3.5
    elif n >= 15:
        ic = 4.5
    b2s2: float = (4.0 / np.pi) / (ic * ic)

    denominator: float = 1.0 + ((m * (m - 1.0) * (m + 1.0) / 12.0) * b2s2)
    asym_wts: np.ndarray = np.repeat(np.nan, m)
    for r in range(m):
        numerator = ((r + 1.0) - (m + 1.0) / 2.0) * b2s2
        asym_wts[r] = (
            sym_weights[r] + sum_residual + (numerator / denominator) * sum_end
        )

    asym_wts.flags.writeable = False
    _hma_cache[cache_key] = asym_wts
    return asym_wts


def hma(series: pd.Series, n: int) -> pd.Series:
    """Calculate an n-term Henderson Moving Average.

    The Henderson moving average is used by the ABS for trend estimation.
    It provides smooth trend estimates with proper handling of endpoints
    using asymmetric weights.

    Args:
        series: pandas Series (must be ordered, contiguous, without NaN)
        n: Number of terms (must be odd, >= 3)

    Returns:
        pandas Series with Henderson moving average applied

    Raises:
        TypeError: If series is not a pandas Series or n is not an int
        ValueError: If series contains NaN, n < 3, n is even, or series too short

    """
    if not isinstance(series, pd.core.series.Series):
        raise TypeError("The series argument must be a pandas Series")
    if series.isna().sum() > 0:
        raise ValueError("The series argument must not contain missing data")
    if not isinstance(n, int):
        raise TypeError("The n argument must be an integer")
    minimum_n: int = 3
    if n < minimum_n:
        raise ValueError(f"The n argument must be >= {minimum_n}")
    if n % 2 == 0:
        raise ValueError("The n argument must be odd")
    if len(series) < n:
        raise ValueError(f"The series (len={len(series)}) is shorter than n")

    sym_weights = hma_symmetric_weights(n)
    mid_point: int = int((n - 1) // 2)

    # Middle section using rolling window
    henderson = series.rolling(n, min_periods=n, center=True).apply(
        lambda x: x.mul(sym_weights).sum()
    )

    # Tails using asymmetric weights
    for i in range(1, mid_point + 1):
        asym_wts = hma_asymmetric_weights(mid_point + i, sym_weights)
        henderson.iloc[i - 1] = (series.iloc[: (i + mid_point)] * asym_wts[::-1]).sum()
        henderson.iloc[-i] = (series.iloc[(-mid_point - i) :] * asym_wts).sum()

    return henderson


# --- Testing ---

if __name__ == "__main__":
    print("Testing Henderson Moving Average...\n")

    # Check symmetric weights
    N = 9
    weights = hma_symmetric_weights(N)
    print(f"{N} symmetric weights: {weights}")
    print(f"Sum of weights: {weights.sum()} (should be 1.0)\n")

    # Check asymmetric weights
    M = 7
    asym_weights = hma_asymmetric_weights(M, weights)
    print(f"{M} asymmetric weights: {asym_weights}")
    print(f"Sum of weights: {asym_weights.sum()} (should be 1.0)\n")

    # Test on simple data
    LENGTH = 30
    test_series = pd.Series(range(LENGTH))
    result = hma(test_series, N)
    print(f"HMA of [0, 1, 2, ..., {LENGTH-1}]:")
    print(result.head(10))
