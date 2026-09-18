"""u* read off the inverted Phillips curve, quarter by quarter, then smoothed.

**Nothing is estimated here.** The state-space model in `estimate.py` puts a
law of motion on u* and lets a weak likelihood fill it in; this does the
opposite. It asks what u* each quarter's inflation implies on its own, and
then makes a slow series out of those readings by smoothing them.

    u*_t = u_t x (1 + (pi_t - pi^e_t) / beta)

`pi^e` is the regime-dependent expectation built in `observations.py`: adaptive
anchored, then a trailing average, then measured from 1983Q1. `beta` is the divisor and it is
NOT estimated here, which is this module's one borrowed number. See
`resolve_beta`.

**Why this is not just a smoothed unemployment rate.** Smoothing the identity
above gives

    smooth(u*) = smooth(u) + smooth(u x (pi - pi^e)) / beta

so the second term is the entire information content. On the 1959-2026 sample
its standard deviation is 0.74, 0.51 and 0.38 times the first term's at
windows of 9, 21 and 41 quarters. Substantial, and it shrinks as the window
grows, which is the finding below.

**Smoothing harder makes it worse.** A long moving average of unemployment
does not remove the 1990s recession, because the recession outlasts the
window, so cyclical unemployment leaks through as if it were u*. The surprise
term is exactly what corrects that, and over-smoothing kills the correction.
Against the four readings `long_run_ustar` derives by a different method, a
9-quarter window wins on three of four and a 41-quarter window is worse on
three of four while its 2025-26 reading runs from 5.26 up to 7.02.

So "slow moving" and "accurate" pull against each other, and the default
window is short for that reason rather than by oversight.
"""

import numpy as np
import pandas as pd

from src.data.henderson import hma
from src.models.regime_ustar.config import ModelConfig

# Henderson terms to report. Odd, as the filter requires. 9 quarters is a bit
# over two years and is where the sweep lands; the longer two are carried so a
# reader sees the degradation rather than being told about it.
DEFAULT_WINDOWS: tuple[int, ...] = (9, 21, 41)
DEFAULT_WINDOW: int = 9


def implied_ustar(frame: pd.DataFrame, beta: float) -> pd.Series:
    """Return the per-quarter inversion, unsmoothed.

    Wild by construction: it divides an inflation surprise by a coefficient
    near a half and adds the result to the unemployment rate. The smoothing is
    what makes it readable, and the raw series is kept so a chart can show how
    much of the answer the smoothing supplied.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive to invert the Phillips curve, got {beta}")
    return (frame["u"] * (1.0 + frame["surprise"] / beta)).rename("implied")


def smoothed(implied: pd.Series, window: int) -> pd.Series:
    """Return the Henderson trend through the inversion.

    Henderson rather than a centred mean because it carries asymmetric end
    weights, so the series reaches the last observed quarter instead of
    stopping half a window short. That matters here: the current reading is the
    one anybody will ask about.
    """
    return hma(implied.dropna(), window).rename(f"hma{window}")


def resolve_beta(config: ModelConfig, override: float | None = None) -> tuple[float, str]:
    """Return the Phillips slope to divide by, and where it came from.

    This module estimates nothing, so `beta` has to arrive from outside. In
    order of preference: an explicit override, then the saved state-space
    trace, then the failure. There is no default constant, deliberately: a
    number baked in here would be quoted as if it had been measured.

    `beta` is the divisor, so it scales every departure from smoothed
    unemployment. A smaller `beta` widens them all. Sweep it.
    """
    if override is not None:
        return override, f"supplied ({override:.3f})"

    import arviz as az  # noqa: PLC0415 — only needed on this path

    path = config.output_dir / f"{config.prefix}_trace.nc"
    if not path.is_file():
        raise FileNotFoundError(
            f"no saved trace at {path} to take beta from: run the model first, or pass --beta",
        )
    trace = az.from_netcdf(str(path))
    posterior = getattr(trace, "posterior", None)
    if posterior is None:
        raise TypeError(f"the trace at {path} has no posterior group")
    beta = float(np.median(np.asarray(posterior["beta"])))
    return beta, f"state-space posterior median ({beta:.3f})"


def build(frame: pd.DataFrame, beta: float, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """Return the raw inversion, each Henderson trend, and the two input series."""
    implied = implied_ustar(frame, beta)
    out = pd.DataFrame({"u": frame["u"], "implied": implied})
    for window in windows:
        out[f"hma{window}"] = smoothed(implied, window)
    return out


def information_share(frame: pd.DataFrame, beta: float, windows: tuple[int, ...]) -> pd.DataFrame:
    """Return, per window, how much of the smoothed series is NOT smoothed unemployment.

    The decomposition of the identity in this module's docstring. `ratio` near
    zero is the failure mode: it would mean the inflation data contributed
    nothing and the answer is a moving average of the unemployment rate.
    """
    rows = []
    for window in windows:
        su = hma(frame["u"].dropna(), window)
        ss = hma((frame["u"] * frame["surprise"] / beta).dropna(), window)
        rows.append({
            "window": window,
            "years": round(window / 4, 1),
            "sd smooth(u)": float(su.std()),
            "sd correction": float(ss.std()),
            "ratio": float(ss.std() / su.std()),
            "corr with smooth(u)": float((su + ss).corr(su)),
        })
    return pd.DataFrame(rows)
