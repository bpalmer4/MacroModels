"""Observation assembly for the joint y*/u* model.

Seven series on one aligned quarterly sample:

- log GDP x 100                        (ABS 5206.0, chain volume, SA)
- trimmed mean inflation, y/y %        (ABS 6401.0)  -> defines the gap
- trimmed mean inflation, q/q %        (ABS 6401.0)  -> the Phillips curve's LHS
- unemployment rate, %                 (ABS 1364.0.15.003 via `labour_force`)
- inflation expectations, %            (the `expectations` model's saved output)
- import price growth, lagged annual   (ABS 6457.0)
- GSCPI, lagged                        (live series)

**Two inflation horizons, and it is a live specification choice rather than
housekeeping.** `ystar` defines the gap on the four-quarter rate, because there
inflation is a regressor and never a dependent variable, so overlapping
observations cost nothing and "at target" is an annual concept. `ustar`
estimates the Phillips curve on the quarterly rate, because there inflation is
the dependent variable and a four-quarter series shares three of its four
quarters with its own lag.

Here inflation is both at once, which neither parent had to face. Keeping the
horizons apart is the only thing stopping the Phillips curve's regressor from
containing its own left-hand side exactly. On the quarterly basis the gap is
c·(4·pi_q - anchor), an exact multiple of the Phillips curve's dependent
variable, so that correlation is 1.0 by construction. On the annual basis it is
0.828, measured on this sample.

That is a reason to prefer "annual", not a reason to think the problem is
handled. The correlation is not the same quantity as the contamination share of
gamma_pi, which in `ustar` is 21.4% of the regressor's variance, and nothing
here has established what that share becomes under either basis.
`ModelConfig.gap_pi_basis` exists so the comparison can be run rather than
argued.

Unlike `ustar`, nothing here is read from a completed model run except
expectations. The output gap is estimated rather than imported, which is the
point of the exercise.
"""

import numpy as np
import pandas as pd

from src.data.expectations_model import get_model_expectations_unanchored
from src.data.gdp import get_log_gdp
from src.data.gscpi_live import get_gscpi_qrtly_live
from src.data.import_prices import get_import_price_growth_lagged_annual
from src.data.inflation import get_trimmed_mean_annual, get_trimmed_mean_qrtly
from src.data.labour_force import get_unemployment_rate_qrtly
from src.models.common.sources import SourceSet

_NAME_WIDTH = 34


def _gscpi_lagged(index: pd.Index, sources: SourceSet, lag: int = 2) -> pd.Series:
    """Return the GSCPI, lagged, unmasked, from the live source.

    Both choices are `ustar`'s and the reasons carry over unchanged. Unmasked,
    so the coefficient is identified on the whole history rather than on the
    pandemic alone. Live, because the checked-in workbook stops at 2024Q1 while
    the published series runs past 2026, and unmasked those quarters matter.

    Zero-filled before the index begins in 1998Q1: GSCPI is standardised in
    deviations from its own mean, so zero is neutral pressure rather than a
    hole in the data, and the alternative would cost five years of sample.
    """
    gscpi = sources.take(get_gscpi_qrtly_live(), "GSCPI, lagged", key="gscpi").astype(float)
    return gscpi.shift(lag).reindex(index).fillna(0.0)


def _align(columns: dict[str, pd.Series], start: str | None, end: str | None) -> pd.DataFrame:
    """Put the series on one quarterly index, drop partial rows, trim the sample."""
    df = pd.DataFrame(columns)

    period_index = df.index
    if not isinstance(period_index, pd.PeriodIndex):
        period_index = pd.PeriodIndex(period_index, freq="Q")
    df.index = period_index.asfreq("Q")
    df = df.dropna()

    if start:
        df = df.loc[df.index >= pd.Period(start, "Q")]
    if end:
        df = df.loc[df.index <= pd.Period(end, "Q")]

    if df.empty:
        raise ValueError(f"No observations remain for start={start!r}, end={end!r}")

    return df


def build_observations(
    start: str | None = "1993Q1",
    end: str | None = None,
    *,
    gap_pi_basis: str = "annual",
    include_phillips: bool = True,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex, pd.DataFrame, SourceSet]:
    """Build observation arrays for joint estimation.

    `gap_pi_basis` selects the series the gap is defined on. "quarterly" is the
    quarterly rate multiplied by four, so it sits on the anchor's scale; see
    `ModelConfig.gap_pi_basis` for why this is a switch and not a settled
    choice.

    Returns:
        Tuple of:
          - obs: dict of numpy arrays keyed by variable name
          - obs_index: the aligned PeriodIndex
          - chart_obs: DataFrame of the same series, for charting
          - sources: the providers behind those series, for the chart footers

    """
    sources = SourceSet()

    if gap_pi_basis == "quarterly":
        gap_pi = sources.take(get_trimmed_mean_qrtly(), "trimmed mean q/q ann.", key="pi_gap") * 4.0
    else:
        gap_pi = sources.take(get_trimmed_mean_annual(), "trimmed mean y/y", key="pi_gap")

    columns: dict[str, pd.Series] = {
        "log_gdp": sources.take(get_log_gdp(), "log GDP", key="log_gdp"),
        "pi_gap": gap_pi,
        "u": sources.take(get_unemployment_rate_qrtly(), "unemployment rate", key="u"),
    }

    if include_phillips:
        columns["pi_qtr"] = sources.take(get_trimmed_mean_qrtly(), "trimmed mean q/q", key="pi_qtr")
        try:
            columns["pi_exp"] = sources.take(
                get_model_expectations_unanchored(), "expectations", key="pi_exp",
            ).astype(float)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "No saved expectations model output found. The Phillips curve reads the "
                "unanchored expectations series, so the expectations model must be run "
                "first: ./run-expectations.sh. To estimate without it, use --no-phillips, "
                "but note that u* is then a trend through unemployment rather than a NAIRU.",
            ) from exc
        columns["d4pm"] = sources.take(
            get_import_price_growth_lagged_annual(), "import price growth", key="d4pm",
        )
        columns["gscpi"] = _gscpi_lagged(columns["pi_qtr"].index, sources)

    if verbose:
        print("Input series coverage:")
        for key, series in columns.items():
            clean = series.dropna()
            print(
                f"  {sources.label(key):<{_NAME_WIDTH}} {clean.index.min()} -> "
                f"{clean.index.max()}  n={len(clean)}",
            )

    df = _align(columns, start, end)

    if verbose:
        print(f"\nAligned sample: {df.index.min()} to {df.index.max()}  ({len(df)} quarters)")

    obs = {name: df[name].to_numpy(dtype=float) for name in df.columns}
    index = df.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError("aligned observations must carry a PeriodIndex")

    return obs, index, df, sources
