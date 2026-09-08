"""Observation assembly for the ustar model.

Four series on one aligned quarterly sample:

- unemployment rate, %                 (ABS 1364.0.15.003 via `labour_force`)
- output gap and its posterior sd      (a completed `ystar` run)
- trimmed mean inflation, annual %     (ABS 6401.0)
- inflation expectations, %            (the `expectations` model's saved output)

Nothing here is estimated. The output gap arrives as two arrays — a per-quarter
mean and sd — rather than a single path, so the model can carry `ystar`'s
uncertainty rather than treating a median as data.

Run order is therefore `expectations` -> `ystar` -> `ustar`. Both inputs
are read from saved output; neither is re-estimated.
"""

import numpy as np
import pandas as pd

from src.data.expectations_model import get_model_expectations_unanchored
from src.data.gscpi_live import get_gscpi_qrtly_live
from src.data.import_prices import get_import_price_growth_lagged_annual
from src.data.inflation import get_trimmed_mean_qrtly
from src.data.labour_force import get_unemployment_rate_qrtly
from src.models.common.sources import SourceSet
from src.models.ystar.results import load_results

_NAME_WIDTH = 24


def _gap_moments(
    gap_source: str,
    gap_prefix: str,
    sources: SourceSet,
) -> tuple[pd.Series, pd.Series]:
    """Return the per-quarter posterior mean and sd of the given output gap.

    The sd is the whole point of returning moments rather than a path: it is
    what the measurement-error prior uses, and it differs sharply between the
    two sources. The defined gap's only uncertainty is uncertainty in `c`; the
    actual gap also carries potential's own, so its band is several times wider.
    """
    try:
        results = load_results(prefix=gap_prefix)
    except FileNotFoundError as exc:
        # The raw failure is an HDF5 "unable to open file" several frames down,
        # which says nothing about which model was missing or why this one
        # wanted it. The run order is documented in six other places; this is
        # the one a person actually meets.
        raise FileNotFoundError(
            f"No saved ystar run found under prefix {gap_prefix!r}. "
            "The u* model takes the output gap as given rather than estimating "
            "it, so ystar must be run first: ./run-ystar.sh "
            "(and ./run-expectations.sh before that, for the expectations "
            "series this model also reads).",
        ) from exc

    # The gap arrives as data, so whatever the run that produced it was built
    # from is an input to this model too.
    recorded = SourceSet.from_records(results.constants.get("sources"))
    if recorded is not None:
        for source, cat in recorded.records:
            sources.add(source, cat)

    posterior = (
        results.output_gap_posterior()
        if gap_source == "defined"
        else results.actual_output_gap_posterior()
    )
    return posterior.mean(axis=1), posterior.std(axis=1)


def _gscpi_lagged(index: pd.Index, sources: SourceSet, lag: int = 2) -> pd.Series:
    """Return the GSCPI, lagged, with no COVID mask.

    Two departures from `nairu`, both deliberate.

    **No mask.** The NAIRU model keeps GSCPI only over 2020Q1-2023Q2 and zeroes
    it elsewhere, which leaves 14 non-zero quarters out of the sample. Unmasked,
    the index is a regressor throughout, so the coefficient is identified on the
    whole history rather than on the pandemic alone, and the model can say
    whether supply-chain pressure mattered outside it.

    **Live source.** With the mask, the static workbook's staleness was harmless
    because it stopped after the masked window closed. Unmasked, it is not: the
    checked-in file ends 2024Q1 while the published series runs to 2026Q2, so
    nine quarters at the business end would be missing.

    Zero-filled before the index begins in 1998Q1. GSCPI is standardised in
    deviations from its own mean, so zero is neutral pressure, not a gap in the
    data, and the alternative would truncate the sample by five years.
    """
    gscpi = sources.take(get_gscpi_qrtly_live(), "GSCPI (live, lagged)", key="gscpi").astype(float)
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
    gap_source: str = "defined",
    gap_prefix: str = "ystar",
    include_phillips: bool = True,
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], pd.PeriodIndex, pd.DataFrame, SourceSet]:
    """Build observation arrays for ustar estimation.

    Returns:
        Tuple of:
          - obs: dict of numpy arrays keyed by variable name
          - obs_index: the aligned PeriodIndex
          - chart_obs: DataFrame of the same series, for charting
          - sources: the providers behind those series, for the chart footers

    """
    sources = SourceSet()
    gap_mean, gap_sd = _gap_moments(gap_source, gap_prefix, sources)

    columns = {
        "u": sources.take(get_unemployment_rate_qrtly(), "Unemployment rate", key="u"),
        "gap_mean": gap_mean,
        "gap_sd": gap_sd,
    }
    sources.note("gap_mean", f"Output gap ({gap_source})")
    sources.note("gap_sd", "Output gap sd")

    if include_phillips:
        # Quarterly inflation, the unanchored expectations series, and the two
        # supply shocks. The target is a constant, not a series, so it is not
        # loaded here.
        #
        # Unanchored, not Target Anchored: the anchored series is constructed
        # with a 2.5% anchor post-1998, so its distance from 2.5 is near zero by
        # construction and the excess term would be testing nothing. The
        # unanchored median is the one that can actually drift away.
        columns["pi"] = sources.take(get_trimmed_mean_qrtly(), "Trimmed mean (qtrly)", key="pi")
        try:
            columns["pi_exp"] = sources.take(
                get_model_expectations_unanchored(), "Expectations", key="pi_exp",
            ).astype(float)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "No saved expectations model output found. The Phillips curve "
                "reads the unanchored expectations series, so the expectations "
                "model must be run first: ./run-expectations.sh. To estimate u* "
                "without it, use --no-phillips (Okun only).",
            ) from exc
        columns["d4pm"] = sources.take(
            get_import_price_growth_lagged_annual(), "Import price growth", key="d4pm",
        )
        columns["gscpi"] = _gscpi_lagged(columns["pi"].index, sources)

    if verbose:
        print("Input series coverage:")
        for key, series in columns.items():
            clean = series.dropna()
            label = sources.label(key)
            print(f"  {label:<{_NAME_WIDTH}} {clean.index.min()} -> {clean.index.max()}  n={len(clean)}")

    df = _align(columns, start, end)

    if verbose:
        print(f"\nAligned sample: {df.index.min()} to {df.index.max()}  ({len(df)} quarters)")

    obs = {name: df[name].to_numpy(dtype=float) for name in df.columns}
    index = df.index
    if not isinstance(index, pd.PeriodIndex):
        raise TypeError("aligned observations must carry a PeriodIndex")

    return obs, index, df, sources
