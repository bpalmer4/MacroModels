"""Command-line entry point: the real TWI gap from commodity prices."""

import argparse

import mgplot as mg

from src.data.cash_rate import get_cash_rate_monthly
from src.models.twi_gap import CHART_DIR, DARK_SD, NEUTRAL_SD
from src.models.twi_gap.analysis import band_shares, build_frame, fit_gap, robustness
from src.models.twi_gap.charts import backplane_chart

DECIMALS = 3
RECENT = 6  # quarters of the gap printed


def main() -> None:
    """Print the fit and the robustness check, and write the chart."""
    parser = argparse.ArgumentParser(description="The real TWI gap from commodity prices")
    parser.add_argument("--chart-dir", default=None, help="Override the chart directory")
    args = parser.parse_args()

    frame = build_frame()
    fit = fit_gap(frame)
    print(f"Sample {fit.gap.index[0]} to {fit.gap.index[-1]} ({len(fit.gap)} quarters)")
    print(f"log real TWI = {fit.alpha:.3f} + {fit.beta:.3f} x log commodity prices (US$)")
    print(f"  beta se {fit.beta_se:.3f} (OLS; not valid for inference on cointegrated levels)")
    print(f"  R2 {fit.r2:.3f}, gap sd {fit.sd:.2f}%, Engle-Granger p {fit.eg_p:.3f}")

    shares = band_shares(fit)
    for band in (NEUTRAL_SD, DARK_SD):
        print(f"  within ±{band:g} sd: {shares[band]:.1%} of quarters")

    z = fit.gap / fit.sd
    print("\n=== Recent gap ===")
    recent = fit.gap.tail(RECENT)
    for period, value, sds in zip(recent.index, recent, z.tail(RECENT), strict=True):
        print(f"  {period}: {value:+.1f}% ({sds:+.2f} sd)")
    dark = z[z.abs() >= DARK_SD]
    print(f"\nBeyond ±{DARK_SD:g} sd: {', '.join(str(p) for p in dark.index)}")

    print("\n=== Robustness: fit by start year ===")
    print(robustness(frame).round(DECIMALS).to_string())

    chart_dir = args.chart_dir or str(CHART_DIR)
    mg.set_chart_dir(chart_dir)
    mg.clear_chart_dir()
    backplane_chart(get_cash_rate_monthly().data, fit, shares)
    print(f"\nCharts written to: {chart_dir}")


if __name__ == "__main__":
    main()
