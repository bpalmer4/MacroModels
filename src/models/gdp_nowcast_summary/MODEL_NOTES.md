# GDP Nowcast Summary: Model Notes

**NOT A MODEL.** This runs every GDP nowcast for the next unpublished quarter and draws
their distributions on one chart, for quarterly growth and again for annual growth. Nothing
here is estimated.

```bash
./run-gdp-nowcast-summary.sh       # each model prints its own summary too
./run-gdp-nowcast-summary.sh -q    # just the summary table
```

The nowcasts save nothing, so each run re-runs all of them.

## What each curve is

Every curve is a weighted mixture of normals, f(x) = Σ wᵢ N(x; mᵢ, sᵢ). Each model's own
headline is a median: for the single normals that is also their mean.

| source | components | sd |
|---|---|---|
| Bridge | one | the sigma behind its own bands (RMSE of its pseudo real-time forecast record since 2015) |
| DFM | one | the sigma behind its own bands (Kalman prediction error) |
| BVAR | one | the sigma behind its own bands (conditional Gaussian) |
| Components | one | the statistical-discrepancy sigma behind its own bands |

Each curve is the distribution the model itself reports: its printed bands are quantiles of
the curve, and its headline is the curve's median. The run fails if a charted median is not
the model's own headline.

**Medians and P(<0)** come from the mixture CDF, Σ wᵢ Φ((x − mᵢ)/sᵢ): P(<0) is its value at
zero, the median its root at one half.

## Per capita

Per capita growth subtracts population growth from each nowcast, so every curve shifts left
by the same amount and P(per capita < 0) is each model's CDF at population growth.

Population is total resident population (ERP), the denominator of the ABS's own GDP per
capita. ERP is published about six months late, so the target quarter comes from the
forward proxy in `src/data/population.py`, built from the monthly Labour Force civilian
population 15+. The run fails unless the Labour Force has all three months of the target
quarter.

The proxy is an annual change. Its quarterly rate is the geometric one that compounds to
the annual rate over four quarters, (1 + annual)^(1/4) − 1, not a quarter of it. The rate is
put in logs before subtracting, because the nowcasts' Q/Q growth is a log difference.
Annual per capita is exact: (1 + GDP annual) / (1 + population annual) − 1.

The quarterly rate is the year's average quarter, so any variation in population growth
between quarters within the year is not in it. That avoids the seasonality in quarterly
ERP, which is published unadjusted.

## Components is left off until complete

Its band measures only the statistical discrepancy, so while any component is still
unpublished the curve would show a certainty the model does not have. It is charted once
every component is in, and the run says which are pending until then.

## The annual chart

Each Q/Q value is carried to annual growth with the same `compute_tty` the models use, which
rolls the last published level forward. The map is monotonic, so medians carry across directly
and each density is rescaled by dq/d(annual) = 100 / (100 + annual).

**Earlier quarters are held at their published values.** The annual curves carry no
uncertainty from revisions to those quarters, which arrive with the same release.

As a check, each model's own annual figure must come out of that conversion unchanged. The
run fails if it does not, since that would mean the summary and the model used different GDP
data.
