# GDP Nowcast — Components (Expenditure Identity)

A **T-1 nowcast**: run the day before the Quarterly National Accounts (5206.0),
it reconstructs quarter-on-quarter GDP growth as the sum of expenditure
components' **contributions to growth**, each read from its own source release
that lands a day (or weeks) ahead of GDP.

```
GDP growth (ppt) =  Household consumption  +  Government consumption
                 +  Private investment  +  Public investment
                 +  Inventories  +  Net exports   [ + statistical discrepancy ]
```

Private and public GFCF are kept as **separate** stack segments (not merged into
the ABS-style single GFCF bar) because their T-1 reliability is opposite: public
investment is accounting-exact from GFS (MAE 0.010) while private investment is
the bridged, AI-capex-driven, import-offset-prone piece (MAE 0.200). Splitting
keeps that difference legible.

This is a structural accounting build-up — complementary to, not a competitor of,
the statistical nowcasts (`gdp_nowcast_bridge`, `_dfm`, `_bvar`), which regress
GDP growth on indicator panels. Its edge: at T-1 the two most volatile
contributors (inventories, net exports) and government are **measured, not
forecast**. Its output is interpretable — a stacked-ppt decomposition telling you
*where* growth comes from, not just a headline number.

## Contribution formula

Each component's contribution to q/q growth, in chain volume measures (CVM):

```
contribution_T (ppt) = Δ(component_T) / GDP_{T-1} × 100
```

`GDP_{T-1}` is the last *published* real GDP (the denominator is lagged by index,
so the T-quarter source — out a day early — divides by the last published GDP).
Inventories enter GDP as a flow, so they take a **second** difference of the
level: `(Δlevel_T − Δlevel_{T-1}) / GDP_{T-1} × 100`.

## Components and sources (at T-1)

| Component | Source @ T-1 | Method | Backtest MAE (ex-COVID) |
|---|---|---|---|
| Household consumption | 5682.0 t.5682015 CVM index (~5 wk) | **growth bridge → level** | 0.232 |
| Government consumption | GFS Table 15 CVM $m (~1 day) | accounting-exact | 0.009 |
| Private GFCF | 5625.0 capex + 8755.0 construction (~1–3 wk) | **contribution bridge** | 0.200 |
| Public GFCF | GFS Table 15 CVM $m (~1 day) | accounting-exact | 0.010 |
| Inventories | 5676.0 t.5676001 CVM $m level (~1 day) | accounting-exact | 0.210 |
| Net exports | 5302.0 t.530205 CVM $m (~1 day) | accounting-exact | 0.082 |

- **Accounting-exact**: real $m CVM levels that map straight onto the GDP
  identity. Government consumption tracks the published NA contribution almost
  perfectly (MAE 0.009) — confirming GFS Table 15 is CVM and needs no deflator.
- **Household consumption (growth bridge → level)**: the HSI is a CVM *index*
  covering only the volatile, transaction-based ~⅔ of consumption (it maps to
  HFCE at slope ~0.59, not 1 — see `diagnostics.plot_source_vs_na`). It is
  handled the same way as the exact components: fit HFCE *growth* on HSI growth
  over the **ex-COVID** history before T-1 (COVID broke the relationship), predict
  the target-quarter HFCE growth, roll the last HFCE level forward by it, and take
  `ΔHFCE / GDP_{t-1} × 100`. This uses the *actual* current consumption share via
  real levels — no embedded average share, no rounding — so consumption is no
  longer a special case. It remains an inference with moderate, irreducible error
  (~±0.29 ppt 1σ): no T-1 source *is* household consumption.
- **Private GFCF (contribution bridge)**: capex + construction miss IP products
  and some industries, so the published private-GFCF contribution is regressed on
  their growth (expanding window, no look-ahead). The weakest leg (capex maps to
  GFCF at R² ~0.3).
- **Statistical discrepancy**: the residual that makes the five stacked
  components sum to headline GDP (GDP is the average of the I/E/P measures). It
  is **zero in the central nowcast** and sized into the uncertainty band from the
  recent distribution of the published discrepancy.

## Two hard-won data gotchas (ported from the ~/ABS notebooks)

1. **Re-referencing guard** (`data.reref_factor`, live only). ABS CVM are
   re-referenced annually at the September accounts. The source releases drop a
   day *before* the accounts, so in that straddle the source is on the new
   reference year while the last published GDP is still on the old one. A CVM
   level is a clean scalar under re-referencing, so the source is down-weighted
   onto the GDP vintage's basis using the median ratio of the current series to
   the vintage that was current when GDP last printed (`history=` fetch). Applied
   to inventories and to exports/imports separately (net exports is a small,
   heavily-leveraged balance of two large aggregates). Outside the straddle the
   factor is ~1.0 and the extra fetch is skipped.

2. **Quarterly household spending retrieval** (`data.household_spending_cvm_level`).
   The quarterly CVM table 5682015 only ships with the monthly 5682.0 release
   that lands on a quarter-end month. Because the target quarter is known up
   front, fetch that snapshot directly via `history=<quarter-end month of T>`
   rather than the download-then-check-then-fallback dance — if the quarter isn't
   in it, it isn't there. Anchored to the target quarter, not to `today`.

## Architecture

One as-of-parameterised contribution path (`model._contribute`) is shared by the
live run and the backtest, so a component becomes a number in exactly one place.

- `data.py` — component sources, published contributions, household trick,
  re-referencing guard. Everything is a `Q-DEC` quarterly `pd.Series`.
- `model.py` — `AsOf` information set, `_contribute` (the shared math), the two
  OLS bridges, `NowcastResult`, text summary, the stacked contributions chart,
  and the live CLI (`run_nowcast`).
- `backtest.py` — replays the nowcast, reports headline + per-component error.

```bash
./run-gdp-nowcast-components.sh                                    # live (T-1)
uv run python -m src.models.gdp_nowcast_components.backtest        # backtest
```

Output: a per-component text table + the **Contributions to Quarterly GDP
Growth** chart (`charts/GDP-Nowcast-Components/`), matching the ABS 5206 chart
with the unpublished quarter appended as the final bar. Backtest artefacts land
in `model_outputs/gdp_nowcast_components/`.

## Backtest results (2015Q1–latest, pseudo real-time)

```
Headline (summed nowcast vs published GDP growth, ppt):
  ex-COVID      n= 31  MAE=0.553  RMSE=0.766  bias=+0.191
```

Read alongside the per-component table above:

- The **accounting-exact** pieces are tight (government 0.009, net exports 0.082).
- **Inventories** (0.210) is the noisiest exact component — the 5676 private
  non-farm Δlevel is a partial-coverage proxy (farm + public excluded), and
  second-differencing a CVM level amplifies noise.
- **Household consumption** (0.232) and **private GFCF** (0.200) are the bridged
  pieces and carry the most error. The level-path consumption bridge (ex-COVID)
  lowered household MAE from 0.257 and trimmed its over-prediction bias
  (0.129 → 0.111); the residual headline bias (+0.19) now sits mostly in private
  GFCF and inventories.

### Caveats

- **Pseudo real-time**: inputs and evaluation use latest-vintage data truncated
  to the as-of set, so the September re-referencing straddle (a live-only effect)
  is not exercised and live Sep-quarter uncertainty is slightly understated.
- **GFS history**: the GFS workbook's Table 15 only spans ~2022Q4 onward, which
  would otherwise cap the full-identity backtest at ~12 quarters. Government is
  accounting-exact, so the backtest substitutes the published NA government
  contribution before GFS reaches (`gov_fallback=True`, flagged per quarter),
  unlocking the long household / GFCF / inventories window. Live runs always use
  GFS.

## Open improvement avenues

1. **Inventories** — try the all-sector NA-matched changes series, or smooth the
   second difference; it's the weakest exact component.
2. **Household consumption** — now a growth-bridge→level path (ex-COVID), which is
   about as far as the HSI can be pushed: it sees only ~⅔ of consumption, so the
   leg is an irreducible moderate-error inference (~±0.29 ppt 1σ). Per-category
   decomposition was considered and rejected — it can't reach the ~⅓ the HSI never
   covers (rent, electricity, comms, education, financial), which the aggregate
   intercept already approximates.
3. **Private GFCF** — capex + construction under-cover (IP products); add an
   IP-products trend or a fitted coverage scale-up.
4. **Uncertainty band** — replace the discrepancy-only band with the empirical
   backtest RMSE, which captures bridge error too.
5. **Headline debias** — a small intercept correction would remove the +0.21 ppt
   bias, at the usual cost to turning-point tracking (see the DFM notes).
