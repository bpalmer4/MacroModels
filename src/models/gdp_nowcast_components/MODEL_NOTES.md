# GDP Nowcast — Components (Expenditure Identity)

A **T-0 nowcast**: run the day before the Quarterly National Accounts (5206.0)
are published, it reconstructs quarter-on-quarter GDP growth as the sum of
expenditure components' **contributions to growth**, each read from its own
source release that lands a day (or weeks) ahead of GDP. T-0 here matches the
sibling nowcast models' month-indexed publication cycle (T-3m … T-0): it is the
run timing, the complete information set the day before GDP. It is distinct from
the quarter index `T` used in the contribution formula below, where `T` is the
*target quarter* and `T-1` its predecessor.

```
GDP growth (ppt) =  Household consumption  +  Government consumption
                 +  Private investment  +  Public investment
                 +  Inventories  +  Net exports   [ + statistical discrepancy ]
```

Private and public GFCF are kept as **separate** stack segments (not merged into
the ABS-style single GFCF bar) because their T-0 reliability is opposite: public
investment is accounting-exact from GFS (MAE 0.010) while private investment is
the bridged, AI-capex-driven, import-offset-prone piece (MAE 0.200). Splitting
keeps that difference legible.

This is a structural accounting build-up — complementary to, not a competitor of,
the statistical nowcasts (`gdp_nowcast_bridge`, `_dfm`, `_bvar`), which regress
GDP growth on indicator panels. Its edge: at T-0 net exports and government are
**measured, not forecast**, and inventories — the other volatile contributor — is
anchored on a measured prior-quarter flow. Its output is interpretable — a
stacked-ppt decomposition telling you
*where* growth comes from, not just a headline number.

## Contribution formula

Each component's contribution to q/q growth, in chain volume measures (CVM):

```
contribution_T (ppt) = Δ(component_T) / GDP_{T-1} × 100
```

`GDP_{T-1}` is the last *published* real GDP (the denominator is lagged by index,
so the T-quarter source — out a day early — divides by the last published GDP).
Inventories enter GDP as a flow, so their contribution is a difference *of
flows*: `(flow_T − flow_{T-1}) / GDP_{T-1} × 100`. Only `flow_T` needs a proxy.
`flow_{T-1}` is the NA changes-in-inventories published with last quarter's
accounts, so it is read from 5206 rather than differenced out of the 5676 stock:
proxying it too injected the 5676 coverage gap (farm and public excluded) a
second time with the opposite sign, so one bad stock reading cost two quarters.
`flow_T` is the 5676 stock change put on the NA basis by an OLS fitted strictly
before `T` (the two series measure different aggregates, so the raw stock change
is not commensurate with `flow_{T-1}`).

## Components and sources (at T-0)

| Component | Source @ T-0 | Method | Backtest MAE (ex-COVID) |
|---|---|---|---|
| Household consumption | 5682.0 t.5682015 CVM index (~5 wk) | **growth bridge → level** | 0.220 |
| Government consumption | GFS Table 15 CVM $m (~1 day) | accounting-exact | 0.015 |
| Private GFCF | 5625.0 capex + 8755.0 construction (~1–3 wk) | **contribution bridge** | 0.217 |
| Public GFCF | GFS Table 15 CVM $m (~1 day) | accounting-exact | 0.018 |
| Inventories | 5676.0 t.5676001 CVM $m level (~1 day) + 5206 NA flow @ T-1 | **flow bridge, NA-anchored** | 0.223 |
| Net exports | 5302.0 t.530205 CVM $m (~1 day) | accounting-exact | 0.070 |

- **Accounting-exact**: real $m CVM levels that map straight onto the GDP
  identity. Government consumption tracks the published NA contribution almost
  perfectly (MAE 0.015) — confirming GFS Table 15 is CVM and needs no deflator.
- **Inventories (flow bridge, NA-anchored)**: the 5676 private non-farm stock is
  the only T-0 source, but it is a partial-coverage proxy (farm and public
  excluded), and the contribution differences two flows. Anchoring the T-1 flow
  on the published NA number is what does the work: on a like-for-like vintage it
  cut the component's RMSE from 0.357 to 0.260 over 2023+ and 0.264 to 0.178 over
  2024+. Rescaling the T flow onto the NA basis adds a little more (to 0.249 and
  0.170). Bridging *without* anchoring achieves nothing (0.351 / 0.265), which is
  what identifies the T-1 term rather than the proxy's scale as the fault.
- **Household consumption (growth bridge → level)**: the HSI is a CVM *index*
  covering only the volatile, transaction-based ~⅔ of consumption (it maps to
  HFCE at slope ~0.59, not 1 — see `diagnostics.plot_source_vs_na`). It is
  handled the same way as the exact components: fit HFCE *growth* on HSI growth
  over the **ex-COVID** history before the target quarter (COVID broke the
  relationship), predict
  the target-quarter HFCE growth, roll the last HFCE level forward by it, and take
  `ΔHFCE / GDP_{t-1} × 100`. This uses the *actual* current consumption share via
  real levels — no embedded average share, no rounding — so consumption is no
  longer a special case. It remains an inference with moderate, irreducible error
  (~±0.29 ppt 1σ): no T-0 source *is* household consumption.
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

   **The diagnostics needed a fallback that the model itself does not.** Gotcha 2 means the
   loader returns an *empty* series for a quarter whose snapshot has not shipped, and
   `model.run_nowcast` degrades correctly: every component prints "pending release" and the
   nowcast is `nan`, which is right for T-0 early in a quarter. `diagnostics._hh_target_month`
   did not, and fed the empty frame straight into `np.polyfit`, which raised
   `TypeError: expected non-empty vector for x` and took the whole run down at exit code 1. It
   now steps back to the most recent quarter whose snapshot exists, over at most
   `_HH_SNAPSHOT_LOOKBACK` = 4 quarters, and caches the resolved tag because the loader is not
   itself cached and each call is a snapshot fetch.

   The distinction is worth keeping in mind when adding checks here: these charts are historical
   fits and never needed the target quarter at all, whereas the model does. On 2026-09-08 the
   crash reproduced with the target at 2026Q3 and the fallback resolving to `jun-2026`, restoring
   n=41 on the household fit from n=0.

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
./run-gdp-nowcast-components.sh                                    # live (T-0)
uv run python -m src.models.gdp_nowcast_components.backtest        # backtest
```

Output (`charts/GDP-Nowcast-Components/`): a per-component text table reporting
both the Q/Q nowcast and its annual (TTY) re-expression, plus the
component-specific **Contributions to Quarterly GDP Growth** stacked chart
(matching the ABS 5206 chart with the unpublished quarter appended as the final
bar) and the per-component history charts. It also emits the three **standard
nowcast charts** shared with the sibling models (`Q/Q fan`, `TTY fan`, and the
combined annual-line + quarterly-bars chart) via
`src/models/common/nowcast_charts.py` (`plot_nowcast_charts`) and
`src/models/common/nowcast_core.py` (`compute_tty`) — the same shared helpers the
bridge/DFM/BVAR models use, with a goldenrod CI fan.
The annual band is the symmetric Q/Q discrepancy band rolled through the same
Q/Q→annual conversion. Backtest artefacts land in
`model_outputs/gdp_nowcast_components/`.

## Backtest results (2015Q1–latest, pseudo real-time)

```
Headline (summed nowcast vs published GDP growth, ppt):
  ex-COVID      n= 32  MAE=0.477  RMSE=0.672  bias=+0.044
```

Read alongside the per-component table above:

- The **accounting-exact** pieces are tight (government 0.015, net exports 0.070).
- **Inventories** (0.223) is no longer the drag it was. The NA-anchored flow
  bridge (above) is what fixed it; the headline gain is larger than the
  component's own, because the old formula's error was serially correlated by
  construction. On a like-for-like vintage the headline went from RMSE 0.578 to
  0.434 over 2023+, and the ex-COVID headline bias collapsed from +0.191 to
  +0.044 — most of that bias had been the inventories term.
- **Household consumption** (0.220) and **private GFCF** (0.217) are the bridged
  pieces and now carry the most error. The level-path consumption bridge
  (ex-COVID) lowered household MAE from 0.257 and trimmed its over-prediction
  bias; the residual headline bias now sits mostly in private GFCF.
- Read the headline number over the **whole** 2015Q1 window with care: the
  early years fit the private-GFCF bridge on a handful of quarters (it needs only
  `len(names) + 5`), so 2016-2019 headline RMSE is 0.746 against 0.396 over
  2024+. That era reflects an expanding window starting cold, not current
  accuracy. Comparisons against the sibling nowcasts must use a common window
  *and* a common vintage.

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

1. **Inventories** — done: the T-1 flow now comes from the NA all-sector
   changes series and the T flow is bridged onto that basis. What is left is a
   small under-prediction (bias -0.10 over 2023+), which is the farm and public
   coverage gap being carried by the bridge's intercept rather than measured.
   A farm-inventories proxy (ABS crop estimates) is the only obvious lead on it.
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
