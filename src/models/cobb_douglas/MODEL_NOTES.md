# Cobb-Douglas MFP Decomposition

Deterministic growth accounting. GDP growth is split into contributions from capital, hours and a residual, and a potential output path is built by cumulating the trend components. No estimation, no sampling, no parameters fitted to data: everything here is arithmetic on ABS series plus three choices made in advance.

```
Y = A · K^α · L^(1−α)

g_Y      = g_MFP + α·g_K + (1−α)·g_L        the identity, in log growth
g_MFP    = g_Y − α·g_K − (1−α)·g_L          MFP as the Solow residual
g_pot    = α·g_K_trend + (1−α)·g_L_trend + g_MFP_trend
log_pot  = cumulated g_pot, reset to actual GDP at the anchor dates
```

Three choices, none estimated: `alpha` = 0.30, HP `lambda` = 1600, and the four anchor dates.

This is the oldest model in the repo and it predates the conventions the later packages follow. It saves nothing to `model_outputs/`, has no config dataclass, and recomputes from ABS on every run, which takes about 30 seconds.

---

## What it is for

Two things, and it is worth keeping them apart because they are of very different quality.

**Growth accounting.** How much of Australian output growth came from more capital, more hours, and neither. This is the model's real product. It is close to an identity, it needs only `alpha`, and the decade table below is a defensible description of what happened.

**A potential output path.** Built by cumulating the trend contributions. This is weaker and the model says so in its own docstring: *"the output gap from this model is notional only, it is not disciplined by inflation dynamics."* Nothing in this package positions the level of potential output. See "The re-anchoring problem" and "Relationship to `ystar`".

---

## Data

| Series | Source | Role |
|---|---|---|
| GDP, chain volume, SA | `gdp.get_gdp(gdp_type="CVM", seasonal="SA")` | output |
| Net capital stock, quarterly | `get_capital_stock_qrtly()` | K |
| Hours worked, quarterly | `get_hours_worked_qrtly()` | L |
| Trimmed mean annual inflation | `get_trimmed_mean_annual()` | cross-check only, not an input |
| Capital share of income | `get_capital_share()` | chart only, `alpha` does not read it |
| Labour force growth, ULC, hourly COE | `src.data` | comparison charts only |

Sample **1985Q2 to 2026Q2, 165 quarters**, set by where the three core series overlap after `dropna()`.

Growth rates are log differences times 100, which is what makes the contributions additive.

---

## The three choices

### `alpha` = 0.30

The capital share of income. Fixed, and it does not read the capital share series the package loads for charting. That series has a sample mean of **0.303**, so 0.30 is the right central value, but it ranges from **0.21 to 0.41** and its latest reading is **0.336**. A fixed `alpha` is therefore a stand-in for something that moves, and the drift is upward.

`sensitivity_analysis_alpha` measures what that costs. Across `alpha` from 0.20 to 0.40, mean quarterly raw MFP moves from 0.208% to 0.121%, and the post-2015 trend from 0.084% to 0.074%. So the recent MFP story is nearly `alpha`-proof while the historical average is not: doubling `alpha` almost halves measured long-run MFP.

### HP(1600)

Applied three times, to capital growth, hours growth and the Solow residual, to separate trend from cycle. HP brings its own endpoint problem, and here it lands on the most-watched number, since trend MFP at the sample end is what drives the current potential growth estimate.

### The four anchor dates

`["1990Q1", "2000Q1", "2008Q1", "2019Q4"]`, described in the code as business cycle peaks. At each, potential is reset to actual GDP, so **the output gap is zero by construction at those four dates**, and at the start of the sample.

---

## Results (2026Q2 vintage)

Mean annual growth contributions, by decade:

| Decade | Capital | Labour | MFP | Total |
|---|---|---|---|---|
| 1990s | 0.88 | 0.75 | 1.59 | 3.31 |
| 2000s | 1.34 | 1.10 | 0.56 | 3.01 |
| 2010s | 1.11 | 1.08 | 0.44 | 2.56 |
| 2020s | 0.53 | 1.24 | 0.10 | **1.91** |

**The whole of the decline is capital and MFP.** Labour's contribution is higher in the 2020s than in any earlier decade, at 1.24. Capital has halved since the 2000s, from 1.34 to 0.53. MFP has fallen from 1.59 in the 1990s to 0.10, and trend MFP growth at 2026Q2 is **−0.02% p.a.**

Potential growth, year-ended, is still falling:

| | % |
|---|---|
| 2025Q2 | 2.01 |
| 2025Q3 | 1.97 |
| 2025Q4 | 1.93 |
| 2026Q1 | 1.90 |
| 2026Q2 | **1.86** |

Output gap at 2026Q2: **−0.89%**. Read the caveats before using that number.

---

## The re-anchoring problem

Re-anchoring is necessary, not careless, and the reason is the fixed coefficients. With `alpha` constant and the input trends smooth, cumulating potential growth from a single base accumulates level error whenever the underlying speed limit is moving, and the level smears within each block. The resets discard that drift.

The size of the drift is printed on every verbose run, as the gap immediately before each reset:

| anchor | gap discarded |
|---|---|
| 1990Q1 | +0.89% |
| 2000Q1 | −0.78% |
| 2008Q1 | −0.24% |
| 2019Q4 | −0.24% |

Roughly 0.08pp a year of level drift over the earlier blocks, less later.

**What this costs.** The output gap is not a measurement of anything. At 2026Q2 it means "cumulated actual growth less trend growth since 2019Q4", and it rests on output having been exactly at potential in 2019Q4, which is asserted rather than found. On the evidence of the period, Australian GDP growth had already fallen from 3.11% year-ended in 2018Q2 to 1.66% by 2019Q2, so an assumption of zero gap at 2019Q4 is not obviously right.

**What it does not cost.** Re-anchoring never touches `g_potential`, which is computed over the whole sample from HP-filtered inputs (`model.py:280-289`) and only then cumulated (`model.py:311-330`). So the **growth** path is unaffected by the anchor choice, and it is the part of this model's potential output work that can be compared with other estimates.

---

## The Phillips curve cross-check

The model computes, but does not use, the relationship between its own gap and inflation's deviation from 2.5:

| | |
|---|---|
| correlation, gap against inflation deviation | **0.160** |
| slope | 0.249 |

and prints "weak relationship, gap may not capture demand pressure well". This is the model marking its own homework honestly. A gap correlating 0.16 with inflation is not carrying much information about demand pressure.

---

## What it cannot say

1. **Nothing about the level of potential output.** The level is set by the anchor dates and by nothing else. This is the model's own stated position, not an external criticism.
2. **MFP is a residual and absorbs everything.** Measurement error in capital or hours, changes in composition, capacity utilisation and any misspecification of the production function all land in MFP by construction. "MFP fell" and "we mismeasured capital services" are the same observation here.
3. **`alpha` is fixed at a value the data say is moving.** The capital share has ranged 0.21 to 0.41 and is currently 0.336. The sensitivity table bounds the damage for recent MFP but not for the historical decomposition.
4. **HP endpoint bias falls on the most-used number.** Trend MFP at the sample end drives current potential growth, and it is exactly where an HP trend is least reliable.
5. **No uncertainty of any kind is reported.** Everything here is a point estimate. There are no bands because there is no estimation.

---

## Relationship to `ystar`

The two models answer overlapping questions from disjoint information, which makes them a useful pair.

| | `cobb_douglas` | `ystar` |
|---|---|---|
| information | capital, hours, factor shares | GDP, trimmed mean inflation |
| potential growth, 2026Q2 | 1.86 | 2.14 [1.77, 2.51] |
| how the level is set | reset to actual at four anchor dates | defined by inflation at target |
| output gap, 2026Q2 | −0.89 (notional) | +0.51 [0.31, 0.72] |

**Compare the growth paths, not the gaps.** 1.86 against 2.14 is a genuine cross-check between methods sharing no equations and almost no data, and 0.28pp apart is close. The gaps are not comparable, because this model does not claim to locate the level and `ystar` does. See `ystar/MODEL_NOTES.md`, "External comparison".

The `nairu` package also derives its deterministic r* from this model's potential growth, so changes here propagate there.

---

## File structure

```
cobb_douglas/
├── model.py         everything: loading, accounting, potential, charts, CLI
└── MODEL_NOTES.md   this file
```

One module. The pipeline is `load_data` to `calculate_growth_rates` to `calculate_solow_residual` to `extract_mfp_trend` to `calculate_potential_gdp`, wrapped by `run_decomposition`, which returns a `DecompositionResult` holding every series. Sixteen charts are written to `charts/cobb_douglas/` by `plot_all`.

## Commands

```bash
./run-cd.sh              # decomposition + charts
./run-cd.sh -v           # adds the summary tables quoted above
```

Or from Python, when the series rather than the charts are wanted:

```python
from src.models.cobb_douglas.model import run_decomposition
result = run_decomposition()            # start, end, alpha, anchor_points all overridable
result.potential["g_potential"]         # quarterly potential growth, %
result.potential["output_gap"]          # notional, see above
```

Nothing is cached or saved, so every run re-fetches from ABS.

---

## If this is picked up again

1. **Read `alpha` from the capital share series** instead of fixing it at 0.30, or at least document why a constant is preferred. The series is already loaded for charting.
2. **Justify or revisit the anchor dates.** 2019Q4 in particular assumes a zero gap in a quarter when growth had already slowed markedly. Their sensitivity has never been tested, and it drives the entire gap series.
3. **Consider dropping the gap.** The docstring already calls it notional, `ystar` now produces one that is disciplined by inflation, and a notional gap that is charted and printed invites use.
4. **The growth accounting deserves better billing than the gap.** It is the solid part of this package and the decade table is its best output.
