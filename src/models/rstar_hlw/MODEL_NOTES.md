# HLW Bayesian r-star Model — Model Notes

A Bayesian (PyMC + NumPyro NUTS) implementation of the Holston-Laubach-Williams 2017 model, applied to Australian quarterly data. Estimates the natural rate of interest r* jointly with potential output, trend growth, and the output gap.

## The story

The original goal: estimate r* for Australia with proper Bayesian uncertainty bands and feed the posterior into the NAIRU model's IS curve, replacing its deterministic Cobb-Douglas r*.

The eight specifications (Resolutions A–H below) were not designed upfront as a sweep. They were built one at a time, each as the previous one's failure clarified what the problem actually was:

- **A** (canonical HLW) came back with a dead z-state and r* ≈ g. That raised three competing diagnoses: missing market information, a mis-specified IS curve, or a latent that cannot be identified at all.
- **B** tested the first — add the indexed-bond observation. The model simply relabelled the bond yield as r*; the HLW machinery became decorative.
- **C** accepted the diagnosis B implied and replaced the unidentifiable latent with a deterministic blend of the two observable anchors, weighted by a single scalar α. Clean — but the answer turned out to be whatever the α-prior said.
- **D** tested the second diagnosis — maybe A failed because the closed-economy IS curve was missing the SOE channels. It wasn't: same dead z with the open-economy block wired in.
- **E** and **F** checked whether C's deterministic identity was over-constraining the answer, by giving the IS curve room (an AR(1) z) to disagree with the blend — alone (E) and with the SOE block (F). It declined both times.
- **G** and **H** were the last degrees of freedom: hand the α-prior's shape to the data (G), then let α drift over time (H). The data handed the shape straight back, and pulled α_t flat.

What began as "estimate r* for Australia" became "understand why r* cannot be estimated from Australian data" — and the resolutions are the record of that shift, not a menu of alternatives.

Across the eight, the same pattern returned. **Each spec gave back the structural assumption it imposed**, plus a Monte Carlo whisker. The IS curve does not pin r* in Australian data. The rate channel is too weak: a_r ≈ −0.04 against σ_IS ≈ 0.70 and an in-sample r-gap sd of ~2pp — a signal-to-noise ratio of about 0.11. The signal sits more than an order of magnitude below the noise floor. The IS curve cannot resolve r* out of the data independently of whatever structural assumption is imposed on it.

This is **Buncic-Pagan-Robinson 2023** ("On Constructing a Country-Specific Time Series for the Natural Rate of Interest") made concrete on Australian data: when latent shocks meet or exceed identifying observables, the latent is not point-identified — the posterior is essentially the prior projected through the structural model. We confirm it three independent ways inside the project:

- **σ_z prior sweep on E**: the posterior CI on r* scales linearly with the prior σ_z (1.27pp → 5.35pp as σ_z runs 0.05 → 0.50). The data has nothing to say about how uncertain r* should be.
- **α-prior shape sweep on C**: the posterior 90% range tracks the prior 90% range almost 1:1, regardless of which Beta we choose.
- **Time-varying α_t (H)**: when α is allowed to drift period-by-period under a permissive logit-RW prior (~1pp/quarter latitude at α=0.5), it lands flat — sd 0.002 across 158 quarters, total drift 0.01pp end-to-end. The data does not even want time-variation.

The project did not produce *the* r* estimate for Australia. That estimate cannot be produced from this data. **Nonetheless the exercise was useful** — three things came out of it that are worth more than a single point estimate would have been:

1. **A diagnostic framework.** Every credible Australian r* estimate — the model's variants, Bullock's stated working view, McCririck-Rees, Ellis 2022, IMF Article IV — sits on a single bond-vs-growth axis that the α-blend formalises. α near 0 weights the bond anchor (yields-implied real rate, less a constant term premium); α near 1 weights the structural anchor (trend growth); α ≈ 0.5 splits the difference. Disagreements about r* are disagreements about α. The model gives a clean way to read them as such.
2. **An honest uncertainty range.** The cross-resolution spread of r* latest is ~1.0pp (1.48% B → 2.43% A). That spread is the honest band. Anything tighter is the prior projected through a likelihood that has no power to challenge it.
3. **A forward-testable framework.** The cross-resolution disagreement will resolve as future inflation outcomes arrive (see "What will resolve the disagreement"). The model is a snapshot, not a verdict.

The rest of these notes work through the eight resolutions individually, the cross-resolution evidence, the diagnostic framing, and an iteration log.

## Sample, data, and the indexed-yield fill

**Sample**: 1986Q3 → 2026Q1 (159 contiguous quarters as at the June 2026 run; the end advances with each quarterly re-run). The start is pinned by the indexed bond yield series — `observations.py` joins all series and drops incomplete quarters, so the 1980Q1 CLI default start is a no-op and the effective floor is 1986Q3. Pre-1993 Australia had no inflation target and a different monetary regime — including 1983Q1–1992Q4 worsens identification. The HLW-NAIRU integration plan is to use HLW r* from 1993Q1 onward and fall back to the Cobb-Douglas r* before that.

**Data inputs**

| Series | Source | Notes |
|--------|--------|-------|
| log GDP × 100 | `gdp.get_log_gdp()` | ABS 5206.0 chain volume, SA |
| Cash rate | `cash_rate.get_cash_rate_qrtly()` | RBA OCR + historical interbank, end of quarter |
| π_exp (annualised %) | `expectations_model.get_model_expectations_unanchored()` | Project's own signal-extraction model, unanchored variant |
| π_4 (annual trimmed mean) | `inflation.get_trimmed_mean_annual()` | ABS 6401.0 |
| Fiscal impulse (lag 1) | `gov_spending.get_fiscal_impulse_lagged_qrtly()` | Same series the NAIRU model uses |
| Indexed 10y bond yield | `bonds.get_indexed_yield_filled()` | RBA F2 (1986+); see fill below |
| Linear trend (g anchor) | derived in `observations.py` | Linear regression of YoY GDP growth over the sample |

**Indexed yield gap fill**. RBA's F2 series for the 10y indexed bond yield has a 5-quarter gap (2013Q3–2014Q3) when Treasury was transitioning between the maturing 2020 indexed benchmark and the new 2025 benchmark. `get_indexed_yield_filled()` patches it with `nominal_10y − interpolated breakeven`. Breakeven inflation is anchored and moves slowly; the nominal 10y is observed throughout the gap and contributes the actual real-rate dynamics — including the 2013 taper-tantrum spike that a naïve interpolation of indexed_10y would miss. Filling the gap reduced divergences across most resolutions (A: 4,050→111; B: 11,458→3,349; C: 428→193; E: 819→173).

## Model structure (shared across resolutions)

All resolutions share the same potential output, Phillips curve, and (where wired in) soft anchor on g.

**Latent states**

| State | Symbol | Equation | Units |
|-------|--------|----------|-------|
| Trend growth | g_t | g_t = g_{t-1} + ε_g | annualised % |
| Potential output | y*_t | y*_t = y*_{t-1} + g_{t-1}/4 + ε_{y*} | log × 100 |

**Observation equations**

1. **IS curve** (HLW 2017 form, with fiscal impulse; D and F additionally include open-economy regressors — ToT growth, real TWI change, RBA ICP A$ growth):
   `log_gdp_t = y*_t + a_y1·y_gap_{t-1} + a_y2·y_gap_{t-2} + (a_r/2)·(r_gap_{t-1} + r_gap_{t-2}) + γ_fi·fiscal_{t-1} + ε_IS`
2. **Phillips curve** (annual trimmed mean, anchor-augmented):
   `π_4_t = π_exp_t + b_y · y_gap_{t-1} + ε_π`
3. **Soft anchor on g** (C, D, E, F, G, H only):
   `linear_trend_t = g_t + ε_trend`,  `ε_trend ~ N(0, 2.0)` (σ fixed; a free σ collapses to ~0.02 and turns the soft anchor into a hard constraint).

The differences between resolutions are entirely in the **r\* identity** and in whether the open-economy IS-curve regressors are wired in.

**Reparameterisation choices that matter:** trend_growth is centred (non-centring breaks the doubly-cumulated y* equation — see iteration 4); σ_g uses a tight HalfNormal(0.04) — loosening it without an external anchor on g blows up divergences; r_innovation is non-centred (this fixes the σ_r funnel and was the move that took iteration 12's divergence count from 908 to 156).

## The eight resolutions

All resolutions run on the gap-filled 158-quarter sample (1986Q3–2025Q4) with NumPyro NUTS, 5 chains × (3,500 tune + 10,000 draws), seed=42.

### Resolution A — canonical HLW (closed economy)

`r*_t = g_t + z_t` with z an AR(1)-reparameterised non-centred random walk. IS curve closed-economy. No anchor on g.

- z posterior collapses; r* tracks g almost exactly; r* span 0.42pp over the sample.
- a_r ≈ −0.033, σ_IS 0.68; r* range [2.41, 2.83]%, latest 2.43%.
- 111 divergences; r_star R-hat 1.020, ESS 243 — the worst sampling of the well-behaved resolutions; the poor sampling is itself diagnostic of identification failure.

**What we learnt**: Buncic's finding for Euro Area / UK / Canada also holds for Australia. λ_z is essentially zero in canonical HLW; the z latent is dead; r* ≈ g. McCririck-Rees (RBA 2017) and Ellis (RBA 2022) report similar identification failure. A is the textbook canonical-HLW-on-an-SOE baseline.

### Resolution B — canonical + indexed-bond observation

Same `r*_t = g_t + z_t` structure plus an observation equation `indexed_10y_t = r*_t + tp + ε_tp` with constant tp.

- z absorbs essentially all r* dynamics; g stays ~2.7–3.1%; r* tracks indexed_10y minus a constant term premium of ~0.93pp.
- r* span 6.51pp; range [−1.64, 4.87]%, trough −1.64% (2020Q4), latest 1.48%.
- 3,349 divergences (worst of any resolution); r_star R-hat 1.010, ESS 366.

**What we learnt**: with a strong external identifier, the data does not "estimate" r* — it relabels the bond yield as r*. The negative trough is a direct consequence of the constant-tp assumption pushing post-GFC term-premium compression into r* itself. The HLW machinery becomes decorative. **However**, B's level (1.48%) is the closest of any resolution to Bullock's stated working view. And B's mechanism — r* moves with the bond yield — is the implicit framework most central banks use.

### Resolution C — deterministic blend

`r*_t = α·g_t + (1 − α)·(indexed_10y_t − k) + ε_t`, α ~ Beta(1, 1) Uniform, k ~ TruncatedNormal(0.5, 0.5, lower=0), ε_t ~ N(0, σ_r) i.i.d. The unidentified third latent state (z) is replaced by a deterministic convex combination of two observable-driven anchors plus a small i.i.d. noise term.

- α posterior median 0.56, 90% HDI [0.07, 0.96] — the HDI tracks the Uniform prior's range almost 1:1.
- k ≈ 0.62, σ_r ≈ 0.20, σ_g ≈ 0.043, a_r ≈ −0.035, σ_IS ≈ 0.68.
- r* range [0.82, 4.13]%, trough 0.82% (2020Q4), latest 2.19%.
- 193 divergences; r_star R-hat 1.000, ESS 6,133.

**What we learnt**: by collapsing the unidentified third latent state to a single scalar (α), C produces a clean, in-range r* path — but the level is mostly an artefact of the α prior. The Uniform-prior posterior shows a weak data tilt toward higher α (density rising ~1.75× from α=0 to α=1), but the tilt is not strong enough to overrule a tighter prior. C's r* is what the analyst's α-prior implies, dressed in Bayesian language.

### Resolution D — canonical r* + open-economy IS curve

`r*_t = g_t + z_t` (same identity as A) but with the IS curve augmented by SOE regressors: ToT growth, real TWI change, RBA ICP (A$) growth — all lagged. Soft anchor on g wired in.

- σ_z piles at the lower bound; z still dead. SOE coefficients all small (γ_tot ~0.01 piled at 0; γ_twi ~−0.01; γ_icp ~0.02).
- a_r ≈ −0.032, σ_IS ≈ 0.668 — within MC error of A and C.
- r* range [1.77, 2.68]%, span 0.91pp, latest 1.77%.
- 85 divergences; r_star R-hat 1.010, ESS 548.

**What we learnt**: the hypothesis was that A's failure might be mis-specification — closed-economy IS curve missing SOE channels. D refutes that directly. Adding the three principal SOE channels does not change the answer: σ_IS unchanged, a_r unchanged, z still dead. D upgrades the canonical-HLW-fails-for-AU finding from "weak rate channel in a closed-economy model" to "weak rate channel even with the SOE block".

### Resolution E — blend + AR(1) z (soft-anchor variant of C)

`r*_t = α·g_t + (1 − α)·(indexed_10y_t − k) + z_t` with z AR(1), σ_z fixed at 0.15, ρ_z ~ TN(0.95, 0.03). Generalises C (σ_z = 0 → C) and A (α = 1 → A).

- z posterior median range ~[−0.08, +0.04] over 40 years (span ~0.12pp); z mean absolute 0.041pp; z latest ≈ 0.
- α 0.57 [0.08, 0.96] — within MC error of C; ρ_z ≈ 0.95 (≈ prior).
- r* range [0.83, 4.09], trough 0.83% (2020Q4), latest 2.20% — virtually identical to C.
- 173 divergences; r_star R-hat 1.000, ESS 12,420.

**What we learnt**: softening C's deterministic identity to a soft anchor lets the IS curve disagree with the blend if it wants to. It doesn't. The IS curve has no information to put into z above the blend. C's deterministic identity is not over-constraining the answer; the blend is what r* is.

### Resolution F — E's r* + open-economy IS curve

E combined with D's SOE-block IS curve. Tests whether softening the anchor (giving z room) plus richer IS-curve regressors finds something neither alone can.

- z mean abs 0.051pp (vs E's 0.041); same dead patterns on γ_tot/γ_twi/γ_icp as D.
- α 0.53 [0.06, 0.95]; r* range [0.63, 4.13], latest 2.15%.
- 249 divergences; r_star R-hat 1.000, ESS 16,565 (cleanest of any resolution).

**What we learnt**: closes the empirical loop. Across A–F, σ_IS sits at 0.70 ± 0.02 and a_r at −0.04 ± 0.01. The IS curve's rate channel is a property of Australian data; no choice of r* identification or IS-curve regressor specification makes it strong.

### Resolution G — blend with hierarchical Beta(a, b) on α

Same r* identity as C but with a hierarchical prior on α: a, b ~ Uniform(0.25, 2); α ~ Beta(a, b). Lets the data choose the Beta shape.

- a_hyper 1.18 [0.35, 1.92]; b_hyper 1.09 [0.33, 1.91]; α 0.58 [0.03, 0.99].
- α posterior is right-tilted with mild endpoint pile-up (density ~1.7 at α=0, ~1.0 in middle, ~3.0 at α=1). Both hyperparameters land slightly above 1 — under a flat hyperprior, the data mildly prefers bell-shaped Betas over U-shapes.
- r* range [0.89, 4.11], trough 0.89% (2020Q4), latest 2.20% — within MC error of C.
- 181 divergences; r_star R-hat 1.000, ESS 6,169.

**What we learnt**: even when the prior shape is handed over to the data, the data hands it back with effectively no information added — both hyperparameters straddle 1 with very wide HDIs. An earlier version of G used `HalfNormal(1)` as the hyperprior on (a, b); HalfNormal(1) puts ~58% of its mass below 1, mildly favouring U-shaped Betas, and produced a strongly bimodal α posterior (densities ~4.4 / 0.6 / 8.4). Switching to the neutral `Uniform(0.25, 2)` flattened the bimodality. **Endpoint stacking only appears when the (hyper)prior allows Beta shape parameters below 1**, putting the Beta in U-shape territory — it's a constraint structure × diffuse-signal interaction, not a data preference for extreme α.

G is the implementation default in `estimate.py` — not because it is "the answer", but because among the blend variants it makes the analyst's α choice as data-driven as possible. The median r* path is essentially identical to C.

G is the end point of the sequence, not the winner of a horse race: by the time it was built, A–F had established that the data would not adjudicate, so the remaining honest move was to make the one analyst-chosen quantity (α) as data-driven as the framework allows — and document that the data declines even that.

#### The G story — the honest answer

What is Australia's r*, the real neutral interest rate? Resolution G gives the honest answer: the data cannot tell you.

r* in G is a blend of two anchors — trend growth (structural) and the real bond yield (market) — with the weight α left for the data to pin down. It doesn't pin it down. The α posterior is nearly flat from 0 to 1, so the model splits into two stories: draws either track trend growth or the bond yield, with little in between. The bimodal draw-cloud chart shows the split and the current level of each mode.

The blended median is the average of those two stories, not a value the model actually settles on — almost no single posterior draw sits at it. Quoting it as "the estimate" overstates what the data identifies.

Why so weak? The IS curve's rate channel sits an order of magnitude below the quarterly noise (a_r ≈ −0.04 against σ_IS ≈ 0.70 — see "The story" above), so the model returns the prior with a thin layer of likelihood: the Buncic-Pagan-Robinson non-identification result, on Australian data.

So which anchor? The data won't say, but two outside pieces of evidence lean to the bond/market reading: the post-GFC decade (a low yield-anchored r* fits the low-inflation outcome better than a high growth-anchored one), and RBA commentary describing the policy stance as having eased materially with no change in the cash rate (see the Bullock cross-validation section below) — only a yield-anchored r* can move that fast; trend growth can't.

On the bond-market reading, neutral nominal is the mode-conditional r* plus inflation expectations, and the policy stance should be judged against that — this is why the international comparison chart plots the bond-market mode rather than the blended median. But the real finding isn't a number. It's that anyone quoting a single confident r* for Australia is showing you their prior, not the data.

A corollary: the market-anchored r* moves. When it rises, the gap between the cash rate and neutral narrows without the central bank doing anything — a stance that looks mildly restrictive can drift toward neutral within quarters. Of course, what goes up can also come down.

### Resolution H — blend with time-varying α_t (logit-RW)

Same r* identity as C but with α_t a time-varying latent on the logit scale: logit_α_0 ~ N(0, 2); logit_α_t = logit_α_{t−1} + σ_a · ε_α_t with σ_a fixed at 0.05 (which permits ~1pp/quarter drift at α=0.5); α_t = sigmoid(logit_α_t).

- α_t std deviation across 158 quarters: **0.002**. End-to-end drift: **0.01pp** (0.583 in 1989 → 0.593 in 2023). Effectively a horizontal line at ~0.59.
- r* range [0.92, 4.11], trough 0.92% (2020Q4), latest 2.20%.
- 206 divergences; r_star R-hat 1.000, ESS 6,246; alpha_rstar ESS 88,394.

**What we learnt**: the chain *actively pulls α_t toward a constant* despite ample room to drift. The IS curve has no era-specific signal about which anchor matters more in different periods. Bullock's "r* has shifted upward" framing — which would imply α_t drifting toward 0 in recent years — is not supported by the data inside this model. H closes one final degree of freedom: even time-variation in α can't be coaxed out of the Australian data.

## Cross-resolution summary

| | A | B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|---|---|
| Divergences | 111 | 3,349 | 193 | 85 | 173 | 249 | 181 | 206 |
| r* span (median) | 0.42 | 6.51 | 3.31 | 0.91 | 3.26 | 3.50 | 3.22 | 3.19 |
| r* trough | 2.41% | −1.64% | 0.82% | 1.77% | 0.83% | 0.63% | 0.89% | 0.92% |
| r* latest (2025Q4) | 2.43% | 1.48% | 2.19% | 1.77% | 2.20% | 2.15% | 2.20% | 2.20% |
| z status | dead | wild | n/a | dead | dead | dead | n/a | n/a |
| `r_star` R-hat (ESS) | 1.020 (243) | 1.010 (366) | 1.000 (6,133) | 1.010 (548) | 1.000 (12,420) | 1.000 (16,565) | 1.000 (6,169) | 1.000 (6,246) |
| `a_r` median | −0.033 | −0.031 | −0.035 | −0.032 | −0.035 | −0.036 | −0.034 | −0.034 |
| `σ_IS` median | 0.68 | 0.68 | 0.68 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| `α` posterior | n/a | n/a | 0.56 [0.07, 0.96] | n/a | 0.57 [0.08, 0.96] | 0.53 [0.06, 0.95] | 0.58 [0.03, 0.99] | α_t flat at ~0.59 |

The picture: **σ_IS and a_r are flat across all eight specs**. r* tracks whichever observable the structural identity admits — g (A, D), bond yield (B), or the blend (C, E, F, G, H). The IS curve does not adjudicate.

## The diagnostic value

Every credible Australian r* estimate sits on a single bond-vs-growth axis:

- α near 0 (B / Bullock / market-leaning frameworks): bond yield carries the structural signal; r* latest ~1.5%.
- α ≈ 0.5 (C, E, F, G, H): equal-weight blend; r* latest ~2.2%.
- α near 1 (A / D / canonical HLW for SOEs): trend growth carries the structural signal; r* latest 1.8–2.4%.

The data inside this 158-quarter sample does not distinguish between these positions. The choice of α is a *prior* commitment.

### Bullock cross-validation (May 2026)

Governor Bullock characterised cash 4.35% as "a bit restrictive, but less restrictive than 16 months ago, due to shifts in r*". This is informative on both *level* and *dynamics*:

- **Level**: real cash ≈ 4.35 − 2.7 = 1.65%. "A bit restrictive" implies r* ≈ 1.0–1.4%.
- **Dynamics**: r* has *risen* by ~0.3–0.5pp over 16 months at unchanged 4.35% nominal cash.

| Resolution | r* latest | Implied r-gap | Bullock-consistent? |
|---|---|---|---|
| A | 2.43% | −0.78pp | accommodative — incompatible |
| **B** | **1.48%** | **+0.17pp** | mildly restrictive — **closest** |
| C | 2.19% | −0.54pp | accommodative — incompatible |
| D | 1.77% | −0.12pp | essentially neutral |
| E/F/G/H | ~2.20% | ~−0.55pp | accommodative — incompatible |

The bond yield (indexed_10y) rose from 2.21% (2025Q1) to 2.40% (2025Q4) — a 0.2pp move. B (full transmission) gives ~0.5pp r* rise → consistent with Bullock. C/E/F/G (partial transmission at α~0.5) give ~0.2pp rise → partial consistency. A/D (zero transmission) → incompatible.

**The deflationary caveat.** Bullock's view is a model preference, not external evidence. The RBA's r* methodology — like most central banks' — almost certainly weights the bond market heavily; that's standard practice. B's level being close to Bullock's reflects two methods reading the same signal, not independent confirmation. The published Australian estimates we cited (McCririck-Rees, Ellis 2022, IMF Article IV) almost all lean on bond-market information in some way.

Bullock's view sits at α ≈ 0 on the same axis everything else does. It is one defensible position. It is not the truth. These are judgements, not estimates.

### What will resolve the disagreement

Forward inflation outcomes. Bullock's view is testable.

| Inflation outcome over next 12–24 months | Implies | Vindicates |
|---|---|---|
| Falls to target within ~12 months | real cash 1.65% was above r* → r* near 1.0–1.4% | B and Bullock |
| Stays elevated 18+ months despite 4.35% | real cash 1.65% was not above r* → r* > 1.4% | C / G / H (~2.20%) |
| Falls only after further hikes | r* 1.5–2.0% — middle of the range | D, or somewhere mid-spectrum |

Recommended forward-test cadence: re-estimate at Q4 2026 (~6 months out, first read), Q2 2027 (~12 months out, meaningful test horizon), Q4 2027 / 2028 (~18 months out, definitive read absent a major exogenous shock). As real disinflation outcomes arrive, the joint posterior over α / σ_z / hyperparameters will shift in ways the current sample's likelihood cannot anticipate.

## Implications for downstream use

1. **A single number with credible bands overstates precision.** The within-resolution CI is mostly the prior projected through the model. The honest band is the cross-resolution spread, ~1.0pp.

2. **For NAIRU integration**, the median r* series from a blend resolution (C, G, or H — all near-identical at the median) is a defensible input, but it is one possible r* path, not a separately-identified structural quantity. The downstream NAIRU model should be aware that this r* is mostly an externally-imposed input chosen by the analyst's α weighting.

3. **The leverage is in richer external anchors, not in richer IS-curve identification.** The IS curve's weakness is a property of the data. No respec inside the HLW framework changes that. See "What was *not* tried".

Treat the model as a framework for thinking about r*, not a measurement of it.

## NAIRU integration

The original motivation. Once a single canonical run is settled, the path forward is sequential coupling:

1. Run the rstar pipeline at 1993Q1 start.
2. Save the posterior median r* series to disk.
3. In `src/models/nairu/observations.py`, expose an option to load that series into `obs["det_r_star"]` instead of the Cobb-Douglas r* derived from `compute_r_star()`.
4. Pre-1993 r* values can be backward-extrapolated (hold the 1993Q1 estimate, or a simple trend) or fall back to Cobb-Douglas.

Fan integration (propagating r* uncertainty through to NAIRU) would require either a joint model or a Monte Carlo loop over r* samples. Not implemented.

## Switching resolutions

```bash
./run-rstar-hlw.sh -v                          # default G
./run-rstar-hlw.sh -v --resolution C
./run-rstar-hlw.sh -v --resolution A           # canonical HLW, closed economy
./run-rstar-hlw.sh -v --resolution B           # canonical + indexed-bond observation
./run-rstar-hlw.sh -v --resolution D           # canonical + open-economy IS curve
./run-rstar-hlw.sh -v --resolution E           # blend + AR(1) z
./run-rstar-hlw.sh -v --resolution F           # E + open-economy IS curve
./run-rstar-hlw.sh -v --resolution H           # blend with time-varying alpha_t
```

## Australia vs G3 r* comparison chart

`analyse.py` always emits an overlay of the model's median r* against the NY Fed Holston-Laubach-Williams r* estimates for US, Euro Area, and Canada (`src/data/world_rstar.py`). The three foreign series are plotted individually — no trade-weighted aggregate. The chart is purely descriptive — none of the foreign r* series are observations in the model. `analyse.py` calls `get_world_rstar(force_download=True)` so each run pulls the latest published file.

## Iteration log

Sampler diagnostics across the major model variants. All runs use NumPyro NUTS, 5 chains × (3,500 tune + 10,000 draws), `target_accept = 0.90` throughout (one experimental run at 0.97 was reverted — see "What was tried that didn't work").

| Iteration | Spec | Sample | Divergences | Substantive answer |
|---|---|---|---|---|
| 1 | Canonical (z RW, no fiscal, no bond) | 1983Q1 | 2 | r* ≈ g, span 0.4pp (dead z) |
| 2 | + fiscal impulse, λ_z reparam, centred z | 1983Q1 | 54 | r* span 0.5pp (z still dead) |
| 3 | + non-centred z, AR-only z, 1993Q1 sample | 1993Q1 | 616 | z alive (span 0.28pp), r* latest 2.10% |
| 4 | + non-centred trend_growth | 1993Q1 | 5,976 | Broken — doubly-cumulated cumsum |
| 5 | AR(1) z + looser λ_z + indexed-bond observation (Resolution B) | 1993Q1 | 5,107 | r* span 6.2pp, trough −0.30% |
| 6 | Blend r* = α·g + (1−α)·(indexed_10y − k) + ε (Resolution C) | 1993Q1 | 1,197 | r* span 3.15pp, trough 1.3%, α 0.52 |
| 7 | C + regime-switching α at 2008Q4 | 1993Q1 | 6,575 | α_pre 0.55, α_post 0.48 — regimes not separated; reverted |
| 8 | C, σ_g loosened to HalfNormal(0.10) | 1993Q1 | 5,771 | g slope clean, but r_star ESS 20 — unreliable |
| 9 | C + HMA(13) anchor on g (free σ_trend_obs) | 1993Q1 | 8,276 | σ_trend_obs collapsed to 0.022; COVID dip bled into g |
| 10 | C + linear-trend anchor on g, loose σ_g | 1993Q1 | 9,967 | g shape clean but funnel returned at endpoints |
| 11 | C + linear-trend anchor, tight σ_g, fixed σ_trend_obs=2.0 | 1993Q1 | 2,685 | r* span 3.58pp, trough 1.05%; clean sampling |
| 12 | Iteration 11 + non-centred r_innovation | 1980Q1 | 156 | σ_r funnel eliminated; r_innovation R-hat 1.001 |
| 13 | C + regime-switching α at 2011Q3–2021Q4 | 1980Q1 | 267 | α_normal 0.54, α_divergence 0.50 — regimes not separated; reverted |
| 14 | C with intercept c replacing k | 1980Q1 | 359 | c = −0.13, g unchanged at 2.66%; reverted |
| 15 | C with intercept c alongside k | 1980Q1 | 2,104 | c/k collinear; reverted |
| 16 | Resolution D: canonical + open-economy IS | 1986Q3 | 33 | a_r unchanged, z still dead, r* ≈ g |
| 17 | Resolution E: blend + AR(1) z, σ_z=0.15 | 1986Q3 | 302 | z mean abs 0.04pp; r* matches C |
| 18 | Resolution F: E + open-economy IS | 1986Q3 | 156 | All SOE coefs dead; r* matches C |
| 19 | σ_z prior sweep on E (σ_z ∈ {0.05, 0.15, 0.30, 0.50, 1.00}) | 1986Q3 | 270–1,583 | r* CI scales linearly with σ_z; median r* moves only 0.20pp across σ_z ∈ [0.05, 0.50] |
| 20 | α prior default Beta(2, 2) → Beta(1, 1) (Uniform) | 1986Q3 | 205–819 | Posterior 90% range tracks Uniform's range; data weakly tilts toward higher α |
| 21 | Hierarchical Beta(a, b) on C, a, b ~ HalfNormal(1) | 1986Q3 | 365 | Strongly bimodal α posterior — artefact of HalfNormal mass below 1 allowing U-shapes |
| 22 | Promoted hierarchical Beta to Resolution G | 1986Q3 | 365 | Same spec as 21, wired as a first-class resolution |
| 23 | Indexed-yield gap filled (2013Q3–2014Q3); A–G re-estimated on 158-quarter sample | 1986Q3 | 85–3,349 | Substantive r* paths shift <0.05pp; divergences drop dramatically |
| 24 | G hyperprior HalfNormal(1) → Uniform(0.25, 2) | 1986Q3 | 181 | a/b move from 0.74/0.62 to 1.18/1.09; α bimodality flattens — endpoint stacking was the prior's contribution |
| 25 | Resolution H: time-varying α_t via logit-RW (σ_a fixed at 0.05) | 1986Q3 | 206 | α_t flat (sd 0.002); r* matches C/G; data does not want time-variation in α |

Iteration 12 applies a standard non-centred reparameterisation to r_innovation: sample `r_innovation_raw ~ N(0, 1)` and scale by σ_r, rather than sampling `r_innovation ~ N(0, σ_r)` directly. This eliminates the Neal's funnel that formed when σ_r was small. Iterations 7 and 13 both tested regime-switching α (GFC split, then 2011Q3–2021Q4 "great divergence"). Both failed: α posteriors overlap heavily regardless of where the break is placed. Iterations 8–10 explored loosening σ_g — the lesson is that σ_g must stay tight or the y*-equation funnel returns; the working combination is tight σ_g + linear-trend soft anchor + fixed σ_trend_obs.

## What was tried that didn't work

| Approach | Outcome | Why it failed |
|----------|---------|---------------|
| `target_accept = 0.97` | Worse | Hides the geometry, doesn't fix it |
| Non-centring `trend_growth` (manual cumsum) | 5,976 divergences | y* already uses g[:-1]/4 inside its own cumsum — doubly-cumulated structure breaks NUTS gradients |
| Tighter σ_g and σ_z | Lowered divergences but killed z | Model becomes effectively univariate trend extraction |
| Constant tp in indexed-bond observation (B) | r* trough −1.64%, 3,349 divergences | Post-GFC term-premium compression pushed into r* level; HLW machinery becomes decorative |
| 1980Q1 sample start | Worse identification | Pre-1993 regime contaminates Phillips curve |
| Survey-based PIE_RBAQ for π_exp | Rejected upfront | Project has its own expectations model |
| Regime-switching α (GFC split, then 2011Q3–2021Q4) | Regimes not separated in either case | The data has too little identifying power on α to support time-variation |
| Loose σ_g without an anchor on g | Divergences blow up, ESS collapses | σ_g and σ_y* funnel without external info on g |
| HMA(13) of YoY GDP growth as g anchor (free σ) | σ_trend_obs collapsed to 0.022; COVID dip bled into g | Free measurement σ over-fits when it's the only constraint on g; HMA carries cyclical contamination into the "trend" |
| Slope-based time-varying k: k_t = k0 + k_slope · (10y_nominal − cash) | 10,655 divergences, ESS 17, r* path barely changed | k_t added flexibility without identifying information |
| Intercept c replacing k | 359 divergences; g unchanged | An intercept shifts level but doesn't release g |
| Intercept c alongside k | 2,104 divergences | c and (1−α)·k near-collinear |
| Time-varying term premium in indexed-bond equation | Rejected conceptually | Re-introduces the canonical identification failure |
| Mid-life fan-integration of r* into NAIRU | Not implemented | Sequential coupling (median series → NAIRU `obs["det_r_star"]`) is the realistic path |

## What was *not* tried (potential future work)

- **Term-structure block** (Bauer-Rudebusch 2020 "Falling Stars"): full arbitrage-free yield curve with multiple maturities. Would let term premium be a proper latent without re-introducing identification problems. Significant engineering effort.
- **Long-run survey expectations** (Del Negro et al 2017): use Consensus Economics 6–10y forecasts of the cash rate as an additional observation. Would pin r*'s long-run mean.
- **Convenience-yield observation** (Szoke-Vazquez-Grande-Xavier 2024 FEDS Note): would need long-history Australian AA corporate bond yield data — RBA F3 only goes back to ~2005.
- **AR(1) trend growth**: would mean-revert g, possibly stabilising σ_g. Risk: changes the long-run interpretation.

## File structure

```
src/models/rstar_hlw/
├── observations.py           # Data loading (incl. indexed_10y, linear-trend g anchor, SOE regressors)
├── equations/
│   ├── trend_growth.py       # g state equation (centred RW) + soft observation on g
│   ├── potential.py          # y* state equation
│   ├── r_star.py             # r* = α·g + (1−α)·(indexed_10y − k) + ε   (C and G)
│   ├── r_star_blended_z.py   # r* = α·g + (1−α)·(indexed_10y − k) + z   (E and F)
│   ├── z_star.py             # r* = g + z (A, B, D)
│   ├── is_curve.py           # IS curve; fiscal + opt-in SOE block (ToT, TWI, ICP)
│   ├── phillips.py           # Phillips curve on annual π
│   └── indexed_bond.py       # separate indexed_10y observation (B only)
├── estimate.py               # Model assembly, NUTS sampling, save/load; resolution dispatch
├── results.py                # RStarResults dataclass
├── analyse.py                # Fan charts: r*, output gap, g, decomposition, alpha posterior
├── run.py                    # CLI: --resolution, --start, --estimate-only, --skip-estimate, --seed
└── MODEL_NOTES.md            # This file

run-rstar-hlw.sh              # Shell wrapper
```

External data dependencies for the SOE block and the comparison chart:
- `src/data/tot.py` — ABS Terms of Trade quarterly % change
- `src/data/twi.py` — RBA TWI quarterly change (lag-1 helper)
- `src/data/commodity_prices.py` — RBA Index of Commodity Prices (A$), table I2
- `src/data/world_rstar.py` — NY Fed HLW r* for US/Euro/Canada (chart only, not in model)

## References

- Holston, Laubach, Williams (2017): "Measuring the Natural Rate of Interest"
- Holston, Laubach, Williams (2023 update): NY Fed Staff Report 1063
- Lewis, Vazquez-Grande (2019): "Measuring the Natural Rate of Interest" — λ_z reparameterisation, AR(1) z. Code: https://github.com/kflewis/rStarLVGPublic
- Buncic (2021): "On a standard method for measuring the natural rate of interest" — MUE critique. Code: https://github.com/4db83/Issues-with-HLWs-natural-rate-Code
- Buncic, Pagan, Robinson (2023): "On Constructing a Country-Specific Time Series for the Natural Rate of Interest" — formal identification critique. The "shocks ≥ observables → posterior is the prior projected through the model" finding underlies these notes.
- Szoke, Vazquez-Grande, Xavier (2024 FEDS Note): "Convenience Yield as a Driver of r*"
- Del Negro, Giannone, Giannoni, Tambalotti (2017): "Safety, Liquidity, and the Natural Rate of Interest". Code: https://github.com/FRBNY-DSGE/rstarBrookings2017
- Bauer, Rudebusch (2020): "Interest Rates Under Falling Stars"
- McCririck, Rees (RBA Bulletin Sep 2017): "The Neutral Interest Rate"
- Ellis (RBA speech 2022): "The Neutral Rate: The Pole-star Casts Faint Light"
