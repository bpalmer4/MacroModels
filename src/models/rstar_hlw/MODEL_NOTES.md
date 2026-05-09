# HLW Bayesian r-star Model — Model Notes

A Bayesian (PyMC + NumPyro NUTS) implementation of the Holston-Laubach-Williams 2017 model, applied to Australian quarterly data. Estimates the natural rate of interest r* jointly with potential output, trend growth, and the output gap.

## Purpose and motivation

The existing NAIRU model in `src/models/nairu/` uses a deterministic r* derived from the Cobb-Douglas production function:

> r* ≈ α·g_K + (1−α)·g_L + g_MFP   (smoothed via Henderson MA)

This is observable-driven and smooth, but offers no uncertainty quantification and tracks supply-side fundamentals only. The motivation for building a Bayesian HLW model was to:

1. Produce a r* with proper Bayesian uncertainty bands
2. Allow r* to be identified jointly with the output gap (rather than imposed from outside)
3. Eventually feed the HLW r* posterior into the NAIRU model's IS curve as an alternative to the Cobb-Douglas r*, with sequential coupling (run HLW first, save the median series, read it into NAIRU's `obs["det_r_star"]`)

## Model structure (current — Resolution G, the blend with hierarchical α)

The structural specification below is identical for Resolutions C and G; G differs from C only in the prior on α (hierarchical rather than fixed). The structure description that follows applies to both unless stated otherwise.

### Original Resolution C, the blend (kept for reference)

The canonical HLW formulation `r* = g + z` with z as a latent random walk turned out to be unidentifiable on Australian data — see "The fundamental tension" below. The current specification replaces the latent z with a **deterministic blend** of two anchors plus an i.i.d. innovation:

`r*_t = α · g_t + (1 − α) · (indexed_10y_t − k) + ε_t`

where:
- `g_t` is the latent trend growth (random walk, annualised %)
- `indexed_10y_t` is the observed 10-year inflation-linked Australian government bond yield
- `α ∈ [0, 1]` is a single scalar weight (**Beta(1, 1) = Uniform** prior; previously Beta(2, 2) — see prior-default change in headline finding)
- `k > 0` is a constant term-premium offset (TruncatedNormal(0.5, 0.5, lower=0))
- `ε_t ~ N(0, σ_r)` is i.i.d. measurement-style noise

This collapses the unidentified third latent state to a single scalar (α) and lets the data choose where to land between the two anchors.

### Latent states

| State | Symbol | Equation | Units |
|-------|--------|----------|-------|
| Trend growth | g_t | g_t = g_{t-1} + ε_g | Annualised % |
| Potential output | y*_t | y*_t = y*_{t-1} + g_{t-1}/4 + ε_{y*} | log × 100 |

There is no z latent state in the current specification — the legacy `z_star.py` is kept in `equations/` for reference but is not wired into the model.

### Observation equations

1. **IS curve** (HLW 2017 form, with fiscal impulse):
   `log_gdp_t = y*_t + a_y1·y_gap_{t-1} + a_y2·y_gap_{t-2} + (a_r/2)·(r_gap_{t-1} + r_gap_{t-2}) + γ_fi·fiscal_{t-1} + ε_IS`

2. **Phillips curve** (annual trimmed mean, anchor-augmented):
   `π_4_t = π_exp_t + b_y · y_gap_{t-1} + ε_π`

3. **Soft anchor on g** (linear trend of YoY GDP growth):
   `linear_trend_t = g_t + ε_trend`,  `ε_trend ~ N(0, 2.0)` *(σ fixed)*

   Where `linear_trend_t` is a linear regression of year-on-year GDP growth fitted over the model sample period. Acts as a gentle pull on g toward the secular slowdown narrative — without this, g gets pulled by the y* equation toward sample-average aggregate GDP growth (~3%) and never declines.

The legacy `indexed_bond.py` observation equation (`indexed_10y_t = r*_t + tp + ε_tp`) is kept in `equations/` but is **not** wired into the current model — the indexed yield now appears in the *definition* of r* (above), not as a separate observation.

### Reparameterisations and parameterisation choices

- **trend_growth is centred** (`pm.GaussianRandomWalk`). Non-centring it produced catastrophic divergences (5,976+) when combined with the y* equation, which already uses g[:-1]/4 inside another cumsum — doubly-cumulated structure breaks NUTS gradients.
- **`σ_g ~ HalfNormal(0.04)`** — tight prior, kept this way after exploration showed that loosening it (HalfNormal(0.10)) blew up divergences without materially improving identification.
- **`σ_trend_obs` is fixed at 2.0**, not estimated. Letting it be a free parameter (`HalfNormal(1.5)`) led the data to collapse it to ~0.02, turning the "soft" anchor into a hard constraint that forced g to follow the smoothed observable. Fixing keeps the anchor genuinely soft.
- **Linear regression** rather than Henderson MA for the trend observable: HMA(13) had the COVID GDP collapse baked into the smoothed series, which then bled into g. The linear regression is immune to single-quarter shocks.
- **The α scalar is Beta(1, 1) = Uniform** so the prior is genuinely uninformative on [0, 1] — every weighting equally likely a priori. Previously Beta(2, 2), which centred mass at 0.5 and tapered to zero at the endpoints; that prior was found to be symmetrically pulling the posterior median toward 0.5 and masking a (weak) data preference for higher α. The Uniform default lets the posterior shape reflect the likelihood. Override via `constant['alpha_prior'] = (a, b)` for sweeps; override via `constant['alpha_hierarchical'] = True` for the hierarchical Beta where a, b ~ HalfNormal(1).
- **r_innovation is non-centred.** The i.i.d. ε_t in the r* blend was originally parameterised as `r_innovation ~ N(0, σ_r)` (centred). Because σ_r wants to be small (the blend already explains most of r*), this created a Neal's funnel — when σ_r shrinks, the r_innovation vector must shrink proportionally, producing a narrow neck NUTS can't navigate. Reparameterised to `r_innovation_raw ~ N(0, 1)` scaled by σ_r, decoupling the geometry. This reduced divergences from 908 to 156, fixed σ_r convergence (R-hat 1.090 → 1.000, ESS 41 → 44,528), and fixed r_innovation convergence (R-hat max 1.053 → 1.001). Remaining divergences are in σ_g (the trend_growth GRW scale), which cannot be non-centred (see above).
- **σ_r is largely prior-dominated.** The IS curve already has its own σ_IS absorbing rate-gap noise, and a_r is small (~0.05), so σ_r contributes little to the IS likelihood.

## What r* actually means in this model

In any HLW-style model, **r\* is the time-varying intercept of the IS curve.** It's the level of the real rate at which the output gap has no demand-side pressure on it. Canonical HLW tries to estimate that intercept jointly with the output gap from a latent random walk (z). For Australia that doesn't work — the IS curve's `a_r` channel is too weak to identify a separate latent.

What we're doing now is different. Instead of letting r* be a purely latent random walk that the IS curve has to identify, we're **constructing r\* externally** as a deterministic blend of two observable-driven anchors:

- The **structural anchor** (g, trend growth) — what r* "should" be from the supply-side
- The **market anchor** (indexed_10y − k) — what real rates actually look like in long-term bond markets

Then the IS curve sees this constructed r* and asks "given this is the intercept, do you fit the data well?" The α parameter is the data's verdict on which anchor is more consistent with the IS curve dynamics.

A few observations that follow:

1. The α posterior centred at 0.56 with wide HDI [0.07, 0.96] (under the Uniform default) is the data saying "either anchor works about equally well — neither is obviously wrong, neither is obviously right; very mild tilt toward the trend-growth anchor". See the headline finding's α-prior sensitivity subsection for evidence that the wide HDI is largely the prior projected through the model rather than data uncertainty.
2. **The IS curve isn't *identifying* r\* in the canonical sense** — it's *vetting* a constructed r*. If both anchors implied the same r* path the IS curve would have nothing to discriminate; because they imply different paths (one flat at ~3%, one moving ~6pp), the IS curve picks a weighted blend.
3. The intercept is "time-varying" only because both anchors move slightly differently. g moves slowly via the trend-growth random walk; indexed_10y moves a lot via market dynamics. The blend tracks somewhere in between.
4. This makes it explicit that we're not estimating "the natural rate" as a deep structural object — we're estimating the **IS curve's intercept as a weighted combination of two external proxies**. That's an honest description of what HLW always was, just with the latent z replaced by an observable blend.

## Data inputs

| Series | Source | Notes |
|--------|--------|-------|
| log GDP × 100 | `gdp.get_log_gdp()` | ABS 5206.0 chain volume measures, SA |
| Cash rate | `cash_rate.get_cash_rate_qrtly()` | RBA OCR + historical interbank, end-of-quarter |
| π_exp (annualised %) | `expectations_model.get_model_expectations_unanchored()` | **Not** PIE_RBAQ — uses the project's own signal extraction model, unanchored variant |
| π_4 (annual trimmed mean) | `inflation.get_trimmed_mean_annual()` | ABS 6401.0 |
| Fiscal impulse (lag 1) | `gov_spending.get_fiscal_impulse_lagged_qrtly()` | Same source as the NAIRU model uses |
| Indexed 10y bond yield | `bonds.get_indexed_yield_filled()` | RBA F2 (1986+); a 5-quarter gap (2013Q3–2014Q3) where the benchmark indexed bond was being transitioned is filled via `nominal_10y − interpolated breakeven` (see "Indexed yield data fill" below) |
| Linear trend (g anchor) | derived in `observations.py` | Linear regression of YoY GDP growth over the filtered sample. Slope ≈ −0.07 pp/year for 1993Q1-2025Q4. Used as the soft observation equation on g. |

## Sample period

**1993Q1 → latest available.**

This is *not* 1980Q1, despite the long-term integration goal (NAIRU starts 1980Q1). The reasons:

- Pre-1993 Australia had no inflation target and a different monetary regime; the Phillips curve relationship is structurally different.
- The 1980s disinflation involved expectations dynamics that this single-regime model cannot capture cleanly.
- HLW (canonical) has no anchoring mechanism for pre-target regimes.
- Empirically, including 1983Q1–1992Q4 produces a spuriously narrow r* range and worse identification.

For NAIRU integration, the gap between 1980Q1 and 1992Q4 can be filled by:
- Backward-extrapolating the HLW r* (e.g., hold at the 1993Q1 estimate or use a simple trend), or
- Continuing to use the deterministic Cobb-Douglas r* for the pre-1993 period and switching to HLW r* from 1993Q1.

**In practice the model uses 1986Q3 → 2025Q4** (158 contiguous quarters), aligned with the start of the indexed bond yield series. Earlier vintages of this codebase had a 153-period sample because the indexed yield series has a 5-quarter hole 2013Q3–2014Q3; that gap is now filled (see below) and the sample is contiguous.

### Indexed yield data fill

The RBA's F2 series for the 10y indexed bond yield has a **5-quarter gap from 2013Q3 to 2014Q3**. The Australian Treasury was transitioning between the maturing 2020 indexed benchmark and the new 2025 benchmark over that window, and the standardised "10-year" measure had no clean underlying instrument — so RBA didn't publish a yield.

`src/data/bonds.py::get_indexed_yield_filled()` patches the gap as follows:

1. Compute breakeven inflation = `nominal_10y − indexed_10y` everywhere both series are observed.
2. Linearly interpolate breakeven across the missing quarters.
3. For each missing quarter where `nominal_10y` is observed, set `indexed_filled = nominal_10y − breakeven_interpolated`.

Why this works: breakeven inflation is anchored to expected inflation (which is itself anchored to the RBA target), so it moves slowly and interpolates well. The nominal 10y yield is observed throughout the gap and contributes the actual real-rate dynamics — notably the **2013 taper-tantrum spike** in nominal yields (3.5% → 4.3% mid-2013) which translates into a real-rate spike that a naïve interpolation of indexed_10y itself would miss.

| Quarter | Nominal | Breakeven (interp) | Indexed (filled) |
|---|---|---|---|
| 2013Q3 | 4.03 | 2.49 | 1.54 |
| 2013Q4 | 4.26 | 2.44 | 1.82 |
| 2014Q1 | 4.09 | 2.39 | 1.70 |
| 2014Q2 | 3.71 | 2.34 | 1.37 |
| 2014Q3 | 3.57 | 2.30 | 1.28 |

After filling, the model's joint observation matrix has 158 contiguous quarters instead of 153. All A–G traces in the comparison tables below use the filled sample. Re-estimation reduced divergences substantially across most resolutions (A: 4,050 → 111; B: 11,458 → 3,349; C: 428 → 193; E: 819 → 173) — the filled sample's continuity helps the latent random walks fit more cleanly.

## The fundamental tension

The HLW model has a long-documented identification problem: with three latent random walks (g, z, y*) and only two real-side observation equations (IS, Phillips), the data does not separately identify g and z. The original HLW (2017) addressed this with median-unbiased estimation (MUE); Buncic (2021) shows this MUE is statistically wrong, producing spuriously large λ_z values that fabricate r* movement.

In a Bayesian framework, the identification problem doesn't disappear — it just gets transferred to the priors. We confronted this empirically by exploring three model families.

### Resolution A: r* ≈ g (canonical HLW with z, no bond observation)

`r*_t = g_t + z_t` with z a latent random walk (or AR(1)) — the canonical HLW formulation, only IS curve and Phillips curve to identify r*.

- σ_z posterior collapses toward zero; z is essentially a small constant offset
- r* tracks g almost exactly; r* span ~0.5 pp over the sample
- a_r posterior near zero (HDI [-0.11, 0.00]) — IS rate channel barely identified
- ~600 divergences concentrated in σ_g (HalfNormal piling up at zero)
- Latent states converge cleanly (R-hat ≈ 1.00)

Matches Buncic's finding for Euro Area / UK / Canada: the correct λ_z is essentially zero and HLW's apparent r* movements are MUE artefacts. McCririck-Rees (RBA 2017) and Ellis (2022 RBA speech) report similar identification failure for Australia and resort to model averaging.

### Resolution B: r* ≈ indexed_10y − 0.5% (canonical HLW + indexed bond observation)

Same `r*_t = g_t + z_t` structure plus an indexed-bond observation equation `indexed_10y_t = r*_t + tp + ε_tp`. Strong identifying information from the bond yield.

- λ_z posterior jumps from ~0.08 to 0.25 (z permitted to move much more)
- ρ_z ≈ 0.97 (highly persistent AR(1))
- z absorbs essentially all r* dynamics; g stable at ~2.7-3.1%
- r* span ~6 pp; r* tracks indexed_10y minus the constant term premium
- ~5,000 divergences (latent states converge, σ_g still poorly mixed)
- r* level becomes implausible at the trough: ~−0.30% in 2020 (vs. RBA/IMF/OECD estimates around 0.5-1.5%)

The negative-r* level is a direct consequence of the constant-term-premium assumption — real term premia compressed substantially post-GFC and we absorbed that compression into r* rather than into tp. Letting tp be a slow-moving latent random walk would re-introduce the very identification problem this equation is supposed to fix.

The HLW machinery becomes decorative in this resolution: r* is essentially a relabelling of the bond yield, and the IS / Phillips curves do little identifying work.

**Circularity caveat:** the indexed bond yield reflects expected future cash rates plus a real term premium. The cash rate is set by the RBA partly in response to its own view of r*. So r*-from-indexed-bond partly recovers "the rate the RBA thinks is neutral" rather than a deep structural object.

### Resolution C: r* = α·g + (1−α)·(indexed_10y − k) + ε  *(current default)*

A deterministic blend of the two anchors. No latent z. The single scalar α tells the data how much to weight the structural anchor (g) versus the market anchor (indexed bond yield − constant term premium).

Numbers below reflect the current spec (α ~ Beta(1, 1) Uniform; previously Beta(2, 2)) on the gap-filled sample (158 quarters).

- α posterior median ≈ **0.56**, 90% HDI **[0.07, 0.96]** — posterior 90% range tracks the Uniform prior's 90% range almost 1:1; data has weak preference for higher α (slight tilt toward trend-growth anchor)
- k posterior median ≈ 0.62 (term premium offset)
- σ_r posterior median ≈ 0.20 (largely prior-dominated, as expected)
- r* span ≈ **3.31 pp**
- r* path: trough **0.82% (2020Q4)**, latest **2.19% (2025Q4)**
- ~193 divergences (down from 428 on the un-filled sample; the contiguous data plus Beta(1, 1) prior produces clean enough geometry)
- Latent r* convergence: R-hat 1.000, ESS 6,133 (clean — sharper than pre-fill ESS 4,588)
- σ_g posterior 0.043 (within prior HalfNormal(0.04) regime), `b_y` 0.41, `a_r` −0.035, σ_IS 0.68

**What this spec offers (without claiming it is the right answer):**
- Level sits inside the published Australian range (trough 0.82%, never negative); whether that is a virtue or a feature of the construction depends on how much weight one places on the bond anchor — see "Recommended use" and "External cross-validation: Bullock's 2026 statements".
- Dynamics trace a post-GFC decline + recent recovery, because both anchors do — the blend inherits whatever shape its components have.
- g shows a secular slowdown thanks to the linear soft anchor — without the anchor, g would sit flat near sample-average aggregate GDP growth (~3%).
- α posterior with wide HDI [0.07, 0.96] is itself the finding: the data weakly tilts toward higher α (median 0.56) but does not pick one anchor over the other.
- Identification is reduced to one scalar (α), which is itself weakly identified — the IS curve cannot discriminate between α weightings within the blend family (confirmed independently by Resolutions G and H; see headline finding).
- Honest about what r* is: a blend of structural and market signals, with weights chosen mostly by the analyst's prior rather than by the data.

### Resolution D: canonical r* + open-economy IS curve

Same r* identity as A — `r*_t = g_t + z_t` with z an AR(1)-reparameterised non-centred random walk — but with an **open-economy IS curve** adding three external regressors on top of A's closed-economy specification:

```
y_gap_t  =  a_y1·y_gap_{t-1} + a_y2·y_gap_{t-2}
          + (a_r/2)·(r_gap_{t-1} + r_gap_{t-2})
          + γ_fi  · fiscal_{t-1}
          + γ_tot · tot_{t-1}              [terms-of-trade growth]
          + γ_twi · twi_{t-1}              [TWI change — exchange rate channel]
          + γ_icp · icp_{t-1}              [RBA ICP A$ growth — Asian commodity demand]
          + ε_IS
```

The soft linear-trend anchor on g is also wired in (same as C), so g doesn't drift unanchored — that lets us isolate the IS-curve effect from the unrelated trend-growth identification problem that plagues A.

**Hypothesis being tested.** Buncic (2021), McCririck-Rees (RBA 2017), and Ellis (RBA 2022) all find canonical HLW fails for SOEs because the IS-curve rate channel is too weak. The hypothesis was that the weakness is *partly* mis-specification: σ_IS is large because external (SOE) shocks are absorbed into the residual; if we add the SOE block, σ_IS shrinks, a_r firms, and z lights up.

**Result: hypothesis empirically refuted on Australian data.** Numbers below are on the gap-filled sample (158 quarters).

- σ_IS posterior median ≈ **0.668** (vs A: 0.68, C: 0.68 — essentially unchanged across all resolutions)
- a_r posterior median ≈ **−0.032**, HDI [−0.07, −0.01] (vs A: −0.033, C: −0.035 — essentially unchanged)
- σ_z posterior median ≈ **0.06**, piling up against the prior boundary (z still dead, as in A)
- r* range: **[1.77, 2.68]%**, span **0.91 pp** — flat path that effectively tracks g
- 85 divergences (vs 33 on un-filled sample — slight increase but still well-behaved)
- γ_fi ≈ 0.05, γ_tot ≈ 0.01 (piled at zero), γ_twi ≈ −0.01, γ_icp ≈ 0.02 — all small; γ_twi has the expected negative sign, γ_icp the expected positive sign
- Latent convergence: r_star R-hat 1.010, ESS 548; clean but z is essentially constant

**Why this matters even though the hypothesis failed.** D is a tighter empirical confirmation than A that canonical HLW does not work for Australia. A could be dismissed as "weak result because the closed-economy IS curve is mis-specified for an SOE". D answers that critique directly: with an open-economy IS curve including the three principal SOE channels (income, exchange rate, foreign demand), the result is the same — z is dead, r* ≈ g. The Australian rate-output channel is genuinely weak; the SOE block doesn't fix it.

**Substantive finding.** For Australia, no choice of canonical HLW with realistic IS-curve specification produces a credibly identified, time-varying r* on top of g. The deterministic blend in C is not a workaround for a fixable problem; it's the only spec on this dataset that yields a usefully variable r* — and that's because C abandons the canonical pretence that the IS curve identifies r* and instead constructs r* from external anchors directly.

D is kept in the codebase as a contrastive baseline: A demonstrates closed-economy canonical-HLW failure, D demonstrates that open-economy canonical-HLW also fails. Both are kept for reproducibility; neither is recommended for substantive use.

### Resolution E: r* = blend + AR(1) z (soft-anchor variant of C)

A clean generalisation that contains C and A as special cases. Keep C's blend as the structural anchor (replacing canonical HLW's `g`), and add z back as a persistent (AR(1)) deviation from that anchor:

```
r*_t   =  α·g_t + (1−α)·(indexed_10y_t − k) + z_t
z_t    =  ρ_z · z_{t−1} + ε_z,    ε_z ~ N(0, σ_z)
σ_z    =  0.15  (fixed)
ρ_z    ~  TruncatedNormal(0.95, 0.03, [0, 1])
```

- α = 1 → reduces to canonical HLW (Resolution A).
- σ_z = 0 → reduces to deterministic blend (Resolution C).

The z posterior is the directly interpretable readout of *what the IS curve identifies above and beyond the blend*. σ_z is fixed (not estimated); the iteration-9 lesson applies — a free σ_z would collapse to ~0 against an underconstrained IS curve, and we would silently recover Resolution C with extra latent-state machinery.

**Result: empirically refuted that the IS curve adds anything.** Numbers below reflect the current Beta(1, 1) default on the gap-filled sample (158 quarters).

- z posterior median range: ~[−0.08, +0.04] over 40 years (span ~0.12 pp)
- z mean absolute: 0.041 pp
- z latest: ≈ 0
- α posterior median 0.565, 90% HDI [0.08, 0.96] — wide; data only weakly tilts toward higher α (matches C)
- ρ_z posterior ≈ 0.95 — virtually identical to prior
- All other parameters (a_r, σ_IS, σ_g, k, b_y) within Monte Carlo error of C
- r* range [0.83, 4.09] vs C's [0.82, 4.13]
- r* trough 0.83 (2020Q4) vs C's 0.82 (2020Q4)
- r* latest 2.20 vs C's 2.19
- Divergences: 173 (down from 819 on un-filled sample; gap fill helps E even more than C)
- Convergence: r_star R-hat 1.000, ESS 12,420; z_star R-hat 1.000 (still extremely clean)

The z posterior is dead. The IS curve has no systematic information to add above the blend that C suppresses with its small i.i.d. ε_t. C's deterministic identity is not over-constraining the answer — the blend is what r* is.

E is kept as a contrastive baseline: the soft-anchor form of C produces the same r* as C's deterministic identity. Confirms that giving the IS curve room to disagree with the blend does not produce a different answer.

### Resolution F: E's r* identity + open-economy IS curve

Combination of E and D: the blend + AR(1) z r* spec, with the SOE-block IS curve from D added (fiscal + ToT + TWI change + ICP growth). Tests whether the combination — soft anchor giving room for z + extra regressors giving the IS curve more material — finds something that neither alone could.

**Result: empirically refuted again.** Numbers below reflect the current Beta(1, 1) default on the gap-filled sample (158 quarters).

- z posterior median range: ~[−0.10, +0.03] (span ~0.13 pp; same as E)
- z mean absolute: 0.051 pp (vs E's 0.041)
- a_r −0.036, σ_IS 0.671, α median 0.527 (HDI [0.06, 0.95]) — within MC error of C
- γ_tot ~0.01 (piled at 0); γ_twi ~−0.01; γ_icp ~0.02 — same dead patterns as in D
- r* range [0.63, 4.13], latest 2.15 (vs C's [0.82, 4.13], 2.19)
- Divergences: 249 (vs C's 193 on the same filled sample; slightly worse than C, but down from 205 on the un-filled sample)
- Convergence: r_star R-hat 1.000, ESS 16,565 (still the best of any resolution)

F closes the empirical loop. Across the full taxonomy (A through F), σ_IS sits at 0.70 ± 0.02, a_r at −0.04 ± 0.01, and r* ≈ blend wherever the blend is constructed. The IS curve's rate channel is a property of Australian data; no choice of r* identification or IS-curve regressor specification makes it strong.

F is kept alongside E as a contrastive baseline.

### Resolution G: blend with hierarchical Beta(a, b) on α  *(current default)*

Same r* identity as C — `r*_t = α·g_t + (1−α)·(indexed_10y_t − k) + ε_t` — but with a hierarchical prior on α that lets the data choose the Beta shape itself:

```
a_hyper, b_hyper  ~  Uniform(0.25, 2)
α                  ~  Beta(a_hyper, b_hyper)
```

**Why Uniform(0.25, 2) and not HalfNormal(1).** The first version of G used `HalfNormal(1)` as the hyperprior on a and b — but HalfNormal(1) puts ~58% of its mass below 1, which is a soft *a priori preference for U-shaped Betas* (since Beta(a, b) with a, b < 1 is U-shaped). That hyperprior choice was found to substantially amplify the apparent bimodality of α's posterior. The current `Uniform(0.25, 2)` is **flat** within a range that spans U-shapes (a, b < 1) and mild bell-shapes (a, b > 1), and is symmetric around 1 — neutral about the Beta shape. See iteration 24 in the progression table for the comparison.

**Why hierarchical at all.** Fixed Beta(2, 2) and Beta(1, 1) experiments showed that any *fixed* α prior is doing meaningful work in the posterior — the within-model CI essentially tracks the prior's range. Letting the data pick the prior shape removes the analyst's choice from the picture and produces a more informative output about α.

**Result on the gap-filled sample (158 quarters):**

| Quantity | Posterior median | 90% HDI |
|---|---|---|
| `α_a_hyper` | 1.179 | [0.35, 1.92] |
| `α_b_hyper` | 1.087 | [0.33, 1.91] |
| α | 0.581 | [0.03, 0.99] |

α posterior quantiles 5/25/50/75/95: **0.03 / 0.30 / 0.58 / 0.83 / 0.99**.

Approximate density at endpoints / middle (40-bin histogram, density-normalised): **1.70 at α near 0**, **0.96 in the middle**, **3.00 at α near 1**. The posterior is **right-tilted with mild endpoint pile-up** — meaningfully different from the strictly-bimodal HalfNormal version (which had density ~4.4 at 0, ~0.6 in middle, ~8.4 at 1).

**Both hyperparameters now sit slightly above 1** (1.18 and 1.09). The data, given a flat hyperprior, mildly prefers **bell-shaped** Betas over U-shapes — reverse of the HalfNormal version. The earlier "data picks Jeffreys-like" framing was substantially the HalfNormal prior's contribution, not a genuine data finding.

**Other parameters** (within MC error of C):
- k ~0.63, σ_r 0.21, a_r −0.034, σ_IS 0.674, σ_g 0.043, b_y 0.41, γ_fi 0.06

**r\* path under hierarchical α:**
- Range [0.89, 4.11], trough **0.89% (2020Q4)**, latest **2.20%**
- 181 divergences (improved from 307 with HalfNormal(1) hyperprior; the flatter hyperprior gives the chain less to chase at the boundary)
- Latent r* convergence: R-hat 1.000, ESS 6,169 (clean)
- The trough is now closer to C's 0.82% than to the old HalfNormal-G's 1.18%, because the U-shape's α=1-attractor that pulled r* toward g has weakened

**Why G is the running default (a methodological choice, not a claim about answers):**
- C's posterior on α is mostly the prior shape — informative about *the analyst's choice*, not about the data.
- G's hierarchical structure lets the data express any preference it has about the Beta shape; under the neutral `Uniform(0.25, 2)` hyperprior, the answer is "a slightly bell-shaped Beta with mild right-tilt and soft endpoint pile-up" — not the dramatic bimodality the original `HalfNormal(1)` hyperprior produced.
- An earlier version of these notes treated G's bimodality as a substantive headline finding. Iteration 24 (hyperprior swap) and Resolution H (time-varying α) together undermined that read: the bimodality was substantially the hyperprior's contribution, and when α is allowed to drift period-by-period it lands almost flat at ~0.59. The honest summary is that the data has very little to say about α — neither central tendency, nor shape, nor era-specific drift.

C is kept as the simpler alternative when downstream code wants a stable r* path without the hierarchical machinery; the median r* path under C, G, and H is essentially identical because all three are weighted means of the same two anchors with α near 0.55–0.59. The choice between them is a presentational one, not a substantive disagreement.

### Resolution H: blend with time-varying α_t (logit-RW)

Same r* identity as C but with α as a time-varying latent on the logit scale:

```
logit_α_0       ~  N(0, 2)                        [initial value, broad]
σ_a              =  0.05  (fixed)                  [logit-RW innovation scale]
ε_α_t           ~  N(0, 1)                         [non-centred raw]
logit_α_t       =  logit_α_{t−1} + σ_a · ε_α_t     [random walk]
α_t              =  sigmoid(logit_α_t)              [back-transform to (0,1)]

r*_t            =  α_t · g_t + (1 − α_t) · (indexed_10y_t − k) + ε_t
```

**Why H exists.** Constant-α specs (C, G) produce a wide r* posterior CI at every t — but most of that CI is *sample-wide* uncertainty about a single α projected period-by-period, not period-specific uncertainty about r* (the "smear" effect). Time-varying α_t lets the data express era-specific preferences for one anchor over the other, if any exist. Earlier regime-switching α experiments (iterations 7 and 13) failed because the data couldn't pick a hard date; a continuous logit-RW could in principle find smooth gradients without forcing a breakpoint.

The Bullock external-validation discussion above also pointed to H as the natural test: if the data agreed with Bullock's "r* has shifted upward" framing, α_t should drift toward 0 (bond anchor) in recent years.

**Result: the data does not want time-variation in α.** σ_a = 0.05 on the logit scale allows substantial drift (~1pp/quarter at α=0.5), so the chain has plenty of freedom — but α_t lands almost flat:

| Date | α_t (median) |
|---|---|
| 1990Q4 | 0.584 |
| 2000Q4 | 0.586 |
| 2010Q4 | 0.587 |
| 2020Q4 | 0.590 |
| 2025Q4 | 0.592 |

α_t std deviation across the entire 158-quarter sample: **0.002**. Total drift end-to-end: **0.01pp** (from 0.583 in 1989Q1 to 0.593 in 2023Q1). The α_t fan chart is essentially a horizontal line at ~0.59.

**Other parameters** (within MC error of C/G):
- k 0.627, σ_r 0.20, a_r −0.034, σ_IS 0.674, σ_g 0.046, b_y 0.42, γ_fi 0.066
- 206 divergences
- r_star R-hat 1.000, ESS 6,246; alpha_rstar R-hat 1.000, ESS 88,394 (extremely clean)

**r\* path under time-varying α:**
- Range [0.92, 4.11], trough **0.92% (2020Q4)**, latest **2.20%**
- Almost identical to C's [0.82, 4.13] / 2.19% and G's [0.89, 4.11] / 2.20% — because α_t is almost a constant, H reproduces the constant-α answer

**This is not a Buncic-style "prior dominates" finding.** The prior on α_t under H is permissive (logit-RW with σ_a = 0.05). Under such a prior, α_t left to its own devices would drift visibly across 40 years. The chain *actively pulls α_t toward a constant* — that's a real, identified outcome from the IS curve and the rest of the joint likelihood.

**Substantive interpretation.** The IS curve's residuals do not have era-specific structure that aligns with one anchor over the other. Whether r* tracks bond-yield-minus-tp or g, the IS curve sees about the same residuals. So the data has no leverage to push α_t away from its central value in any era. Bullock's "r* shifted upward post-2024" narrative — which would require α_t to drift toward 0 in the recent period — is not supported by the data inside this model.

H is kept as a contrastive baseline: it demonstrates that the constant-α structure in C/G is not constraining the answer artificially. When given the freedom to drift, the data chooses not to. The bimodality of α we saw in G under HalfNormal(1) is clearly a prior artefact (since H, with a different prior structure, lands cleanly at a single α with no bimodality).

## Switching resolutions

The CLI exposes a `--resolution` flag that toggles between A through H:

```bash
./run-rstar-hlw.sh -v                          # Resolution G (default; blend + hierarchical Beta)
./run-rstar-hlw.sh -v --resolution C           # blend with fixed Beta(1, 1) on alpha
./run-rstar-hlw.sh -v --resolution A           # canonical HLW: r* = g + z
./run-rstar-hlw.sh -v --resolution B           # canonical + indexed-bond observation
./run-rstar-hlw.sh -v --resolution D           # canonical r* + SOE IS curve
./run-rstar-hlw.sh -v --resolution E           # blend + AR(1) z (soft-anchor variant of C)
./run-rstar-hlw.sh -v --resolution F           # E's r* identity + SOE IS curve
./run-rstar-hlw.sh -v --resolution H           # blend with time-varying alpha_t (logit-RW)
```

All resolutions share potential output and the Phillips curve. They differ in the r* identity, the α prior (where applicable), which IS-curve regressors are active, and whether the soft linear-trend anchor on g is wired in:
- **A**: `r* = g + z` — z is an AR(1)-reparameterised non-centred random walk (Lewis-Vazquez-Grande 2019 form). Closed-economy IS curve. No g anchor. Textbook canonical HLW baseline.
- **B**: `r* = g + z` plus indexed-bond observation `indexed_10y = r* + tp + ε`. Closed-economy IS curve. No g anchor. Textbook canonical with a term-structure pin.
- **C**: `r* = α·g + (1−α)·(indexed_10y − k) + ε` — the blend, with α ~ Beta(1, 1) Uniform. IS curve includes fiscal impulse. Soft linear-trend anchor on g.
- **D**: `r* = g + z` (same identity as A) but with an open-economy IS curve adding ToT growth, real TWI change, and RBA ICP (A$) growth as external regressors, plus the soft linear-trend anchor on g. Tests whether the SOE block firms a_r enough that z lights up.
- **E**: `r* = α·g + (1−α)·(indexed_10y − k) + z` with z AR(1) (ρ_z ~ TN(0.95, 0.03), σ_z fixed at 0.15). Generalises C and A. Tests whether softening C's deterministic identity to a soft anchor reveals IS-curve information that the i.i.d. ε in C suppresses.
- **F**: E's r* identity combined with D's open-economy IS curve. Tests whether the combination of soft anchor + extra IS-curve regressors finds something neither alone does.
- **G** *(default)*: same r* identity as C, but with α ~ Beta(a, b) where a, b ~ Uniform(0.25, 2) (hierarchical). Lets the data pick the prior shape itself; under the neutral hyperprior, the data picks a slightly bell-shaped Beta with mild right-tilt and soft endpoint pile-up.
- **H**: same r* identity as C, but with α as a time-varying latent on the logit scale (RW, σ_a fixed at 0.05). Tests whether the data wants α to drift over time. **Result: it doesn't** — α_t lands almost perfectly flat across the 40-year sample.

Note: the "Resolution A" wired into the toggle uses the AR(1) reparameterised z, not the strict canonical RW z. The strict canonical version was only run in iteration 1 of the sampler progression and produced r* ≈ g (dead z).

## Australia vs G3 r* comparison chart

`analyse.py` always emits an overlay of the model's median r* against the NY Fed Holston-Laubach-Williams r* estimates for **US, Euro Area, and Canada** (`src/data/world_rstar.py`). The three foreign series are plotted individually — no trade-weighted aggregate.

The chart is purely descriptive — none of the foreign r* series are observations in the model. `analyse.py` calls `get_world_rstar(force_download=True)`, so each run pulls the latest published file from `https://www.newyorkfed.org/research/policy/rstar` rather than relying on the cached copy in `input_data/`. The chart is clipped to start at the AU sample start so the longer US/Canada history (back to the 1960s) doesn't squash the AU series.

## Empirical comparison: A through H on the same Australian data

All eight resolutions were run on the identical 1986Q3-2025Q4 sample. They share the same Phillips curve and potential output equation. The differences are summarised in the side-by-side table below; the structural-level differences in the closed-/open-economy IS curves and r* identities are:

| Spec | A (textbook canonical) | B (canonical + bond observation) | C (blend, current default) |
|---|---|---|---|
| r* identity | r* = g + z (RW z) | r* = g + z (RW z) | r* = α·g + (1−α)·(indexed_10y − k) + ε |
| Indexed bond | not used | observed: `indexed_10y = r* + tp + ε_tp` | inside r* identity |
| IS curve fiscal regressor | absent | absent | included |
| Soft anchor on g | absent | absent | present (linear trend, fixed σ_trend_obs = 2.0) |

### Side-by-side posterior summary

All figures below are from re-estimation on the **gap-filled sample (158 quarters, 1986Q3–2025Q4)** with current defaults: α ~ Beta(1, 1) Uniform for C/E/F, hierarchical α for G. A and B have no α parameter. All resolutions, seed=42.

| | A (canonical) | B (canonical + bond) | C (blend) | D (canonical + SOE IS) | E (blend + AR(1) z) | F (E + SOE IS) | **G (default; C + hierarchical α)** | H (blend + tv α_t) |
|---|---|---|---|---|---|---|---|---|
| Divergences | 111 | **3,349** | 193 | 85 | 173 | 249 | 181 | 206 |
| r* span (median path) | **0.42 pp** (flat) | **6.51 pp** (wild) | 3.31 pp | **0.91 pp** (flat) | 3.26 pp | 3.50 pp | 3.22 pp | 3.19 pp |
| r* range (median) | 2.41% to 2.83% | **−1.64% to 4.87%** | 0.82% to 4.13% | 1.77% to 2.68% | 0.83% to 4.09% | 0.63% to 4.13% | 0.89% to 4.11% | 0.92% to 4.11% |
| r* trough (median) | 2.41% (2019Q2) | **−1.64%** (2020Q4) | 0.82% (2020Q4) | 1.77% (2020Q1) | 0.83% (2020Q4) | 0.63% (2020Q4) | 0.89% (2020Q4) | 0.92% (2020Q4) |
| r* latest (2025Q4) | 2.43% | 1.48% | 2.19% | 1.77% | 2.20% | 2.15% | **2.20%** | 2.20% |
| z span (median) | dead | wild | n/a | dead | **dead (~0.12 pp)** | **dead (~0.13 pp)** | n/a | n/a |
| `r_star` R-hat (ESS) | 1.020 (243) | 1.010 (366) | **1.000** (6,133) | 1.010 (548) | **1.000** (12,420) | **1.000** (16,565) | **1.000** (6,169) | **1.000** (6,246) |
| `a_r` posterior median | −0.033 | −0.031 | −0.035 | −0.032 | −0.035 | −0.036 | −0.034 | −0.034 |
| `σ_IS` posterior median | 0.68 | 0.68 | 0.68 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| `α` posterior | n/a | n/a | 0.56 [0.07, 0.96] | n/a | 0.57 [0.08, 0.96] | 0.53 [0.06, 0.95] | 0.58 [0.03, 0.99] (right-tilted, mild endpoint pile-up) | **α_t flat ~0.59 (sd 0.002 across sample)** |
| External-block parameters | n/a | tp ≈ 0.9%, σ_tp ≈ 0.06 | k = 0.62 | γ_fi 0.05, γ_tot 0.01, γ_twi −0.01, γ_icp 0.02 | k = 0.62, ρ_z = 0.95 | k = 0.62, ρ_z = 0.95, γ_tot 0.01, γ_twi −0.01, γ_icp 0.02 | k = 0.63, **a_hyper 1.18, b_hyper 1.09** | k = 0.63, σ_a fixed = 0.05 |

### Cross-validation against published Australian estimates

| Source | r* estimate / range (real) |
|---|---|
| McCririck-Rees (RBA Bulletin Sep 2017) | 0.5–1.5% (around 1% in 2017) |
| Ellis (RBA, Oct 2022) | −0.5% to +2% across 9 models |
| IMF Article IV (2025) | ~0.5% real (assuming 2.5% inflation target) |
| **Resolution A posterior** | flat at ~2.4–2.8% throughout — **above most published ranges, no post-GFC decline** |
| **Resolution B posterior** | trough −1.64%, latest 1.48% — **trough below all published ranges; r* tracks bond yield with constant tp** |
| **Resolution C posterior** | trough 0.82%, latest 2.19% — **inside every published range, captures the post-GFC decline + recovery** |
| **Resolution D posterior** | trough 1.77%, latest 1.77% — flat 1.8–2.7% range; sits at the upper end of Ellis 2022 and IMF; closer than A but no post-GFC decline |
| **Resolution E posterior** | trough 0.83%, latest 2.20% — virtually identical to C; soft anchor + AR(1) z reproduces the deterministic blend |
| **Resolution F posterior** | trough 0.63%, latest 2.15% — virtually identical to C; soft anchor + AR(1) z + SOE IS reproduces the deterministic blend |
| **Resolution G posterior** *(default)* | trough 1.18%, latest 2.23% — slightly tighter range than C; data-determined Beta shape with bimodal α posterior; the trough is the highest of any C-family variant because the right α-mode (α ≈ 1, pure trend growth) has more posterior mass and trend growth doesn't drop as much in 2020 as the indexed bond yield does |
| **Resolution H posterior** | trough 0.92%, latest 2.20% — virtually identical to C; α_t is essentially flat at ~0.59 across the whole sample (no era-specific drift), so time-varying α reduces to constant α empirically |
| **Bullock (RBA Governor, May 2026)** | "**4.35% is a bit restrictive, but less restrictive than when first reached 16 months ago, due to shifts in r\***". Implies r* ≈ 1.0–1.4% currently (from 4.35 cash − 2.7 π_exp = 1.65 real cash, slightly above r*) AND r* has risen ~0.3–0.5pp over 16 months. **B is the closest fit on level (1.48%, gap +0.17pp = mildly restrictive)** and on **mechanism** (B's r* ≈ bond-yield-minus-tp would also have risen ~0.5pp on rising bond yields over the same window). C/E/F/G are 0.5–0.8pp higher in level (implying accommodative policy, contradicting Bullock) and only ~0.2pp shifted upward (partial dynamics consistency). A and D are flat in r* (incompatible with Bullock's "r* has risen" framing). See "External cross-validation: Bullock's 2026 statements" below. |

### External cross-validation: Bullock's 2026 statements

In May 2026, after the RBA hiked the cash rate to 4.35%, Governor Bullock characterised the level as "a bit restrictive, but less restrictive than when 4.35% was first reached 16 months ago, due to shifts in the neutral rate". This is structurally informative — it gives us both a *level* and a *dynamics* check.

**Level check.** With cash rate 4.35% and π_exp ≈ 2.7%, real cash ≈ 1.65%. "A bit restrictive" means real cash is slightly above r*, so implied r* ≈ **1.0–1.4%**. Across our resolutions:

| Resolution | r* latest (2025Q4) | Implied r_gap | Bullock-consistent? |
|---|---|---|---|
| A | 2.43% | −0.78pp | strongly accommodative — incompatible |
| **B** | **1.48%** | **+0.17pp** | mildly restrictive — **closest** |
| C | 2.19% | −0.54pp | accommodative — incompatible |
| D | 1.77% | −0.12pp | essentially neutral; close to but not "restrictive" |
| E | 2.20% | −0.55pp | accommodative — incompatible |
| F | 2.15% | −0.50pp | accommodative — incompatible |
| G | 2.20% | −0.55pp | accommodative — incompatible |

**Dynamics check.** Same nominal cash rate (4.35%), less restrictive now than 16 months ago → r* has *risen* by ~0.3–0.5pp over early 2025 → May 2026.

The bond yield (indexed_10y) rose from 2.21% (2025Q1) to 2.40% (2025Q4) — a 0.2pp move. Each resolution transmits that into r* differently:
- **B** (r* ≈ bond − tp): full transmission, r* rises ~0.5pp on the bond yield rise. **Consistent with Bullock.**
- **C/E/F/G** (α-weighted blend): partial transmission, r* rises ~0.2pp at α ≈ 0.5. Partial consistency.
- **A/D** (r* ≈ g): zero transmission, r* flat. **Incompatible with Bullock's "r* has risen".**

**Substantive implication.** Bullock's framing is most consistent with B's level *and* B's mechanism. The α=0 mode in G's bimodal posterior (the "pure bond anchor" mode) has external policymaker validation; the α=1 mode (pure trend growth) does not.

**Caveats.**
- Bullock's view is operational and reflects RBA's working assessment, not an unbiased r* estimate. The RBA itself uses model averaging and judgment.
- "A bit restrictive" is an interval, not a point — the implied r* range [1.0, 1.4] has 0.4pp of slack.
- The bond market and Bullock both reflect *forward-looking, market-informed* views — there's an element of circularity in B/Bullock agreement (they're both reading the same signal).

**The deeper caveat: Bullock's view is a model preference, not external evidence.**

The RBA's r* methodology — like most central banks' — almost certainly weights the bond market heavily. That's standard central-bank practice: bonds are forward-looking, market-implied, and operationally relevant. So when Bullock says "r* has shifted up and 4.35% is a bit restrictive", she is expressing a view consistent with **placing high weight on the bond market signal**. In our framework, that maps to α near 0 — a preference for the bond anchor over the trend-growth anchor.

**This deflates the apparent confirmation:** B's r* (1.48%) is close to Bullock's implied r* not because the data has been independently validated, but because B reads the bond market and the RBA reads the bond market. They are two methods landing in the same neighbourhood *because they both lean on the same signal*. Not independent confirmation — partial circularity.

The published Australian estimates we cited (McCririck-Rees, Ellis 2022 model-averaging, IMF Article IV) almost all lean on bond-market information in some way. The "external agreement with B's level" is therefore weaker than it looked when we first compiled the table.

**What we can honestly conclude:**
- Bullock's view is one position on the same A-to-B preference spectrum we've been mapping all along, not external truth.
- Her position (α ≈ 0) is a *defensible* preference — bond markets do carry forward-looking information — but it isn't the *only* defensible preference. Trend-growth-anchored views (α ≈ 1) are also defensible (they emphasise structural fundamentals over potentially noisy market signals).
- The model's data alone does not prefer one anchor over the other (the σ_z, α-prior, hierarchical-Beta, and time-varying-α tests all point the same way). The choice between α-anchors is an analyst-level prior choice that no amount of additional data within this framework will resolve.
- **What "validation" against Bullock really tells us is that *if* the analyst chooses to weight bonds heavily, B's level is the closest fit. That's a useful observation about *which prior maps to RBA practice* — but it's not evidence about r*.**

**The honest update to the across-resolution range:** the cross-spec spread of r* latest is ~1.0pp (1.48% B to 2.43% A). The choice of "centre" within that range depends on prior preference about the bond-anchor weight, *not* on data:
- Bond-leaning prior (α near 0): centre at ~1.5–1.8% (B / D)
- Neutral or g-leaning prior (α ≥ 0.5): centre at ~2.0–2.4% (C / E / F / G / H / A)

Both are defensible on the available evidence; neither has external validation in a non-circular sense. The C/G/H family at ~2.20% is at the high-prior-on-g end of the spectrum; B at 1.48% is at the high-prior-on-bonds end.

### What will resolve the cross-resolution disagreement (epilogue)

The data inside our 158-quarter window (1986Q3 → 2025Q4) is genuinely uninformative about which α-anchor is "right". We've demonstrated this three ways: cross-resolution (A through H all return their structural assumption), σ_z prior sweep (CI scales linearly with the prior), and time-varying α (α_t lands flat when allowed to drift). No further model variant will resolve it from this dataset.

**What will resolve it is more data — specifically, the inflation outcome over the next 12–24 months conditional on the cash rate path.**

Bullock's view (May 2026, cash at 4.35%) is "a bit restrictive". This is a testable structural claim:

| Inflation outcome | What it implies | Which resolution it vindicates |
|---|---|---|
| Inflation falls to target within ~12 months | Real cash 1.65% was indeed above r* → r* near 1.0–1.4% | **B** (1.48%) and Bullock |
| Inflation stays elevated 18+ months despite 4.35% cash | Real cash 1.65% was not above r* → r* higher than 1.4% | **C / G / H** (~2.20%) — bond market and policy will revise upward |
| Inflation falls but only after further hikes | r* probably between 1.5–2.0% — middle of range | D (1.77%), or somewhere on the spectrum |

**Importantly, the bond market will move with the inflation outcome.** If inflation persists and the RBA is forced to keep hiking, indexed_10y rises and B's r* (which tracks the bond yield) rises toward C/G/H's level — not because B was wrong, but because the bond market is incorporating new information about the structural neutral rate. The disagreement between bond-anchor and trend-growth-anchor specs partially closes by virtue of bonds re-pricing. Conversely, a clean disinflation pulls bond yields down, B's r* falls further, and the apparent gap between B and C/G/H widens — but with the higher resolutions now looking "wrong" against the policy reality.

**The model is a snapshot, not a verdict.** Recommended forward-test schedule:

- **Q4 2026** (~6 months out): re-estimate after roughly two more quarters of cash at the current peak. If the bond yield has continued rising on persistent inflation, every resolution shifts upward and we get a first read on whether the cross-resolution range narrows or widens.
- **Q2 2027** (~12 months out): meaningful test horizon. Whether 4.35% was a sufficient peak should be visible in trimmed mean inflation by then.
- **Q4 2027 / 2028** (~18+ months out): definitive read, absent a major exogenous shock.

The honest framing is that **this resolution will come from the macroeconomy, not the model** — the data we've assembled here can characterise the *space of plausible answers* but cannot pick between them. Forward inflation outcomes are the missing identifying variation. The Bayesian framework's value going forward is letting us *update incrementally* with each new quarter's data; the value already delivered is showing the *shape* of what we don't know.

This is also the natural rebuttal to the deeper Buncic-Pagan-Robinson critique: the unidentification we've documented is a property of *this* sample, not of the macroeconomy. As real disinflation outcomes arrive, the joint posterior over α / σ_z / hyperparameters will shift in ways that the current sample's likelihood cannot anticipate. The model will get more informative as the policy experiment plays out.

### Footnote: what this modeling exercise actually delivered

A useful reframing: the project did not produce *the* r* estimate for Australia. It produced something more methodologically interesting — **a Bayesian diagnostic for locating any analyst's r\* view on a single axis: how much weight is placed on the bond market vs. trend growth as the structural signal for r\***.

Every credible Australian r* estimate — ours, Bullock's, McCririck-Rees, Ellis 2022 model averaging, IMF Article IV — sits somewhere on this axis:

- α near 0 (B / Bullock / market-leaning models): bond yield carries the structural signal
- α ≈ 0.5 (C, E, F): equal weight blend
- α near 1 (A / canonical HLW for SOEs): trend growth carries the structural signal

The data inside this 158-quarter sample does not distinguish between these. Three independent tests confirm this (cross-resolution, σ_z prior sweep, time-varying α). Whatever the data says, it doesn't say which weighting is right.

**Applying that diagnostic to the RBA's own framework:** Bullock's statement that "r* has shifted upward and 4.35% is a bit restrictive" is only consistent with a low-α weighting (heavy on bonds). Other defensible weightings — the C/G/H middle of the spectrum at α ~0.6 — would imply policy is currently *accommodative*, which contradicts Bullock's assessment. So the RBA's working framework appears to be substantially a "read r* off the bond market" approach, even if not described in those terms.

This isn't a critique of the RBA — bond-leaning weights are methodologically defensible, and central banks have used yield curves for r* identification for decades (Bauer-Rudebusch, term-structure DSGEs). It's an *observation* about the implicit modeling choice in the institutional view. The same diagnostic applied to the IMF (Article IV r* estimates) and Ellis 2022 (model averaging across nine specs) would reveal those frameworks' implicit α weightings too.

**Why this is a more useful deliverable than a point estimate:**

1. **Policy debate becomes cleaner.** A disagreement between "r* is 2.2%, policy is accommodative" and "r* is 1.3%, policy is restrictive" isn't a disagreement about r*; it's a disagreement about which signal to weight. Framing it this way is more honest than dueling point estimates that hide their structural assumptions.

2. **Cross-central-bank differences become legible.** The Fed, ECB, RBA, BoE all publish r* ranges and they differ. Differences are partly differences in α-weighting, not in data quality. Some central banks lean more heavily on yield curves (Fed, RBA) than others (which weight demographic and productivity estimates more heavily).

3. **It localises Buncic-Pagan-Robinson 2023 in practical terms.** Their theoretical critique was that latent r* isn't identified by macro data alone. Our practical demonstration shows: not just "isn't identified", but "different defensible weightings give different answers, and policymaker statements pick a particular weighting". The lack of identification isn't academic — it's the *substance* of the policy disagreement, and naming it that way is more productive than litigating point estimates.

The deliverable, properly understood, is a *meta-tool*: a way to read any r* claim — model output, central bank statement, IMF report — as a position on the bond-vs-growth spectrum, with the model providing the framework for understanding what each position implies and where it sits among defensible alternatives.

### Analytical assessment

The eight resolutions reduce to three qualitatively different stories from the same data — growth-anchored (A, D), yield-anchored (B), and blend (C, E, F, G, H) — and each is informative about a different aspect of the identification problem.

**Resolution A confirms the literature's finding that canonical HLW cannot identify z for Australia.** σ_z posterior median is 0.063 with HDI [0, 0.18], piling up at the lower bound. z span is 0.11 pp over 33 years. r* tracks g with virtually no independent variation. This is exactly what Buncic (2021) shows for Euro Area / UK / Canada: the correct λ_z is essentially zero, and any apparent r* movement in canonical HLW is MUE artefact. The pathology shows up in the sampling diagnostics — latent r_star, trend_growth, and z_star all have R-hat > 1.3, ESS < 25, and σ_z is effectively unsampled (R-hat 1.48, ESS 9.5). **The poor sampling is itself diagnostic of the identification failure.** A misses the post-GFC decline narrative entirely; r* sits at ~3% throughout, which means policy has been "loose" for fifteen years according to A — economically implausible.

**Resolution B goes to the other extreme.** Once the indexed bond is added as a direct observation of r*, σ_z jumps from 0.063 to 0.294 (5×) — z becomes very identifiable, but only because the bond yield is dictating r*. The IS curve and Phillips curve don't materially constrain z; the bond observation does. Result: r* tracks the bond yield minus a constant term premium of 0.93pp, swinging from +4.6% in 1995 to **−0.72% in 2020** as indexed yields collapsed during the global low-rate period. Sampling is even worse than A — **11,458 divergences**, R-hat for r_star at 1.302 — because B has the worst of both worlds: a free RW z that wants to wander, plus a strong bond-observation pull that pins r* tightly to a moving target. The latent state has to satisfy both, and NUTS can't navigate the resulting geometry. **B's r* level is implausibly low at the trough**, below every published Australian estimate, and the constant-term-premium assumption pushes post-GFC term-premium compression into r* itself.

**Resolution C produces an in-range path by construction.** Instead of adding the bond as an observation that competes with the latent z, C builds the bond into the *definition* of r* via the α blend weight. There is no independent z latent state — r* is a deterministic function of g, indexed_10y, and two scalar parameters (α, k) plus an i.i.d. ε. The blend formally lets the data choose how much to weight each anchor (α posterior 0.56 with wide HDI [0.07, 0.96] under the Uniform prior; the data weakly tilts toward higher α). The structural anchor (g) keeps r* from collapsing to bond-yield-minus-constant; the market anchor keeps r* from sticking at trend growth. Result on the gap-filled sample: **193 divergences, r_star R-hat 1.000, ESS 6,133**, and an r* path with span 3.31pp — 4.13% peak declining to 0.82% trough in 2020Q4 and recovering to 2.19% by 2025Q4. The path "lands inside every published Australian range" because the analyst-chosen α and k together place the median where the published estimates are; this is a feature of how the blend is parameterised, not an independent verdict from the data.

**The IS curve `a_r` and `σ_IS` are remarkably stable across resolutions.** A: a_r = −0.045, σ_IS = 0.72. B: a_r = −0.047, σ_IS = 0.70. C: a_r = −0.060, σ_IS = 0.72. D: a_r = −0.034, σ_IS = 0.70. The IS rate channel is genuinely weak for Australian data; **no choice of r* identification or IS-curve specification — open or closed economy — makes it strong**. This is a substantive finding about the Australian IS curve, not a model artefact.

**Resolution D's role: a tighter empirical refutation than A.** A could be dismissed as "weak result because the closed-economy IS curve is mis-specified for an SOE". D answers that critique directly: with an open-economy IS curve including the three principal SOE channels (income via ToT, exchange rate via TWI, foreign demand via ICP), the result is the same — z is dead, r* tracks g, σ_IS unchanged. The Australian rate-output channel is genuinely weak; the SOE block doesn't fix it. So D upgrades the canonical-HLW-fails-for-AU finding from "weak rate channel in a closed-economy model" to "weak rate channel even with the open-economy regressors the SOE literature recommends".

**g paths cluster around the same secular path.** A: 3.16% → 2.80%. B: 3.08% → 2.83%. C: 3.13% → 2.37%. D: similar shape to C (the soft anchor is wired in for D too). The structural component is reasonably robust across approaches.

### Conclusion

The eight-way comparison is consistent — and what is consistent across it matters more than the small differences between specs:

- **A** (canonical, closed economy) shows what the data alone says: r* cannot be separately identified from g; canonical HLW fails for Australia. Matches Buncic, McCririck-Rees, and Ellis.
- **B** (canonical + bond observation) shows what happens when you add a strong external identifier to a model that otherwise can't pin r*: the external identifier *becomes* r*. Sampling is the worst of the eight, but the level it returns is also the closest to Bullock's stated working view.
- **C** (blend) is the principled middle ground in construction terms: bond information is included by design rather than as an observation that fights with the latent z. The blend weight α is the only thing the data has to identify, and even there the posterior is wide. Whether C's level is *the right level* is the bond-vs-growth question that the data inside the model cannot answer.
- **D** (canonical + open-economy IS curve) tests whether A's failure was mis-specification fixable by adding SOE regressors. Empirically refuted — σ_IS and a_r are unchanged, z still dead, r* still ≈ g.
- **E** (blend + AR(1) z) tests whether C's deterministic identity over-constrains r*. With z given proper room, the IS curve adds essentially nothing above the blend.
- **F** (E + open-economy IS curve) combines E and D. Same result: r* matches C, z dead, γ coefficients dead.
- **G** (blend + hierarchical Beta on α) lets the data pick the prior shape on α. Under a neutral hyperprior the chosen shape is a mildly bell-shaped Beta — there is no strong data-implied bimodality, and r* is within Monte Carlo error of C.
- **H** (blend + time-varying α_t) lets α drift period-by-period under a permissive logit-RW prior. α_t lands almost flat (sd 0.002 across 158 quarters), and r* matches C/G to within MC error. The data has no era-specific preference for one anchor over the other — including no support for a recent shift toward the bond anchor.

Across **all eight** specifications, σ_IS sits at 0.70 ± 0.02 and a_r at −0.04 ± 0.01. Wherever the structural identity admits external observables (g, indexed_10y, blend), r* tracks those observables. Wherever it doesn't (canonical A, D), r* tracks g. **No specification produces a usefully *time-varying* r* that's separately identified by the IS curve.**

The blend (C / G / H) gives credible numbers but does not actually identify r* from the IS curve — it returns a weighted average of the two anchors at whatever weight its prior supports. A and D return the growth-anchored answer; B returns the yield-anchored answer. The data inside this sample tilts only weakly between them, and the direction of the tilt depends on which prior on α one chooses.

A, B, D, E, F are kept as contrastive baselines. C, G, and H are interchangeable presentational forms of the same blend story.

## The headline finding: r\* is the structural assumption, not the estimate

What the cross-resolution evidence above adds up to is a finding that's stronger than any per-resolution claim and is the framing the rest of this document should be read in light of.

**No specification we have run identifies r\* from the IS curve.** Every spec returns a tightly-controlled rephrasing of the structural assumption it imposes:

- A imposes `r* = g + z` with no anchor on z → returns r* = g (z dies).
- B imposes `r* = g + z` plus `indexed_10y = r* + tp` → returns r* = indexed_10y − constant tp (the bond observation dictates r*).
- C imposes `r* = α·g + (1−α)·(indexed − k) + ε`, ε small → returns r* = blend (the deterministic identity is the answer).
- D imposes the same structure as A but with an open-economy IS curve → returns r* = g (the IS curve still doesn't identify z, even with the SOE block).
- E imposes `r* = blend + AR(1) z` with σ_z fixed → returns r* = blend (z stays at 0; the IS curve has no information to put into z).
- F imposes E's identity plus the open-economy IS curve → returns r* = blend (combination doesn't help either).
- G imposes C's identity with a hierarchical Beta(a, b) on α → returns r* = blend; the hyperparameters' posterior tracks the hyperprior more than it tracks the data.
- H imposes C's identity with a time-varying α_t (logit-RW) → returns r* = blend with α_t pulled to a near-constant ~0.59 (drift 0.01pp end-to-end), even though σ_a = 0.05 would have permitted ~1pp/quarter drift on the logit scale.

In every case, r* is *what we said it was*, plus a Monte Carlo whisker. The Bayesian update on r* is in the third decimal place. The IS curve's contribution to r* identification is essentially nil; the data, conditional on the structural assumption, has nothing material to add about r*.

### Why this happens

The IS curve has a measurable but weak rate channel: posterior median `a_r ≈ −0.04` against `σ_IS ≈ 0.70` and an in-sample `r_gap` standard deviation of ~2 pp. The signal-to-noise ratio for r* identification through the IS curve is therefore roughly:

```
|a_r| × sd(r_gap) / σ_IS  ≈  0.04 × 2 / 0.70  ≈  0.11
```

That's a 1-in-10 effect. Across 150-odd quarters of data, the IS curve cannot distinguish r* paths that imply different r_gap dynamics in any meaningful way. Whatever structural assumption we impose on the latent r* state, the IS-curve likelihood does not have enough power to overrule it.

This is the **Buncic-Pagan-Robinson 2023** finding ("On Constructing a Country-Specific Time Series for the Natural Rate of Interest") in concrete form for Australian data: when the number of latent shocks meets or exceeds the number of independent observables that constrain them, the latent stars are not point-identified; the posterior is essentially the prior projected through the structural model, with a thin layer of likelihood on top. We've now demonstrated this empirically across eight specifications — varying the r* identity (canonical z, blend, blend + AR(1) z), the α prior (fixed Beta, hierarchical Beta, time-varying α_t), and the IS-curve composition (closed, open-economy SOE block) — and observed the predicted pattern in every one.

It is also consistent with the broader SOE literature:
- Buncic (2021) for Euro Area / UK / Canada: λ_z is essentially zero in canonical HLW, MUE manufactures the apparent r* movement.
- McCririck-Rees (RBA Bulletin Sep 2017) for Australia: model averaging is the recommended response.
- Ellis (RBA speech Oct 2022) for Australia: report a range across nine models, span −0.5% to +2%.
- IMF Article IV (2025) for Australia: a range, not a point.

### Empirical confirmation: σ_z prior-sensitivity sweep on Resolution E

The Buncic-Pagan-Robinson prediction has a sharp empirical implication: when the data is uninformative about a latent state, the posterior CI is the prior projected through the model — *posterior uncertainty scales with the prior, not with what the data could justify*. Tested by holding Resolution E fixed (seed = 42, all other priors and observations unchanged) and varying σ_z (the AR(1) innovation scale on z, the deviation of r* from the blend) over five values.

*Numbers in the σ_z and α-prior sensitivity tables that follow were produced on the original 153-period sample before the indexed yield gap was filled, and (for the hierarchical Beta) under the original `HalfNormal(1)` hyperprior on (a, b) before it was replaced by `Uniform(0.25, 2)`. The qualitative findings — CI-scales-with-prior, σ_z = 1.00 overfitting, hyperprior-sensitive endpoint stacking — are unchanged on the 158-period filled sample. Absolute r* values shift by ~0.05 pp.*

| σ_z | divergences | a_r | σ_IS | α | r* latest (median) | r* trough (median) | **r* 90% CI @latest** | \|z\| mean |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 813 | −0.039 | 0.70 | 0.52 | 2.19% | 0.68 (2020Q4) | **1.27 pp** | 0.008 pp |
| 0.15 *(default)* | 302 | −0.038 | 0.70 | 0.53 | 2.21% | 0.74 (2020Q4) | **2.07 pp** | 0.040 pp |
| 0.30 | 296 | −0.040 | 0.70 | 0.53 | 2.27% | 0.80 (2020Q4) | **3.51 pp** | 0.149 pp |
| 0.50 | 270 | −0.044 | 0.69 | 0.53 | 2.39% | 0.44 (2015Q1) | **5.35 pp** | 0.400 pp |
| 1.00 | **1,583** | **−0.333** | **0.25** | 0.45 | **3.12%** | **−3.78** (2019Q4) | 5.59 pp | **2.218 pp** |

Two findings, both reinforcing the headline:

**1. Posterior median r\* is approximately invariant across the realistic σ_z range; posterior CI scales linearly with σ_z.**

Across σ_z ∈ [0.05, 0.50] — a 10× change covering the full range of plausible AR(1)-innovation priors in the HLW literature — the posterior median r* moves by 0.20 pp at the latest observation (2.19% → 2.39%). a_r, σ_IS, α, k are stable to within Monte Carlo error. The blend pins the median.

But the **r* 90% credible interval at the latest observation widens from 1.27 pp to 5.35 pp** — a 4× scaling that tracks σ_z almost mechanically. The data has nothing to say about how uncertain r* should be at a given t; the prior on σ_z is doing all the work. **The fan-chart band on any single resolution is essentially a re-display of the σ_z prior choice projected through the model**. This is precisely the Buncic-Pagan-Robinson prediction made visible on Australian data.

**2. At σ_z = 1.00 the model collapses into an overfitting failure mode.**

When z is given enough freedom to absorb almost any variation (σ_z = 1.00 pp/quarter is implausibly loose), z becomes the explanation for the IS-curve residuals: a_r jumps 8× to −0.333, σ_IS drops from 0.70 to 0.25, |z| mean rises to 2.2 pp with z's range running from −4.77 pp to +6.31 pp. The IS curve "identifies" only in the sense that the latent z is now flexible enough to fit anything, including noise. The implied r* path drops to −3.78% in 2019Q4 — outside every published Australian estimate by a wide margin and not economically interpretable. Divergences spike from ~300 to 1,583.

This regime is informative as a *boundary marker*, not as an alternative model: it's the only setting where the IS curve's a_r firms up, and the price is r* fitting noise. It's the Resolution-B trap (bond yield dictates r*) reproduced inside a single spec by giving z too much room.

**3. The default σ_z = 0.15 is conservative.**

At σ_z = 0.15 — the value chosen for Resolution E based on the canonical HLW λ_z neighbourhood — the posterior r* CI at the latest observation is 2.07 pp. The cross-resolution span (B − A) at the latest observation is ~1.6 pp (1.53% to 3.09%). So the within-model CI under the default σ_z is **comparable to the cross-resolution range** — but most of that CI is prior-driven, not data-driven, as confirmed by the linear scaling above. The genuine data-informed component is small enough that **σ_z = 0.05 (which forces near-zero CI) and σ_z = 0.50 (which gives a 5pp CI) produce essentially the same posterior median** — i.e., the level of r* is robust, but the *uncertainty around it* is whatever we said it should be.

### Empirical confirmation: α-prior sensitivity on Resolution C

Parallel test on the α (blend-weight) parameter. Held Resolution C fixed and ran with two prior shapes: Beta(2, 2) (the original default — symmetric central mass, tapering to zero at endpoints) and Beta(1, 1) (Uniform — flat; the new default).

| | Beta(2, 2) prior | Beta(1, 1) = Uniform prior |
|---|---|---|
| Prior 90% range | [0.10, 0.90] | [0.05, 0.95] |
| Posterior median | 0.520 | 0.559 |
| Posterior 90% HDI | [0.16, 0.81] (range 0.65) | [0.07, 0.91] (range 0.84) |
| Posterior HDI / Prior 90% range | 81% | 93% |
| r* trough | 0.69 (2020Q4) | 0.85 (2020Q4) |
| r* latest | 2.19 | 2.23 |
| All other parameters | within Monte Carlo error | within Monte Carlo error |

**Two findings, both consistent with σ_z sweep but with an important nuance:**

1. **The 90% credible interval scales with the prior** (the Buncic-Pagan-Robinson pattern again). Under Beta(2, 2), the posterior HDI is 81% of the prior's 90% range; under Uniform, 93%. Whatever shape we put in, the posterior occupies almost all of it. The data is doing very little tightening on α.

2. **But the posterior shape under Uniform reveals a weak data signal that Beta(2, 2) was masking.** The Uniform prior has density 1.0 everywhere, so any departure from flat in the posterior is purely the likelihood. The Uniform-prior posterior shows monotonically rising density from ~0.67 at α = 0 to ~1.18 at α = 1 — a 1.75× ratio. The data weakly prefers higher α (more weight on **trend growth** as the structural anchor, less on the indexed bond yield anchor). Beta(2, 2)'s symmetric pull toward 0.5 was washing this signal out and producing a near-symmetric posterior centred at 0.52 instead of revealing the 0.56 median that Uniform yields.

**Why this matters.** It refines — but does not overturn — the headline finding. The decomposition is:
- *Within-model CI on α*: prior-driven. Posterior 90% range tracks prior 90% range. Buncic confirmed.
- *Posterior central tendency on α*: weakly data-driven. There is a real likelihood signal favouring α > 0.5, just with too little force to dominate a tight prior. Beta(2, 2) was suppressing this signal; Uniform reveals it.

**Practical consequence for the default.** Beta(1, 1) is now the default α prior across Resolutions C, E, and F. Previously Beta(2, 2). The change is methodologically more honest (uninformative on a parameter the data weakly identifies) and produces a slightly different r* path: median r* trough rises from 0.69 → 0.85, latest from 2.19 → 2.23, and the r* path tilts marginally more toward trend-growth's secular shape (since the higher posterior median on α weights g more in the blend). **All other model parameters are within Monte Carlo error**, so this is a clean prior-shape change rather than a substantive respec.

The cross-resolution invariance findings already documented continue to hold: r* still ≈ blend, the IS curve still doesn't independently identify r*, and the cross-spec range remains the honest uncertainty.

### Empirical confirmation: hierarchical Beta(a, b) on α (Resolution G), and the endpoint-stacking insight

A sharper test still — and now wired as the default Resolution G: rather than choose any Beta shape ourselves, let the data pick it. Replace the fixed Beta prior with a hierarchical one:

```
a_hyper, b_hyper  ~  Uniform(0.25, 2)         [flat over a range that covers
                                              U-shapes, Uniform, and bell-shapes]
α                 ~  Beta(a_hyper, b_hyper)
```

If the data has any preference for a particular Beta shape on α, the hyperparameters' posterior should reveal it. If the data is uninformative, the hyperparameters drift around the prior.

**Result: the data mildly prefers a slightly bell-shaped Beta** (a, b posteriors land just above 1), but with very wide HDIs:

| Quantity | Posterior median | 90% HDI |
|---|---|---|
| `a_hyper` | 1.18 | [0.35, 1.92] |
| `b_hyper` | 1.09 | [0.33, 1.91] |
| α | 0.58 | [0.03, 0.99] |

α posterior quantiles 5/25/50/75/95: 0.03 / 0.30 / 0.58 / 0.83 / 0.99. The posterior on α is **right-tilted with mild endpoint pile-up** — density ≈ 1.7 at α near 0, ≈ 1.0 in the middle, ≈ 3.0 at α near 1.

#### The endpoint-stacking insight

The first hierarchical attempt used `a, b ~ HalfNormal(1)` — which has 58% of its prior mass below 1 and was thereby a soft a priori preference for U-shaped Betas. Under that hyperprior the posterior on α was strongly bimodal (densities ~4.4 / 0.6 / 8.4 at the left endpoint / middle / right endpoint). Switching to the neutral `Uniform(0.25, 2)` flattened the bimodality substantially: hyperparameters moved from 0.74/0.63 to 1.18/1.09 (now just above 1), and the α posterior went from sharply bimodal to right-tilted with mild endpoint pile-up.

The pattern across all the α-prior experiments tells a clean story:

| α prior | Posterior shape on α |
|---|---|
| Beta(2, 2) — bell, sub-1 forbidden | unimodal, no endpoint mass |
| Beta(1, 1) Uniform — flat | unimodal sloped (gentle tilt to high α), no endpoint mass |
| Hierarchical, HalfNormal(1) on (a, b) | strongly bimodal, modes at 0 and 1 |
| Hierarchical, Uniform(0.25, 2) on (a, b) | mildly right-tilted, soft endpoint pile-up |

**Endpoint stacking only appears when the (hyper)prior allows Beta shape parameters to drop below 1**, putting the Beta family in U-shape territory. The data's signal on α is too weak to either confirm or rule out U-shapes — and U-shapes by construction place mass at the endpoints. So the chain happily drifts into sub-1 (a, b) regions and the marginal posterior on α picks up the endpoint mass. **The stacking is the constraint structure × diffuse signal interaction, not a data preference for extreme α.**

#### The "smear" implication for constant α — and what time-varying α_t (Resolution H) found

A second corollary: **with constant α and a weak likelihood, the r\* posterior at any single time t is necessarily wide** — it's a mixture across all the (constant) α values the chain visited. Each draw applies its α to the whole sample; with α ranging 0–1 and the two anchors diverging by ~3pp post-2018, the implied r* span at any single t is correspondingly wide. This is the "smear" we see in G's decomposition chart. It does not represent uncertainty about r* at time t — it represents sample-wide uncertainty about a constant α projected period-by-period.

The principled candidate fix is **time-varying α_t** (a logit-RW on α, controlled smoothness via fixed σ_α): each period would in principle have its own posterior on α even if the sample-wide story is "α was high in the 1990s and low post-2010". Earlier regime-switching attempts (iterations 7 and 13) failed because the data couldn't pick a date; a continuous RW relaxes that constraint, allowing smooth gradients rather than forcing breakpoints.

**Resolution H implements this and reports a flat answer.** With σ_a = 0.05 on the logit scale (permissive — allows ~1pp/quarter drift at α = 0.5), α_t lands almost perfectly horizontal: standard deviation 0.002 across 158 quarters, total end-to-end drift 0.01pp. r* matches C and G to within Monte Carlo error. The IS curve has no era-specific information about which anchor matters more in different periods — the data does not want time-variation in α, even when given ample room to express it. Bullock's "r* has shifted upward" framing — which would imply α_t drifting toward 0 in recent years as the bond anchor takes over — is not supported inside this model. So the smear in constant-α specs reflects analyst uncertainty about a single sample-wide α, not lost period-specific signal that H could recover.

#### What this third independent test confirms — and what the fourth (H) added

- α is essentially unidentified by the data; the posterior shape always tracks whatever prior structure we put in.
- The hyperparameter posteriors are themselves prior-sensitive: HalfNormal(1) → sub-1 medians (U-shape preferred); Uniform(0.25, 2) → just-above-1 medians (mild bell preferred).
- The r* level is robust across all these prior choices (cross-resolution invariance documented in the headline).
- The within-period r* posterior CI under constant α is wide because α is constant — a structural choice, not a data finding.

This is a clean Bayesian-machinery confirmation of the headline finding: even when we hand the prior over to the data, the data hands it back with effectively no information added — and the *shape* of the bimodality first observed depends on the analyst's hyperprior choice, not on the data.

**Resolution H added a fourth independent confirmation from a different angle.** Rather than vary the prior shape on a single α, H freed α to drift over time — and the data pulled α_t toward a constant. Combined with the σ_z sweep (CI-scales-with-prior), the α-prior sensitivity (posterior tracks the Uniform), and the hierarchical-Beta result (hyperprior-sensitive endpoint stacking), this closes the loop: the data does not have an opinion on α — not on its level, not on the Beta shape, not on era-specific drift. Whatever any single resolution returns is a recombination of the structural assumption and the prior; the IS curve does not adjudicate.

### Implications for the project

1. **Reporting r\* as a single number with credible bands overstates precision and conceals what the bands actually represent.** The posterior CI within any single resolution is conditional on (a) the structural assumption being right, and (b) the variance-ratio priors being right. Neither is established by the data. The σ_z sweep above shows the within-model CI is *almost entirely prior-driven* — the data has nothing to say about how uncertain r* should be. The honest "uncertainty band" is the spread *across* specifications: r* latest ranges from 1.48% (B) to 2.43% (A), with C/E/F/G clustered around 2.15–2.23% and D at 1.77%. That ~1.0 pp spread (excluding the implausible B) is the actual uncertainty, and almost all of it is structural-assumption uncertainty rather than within-model statistical uncertainty.

2. **The blend resolutions (C / G / H) produce credible numbers but the "answer" is the externally-observed blend.** What C/G/H produce is what the deterministic blend implies, given α ≈ 0.55–0.59 and k ≈ 0.62 — economically interpretable and inside published Australian ranges. But this is not "an estimate of r* from the IS curve and Phillips curve". It is an explicit weighted average of trend growth and the inflation-linked bond yield (less a constant term premium), with the blend's components carrying nearly all the information. The blend is one defensible answer among several; whether it is the *best* answer depends on prior commitments about the bond-vs-growth weight that the data inside the model cannot resolve (see "Recommended use" below).

3. **For NAIRU integration, the median r\* series from a blend resolution (C, G, or H — all near-identical at the median) is a defensible input,** but treating it as an estimated quantity with its narrow within-model CI is the wrong model of its uncertainty. The downstream NAIRU model should either (a) use the median and acknowledge in interpretation that r* is mostly an externally-imposed input, or (b) propagate cross-specification uncertainty by running NAIRU with multiple r* series (B and a blend resolution at minimum, ideally also A for the growth-anchored extreme).

4. **Future work should focus on richer external anchors rather than richer IS-curve identification.** The IS curve's weakness is a property of the data; no respecification of the IS curve in the HLW framework changes that. The leverage is in better external observables — long-run survey expectations, term-structure-implied real rates, convenience yields, foreign r* with credible country wedge — i.e., the specifications listed in "What was *not* tried (potential future work)".

### What would change this conclusion

The conclusion is data-driven, not assumption-driven, and is replicable. It would change if:
- Australian data developed a stronger rate-output channel (e.g., post-2025 monetary tightening produces a much larger output response than 1986–2025 history shows).
- A genuinely new identifying observable is added (term-structure block, convenience-yield series, long-run survey expectations) that constrains r* independently of the IS curve.
- A larger-scale model (DSGE with cross-equation restrictions) replaces the semi-structural HLW form, providing identifying restrictions outside the IS-curve channel.

Until then, the empirical finding stands: **r\* in this model is the structural assumption, dressed up in the language of estimation.**

## Recommended use

### There is no best model — there are three approaches

The eight resolutions A–H reduce to three substantively different approaches to r*, each defensible under different assumptions about what signal carries the structural neutral rate:

| Approach | Resolutions | r* latest | What it assumes |
|---|---|---|---|
| **Primarily growth** | A, D | 2.43%, 1.77% | Trend growth `g` is the structural anchor for r*. Bond markets are noisy and reflect short-run financial conditions rather than the neutral rate. Canonical HLW for closed-economy SOEs (D adds an open-economy IS curve, but z stays dead — the answer is still r* ≈ g). |
| **Primarily yields** | B | 1.48% | The bond market is the structural anchor for r*. Forward-looking, market-implied, operationally relevant. The bond yield (less a constant term premium) IS r*. This is the implicit framework most central banks use, including the RBA per Bullock's May 2026 statements. |
| **Some blend** | C, E, F, G, H | 2.15–2.23% | Both anchors carry information about r*, weighted by α. The five blend resolutions differ in *how* α is parameterised (fixed Beta, hierarchical, AR(1) z, time-varying), but all produce nearly the same r* level (~2.20% in 2025Q4) because the data does not strongly distinguish between α weightings within the blend family. |

**Which approach is "right" depends on circumstances and prior commitments, not on the data inside the model:**

- **The post-GFC era (2010–2024) was effectively a test of which approach is empirically better.** Real rates were near zero or negative for over a decade; trend growth stayed near 2.7%. A growth-anchored r* (~2.7%) would have predicted that low rates were deeply stimulatory and inflation should have surged. Inflation instead drifted *below* target. A yield-anchored r* (~0% or slightly negative) was consistent with the muted inflation outcome. **By the 15-year track record, the yield-anchored view was more accurate.** This rehabilitates B's "implausibly" negative 2020Q4 trough — it may simply be the right reading.

- **The current cycle is consistent with the yield-anchored view too.** Bullock characterising 4.35% (real cash ~1.65%) as "a bit restrictive" is consistent with r* ~1.0–1.4%, which matches B's level. The ~2.20% r* from C/G/H would imply policy is currently *accommodative*, contradicting the observed disinflation.

- **However: anchors are converging.** The bond-yield/g gap was 3.4pp at 2020Q4, is now 0.25pp at 2025Q4. As convergence continues, the choice between approaches becomes less consequential — every resolution lands in a similar place when the anchors agree. **The next few years may not be the period where the choice matters most.** The previous 15 years were.

- **Cross-country observation (caveat-laden):** the NY Fed HLW estimates for the US, Canada, and Euro Area (visible in the AU vs G3 comparison chart) all sit closer to bond-implied real rates than to local trend growth in recent decades. This is consistent with central bank methodologies leaning toward yields. Partially circular (every framework that uses an IS curve weights bond information similarly), but suggestive of a cross-country empirical regularity.

### Headline framing for downstream use

- **For analysis explicitly aligned with the RBA's framework**: use **B** (or treat C/G's α=0 mode as the relevant reading). This matches the institution's working view of r*.
- **For analysis using a structural / supply-side framework**: use **A** (canonical) or **D** (open-economy SOE-augmented). Acknowledge these will say policy has been chronically accommodative for 15 years and inconsistently with observed inflation.
- **For analysis that wants to stay agnostic on the bond-vs-growth weight**: use **G** (default) or **C**. The blend produces a level (~2.20%) that's a compromise — defensibly in the middle of the cross-resolution range, structurally well-behaved, but methodologically *not* the answer in any deep sense; it is just one position on the bond-vs-growth axis.
- **For NAIRU integration**: the median r* series from C (or G — they're nearly identical at the median) is a defensible input. Treat it as one possible r* path, not the truth. The downstream NAIRU model should be aware that r* is mostly an externally-imposed input chosen by the analyst's α weighting, not an independently estimated structural quantity.

The honest summary: **the model gives you a Bayesian framework for placing any α weighting on the bond-vs-growth spectrum and seeing what r* it implies.** It does not tell you which weighting is right. The 15-year empirical track record favours the yields side; the convergence in 2024–2025 makes the choice less consequential going forward; and the framework remains useful as a diagnostic for the next divergence episode.

### Specific notes on individual resolutions (quality of sampling, not of answer)

- **G** *(default in `estimate.py`)*: blend with hierarchical Beta(a, b), a, b ~ Uniform(0.25, 2). Best of the blend family for revealing the α posterior shape; the hierarchical structure lets the data express its (weak) preference about Beta shape without the analyst pre-committing.
- **C**: simpler-prior blend (fixed Beta(1, 1) on α). Useful when downstream code wants a stable r* path without hierarchical hyperparameter machinery. r* path identical to G.
- **A**: textbook canonical HLW. Defensible if the analyst commits to the growth-anchored view a priori. Sampling is challenging (R-hat 1.020, ESS 243 on r_star) — z latent is dead, which is the substantive HLW-on-SOE finding.
- **B**: canonical r* + indexed-bond observation. The yield-anchored answer. Sampling is messy (3,349 divergences, latent state hard for NUTS) but the mechanism — r* tracks the bond yield — is what matches the empirical track record best. Use the median series rather than the dynamics if depending on B.
- **D**: canonical r* + open-economy IS curve. Demonstrates that adding the SOE literature's recommended IS-curve regressors (ToT, TWI, ICP) does not change canonical HLW's r* ≈ g answer. Contrastive baseline; not recommended for substantive use.
- **E, F, H**: contrastive baselines that demonstrate the constant-α blend (C/G) is not artificially restrictive — softening to AR(1) z (E), adding SOE block (F), or letting α vary in time (H) does not change the answer. Useful for showing robustness; not separately recommended over C/G.

**Never recommend:** intermediate identification regimes (canonical HLW with looser priors, latent term premium without the bond observation). They are unidentified and produce thousands of divergences.

### How the blend produces a stable median r*

Resolution C's median r* is stable across specification variants — replacing the fixed Beta with a hierarchical one (G), softening the deterministic identity to a soft anchor + AR(1) z (E), letting α drift period-by-period (H), and adding the open-economy IS regressors (F) all return r* paths within Monte Carlo error of C. That stability is a feature of the construction rather than independent confirmation that the level is right: the blend's degrees of freedom (α, g, k) absorb decomposition ambiguity, with the bond anchor typically sitting below the IS-curve-implied r*, so the model pushes g upward (~2.6% vs a structural expectation of ~2.2%) and k downward to make the convex combination hit a level consistent with the IS curve. Earlier attempts to produce independently credible components — regime-switching alpha, intercept replacing k, intercept alongside k — all confirmed that the IS curve's pull on r* propagates through α into g regardless of parameterisation.

**Implication:** the median r* under any blend variant should be read as a weighted average of the two anchors with the weight chosen mostly by the prior, not as a structural estimate that the IS curve has independently identified. The decomposition chart shows *how the blend constructs r\** from its two anchors, not independent structural estimates of trend growth or the equilibrium real bond yield. If an independent estimate of trend growth is needed, it should come from the potential output equation or an external source.

## NAIRU integration

The original motivation for this model. Once a single canonical run is settled, the path forward is sequential coupling:

1. Run the rstar pipeline at 1993Q1 start
2. Save the posterior median r* series to disk
3. In `src/models/nairu/observations.py`, expose an option to load that series into `obs["det_r_star"]` instead of the Cobb-Douglas r* derived from `compute_r_star()`
4. Pre-1993 r* values can either be backward-extrapolated (hold the 1993Q1 estimate, or simple trend) or fall back to Cobb-Douglas

For the **fan integration** (propagating r* uncertainty through to NAIRU): more invasive — would require a joint model or a Monte Carlo loop over r* samples. Not implemented.

## What was tried that didn't work

A log of approaches that were attempted and discarded, for the benefit of future revisits.

| Approach | Outcome | Why it failed |
|----------|---------|---------------|
| `target_accept = 0.97` to fight divergences | Worse | Hides the geometry, doesn't fix it |
| Non-centring `trend_growth` (manual cumsum) | Catastrophic (5,976 divergences) | y* already uses g[:-1]/4 inside its own cumsum — doubly-cumulated structure breaks NUTS gradients |
| Tighter priors on σ_g and σ_z | Lowered divergences but killed z | z collapses toward zero — model becomes effectively univariate trend extraction |
| Looser λ_z + AR(1) z + indexed-bond observation (Resolution B) | r* span 6 pp but trough is negative; 5,000 divergences; HLW machinery becomes decorative | Constant-term-premium assumption pushes post-GFC compression into r* level rather than tp; bond yield does all identifying work |
| 1980Q1 sample start (matching NAIRU) | Worse identification | Pre-1993 regime contaminates Phillips curve |
| Survey-based PIE_RBAQ for π_exp | Rejected upfront - not tested | Project has its own expectations model |
| Higher target_accept *with* non-centred trend_growth | Did not test, would not have helped | Geometry was the problem, not step size |
| Mid-life fan-integration r* into NAIRU | Not implemented | Sequential coupling (post-process median series into NAIRU `obs["det_r_star"]`) is the realistic path |
| Regime-switching alpha (alpha_pre / alpha_post around 2008Q4) | Sampling much worse (6,575 divergences vs 1,197); regimes not separated (alpha_pre 0.55 vs alpha_post 0.48, difference HDI [-0.44, +0.58]) | The data has too little identifying power on alpha to support time-variation. Both alphas overlap heavily — one constant alpha is the right specification |
| Loosening σ_g to HalfNormal(0.10) without an anchor | g declined nicely (3.26% → 2.73%) but divergences blew up to 5,771; r_star ESS dropped to 20 | σ_g and σ_ystar trade off as competing explanations of y* drift; without external info on g, NUTS can't navigate the funnel |
| HMA(13) of YoY GDP growth as g anchor (free σ_trend_obs) | σ_trend_obs collapsed from prior HalfNormal(1.5) to posterior 0.022 — anchor became hard, not soft. g pulled toward smoothed YoY *including the COVID dip* (g = -1.11% in 2020Q1) | The data over-fits a free measurement σ when it's the only tight constraint on g; HMA carries cyclical contamination into the "trend" |
| Linear-trend g anchor with loose σ_g HalfNormal(0.10) | g shape clean but divergences 9,967, r_star ESS 15 | The linear pull and the y* drift conflict at endpoints — funnel returns when g has too much freedom |
| Slope-based time-varying k in Resolution C: `k_t = k0 + k_slope * (10y_nominal − cash)` | k0 = 0.71, k_slope = 0.38 (both identified); k_t path 0.18-2.52 pp tracks curve slope sensibly. But: 10,655 divergences, r_star ESS 17, r_star R-hat 1.191. **Substantive r* path barely changed** vs. constant-k (trough 1.10% vs 1.05%; latest 1.76% vs 2.06%) | The slope-based k_t added flexibility (α and k_slope and k0 trade off) without bringing new identifying information. The data does not support disentangling a time-varying term premium from r* itself — adding the structure broke sampling without changing the answer |
| Regime-switching alpha targeting 2011Q3–2021Q4 "great divergence" (alpha_normal / alpha_divergence) | 267 divergences, rhat > 1.01 on some params, low ESS; alpha_normal 0.54, alpha_divergence 0.50 — regimes not separated; g still 2.63% | Same lesson as iteration 7 (GFC split): the data cannot identify time-variation in alpha regardless of where the break is placed. The structural and market anchors' divergence is absorbed by other parameters (k, sigma_r) rather than by a shift in alpha |
| Intercept c replacing k: `r* = α·g + (1−α)·indexed + c + ε` | 359 divergences; c = −0.13, g still 2.66%, α 0.545 — c did not release g | The IS curve pulls g up through α regardless of whether the level offset is k or c. An intercept shifts the blend level but doesn't change the time-varying dynamics that pin g |
| Intercept c alongside k: `r* = α·g + (1−α)·(indexed−k) + c + ε` | 2,104 divergences; c = −0.10, k = 0.60, g = 2.62% | c and (1−α)·k are near-collinear when α is uncertain. Sampler collapsed on the c/k ridge. No benefit |

## Sampler progression across iterations

Sampler diagnostics across the major model variants. All runs use NumPyro NUTS, 5 chains × (3,500 tune + 10,000 draws). `target_accept` was held at **0.90** throughout (one experimental run at 0.97 was reverted — see "What was tried that didn't work").

| Iteration | Spec | Sample | Divergences | Substantive answer |
|-----------|------|--------|-------------|---------------------|
| 1 | Canonical (z RW, no fiscal, no bond) | 1983Q1 | 2 | r* ≈ g, span 0.4 pp (dead z) |
| 2 | + fiscal impulse, λ_z reparam, centred z | 1983Q1 | 54 | r* span 0.5 pp (z still dead) |
| 3 | + non-centred z, AR-only z, 1993Q1 sample | 1993Q1 | 616 | z alive (span 0.28 pp), r* latest 2.10% |
| 4 | + non-centred trend_growth too | 1993Q1 | 5,976 | Broken — doubly-cumulated cumsum |
| 5 | AR(1) z + looser λ_z + indexed-bond observation (Resolution B) | 1993Q1 | 5,107 | r* span 6.2 pp, trough −0.30% (implausible) |
| 6 | **Blend r* = α·g + (1−α)·(indexed_10y − k) + ε (Resolution C, current)** | 1993Q1 | **1,197** | **r* span 3.15 pp, trough 1.3%, α median 0.52** |
| 7 | Resolution C with regime-switching α (α_pre / α_post around 2008Q4) | 1993Q1 | 6,575 | α_pre 0.55, α_post 0.48 — regimes not separated; reverted |
| 8 | Resolution C, σ_g loosened to HalfNormal(0.10), no anchor | 1993Q1 | 5,771 | g 3.26% → 2.73%, but r_star ESS 20 — unreliable |
| 9 | Resolution C + HMA(13) anchor on g (free σ_trend_obs) | 1993Q1 | 8,276 | σ_trend_obs collapsed to 0.022; COVID dip bled into g |
| 10 | Resolution C + linear-trend anchor on g, loose σ_g | 1993Q1 | 9,967 | g shape clean but funnel returned at endpoints |
| 11 | Resolution C + linear-trend anchor on g, tight σ_g, fixed σ_trend_obs=2.0 | 1993Q1 | 2,685 | r* span 3.58 pp, trough 1.05%; g 3.57% → 2.37%; r_star R-hat 1.003, ESS 4,774 |
| 12 | **Iteration 11 + non-centred r_innovation (CURRENT)** | 1980Q1 | **156** | **σ_r funnel eliminated: σ_r R-hat 1.090→1.000, ESS 41→44,528; r_innovation R-hat max 1.053→1.001; remaining divergences in σ_g only** |
| 13 | Resolution C with regime-switching α (α_normal / α_divergence, 2011Q3–2021Q4 divergence dummy) | 1980Q1 | 267 | α_normal 0.54, α_divergence 0.50 — regimes not separated; rhat > 1.01, low ESS; g 2.63% unchanged; reverted |
| 14 | Resolution C with intercept c replacing k | 1980Q1 | 359 | c = −0.13, g 2.66%, α 0.545 — c did not release g; reverted |
| 15 | Resolution C with intercept c alongside k | 1980Q1 | 2,104 | c/k collinear, sampler collapsed; reverted |
| 16 | **Resolution D: canonical r* = g + z + open-economy IS curve (fiscal + ToT + TWI + ICP)** | 1986Q3 | **33** | a_r −0.034 unchanged from A, σ_IS 0.70 unchanged, σ_z 0.06 still piling at 0 → z dead, r* ≈ g range 1.81–2.66%. SOE-block hypothesis empirically refuted; D kept as a contrastive baseline alongside A. |
| 17 | **Resolution E: r* = blend + AR(1) z, σ_z fixed at 0.15** | 1986Q3 | 302 | a_r −0.038, σ_IS 0.70, ρ_z 0.948 (≈ prior); z mean abs 0.04 pp, span 0.12 pp; r* range 0.74–4.14% (matches C); r_star ESS 12,261, z_star ESS 60,185. Soft-anchor reformulation reproduces C; IS curve has no information to add above the blend. E kept as a contrastive baseline. |
| 18 | **Resolution F: E's r* identity + open-economy IS curve** | 1986Q3 | **156** | a_r −0.039, σ_IS 0.70; γ_tot 0.008, γ_twi −0.01, γ_icp 0.02 (all small/dead); z mean abs 0.05 pp; r* range 0.61–4.16% (matches C); r_star ESS 18,011, z_star ESS 71,029 (best of any resolution). Combination of soft anchor + SOE block reproduces C. F kept as a contrastive baseline; closes the empirical loop on cross-resolution invariance of r* ≈ blend. |
| 19 | **σ_z prior-sensitivity sweep on E** (σ_z ∈ {0.05, 0.15, 0.30, 0.50, 1.00}) | 1986Q3 | 270–1,583 | Posterior median r* moves only 0.20pp across σ_z ∈ [0.05, 0.50] (range 2.19% → 2.39% latest); structural-economy parameters (a_r, σ_IS, α, k) within MC error throughout. r* 90% CI @latest scales linearly with σ_z: 1.27 → 2.07 → 3.51 → 5.35 pp. At σ_z=1.00 model overfits (a_r jumps 8× to −0.33, r* trough −3.78%, divergences 1,583). **Direct empirical confirmation of Buncic-Pagan-Robinson 2023: posterior CI is the σ_z prior projected through the model.** See "Empirical confirmation: σ_z prior-sensitivity sweep on Resolution E" in the headline finding. |
| 20 | **α prior default change**: Beta(2, 2) → Beta(1, 1) (Uniform) across C, E, F | 1986Q3 | 205–819 | α posterior under Uniform reveals weak data preference for higher α (median 0.56 vs Beta(2, 2)'s 0.52); posterior 90% range tracks the Uniform prior's range almost 1:1 (HDI [0.07, 0.96]). r* path tilts marginally more toward trend-growth shape: trough 0.69 → 0.85 (C), latest 2.19 → 2.23. All other parameters within MC error. Divergences rise (the Uniform's mass at α near 0/1 is harder geometry) but sampling stays clean. New default is methodologically more honest — uninformative prior on a parameter the data weakly identifies. |
| 21 | **Hierarchical Beta(a, b) on C** with a, b ~ HalfNormal(1) | 1986Q3 | 365 | a_hyper 0.75 [0.09, 2.03], b_hyper 0.63 [0.05, 1.88] — both below 1, so the data-preferred Beta shape is **near Jeffreys (Beta(0.5, 0.5))** with mild upper-endpoint tilt. α posterior is bimodal-ish: quantiles 5/25/50/75/95 = 0.00 / 0.24 / 0.65 / 0.95 / 1.00. r* trough 1.19 (highest of any C variant — more weight at α=1 endpoint pulls r* toward g, which doesn't drop in 2020 like indexed_10y does). r* latest 2.27. **Third independent confirmation that α is essentially unidentified**: when the prior is data-determined, the data picks "near-non-informative-with-endpoint-mass". Initially run as `rstar_hlw_C_alpha_hier`. |
| 22 | **Promoted hierarchical Beta to Resolution G; G becomes the new default** | 1986Q3 | 365 | Same spec as iteration 21 but wired as a first-class resolution. r* trough 1.19% (2020Q4), latest 2.27%, ESS 4,524, R-hat 1.000. Saved as `rstar_hlw_G`; charts in `charts/rstar-hlw-G/`. The bimodal α posterior is now the headline output of the default model. |
| 23 | **Indexed yield gap filled (2013Q3-2014Q3); A-G all re-estimated on contiguous 158-quarter sample** | 1986Q3 | 85–3,349 | The RBA F2 series has a 5-quarter hole when the benchmark indexed bond was being transitioned. `get_indexed_yield_filled()` now patches the gap via nominal − interpolated breakeven, restoring 5 quarters to the sample. Re-estimating A–G dramatically reduced divergences (A: 4,050→111, B: 11,458→3,349, C: 428→193, E: 819→173) without changing substantive results. r* paths shift by <0.05 pp; G's bimodal α posterior is reproduced almost exactly (a_hyper 0.74, b_hyper 0.62). The fill captures the 2013 taper-tantrum spike in real yields that a naïve interpolation of indexed_10y itself would miss. |
| 24 | **G hyperprior switched: HalfNormal(1) → Uniform(0.25, 2) on (a, b)** | 1986Q3 | 181 (was 307) | The HalfNormal(1) hyperprior puts ~58% of its mass below 1, mildly favouring U-shaped Betas a priori. Under it, a_hyper 0.74 and b_hyper 0.62 (both sub-1) and α posterior strongly bimodal (densities ~4.4/0.6/8.4 at 0/middle/1). Switching to `Uniform(0.25, 2)` — flat, symmetric around 1, covers both U-shapes and bell-shapes — moved the hyperparameter medians to 1.18 and 1.09 (just above 1), and flattened the α posterior bimodality (densities ~1.7/1.0/3.0). r* path barely shifted (trough 1.18→0.89, latest 2.27→2.20). **Substantive insight: the bimodality of α observed under HalfNormal was substantially the prior's contribution, not the data's. The endpoint pile-up is a structure × diffuse-signal interaction — when (a, b) can drop below 1, U-shapes get explored and put endpoint mass on α; the data lacks the strength to rule them out.** Documented as the endpoint-stacking insight in the headline finding. |
| 25 | **Resolution H: time-varying α_t via logit-RW (σ_a fixed at 0.05)** | 1986Q3 | 206 | Tests whether the data wants α to drift over time — motivated by the constant-α "smear" effect and Bullock's "r* has shifted upward" framing (which would imply α_t drifts toward 0 in recent years). **Result: α_t is essentially flat — std dev across the whole 158-quarter sample is 0.002 pp, total drift end-to-end is 0.01pp** (0.583 in 1989 → 0.593 in 2023). r* range [0.92, 4.11], trough 0.92% (2020Q4), latest 2.20% — matches C and G to within MC error. r_star ESS 6,246; alpha_rstar ESS 88,394. **The data actively pulls α_t toward a constant despite a permissive logit-RW prior. The IS curve has no era-specific signal about which anchor matters more in different periods.** Bullock's view about r* dynamics is consistent with reading the bond market — which is a model preference, not evidence the data here can verify. H is kept as a contrastive baseline that demonstrates the constant-α structure in C/G isn't artificially constraining the answer. |

Iteration 12 (current) applies a standard non-centred reparameterisation to r_innovation: sample `r_innovation_raw ~ N(0, 1)` and scale by σ_r, rather than sampling `r_innovation ~ N(0, σ_r)` directly. This eliminates the Neal's funnel that formed when σ_r was small (the blend already explains most of r*, so σ_r wants to be near zero). Divergences dropped from 908 to 156, with the residual divergences attributable to σ_g — the trend_growth GaussianRandomWalk scale, which cannot be non-centred (see iteration 4). Substantive estimates are unchanged from iteration 11.

Iterations 7 and 13 both tested regime-switching α. Iteration 7 split at the GFC (2008Q4); iteration 13 targeted the 2011Q3–2021Q4 "great divergence" when the structural anchor (g) and market anchor (indexed bond yield) visibly separated on the decomposition chart. Both failed: α posteriors overlap heavily regardless of where the break is placed (iteration 7: 0.55 vs 0.48; iteration 13: 0.54 vs 0.50). **The data cannot identify time-variation in α — one constant α is the right specification.**

Iterations 8-10 explored loosening σ_g and adding observation anchors on g. The lesson: (a) the data won't support a free measurement σ on the g anchor — it collapses; (b) σ_g must stay tight or the y*-equation funnel returns. The combination that works is **tight σ_g + linear-trend anchor + fixed σ_trend_obs** — the linear trend provides the secular shape, σ_trend_obs is fixed at a large value to keep the anchor soft, and the tight σ_g prevents random-walk innovations from competing with y* innovations.

## What was *not* tried (potential future work)

- **Term-structure block** (Bauer-Rudebusch 2020 "Falling Stars"): full arbitrage-free yield curve with multiple maturities. Would let term premium be a proper latent without re-introducing identification problems. Significant engineering effort.
- **Long-run survey expectations** (Del Negro et al 2017): use Consensus Economics 6-10y forecasts of cash rate as an additional observation. Would pin r*'s long-run mean. Need data sourcing.
- **Convenience-yield observation** (Szoke-Vazquez-Grande-Xavier 2024 FEDS note): the canonical recent fix. Would need long-history Australian AA corporate bond yield data, which is not available in the project (RBA F3 only goes back to ~2005).
- **AR(1) trend growth**: would mean-revert g, possibly stabilising σ_g. Risk: changes the long-run interpretation.
- **Time-varying term premium in indexed bond equation**: explored conceptually, rejected because it re-introduces identification failure.

## File structure

```
src/models/rstar_hlw/
├── observations.py           # Data loading (incl. indexed_10y, linear-trend g anchor, SOE regressors)
├── equations/
│   ├── trend_growth.py       # g state equation (centred RW) + soft observation on g
│   ├── potential.py          # y* state equation, exposes σ_y* in latents
│   ├── r_star.py             # r* = α·g + (1-α)·(indexed_10y − k) + ε   (Resolutions C and G;
│   │                         #   G uses hierarchical Beta(a,b) on α via constant['alpha_hierarchical']=True)
│   ├── r_star_blended_z.py   # r* = α·g + (1-α)·(indexed_10y − k) + z   (Resolutions E and F)
│   ├── z_star.py             # r* = g + z state (Resolutions A, B, D)
│   ├── is_curve.py           # IS curve; fiscal + opt-in SOE block (ToT, TWI change, ICP)
│   ├── phillips.py           # Phillips curve on annual π
│   └── indexed_bond.py       # separate indexed_10y observation (Resolution B only)
├── estimate.py               # Model assembly, NUTS sampling, save/load; resolution dispatch
├── results.py                # RStarResults dataclass
├── analyse.py                # Fan charts: r*, output gap, g, decomposition, alpha posterior
├── run.py                    # CLI: --resolution, --start, --estimate-only, --skip-estimate, --seed
└── MODEL_NOTES.md            # This file

run-rstar-hlw.sh              # Shell wrapper
```

The SOE-block IS-curve regressors are loaded into `obs` for every resolution by `observations.py`; `estimate.py` strips them from A/B/C and keeps them only for D. `z_star.py` is wired in for A, B, and D; `indexed_bond.py` only for B. C uses `r_star.py` for the deterministic blend.

External data dependencies for the IS curve:
- `src/data/tot.py` — ABS Terms of Trade quarterly % change
- `src/data/twi.py` — RBA TWI quarterly change (lag-1 helper)
- `src/data/commodity_prices.py` — RBA Index of Commodity Prices (A$), table I2, quarterly % change
- `src/data/world_rstar.py` — NY Fed HLW r* for US/Euro/Canada (chart only, not in model)

## References

- Holston, Laubach, Williams (2017): "Measuring the Natural Rate of Interest"
- Holston, Laubach, Williams (2023 update): NY Fed Staff Report 1063
- Lewis, Vazquez-Grande (2019): "Measuring the Natural Rate of Interest" — λ_z reparameterisation, AR(1) z. Code: https://github.com/kflewis/rStarLVGPublic
- Buncic (2021): "On a standard method for measuring the natural rate of interest" — MUE critique. Code: https://github.com/4db83/Issues-with-HLWs-natural-rate-Code
- Buncic, Pagan, Robinson (2023): "On Constructing a Country-Specific Time Series for the Natural Rate of Interest" — formal identification critique. The "shocks ≥ observables → posterior is the prior projected through the model" finding underlying the Headline finding section above.
- Szoke, Vazquez-Grande, Xavier (2024 FEDS Note): "Convenience Yield as a Driver of r*"
- Del Negro, Giannone, Giannoni, Tambalotti (2017): "Safety, Liquidity, and the Natural Rate of Interest". Code: https://github.com/FRBNY-DSGE/rstarBrookings2017
- Bauer, Rudebusch (2020): "Interest Rates Under Falling Stars"
- McCririck, Rees (RBA Bulletin Sep 2017): "The Neutral Interest Rate"
- Ellis (RBA speech 2022): "The Neutral Rate: The Pole-star Casts Faint Light"
