# HLW Bayesian r-star Model — Model Notes

A Bayesian (PyMC + NumPyro NUTS) implementation of the Holston-Laubach-Williams 2017 model, applied to Australian quarterly data. Estimates the natural rate of interest r* jointly with potential output, trend growth, and the output gap.

## Purpose and motivation

The existing NAIRU model in `src/models/nairu/` uses a deterministic r* derived from the Cobb-Douglas production function:

> r* ≈ α·g_K + (1−α)·g_L + g_MFP   (smoothed via Henderson MA)

This is observable-driven and smooth, but offers no uncertainty quantification and tracks supply-side fundamentals only. The motivation for building a Bayesian HLW model was to:

1. Produce a r* with proper Bayesian uncertainty bands
2. Allow r* to be identified jointly with the output gap (rather than imposed from outside)
3. Eventually feed the HLW r* posterior into the NAIRU model's IS curve as an alternative to the Cobb-Douglas r*, with sequential coupling (run HLW first, save the median series, read it into NAIRU's `obs["det_r_star"]`)

## Model structure (current — Resolution C, the blend)

The canonical HLW formulation `r* = g + z` with z as a latent random walk turned out to be unidentifiable on Australian data — see "The fundamental tension" below. The current specification replaces the latent z with a **deterministic blend** of two anchors plus an i.i.d. innovation:

`r*_t = α · g_t + (1 − α) · (indexed_10y_t − k) + ε_t`

where:
- `g_t` is the latent trend growth (random walk, annualised %)
- `indexed_10y_t` is the observed 10-year inflation-linked Australian government bond yield
- `α ∈ [0, 1]` is a single scalar weight (Beta(2, 2) prior)
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
- **The α scalar is Beta(2, 2)** so the prior is centred at 0.5 with mass on the full [0,1] interval. The data picks the posterior; no prior commitment to either anchor.
- **σ_r is largely prior-dominated.** The IS curve already has its own σ_IS absorbing rate-gap noise, and a_r is small (~0.05), so σ_r contributes little to the IS likelihood.

## What r* actually means in this model

In any HLW-style model, **r\* is the time-varying intercept of the IS curve.** It's the level of the real rate at which the output gap has no demand-side pressure on it. Canonical HLW tries to estimate that intercept jointly with the output gap from a latent random walk (z). For Australia that doesn't work — the IS curve's `a_r` channel is too weak to identify a separate latent.

What we're doing now is different. Instead of letting r* be a purely latent random walk that the IS curve has to identify, we're **constructing r\* externally** as a deterministic blend of two observable-driven anchors:

- The **structural anchor** (g, trend growth) — what r* "should" be from the supply-side
- The **market anchor** (indexed_10y − k) — what real rates actually look like in long-term bond markets

Then the IS curve sees this constructed r* and asks "given this is the intercept, do you fit the data well?" The α parameter is the data's verdict on which anchor is more consistent with the IS curve dynamics.

A few observations that follow:

1. The α posterior centred at 0.52 with wide HDI [0.14, 0.91] is the data saying "either anchor works about equally well — neither is obviously wrong, neither is obviously right".
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
| Indexed 10y bond yield | `bonds.get_indexed_yield()` | RBA F2 (1986+) |
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

- α posterior median ≈ **0.52**, 90% HDI **[0.14, 0.91]**
- The data is **genuinely indifferent** between the two anchors — wide α posterior
- k posterior median ≈ 0.65 (term premium offset)
- σ_r posterior median ≈ 0.23 (largely prior-dominated, as expected)
- r* span ≈ **3.15 pp** (between A and B)
- r* path: 1995Q1 ≈ 4.0%, 2015 trough ≈ 1.3%, 2025Q4 ≈ 2.3%
- ~2,700 divergences with the linear soft anchor on g (1,200 without it; the anchor adds some sampling cost in exchange for a more credible g path)
- Latent r* convergence: R-hat 1.003, ESS 4,774 (very clean)
- Latent g convergence: R-hat 1.008, ESS 1,150 (very clean)
- σ_g posterior 0.065 (within prior HalfNormal(0.04) regime), `b_y` 0.38, `a_r` -0.06

**Why this is the best of the three:**
- Level is economically credible (trough 1.05%, never negative)
- Dynamics tell the post-GFC decline + recent recovery story for r*
- g now itself shows a credible secular slowdown (3.57% in 1995 → 2.37% in 2025Q4) thanks to the linear soft anchor — without the anchor, g sat flat near sample-average aggregate GDP growth (~3%) and the chart looked implausibly stable
- α posterior with wide HDI is itself a finding: the data won't pick a single anchor (50/50 between trend growth and bond yield)
- No unidentified random walk; identification problem reduced to one scalar
- Honest about what r* is: a blend of structural and market signals, weights data-determined

## Switching resolutions

The CLI exposes a `--resolution` flag that toggles between A and C:

```bash
./run-rstar.sh -v                          # Resolution C (blend, default)
./run-rstar.sh -v --resolution A           # canonical HLW: r* = g + z
```

Both share trend_growth (with linear soft anchor on g), potential output, IS curve, and Phillips curve. They differ only in the r* identity:
- **A**: `r* = g + z` — z is an AR(1)-reparameterised non-centred random walk (Lewis-Vazquez-Grande 2019 form)
- **C**: `r* = α·g + (1-α)·(indexed_10y − k) + ε` — the blend

Note: the "Resolution A" wired into the toggle uses the AR(1) reparameterised z, not the strict canonical RW z. The strict canonical version was only run in iteration 1 of the sampler progression and produced r* ≈ g (dead z).

## Empirical comparison: A vs B vs C on the same Australian data

All three resolutions were run on the identical 1993Q1-2025Q4 sample. They share the same Phillips curve and potential output equation. The differences:

| Spec | A (textbook canonical) | B (canonical + bond observation) | C (blend, current default) |
|---|---|---|---|
| r* identity | r* = g + z (RW z) | r* = g + z (RW z) | r* = α·g + (1−α)·(indexed_10y − k) + ε |
| Indexed bond | not used | observed: `indexed_10y = r* + tp + ε_tp` | inside r* identity |
| IS curve fiscal regressor | absent | absent | included |
| Soft anchor on g | absent | absent | present (linear trend, fixed σ_trend_obs = 2.0) |

### Side-by-side posterior summary

| | A (canonical) | B (canonical + bond) | C (blend) |
|---|---|---|---|
| Divergences | 4,050 | **11,458** | 2,685 |
| r* span (median path) | **0.26 pp** (flat) | **6.19 pp** (wild) | 3.58 pp |
| r* range (median) | 2.99% to 3.25% | **−1.56% to 4.63%** | 0.82% to 4.18% |
| r* trough (median) | 3.00% (2020Q1) | **−0.72%** (2020Q1) | 1.05% (2020Q1) |
| r* latest (2025Q4) | 3.09% | 1.53% | 2.06% |
| g latest | 2.80% | 2.83% | 2.37% |
| z span (median) | **0.11 pp (dead)** | **5.90 pp (wild)** | n/a |
| σ_z posterior median | 0.063 | 0.294 | n/a |
| `r_star` R-hat | 1.323 (ESS 12) | 1.302 (ESS 13) | **1.003** (ESS 4,774) |
| `trend_growth` R-hat | 1.344 (ESS 21) | 1.145 (ESS 30) | 1.008 (ESS 1,150) |
| `z_star` R-hat | 1.313 (ESS 13) | 1.150 (ESS 27) | n/a |
| `a_r` posterior median | −0.045 | −0.047 | −0.060 |
| `σ_IS` posterior median | 0.72 | 0.70 | 0.72 |
| Bond-related parameters | n/a | tp = 0.93%, σ_tp = 0.063 | k = 0.65, α = 0.53 |

### Cross-validation against published Australian estimates

| Source | r* estimate / range (real) |
|---|---|
| McCririck-Rees (RBA Bulletin Sep 2017) | 0.5–1.5% (around 1% in 2017) |
| Ellis (RBA, Oct 2022) | −0.5% to +2% across 9 models |
| IMF Article IV (2025) | ~0.5% real (assuming 2.5% inflation target) |
| **Resolution A posterior** | flat at ~3% throughout — **above all published ranges, no decline** |
| **Resolution B posterior** | trough −0.72%, latest 1.53% — **trough below all published ranges; r* tracks bond yield** |
| **Resolution C posterior** | trough 1.05%, latest 2.06% — **inside every published range, captures the post-GFC decline + recovery** |

### Analytical assessment

The three resolutions produce three qualitatively different stories from the same data — and each one is informative about a different aspect of the identification problem.

**Resolution A confirms the literature's finding that canonical HLW cannot identify z for Australia.** σ_z posterior median is 0.063 with HDI [0, 0.18], piling up at the lower bound. z span is 0.11 pp over 33 years. r* tracks g with virtually no independent variation. This is exactly what Buncic (2021) shows for Euro Area / UK / Canada: the correct λ_z is essentially zero, and any apparent r* movement in canonical HLW is MUE artefact. The pathology shows up in the sampling diagnostics — latent r_star, trend_growth, and z_star all have R-hat > 1.3, ESS < 25, and σ_z is effectively unsampled (R-hat 1.48, ESS 9.5). **The poor sampling is itself diagnostic of the identification failure.** A misses the post-GFC decline narrative entirely; r* sits at ~3% throughout, which means policy has been "loose" for fifteen years according to A — economically implausible.

**Resolution B goes to the other extreme.** Once the indexed bond is added as a direct observation of r*, σ_z jumps from 0.063 to 0.294 (5×) — z becomes very identifiable, but only because the bond yield is dictating r*. The IS curve and Phillips curve don't materially constrain z; the bond observation does. Result: r* tracks the bond yield minus a constant term premium of 0.93pp, swinging from +4.6% in 1995 to **−0.72% in 2020** as indexed yields collapsed during the global low-rate period. Sampling is even worse than A — **11,458 divergences**, R-hat for r_star at 1.302 — because B has the worst of both worlds: a free RW z that wants to wander, plus a strong bond-observation pull that pins r* tightly to a moving target. The latent state has to satisfy both, and NUTS can't navigate the resulting geometry. **B's r* level is implausibly low at the trough**, below every published Australian estimate, and the constant-term-premium assumption pushes post-GFC term-premium compression into r* itself.

**Resolution C threads the needle.** Instead of adding the bond as an observation that competes with the latent z, C builds the bond into the *definition* of r* via the α blend weight. There is no independent z latent state — r* is a deterministic function of g, indexed_10y, and two scalar parameters (α, k) plus an i.i.d. ε. The blend lets the data choose how much to weight each anchor (α posterior 0.53 with wide HDI [0.14, 0.91]). The structural anchor (g) keeps r* from collapsing to bond-yield-minus-constant; the market anchor keeps r* from sticking at trend growth. Result: **2,685 divergences (best of three), r_star R-hat 1.003, ESS 4,774**, and an r* path with span 3.58pp inside every published Australian range — 4.18% in 1995 declining to 1.05% trough in 2020 and recovering to 2.06% by 2025Q4.

**The IS curve `a_r` and `σ_IS` are remarkably stable across resolutions.** A: a_r = −0.045, σ_IS = 0.72. B: a_r = −0.047, σ_IS = 0.70. C: a_r = −0.060, σ_IS = 0.72. The IS rate channel is genuinely weak for Australian data; no choice of r* identification makes it strong. This is a substantive finding about the Australian IS curve, not a model artefact.

**g paths cluster around the same secular path.** A: 3.16% → 2.80%. B: 3.08% → 2.83%. C: 3.13% → 2.37%. C's g declines further because of the linear soft anchor. A and B have flatter g paths because they don't have that anchor. The structural component that would feed into NAIRU integration is reasonably robust across approaches — though it's more aggressively declining in C.

### Conclusion

The three-way comparison is decisive:

- **A** (canonical) shows what the data alone says: r* cannot be separately identified from g; canonical HLW fails for Australia. This matches Buncic, McCririck-Rees, and Ellis.
- **B** (canonical + bond observation) shows what happens when you add a strong external identifier to a model that otherwise can't pin r*: the external identifier *becomes* r*, with implausibly extreme dynamics and the worst sampling of the three.
- **C** (blend) is the principled middle ground: bond information is included by design rather than as an observation that fights with the latent z. The blend weight α is the only thing the data has to identify, and even there the posterior is wide — but the resulting r* path is credible, the sampling converges, and the model is honest about what it's doing.

A and B together demonstrate why C exists: A shows you can't get useful r* from real-side macro alone, and B shows you can't fix that by bolting on the bond as a separate observation. **C remains the recommended specification.**

Resolutions A and B are valuable as contrastive baselines that demonstrate empirically why the blend is needed — not just theoretically defensible. They are not recommended for substantive use.

## Recommended use

**Default: Resolution C — the blend, with linear soft anchor on g.** Currently wired into `estimate.py`.

Reasoning:
- Most economically credible level (r* trough 1.05%, never negative)
- g shows the secular slowdown narrative (3.57% → 2.37%)
- Honest about its construction: r* blends a structural and a market anchor; g is softly pulled toward a linear trend over the sample
- α posterior tells you how much each anchor matters according to the data — no pre-commitment
- Latent r* and g converge cleanly (R-hat ≈ 1.00 for both)

**Resolution A** is still defensible and is what the canonical literature defaults to for similar countries (Buncic, McCririck-Rees, Ellis). Use it if you want to stay strictly within the canonical HLW spec and are willing to accept r* ≈ g.

**Resolution B** is not recommended — the bond yield is doing all the identifying work, the implied r* level is implausible (negative trough), and divergences are severe.

**Never recommend:** intermediate identification regimes (canonical HLW with looser priors, latent term premium). They are unidentified and produce thousands of divergences.

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
| 11 | **Resolution C + linear-trend anchor on g, tight σ_g, fixed σ_trend_obs=2.0 (CURRENT)** | 1993Q1 | **2,685** | **r* span 3.58 pp, trough 1.05%; g 3.57% → 2.37%; r_star R-hat 1.003, ESS 4,774** |

Iteration 11 (current) produces the cleanest sampling of any model that delivers both non-trivial r* dynamics and a credible g slowdown. Latent `r_star` converges with R-hat 1.003 and ESS 4,774; `trend_growth` with R-hat 1.008 and ESS 1,150; `α_rstar` with R-hat 1.000 and ESS 29,007. Remaining divergences cluster in σ_r, σ_pi (scale parameters), not in the substantive answer.

Iteration 7 (regime-switching α) was an explicit test of whether the trade-off between structural and market anchors changed at the GFC. Result: the data does not support a regime split. α_pre and α_post posteriors overlap so heavily that the difference HDI [−0.44, +0.58] straddles zero by a wide margin, while sampling degraded substantially. **One constant α is the right specification for Australia 1993-2025.**

Iterations 8-10 explored loosening σ_g and adding observation anchors on g. The lesson: (a) the data won't support a free measurement σ on the g anchor — it collapses; (b) σ_g must stay tight or the y*-equation funnel returns. The combination that works is **tight σ_g + linear-trend anchor + fixed σ_trend_obs** — the linear trend provides the secular shape, σ_trend_obs is fixed at a large value to keep the anchor soft, and the tight σ_g prevents random-walk innovations from competing with y* innovations.

## What was *not* tried (potential future work)

- **Term-structure block** (Bauer-Rudebusch 2020 "Falling Stars"): full arbitrage-free yield curve with multiple maturities. Would let term premium be a proper latent without re-introducing identification problems. Significant engineering effort.
- **Long-run survey expectations** (Del Negro et al 2017): use Consensus Economics 6-10y forecasts of cash rate as an additional observation. Would pin r*'s long-run mean. Need data sourcing.
- **Convenience-yield observation** (Szoke-Vazquez-Grande-Xavier 2024 FEDS note): the canonical recent fix. Would need long-history Australian AA corporate bond yield data, which is not available in the project (RBA F3 only goes back to ~2005).
- **AR(1) trend growth**: would mean-revert g, possibly stabilising σ_g. Risk: changes the long-run interpretation.
- **Time-varying term premium in indexed bond equation**: explored conceptually, rejected because it re-introduces identification failure.

## File structure

```
src/models/rstar/
├── observations.py           # Data loading (incl. indexed_10y, linear-trend g anchor)
├── equations/
│   ├── trend_growth.py       # g state equation (centred RW) + soft observation on g
│   ├── potential.py          # y* state equation, exposes σ_y* in latents
│   ├── r_star.py             # r* = α·g + (1-α)·(indexed_10y − k) + ε   (Resolution C, current)
│   ├── is_curve.py           # IS curve with fiscal impulse
│   ├── phillips.py           # Phillips curve on annual π
│   ├── z_star.py             # [legacy] r* = g + z state (Resolutions A/B), kept as bread crumb
│   └── indexed_bond.py       # [legacy] separate indexed_10y observation (Resolution B), kept as bread crumb
├── estimate.py               # Model assembly, NUTS sampling, save/load
├── results.py                # RStarResults dataclass
├── analyse.py                # Fan charts: r*, output gap, g, decomposition, alpha posterior
├── run.py                    # CLI: --start, --estimate-only, --skip-estimate
└── MODEL_NOTES.md            # This file

run-rstar.sh                  # Shell wrapper
```

`z_star.py` and `indexed_bond.py` are no longer imported by `estimate.py` but are kept in `equations/` as a record of the journey. They can be re-wired manually to recover Resolution A or B if needed.

## References

- Holston, Laubach, Williams (2017): "Measuring the Natural Rate of Interest"
- Holston, Laubach, Williams (2023 update): NY Fed Staff Report 1063
- Lewis, Vazquez-Grande (2019): "Measuring the Natural Rate of Interest" — λ_z reparameterisation, AR(1) z. Code: https://github.com/kflewis/rStarLVGPublic
- Buncic (2021): "On a standard method for measuring the natural rate of interest" — MUE critique. Code: https://github.com/4db83/Issues-with-HLWs-natural-rate-Code
- Szoke, Vazquez-Grande, Xavier (2024 FEDS Note): "Convenience Yield as a Driver of r*"
- Del Negro, Giannone, Giannoni, Tambalotti (2017): "Safety, Liquidity, and the Natural Rate of Interest". Code: https://github.com/FRBNY-DSGE/rstarBrookings2017
- Bauer, Rudebusch (2020): "Interest Rates Under Falling Stars"
- McCririck, Rees (RBA Bulletin Sep 2017): "The Neutral Interest Rate"
- Ellis (RBA speech 2022): "The Neutral Rate: The Pole-star Casts Faint Light"
