# DSGE Modeling: Post-GFC Challenges

Working notes on the fundamental mismatch between DSGE theory and post-GFC observations.

## What We Built

A 4-equation New Keynesian DSGE model for Australia:

1. **IS Curve**: Output gap depends on expected future output and real interest rate
2. **Price Phillips Curve**: Inflation depends on expected future inflation and output gap
3. **Wage Phillips Curve**: Wage inflation similarly determined
4. **Taylor Rule with Smoothing**: Interest rate responds to inflation and output gap

Solved via Blanchard-Kahn method, estimated by maximum likelihood using Kalman filter.

## Estimation Results

With 4 observables (output gap, inflation, interest rate, wage inflation) from 1993Q1-2025Q3:

| Parameter | Estimate | Bound | Issue |
|-----------|----------|-------|-------|
| κ_p (Phillips slope) | 0.01 | Lower | Flat Phillips curve |
| φ_π (Taylor inflation) | 1.01 | Lower | Minimal inflation response |
| φ_y (Taylor output) | 2.00 | Upper | Maximum output response |
| ρ_i (rate smoothing) | 0.92 | Interior | Reasonable |

The optimizer wants the Taylor rule to be as flexible as possible - minimal systematic response to inflation, maximum response to output, high smoothing.

This is not a coding, filtering, or identification problem. The likelihood is telling us the model is wrong.

## What the Model Still Gets Right

The NK-DSGE framework is not useless - it's incomplete for the post-GFC period. It still captures:

- **Pre-GFC dynamics**: When policy rates moved freely and transmitted through conventional channels
- **Inflation anchoring**: The role of credible monetary policy in stabilising expectations
- **Short-run responses outside the ZLB**: Conventional rate changes still affect output and inflation directionally
- **Qualitative transmission**: The signs and relative magnitudes of structural shocks

The failure is specific: the model breaks down when the policy rate diverges from the rate relevant for private decisions, and when asset market dynamics dominate goods market dynamics.

## The LW/HLW Response

Before diagnosing the deeper problem, it's worth noting how the profession responded initially.

Laubach-Williams (2003) and Holston-Laubach-Williams (2017) emerged because:

1. Standard Taylor rules with constant r* were failing
2. Needed to track time-varying natural rate
3. Semi-structural approach - doesn't impose full DSGE optimisation
4. Links r* to trend growth (when growth fell, r* fell)

This was a pragmatic response: acknowledge r* moves, estimate it as an unobserved state, sidestep full DSGE structure.

But even LW/HLW doesn't explain the deeper disconnect that emerged.

## The Taylor Rule Problem

Central observation: **"Since 2008, Taylor hasn't described central bank action at all."**

The standard Taylor rule assumes:
- Constant equilibrium real rate r*
- Systematic response to inflation deviations
- Policy rate equals the rate relevant for private decisions

Post-GFC reality:
- r* collapsed (near zero or negative)
- Forward guidance and QE replaced rate movements
- Zero lower bound constrained policy for extended periods
- Transmission mechanism shifted from interest rate channel to asset prices

This isn't claiming central banks were random - it's saying the mapping from instrument to private behaviour collapsed.

## The r* vs Market Returns Disconnect

This is the deeper puzzle that no unified general-equilibrium framework currently explains.

| Measure | Post-GFC Level |
|---------|----------------|
| Policy rates | Near zero |
| Estimated r* (LW/HLW) | Near zero or negative |
| Equity returns | ~7% real |
| Housing returns | Strong |
| Corporate bond spreads | Compressed |

If r* is the rate that equilibrates savings and investment, arbitrage should close the gap with market returns. It didn't.

### Two Pricing Kernels Diverged

The post-GFC period saw a fundamental split:

1. **Goods-market pricing kernel**: Low r*, persistent slack, muted inflation response to policy
2. **Asset-market pricing kernel**: High required returns, elevated valuations, compressed risk premia

Standard macro theory assumes these must equilibrate. They didn't - for over a decade.

### Implications

1. **Transmission mechanism shifted**: Low policy rates affected the economy through:
   - Wealth effects (asset price appreciation)
   - Financial conditions (credit availability)
   - NOT the traditional IS curve (consumption smoothing via interest rate)

2. **The DSGE IS curve is misspecified**:
   ```
   ŷ = E[ŷ'] - σ·(i - E[π])
   ```
   This assumes the policy rate is the rate relevant for private decisions. When firms price off WACC, equity hurdle rates, and cost of capital >> policy rate, the Euler equation loses empirical meaning. Households face credit spreads, not policy rates. Asset holders and non-holders respond differently.

## What Theory Is Missing

We have fragments, not a coherent framework:

| Phenomenon | Observation | Theoretical Gap |
|------------|-------------|-----------------|
| Reach for yield | Investors accepted compressed spreads | Standard expected-utility models cannot rationalise this without implausible beliefs or constraints |
| QE effects | Asset prices up, goods prices flat | Should move together in standard models |
| Buybacks > investment | Corporations returned capital | Should invest if returns > cost of capital |
| Persistent low rates | Didn't stimulate investment | IS curve says they should |
| Secular stagnation | Describes symptoms | Mechanism unclear |

### DSGE Assumptions That Broke Down

1. **Representative agent**: No heterogeneity between asset holders and non-holders
2. **One interest rate**: No wedge between policy rate and market returns
3. **Rational arbitrage**: No persistent mispricings or "reach for yield"
4. **Investment = savings**: At equilibrium rate - which didn't hold

## Possible Extensions (Not Fully Theorised)

1. **Financial accelerator** (Bernanke-Gertler-Gilchrist): Asset prices affect borrowing constraints
2. **Collateral constraints** (Iacoviello): Housing values affect credit
3. **Wealth effects in IS curve**: Ad-hoc but captures the mechanism
4. **HANK models**: Heterogeneous agents with portfolio choice - frontier research

None of these fully resolves the disconnect. They're partial repairs to a framework that may be fundamentally incomplete for the post-GFC environment.

## Implications for Empirical Macro

1. **Semi-structural models may be more honest**: They don't pretend to microfoundations for transmission mechanisms that broke down. A NAIRU/output gap model estimated via state-space methods acknowledges what we can measure without imposing a theoretical structure that contradicts the data.

2. **Production function approach**: Deriving r* from MPK (Cobb-Douglas) gives the return on capital, which stayed elevated - different from LW/HLW's goods-market r*. This distinction matters.

3. **Reduced-form may outperform structural**: When the theory is wrong, flexible statistical models may fit better and forecast more reliably.

4. **Humility required**: The profession doesn't have a unified theory that explains 2008-2024.

## Summary

The challenge isn't just estimation or computation. It's that:

> **We lack a unified, closed-form general-equilibrium framework that explains what happened.**

DSGE models assume away the features that would explain the post-GFC world: heterogeneous agents, multiple pricing kernels, asset-market segmentation from goods markets. Until theory catches up, structural estimation will struggle with parameters hitting bounds, implausible values, and poor out-of-sample performance.

The path forward likely requires either:
1. Fundamental theoretical innovation (new framework)
2. Honest semi-structural approaches that acknowledge what we don't know
3. Both

For applied work, this suggests semi-structural methods - like state-space NAIRU/output gap models with production function foundations - may be the appropriate tool. They don't claim to solve what theory hasn't solved.

---

*Working notes from DSGE exploration, December 2025*
