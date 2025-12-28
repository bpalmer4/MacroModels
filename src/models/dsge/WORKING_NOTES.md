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

## A Speculative Hypothesis: The Production Structure Shifted

The observations above are usually explained as market failures, financial frictions, or cyclical headwinds. But there may be a deeper structural explanation.

**Hypothesis**: Economic growth decoupled from capital accumulation. If true, r* is no longer tied to GDP growth in any fundamental sense.

The core claim is that modern growth is increasingly generated by non-rival, intangible, and rent-bearing inputs, breaking the historical link between output growth and capital scarcity that r* prices.

### The Classical Chain (Pre-GFC)

Standard macro theory assumes:

1. Growth comes from capital deepening plus labour
2. Capital is scarce and rival
3. Investment demand rises with growth
4. r* equilibrates saving and investment
5. Therefore r* ≈ trend growth rate

This logic is embedded in Ramsey-Cass-Koopmans, Solow, and standard DSGE models. The Euler equation assumes households smooth consumption against a rate that prices intertemporal capital scarcity.

### What May Have Changed

Post-2000s growth increasingly came from:

- Software and algorithms
- Data and platforms
- Network effects
- Brand and organisational capital
- Winner-take-most market structures

These have radically different properties from physical capital:

| Property | Physical Capital | Intangible Capital |
|----------|------------------|-------------------|
| Rival? | Yes | Often no |
| Requires external finance? | Yes | Often internal cash flow |
| Scales with investment? | Linearly | Super-linearly (or not at all) |
| Appears in measured investment? | Yes | Poorly |
| Returns accrue to | Marginal product | Rents, market power |

This is not a claim that intangibles dominate all production, but that *marginal* growth increasingly does not require marginal tangible capital.

If this characterisation is roughly correct, several implications follow:

### Implication 1: The Investment-Growth Link Broke

You can grow output without accumulating much measured capital. A software platform scales globally with minimal marginal investment. A network effect creates value through adoption, not capital expenditure.

This would break the classical chain at the root: higher growth no longer implies higher investment demand, so r* need not rise with g.

### Implication 2: MPK for Tangible Capital Stagnated

In standard theory: higher growth → higher marginal product of capital → higher r*.

But if growth accrues to intangibles and rents rather than reproducible capital, the marginal product of *measured* tangible capital can stagnate even as GDP grows.

### Implication 3: Saving Has Nowhere Productive to Go

The economy still generates saving, but growth no longer creates enough tangible investment demand to absorb it.

The equilibrium real rate must fall until:
- The marginal saver is indifferent
- Marginal low-productivity projects get funded
- Or saving is absorbed by asset price inflation

This would explain r* collapsing without implying the economy is stagnating.

### Implication 4: Asset Returns Reflect Rents, Not Capital Scarcity

This may resolve the "two pricing kernels" puzzle. Equity returns reflect:

- Monopoly rents from market power
- Scalability of intangible-heavy business models
- Option value and concentration risk

These are not returns to aggregate capital scarcity. They're returns to *specific* assets that captured the intangible-driven growth.

So r* (goods-market rate) can fall while asset returns stay elevated. No arbitrage closes this gap because:

- Rents are not reproducible capital
- Markets are segmented (you can't arbitrage Apple's market power)
- High returns compensate for concentration risk, not intertemporal substitution

### Why the GFC May Have Been a Trigger, Not a Cause

On this view, the GFC:

- Forced deleveraging that revealed overcapacity in tangible investment
- Accelerated digital substitution (already underway)
- Legitimised ultra-low safe rates
- Exposed and locked in an underlying structural shift

The shift was already happening; the crisis made it visible.

### Caveats and Limitations

This remains speculative. Several objections apply:

1. **Measurement**: Intangible investment is increasingly captured in national accounts (R&D capitalisation since SNA 2008). The "missing capital" story may be overstated.

2. **Timing**: The shift should have been gradual, but r* fell abruptly post-GFC. This could reflect threshold effects, or the GFC may have been more causal than this hypothesis allows.

3. **Cross-country variation**: If this is a technology-driven shift, it should affect all advanced economies similarly. It largely has - but idiosyncratic factors (demographics, policy) also matter.

4. **Unfalsifiable?**: Without a formal model, this risks being a just-so story. The hypothesis needs to generate testable predictions beyond "r* fell." One candidate: industries with high intangible intensity should show high returns, low capex, and weak sensitivity to interest rates. If intangible-heavy sectors respond to rate changes as strongly as tangible-heavy sectors, the hypothesis is in trouble.

5. **Alternative explanations**: Demographics (ageing → more saving), inequality (rich save more), safe asset shortage, and secular stagnation all offer competing or complementary accounts.

### Why This Still Has No Unified Theory

Formalising this hypothesis would require:

- Endogenous market power
- Non-rival capital in production
- Segmented asset markets
- Heterogeneous agents with portfolio choice
- Financial dominance over goods markets

No closed-form general equilibrium model does all of this tractably. So we keep getting "what happened" decompositions without a unified "why."

### Connection to DSGE Failures

If this hypothesis has merit, the DSGE failures documented above are not primarily about:

- Financial frictions (though those matter)
- Zero lower bound (though that constrained policy)
- Expectations formation (though that's hard to model)

They're about the **production function**. Standard DSGE assumes:

```
Y = F(K, L, A)  where K is rival, depreciates, requires finance
```

But if reality is closer to:

```
Y = F(K_tangible, K_intangible, L, A)
where K_intangible is non-rival, self-financing, and scales without investment
```

...then the Euler equation's interest rate is pricing the wrong thing. The rate that clears tangible capital markets has limited relevance for aggregate growth dynamics.

This would explain why the IS curve coefficient hits bounds, why Taylor rules stopped describing policy, and why "reach for yield" is rational when productive tangible investment is saturated.

### AI and the App Economy: The Hypothesis Intensified

AI may be the limiting case of intangible-driven growth - and a natural test of the hypothesis.

**The pattern**:

| Traditional production | AI/App economy |
|-----------------------|----------------|
| Factory, equipment, inventory | Laptop, cloud credits, API calls |
| Scales with capital investment | Scales with users and data |
| Marginal cost > 0 | Marginal cost ≈ 0 |
| Returns to reproducible capital | Returns to network effects, brand, model weights |
| Finance-constrained | Talent-constrained |

A foundation model costs $100M+ to train, but once trained it's non-rival - it serves billions simultaneously at near-zero marginal cost. The "capital" is the model weights, the data, the network effect. None of this requires ongoing finance at the margin.

**Capital-intensive at the frontier, capital-free downstream**: Nvidia and hyperscalers absorb the capex. The rest of the economy accesses AI via API. Productivity may rise across the economy without corresponding investment demand.

**This fits the falsification test**: The hypothesis predicts that intangible-intensive industries should show high returns, low capex, and weak sensitivity to interest rates. The 2022-2024 period offers a natural experiment:

- Rates rose from near-zero to 5%+
- AI investment and adoption *accelerated*
- Nvidia's returns were extraordinary
- Downstream AI adopters needed no capital expenditure

If the traditional transmission mechanism worked, rate hikes should have slowed investment. Instead, the most intangible-intensive sector boomed. This is consistent with the hypothesis - though one episode is not definitive.

**What would an AI bubble/bust mean?**

If AI valuations collapse, it tests the hypothesis in an interesting way:

1. **If the hypothesis is correct**: A bust would destroy rents and equity valuations, but r* would stay low. The underlying dynamic - growth doesn't require broad capital investment - would persist. Asset prices would crash while the safe rate remained depressed. This would mirror the 2000 dot-com bust: equity carnage, but no restoration of the capital-growth link.

2. **What would falsify the hypothesis**: If an AI bust led to r* *rising* - because "real" capital-intensive growth resumed and absorbed saving - that would suggest the intangibles story was overstated. The bust would have revealed that the divergence was cyclical froth, not structural shift.

3. **The dot-com parallel**: After 2000, equity values collapsed but r* continued falling. Growth eventually resumed through the surviving platforms (Google, Amazon). The bust reallocated rents among firms; it didn't restore the link between growth and capital scarcity. If AI follows this pattern, the hypothesis survives. If it doesn't, the hypothesis needs revision.

4. **A more nuanced outcome**: AI may require sustained massive investment at the frontier (chips, data centers, power generation) while remaining capital-free downstream. A bust could reveal that frontier AI is more capital-intensive than software - but the downstream economy, which uses AI without investing, might still decouple from r*. The hypothesis might hold for most of the economy while failing for the AI hyperscalers themselves.

The honest position: we don't know yet. AI is either the hypothesis's strongest confirmation or a sector that will eventually reveal its limits. Time will tell which.

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
