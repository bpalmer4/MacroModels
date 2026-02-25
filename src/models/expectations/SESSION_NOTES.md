# Session Notes - Expectations Model Refactoring

## Completed: All Four Models

### Long Run Model (`market`)

Simplified:
- `breakeven = π_exp + ε` (no α, no λ)
- Nominal bonds pre-1988Q3 (2yr overlap with breakeven which starts 1986Q3)
- Fixed innovation variance (0.12/0.075)
- r* estimated ~3.8% (weakly identified - see note below)

Latest estimate: **2.23%** (2025Q3)

---

### Short Run Model (`short`)

Simplified:
- `market_1y = π_exp + ε` (no α, no λ)
- `inflation = π_exp + ε` (shared σ with survey)
- Estimated innovation variance (σ_early ~0.4, σ_late ~0.14)
- No nominal bonds, no HCOE
- Headline CPI (pre-1993 only) remains

Latest estimate: **3.02%** (2025Q3)

---

### Target Anchored Model (`target`)

Expanded to use all four survey measures:
- market_1y, breakeven, business, market_yoy (all with α, λ)
- GST adjustment for both market_1y and market_yoy (1999Q3-2000Q3)
- Estimated innovation variance (σ_early ~0.30, σ_late ~0.07)
- Target anchor σ = 0.35 (was 0.3)
- r* estimated ~5.4%

**Parameter estimates:**
| Series | α | λ | σ_obs |
|--------|-----|-----|-------|
| market_1y | -0.44 | 0.18 | 0.37 |
| breakeven | -0.46 | 0.16 | 0.58 |
| business | -0.60 | -0.12 | 0.90 |
| market_yoy | 0.04 | -0.01 | 0.13 |

Latest estimate: **2.58%** (2025Q3)

---

### Unanchored Model (`unanchored`)

Same as Target Anchored but without the 2.5% target anchor:
- All four survey measures with α, λ
- Fixed innovation variance (σ_early=0.30, σ_late=0.07) to avoid funnel geometry
- No target anchor observation

**Parameter estimates:**
| Series | α | λ | σ_obs |
|--------|-----|-----|-------|
| market_1y | -0.03 | 0.00 | 0.35 |
| breakeven | -0.23 | 0.06 | 0.62 |
| business | -0.39 | -0.22 | 0.84 |
| market_yoy | 0.48 | -0.20 | 0.15 |

Latest estimate: **2.75%** (2025Q3)

**Gap vs Target:** -17bp (anchor pulling down modestly)

---

## Validation vs RBA PIE_RBAQ

| Period | Correlation | RMSE |
|--------|-------------|------|
| Post-1993 | 0.94 | 0.12pp |
| Post-1998 | 0.92 | 0.06pp |
| 2009-2019 | 0.95 | 0.06pp |

Excellent convergence post-inflation targeting.

---

## r* Identification Note

In the Fisher equation `nominal = π_exp + r* + (π_exp × r*/100)`, r* and π_exp are only weakly identified from each other. With limited overlap, the model finds a combination that fits nominal yields, but the decomposition is uncertain. A lower r* estimate means π_exp adjusts higher to compensate, and vice versa.

- **Target/Unanchored**: Nominal bonds through 1993Q3 (7yr overlap with breakeven) → r* ~5.4%
- **Long Run (market)**: Nominal bonds through 1988Q3 (2yr overlap with breakeven) → r* ~3.8%

---

## Key Files
- `src/models/expectations/stage1.py` - model building
- `src/models/expectations/stage2.py` - diagnostics/plots
- `src/models/expectations/common.py` - constants (ANCHOR_SIGMA=0.35)
- `src/models/expectations/MODEL_NOTES.md` - full documentation

---

## Model Equations Summary

### Long Run
```
breakeven = π_exp + ε
nominal   = π_exp + r* + (π_exp × r*/100) + ε   [pre-1995Q3, 2yr overlap]
Innovation: Fixed σ=0.12 (early) / σ=0.075 (late)
```

### Short Run
```
market_1y = π_exp + ε
inflation = π_exp + ε                            [shared σ]
headline  = π_exp + ε                            [pre-1993 only]
Innovation: Estimated σ_early ~0.4, σ_late ~0.14
```

### Target Anchored
```
market_1y   = π_exp + α + λ × π_{t-1} + ε
breakeven   = π_exp + α + λ × π_{t-1} + ε
business    = π_exp + α + λ × π_{t-1} + ε
market_yoy  = π_exp + α + λ × π_{t-1} + ε
inflation   = π_exp + ε
headline    = π_exp + ε                          [pre-1993 only]
nominal     = π_exp + r* + (π_exp × r*/100) + ε  [pre-1993Q3]
hcoe        = π_exp + mfp + adjustment + ε
2.5         ~ N(π_exp, 0.35)                     [post-1998Q4]
Innovation: Estimated σ_early ~0.30, σ_late ~0.07
```

### Unanchored
```
market_1y   = π_exp + α + λ × π_{t-1} + ε
breakeven   = π_exp + α + λ × π_{t-1} + ε
business    = π_exp + α + λ × π_{t-1} + ε
market_yoy  = π_exp + α + λ × π_{t-1} + ε
inflation   = π_exp + ε
headline    = π_exp + ε                          [pre-1993 only]
nominal     = π_exp + r* + (π_exp × r*/100) + ε  [pre-1993Q3]
hcoe        = π_exp + mfp + adjustment + ε
# NO target anchor
Innovation: Fixed σ_early=0.30, σ_late=0.07
```
