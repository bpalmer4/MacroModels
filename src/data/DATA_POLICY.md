# Data Layer Policy

This document defines the conventions for functions in `src/data/`.

## Naming Conventions

### `get_*` — Data Retrieval Only

Functions prefixed with `get_` retrieve raw data or perform pure algebraic derivations. They must NOT apply smoothing, filtering, or other statistical transformations.

**Allowed operations:**
- Loading data from ABS/RBA sources
- Unit conversions (e.g., monthly to quarterly averaging)
- Algebraic identities (e.g., `LP = Δhcoe - Δulc`)
- Log differences for growth rates
- Lagging (shift operations)
- Reindexing/alignment

**NOT allowed:**
- HP filtering
- Henderson moving averages (HMA)
- Rolling windows for smoothing
- Exponential smoothing
- Any trend extraction
- Truncating date ranges

**Data range policy:**
- Return ALL available data from the source — do not truncate or filter date ranges
- Date range alignment happens downstream in `observations.py`, not in individual getters
- This ensures maximum flexibility and avoids data loss

**Examples:**
```python
get_capital_growth_qrtly()      # Raw log difference
get_mfp_growth(...)             # Algebraic: LP - α × capital_deepening
get_labour_productivity_growth(...) # Algebraic: Δhcoe - Δulc
```

### `compute_*` — Transformations Allowed

Functions prefixed with `compute_` perform calculations that may include statistical transformations like smoothing or filtering.

**Allowed operations:**
- Everything in `get_*`, plus:
- HP filtering
- Henderson moving averages
- Rolling calculations
- Trend extraction
- Flooring/capping

**Examples:**
```python
compute_r_star(...)             # HMA smoothing of potential growth
compute_mfp_trend_floored(...)  # HP filter + floor at zero
```

## Return Types

All time series data must be returned with a `pd.PeriodIndex`:

```python
# Correct - PeriodIndex
pd.Series([1.2, 1.3], index=pd.PeriodIndex(['2020Q1', '2020Q2'], freq='Q'))

# Incorrect - DatetimeIndex
pd.Series([1.2, 1.3], index=pd.to_datetime(['2020-03-31', '2020-06-30']))
```

**Why PeriodIndex:**
- Unambiguous period representation (Q1 2020 vs a specific date)
- Consistent frequency handling
- Clean period arithmetic
- Avoids end-of-period vs start-of-period ambiguity

## Where Transformations Happen

Most data transformations for model inputs should occur in `observations.py`, which assembles the observation matrix. Individual data modules should provide clean `get_*` functions that `observations.py` can transform as needed.

Exception: Well-defined derived quantities (like `compute_r_star`) that have a specific economic meaning can include their defining transformations.

## Adding New Data Series

1. Create a `get_*` function that returns raw/algebraic data
2. If smoothing is needed for model input, apply it in `observations.py`
3. Only create a `compute_*` function if the transformed version has standalone economic meaning
