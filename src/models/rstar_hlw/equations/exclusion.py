"""Dropping a window of quarters from an observation equation's likelihood.

The lockdown quarters are the reason this exists. Potential output took part
of the 2020 collapse as a fall in *potential* and never caught back up, so the
recovery read as Australia running seven per cent above capacity through
2022-23 (see MODEL_NOTES.md, "potential output is broken"). The fix that
`ystar` already uses is to stop asking the model to explain those quarters at
all: the states still run through the window under their own priors, and what
goes is the claim that potential plus a cyclical gap should account for what
output and inflation did while the economy was shut.

A `keep` mask is a boolean array over the WHOLE estimation sample. Each
observation equation covers a sub-range of it (the IS curve starts once it has
its lags, the Phillips curve one quarter in), so `drop_excluded` takes the
offset of the first quarter the likelihood covers and slices the mask to match.

This is safe here for the same reason it is safe in `ystar`'s white-noise
branch and unsafe in its AR(1) one: both equations below are written on
contemporaneous quantities with LAGGED DATA on the right-hand side, not as a
recursion in a latent residual, so removing rows from the middle removes those
rows and nothing else. Quarters just after the window still carry lockdown
gaps as regressors. That is unavoidable and intended: those gaps are data
minus a latent, not something the exclusion can or should suppress.
"""

import numpy as np
import pytensor.tensor as pt


def drop_excluded(
    keep: np.ndarray | None,
    first: int,
    fitted: pt.TensorVariable,
    observed: np.ndarray,
) -> tuple[pt.TensorVariable, np.ndarray]:
    """Restrict a likelihood to the kept quarters.

    Args:
        keep: boolean mask over the full estimation sample, or None for no
            exclusion.
        first: index into the full sample of the quarter that decides whether
            row 0 is kept, so row `i` is kept when `keep[first + i]` is True.
            That is the row's own date in the IS curve, and the date of the
            output gap it uses in the Phillips curve.
        fitted: the equation's predicted values.
        observed: the matching observed values.

    Returns:
        The pair with excluded rows removed (or unchanged when keep is None).

    """
    if keep is None:
        return fitted, observed

    if not isinstance(keep, np.ndarray) or keep.dtype != bool:
        raise TypeError("keep must be a boolean numpy array over the estimation sample")

    window = keep[first:first + len(observed)]
    if window.shape != observed.shape:
        raise ValueError(
            f"keep has {len(keep)} quarters, so from offset {first} it covers "
            f"{window.shape[0]}, but the likelihood covers {observed.shape[0]}",
        )

    rows = np.flatnonzero(window)
    return fitted[rows], observed[rows]
