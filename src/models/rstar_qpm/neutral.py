"""Short-run neutral: the rate that closes the output gap at a horizon.

At each quarter t, hold the real cash rate at a constant level from t+1, hold
the world rate and the wedge where they are, switch every shock off, and run
the IS curve and the exchange rate forward. Short-run neutral is the constant
rate at which the output gap is zero `horizon` quarters on.

The projection is linear in the rate, so it is solved rather than searched:

    ygap_{t+H}(x) = A + B x,     x = rate held - r*_t,     x* = -A / B

`A` is where the gap would be with the rate at trend r*, which is the momentum
the current shocks leave behind. `B` is the cumulative multiplier of a 1pp
rate gap held for H quarters. Short-run neutral is r*_t + x*.

`B` IS THE WHOLE STORY. It is built from b2, b3, kappa and the persistences,
and it divides. A small multiplier turns any momentum into a large deviation
from trend r*, so `B` is saved per draw and should be read beside the result.

The real rate at t itself is sunk: it has already hit the gap at t+1 through
the lagged IS curve, so it enters the projection as data.
"""

from collections.abc import Mapping

import numpy as np

from src.models.rstar_qpm.state_space import QGAP, YGAP, Prepared, W


def gap_projection(
    states: np.ndarray,
    p: Mapping[str, float],
    prep: Prepared,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return A and the multiplier B per quarter, for one draw.

    `states` is (n, N_STATES). B varies by quarter because the cash-flow
    channel scales with the debt the cash rate works on.

    CASH FLOW. Holding the rate from t+1 moves the nominal cash rate once, from
    i_t to the held real rate plus expected inflation, so debt servicing jumps
    once by delta x E x that change and hits the gap a quarter later. Exposure
    is held at its latest value over the projection.
    """
    d = prep.data
    r, rw, i, pie, exposure = d["r"], d["rw"], d["i"], d["pie"], d["exposure"]
    wedge, ygap, qgap = states[:, W], states[:, YGAP], states[:, QGAP]
    rstar = rw + wedge

    # Debt servicing's one-off move at t+1: with the rate held at trend r*
    # (A), and per 1pp held above it (B).
    dsr_a = p["delta"] * exposure * (rstar + pie - i)
    dsr_b = p["delta"] * exposure

    # A: the rate held at trend r*, so the held rate gap is zero from t+1.
    y_a = p["b1"] * ygap - p["b2"] * (r - rw - wedge) - p["b3"] * qgap
    q_a = p["rho_q"] * qgap
    # B: the response to a 1pp rate gap held from t+1. Nothing at t+1 on the
    # output side, since the IS curve is lagged; the exchange rate moves at once.
    y_b, q_b = np.zeros_like(y_a), np.full_like(y_a, p["kappa"])
    for h in range(2, horizon + 1):
        cash_a, cash_b = (dsr_a, dsr_b) if h == CASH_FLOW_QUARTER else (0.0, 0.0)
        y_a, q_a = p["b1"] * y_a - p["b3"] * q_a - p["b4"] * cash_a, p["rho_q"] * q_a
        y_b, q_b = (p["b1"] * y_b - p["b2"] - p["b3"] * q_b - p["b4"] * cash_b,
                    p["rho_q"] * q_b + p["kappa"])
    return y_a, y_b


# The quarter the cash-flow effect reaches the gap: the rate moves at t+1,
# debt servicing with it, and the IS curve reads debt servicing with a lag.
CASH_FLOW_QUARTER = 2


def rate_response(p: Mapping[str, float], horizon: int, exposure: float) -> dict[str, np.ndarray]:
    """Return the output gap's response to a 1pp real rate gap held from quarter 1.

    Index h-1 holds the response h quarters on. Three channels, separated by
    switching them on in turn: `direct` is the real rate alone, adding the
    exchange rate (kappa) gives `exchange_rate`, adding debt servicing gives
    `cash_flow`. `total` has all three, and at the model horizon it is B.
    `exposure` is the debt the cash rate works on, relative to income.
    """
    def path(kappa: float, cash: float) -> np.ndarray:
        out = np.zeros(horizon)
        y_b, q_b = 0.0, kappa
        for h in range(2, horizon + 1):
            shock = cash if h == CASH_FLOW_QUARTER else 0.0
            y_b, q_b = p["b1"] * y_b - p["b2"] - p["b3"] * q_b - p["b4"] * shock, p["rho_q"] * q_b + kappa
            out[h - 1] = y_b
        return out

    cash = p["delta"] * exposure
    direct = path(0.0, 0.0)
    with_fx = path(p["kappa"], 0.0)
    total = path(p["kappa"], cash)
    return {"total": total, "direct": direct,
            "exchange_rate": with_fx - direct, "cash_flow": total - with_fx}


def short_run_neutral(
    states: np.ndarray,
    p: Mapping[str, float],
    prep: Prepared,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return real short-run neutral per quarter, and the multiplier B per quarter."""
    momentum, multiplier = gap_projection(states, p, prep, horizon)
    rstar = prep.data["rw"] + states[:, W]
    return rstar - momentum / multiplier, multiplier
