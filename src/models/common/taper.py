"""A driftless random walk whose innovation sd tapers from loose to tight.

    x_t = x_{t-1} + sigma_t x z_t,   z_t ~ N(0, 1)
    sigma_t = late + (early - late) x w_t

with `w_t` falling linearly from 1 at the sample start to 0 at `end`, and 0
after it. The walk imposes no shape, only how far the state may move each
quarter, and lets that allowance change once over the sample: loose while a
slow transition runs, tight once it is done.

Non-centred: the innovations are standard normals scaled by the schedule, so
the sampler sees the same geometry whatever the step size.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


def taper_schedule(obs_index: pd.PeriodIndex, early: float, late: float, end: str) -> np.ndarray:
    """Return the innovation sd for each quarter of `obs_index`.

    Linear in quarters from `early` at the first quarter to `late` at `end`,
    flat at `late` after it.
    """
    span = float((pd.Period(end, freq="Q") - obs_index[0]).n)
    if span <= 0:
        raise ValueError(f"the taper must end after the sample start {obs_index[0]}, not at {end}")
    steps = np.arange(len(obs_index), dtype=float)
    weight = np.clip(1.0 - steps / span, 0.0, 1.0)
    return late + (early - late) * weight


def tapered_walk(
    model: pm.Model,
    obs_index: pd.PeriodIndex,
    *,
    name: str,
    early: float,
    late: float,
    end: str,
    init_mu: float,
    init_sd: float,
) -> pt.TensorVariable:
    """Add the tapered walk to `model` as the deterministic `name`, and return it.

    The first quarter carries a Normal(`init_mu`, `init_sd`) prior; each later
    quarter adds one scaled innovation.
    """
    sigma = taper_schedule(obs_index, early, late, end)
    with model:
        init = pm.Normal(f"{name}_init", mu=init_mu, sigma=init_sd)
        z = pm.Normal(f"z_{name}", mu=0.0, sigma=1.0, shape=len(obs_index) - 1)
        steps = pt.cumsum(z * pt.as_tensor_variable(sigma[1:]))
        return pm.Deterministic(name, pt.concatenate([[init], init + steps]))
