"""Random-walk states that collapse to a constant when their scale is zero.

HLW's staged estimation holds a state still in the stages before the stage
that measures it: trend growth while lambda_g is being read off potential,
z while lambda_z is being read off the IS curve. The object that describes
is a Gaussian random walk of zero innovation scale, which PyMC will not
build, since a distribution needs a positive scale.

A scale of exactly zero therefore returns a constant path at a freely
estimated level. That is what "held constant" means in the papers: the state
does not MOVE, not that its value is known. The level stays a parameter and
the data still choose it.
"""

import pymc as pm
import pytensor.tensor as pt


def walk_or_level(
    model: pm.Model,
    name: str,
    *,
    sigma: float | pt.TensorVariable,
    init_mu: float,
    init_sigma: float,
    steps: int,
    non_centred: bool = False,
) -> pt.TensorVariable:
    """Build a driftless Gaussian random walk, or a constant when `sigma` is zero.

    `sigma` may be a tensor, in which case the walk is always built: a scale
    derived from another parameter is not zero in any sense the graph can
    resolve before sampling.

    The constant branch registers `<name>_level` as the free scalar and
    `<name>` as the path, so downstream code indexes the state the same way
    in both branches.

    `non_centred` draws standardised increments and applies the scale
    afterwards. It is the same model in different coordinates: with the scale
    estimated, a centred walk's states are spread by exactly the parameter
    being sampled, so the space narrows to a neck wherever that parameter is
    small, and one step size cannot serve both the neck and the mouth.
    Standardising removes the dependence from the geometry.

    IT IS OPT-IN, AND NOT ALWAYS RIGHT. Centred sampling wins where the data
    pin the states down, non-centred where they do not. Trend growth is
    informed through potential's drift, and non-centring it produced
    catastrophic divergences (see `trend_growth.py`). z is the other case
    entirely: nothing observes it.
    """
    with model:
        if isinstance(sigma, float) and sigma == 0.0:
            level = pm.Normal(f"{name}_level", mu=init_mu, sigma=init_sigma)
            return pm.Deterministic(name, pt.full((steps + 1,), level))

        if non_centred and not isinstance(sigma, float):
            init = pm.Normal(f"{name}_init", mu=init_mu, sigma=init_sigma)
            eps = pm.Normal(f"{name}_eps", mu=0.0, sigma=1.0, shape=steps)
            return pm.Deterministic(
                name,
                pt.concatenate([[init], init + sigma * pt.cumsum(eps)]),
            )

        return pm.GaussianRandomWalk(
            name,
            mu=0,
            sigma=sigma,
            init_dist=pm.Normal.dist(mu=init_mu, sigma=init_sigma),
            steps=steps,
        )
