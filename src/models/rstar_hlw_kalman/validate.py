"""Check the state-space form by recovering known parameters from simulated data.

THE POINT. A state-space form can be subtly wrong in ways nothing downstream
reveals: a lag indexed one period out, a sign flipped, a shock loaded on the
wrong row. Fitting it to real data would still produce plausible-looking
output, because the model would simply be describing a different model's
world. Simulating from the form itself and recovering the parameters that
generated the data is the test that catches those.

It is not a test of whether HLW describes Australia. It is a test of whether
the matrices in `state_space.py` say what the equations in the docstring say.

Run:
    uv run python -m src.models.rstar_hlw_kalman.validate
"""

import numpy as np
import pandas as pd
from scipy import optimize

from src.models.rstar_hlw_kalman.state_space import (
    N_STATES,
    Params,
    log_likelihood,
    transition,
)

# Parameters the simulated world runs on. Chosen near the MCMC posterior so
# the test exercises the region the real model occupies, not a comfortable one.
TRUTH = Params(
    a_y1=0.95,
    a_y2=-0.05,
    a_r=-0.06,
    b_y=0.28,
    sigma_is=0.55,
    sigma_pi=0.55,
    sigma_ystar=0.30,
    sigma_g=0.10,
    sigma_z=0.10,
)

# Long enough that the variances are identifiable in principle. Real samples
# are 134 quarters, and a form that only recovers at 4,000 is still correct
# but tells you the data cannot pin it, which is a separate finding.
N_PERIODS = 4_000
SEED = 42

# Starting values for the optimiser, deliberately away from TRUTH so the test
# cannot pass by never moving.
START = Params(
    a_y1=0.80,
    a_y2=-0.10,
    a_r=-0.10,
    b_y=0.15,
    sigma_is=1.0,
    sigma_pi=1.0,
    sigma_ystar=0.60,
    sigma_g=0.20,
    sigma_z=0.20,
)

# An eigenvalue this close to one is a unit root: y*, g and z are random
# walks, so three are expected and anything else means T is miswired.
_UNIT_ROOT_TOLERANCE = 0.999

_FIELDS = (
    "a_y1", "a_y2", "a_r", "b_y",
    "sigma_is", "sigma_pi", "sigma_ystar", "sigma_g", "sigma_z",
)


def simulate(p: Params, n: int, seed: int) -> dict[str, np.ndarray]:
    """Generate observations from the state space at `p`.

    Built from the EQUATIONS rather than from the matrices, so that a mistake
    in the matrices cannot hide by being made twice.
    """
    rng = np.random.default_rng(seed)

    ystar = np.zeros(n)
    g = np.zeros(n)
    z = np.zeros(n)
    ystar[0], g[0], z[0] = 1200.0, 3.0, 0.5
    for t in range(1, n):
        ystar[t] = ystar[t - 1] + g[t - 1] / 4.0 + rng.normal(0, p.sigma_ystar)
        g[t] = g[t - 1] + rng.normal(0, p.sigma_g)
        z[t] = z[t - 1] + rng.normal(0, p.sigma_z)

    # An exogenous real rate with roughly the variability of the real thing.
    real = 2.0 + rng.normal(0, 1.5, size=n)
    pi_exp = 2.5 + rng.normal(0, 0.5, size=n)

    log_gdp = np.zeros(n)
    log_gdp[:2] = ystar[:2]
    pi_4 = np.zeros(n)
    pi_4[0] = pi_exp[0]
    for t in range(2, n):
        gap1 = log_gdp[t - 1] - ystar[t - 1]
        gap2 = log_gdp[t - 2] - ystar[t - 2]
        rgap1 = real[t - 1] - (g[t - 1] + z[t - 1])
        rgap2 = real[t - 2] - (g[t - 2] + z[t - 2])
        log_gdp[t] = (
            ystar[t]
            + p.a_y1 * gap1
            + p.a_y2 * gap2
            + (p.a_r / 2.0) * (rgap1 + rgap2)
            + rng.normal(0, p.sigma_is)
        )
    for t in range(1, n):
        pi_4[t] = (
            pi_exp[t]
            + p.b_y * (log_gdp[t - 1] - ystar[t - 1])
            + rng.normal(0, p.sigma_pi)
        )

    return {
        "log_gdp": log_gdp,
        "cash_rate": real + pi_exp,   # the model forms real = cash_rate - pi_exp
        "pi_exp": pi_exp,
        "pi_4": pi_4,
    }


def _to_params(x: np.ndarray) -> Params:
    return Params(**dict(zip(_FIELDS, (float(v) for v in x), strict=True)))


def _to_vector(p: Params) -> np.ndarray:
    return np.array([getattr(p, f) for f in _FIELDS], dtype=float)


def estimate(obs: dict[str, np.ndarray], start: Params) -> Params:
    """Maximum likelihood over the nine scalars."""
    def negative_ll(x: np.ndarray) -> float:
        ll = log_likelihood(obs, _to_params(x))
        return -ll if np.isfinite(ll) else 1e12

    # The five sds must stay positive; the slopes are left free so a wrong
    # sign shows up as a wrong answer rather than being clipped out of sight.
    bounds = [(None, None)] * 4 + [(1e-4, None)] * 5
    result = optimize.minimize(
        negative_ll, _to_vector(start), method="L-BFGS-B", bounds=bounds,
    )
    return _to_params(result.x)


def main() -> None:
    """Simulate, re-estimate, and report how close the recovery is."""
    print(f"Simulating {N_PERIODS} quarters from the state space at TRUTH...")
    obs = simulate(TRUTH, N_PERIODS, SEED)

    t_mat, _ = transition()
    print(f"  T has {int(np.sum(np.abs(np.linalg.eigvals(t_mat)) > _UNIT_ROOT_TOLERANCE))} unit "
          f"eigenvalues of {N_STATES} (expect 3: y*, g, z)")
    print(f"  log-likelihood at TRUTH: {log_likelihood(obs, TRUTH):,.1f}")
    print(f"  log-likelihood at START: {log_likelihood(obs, START):,.1f}")

    print("\nMaximising...")
    fitted = estimate(obs, START)

    table = pd.DataFrame({
        "truth": [getattr(TRUTH, f) for f in _FIELDS],
        "start": [getattr(START, f) for f in _FIELDS],
        "recovered": [getattr(fitted, f) for f in _FIELDS],
    }, index=list(_FIELDS))
    table["error"] = table["recovered"] - table["truth"]
    print()
    print(table.to_string(float_format=lambda v: f"{v:8.4f}"))
    print(f"\n  log-likelihood at the optimum: {log_likelihood(obs, fitted):,.1f}")

    errors = table["error"].abs()
    worst = str(errors.idxmax())
    print(f"\n  largest error: {worst} off by {float(errors.max()):+.4f}")
    print("  Recovery to a few hundredths means the matrices agree with the")
    print("  equations. Anything systematically off means they do not, and no")
    print("  result from this package should be believed until it is fixed.")


if __name__ == "__main__":
    main()
