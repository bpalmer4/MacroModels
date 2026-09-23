"""Configuration for the rstar_qpm model.

A semi-structural small-open-economy gap model, QPM style, written as a linear
Gaussian state space so the states integrate out by Kalman filter and NUTS runs
on the parameters alone.

    y*_t   = y*_{t-1} + g_{t-1}/4 + e              potential, 100 x log GDP
    g_t    = g_{t-1} + e                           trend growth, annualised
    ygap_t = b1 ygap_{t-1} - b2 (r_{t-1} - r*_{t-1}) - b3 qgap_{t-1} - b4 dDSR_{t-1} + e   IS
    dDSR_t = delta x E_t x (i_t - i_{t-1}) + e     cash flow: repayments follow the cash rate
    r*_t   = rw_t + w_t,   w_t = w_{t-1} + e       world real rate (data) + AU wedge
    q*_t   = q*_{t-1} + e                          equilibrium real TWI, 100 x log
    qgap_t = rho_q qgap_{t-1} + kappa (r_t - r*_t) + e                      UIP, in gaps

    y_t    = y*_t + ygap_t
    q_t    = q*_t + qgap_t
    pi_t   = a1 pi_{t-1} + (1 - a1) pie_t + a2 ygap_{t-1} + a3 (m_{t-1} - pie_{t-1}) + e
    i_t    = rho_i i_{t-1} + (1 - rho_i)(r*_t + pie_t + phi_pi (pi4_t - target) + phi_y ygap_t) + e
    f_t    = r*_t + bias + e                        the AOFM 5y5y forward, real

`DSR` is household interest payments over disposable income, and `E_t`, last
quarter's DSR over last quarter's standard variable mortgage rate, is the
household debt the cash rate works on, relative to income.

`r` is the real cash rate on MEASURED expectations, `r = i - pie`. The real
TWI rises on appreciation, so a positive rate gap appreciates (`kappa > 0`) and
an appreciation drags on output (`b3 > 0`).

EXPECTATIONS ARE MEASURED, NOT MODEL-CONSISTENT. That is what keeps this a
plain linear state space whose likelihood NUTS can differentiate. The cost is
that nothing here is forward-looking in the rational-expectations sense: the
exchange rate responds to today's rate gap, not to the expected path.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from src.paths import MODEL_OUTPUTS

DEFAULT_OUTPUT_DIR = MODEL_OUTPUTS

# Cleveland Fed 10-year expected real rate, less the Kim-Wright 10-year term
# premium: a world real rate with the US premium taken out.
WORLD_REAL_SERIES = "REAINTRATREARAT10Y"
KIM_WRIGHT_SERIES = "THREEFYTP10"

# AOFM decomposition behind the 5y5y forward. "bc" is the loader's default.
FORWARD_METHOD = "bc"

# The mortgage rate behind debt exposure. The standard variable rate, because
# it runs from 1959; the discounted rate borrowers actually pay starts only in
# 2004. The standard rate overstates what is paid once discounts widened, so
# exposure is understated in later years; `delta` absorbs the level of that
# error but not its drift.
LENDING_RATE = "housing_oo_standard"

# How fast the Australian wedge, and so trend r*, may move: its innovation sd
# per quarter, imposed. Slow, in keeping with a neutral rate that drifts.
SIGMA_W_DEFAULT = 0.10

# The IS curve's rate terms: the real rate gap, the exchange rate gap and debt
# servicing. Switching the IS curve off fixes all three at zero.
IS_PARAMETERS = ("b2", "b3", "b4")

# Short-run neutral is the constant real rate, held from next quarter, that
# closes the output gap this many quarters ahead. Twelve follows the
# three-year horizon the FRB/US-based measures are usually described with.
# ASSUMPTION: that description has not been checked against a Fed source.
DEFAULT_HORIZON = 12

# How many posterior draws are pushed through the simulation smoother to give
# the state paths and short-run neutral their bands.
DEFAULT_STATE_DRAWS = 1_000


@dataclass(frozen=True)
class Prior:
    """One prior. `kind` is "normal", "half", "trunc" (Normal truncated at 0), "beta" or "invgamma".

    For "beta", `mu` and `sd` are the two shape parameters. For "invgamma" they
    are the mean and sd, converted to the shape and scale by `invgamma_shape`.
    """

    kind: str
    mu: float
    sd: float

    def invgamma_shape(self) -> tuple[float, float]:
        """InverseGamma (alpha, beta) with this prior's mean and sd."""
        alpha = 2.0 + (self.mu / self.sd) ** 2
        return alpha, self.mu * (alpha - 1.0)

    def _evaluate(self, grid: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Density on `grid`, mean and sd, from the matching scipy distribution."""
        if self.kind == "beta":
            dist = stats.beta(self.mu, self.sd)
        elif self.kind == "trunc":
            dist = stats.truncnorm(-self.mu / self.sd, np.inf, loc=self.mu, scale=self.sd)
        elif self.kind == "half":
            dist = stats.halfnorm(scale=self.sd)
        elif self.kind == "normal":
            dist = stats.norm(self.mu, self.sd)
        elif self.kind == "invgamma":
            alpha, beta = self.invgamma_shape()
            dist = stats.invgamma(alpha, scale=beta)
        else:
            raise ValueError(f"unknown prior kind {self.kind!r}")
        return np.asarray(dist.pdf(grid), dtype=float), float(dist.mean()), float(dist.std())

    def pdf(self, grid: np.ndarray) -> np.ndarray:
        """Prior density on `grid`."""
        return self._evaluate(grid)[0]

    def moments(self) -> tuple[float, float]:
        """Prior mean and sd, whatever the family."""
        _, mean, sd = self._evaluate(np.zeros(1))
        return mean, sd


def _default_priors() -> dict[str, Prior]:
    return {
        # --- IS curve ---
        "b1": Prior("beta", 6.0, 2.0),          # gap persistence, mean 0.75
        # pp of output gap per pp of real rate gap, one quarter on. SIGN IMPOSED:
        # truncated at zero, so the posterior cannot report a wrong-signed curve.
        "b2": Prior("trunc", 0.10, 0.10),
        # pp of output gap per 1% real TWI gap, one quarter on. Sign imposed.
        "b3": Prior("trunc", 0.03, 0.03),
        # pp of output gap per pp of income added to interest payments, one
        # quarter on: the cash-flow channel. Sign imposed.
        "b4": Prior("trunc", 0.2, 0.2),
        # --- Cash flow ---
        # Share of a cash rate change that reaches debt servicing within the
        # quarter, per unit of debt exposure. Sign imposed.
        "delta": Prior("trunc", 0.7, 0.3),
        "sigma_d": Prior("half", 0.0, 0.3),
        # --- Exchange rate ---
        # % real TWI per pp of real rate gap, same quarter. Sign imposed.
        "kappa": Prior("trunc", 2.0, 2.0),
        "rho_q": Prior("beta", 6.0, 2.0),
        # --- Phillips curve (quarterly annualised trimmed mean) ---
        "a1": Prior("beta", 2.0, 2.0),
        "a2": Prior("trunc", 0.20, 0.20),
        "a3": Prior("trunc", 0.05, 0.05),
        # --- Policy rule ---
        "rho_i": Prior("beta", 6.0, 2.0),
        "phi_pi": Prior("trunc", 1.5, 0.5),
        "phi_y": Prior("trunc", 0.5, 0.25),
        # --- Forward ---
        "forward_bias": Prior("normal", 0.0, 0.5),
        # --- State innovations ---
        # Not identified: simulated at 0.172, the estimate came back 0.094 and
        # 0.076, and left free the likelihood favours potential absorbing the
        # cycle. So this prior sets how smooth potential is. Centred on the
        # 0.17 this model leaned toward under a loose prior, not borrowed from
        # elsewhere. InverseGamma has no mass at zero, which removes the
        # rigid-potential end of the ridge the sampler was wandering.
        "sigma_ystar": Prior("invgamma", 0.17, 0.05),
        "sigma_g": Prior("half", 0.0, 0.10),
        "sigma_ygap": Prior("half", 0.0, 1.0),
        "sigma_w": Prior("half", 0.0, 0.15),
        "sigma_qstar": Prior("half", 0.0, 2.0),
        "sigma_qgap": Prior("half", 0.0, 5.0),
        # --- Observation residuals ---
        "sigma_pi": Prior("half", 0.0, 2.0),
        "sigma_i": Prior("half", 0.0, 0.5),
        "sigma_f": Prior("half", 0.0, 0.5),
    }


@dataclass
class ModelConfig:
    """Specification and sample."""

    # The start of inflation targeting: before it the rule's target is not 2.5.
    start: str = "1993Q1"
    end: str | None = None
    target: float = 2.5

    # The lockdown quarters leave the GDP observation. The states carry on
    # through them, so potential and the gap are interpolated rather than
    # handed a 7% one-quarter swing to explain.
    exclude_start: str | None = "2020Q2"
    exclude_end: str | None = "2021Q3"

    # Quarters with an average cash rate below this leave the policy-rule
    # observation: at the lower bound the rule's prescription was not
    # deliverable, so scoring it would read the floor as a weak response.
    rule_floor: float = 0.5

    # Measurement error on the two exact identities, y = y* + ygap and
    # q = q* + qgap. Not zero only to keep the filter's innovation variance
    # well conditioned; 0.01 on a 100 x log scale is a hundredth of a per cent.
    identity_sd: float = 0.01

    horizon: int = DEFAULT_HORIZON
    state_draws: int = DEFAULT_STATE_DRAWS

    # Off, the 5y5y forward leaves the likelihood and no longer seeds the
    # wedge's starting value, so the level of trend r* has to come from the
    # rest of the system. A test of how much the structure alone can pin.
    use_forward: bool = True

    # Off, the IS curve is switched off: rates no longer move the output gap
    # (b2, b3 and b4 are fixed at zero and not estimated). The level of trend r*
    # then rests on the forward, the rule and the exchange rate. Short-run
    # neutral and the transmission charts do not exist without it.
    use_is: bool = True

    # The wedge's innovation sd, imposed; None estimates it. Left free it runs
    # fast enough for trend r* to follow the forward quarter by quarter, so the
    # forward is fitted almost exactly and the structure has no say. Imposing a
    # slow value lets the forward set the level on average while its short-run
    # wiggles go to its residual. An identification choice, not an estimate.
    sigma_w_fixed: float | None = SIGMA_W_DEFAULT

    @property
    def fixed(self) -> dict[str, float]:
        """Parameters held at a value rather than estimated, under these settings."""
        fixed = {} if self.use_is else dict.fromkeys(IS_PARAMETERS, 0.0)
        if self.sigma_w_fixed is not None:
            fixed["sigma_w"] = self.sigma_w_fixed
        return fixed

    @property
    def estimated_priors(self) -> dict[str, Prior]:
        """The priors actually sampled: every prior less the fixed parameters."""
        return {name: p for name, p in self.priors.items() if name not in self.fixed}

    priors: dict[str, Prior] = field(default_factory=_default_priors)
    output_dir: Path = DEFAULT_OUTPUT_DIR

    @property
    def constants(self) -> dict[str, float | str | list[float] | dict[str, list[float | str]]]:
        """Settings recorded with the trace, so analysis reads what was run.

        The priors go in too: the analysis compares each posterior with ITS
        prior, and reading the current config instead would silently pair an
        old run with new priors.
        """
        return {
            "priors": {name: [p.kind, p.mu, p.sd] for name, p in self.estimated_priors.items()},
            "start": self.start,
            "end": self.end or "",
            "target": self.target,
            "exclude_start": self.exclude_start or "",
            "exclude_end": self.exclude_end or "",
            "rule_floor": self.rule_floor,
            "identity_sd": self.identity_sd,
            "horizon": float(self.horizon),
            "use_forward": float(self.use_forward),
            "use_is": float(self.use_is),
            "sigma_w_fixed": float("nan") if self.sigma_w_fixed is None else float(self.sigma_w_fixed),
        }
