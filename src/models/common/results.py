"""The plumbing every saved posterior needs, without the economics.

A model's results class exists to answer questions about that model: what u* is,
how the gap decomposes, which term carries the disinflation. None of that
belongs here. What does belong here is the machinery underneath it, which is the
same whatever the model estimated: reach the posterior group, pull a latent out
as time x draw, pull a scalar out as a flat array, render the sources footer.

`kw_only` because subclasses add fields of their own and some of them are
required. Positionally, a required field cannot follow the defaulted ones here;
by keyword the question does not arise. Every construction site already passes
keywords.
"""

from dataclasses import dataclass, field
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from src.models.common.sources import footer_from_constants


@dataclass(kw_only=True)
class PosteriorResults:
    """A sampled model's trace, the data it saw, and how to read them."""

    trace: xr.DataTree
    obs_index: pd.PeriodIndex
    constants: dict[str, Any] = field(default_factory=dict)
    # Footers, headers and shaded windows for this run's charts. Written and read
    # only through `common.chart_annotations`.
    chart_annotations: dict[str, object] = field(default_factory=dict)

    @property
    def posterior(self) -> xr.Dataset:
        """The trace's posterior group as a Dataset."""
        return trace_group(self.trace, "posterior")

    def _vector(self, var_name: str) -> pd.DataFrame:
        """Return a time x draw DataFrame for a vector-valued latent."""
        return vector_draws(self.posterior[var_name], self.obs_index)

    def _scalar(self, var_name: str) -> np.ndarray:
        """Return the flattened posterior draws for a scalar parameter."""
        return np.asarray(self.posterior[var_name].values).ravel()

    @property
    def source_footer(self) -> str | None:
        """The "Built using: ..." line for this run's inputs, or None for an older run.

        Runs saved before `build_observations` began recording where its series
        came from carry no "sources" key, so the charting module falls back to
        its own constant.
        """
        return footer_from_constants(self.constants)


# Summary tables report a 94% highest-density interval, the interval every
# printed table in this repo has carried, and three decimal places.
SUMMARY_CI_PROB = 0.94
SUMMARY_DECIMALS = 3
# The interval's column names in the summary table, as ArviZ builds them.
SUMMARY_HDI_LOWER = f"hdi{round(SUMMARY_CI_PROB * 100)}_lb"
SUMMARY_HDI_UPPER = f"hdi{round(SUMMARY_CI_PROB * 100)}_ub"


def posterior_summary(
    trace: xr.DataTree,
    var_names: list[str] | None = None,
    ci_prob: float = SUMMARY_CI_PROB,
    decimals: int | None = SUMMARY_DECIMALS,
) -> pd.DataFrame:
    """ArviZ's summary table: mean, sd, HDI bounds, ESS, R-hat and MCSE per parameter.

    ArviZ's own default interval is an equal-tailed one, and its "auto" rounding
    rounds each value to what its MCSE justifies, so the raw numbers are asked
    for and rounded here, or not at all when `decimals` is None, for callers
    that compute on the values rather than print them.
    """
    summary = az.summary(trace, var_names=var_names, ci_kind="hdi", ci_prob=ci_prob, round_to="none")
    if not isinstance(summary, pd.DataFrame):
        raise TypeError(f"az.summary returned {type(summary).__name__}, expected a DataFrame")
    return summary if decimals is None else summary.round(decimals)


def trace_group(trace: xr.DataTree, name: str) -> xr.Dataset:
    """Return one group of a trace (posterior, sample_stats, ...) as a Dataset.

    A trace is an xarray DataTree whose groups are child nodes. A node is not a
    Dataset, so callers that want Dataset methods get one here; `to_dataset`
    shares the node's arrays rather than copying the draws. Narrowing once also
    turns a missing group into a clear failure rather than a KeyError several
    frames deep.
    """
    node = trace[name] if name in trace.children else None
    if not isinstance(node, xr.DataTree):
        raise TypeError(f"trace has no {name} group - was it loaded from a completed run?")
    return node.to_dataset()


def vector_draws(values: xr.DataArray, index: pd.Index | None = None) -> pd.DataFrame:
    """Flatten a (chain, draw, time) latent into a time x draw DataFrame."""
    # xarray's .stack, not pandas' — PD013 does not apply.
    stacked = values.stack(sample=("chain", "draw"))
    return pd.DataFrame(np.asarray(stacked.values), index=index)
