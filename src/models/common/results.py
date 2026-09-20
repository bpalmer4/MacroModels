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

    trace: az.InferenceData
    obs_index: pd.PeriodIndex
    constants: dict[str, Any] = field(default_factory=dict)

    @property
    def posterior(self) -> xr.Dataset:
        """The trace's posterior group, narrowed at runtime.

        `az.InferenceData` builds its groups dynamically, so a static checker
        cannot see `.posterior` and every access reads as an attribute error.
        Narrowing here once means the rest of the class is checkable, and it
        turns a missing group into a clear failure rather than an AttributeError
        several frames deep.
        """
        posterior = getattr(self.trace, "posterior", None)
        if not isinstance(posterior, xr.Dataset):
            raise TypeError("trace has no posterior group — was it loaded from a completed run?")
        return posterior

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


def vector_draws(values: xr.DataArray, index: pd.Index | None = None) -> pd.DataFrame:
    """Flatten a (chain, draw, time) latent into a time x draw DataFrame."""
    # xarray's .stack, not pandas' — PD013 does not apply.
    stacked = values.stack(sample=("chain", "draw"))  # noqa: PD013
    return pd.DataFrame(np.asarray(stacked.values), index=index)
