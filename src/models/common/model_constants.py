"""The settings a run imposed rather than estimated, carried on its PyMC model.

Equation functions receive the model and return only a description, so the
model is the one object that can carry what they fixed back to the estimator,
which saves it with the trace as `constants`.

The record is an attribute on an object PyMC owns. Its name carries an unusual
suffix so it is unlikely to collide with anything PyMC adds, and it is used
only in this file: everything else goes through the functions below.
"""

from collections.abc import Mapping
from typing import Any

import pymc as pm

# Deliberately unusual: a plain name could silently collide with a PyMC attribute.
_ATTRIBUTE = "fixed_constants_macromodels"


def _record(model: pm.Model) -> dict[str, Any]:
    """Return the model's record, creating it if absent."""
    record = getattr(model, _ATTRIBUTE, None)
    if record is None:
        record = {}
        setattr(model, _ATTRIBUTE, record)
    if not isinstance(record, dict):
        raise TypeError(f"pm.Model.{_ATTRIBUTE} is a {type(record).__name__}, not our dict: a name collision")
    return record


def attach(model: pm.Model, dictionary: Mapping[str, Any]) -> None:
    """Record several constants at once."""
    _record(model).update(dictionary)


def record_constant(model: pm.Model, name: str, value: object) -> None:
    """Record one constant."""
    _record(model)[name] = value


def remove_constant(model: pm.Model, name: str) -> None:
    """Remove one constant, if recorded."""
    _record(model).pop(name, None)


def get_constant(model: pm.Model, name: str) -> object:
    """Return one recorded constant."""
    return _record(model)[name]


def get_dictionary(model: pm.Model) -> dict[str, Any]:
    """Return a copy of every recorded constant."""
    return dict(_record(model))
