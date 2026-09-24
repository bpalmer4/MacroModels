"""The per-run annotations a chart draws, carried on the results it is handed.

Footers, a header, and the date windows a chart shades belong to a run, not to
a chart module: `ystar_ustar` draws `ystar`'s and `ustar`'s charts for its own
run, and needs them to name its model, its sources and its excluded quarters.
Kept as module globals, that meant reaching into the parents to overwrite their
state and restoring it afterwards. Carried on the results object instead, each
run labels its own charts and nothing leaks into the next.

The record is a field on `PosteriorResults`, and it is read and written only
through the functions below. A chart asks for an annotation with its own module
constant as the default, so a run that attaches nothing draws exactly what that
module always drew.
"""

from src.models.common.results import PosteriorResults

# The names a run can attach. Constants, so a misspelt key fails at import
# rather than silently falling back to the default.
LFOOTER = "lfooter"
LFOOTER_BAND = "lfooter_band"
# The source line for a run saved before its inputs were recorded; a current
# run names its own sources and never reaches this.
RFOOTER_FALLBACK = "rfooter_fallback"
ACTUAL_GAP_HEADER = "actual_gap_header"
# Quarters that carried no likelihood, shaded so a fitted line is not read
# across them.
EXCLUDED_WINDOW = "excluded_window"
# Early quarters where the level is set by the state law rather than the data.
UNIDENTIFIED_WINDOW = "unidentified_window"

_NAMES = frozenset({LFOOTER, LFOOTER_BAND, RFOOTER_FALLBACK, ACTUAL_GAP_HEADER, EXCLUDED_WINDOW, UNIDENTIFIED_WINDOW})

Window = tuple[str, str]


def attach(results: PosteriorResults, **annotations: str | Window | None) -> None:
    """Record annotations for this run's charts. None records "no window"."""
    unknown = set(annotations) - _NAMES
    if unknown:
        raise KeyError(f"unknown chart annotation(s): {sorted(unknown)}")
    results.chart_annotations.update(annotations)


def text(results: PosteriorResults, name: str, default: str) -> str:
    """Return a text annotation, or `default` when the run attached none."""
    value = results.chart_annotations.get(name, default)
    if not isinstance(value, str):
        raise TypeError(f"chart annotation {name!r} is a {type(value).__name__}, not text")
    return value


def window(results: PosteriorResults, name: str) -> Window | None:
    """Return a shaded window as (first, last) quarter, or None when there is none."""
    match results.chart_annotations.get(name):
        case None:
            return None
        case (str() as first, str() as last):
            return first, last
        case other:
            raise TypeError(f"chart annotation {name!r} is not a (first, last) quarter pair: {other!r}")
