"""Chart styling shared across models, so it cannot drift between them.

Only things that must look identical everywhere belong here. A chart that is
one model's own, drawing that model's own series with its own captions, stays
with that model even when another has a function of the same name.
"""

from typing import Any

# The pandemic window that several models exclude from their likelihoods.
#
# Deliberately plain: the shading marks quarters that carry no likelihood, so
# it should read as an absence rather than as a highlighted episode. Behind
# the lines and the credible-interval band.
#
# `label` puts it in the legend, which is where a reader looks to find out
# what a shaded band means. "excluded from fit" rather than "pandemic" alone:
# the claim is about the estimation, not the epidemiology, and a reader who
# sees only "pandemic" will take the shading for an episode marker.
#
# Gold rather than orange: several of these charts draw their headline series
# in darkorange, and an orange wash behind an orange line costs contrast where
# it is needed most. Gold at low alpha reads as a warm highlight against both
# the orange lines and the cornflower credible-interval band.
_EXCLUDED_SPAN: dict[str, Any] = {
    "color": "gold",
    "alpha": 0.20,
    "zorder": -1,
    "label": "Pandemic: excluded from fit",
}


def ustar_structure_note(constants: dict[str, Any]) -> str:
    """Describe the structure imposed on u*, for a chart footer.

    Built from the run's own recorded constants, so a chart cannot describe a
    structure the run did not use. The knot count is the thing a reader cannot
    infer from the fitted curve: one knot and three give very different
    freedom, and the charts were silent about which was in force.

    Empty for a trace saved before the structure was recorded, which is the
    right failure: better to say nothing than to state a default that may not
    be what produced the line.
    """
    law = constants.get("ustar_structure")
    if not isinstance(law, str):
        return ""
    if law != "spline":
        return f"u*: {law}. "
    knots = [k for k in str(constants.get("spline_knots", "")).split(",") if k]
    if not knots:
        return "u*: spline. "
    count = f"{len(knots)} knot" + ("s" if len(knots) > 1 else "")
    return f"u*: spline, {count} ({', '.join(knots)}). "


def excluded_span_style() -> dict[str, Any]:
    """Return the styling for the excluded-window span.

    Shared because more than one model draws the same window on its own
    charts and the whole point is that it looks identical wherever it
    appears. A copy in each package is how it would drift.

    A fresh dict each call, since callers add `xmin`, `xmax` and their own
    label to it.
    """
    return dict(_EXCLUDED_SPAN)
