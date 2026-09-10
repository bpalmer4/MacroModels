"""r* identified from the RBA's own reaction to inflation.

    r_t - r*_t = lambda · (pi_t - 2.5) + u_t

Two gaps, assumed proportional: the cash rate's distance from neutral and
inflation's distance from target. No IS curve, no world anchor, no bond market.
r*_t is the neutral *nominal* cash rate, the rate that prevails when inflation
sits at target.

This is the third domestic route to r* tried in this repo. The first went
through the output gap (`rstar_hlw`, `is_curve`) and failed because the rate
does not visibly move output. The second imported HLW's `r* = g + z`, which is
the same identification run overseas. This one asks whether the RBA's own
behaviour reveals neutral, and its weak point is stated up front: over
1993-2026 the cash rate and the inflation gap correlate about 0.2, so `lambda`
has little to grip.
"""
