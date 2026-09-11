"""Neutral identified from the RBA's own reaction to inflation.

    b_t = neutral, a slow random walk
    d_t = b_t + lambda · (pi_t - 2.5)     the rule's PRESCRIBED rate
    r_t = d_t + eps_t

Neutral moves slowly; on top of it the RBA responds to inflation away from the
2.5 target. The model splits the observed cash rate into those two pieces.
`b_t` is neutral; `d_t` carries the inflation response on top and is not.
No IS curve, no world anchor, no bond market.

This is the third domestic route to neutral tried in this repo. The first went
through the output gap (`rstar_hlw`, `is_curve`) and failed because the rate
does not visibly move output. The second imported HLW's `r* = g + z`, which is
the same identification run overseas. This one asks whether the RBA's own
behaviour reveals neutral, and its weak point is stated up front: inflation
targeting worked, so most deviations were small and `lambda` is identified
largely by 2008 and 2022-24.
"""
