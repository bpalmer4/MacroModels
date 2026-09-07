"""Joint y* and u* — y* and u* estimated jointly, with the gap partly free.

`ystar` and `ustar` in one likelihood, plus one addition: the output gap gets a
free component `v` alongside the inflation-defined part.

    gap_t = c·(pi_t - 2.5) + v_t

`v` is the point of the exercise. Inside `ystar` it cannot be estimated, because
`v` and the GDP residual `e_c` enter the same equation additively and only their
sum is visible. Here the gap also appears in the Okun equation, so the
covariance between the GDP and unemployment residuals identifies `beta·Var(v)`.
That moment exists only when the two models share a likelihood, and it is the
whole reason to join them.

What it answers: how much of what `ystar` books as GDP noise is cycle that
unemployment can see. If `sigma_v` is near zero, `ystar`'s identity is
vindicated and `beta_okun` = 2.03 is a real finding. If `sigma_v` is large,
`ystar` has been reporting a slice of the cycle and `ustar` has been fed that
slice.

Run order is `expectations` -> `ystar_ustar`. It does not read `ystar` or `ustar`
output; it re-estimates both.
"""
