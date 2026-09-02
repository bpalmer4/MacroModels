"""Equations for the potential_uc model.

Standard API, matching the convention used elsewhere in the project::

    def equation(obs, model, latents, constant) -> str

State equations populate the `latents` dict; observation equations read from
it. Every equation returns a self-describing string for the run log.

Call order matters, because each step consumes what earlier ones produced:

1. ``scale``              creates sigma_c and the fixed-ratio trend sigmas
2. ``trend_hours``        h*      (needs sigma_h_star)
3. ``trend_productivity`` g_lp, lp*  (needs sigma_g_lp, sigma_lp_star)
4. ``output``             y* = h* + lp*, gap = log_gdp - y*, AR(2) cycle
5. ``hours``              hours observation (needs h*, gap)
6. ``phillips``           inflation observation (needs gap)
"""
