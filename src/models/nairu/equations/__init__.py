"""Equation building blocks for the NAIRU + Output Gap model.

Each equation function follows a standard API:

    def equation_name(
        obs: dict[str, np.ndarray],
        model: pm.Model,
        latents: dict[str, Any],
        constant: dict[str, Any] | None = None,
    ) -> str:
        '''One-line description.'''
        ...
        return "Model: mathematical specification"

- obs: Observed data arrays (from build_observations)
- model: PyMC model context (equations add distributions to it)
- latents: Dict of latent variables built so far. State equations
  (NAIRU, potential) ADD their latent to this dict. Observation
  equations READ from it.
- constant: Optional fixed values for coefficients
- Returns: Self-describing model string for reporting

State equations return a string AND populate latents:
    latents["nairu"] = ...
    latents["potential_output"] = ...

Observation equations return a string only.
"""
