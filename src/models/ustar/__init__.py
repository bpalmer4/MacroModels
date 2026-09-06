"""u* from a given output gap: Okun for the level, Phillips for the nominal content.

The output gap is not estimated here. It is read from a completed `ystar`
run and enters as a latent with a measurement-error prior, so its uncertainty
propagates without the supply side being re-estimated.
"""
