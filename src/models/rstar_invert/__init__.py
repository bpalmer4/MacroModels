"""r* by conditional inversion of an IMPOSED IS curve.

Every other r* model here asks the data how strong the rate channel is and reads
r* off the answer. On Australian data that fails, and the failure is measured:
`rstar_hlw` returns a_r = -0.04 +/- 0.01 against sigma_IS ~ 0.70 across all
eight of its specifications.

This package inverts the question. The slope is asserted, a slow-moving neutral
is asserted, and the model reports what r* path those two assertions force. A
run is a conditional statement and never an estimate of r*.
"""
