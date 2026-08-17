"""A neural surrogate for the sodium-boiling phase of an SFR loss-of-flow transient.

Companion code to the paper. Read in this order:

* :mod:`pinn_ulof.params`    -- every physical constant, with its value
* :mod:`pinn_ulof.physics`   -- the governing equations, shared by both solvers
* :mod:`pinn_ulof.reference` -- the stiff reference solution the surrogate is judged against
* :mod:`pinn_ulof.network`   -- the embedding, the trunk, and the hard-constraint ansatz
* :mod:`pinn_ulof.train`     -- residuals, collocation, and the quasi-Newton solve
"""

import jax

# Double precision, enabled before anything creates an array. This is not a preference:
# the quasi-Newton solve builds curvature pairs from differences of gradients, and at
# single-precision residual magnitudes those differences are noise.
jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - the flag is a bool

from pinn_ulof.config import TrainConfig  # noqa: E402
from pinn_ulof.params import AxialParams  # noqa: E402

__all__ = ["AxialParams", "TrainConfig"]
