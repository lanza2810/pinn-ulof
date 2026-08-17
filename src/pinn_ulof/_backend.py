"""Array-module dispatch, so one transcription of each correlation serves both solvers.

The reference solver calls these functions with numpy arrays and the network calls them
with JAX arrays. Writing the physics once and dispatching on the argument type is what
makes "the network and its ground truth solve the same equations" a property of the code
rather than a claim about it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def xp(x: Any) -> Any:  # noqa: ANN401 - deliberately backend-agnostic
    """Return the array module (``numpy`` or ``jax.numpy``) matching ``x``.

    Both JAX flavours resolve here, concrete arrays and tracers alike, because both module
    paths start with ``jax``.
    """
    if type(x).__module__.startswith("jax"):
        import jax.numpy as jnp  # noqa: PLC0415

        return jnp
    return np


def like(x: Any, values: Any) -> Any:  # noqa: ANN401 - deliberately backend-agnostic
    """Return ``values`` as an array of the same backend as ``x``."""
    return xp(x).asarray(values)
