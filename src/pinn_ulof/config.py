"""Hyper-parameters of the shipped surrogate.

Every field is the published value. The measurement fixing each one is stated beside it, so
a number can be checked without leaving the file. Two are exposed as sweep axes on the
command line, ``points`` and ``iters``; the rest are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Hyper-parameters of the shipped surrogate."""

    # --- the two parametric axes ---------------------------------------------------
    #
    # `points` is the size of the fixed collocation set, drawn once and held for the whole
    # solve.
    #
    # Boiling onset is the only quantity the point count buys. At 50 000 iterations and
    # three seeds it is 0.0314 s at 5000 points, 0.0130 s at 10 000 and 0.0127 s at 20 000,
    # so the improvement saturates here; the seed half-range falls from 0.0197 s to
    # 0.0003 s over the same step. The temperature fields, the peak voided length and the
    # saturation margin converge to the same values at every point count measured.
    points: int = 10000
    # Quasi-Newton iterations. Onset error is 0.0160 s at 30 000 and 0.0130 s at 50 000,
    # and its seed half-range falls from 0.0008 s to 0.0003 s; past 50 000 it reaches
    # 0.0115 s at 100 000, which is twice the cost for 0.0015 s.
    iters: int = 50000

    # --- network -------------------------------------------------------------------
    #
    # Random Fourier features, frozen after initialisation. 128 and 256 score
    # indistinguishably from 64 and cost 1.25x and 1.9x per iteration. The width sets how
    # many frequencies the input carries, not what the trunk can represent: the trunk holds
    # 17 029 fitted parameters at every value of this field.
    fourier_features: int = 64
    fourier_scale: float = 2.0
    width: int = 64
    depth: int = 5

    # --- solve -----------------------------------------------------------------------
    #
    # L-BFGS curvature pairs. Passed explicitly: `optax.lbfgs` defaults to 10, and 10
    # against 50 on an identical objective from identical weights is a factor of 17 in the
    # film temperature at 0.974x the cost, because a poorer search direction spends in
    # line-search evaluations what the two-loop recursion saves. 100 pairs costs 1.082x and
    # trades 8% on the fuel field for 15% on the film field.
    lbfgs_history: int = 50
    seed: int = 0

    # --- problem scope ---------------------------------------------------------------
    #
    # Fraction of the 60 s nominal horizon that is trained and scored. The reference stops
    # at 16.5 s, where the coolant leaves the validity range of the sodium property fits,
    # so training beyond that would fit a model outside its own closure range.
    t_train_frac: float = 0.275
