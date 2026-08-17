"""Residuals, collocation, and the quasi-Newton solve. This is the entire training loop.

Draw points, square the residuals, run L-BFGS. There is no first-order stage, no adaptive
sampling, no loss weighting, no curriculum and no resampling.

The collocation set is held fixed because the curvature pairs a quasi-Newton method
accumulates are meaningful only while the objective is unchanged; redrawing during the solve
invalidates that history and degrades the result.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

if TYPE_CHECKING:
    from collections.abc import Callable

from pinn_ulof.config import TrainConfig
from pinn_ulof.network import (
    N_TEMPS,
    AxialPinn,
    horizon,
    normalised_state,
    power_shape,
    state_and_grads,
)
from pinn_ulof.params import AxialParams
from pinn_ulof.physics import (
    continuous_derivatives,
    line_geometry,
    residual_normalisation,
)

#: Residual blocks: one per *temperature*. The void is not among them --- it is closed
#: algebraically on the coolant temperature, so its transport equation is no longer an
#: independent condition and enforcing it as a fifth residual fights the other four.
N_BLOCKS: int = N_TEMPS
#: Fields the network produces: the four temperatures and the void fraction.
N_FIELDS: int = N_TEMPS + 1
#: Time windows the loss is averaged over. See :func:`loss`.
N_CHUNKS: int = 32


# --- collocation --------------------------------------------------------------------
def collocation(cfg: TrainConfig, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Draw the fixed collocation set: ``cfg.points`` uniform over the unit square.

    Uniform in both ``zeta`` and ``t_hat``, drawn once. There is no sampling choice in the
    method: no density, no refinement, no resampling.

    The draw consumes ``split(key)[0]`` and not ``key``. That derivation is load-bearing: it
    is the stream the shipped model and every published ladder were fitted on, and drawing
    from ``key`` directly gives a different set and different results at every rung --
    measured at 5000 points, boiling-onset error moves from 0.0314 s to 0.0103 s.
    """
    k_uniform = jax.random.split(key)[0]
    pts = jax.random.uniform(k_uniform, (cfg.points, 2))
    return pts[:, 0:1], pts[:, 1:2]


# --- residuals ----------------------------------------------------------------------
def residuals(
    model: AxialPinn, p: AxialParams, cfg: TrainConfig, zeta: jax.Array, that: jax.Array
) -> tuple[jax.Array, ...]:
    """Signed residual of each governing equation, in normalised variables.

    The right-hand sides come from :func:`pinn_ulof.physics.continuous_derivatives`, which
    is the same function the reference solver discretises. The network and its ground
    truth therefore solve the same equations by construction rather than by review, which
    is the one structural guarantee worth having in a study like this.

    One block per temperature. The void fraction is closed algebraically on the coolant
    temperature, so the void transport equation is not an independent condition and is not
    residualised; the quasi-steady closure does not satisfy it, and enforcing it as a fifth
    block distorts the temperature fields for a factor of thirty in the converged error.

    Each block is multiplied by a scale that brings the equations to a common magnitude.
    That changes the loss and never the equations: dividing the scale back out recovers
    the physical residual exactly, so the zero set is untouched.
    """
    dT = p.P_0 / (p.w_0 * p.c_c)
    t_end = horizon(p, cfg)
    theta, d_dt, d_dz = state_and_grads(model, p, zeta, that)
    T_f, T_cl, T_s, T_c = (p.T_in + theta[:, k : k + 1] * dT for k in range(N_TEMPS))
    alpha = theta[:, N_TEMPS : N_TEMPS + 1]
    rhs = continuous_derivatives(
        that * t_end,
        T_f,
        T_cl,
        T_s,
        T_c,
        alpha,
        d_dz[:, 3:4] * dT / p.H,  # coolant gradient, back in physical units
        d_dz[:, 4:5] / p.H,  # void gradient
        p,
        line_geometry(p),
        power_shape(p, zeta),
        1.0,
    )
    # d(theta)/d(t_hat) against the physical rate, both in normalised temperature.
    scales = [t_end / dT] * N_TEMPS + [t_end]
    nrm = residual_normalisation(p, t_end)
    return tuple(
        ((d_dt[:, k : k + 1] - scales[k] * rhs[k]) * nrm[k]).squeeze(1) for k in range(N_BLOCKS)
    )


def loss(
    model: AxialPinn, p: AxialParams, cfg: TrainConfig, zeta: jax.Array, that: jax.Array
) -> jax.Array:
    """Squared residual, summed over the equations and averaged over time windows.

    No initial- or boundary-condition terms appear: the ansatz satisfies them exactly, so
    there is nothing to weight against the residual, and the four blocks enter with equal
    weight because the variable scaling has already put them on a common magnitude.

    The average is taken **per time window and then across windows**, not over the points
    directly. A plain mean would let wherever the points happen to lie become a statement
    about where the residual matters, and boiling onset falls at 10.98 s of a 16.5 s
    window, so any density tilted toward early time would weight against the event the
    model exists to predict. Averaging within each of ``N_CHUNKS`` windows first keeps
    sampling density and loss weighting independent: a uniform draw fluctuates in how many
    points land in each window, and this makes the objective insensitive to that.
    """
    e = sum(r**2 for r in residuals(model, p, cfg, zeta, that))
    idx = jnp.clip((that.reshape(-1) * N_CHUNKS).astype(int), 0, N_CHUNKS - 1)
    counts = jnp.bincount(idx, length=N_CHUNKS)
    sums = jnp.bincount(idx, weights=e, length=N_CHUNKS)
    return jnp.mean(sums / jnp.maximum(counts, 1))


# --- the solve ----------------------------------------------------------------------
def train(
    p: AxialParams | None = None,
    cfg: TrainConfig | None = None,
    *,
    verbose: bool = True,
    on_checkpoint: Callable[[int, AxialPinn], None] | None = None,
    checkpoint_every: int = 0,
) -> tuple[AxialPinn, AxialParams, TrainConfig]:
    """Fit the surrogate. Returns ``(model, params, config)``.

    L-BFGS with a strong-Wolfe line search, on one fixed collocation set, for
    ``cfg.iters`` iterations. ``memory_size`` is passed explicitly because `optax.lbfgs`
    defaults to 10; :class:`~pinn_ulof.config.TrainConfig` records what that costs.

    ``checkpoint_every`` splits the loop into blocks and hands the model to
    ``on_checkpoint(iterations_done, model)`` after each, so a long run can be stopped and
    still leave something usable behind. **The split does not change the optimisation.**
    Both the parameters and the optimiser state --- which is where L-BFGS keeps its
    curvature pairs --- are carried from one block into the next, so the sequence of updates
    is the one an unsplit loop would produce; only the boundary at which the compiled loop
    returns to Python moves. Restarting the optimiser at each block instead would throw away
    the curvature memory every block, which is a different and much worse algorithm.
    """
    p = p or AxialParams()
    cfg = cfg or TrainConfig()
    key = jax.random.PRNGKey(cfg.seed)
    k_model, k_points = jax.random.split(key)

    model = AxialPinn(cfg, k_model)
    zeta, that = collocation(cfg, k_points)
    params, static = eqx.partition(model, eqx.is_inexact_array)

    def objective(q: AxialPinn) -> jax.Array:
        return loss(eqx.combine(q, static), p, cfg, zeta, that)

    before = float(objective(params))
    opt = optax.lbfgs(memory_size=cfg.lbfgs_history)
    state = opt.init(params)
    value_and_grad = optax.value_and_grad_from_state(objective)

    def step(_: int, carry: tuple) -> tuple:
        q, s = carry
        value, grads = value_and_grad(q, state=s)
        updates, s = opt.update(grads, s, q, value=value, grad=grads, value_fn=objective)
        return optax.apply_updates(q, updates), s

    t0 = time.perf_counter()
    block = checkpoint_every if checkpoint_every > 0 else cfg.iters
    carry: tuple = (params, state)
    done = 0
    while done < cfg.iters:
        n = min(block, cfg.iters - done)
        carry = jax.lax.fori_loop(0, n, step, carry)
        done += n
        if on_checkpoint is not None and done < cfg.iters:
            on_checkpoint(done, eqx.combine(carry[0], static))
        if verbose and block < cfg.iters:
            print(
                f"[{done}/{cfg.iters}] loss {float(objective(carry[0])):.3e} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
    params = carry[0]
    after = float(objective(params))

    if verbose:
        print(
            f"loss {before:.3e} -> {after:.3e} "
            f"in {cfg.iters} iterations on {cfg.points} points, "
            f"{time.perf_counter() - t0:.0f}s"
        )
    if not np.isfinite(after) or after > before:
        # A bad line-search step can only cost time, never accuracy.
        if verbose:
            print("diverged; keeping the initial state")
        return model, p, cfg
    return eqx.combine(params, static), p, cfg


# --- evaluation ----------------------------------------------------------------------
def predict(
    model: AxialPinn, p: AxialParams, cfg: TrainConfig, zeta: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Evaluate the five fields on a ``(zeta, t)`` grid, in physical units."""
    dT = p.P_0 / (p.w_0 * p.c_c)
    zz, tt = np.meshgrid(np.asarray(zeta), np.asarray(t) / horizon(p, cfg), indexing="ij")
    flat_z = jnp.asarray(zz.reshape(-1, 1))
    flat_t = jnp.asarray(tt.reshape(-1, 1))

    def fn(a: jax.Array, b: jax.Array) -> jax.Array:
        return normalised_state(model, p, a, b)

    theta = np.asarray(jax.vmap(fn)(flat_z, flat_t)).reshape(*zz.shape, N_FIELDS)
    temps = [p.T_in + theta[..., k] * dT for k in range(N_TEMPS)]
    return (*temps, theta[..., N_TEMPS])
