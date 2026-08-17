"""The surrogate: a frozen Fourier embedding, a small MLP, and a hard-constraint ansatz.

The whole network is in this file. Read it top to bottom:

1. :class:`FourierEmbedding` maps the two inputs through fixed sinusoids. Without it no
   optimiser tested here forms a boiling front, because a plain MLP learns low-frequency
   structure long before the high-frequency part and the front is the high-frequency part.
2. :class:`AxialPinn` is the trunk: five outputs, one per field.
3. :func:`normalised_state` wraps the trunk in the ansatz that makes the initial and
   boundary conditions exact rather than penalised. This is the part worth reading twice --
   it is why the loss has no constraint terms and therefore no weights to balance.
4. :func:`state_and_grads` supplies the derivatives the residuals need, by forward-mode
   automatic differentiation of the ansatz itself.

Everything is float64. Curvature pairs are meaningless at single-precision residual
magnitudes, and the quasi-Newton solve is the whole method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from pinn_ulof.physics import quasi_steady_void

if TYPE_CHECKING:
    from pinn_ulof.config import TrainConfig
    from pinn_ulof.params import AxialParams

#: Temperatures in the state vector; the fifth field is the void fraction.
N_TEMPS: int = 4
#: Bound on the ansatz exponent, so a diverging iterate cannot overflow to inf.
_EXP_BOUND: float = 4.0


def _bounded_exp(x: jax.Array) -> jax.Array:
    """``exp`` with a smooth ceiling and floor."""
    return jnp.exp(_EXP_BOUND * jnp.tanh(x / _EXP_BOUND))


class FourierEmbedding(eqx.Module):
    """Random Fourier features, ``x -> [sin(2 pi B x), cos(2 pi B x)]``.

    ``B`` is drawn once and held under ``stop_gradient``: this is a change of input
    coordinates, not a fitted layer. Freezing it matters --- the read-out that follows
    grows with the feature count while the trunk behind it does not, so a wider embedding
    adds parameters without adding fitting capacity.
    """

    B: jax.Array

    def __init__(self, n_in: int, n_features: int, scale: float, key: jax.Array) -> None:
        self.B = jax.random.normal(key, (n_in, n_features)) * scale

    def __call__(self, x: jax.Array) -> jax.Array:
        """Map two inputs to ``2 * n_features`` sinusoidal features."""
        proj = 2.0 * jnp.pi * (x @ jax.lax.stop_gradient(self.B))
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)])


class AxialPinn(eqx.Module):
    """Frozen embedding followed by a ``tanh`` multilayer perceptron with five outputs."""

    embed: FourierEmbedding
    mlp: eqx.nn.MLP

    def __init__(self, cfg: TrainConfig, key: jax.Array) -> None:
        k_embed, k_mlp = jax.random.split(key)
        self.embed = FourierEmbedding(2, cfg.fourier_features, cfg.fourier_scale, k_embed)
        self.mlp = eqx.nn.MLP(
            in_size=2 * cfg.fourier_features,
            out_size=N_TEMPS + 1,
            width_size=cfg.width,
            depth=cfg.depth,
            activation=jnp.tanh,
            key=k_mlp,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Raw network output at one point; the ansatz wraps this."""
        return self.mlp(self.embed(x))


# --- the analytic steady state, which is also the exact initial condition -----------
# Closed forms rather than calls into `params`, which is numpy and would break under a
# JAX tracer. Same expressions, differentiable.
def power_shape(p: AxialParams, zeta: jax.Array) -> jax.Array:
    """Axial power shape, normalised so its mean over the channel is one."""
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    norm = (2.0 / (jnp.pi * k)) * jnp.sin(0.5 * jnp.pi * k)
    return jnp.cos(jnp.pi * k * (zeta - 0.5)) / norm


def _power_integral(p: AxialParams, zeta: jax.Array) -> jax.Array:
    """Cumulative axial power fraction: zero at the inlet, one at the outlet."""
    k = 1.0 / (1.0 + 2.0 * p.power_extrap)
    half = 0.5 * jnp.pi * k
    return (jnp.sin(jnp.pi * k * (zeta - 0.5)) + jnp.sin(half)) / (2.0 * jnp.sin(half))


def _fuel_temperature(q: jax.Array, T_cl: jax.Array, area: float, p: AxialParams) -> jax.Array:
    """Invert the gap flux for ``T_f``: conduction plus radiation, by Newton iteration.

    The radiative term makes this implicit, so it is solved rather than rearranged. Five
    iterations from the conduction-only guess are ample at these temperatures.
    """
    sigma = 5.670374419e-8
    T = T_cl + q / (p.h_gap * area)
    for _ in range(5):
        f = p.h_gap * area * (T - T_cl) + sigma * p.emissivity * area * (T**4 - T_cl**4) - q
        df = p.h_gap * area + 4.0 * sigma * p.emissivity * area * T**3
        T = T - f / df
    return T


def theta0(p: AxialParams, zeta: jax.Array) -> jax.Array:
    """Analytic steady profile, in normalised variables."""
    dT = p.P_0 / (p.w_0 * p.c_c)
    T_c = p.T_in + dT * _power_integral(p, zeta)
    q_fuel = (1.0 - p.gamma_c) * p.P_0 * power_shape(p, zeta) / p.H
    T_cl = T_c + q_fuel / (p.h_clad_coolant * 2.0 * jnp.pi * p.r_co)
    T_f = _fuel_temperature(q_fuel, T_cl, 2.0 * jnp.pi * p.r_fo, p)
    cols = [(T - p.T_in) / dT for T in (T_f, T_cl, T_c, T_c)]
    return jnp.concatenate([*cols, jnp.zeros_like(T_c)], axis=-1)


# --- the ansatz --------------------------------------------------------------------
def normalised_state(
    model: AxialPinn, p: AxialParams, zeta: jax.Array, that: jax.Array
) -> jax.Array:
    r"""``theta(zeta, t_hat)`` with every hard constraint satisfied identically.

    The ansatz is multiplicative,

    .. math:: \theta = \theta_0(\zeta)\,\exp\bigl(\hat{t}\,N(\zeta,\hat{t})\bigr),

    and three constraints then hold for *any* weights:

    * ``exp(0) = 1`` makes the initial condition exactly the steady profile;
    * the exponential is positive, so no temperature falls below the inlet --- which
      matters because the Doppler feedback is logarithmic in the fuel temperature and
      undefined for non-positive arguments;
    * ``theta_0`` for the coolant vanishes at ``zeta = 0``, which pins the single upstream
      boundary condition the advection equation admits, with no separate gate.

    The additive form tried first let the optimiser drive the fuel temperature negative
    while the loss fell, at which point the Doppler term returns NaN.

    The void is not a network output. It is closed algebraically on the coolant
    temperature, and because the superheat switch underflows to exactly zero below
    saturation, the void-free initial and inlet conditions fall out of the closure.
    """
    raw = model(jnp.concatenate([zeta, that]))
    base = theta0(p, zeta)
    temps = base[:N_TEMPS] * _bounded_exp(that * raw[:N_TEMPS])
    dT = p.P_0 / (p.w_0 * p.c_c)
    alpha = quasi_steady_void(p.T_in + temps[3:4] * dT, p)
    return jnp.concatenate([temps, alpha])


def state_and_grads(
    model: AxialPinn, p: AxialParams, zeta: jax.Array, that: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(theta, dtheta/dt_hat, dtheta/dzeta)`` at a batch of points.

    Two forward-mode passes for a map from two inputs to five outputs: one ``jvp`` per
    input direction yields all five components at once, which is cheaper than five reverse
    passes.
    """

    def one(z: jax.Array, h: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        def f(a: jax.Array, b: jax.Array) -> jax.Array:
            return normalised_state(model, p, a, b)

        theta, d_dt = jax.jvp(lambda b: f(z, b), (h,), (jnp.ones_like(h),))
        _, d_dz = jax.jvp(lambda a: f(a, h), (z,), (jnp.ones_like(z),))
        return theta, d_dt, d_dz

    return jax.vmap(one)(zeta, that)


def horizon(p: AxialParams, cfg: TrainConfig) -> float:
    """End of the trained window [s]. ``t_hat = 1`` maps here, not to ``p.t_end``."""
    return float(p.t_end) * cfg.t_train_frac
