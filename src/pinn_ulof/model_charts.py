"""Figures that need a trained model, not just the reference.

Separate from :mod:`pinn_ulof.charts`, which draws from the reference solver alone and can
therefore be regenerated from the physics. These need a checkpoint, so they are drawn on
demand from a saved model.

    uv run pinn-ulof model-figures models/p5000_i50000_f64_s2_...eqx

The reference is solved on the scoring mesh, imported from :mod:`pinn_ulof.verification`.
These figures show differences of order a kelvin, and a coarser reference carries errors of
that size itself, so it would draw both and label them as one; see `docs/04-results.md`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from pinn_ulof import sodium
from pinn_ulof.network import horizon
from pinn_ulof.params import AxialParams
from pinn_ulof.reference import solve_reference
from pinn_ulof.train import predict, residuals
from pinn_ulof.verification import RULER_N_AXIAL, RULER_N_OUT

if TYPE_CHECKING:
    from pinn_ulof.config import TrainConfig
    from pinn_ulof.network import AxialPinn

# The reference mesh and time sampling come from `verification`, which is where they are
# defined. These figures resolve differences of order a kelvin; a reference coarse enough to
# carry its own error of that size would draw both and label them as one.

#: Void fraction above which a cell counts as part of the boiling front.
FRONT_LEVEL: float = 0.5


def _front(alpha: np.ndarray, zeta: np.ndarray) -> np.ndarray:
    """Lowest height at which the void exceeds :data:`FRONT_LEVEL`, per time sample."""
    out = np.full(alpha.shape[1], np.nan)
    for j in range(alpha.shape[1]):
        hit = np.where(alpha[:, j] > FRONT_LEVEL)[0]
        if hit.size:
            out[j] = zeta[hit.min()]
    return out


def draw(model: AxialPinn, cfg: TrainConfig, outdir: Path) -> list[Path]:
    """Draw every model-dependent figure into ``outdir``. Returns the paths written."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = AxialParams()
    traj = solve_reference(replace(p, n_axial=RULER_N_AXIAL), n_out=RULER_N_OUT)
    T_boil = sodium.saturation_temperature(p.p_system) + p.dT_superheat

    fields = predict(model, p, cfg, traj.zeta, traj.t)
    T_c_s, alpha_s = fields[3], fields[4]
    n_t = min(T_c_s.shape[1], traj.T_c.shape[1])
    written = []

    # --- fidelity: the surrogate drawn over the reference ---------------------------
    times = [0.0, 5.0, 10.0, float(traj.t[n_t - 1])]
    idx = [int(np.argmin(np.abs(traj.t[:n_t] - t))) for t in times]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for colour, j in zip(plt.cm.viridis(np.linspace(0, 0.85, len(idx))), idx, strict=True):
        ax[0].plot(traj.zeta, traj.T_c[:, j], "-", color=colour, lw=2.2, alpha=0.55)
        ax[0].plot(traj.zeta, T_c_s[:, j], "--", color=colour, lw=1.4)
        ax[1].plot(
            traj.zeta,
            traj.alpha[:, j],
            "-",
            color=colour,
            lw=2.2,
            alpha=0.55,
            label=f"t = {traj.t[j]:.1f} s",
        )
        ax[1].plot(traj.zeta, alpha_s[:, j], "--", color=colour, lw=1.4)
    ax[0].axhline(T_boil, color="k", ls=":", lw=1, label="saturation")
    ax[0].set(xlabel=r"$\zeta = z/H$", ylabel=r"$T_c$  [K]", title="coolant temperature")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[1].set(xlabel=r"$\zeta = z/H$", ylabel=r"$\alpha$", title="void fraction")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"solid: reference ({RULER_N_AXIAL} nodes)    dashed: surrogate", fontsize=9)
    fig.tight_layout()
    written.append(outdir / "surrogate_vs_reference.png")
    fig.savefig(written[-1], dpi=150)
    plt.close(fig)

    # --- where the error is, against where the residual is --------------------------
    err = np.abs(T_c_s[:, :n_t] - traj.T_c[:, :n_t])
    zz, tt = np.meshgrid(traj.zeta, traj.t[:n_t], indexing="ij")
    blocks = residuals(
        model,
        p,
        cfg,
        jnp.asarray(zz.reshape(-1, 1)),
        jnp.asarray((tt / horizon(p, cfg)).reshape(-1, 1)),
    )
    res = np.asarray(sum(b**2 for b in blocks)).reshape(zz.shape) ** 0.5

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    bar = fig.colorbar(ax[0].pcolormesh(tt, zz, err, cmap="magma", shading="auto"), ax=ax[0])
    bar.set_label(r"$|T_c^{\rm PINN}-T_c^{\rm ref}|$  [K]")
    ax[0].set_title("error against the reference")
    bar = fig.colorbar(
        ax[1].pcolormesh(tt, zz, np.log10(res + 1e-30), cmap="viridis", shading="auto"),
        ax=ax[1],
    )
    bar.set_label(r"$\log_{10}\|\mathcal{R}\|$")
    ax[1].set_title("PDE residual the network minimises")
    for a in ax:
        a.set(xlabel="t  [s]", ylabel=r"$\zeta$")
        a.contour(tt, zz, traj.alpha[:, :n_t], levels=[FRONT_LEVEL], colors="w", linewidths=1.1)
    fig.suptitle("white line: reference boiling front", fontsize=9)
    fig.tight_layout()
    written.append(outdir / "error_vs_residual.png")
    fig.savefig(written[-1], dpi=150)
    plt.close(fig)

    # --- the front, in time ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(
        traj.t[:n_t],
        _front(traj.alpha[:, :n_t], traj.zeta),
        "-",
        lw=2.2,
        alpha=0.6,
        label="reference",
    )
    ax.plot(traj.t[:n_t], _front(alpha_s[:, :n_t], traj.zeta), "--", lw=1.5, label="surrogate")
    ax.set(xlabel="t  [s]", ylabel=r"front height  $\zeta$", title="boiling front position")
    ax.set_xlim(10.5, float(traj.t[n_t - 1]))
    ax.legend(fontsize=9)
    fig.tight_layout()
    written.append(outdir / "front_position.png")
    fig.savefig(written[-1], dpi=150)
    plt.close(fig)
    return written
