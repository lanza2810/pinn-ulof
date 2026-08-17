"""Save and reload a trained surrogate.

A run costs about three hours of CPU; scoring it costs about three minutes, nearly all of
that re-solving the 2560-node reference. Saving the model decouples the two, so a metric
added, a figure drawn or a comparison against a different reference mesh costs minutes
rather than another run, and a fault in the scorer cannot destroy the training.

The file holds the configuration and the weights together, because weights alone are not a
model: reconstructing the skeleton needs the embedding width, the trunk shape and the
horizon, and inferring those from array shapes is guesswork that fails silently. The first
line is JSON, the rest is the serialised tree.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp

from pinn_ulof.config import TrainConfig
from pinn_ulof.network import AxialPinn, normalised_state

if TYPE_CHECKING:
    from pinn_ulof.params import AxialParams

#: Where `train` writes by default. Gitignored: a checkpoint is a build product.
DEFAULT_DIR = Path("models")
#: Grid resolution used by :func:`matches` to compare two models' fields.
_MATCH_POINTS: int = 17


def run_stamp() -> str:
    """Return a unique run id: ``yyyymmddhhmmss`` in UTC, then eight random hex digits.

    The timestamp alone is not an identifier: two runs launched in the same second share it,
    and `default_path` encodes only points, iterations, features and seed, so arms differing
    in any other knob would contend for one filename and the later writer would win. The
    timestamp keeps names sortable; the random suffix makes them unique whatever two runs
    have in common.
    """
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]


def default_path(cfg: TrainConfig, stamp: str | None = None) -> Path:
    """Name a checkpoint after the run that produced it, and when it was produced.

    The configuration alone does not identify a run: the same configuration re-run after a
    code change produces a different model, and two arms differing only in a knob the name
    omits are indistinguishable. `run_stamp` supplies a unique id per run, so no two runs
    can ever contend for a filename whatever they have in common.
    """
    stamp = stamp or run_stamp()
    return DEFAULT_DIR / (
        f"p{cfg.points}_i{cfg.iters}_f{cfg.fourier_features}_s{cfg.seed}_{stamp}.eqx"
    )


def save(path: Path, model: AxialPinn, cfg: TrainConfig) -> Path:
    """Write ``model`` and ``cfg`` to ``path``, stamped with the time. Returns the path.

    The timestamp is stored inside the file as well as in the name, so a renamed or copied
    checkpoint still identifies the run that produced it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"saved_utc": datetime.now(UTC).isoformat(timespec="seconds"), "config": asdict(cfg)}
    # Write to a sibling and rename. `Path.replace` is atomic within a filesystem, so the
    # destination is either absent or a complete checkpoint, never a half-written one under
    # a name that promises a finished run. JAX dispatches asynchronously, so this function
    # is reached before the iterations it saves have run and `tree_serialise_leaves` blocks
    # on the device only when it needs the values: writing in place would leave a zero-byte
    # file for the length of a whole block.
    #
    # The temp name carries the pid so concurrent writers to one destination each get their
    # own. With a fixed suffix they would share a file, the first rename would take it, and
    # the second would fail with FileNotFoundError.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    try:
        with tmp.open("wb") as f:
            f.write((json.dumps(header) + "\n").encode())
            eqx.tree_serialise_leaves(f, model)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def load(path: Path) -> tuple[AxialPinn, TrainConfig, str]:
    """Read back a model, its configuration and the time it was saved.

    The skeleton is rebuilt from the stored configuration and then filled, which is what
    makes the round trip exact: `equinox` deserialises into a tree of matching shapes, so a
    configuration that disagrees with the weights raises rather than loading something
    subtly wrong.
    """
    with Path(path).open("rb") as f:
        header = json.loads(f.readline().decode())
        # Ignore keys a newer file carries and this version does not know about, rather
        # than failing on them; a missing key is a real error and still raises.
        known = {fld.name for fld in fields(TrainConfig)}
        cfg = TrainConfig(**{k: v for k, v in header["config"].items() if k in known})
        skeleton = AxialPinn(cfg, jax.random.PRNGKey(cfg.seed))
        return eqx.tree_deserialise_leaves(f, skeleton), cfg, header["saved_utc"]


def matches(model: AxialPinn, p: AxialParams, other: AxialPinn) -> bool:
    """Report whether two models agree pointwise, which is what a round trip must preserve.

    Compares the fields rather than the weights: two parameter sets that differ but map every
    input to the same output are the same model, and a serialisation bug that permuted a
    layer would pass an array-by-array comparison of the leaves it permuted.
    """
    grid = jnp.linspace(0.0, 1.0, _MATCH_POINTS).reshape(-1, 1)

    def fields(m: AxialPinn) -> jax.Array:
        return jax.vmap(lambda a, b: normalised_state(m, p, a, b))(grid, grid)

    return bool(jnp.array_equal(fields(model), fields(other)))
