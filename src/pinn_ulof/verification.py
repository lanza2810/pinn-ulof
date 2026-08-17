"""Numerical uncertainty of the reference solution, by grid convergence.

The reference is the ruler every accuracy claim is quoted against, so its own error has to
be quantified before any of those claims mean anything. It is a finite-volume
discretisation, so that error is set by the axial mesh and is estimated the standard way:
solve on a sequence of meshes, observe the order of convergence, and extrapolate.

    uv run pinn-ulof verify
    uv run pinn-ulof table models/*.eqx

`verify` produces the reference's uncertainty; `table` scores saved models against the
shipped reference and prints the published comparison. Every number in
`docs/04-results.md` and in the paper's tables is the output of one of the two. Nothing is
transcribed by hand.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pinn_ulof import checkpoint, sodium
from pinn_ulof.params import AxialParams
from pinn_ulof.reference import AxialTrajectory, energy_balance, solve_reference
from pinn_ulof.scoring import onset_by_tangency, relative_l2
from pinn_ulof.train import predict

#: Meshes solved by default. Three is the minimum that lets the order be *observed* rather
#: than assumed, and the extra rungs confirm the order holds over more than one doubling.
MESHES: tuple[int, ...] = (640, 1280, 2560, 5120)
#: Time samples. The solve terminates at the validity horizon, so only the first ~67 land.
N_OUT: int = 241
#: Scalars whose uncertainty is reported, and the unit each is quoted in.
SCALARS: tuple[tuple[str, str], ...] = (
    ("onset", "s"),
    ("zeta", ""),
    ("Lvoid", "m"),
    ("margin", "K"),
)
#: Fields whose relative L2 uncertainty is reported.
FIELDS: tuple[str, ...] = ("T_f", "T_cl", "T_s", "T_c")

#: Axial nodes and time samples of the reference used for scoring.
#:
#: Chosen by what the measurement needs. The ratio of model error to reference uncertainty
#: must exceed four for an accuracy claim to be supportable; on the film temperature it is
#: 4.2 at 2560 nodes, 3.1 at 640 and 1.9 at 320. Solving at 2560 costs 112 s against a run
#: that takes an hour and a half, so cost is not the constraint.
RULER_N_AXIAL: int = 2560
RULER_N_OUT: int = 241

#: The shipped reference's own uncertainty, as printed by `report()` for the scoring mesh.
#: Quoted beside every score so an error below these is recognisable as a measurement of the
#: reference rather than of the model. Re-run `pinn-ulof verify` and update these together
#: with `cli._RULER_N_AXIAL`; they are the one pair of numbers in the package that a mesh
#: change invalidates.
RULER_UNCERTAINTY: dict[str, float] = {
    "fields": 9.7e-5,
    "onset": 0.00064,
    "zeta": 0.0002,
    "Lvoid_frac": 0.00049,
    "margin": 0.036,
}


def solve(n_axial: int, p: AxialParams | None = None) -> tuple[AxialTrajectory, float]:
    """Solve the reference on ``n_axial`` nodes. Returns the trajectory and the seconds."""
    p = p or AxialParams()
    t0 = time.perf_counter()
    traj = solve_reference(replace(p, n_axial=n_axial), n_out=N_OUT)
    return traj, time.perf_counter() - t0


def scalars(traj: AxialTrajectory, p: AxialParams) -> dict[str, float]:
    """Return the four scalars whose convergence is tracked."""
    T_boil = float(sodium.saturation_temperature(p.p_system)) + p.dT_superheat
    t_on, z_on = onset_by_tangency(traj.T_c, traj.zeta, traj.t, T_boil)
    peak = float(traj.T_c.max())
    return {
        "onset": float(t_on),
        "zeta": float(z_on),
        "Lvoid": float(traj.voided_length.max()),
        "margin": float(peak - T_boil),
    }


def richardson(coarse: float, medium: float, fine: float) -> tuple[float, float]:
    """Observed order and extrapolated limit from three values on doubling meshes.

    ``f_exact ~ f_h + (f_h - f_2h) / (2^p - 1)``. Returns ``(nan, nan)`` when the three
    values are not monotone, which means the quantity is not in the asymptotic range and
    extrapolating it would invent a number.
    """
    d1, d2 = medium - coarse, fine - medium
    if d2 == 0.0 or d1 / d2 <= 0.0:
        return float("nan"), float("nan")
    order = float(np.log2(abs(d1 / d2)))
    return order, float(fine + d2 / (2**order - 1))


def field_l2(coarse: AxialTrajectory, fine: AxialTrajectory) -> dict[str, float]:
    """Relative L2 of a coarse solution against a fine one, sampled on the fine nodes."""
    n_t = min(len(coarse.t), len(fine.t))
    out = {}
    for name in FIELDS:
        a, b = getattr(coarse, name), getattr(fine, name)
        interp = np.empty((len(fine.zeta), n_t))
        for j in range(n_t):
            interp[:, j] = np.interp(fine.zeta, coarse.zeta, a[:, j])
        out[name] = float(np.linalg.norm(interp - b[:, :n_t]) / np.linalg.norm(b[:, :n_t]))
    return out


def report(meshes: tuple[int, ...] = MESHES) -> dict[int, dict[str, float]]:
    """Solve the sequence, print the convergence tables, and return the uncertainties.

    The returned mapping is ``{mesh: {quantity: uncertainty}}``. Field uncertainties are
    twice the gap to the next mesh, which is the first-order Richardson estimate; scalar
    uncertainties are the distance to the extrapolated limit.

    At least three meshes are required: the observed order of convergence is measured from
    three successive values, not assumed.
    """
    if len(meshes) < 3:
        msg = f"need at least three meshes to observe the order, got {list(meshes)}"
        raise ValueError(msg)
    p = AxialParams()
    runs = {}
    header = " ".join(f"{k:>13s}" for k, _ in SCALARS)
    print(f"{'mesh':>6s} {'solve':>9s} {header} {'energy bal.':>13s}")
    for n in meshes:
        traj, sec = solve(n, p)
        runs[n] = traj
        s = scalars(traj, p)
        # Summing the four nodal energy equations over z cancels every internal flux and
        # telescopes the upwind advection, so this closes only if the areas, capacities and
        # stencil are all right. It tests the discretisation, not the integrator.
        cells = " ".join(f"{s[k]:13.6f}" for k, _ in SCALARS)
        # `replace(p, n_axial=n)`, not `p`: the geometry must come from the mesh the
        # trajectory was solved on. Pairing a 40-node geometry with a 160-node solution
        # turns a closure of 5e-5 into 0.5.
        closure = energy_balance(traj, replace(p, n_axial=n))
        print(f"{n:6d} {sec:8.1f}s {cells} {closure:13.2e}", flush=True)

    vals = {n: scalars(runs[n], p) for n in meshes}
    print("\nobserved order and extrapolated limit, from the finest three meshes:")
    limits = {}
    for key, unit in SCALARS:
        order, limit = richardson(*(vals[n][key] for n in meshes[-3:]))
        limits[key] = limit
        note = "" if np.isfinite(order) else "   (not monotone; not extrapolated)"
        print(f"  {key:7s} p={order:5.2f}  limit={limit:.6f} {unit}{note}")

    print("\nuncertainty of each mesh:")
    print(
        f"{'mesh':>6s} "
        + " ".join(f"{k:>13s}" for k, _ in SCALARS)
        + "   "
        + " ".join(f"{f:>11s}" for f in FIELDS)
    )
    out: dict[int, dict[str, float]] = {}
    finest = meshes[-1]
    for n in meshes:
        row = {k: abs(vals[n][k] - limits[k]) for k, _ in SCALARS}
        # First-order Richardson on the field norms: the error is twice the gap to the
        # next mesh. The finest mesh has no successor, so it has no estimate here.
        nxt = meshes[meshes.index(n) + 1] if n != finest else None
        row.update({f: 2.0 * field_l2(runs[n], runs[nxt])[f] for f in FIELDS} if nxt else {})
        out[n] = row
        cells = " ".join(f"{row[k]:13.6f}" for k, _ in SCALARS)
        fields = " ".join(f"{row[f]:11.3e}" if f in row else f"{'--':>11s}" for f in FIELDS)
        print(f"{n:6d} {cells}   {fields}")
    return out


def table(paths: list[str]) -> list[dict[str, float]]:
    """Score saved models against the shipped reference and print the published tables.

    One row per model plus the mean and half-range, which is what the paper reports: a
    spread over seeds is the only honest uncertainty for a quantity whose value depends on
    an initialisation.
    """
    p = AxialParams()
    print(f"reference: {RULER_N_AXIAL} axial nodes, {RULER_N_OUT} time samples", flush=True)
    traj = solve_reference(replace(p, n_axial=RULER_N_AXIAL), n_out=RULER_N_OUT)
    rows = []
    for path in paths:
        model, cfg, _ = checkpoint.load(Path(path))
        rows.append(relative_l2(predict(model, p, cfg, traj.zeta, traj.t), traj, p))

    ref = rows[0]
    print(
        f"\nreference   onset {ref['onset_t_tan'] - ref['onset_t_err_tan_s']:.4f} s"
        f"   zeta* {ref['onset_zeta_tan']:.4f}"
        f"   L_void {ref['L_void_max_ref']:.4f} m"
        f"   margin {ref['margin_K_ref']:+.2f} K"
    )
    quantities = (
        ("T_f", "{:.3e}"),
        ("T_cl", "{:.3e}"),
        ("T_s", "{:.3e}"),
        ("T_c", "{:.3e}"),
        ("onset_t_tan", "{:.4f}"),
        ("onset_t_err_tan_s", "{:.4f}"),
        ("onset_zeta_tan", "{:.4f}"),
        ("L_void_max", "{:.4f}"),
        ("max_T_c", "{:.1f}"),
        ("margin_K", "{:+.2f}"),
    )
    width = max(len(k) for k, _ in quantities)
    header = " ".join(f"{Path(x).stem[-12:]:>13s}" for x in paths)
    print(f"\n{'quantity':{width}s} {header}   {'mean +/- half-range':>24s}")
    for key, fmt in quantities:
        vals = [r[key] for r in rows]
        mean, half = (min(vals) + max(vals)) / 2, (max(vals) - min(vals)) / 2
        cells = " ".join(f"{fmt.format(v):>13s}" for v in vals)
        print(f"{key:{width}s} {cells}   {fmt.format(mean):>11s} +/- {fmt.format(half)}")
    return rows


#: Metrics carried by the ladder, and the ``RULER_UNCERTAINTY`` key each is normalised by.
#: Every entry is an error against the reference divided by the reference's own uncertainty
#: in the same units, so one axis carries all seven and the value four is the point below
#: which a difference is no longer resolvable.
LADDER_METRICS: tuple[tuple[str, str], ...] = (
    ("T_f", "fields"),
    ("T_cl", "fields"),
    ("T_s", "fields"),
    ("T_c", "fields"),
    ("onset", "onset"),
    ("Lvoid", "Lvoid_frac"),
    ("margin", "margin"),
)


#: Quantities carried alongside the errors, in physical units, because the published tables
#: quote the value and not only the distance from the reference.
LADDER_VALUES: tuple[str, ...] = ("onset_t", "L_void_m", "margin_K")


def _ladder_values(m: dict[str, float]) -> dict[str, float]:
    """Return the front quantities in physical units, as the tables quote them."""
    return {
        "onset_t": m["onset_t_tan"],
        "L_void_m": m["L_void_max"],
        "margin_K": m["margin_K"],
    }


def _ladder_errors(m: dict[str, float]) -> dict[str, float]:
    """Error against the reference for each ladder metric, in the ruler's own units."""
    return {
        "T_f": m["T_f"],
        "T_cl": m["T_cl"],
        "T_s": m["T_s"],
        "T_c": m["T_c"],
        "onset": abs(m["onset_t_err_tan_s"]),
        "Lvoid": abs(m["L_void_max"] - m["L_void_max_ref"]) / m["L_void_max_ref"],
        "margin": abs(m["margin_K"] - m["margin_K_ref"]),
    }


def ladder(paths: list[str], out: Path) -> dict[str, Any]:
    """Score checkpoints, group them by ``(points, iters)``, and write the ladder.

    One reference solve serves every checkpoint. Rows are keyed by the configuration read
    back from each file rather than from its name, so a mis-named checkpoint groups by what
    it actually is. Each row carries the mean and half-range over the seeds present.

        uv run pinn-ulof ladder models/p10000_i*.eqx --out paper/data/ladder.json
    """
    p = AxialParams()
    print(f"reference: {RULER_N_AXIAL} axial nodes, {RULER_N_OUT} time samples", flush=True)
    traj = solve_reference(replace(p, n_axial=RULER_N_AXIAL), n_out=RULER_N_OUT)

    rows: dict[tuple[int, int], list[dict[str, float]]] = {}
    traj_ref: dict[str, float] = {}
    for path in paths:
        model, cfg, _ = checkpoint.load(Path(path))
        m = relative_l2(predict(model, p, cfg, traj.zeta, traj.t), traj, p)
        rows.setdefault((cfg.points, cfg.iters), []).append(_ladder_errors(m) | _ladder_values(m))
        traj_ref = {
            "onset_t": m["onset_t_tan"] - m["onset_t_err_tan_s"],
            "L_void_m": m["L_void_max_ref"],
            "margin_K": m["margin_K_ref"],
        }
        print(f"  scored {Path(path).name}", flush=True)

    arms = []
    for (points, iters), seeds in sorted(rows.items()):
        arm: dict[str, Any] = {"points": points, "iters": iters, "seeds": len(seeds)}
        for name in [k for k, _ in LADDER_METRICS] + list(LADDER_VALUES):
            vals = [s[name] for s in seeds]
            arm[name] = {
                "mean": (min(vals) + max(vals)) / 2.0,
                "half": (max(vals) - min(vals)) / 2.0,
            }
        arms.append(arm)

    ref = {
        "onset_t": traj_ref["onset_t"],
        "L_void_m": traj_ref["L_void_m"],
        "margin_K": traj_ref["margin_K"],
    }
    data = {
        "ruler": dict(RULER_UNCERTAINTY),
        "n_axial": RULER_N_AXIAL,
        "reference": ref,
        "arms": arms,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"\nwrote {out}: {len(arms)} arms over {sum(a['seeds'] for a in arms)} checkpoints")
    return data


#: Column format of the appendix ladder tables: metric, scale, and decimals.
_LADDER_COLUMNS: tuple[tuple[str, float, int], ...] = (
    ("T_f", 1e4, 2),
    ("T_cl", 1e4, 2),
    ("T_s", 1e4, 2),
    ("T_c", 1e4, 2),
    ("onset", 1.0, 4),
    ("L_void_m", 1.0, 4),
    ("margin_K", 1.0, 2),
)


def latex_rows(data: dict[str, Any], points: int) -> str:
    """Return the body rows of one appendix ladder table, as LaTeX.

    Generated rather than transcribed. Copying ninety numbers by hand put two of them one
    unit out in the last digit, which is a defect no amount of proofreading finds reliably.
    """
    out = []
    for arm in sorted((a for a in data["arms"] if a["points"] == points), key=lambda a: a["iters"]):
        cells = []
        for key, scale, dp in _LADDER_COLUMNS:
            sign = "+" if key == "margin_K" else ""
            m, h = arm[key]["mean"] * scale, arm[key]["half"] * scale
            cells.append(f"${m:{sign}.{dp}f} \\pm {h:.{dp}f}$")
        out.append(f"{arm['iters']:,}".replace(",", "\\,") + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def check_tables(tex: Path, data: Path) -> int:
    """Verify every ladder value in ``data`` appears in ``tex``. Returns a count of misses.

    The appendix tables are the data file rendered; if a value is absent, one of the two has
    moved without the other.
    """
    body = Path(tex).read_text()
    d = json.loads(Path(data).read_text())
    missing = []
    for points in sorted({a["points"] for a in d["arms"]}):
        missing.extend(line for line in latex_rows(d, points).splitlines() if line not in body)
    for line in missing:
        print(f"::error::row absent from {tex}: {line}")
    print(f"{len(missing)} ladder rows absent from {tex}")
    return len(missing)
