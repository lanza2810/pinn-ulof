"""Command line: run the reference, train the surrogate, draw the figures.

    pinn-ulof reference                 # the ground truth, on its own
    pinn-ulof train                     # the shipped run: 5000 points, 50 000 iterations
    pinn-ulof train --points 3000 --iters 30000 --fourier 128 --memory 100
    pinn-ulof score models/p5000_i50000_f64_s0_20260813122657.eqx   # re-score a saved run
    pinn-ulof figures                   # every figure in the paper
    pinn-ulof model-figures MODEL       # the figures that need a trained model

``train`` saves the fitted model before scoring it, into ``models/`` (gitignored). That
separation is deliberate: training costs hours and scoring costs minutes, so any later
question about a finished run --- a new metric, a different reference mesh, a figure ---
is answered with ``score`` instead of by training again.

``train`` is the published configuration by default. Three flags are the axes the paper
sweeps --- ``--points``, ``--iters`` and ``--fourier``. ``--memory`` and ``--seed``
reproduce the remaining ablations. Nothing else is exposed, because nothing else was
varied in the published result.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from pinn_ulof.config import TrainConfig
from pinn_ulof.params import AxialParams
from pinn_ulof.verification import MESHES as _V_MESHES
from pinn_ulof.verification import RULER_N_AXIAL, RULER_N_OUT
from pinn_ulof.verification import RULER_UNCERTAINTY as _RU

if TYPE_CHECKING:
    from pinn_ulof.network import AxialPinn


def _score(model: AxialPinn, p: AxialParams, cfg: TrainConfig) -> None:
    """Score a trained surrogate against the reference and print the comparison."""
    from pinn_ulof.reference import solve_reference
    from pinn_ulof.scoring import relative_l2
    from pinn_ulof.train import predict

    traj = solve_reference(replace(p, n_axial=RULER_N_AXIAL), n_out=RULER_N_OUT)
    # `relative_l2` returns the field errors and the front metrics together, so one call
    # gives every published quantity.
    m = relative_l2(predict(model, p, cfg, traj.zeta, traj.t), traj, p)

    print(f"\nrelative L2 against the reference   (reference's own: {_RU['fields']:.1e})")
    for name in ("T_f", "T_cl", "T_s", "T_c"):
        print(f"  {name:5s} {m[name]:.3e}")
    print("\nboiling front")
    print(
        f"  onset time error     {m['onset_t_err_tan_s']:.4f} s"
        f"       (reference's own: {_RU['onset']:.5f} s)"
    )
    print(
        f"  peak voided length   {m['L_void_max']:.4f} m"
        f"      (reference {m['L_void_max_ref']:.4f} m,"
        f" {100 * m['L_void_max'] / m['L_void_max_ref']:.1f}%)"
    )
    print(
        f"  saturation margin    {m['margin_K']:+.2f} K      (reference {m['margin_K_ref']:+.2f} K)"
    )


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911 - one return per sub-command
    """Entry point. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    ref = sub.add_parser("reference", help="solve the reference and report its end state")
    # The scoring mesh, imported from `verification` rather than restated here. Every
    # published accuracy number is quoted against it, so this is the mesh that reproduces
    # them: at 160 nodes the peak voided length is 0.3812 m against this mesh's 0.3785 m.
    ref.add_argument(
        "--n-axial",
        type=int,
        default=RULER_N_AXIAL,
        help=f"axial nodes; default {RULER_N_AXIAL} is the mesh every published score is "
        f"quoted against",
    )

    # An instance, not the class: `slots=True` makes `TrainConfig.points` a slot
    # descriptor rather than the default value, which argparse would pass through.
    d = TrainConfig()
    tr = sub.add_parser("train", help="train the surrogate and score it")
    tr.add_argument(
        "--points", type=int, default=d.points, help="size of the fixed collocation set"
    )
    tr.add_argument("--iters", type=int, default=d.iters, help="quasi-Newton iterations")
    tr.add_argument(
        "--fourier",
        type=int,
        default=d.fourier_features,
        help="random Fourier features in the input embedding",
    )
    tr.add_argument("--seed", type=int, default=d.seed)
    tr.add_argument(
        "--memory",
        type=int,
        default=d.lbfgs_history,
        help="quasi-Newton curvature pairs retained (the L-BFGS memory)",
    )
    tr.add_argument(
        "--out",
        default=None,
        help="where to write the trained model; default names it after the run",
    )
    tr.add_argument(
        "--checkpoint-every",
        type=int,
        default=10000,
        metavar="N",
        help="save the model every N iterations so a long run can be stopped (0 to disable)",
    )

    sc = sub.add_parser("score", help="score a saved model against the reference")
    sc.add_argument("model", help="path written by `train`")

    fig = sub.add_parser("figures", help="draw every figure in the paper")
    fig.add_argument("--outdir", default="figures")

    vf = sub.add_parser("verify", help="grid-convergence uncertainty of the reference")
    vf.add_argument(
        "--meshes",
        default=",".join(str(n) for n in _V_MESHES),
        help="comma-separated axial node counts, coarse to fine, each double the last",
    )

    tb = sub.add_parser("table", help="score saved models and print the published tables")
    tb.add_argument("models", nargs="+", help="paths written by `train`")

    ld = sub.add_parser("ladder", help="score checkpoints across budgets into a data file")
    ld.add_argument("models", nargs="+", help="paths written by `train`, any budget or size")
    ld.add_argument("--out", default="paper/data/ladder.json")

    lr = sub.add_parser("ladder-rows", help="emit or check the appendix ladder tables")
    lr.add_argument("--data", default="paper/data/ladder.json")
    lr.add_argument("--points", type=int, help="print the LaTeX body rows for this count")
    lr.add_argument("--check", default=None, help="verify every row appears in this .tex")

    mf = sub.add_parser("model-figures", help="draw the figures that need a trained model")
    mf.add_argument("model", help="path written by `train`")
    mf.add_argument("--outdir", default="figures/model")

    args = ap.parse_args(argv)

    if args.command == "reference":
        from pinn_ulof.reference import solve_reference

        p = AxialParams(n_axial=args.n_axial)
        traj = solve_reference(p, n_out=RULER_N_OUT)
        print(f"solved to t = {traj.t[-1]:.3f} s on {args.n_axial} nodes")
        print(f"  peak coolant temperature  {traj.T_c.max():.1f} K")
        print(f"  peak voided length        {traj.voided_length.max():.4f} m")
        return 0

    if args.command == "train":
        from pinn_ulof import checkpoint
        from pinn_ulof.train import train

        cfg = replace(
            TrainConfig(),
            points=args.points,
            iters=args.iters,
            fourier_features=args.fourier,
            seed=args.seed,
            lbfgs_history=args.memory,
        )
        # One stamp for the whole run, so intermediate checkpoints and the final model
        # group together and sort by the budget they were taken at.
        stamp = checkpoint.run_stamp()

        def _intermediate(done: int, m: AxialPinn) -> None:
            at = checkpoint.save(
                checkpoint.default_path(replace(cfg, iters=done), stamp),
                m,
                replace(cfg, iters=done),
            )
            print(f"saved {at}", flush=True)

        model, p, cfg = train(
            AxialParams(),
            cfg,
            on_checkpoint=_intermediate if args.checkpoint_every > 0 else None,
            checkpoint_every=args.checkpoint_every,
        )
        # Saved before scoring, and unconditionally: training is hours and scoring is
        # minutes, so a fault in the scorer must not be able to destroy the run.
        out = checkpoint.save(args.out or checkpoint.default_path(cfg, stamp), model, cfg)
        print(f"saved {out}")
        _score(model, p, cfg)
        return 0

    if args.command == "score":
        from pinn_ulof import checkpoint

        model, cfg, saved = checkpoint.load(args.model)
        print(
            f"loaded {args.model}\n"
            f"  trained {saved}: {cfg.points} points, {cfg.iters} iterations, "
            f"{cfg.fourier_features} features, seed {cfg.seed}"
        )
        _score(model, AxialParams(), cfg)
        return 0

    if args.command == "verify":
        from pinn_ulof.verification import report

        report(tuple(int(x) for x in args.meshes.split(",")))
        return 0

    if args.command == "table":
        from pinn_ulof.verification import table

        table(args.models)
        return 0

    if args.command == "ladder":
        from pinn_ulof.verification import ladder

        ladder(args.models, Path(args.out))
        return 0

    if args.command == "ladder-rows":
        import json as _json

        from pinn_ulof.verification import check_tables, latex_rows

        if args.check:
            return 1 if check_tables(Path(args.check), Path(args.data)) else 0
        data = _json.loads(Path(args.data).read_text())
        print(latex_rows(data, args.points))
        return 0

    if args.command == "model-figures":
        from pinn_ulof import checkpoint, model_charts

        model, cfg, saved = checkpoint.load(args.model)
        print(f"loaded {args.model} (trained {saved})")
        for path in model_charts.draw(model, cfg, Path(args.outdir)):
            print(f"wrote {path}")
        return 0

    from pinn_ulof.charts import generate_all

    for path in generate_all(Path(args.outdir)):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
