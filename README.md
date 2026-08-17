# A neural surrogate for SFR sodium boiling — companion code

Companion to the paper in [`paper/`](paper/). One model, one optimiser, one published
configuration, so the code can be read alongside the paper rather than searched.

**Start with [`docs/`](docs/)** — it explains the reactor transient, what a
physics-informed neural network is, how this one is built and why, a tour of the code, and
what it does and does not achieve. No machine-learning or reactor background assumed.

## Running it

```bash
uv sync

uv run pinn-ulof reference     # the ground truth on its own, ~2 min
uv run pinn-ulof train         # the published run: saved, then scored
uv run pinn-ulof score MODEL   # re-score a saved run, ~3 min
uv run pinn-ulof figures       # every figure in the paper, into figures/
```

`train` is the published configuration by default — 10 000 collocation points, 50 000
quasi-Newton iterations, 64 Fourier features — and takes about an hour and a half on eight
CPU cores. Three flags expose the axes the paper sweeps:

```bash
uv run pinn-ulof train --points 3000 --iters 30000 --fourier 128
```

Three more reproduce the ablations reported in [`docs/04-results.md`](docs/04-results.md):
`--memory` (the L-BFGS curvature pairs) and `--seed`. Nothing else is exposed, because
nothing else was varied to produce the published result.
Every default is measured rather than inherited; `config.py` gives the reason beside each
value, and [`docs/02-model.md`](docs/02-model.md) gives it at length.

## Layout

`physics.py` holds the governing equations and is called by **both** the reference solver
and the network, so the two solve the same equations by construction rather than by review.
`params.py` holds every constant, `reference.py` the stiff Radau ground truth, `network.py`
the embedding and ansatz, `train.py` the residuals and the solve. The rest is properties,
scoring, figures and the command line — [`docs/03-code.md`](docs/03-code.md) is the tour.

The physics is checked by the shipped run agreeing with an independently discretised
reference solver to that solver's own numerical accuracy. `ruff` and `ty` run under
`pre-commit`.

## Paper

```bash
uv run pinn-ulof figures                  # generated, never committed
latexmk -pdf -outdir=build paper/paper.tex
```

The `paper` workflow runs exactly that on every push touching `paper/` or `src/` and
publishes `paper.pdf` as a downloadable artefact, so the built PDF is available from the
Actions tab without a local TeX installation.

## License

MIT.
