# The code

Fourteen files, about 3700 lines. Nothing is generated or vendored.

## Commands

```bash
uv sync

uv run pinn-ulof reference          # the ground truth on the shipped mesh, ~2 min
uv run pinn-ulof train              # the published run: saved, then scored
uv run pinn-ulof score MODEL        # re-score a saved run, ~3 min
uv run pinn-ulof verify             # grid-convergence uncertainty of the reference
uv run pinn-ulof table MODELS...    # the published comparison table
uv run pinn-ulof ladder MODELS...   # score across budgets -> paper/data/ladder.json
uv run pinn-ulof figures            # the eleven figures drawn from the physics
uv run pinn-ulof model-figures MODEL  # the three that need a trained network
```

`train` defaults to the published configuration — 10 000 collocation points, 50 000
quasi-Newton iterations, 64 Fourier features, double precision — and takes about an hour
and a half on eight CPU cores. It saves the fitted model into `models/` (gitignored, named
for the run and stamped with the UTC time) before scoring it, so a fault in the scorer
cannot destroy the run, and every later question about a finished model is answered by
`score` in about three minutes.

`--checkpoint-every` (default 10 000) saves intermediate models, so a long run can be
stopped. The split does not change the optimisation: parameters and optimiser state both
carry across the boundary, so the update sequence is the one an unsplit loop produces.

Three flags expose the axes the paper sweeps:

```bash
uv run pinn-ulof train --points 3000 --iters 30000 --fourier 128
```

`--memory` and `--seed` reproduce the remaining ablations in
[`04-results.md`](04-results.md).

### Threads

Thread count changes the order of floating-point reductions, so it changes answers, not
only timings. `OMP_NUM_THREADS` does not bind the XLA CPU backend, which sizes its pool from
the machine's core count; CPU affinity does:

```bash
taskset -c 0-7 uv run pinn-ulof train --seed 0
```

## Files

| file | lines | contents |
|---|---:|---|
| [`params.py`](../src/pinn_ulof/params.py) | 403 | every physical constant, with value, units and source |
| [`sodium.py`](../src/pinn_ulof/sodium.py) | 233 | the sodium property correlations |
| [`physics.py`](../src/pinn_ulof/physics.py) | 782 | the governing equations |
| [`reference.py`](../src/pinn_ulof/reference.py) | 453 | the stiff Radau reference solution |
| [`config.py`](../src/pinn_ulof/config.py) | 81 | one frozen dataclass; the whole configuration surface |
| [`network.py`](../src/pinn_ulof/network.py) | 187 | embedding, trunk, ansatz, derivatives |
| [`train.py`](../src/pinn_ulof/train.py) | 246 | collocation, residuals, loss, quasi-Newton solve |
| [`scoring.py`](../src/pinn_ulof/scoring.py) | 197 | the comparison against the reference |
| [`checkpoint.py`](../src/pinn_ulof/checkpoint.py) | 140 | saving and reloading a trained model |
| [`charts.py`](../src/pinn_ulof/charts.py) | 518 | the eleven figures drawn from the reference |
| [`model_charts.py`](../src/pinn_ulof/model_charts.py) | 147 | the three figures that need a network |
| [`cli.py`](../src/pinn_ulof/cli.py) | 226 | the seven sub-commands |
| [`verification.py`](../src/pinn_ulof/verification.py) | 206 | grid convergence, and the published tables |
| [`_backend.py`](../src/pinn_ulof/_backend.py) | 31 | numpy/JAX dispatch |

`physics.continuous_derivatives` is called by both the reference solver and the network's
residual, with the same arguments. The two therefore solve the same equations by
construction. `_backend.py` makes that possible: it dispatches on argument type, so one
transcription of each correlation serves numpy arrays and JAX tracers alike.

`config.py` states the measurement behind each default beside the value.

`scoring.py` defines boiling onset by **tangency** of the level set. Reading onset off a
threshold crossing asks a locally flat function where it crosses a value, and a field error
$\epsilon$ displaces the crossing by $\sqrt{2\epsilon/\kappa}$. The stationarity condition
is linear in the same error and divided by a curvature of 1066 K per unit $\zeta$ squared.

## The scoring mesh

`verification.py` fixes the scoring reference at **2560 axial nodes and 241 time samples**,
in `RULER_N_AXIAL` and `RULER_N_OUT`, beside the `RULER_UNCERTAINTY` derived for that mesh
and the code that derives it. `cli.py` imports them.

The mesh is chosen by what the measurement needs, not by cost. `verification.py` computes
the reference's own uncertainty by grid convergence, and the ratio of model error to that
uncertainty must exceed four for an accuracy claim to be supportable. On the film
temperature the ratio is 4.2 at 2560, 3.1 at 640 and 1.9 at 320. At 40 nodes — the default
in `AxialParams`, which is the training mesh and not a scoring mesh — the reference error
is $4.6\times10^{-3}$ and exceeds the quantity being measured.

Solving at 2560 costs 112 s against 8 s at 160, on a run that takes an hour and a half.

`verification.py` holds both the reference-side and the model-side computations:

```bash
uv run pinn-ulof verify                 # solves 640/1280/2560/5120, prints the uncertainties
uv run pinn-ulof verify --meshes 320,640,1280
uv run pinn-ulof table models/*.eqx     # scores saved models, prints the published table
```

`richardson()` returns `(nan, nan)` when three successive values are not monotone, because
a quantity outside the asymptotic range has no extrapolated limit and producing one would
invent a number.

## Figures

`charts.py` draws from the reference alone, so CI regenerates it from the physics. Two of
its charts need neither: `network` reads the shipped `TrainConfig` and draws the model
structure, and `trajectories` reads `paper/data/ladder.json`. That file is committed —
16 kB — because it is the scored output of runs costing tens of CPU-hours, which CI cannot
repeat; `pinn-ulof ladder` regenerates it from checkpoints. `model_charts.py` needs a trained network. `paper/model/shipped_f64_seed2.eqx` is
the one committed checkpoint in the repository — 200 kB, seed 2 of the shipped three — and
exists so CI can redraw the paper's residual figure from the network on every push.

## Extension points

**Configuration.** `TrainConfig` is frozen:

```python
from dataclasses import replace
from pinn_ulof import AxialParams, TrainConfig
from pinn_ulof.train import train

model, p, cfg = train(AxialParams(), replace(TrainConfig(), points=3000, iters=30000))
```

**Physical parameters.** `replace(AxialParams(), p_system=2.0e5)`. Both solvers pick it up,
so the reference remains a valid ground truth for the changed problem.

**Equations.** Change `physics.py` and both solvers change together.

**Evaluation.** `predict(model, p, cfg, zeta, t)` returns the five fields in physical units
on a grid. The model is a callable function of $(\zeta, \hat{t})$ and can be differentiated
with respect to either input or to its parameters with the usual JAX transformations.

## Checks

`ruff` and `ty` run under `pre-commit`. The physics is checked by the shipped run agreeing
with an independently discretised reference solver to that solver's own numerical accuracy.
That check requires a reference finer than the model it measures: scored against 40 nodes
this model reports $6\times10^{-3}$ on every field and every seed, which is the reference's
discretisation error rather than the model's.

## Paper

```bash
uv run pinn-ulof figures
uv run pinn-ulof model-figures paper/model/shipped_f64_seed2.eqx
latexmk -pdf -outdir=build paper/paper.tex
```

The `paper` workflow runs exactly that on every push touching `paper/` or `src/`, publishes
`paper.pdf` as an artefact, and fails on an unresolved cross-reference or a figure the paper
references but does not embed.
