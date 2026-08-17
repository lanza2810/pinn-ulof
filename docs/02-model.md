# The model

The whole model is [`network.py`](../src/pinn_ulof/network.py) and
[`train.py`](../src/pinn_ulof/train.py).

```
(zeta, t_hat)  ->  Fourier embedding (frozen)  ->  MLP  ->  raw output
                                                              |
                             theta_0(zeta) * exp(t_hat * raw)  |  <- ansatz
                                                              v
                            four temperatures + void fraction
```

Inputs are normalised height $\zeta \in [0,1]$ and normalised time $\hat{t} \in [0,1]$,
where $\hat{t} = 1$ maps to 16.5 s — the end of the sodium property correlations' validity
range, not the 60 s nominal horizon.

## Ansatz

The network output is not the state. The state is

```math
\theta(\zeta, \hat{t}) = \theta_0(\zeta)\, \exp\bigl(\hat{t}\, N(\zeta, \hat{t})\bigr)
```

with $\theta_0$ the analytic steady profile, computed in closed form. Three conditions then
hold for every parameter value:

- **Initial condition.** At $\hat{t} = 0$ the exponent vanishes and $\theta = \theta_0$
  identically.
- **Positivity.** The exponential is positive and $\theta_0$ non-negative, so no temperature
  falls below the inlet. The Doppler feedback is logarithmic in the fuel temperature and
  undefined for non-positive arguments; an additive ansatz permits the optimiser to drive
  $T_f$ negative while the loss falls.
- **Inlet boundary.** The steady coolant profile satisfies $\theta_{c,0}(0) = 0$, which
  pins $T_c(0,t) = T_{\mathrm{in}}$. The advection equation admits exactly one upstream
  condition.

The exponent carries a smooth $\tanh$ ceiling, so a diverging iterate cannot overflow.

Because every condition holds structurally, the loss contains no condition terms and no
weights to balance.

## Residual system

The void fraction is closed algebraically on the coolant temperature, $\alpha = g(T_c)$.
The superheat switch underflows to exactly zero below saturation, so the void-free initial
and inlet conditions follow from the closure.

**The residual therefore has four blocks, one per temperature.** The closure is justified
by timescale separation — vapour fills a node in 0.71 ms against an advective 0.113 s —
and that separation asserts that the void transport equation is in quasi-steady balance.
Once $\alpha$ is a function of $T_c$ it is not an independent unknown, so four unknown
fields under five equations is over-determined, and the fifth equation is the one the
closure approximates. Residualising it costs more than an order of magnitude in converged
error and places boiling onset seconds rather than hundredths of a second late; a five-block
solve at the full budget is worse than a four-block solve at one fifteenth of it.

Each block is scaled to a common magnitude. The scales are constants, so dividing them out
recovers the physical residual and the zero set is unchanged.

## Loss

```math
L(\theta) = \frac{1}{K}\sum_{k=1}^{K}
  \Biggl[ \frac{1}{|I_k|} \sum_{i \in I_k} \sum_{m=1}^{4} r_m(\zeta_i, \hat{t}_i)^2 \Biggr]
```

Points are partitioned into $K = 32$ equal windows in time; each window is averaged, then
the windows are averaged. A plain mean would let wherever the points happen to lie become a
statement about where the residual matters, and boiling onset is at 10.98 s of a 16.5 s
window, so a density tilted toward early time would weight against the event the model
exists to predict. Window averaging keeps sampling density and loss weighting independent,
a uniform draw fluctuates in how many points land in each window, and this makes the
objective insensitive to that fluctuation rather than letting it set the weights.

## Embedding

```math
x \mapsto \bigl[\sin(2\pi B x),\; \cos(2\pi B x)\bigr]
```

$B$ is drawn once and held under `stop_gradient`: a change of input coordinates, not a
fitted layer. Without it no optimiser tested here forms a boiling front, because the front
is the high-frequency part of the solution.

The default is 64 features, giving 128 embedded dimensions. 128 and 256 features are
indistinguishable at this budget and cost 1.25 and 1.9 times as much per iteration. Raising
the embedding to 256 together with the collocation set to 20 000 and the budget to 100 000
does not change that conclusion: it improves the film temperature by 20%, degrades the fuel
and cladding temperatures by 10%, and costs 6.2 times the wall-clock
([04-results.md](04-results.md)).

| embedding | trainable | read-out weight | trunk |
|---:|---:|---:|---:|
| 32 | 21 125 | 4 096 | **17 029** |
| 64 | 25 221 | 8 192 | **17 029** |
| 256 | 49 797 | 32 768 | **17 029** |
| 1024 | 148 101 | 131 072 | **17 029** |

Widening the embedding grows only the read-out. The fitting capacity is 17 029 parameters
at every width, so a wider embedding presents more frequencies without making the model more
expressive.

## Collocation

10 000 points drawn uniformly over $(\zeta, \hat{t})$, once, and held for the whole solve.
There is no sampling choice in the method.

The set is fixed because L-BFGS accumulates curvature information about the objective it is
minimising; redrawing changes that objective and invalidates the accumulated history.

The point count buys boiling-onset accuracy and nothing else. At 50 000 iterations, three
seeds:

| points | onset error (s) | `T_s` | peak voided (m) | margin (K) |
|---|---|---|---|---|
| 5000 | 0.0314 ± 0.0197 | $(3.49 \pm 0.16)\times10^{-4}$ | 0.3792 ± 0.0002 | +68.16 ± 0.03 |
| **10 000** | **0.0130 ± 0.0003** | $(3.64 \pm 0.09)\times10^{-4}$ | 0.3792 ± 0.0002 | +68.16 ± 0.05 |
| 20 000 | 0.0127 ± 0.0011 | $(3.70 \pm 0.08)\times10^{-4}$ | 0.3795 ± 0.0003 | +68.14 ± 0.07 |

Onset improves 2.4-fold from 5000 to 10 000 and then stops; the seed half-range falls
65-fold over the same step. Everything else is flat. 20 000 costs twice as much for
0.0003 s, so 10 000 ships. Below 5000 the rungs are bistable rather than noisy: at 3000
points three seeds span a factor of nine, because they differ in whether the front is found
at all.

## Solver

L-BFGS with a strong-Wolfe line search, 50 000 iterations, 50 curvature pairs, no
first-order stage.

- Adam-family optimisers were explored and dismissed: they could reach comparable accuracy,
  but only after a time-consuming fine tuning of their hyper-parameters.
- Temperature fields saturate near 30 000 iterations; onset improves to 50 000.
- `memory_size = 50`, the retained curvature pairs, measured against 10 and 100 at three
  seeds each. 10 is 17 times worse on the film temperature at 0.97 times the cost; 100 is
  15% better on the film temperature, 8% worse on the fuel temperature, and costs 1.08
  times as much ([04-results.md](04-results.md)). The library default is 10, which makes
  this the one hyper-parameter that must be set explicitly for two implementations of the
  algorithm to be comparable at all.

All arithmetic is float64. Curvature pairs are differences of gradients; at single
precision those differences are dominated by rounding once the residual is small.

## Not used

No adaptive sampling, no residual-based refinement, no loss weighting, no curriculum, no
causal weighting, no second optimiser. Each was measured during the study behind the paper
and none is in the shipped configuration.
