# Results

## Reference uncertainty

The reference is a finite-volume discretisation; its error is set by the axial mesh. Solving
on 640, 1280, 2560 and 5120 nodes gives an observed order of convergence of 1.00 on onset
time and 1.00 on peak voided length, consistent with first-order upwinding, so the
uncertainty of a mesh is its distance from the Richardson limit.

| mesh | solve | onset | onset height | peak voided | margin | `T_s` field |
|---|---|---|---|---|---|---|
| 640 | 31 s | 0.00256 s | 0.00078 | 0.195% | 0.125 K | $3.4\times10^{-4}$ |
| 1280 | 57 s | 0.00128 s | 0.00039 | 0.098% | 0.068 K | $1.5\times10^{-4}$ |
| **2560** | **112 s** | **0.00064 s** | **0.00020** | **0.049%** | **0.036 K** | $9.7\times10^{-5}$ |
| 5120 | 205 s | 0.00032 s | 0.00010 | 0.024% | 0.019 K | — |

Reproduce with `uv run pinn-ulof verify`.

Each doubling costs about twice the wall-clock and buys a factor of two in accuracy. The
whole solve takes seconds against hours to train the model it measures, so the resolution of
the comparison is limited by choice rather than by cost.

The saturation margin converges at order 0.92 rather than 1: it is a maximum over a discrete
field rather than a smooth functional.

A mesh's distance from the finest mesh available is not its error: that mesh is itself in
error, and using it as the datum understates every entry above it. The extrapolated limit is
the datum.

Calibration practice requires a tolerance to sit at least four times above the uncertainty
of the instrument measuring it. A model error below the corresponding entry above is a
measurement of the reference.

## Accuracy

Three seeds at the published configuration, against the 2560-node reference. Reproduce with
`uv run pinn-ulof table models/*.eqx`.

| quantity | reference | surrogate |
|---|---|---|
| boiling onset time (s) | $10.9687 \pm 0.0006$ | $10.9818 \pm 0.0003$ |
| onset height $\zeta^{*}$ | $0.9998 \pm 0.0002$ | $0.9998 \pm 0.0000$ |
| peak voided length (m) | $0.3785 \pm 0.0002$ | $0.3792 \pm 0.0002$ |
| peak coolant temperature (K) | 1237.9 | $1237.1 \pm 0.1$ |
| saturation margin (K) | $+68.94 \pm 0.04$ | $+68.16 \pm 0.05$ |
| `T_f`, relative $L_2$ | $9.1\times10^{-5}$ | $(1.742 \pm 0.023)\times10^{-3}$ |
| `T_cl` | $1.4\times10^{-4}$ | $(2.473 \pm 0.024)\times10^{-3}$ |
| `T_s` | $9.7\times10^{-5}$ | $(3.644 \pm 0.090)\times10^{-4}$ |
| `T_c` | $9.9\times10^{-5}$ | $(3.954 \pm 0.034)\times10^{-4}$ |

Surrogate uncertainties are the half-range over three seeds; reference uncertainties come
from the table above. Only the onset height agrees within the sum of the two.

**Onset time does not.** The surrogate reaches saturation 0.0131 s later than the reference,
against a combined uncertainty of 0.0009 s, so the lateness is a property of the model.
**Nor does the peak voided length**, 0.0007 m long against a combined 0.0004 m. The seed
half-ranges — 0.0003 s and 0.0002 m — are too small for either to be an initialisation
effect.

**The film temperature cannot be resolved by this reference.** Its error is
$3.644\times10^{-4}$ against the reference's own $9.7\times10^{-5}$, a ratio of **3.8** —
below the four a tolerance needs to sit above the uncertainty of the instrument measuring
it. This is not a training-budget problem: the converged film error is $3.5\times10^{-4}$
to $3.7\times10^{-4}$ at every point count and every budget in the ladder below, so the
ruler is what has to improve. `verification.MESHES` already carries 5120; bounding that
mesh's own error needs a 10 240 run. Every other field stays clear: `T_f` at a ratio of 19,
`T_cl` 18, `T_c` 40.

## The converged answer does not depend on the collocation count

Three point counts, every finished rung, three seeds each, all against the same reference.

| points | iters | `T_f` ×10⁻³ | `T_cl` ×10⁻³ | `T_s` ×10⁻⁴ | `T_c` ×10⁻⁴ | onset err (s) | peak voided (m) | margin (K) |
|---|---|---|---|---|---|---|---|---|
| — | ref | 0.091 | 0.14 | 0.97 | 0.99 | — | 0.3785 ± 0.0002 | +68.94 ± 0.04 |
| 5000 | 20 000 | 2.08 ± 0.87 | 3.45 ± 1.48 | 11.6 ± 3.0 | 10.7 ± 3.4 | 0.0408 ± 0.0388 | 0.3763 ± 0.0031 | +67.51 ± 0.60 |
| 5000 | 30 000 | 1.595 ± 0.272 | 2.239 ± 0.373 | 4.90 ± 0.50 | 5.03 ± 0.69 | 0.0438 ± 0.0304 | 0.3787 ± 0.0006 | +68.00 ± 0.18 |
| 5000 | 50 000 | 1.655 ± 0.056 | 2.370 ± 0.093 | 3.49 ± 0.16 | 4.05 ± 0.13 | 0.0314 ± 0.0197 | 0.3792 ± 0.0002 | +68.16 ± 0.03 |
| 5000 | 60 000 | 1.707 ± 0.009 | 2.438 ± 0.025 | 3.51 ± 0.07 | 3.94 ± 0.04 | 0.0293 ± 0.0181 | 0.3792 ± 0.0002 | +68.15 ± 0.02 |
| 5000 | 70 000 | 1.703 ± 0.045 | 2.446 ± 0.051 | 3.47 ± 0.05 | 3.91 ± 0.09 | 0.0275 ± 0.0170 | 0.3792 ± 0.0001 | +68.16 ± 0.02 |
| 5000 | 80 000 | 1.731 ± 0.031 | 2.472 ± 0.042 | 3.35 ± 0.04 | 3.88 ± 0.07 | 0.0240 ± 0.0136 | 0.3791 ± 0.0001 | +68.17 ± 0.02 |
| 5000 | 90 000 | 1.730 ± 0.010 | 2.476 ± 0.007 | 3.35 ± 0.02 | 3.88 ± 0.03 | 0.0219 ± 0.0107 | 0.3791 ± 0.0000 | +68.16 ± 0.02 |
| 5000 | 100 000 | 1.755 ± 0.017 | 2.504 ± 0.008 | 3.33 ± 0.07 | 3.86 ± 0.02 | 0.0202 ± 0.0096 | 0.3791 ± 0.0001 | +68.15 ± 0.01 |
| 10 000 | 20 000 | 1.496 ± 0.052 | 2.130 ± 0.063 | 7.61 ± 0.58 | 6.81 ± 0.14 | 0.0146 ± 0.0132 | 0.3778 ± 0.0011 | +67.97 ± 0.13 |
| 10 000 | 30 000 | 1.740 ± 0.094 | 2.473 ± 0.110 | 4.42 ± 0.12 | 4.39 ± 0.22 | 0.0160 ± 0.0008 | 0.3795 ± 0.0003 | +68.17 ± 0.09 |
| **10 000** | **50 000** | 1.742 ± 0.023 | 2.473 ± 0.024 | 3.64 ± 0.09 | 3.95 ± 0.03 | 0.0130 ± 0.0003 | 0.3792 ± 0.0002 | +68.16 ± 0.05 |
| 10 000 | 100 000 | 1.771 ± 0.005 | 2.518 ± 0.003 | 3.49 ± 0.06 | 3.85 ± 0.003 | 0.0115 ± 0.0011 | 0.3792 ± 0.0001 | +68.17 ± 0.02 |
| 20 000 | 20 000 | 1.747 ± 0.455 | 2.545 ± 0.656 | 8.26 ± 0.87 | 6.83 ± 0.46 | 0.0090 ± 0.0081 | 0.3788 ± 0.0017 | +68.07 ± 0.26 |
| 20 000 | 50 000 | 1.739 ± 0.047 | 2.474 ± 0.053 | 3.70 ± 0.08 | 3.97 ± 0.02 | 0.0127 ± 0.0011 | 0.3795 ± 0.0003 | +68.14 ± 0.07 |
| 20 000 | 60 000 | 1.767 ± 0.061 | 2.514 ± 0.076 | 3.62 ± 0.08 | 3.88 ± 0.07 | 0.0119 ± 0.0015 | 0.3793 ± 0.0001 | +68.16 ± 0.02 |
| 20 000 | 70 000 | 1.776 ± 0.053 | 2.518 ± 0.074 | 3.52 ± 0.06 | 3.87 ± 0.07 | 0.0117 ± 0.0012 | 0.3793 ± 0.0002 | +68.17 ± 0.00 |
| 20 000 | 80 000 | 1.774 ± 0.003 | 2.517 ± 0.004 | 3.50 ± 0.03 | 3.87 ± 0.01 | 0.0113 ± 0.0007 | 0.3792 ± 0.0001 | +68.16 ± 0.00 |
| 20 000 | 90 000 | 1.780 ± 0.022 | 2.528 ± 0.027 | 3.50 ± 0.03 | 3.84 ± 0.01 | 0.0108 ± 0.0006 | 0.3793 ± 0.0001 | +68.18 ± 0.01 |
| 20 000 | 100 000 | 1.786 ± 0.007 | 2.535 ± 0.005 | 3.53 ± 0.05 | 3.84 ± 0.00 | 0.0108 ± 0.0005 | 0.3792 ± 0.0001 | +68.17 ± 0.01 |

All three ladders reach 100 000 iterations. At
10 000 iterations no point count produces a usable front — the voided length is short by 3
to 4 cm and the margin by 8 to 16 K — so that rung is a failure to converge rather than a
less accurate solution.

The three converged tails agree: 20 000 points at 100 000 iterations gives a peak voided
length of 0.3792 m and a margin of +68.17 K, the same to four digits and 0.01 K as 10 000
points at the same budget and as 5000 points at 50 000. Boiling onset is the one quantity
still improving at the end of every ladder, reaching 0.0108 s at 20 000 points and 0.0115 s
at 10 000 — against a reference uncertainty of 0.00064 s, so still resolvable as an error.

**Every converged row lands on the same answer.** From 40 000 iterations to 100 000, every rung
of every count lies between 0.3791 and 0.3795 m in peak voided length and between +68.11 and
+68.18 K in margin — a spread of 0.0004 m across a four-fold range of collocation points,
twice the reference's own uncertainty of 0.0002 m. The 0.0007 m discrepancy is therefore a
property of the formulation, not of sampling or optimisation, and the algebraic void closure
is the natural suspect. Below 40 000 the rungs are still moving: at 30 000 the same quantity
spans 0.3787 to 0.3798.

**Intermediate budgets cross the reference rather than converging to it.** Voided length
runs 0.3763 → 0.3787 → 0.3792 at 5000 points, 0.3778 → 0.3795 → 0.3794 at 10 000, and
0.3788 → 0.3798 → 0.3794 at 20 000. Each passes through 0.3785 on the way out. A rung
selected because it agrees is a zero crossing and will not reproduce; the converged value is
what should be quoted.

**`T_f` and `T_cl` are non-monotone in the budget** at every point count, reaching a minimum
at 20 000–30 000 iterations and then settling 10–15% worse. The half-ranges at the high
rungs are ±5e-6, so this is not seed noise.

## A larger configuration does not help

Measured at 5000 collocation points with an early-time density that is no longer part of
the method. An ablation is a statement about the formulation it was run on, so this section
and the curvature-memory section below are provisional until repeated at the shipped
10 000-point uniform configuration. Neither has been re-measured.

A run at four times the collocation points, four times the embedding width and twice the
iterations — 20 000 points, 256 features, 100 000 iterations, one seed — was scored at every
checkpoint against the same reference.

| iterations | loss | `T_s` | `T_f` | `T_cl` | onset err |
|---|---|---|---|---|---|
| 20 000 | 7.53e-07 | 4.447e-4 | 1.679e-3 | 2.450e-3 | 0.0146 s |
| 40 000 | 1.64e-07 | 3.896e-4 | 1.821e-3 | 2.592e-3 | 0.0103 s |
| 60 000 | 6.48e-08 | 3.349e-4 | 1.782e-3 | 2.537e-3 | 0.0103 s |
| 80 000 | 3.57e-08 | **3.301e-4** | 1.780e-3 | 2.534e-3 | 0.0105 s |
| 100 000 | 2.28e-08 | 3.386e-4 | 1.794e-3 | 2.552e-3 | 0.0106 s |

**The loss and the accuracy decouple.** The loss falls 33-fold across the ladder. `T_s`
improves 1.3-fold and then reverses. Over the last 40 000 iterations the loss drops a
further 2.8-fold while every accuracy metric is flat or slightly worse — four and a half
hours of compute for nothing measurable.

`T_f` and `T_cl` are worse at every budget than at 20 000, rising between 20 000 and 40 000
and never recovering. More optimisation improves the film field and degrades the fuel field,
which is what a residual with no remaining leverage where the error is looks like.

The front quantities saturate by 40 000: onset height, peak voided length, peak coolant
temperature and margin are identical to four digits from there on. The voided length settles
at 0.3793 m against the reference's 0.3785, a 0.2% overshoot no budget corrects.

Against the shipped f64 default:

| | f256, best checkpoint | f64, three seeds |
|---|---|---|
| `T_s` | $3.301\times10^{-4}$ | $(4.14 \pm 0.30)\times10^{-4}$ |
| `T_f` | $1.780\times10^{-3}$ | $(1.613 \pm 0.026)\times10^{-3}$ |
| `T_cl` | $2.534\times10^{-3}$ | $(2.326 \pm 0.016)\times10^{-3}$ |
| onset error | 0.0105 s | $0.0236 \pm 0.0136$ s |
| wall-clock | 32 666 s | 5 292 s |

f256 is 20% better on `T_s` and better on onset; it is 10% worse on `T_f` and `T_cl`, both
outside the f64 seed band in the wrong direction, at 6.2 times the wall-clock. The
configuration is a trade, not an improvement, and its best checkpoint is 80 000 rather than
the 100 000 it was run to.

The f256 arm is one seed. The `T_s` advantage is about two half-ranges of the f64 spread and
is suggestive rather than established; the `T_f` and `T_cl` deficits are larger relative to
their spreads and are the firmer half of the comparison.

## Reproducing

```bash
uv run pinn-ulof train --seed 0
uv run pinn-ulof train --seed 1
uv run pinn-ulof train --seed 2
```

About three hours apiece on eight cores. Each saves its model before scoring, so later
questions cost about three minutes rather than another three hours.

The run is deterministic given the seed. Seed 0 at the shipped configuration scores exactly
this, and it is worth diffing against before trusting a modified environment:

```
  T_f   1.719e-03      T_cl  2.449e-03      T_s   3.554e-04      T_c   3.988e-04
  onset time error     0.0133 s
  peak voided length   0.3791 m      (100.2% of the reference)
  saturation margin    +68.11 K
```

The scores are the strict check rather than the loss: a quasi-Newton run compounds any
perturbation through its curvature history, so four field norms matching to four digits
after 50 000 iterations means the numerical path is unchanged.

At this configuration the seeds agree closely — onset spans 0.0128 to 0.0133 s and the
fields to under 3% — so a single run is nearly as informative as three. That is a property
of 10 000 points, not of the model: at 5000 the same three seeds span 0.0117 to 0.0511 s on
onset, and below 5000 the rungs are bistable rather than noisy, with three seeds at 3000
points spanning a factor of nine because they differ in whether the boiling front is found
at all. A comparative result still quotes three seeds with the per-seed range.

## Curvature memory

Measured at 5000 points on the earlier collocation density; see the note above.

The L-BFGS memory — the number of curvature pairs retained — is the one hyper-parameter
whose library default differs between implementations. Three values were run at the shipped
configuration, three seeds each, three concurrent processes on eight cores apiece so the
wall-clocks are comparable.

| memory | wall-clock (s) | final loss | `T_s` | `T_f` |
|---|---|---|---|---|
| 10 | 5158 ± 24 | $(1.47 \pm 0.44)\times10^{-5}$ | $(7.50 \pm 2.00)\times10^{-3}$ | $(2.43 \pm 0.67)\times10^{-2}$ |
| **50** | 5311 ± 27 | $(2.94 \pm 0.85)\times10^{-7}$ | $(4.14 \pm 0.30)\times10^{-4}$ | $(1.613 \pm 0.027)\times10^{-3}$ |
| 100 | 5736 ± 26 | $(1.23 \pm 0.34)\times10^{-7}$ | $(3.43 \pm 0.08)\times10^{-4}$ | $(1.750 \pm 0.049)\times10^{-3}$ |

| memory | onset error (s) | peak voided | margin (K) |
|---|---|---|---|
| 10 | 0.0750 ± 0.0398 | 94.2 ± 2.0% | 59.42 ± 2.40 |
| **50** | 0.0236 ± 0.0135 | 99.9 ± 0.1% | 68.25 ± 0.12 |
| 100 | 0.0195 ± 0.0125 | 99.8 ± 0.4% | 68.57 ± 0.32 |

**10 is not a cheaper option.** It is 17 times worse on the film temperature and 0.974 times
the cost — the flops the two-loop recursion saves are spent again on line-search evaluations
from a poorer search direction, so there is no time to reinvest in extra iterations. Its
saturation margin is 9 K below the reference and its seed spread is an order of magnitude
wider than the other two: fewer pairs make the result less reproducible as well as less
accurate.

**100 is a trade, not an upgrade.** It is 15% better on `T_s` and 7% better on `T_c`, 8% and
7% *worse* on `T_f` and `T_cl` — outside the memory-50 seed band in the wrong direction —
and costs 1.082 times as much. At equal wall-clock memory 50 buys about 4000 extra
iterations, which from its own trajectory is worth roughly 25% on the loss and does not
close the gap; the ranking on `T_s` survives, and so does the deficit on the fuel and
cladding fields.

50 is therefore kept, now because it is the measured optimum of the three rather than
because it was inherited. The cliff is asymmetric: 10 to 50 is a factor of 17, and 50 to 100
is 0.85 in one direction and 1.08 in the other.

## Open questions

### The objective and the accuracy metrics diverge

The loss falls monotonically for the whole solve. The quantities the model is judged on do
not, and past a point they move in opposite directions. At 10 000 collocation points:

| quantity | 20 000 iters | 100 000 iters | change |
|---|---|---|---|
| `T_f` ×10⁻⁴ | 14.96 ± 0.52 | 17.71 ± 0.05 | **+18%** |
| `T_cl` ×10⁻⁴ | 21.30 ± 0.63 | 25.18 ± 0.03 | **+18%** |
| `T_s` ×10⁻⁴ | 7.61 ± 0.58 | 3.49 ± 0.06 | −54% |
| `T_c` ×10⁻⁴ | 6.81 ± 0.14 | 3.85 ± 0.00 | −43% |
| onset error (s) | 0.0146 ± 0.0132 | 0.0115 ± 0.0011 | −21% |

Eighty thousand further iterations more than halve the film error and improve onset by a
fifth while making the fuel and cladding fields 18% worse. The seed half-ranges at 100 000
are 5e-6 and 3e-6 in relative $L_2$, two orders below the shift, so this is not scatter. The
same non-monotonicity appears at 5000 points (fuel best at 30 000) and at 20 000 points
(fuel best at 40 000).

The mechanism is in the objective. The four blocks are scaled to a common magnitude by fixed
constants taken from the equations' characteristic rates. That equalises them at
initialisation and says nothing about their marginal returns during the solve: once the
fuel and cladding residuals are small, further reduction of the total is cheapest in the
film and coolant blocks, and the optimiser takes it there even where the solid fields must
give ground. A single scalar has one direction of improvement, and past a point it is not
aligned with any individual quantity.

**The open experiment is residual balancing.** Weights that respond to how much each block
is still improving — from gradient norms, from each block's rate of change, or from the
functionals the surrogate will be qualified against — would let the solve keep reducing the
film error without spending the fuel field. Bounded adaptation is essential: unbounded
per-block weights on this system run to 5e6 and make every field worse. Whether such a
scheme moves the converged *limits* or only the path to them is untested.

### Each quantity has its own convergence rate

Four distinct behaviours over one budget axis:

- **solid fields** (`T_f`, `T_cl`) — best at 20 000–40 000, then degrade
- **film and coolant** (`T_s`, `T_c`) — improve monotonically to the end of every ladder
- **peak voided length, margin** — locked to four digits by 30 000, never move again
- **onset time** — improves throughout, slowly, without saturating, at every point count

No single stopping rule serves all four, and stopping when the loss flattens is the worst of
them: the loss is still falling where three of the four have stopped responding. Monitoring
the functionals themselves needs the reference during training, so it needs a held-out
validation transient rather than the scored one.

### Limits further training cannot reach

**Peak voided length** converges to 0.3792 m against the reference's 0.3785 ± 0.0002 m, from
either side, at every collocation count and budget measured. The gap is in the formulation.
The algebraic void closure is the natural suspect — it replaces the void transport equation
with a quasi-steady relation on the coolant temperature, exact only for instantaneous vapour
redistribution, and a 0.2% bias in the extent of the voided region is what that would
produce near the front. Testing it means restoring the transport equation with a relaxation
time and seeing whether the limit moves.

**Boiling onset** never agrees at any configuration. The discrepancy narrows from 0.0314 s
to 0.0115 s as points and budget increase, but the seed scatter narrows faster, so the
disagreement becomes *more* significant: at the shipped configuration it is 0.0131 s against
a combined 0.0009 s. Whether it shares the closure's origin or comes from resolving a
tangency condition on a locally flat field is unresolved.

**Film temperature** has reached the reference's resolution — ratio 3.6 to 3.8 at every
converged configuration. Nothing about the surrogate can be demonstrated on this field until
the reference is refined.

## What the model does not do

**The closed reactivity loop.** The void reactivity functional nearly cancels:

| contribution | value |
|---|---|
| positive-worth region $J^{+}$ | $+4.656\times10^{-4}$ |
| negative-worth region $J^{-}$ | $-1.695\times10^{-4}$ |
| sum $J$ | $+2.962\times10^{-4}$ |
| cancellation ratio | 0.466 |

A relative error $\epsilon$ on each half becomes $2.1\epsilon$ on the sum, so the functional
is ill-conditioned independently of field accuracy. Driving the kinetics from the learned
fields recovers 8 to 16% of the reference integral, while the non-cancelling Doppler
integral over the same fields is reproduced to within 1.7%.

Residual-adaptive sampling does not address it. Dual-weighted-residual theory gives the
leading-order functional error as the residual paired against an adjoint solution. For the
advective coolant operator that adjoint is a step: constant over the lower 72% of the
channel and zero above it, because the void slope underflows to zero wherever the coolant is
subcooled. Every point below the front carries equal sensitivity and every point above it
carries none, so concentrating points at the front — where residual magnitude is largest —
places them where the functional is insensitive.

Evaluated open-loop on the surrogate's own fields the functional is accurate: the positive
half to 1.66–2.66% and the negative half exactly, the latter because that region is fully
voided in surrogate and reference alike and the integral is fixed by geometry. The deficit
arises in the closed loop, where the void reactivity feeds back into the kinetics.

**Beyond 16.5 s.** The sodium property correlations leave their validity range.

**Post-boiling phenomena.** Pin failure, fuel motion and cladding relocation are outside the
model.
