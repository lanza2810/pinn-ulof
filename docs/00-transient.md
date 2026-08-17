# The transient

## What is being modelled

A **sodium-cooled fast reactor** (SFR) uses liquid sodium rather than water to carry heat
out of the core. Sodium conducts heat extremely well and stays liquid over a wide range, so
the core runs far below the coolant's boiling point in normal operation, at atmospheric
pressure rather than the ~155 bar a pressurised water reactor needs.

The sequence modelled here is an **unprotected loss of flow** (ULOF):

1. The primary pumps lose power and the coolant flow coasts down toward a
   natural-circulation floor.
2. The **protection system fails to scram** — that is what "unprotected" means. The chain
   reaction continues at close to full power.
3. Less coolant carries the same heat, so the sodium leaving the core gets hotter.
4. Eventually the coolant reaches its saturation temperature near the top of the channel
   and **boiling begins**.

Step 4 is the event of interest, because of what boiling does next.

## Why boiling is the interesting part

In an SFR the coolant is also a neutron absorber and a moderator. Removing it — replacing
liquid sodium with vapour — changes the neutron balance, and in a large fast core the net
effect is **positive**: voiding the coolant *adds* reactivity, which raises the power, which
makes more vapour. That is a feedback loop with the wrong sign.

It is not unopposed. As the fuel heats, Doppler broadening of the resonance absorption
cross-sections adds *negative* reactivity, and the fuel and structure expand. Whether the
transient turns over or runs away depends on the balance between these, and that balance
depends on **how much of the channel is voided and how fast**.

So the safety-relevant outputs are not average temperatures. They are:

- **when** boiling starts (onset time),
- **where** in the channel it starts (onset height),
- **how far** the voided region extends (peak voided length),
- **how much margin to saturation** remains elsewhere.

These are the quantities [04-results.md](04-results.md) scores.

## The model

One coolant channel is resolved along its axis, with the materials lumped radially. Height
is normalised as $\zeta = z/H \in [0, 1]$.

At each height the state is four temperatures — fuel $T_f$, cladding $T_{cl}$, structure
$T_s$, coolant $T_c$ — plus a void fraction $\alpha$. Six delayed-neutron precursor
concentrations depend on time alone.

### Energy equations

With $C_f$, $C_{cl}$, $C_c$ the heat capacities per unit channel length:

```math
\begin{aligned}
C_f\, \partial_t T_f &= (1-\gamma_c)\, q'(\zeta,t) - q_{fe} \\
C_{cl}\, \partial_t T_{cl} &= q_{fe} - q_{ec} \\
\rho_s c_s t_s\, \partial_t T_s &= -h_{sc}(\alpha)\,(T_s - T_c) \\
C_c\, \partial_t T_c &= \bigl(1 - b(T_c)\bigr)\, q_w - w(t)\, c_c\, \partial_z T_c \\
\partial_t \alpha &= S_v(T_c, \alpha, q_w) - u(t)\, \partial_z \alpha
\end{aligned}
```

The fuel is heated by fission and loses heat to the cladding; the cladding passes it to the
coolant; the coolant carries it away by advection. The last equation transports the void.

Two features matter for what follows. The coolant and void equations contain
$\partial_z$ terms, so this is a **transport problem** — information moves up the channel
at the coolant velocity $u$, and only one boundary condition is admissible, at the inlet.
And the heat-transfer coefficients depend on $\alpha$, so a voided node **insulates its own
wall**: once boiling starts, the cladding there stops being cooled and heats rapidly.

### What drives it

The flow coastdown is prescribed:

```math
w(t) = w_0 \bigl[ f_{\mathrm{nc}} + (1 - f_{\mathrm{nc}})\, e^{-t/\tau_{\mathrm{pump}}} \bigr]
```

with $f_{\mathrm{nc}}$ the natural-circulation floor. Power comes from six-group point
kinetics with a prompt-jump closure, driven by reactivity feedback from the fuel
temperature (Doppler) and the void.

### Closures

Sodium properties — density, enthalpy, saturation temperature, latent heat and the rest —
come from the thirteen numbered correlations of the SAS4A/SASSYS-1 manual's section 12.13.
Their stated validity range is what sets the time horizon: the reference solution is carried
to **16.5 s of a 60 s nominal window**, because that is where the channel leaves the range
over which those property fits are defined. Training past that point would be fitting a
model outside its own closures.

Boiling is a smooth switch on the superheat rather than a discrete phase-change event, and
the void is closed **quasi-steadily** on the coolant temperature — meaning $\alpha$ is
determined by $T_c$ rather than carried as an independent unknown. That choice determines the
residual system; see [02-model.md](02-model.md).

Four departures from the SAS4A formulation are deliberate and are registered, with their
justification, in the paper.

## The reference solution

The equations are discretised on a uniform axial mesh and integrated with a **stiff implicit
Radau solver** to 16.5 s at tolerances of $10^{-8}$. This is the ground truth: the surrogate
never sees it during training and is compared against it only afterwards.

Its own uncertainty is set by that mesh. Scoring uses 2560 nodes, where the uncertainty is
0.00064 s on onset time and 0.049% on peak voided length; `pinn-ulof verify` derives both.
[04-results.md](04-results.md) gives the full table and the reason for the mesh.
