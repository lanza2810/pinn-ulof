# Documentation

| | | |
|---|---|---|
| 00 | [The transient](00-transient.md) | the reactor problem, the governing equations, the reference solution |
| 01 | [Physics-informed neural networks](01-pinn.md) | the method, and the three difficulties it presents |
| 02 | [The model](02-model.md) | the ansatz, the residual system, the embedding, the solver |
| 03 | [The code](03-code.md) | the source files, the commands, the verification tooling |
| 04 | [Results](04-results.md) | reference uncertainty, measured accuracy, open problems |

A neural network is used as the solver. It maps normalised height and time to four
temperatures and a void fraction, and is trained until it satisfies the governing equations
at 10 000 collocation points. There is no training data.

The physical problem is a sodium-cooled fast reactor losing its pumps: the coolant heats,
boils near the top of the channel, and the resulting voids add positive reactivity. The
quantity of interest is when and where boiling starts.

Against a stiff Radau reference at 2560 axial nodes the surrogate reaches a relative $L_2$
of $4.1\times10^{-4}$ on the film temperature and reproduces the peak voided length to
0.05%. Boiling onset is placed 0.024 s late, which at this mesh is a resolved property of
the model rather than of the reference. The closed reactivity loop is not reproduced;
[04-results.md](04-results.md) quantifies why.
