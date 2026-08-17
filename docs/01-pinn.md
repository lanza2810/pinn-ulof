# Physics-informed neural networks

A neural network is used as the solution of a differential equation and trained by
penalising how badly it fails to satisfy that equation.

## The network

A parameterised function. Alternating affine maps and a smooth nonlinearity:

```math
N(x) = W_5\, \tanh\bigl(W_4 \tanh(\cdots \tanh(W_0 x + b_0) \cdots) + b_4\bigr) + b_5
```

The entries of $W_k$ and $b_k$ are the parameters; there are 25 221 of them here. Two
properties matter. With sufficient width the form approximates any continuous function
arbitrarily well. And because it is a composition of elementary operations, exact
derivatives with respect to inputs and parameters follow from the chain rule — automatic
differentiation, exact to machine precision at a cost comparable to one evaluation.

## The method

For a system $\mathcal{R}[u] = 0$ with initial condition $u(z,0) = u_0(z)$, take the
network $N_\theta(z,t)$ as the candidate solution. Automatic differentiation supplies
$\partial_t N_\theta$ and $\partial_z N_\theta$ at any point, so the residual

```math
r_\theta(z, t) = \partial_t N_\theta(z,t) - f\bigl(N_\theta, \partial_z N_\theta\bigr)
```

is computable anywhere and vanishes identically when the network solves the equation.
Minimising its mean square over a set of **collocation points** is the method:

```math
L(\theta) = \frac{1}{n}\sum_{i=1}^{n} r_\theta(z_i, t_i)^2
```

The loss references no known solution, so the method applies to problems with no available
data. The result is a closed-form function, differentiable everywhere, rather than values
on a mesh. The collocation points carry no connectivity and no stencil; their number sets
the cost, not the discretisation.

## Initial and boundary conditions

The residual alone admits many solutions. Conditions are imposed either as additional loss
terms,

```math
L = L_{\text{residual}} + \lambda_1 L_{\text{initial}} + \lambda_2 L_{\text{boundary}},
```

or structurally, by constraining the ansatz. The penalty form introduces weights whose
correct values depend on the problem, the units and the state of training. The structural
form removes them: writing

```math
u_\theta(z,t) = u_0(z) + t\, N_\theta(z,t)
```

makes the initial condition hold for every $\theta$. This repository imposes every
condition structurally, so its loss contains residual terms alone.

## Three difficulties

**Spectral bias.** Training resolves low-frequency structure long before high-frequency
structure. Where the solution contains a front or a boundary layer, the optimisation budget
is spent on the smooth bulk and the sharp feature is left unresolved. The remedy used here
is a Fourier feature map, which presents the inputs at many frequencies at once.

**The loss is a domain average.** A feature occupying 3% of the domain contributes about 3%
of the loss, so further optimisation converges more precisely to a minimiser that is wrong
where the feature is. The loss curve gives no indication.

**The objective is ill-conditioned.** It composes a network with a differential operator.
On a fixed point set it is smooth, deterministic and of modest dimension — the regime of
quasi-Newton methods. Adam-family optimisers were explored and dismissed: they could reach comparable
accuracy, but only after a time-consuming fine tuning of their hyper-parameters.

## Cost

The reference solution here takes seconds and the network takes an hour and a half. The
case for a surrogate rests on what follows: a continuous solution, cheap to evaluate, with
exact derivatives with respect to inputs and parameters.
