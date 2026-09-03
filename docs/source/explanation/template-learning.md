# Learning CELNN templates

Template design can be posed as parameter optimization. The parameter vector contains feedback coefficients, control coefficients, bias, or another chosen subset; an objective measures how the resulting trajectory or output differs from the desired behavior.

The central structure is

```text
parameters → CELNN trajectory → objective → parameter update.
```

CELNN currently exposes two substantially different optimization routes.

## Genetic search: derivative-free optimization

Kozek, Roska, and Chua formulate CNN template learning as an optimization problem and use a genetic algorithm to derive templates. Their motivation includes objectives that may be noisy, discontinuous, or difficult to differentiate because they depend on transient and settled CNN behavior; see {ref}`kozek-roska-chua-1993`.

CELNN’s `celnn.training` implementation follows the broad optimization idea, encoding `(A, B, z)` as real-valued candidate genes and evaluating candidate networks on a dataset. Its DEAP implementation, representation, defaults, and fitness contract are the library’s own and should not be conflated with the historical algorithm.

## Gradient learning through trajectories

Schuler and colleagues define a trajectory-dependent error functional and use calculus of variations to derive gradients with respect to CNN parameters; see {ref}`schuler-et-al-1992`.

Modern PyTorch provides another mechanism for the same high-level dependency: unroll differentiable CELNN steps, compute a loss, and let automatic differentiation propagate gradients through the numerical computation.

Again, the lineage is conceptual rather than identity of algorithms. CELNN’s `DifferentiableCellularNetwork` does not implement the paper’s adjoint/Euler–Lagrange procedure.

## The objective defines what was learned

If optimization minimizes endpoint MSE on a finite dataset, the direct evidence is only about that objective under those conditions. It does not automatically establish:

- stability outside the sampled initial states;
- robustness to perturbations;
- generalization to new inputs;
- solver independence;
- physical realizability.

Those properties require their own objectives, constraints, or validation evidence.

## Search method should follow problem structure

Gradient optimization is efficient when a useful differentiable computation graph exists and local gradient information is informative. Genetic search can be useful when objectives are discontinuous, multimodal, or otherwise unsuitable for gradients, at the cost of many evaluations.

Neither method is universally superior. The scientifically relevant comparison is total evidence for the desired property under a stated computational budget and objective.
