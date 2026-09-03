# Verification model

CELNN treats verification as a chain from meaning to evidence:

```text
definition → representation → invariant → oracle → evidence
```

The goal is not maximum test count. The goal is the smallest set of independent checks that can detect scientifically meaningful regressions.

## 1. Definition

State the mathematical or compatibility property before choosing a test. Examples:

- the derivative equals `-x + A*y(x) + B*u + z`;
- a boundary name has one stable public meaning;
- an exported object appears exactly once in its public reference surface.

## 2. Representation

Identify how the property is encoded: arrays, template offsets, serialized JSON, Python exports, or another structure. Representation mistakes are often different from definition mistakes and deserve separate checks.

## 3. Invariant

Derive something that must remain true. Examples include shape preservation, identity-template behavior, reset semantics, translation invariance away from boundaries, or exact zero residual for an analytic equilibrium.

## 4. Oracle

Choose an independent way to decide whether the invariant holds. Preferred evidence is roughly:

1. exact analytic result;
2. independent trusted reference;
3. differential implementation;
4. metamorphic relation;
5. structural invariant;
6. tolerance-only comparison.

Lower entries are not invalid; they simply provide less independence and require more care.

## 5. Evidence

Run the oracle in the environment that can actually establish the claim. A mocked Torch import can prove reference structure, not gradients. A CPU environment can prove CPU behavior, not CUDA execution.

## Tolerance discipline

Do not widen a tolerance merely to make a regression disappear. First identify the expected numerical error source, scale, dtype, and convergence behavior. If no rationale supports a new tolerance, the failing test is evidence that the implementation or the test model needs investigation.

## Documentation evidence

Executable documentation protects the published example. Production invariants belong in `tests/`. A lab can intentionally expose known technical debt without converting that debt into a permanent required behavior.

The equilibrium lab is the model: it asserts the mathematical counterexample and merely observes the current diagnostic separately.
