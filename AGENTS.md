# CELNN repository instructions

These instructions define the durable engineering contract for maintaining CELNN.
`AICP.aicp` remains project/design context; it is not a second maintainer constitution.
If instructions, AICP, code, or executable tests disagree about scientific meaning,
treat that as a contract conflict and reconcile it before changing behavior.

## Engineering objective

Maintain CELNN as the smallest robust public scientific Python library that preserves
its mathematical meaning across supported execution modes and makes incorrect changes
easy to falsify.

Optimize for scientific correctness, explicit contracts, predictable failure,
reproducibility, compatibility, maintainability, and minimum necessary complexity.

Before adding a mechanism, establish:
1. the invariant it protects;
2. the observable failure without it;
3. the natural owner of that invariant;
4. the evidence that can falsify the claim;
5. why an existing owner is insufficient;
6. that the mechanism removes more meaningful complexity than it introduces.

Prefer: delete -> reuse -> simplify -> consolidate -> explicit invariant -> smallest
remaining mechanism. `NO_CHANGE` and deletion are valid outcomes.

## Scientific ownership

- `celnn.core.dynamics` owns the CelNN differential equation. Do not restate the ODE
  per backend.
- `celnn.core.steppers` owns integration formulas. Public execution paths call those
  steppers instead of inlining updates.
- stencil and boundary semantics have one shared definition; backends implement
  execution, not alternate mathematics.
- classical `CellularNetwork` owns its numerical dtype. Supported classical dtypes are
  `float32` and `float64`; default is `float64`.
- native classical solvers preserve the network dtype end to end.
- SciPy `solve_ivp` is a `float64`-only execution path.
- PyTorch dtype/device follow normal `torch.nn.Module` parameter/buffer ownership.
- backend/device choices may change execution and cost, not the mathematical model.
- plasticity and associative-memory mutable state must remain explicit; do not hide
  sequence/conversation state inside modules.

For scientific behavior use:
definition -> representation -> invariant -> oracle -> evidence.

Prefer evidence in this order when applicable:
known exact/analytical result -> independent reference -> differential comparison ->
metamorphic property -> algebraic/structural invariant -> justified tolerance.
Never widen a tolerance merely to make a test pass.

## Public contracts

Public API, persisted artifacts, errors, supported Python versions, optional extras,
typing metadata, dependency floors, wheel/sdist contents, and installed behavior are
product contracts.

- keep the top-level API deliberate; `celnn.__all__` is its owner;
- adding a public symbol creates a compatibility obligation;
- optional dependencies must remain optional at base import time;
- reject invalid public inputs explicitly rather than silently coercing meaning;
- no silent fallback, dtype change, approximation, overwrite, or scientific claim;
- declared lower dependency bounds must be executable or raised to the oldest version
  actually supported;
- do not introduce convergence or stability claims without a defensible definition
  and oracle.

## Semantic versus operational state

Persist model/user meaning, not the machine on which it happened to run.

Semantic examples: model inputs/state, templates, bias, activation, boundary, dtype,
user metadata.

Operational examples: backend implementation, device placement, host/device transfer
strategy, CI/runtime scheduling.

A saved model must not require the original execution device in order to preserve its
meaning.

## Persistence

`celnn.io.serialization` owns JSON artifact compatibility/versioning.

- new artifacts use the current explicit schema envelope;
- loaders validate envelope, schema version, kind, and payload;
- legacy compatibility must be small, explicit, one-way, and tested;
- unknown incompatible versions fail explicitly;
- never add a migration framework before a concrete supported migration exists;
- durable writes are atomic;
- free-form user metadata is the exception to otherwise closed structural fields.

## Python design

- Python >= 3.10; do not add an upper cap without a known incompatibility.
- Keep setuptools, pytest, Ruff, and Pyright unless a demonstrated problem requires
  replacement.
- Keep modules cohesive and dependency direction obvious.
- Prefer functions and dataclasses to framework-like abstractions.
- Use classes for real stateful lifecycle, public data contracts, or protocols.
- Avoid generic `utils`/`common`/`services` abstractions for new code; put behavior in
  its natural owner.
- Type public APIs and maintained implementation code.
- `Any` is acceptable only at genuine foreign-library seams.
- Preserve lazy optional imports while making public optional symbols statically
  discoverable.
- Validate at public, deserialization, optional-dependency, and backend boundaries.
- Preserve causal, channel, shape, dtype, gradient, and state-ownership semantics.

## Change protocol

For every material change:

1. UNDERSTAND — identify current observable/scientific meaning from authoritative
   evidence, not implementation alone.
2. CONTRACT — state what must remain true and what is intentionally changing.
3. FAILURE — identify plausible observable failure modes.
4. PROOF — choose falsifiers before implementation.
5. DESIGN — use the smallest change in the natural owner.
6. IMPLEMENT — avoid unrelated cleanup.
7. FALSIFY — actively test boundaries, invalid inputs, backend parity, gradients,
   persistence, distribution behavior, and compatibility as applicable.
8. REVIEW — inspect the diff as a maintainer: meaning, ownership, evidence, economy,
   public surface, and installed distribution.

Material architecture decisions use Archer when available. Material change execution
may use Maestro. Repository assurance may use Guardian. Do not create a CELNN-specific
generic engineering skill that duplicates those responsibilities.

## Verification

Evidence must match the claim.

- Ruff proves selected static Python invariants, not runtime behavior;
- Pyright strict proves static type contracts, not numerical correctness;
- `pyright --verifytypes` checks the installed `py.typed` public surface, not runtime
  semantics;
- coverage proves execution, not correctness and is not a project gate by default;
- unit tests prove observed cases/properties, not every scientific claim;
- differential/metamorphic/analytical tests are preferred for scientific equivalence;
- built-distribution checks prove packaging/installability, not source-checkout behavior;
- benchmarks are required before performance architecture changes.

Base-install tests must run without Torch, CuPy, SciPy, Pillow, Matplotlib, or DEAP.
Optional integrations should be tested separately without a Cartesian CI matrix.
Do not claim GPU hardware behavior unless real CUDA execution occurred.

## Documentation and project identity

Use `celnn` and `https://github.com/delbem-research/celnn` as canonical identity.
Documentation explains public meaning, constraints, failure behavior, and scientific
interpretation. Do not duplicate facts already owned by code/types/tests/metadata.
Remove stale statements instead of adding historical explanation unless users need it.

## Definition of done

A change is complete only when every affected material claim has appropriate evidence,
the public/installable artifact still behaves as intended, no invalid state is silently
accepted, and no equivalent simpler design preserves the same guarantees.

Passing tests alone is evidence, not a universal proof of correctness.
