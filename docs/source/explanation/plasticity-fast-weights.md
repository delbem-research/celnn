# Plasticity and fast weights

CELNN’s plasticity module is not part of the canonical cellular ODE. It is a reusable PyTorch subsystem for maintaining transient, per-sample weight changes while keeping ordinary model parameters separate.

## Correlation-driven local updates

A Hebbian-style update increases a weight in proportion to paired pre- and post-synaptic activity. In CELNN’s batched matrix form, the core correlation term is the mean outer product

$$
E[yx^T].
$$

The library’s `HebbianRule` combines this with configurable decay. This equation is documented as the software contract; no claim is made that a particular biological synapse follows it.

## Why naive Hebbian growth needs control

Repeated positive correlation can grow weights without bound. Oja derives a normalized Hebbian-type rule with a response-dependent stabilizing term and shows principal-component behavior for the analyzed linear neuron model; see {ref}`oja-1982`.

CELNN’s `OjaRule` implements the corresponding local structure in batched matrix form. Its optional decay and the surrounding `memory_limit` are additional software mechanisms and should not be attributed to Oja’s derivation.

## Fast weights as transient state

Schmidhuber describes fast-changing weights as a form of short-term memory controlled by another system, with temporary associations represented in weights rather than only in recurrent activations; see {ref}`schmidhuber-1992`.

CELNN adopts the useful separation between slow parameters and transient weight state but uses its own API:

$$
W_{effective}=W_{slow}+\alpha H_{fast}.
$$

The caller owns `H_fast` through `PlasticityState`.

## Explicit state is an architectural decision

Keeping memory outside hidden module mutation has practical consequences:

- batch elements cannot silently share state;
- concurrent sequences can maintain independent memories;
- reset and detach boundaries are visible in caller code;
- checkpoint policy can distinguish learned parameters from transient state.

This is an implementation design choice, not a scientific theorem.

## Plasticity is compositional

A `PlasticLinear` can be used in a feed-forward model, recurrent model, or alongside a differentiable CELNN. When used with CELNN, cross-channel plastic computation is kept outside the canonical cellular derivative and may be supplied explicitly as extra drive where supported.

That separation preserves one source of truth for CELNN dynamics while allowing richer adaptive models to be assembled around it.
