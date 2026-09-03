# Diffusion, filtering, and pattern formation

These ideas are related through spatial operators and mode dynamics, but they are not interchangeable labels. A disciplined interpretation begins with the local operator and asks what it does to spatial modes.

## Filtering in a linear operating regime

If the output map is linear over the relevant state range, a translation-invariant CELNN can be analyzed as a linear spatial dynamical system. Crounse and Chua use this view to derive spatial-frequency responses and explain low-pass filtering, time-varying filtering, and several standard CNN templates. See {ref}`crounse-chua-1995`.

A steady-state filter describes the asymptotic input/output relation of a stable linearized system. A transient filter uses the fact that different spatial modes decay or grow at different rates and stops the evolution at a chosen time.

## Diffusion-like stencils

Discrete second-difference or Laplacian-like stencils are local approximations to diffusion operators after the sign and scaling are chosen appropriately. Crounse and Chua explicitly discuss diffusion and Laplace templates in this spatial-operator context.

In CELNN, recognizing a Laplacian-like coefficient pattern is only the first step. Whether the full dynamics behave diffusively depends on whether that operator is in feedback or control, its sign, the leak term, activation regime, and boundary conditions.

## Pattern selection through unstable modes

A homogeneous state can lose stability when some spatial modes grow instead of decay. Small perturbations then become selectively amplified, producing visible structure. The 1995 tutorial shows how this mechanism appears in simple CNN dynamics and relates it to pattern formation.

This is a stronger explanation than saying merely that “nonlinearity creates patterns”: the relevant question is which modes are amplified, under what operating assumptions, and what eventually limits their growth.

## Relation to reaction–diffusion

Turing showed that diffusion coupled to local reaction dynamics can destabilize a homogeneous equilibrium and create spatial differentiation; see {ref}`turing-1952`.

That result supplies an important conceptual lineage for diffusion-driven pattern formation. It does **not** imply that every CELNN pattern experiment is a Turing pattern. To make that claim for a particular model, the reaction/diffusion correspondence and instability conditions would need to be derived for that model.

## Nonlinear saturation changes the late-time story

Linear mode analysis is usually local to an operating regime. Once growing modes push cells into nonlinear regions, saturation and nonlinear feedback determine amplitude selection and final morphology. A linearized prediction can therefore explain onset without fully predicting the eventual pattern.
