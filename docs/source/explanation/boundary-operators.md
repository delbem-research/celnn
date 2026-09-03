# Boundaries change the operator

A finite array cannot evaluate a centered stencil at its edges without defining values outside the stored domain. A boundary condition supplies those values, and therefore changes the actual operator being applied.

## One stencil, different edge equations

For the second-difference stencil

$$
[1,-2,1],
$$

the first interior-like equation would require a value $x_{-1}$. Under different boundary rules:

- constant zero makes $x_{-1}=0$;
- wrap makes $x_{-1}=x_{N-1}$;
- nearest makes $x_{-1}=x_0$;
- reflection rules construct another mirrored extension.

The coefficient vector did not change, but the equation at the boundary did.

## The boundary is part of model semantics

For a one-shot filter, boundary effects may be confined near an edge. In a recurrent CELNN, an edge change alters a derivative; that state change can then affect neighboring derivatives in later evolution. Boundary assumptions can therefore propagate into the interior.

This is why boundary mode belongs in serialized semantic network configuration and result metadata.

## Backend names are not the public contract

Numerical libraries do not use reflection terminology consistently. CELNN defines public meanings for `reflect` and `mirror` and maps them to backend-specific names so that a backend change does not silently swap the intended extension.

The public semantic contract is more important than the spelling used by NumPy or SciPy internally.

## Boundary choice should come from the modeled domain

Use periodic wrapping for a periodic domain, not because it removes an inconvenient edge artifact. Use a constant exterior when a fixed outside value is part of the model. Use reflection when a mirrored extension is a defensible approximation.

Changing boundaries to make a result “look better” changes the mathematical problem unless that change is independently justified.
