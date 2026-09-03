# Migrate from PyCNN

PyCNN is image-processing oriented and exposes task-named methods. CELNN separates array I/O, reusable templates, continuous dynamics, simulation configuration, and results.

The migration is therefore architectural rather than a one-for-one method rename.

## Replace image paths with arrays

Load image data explicitly outside the core:

```python
from celnn.domains.image import load_grayscale

u = load_grayscale("input.png")
```

The network itself receives `u`, not a file path.

## Replace task methods with templates

For example, use a built-in template as an explicit model component:

```python
from celnn import CellularNetwork
from celnn.templates import EDGE_DETECTION

network = CellularNetwork.from_template(
    EDGE_DETECTION,
    input=u,
    activation="piecewise_linear",
    boundary="reflect",
)
```

This makes coefficients inspectable and reusable rather than hiding them behind an image-specific method name.

## Map the core quantities

- PyCNN-style `A` → CELNN feedback template;
- `B` → control template;
- `Ib` → bias;
- initial condition → `initial_state`;
- time configuration → {py:class}`celnn.SimulationConfig`;
- final processing result → {py:class}`celnn.SimulationResult`.

## Validate behavior instead of assuming equivalence

A template that originated in, or was inspired by, a PyCNN example is not guaranteed to produce byte-identical output after changes in integration, boundary semantics, normalization, or activation. Port the configuration, then define the scientific or application property that must remain true and test that property explicitly.

Built-ins should be treated according to their metadata and demonstration status rather than as claims of universal or optimal parameters.
