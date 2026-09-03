# Choose solvers and execution backends

Solver and backend are independent choices. A **solver** specifies how time evolution is approximated; a **backend** specifies how local array operations are executed.

## CPU reference execution

```python
network = CellularNetwork(input=u, device="cpu")
```

This selects the NumPy backend and requires only the base installation.

## Require CuPy/CUDA execution

```bash
python -m pip install "celnn[gpu]"
```

```python
network = CellularNetwork(input=u, device="gpu")
```

`device="cuda"` is an alias for the same required CuPy path. If CuPy/CUDA is unavailable, required GPU execution fails instead of silently pretending that CPU execution was GPU execution.

## Allow automatic fallback

```python
network = CellularNetwork(input=u, device="auto")
```

`auto` tries the CuPy backend when it reports itself available and otherwise uses NumPy. For experiments where the execution backend is part of the evidence, prefer an explicit device and record result metadata rather than relying on fallback.

## Use SciPy integration

```bash
python -m pip install "celnn[scipy]"
```

```python
config = SimulationConfig(solver="solve_ivp", t_end=2.0, dt=0.02)
result = network.run(config)
```

SciPy changes time integration, not the CELNN equation.

## PyTorch is a separate public execution model

The optional {py:class}`celnn.DifferentiableCellularNetwork` uses PyTorch tensors and parameter ownership for differentiable one-dimensional evolution. It is not selected by `CellularNetwork(device=...)`. See {doc}`differentiable`.

## Evidence rule

Backend parity established by tests supports semantic equivalence within the tested contract. It does not establish performance or hardware claims. In particular, do not claim CUDA performance from a machine that did not execute on CUDA hardware.
