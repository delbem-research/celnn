# Use CELNN

These guides answer concrete implementation questions. They assume the core model is already understood; when a choice needs scientific interpretation, the guide links back to **Learn** or **Explanation** rather than duplicating that material.

## Installation by capability

The base simulator requires only NumPy:

```bash
python -m pip install celnn
```

Install optional capabilities only when needed:

```bash
python -m pip install "celnn[scipy]"   # SciPy solve_ivp
python -m pip install "celnn[gpu]"     # CuPy/CUDA reference backend
python -m pip install "celnn[torch]"   # differentiable, plasticity, memory
python -m pip install "celnn[image]"   # Pillow image I/O
python -m pip install "celnn[viz]"     # Matplotlib visualization helpers
python -m pip install "celnn[ga]"      # DEAP genetic training
```

Plain `import celnn` remains valid without those optional dependencies.

## Core workflow

A typical reference-simulator workflow is:

1. prepare an input array;
2. choose or create a template;
3. construct {py:class}`celnn.CellularNetwork`;
4. configure integration with {py:class}`celnn.SimulationConfig`;
5. run the system;
6. inspect {py:class}`celnn.SimulationResult`.

Start with {doc}`create-network` and {doc}`run-simulation`.

## Advanced capabilities

Template optimization, differentiable PyTorch evolution, plasticity, and associative memory are independent capabilities. Use them because the problem requires them, not as mandatory layers around a basic CELNN simulation.
