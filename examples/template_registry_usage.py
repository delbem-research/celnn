"""Register a template and benchmark a *gigantic* cellular network on GPU.

This example is meant to stress a large accelerator (e.g. an A100 on a
DGX) and measure raw throughput of the CelNN simulation loop.

Everything is tunable via environment variables so you can push the grid
until you fill the GPU:

    CELNN_DEVICE     execution device: auto | gpu | cuda | cpu  (default auto)
    CELNN_GRID_SIZE  side length N of the NxN grid               (default 8192)
    CELNN_DTYPE      float32 | float64
                    (default float32)
    CELNN_T_END      integration end time                        (default 5.0)
    CELNN_DT         integration step                            (default 0.1)
    CELNN_WARMUP     1 to run one untimed warm-up first          (default 1)

Throughput is reported as cell-updates per second
(grid_cells * n_steps / elapsed).
"""

from __future__ import annotations

import os
import time

import numpy as np

from celnn import CellularNetwork, SimulationConfig
from celnn.templates import Template, TemplateRegistry


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _build_network(size: int, device: str, dtype: type) -> CellularNetwork:
    registry = TemplateRegistry()
    registry.register(
        Template(
            name="custom_edge_detector",
            feedback=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            control=[
                [-1.0, -1.0, -1.0],
                [-1.0, 8.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
            bias=-1.0,
            description="Custom edge-like detector",
            tags=["image", "edge", "demo", "benchmark"],
        )
    )
    template = registry.get("custom_edge_detector")

    # A big deterministic input: a bright block on a dark background.
    input_grid = np.zeros((size, size), dtype=dtype)
    q = size // 4
    input_grid[q : 3 * q, q : 3 * q] = 1.0

    return CellularNetwork.from_template(
        template=template,
        input=input_grid,
        activation="piecewise_linear",
        boundary="reflect",
        dtype=dtype,
        device=device,
    )


def _synchronize() -> None:
    """Best-effort GPU sync so timings are accurate on CuPy backends."""
    try:
        import cupy as cp

        cp.cuda.Device().synchronize()
    except Exception:
        pass


def main() -> int:
    device = _env("CELNN_DEVICE", "auto")
    size = int(_env("CELNN_GRID_SIZE", "8192"))
    dtype = (
        np.float64
        if _env("CELNN_DTYPE", "float32") == "float64"
        else np.float32
    )
    t_end = float(_env("CELNN_T_END", "5.0"))
    dt = float(_env("CELNN_DT", "0.1"))
    warmup = _env("CELNN_WARMUP", "1") == "1"

    n_steps = int(round(t_end / dt))
    cells = size * size

    print("=== CelNN GPU benchmark ===")
    print(f"requested device : {device}")
    print(f"grid             : {size} x {size}  ({cells:,} cells)")
    print(f"dtype            : {np.dtype(dtype).name}")
    print(f"t_end / dt       : {t_end} / {dt}  ({n_steps} steps)")

    net = _build_network(size=size, device=device, dtype=dtype)
    print(f"resolved device  : {net.device}")
    print(f"backend          : {net.backend.name}")
    print(
        f"state memory     : ~{cells * np.dtype(dtype).itemsize / 1e9:.2f} GB "
        "per full-grid array"
    )

    config = SimulationConfig(t_end=t_end, dt=dt, solver="euler")

    if warmup:
        print("\nwarm-up run (untimed)...")
        net.run(config)
        _synchronize()

    print("timed run...")
    start = time.perf_counter()
    result = net.run(config)
    _synchronize()
    elapsed = time.perf_counter() - start

    updates = cells * n_steps
    throughput = updates / elapsed if elapsed > 0 else float("nan")

    print("\n=== results ===")
    print("template            :", net.metadata.get("template_name", "custom"))
    print("output shape        :", result.output.shape)
    print(f"elapsed             : {elapsed:.4f} s")
    print(f"cell-updates        : {updates:,}")
    print(f"throughput          : {throughput / 1e9:.3f} G cell-updates/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
