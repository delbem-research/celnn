"""Run a simple edge-detection example on an image or synthetic input."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from celnn import CellularNetwork, SimulationConfig
from celnn.activations import piecewise_linear
from celnn.domains.image import load_grayscale, save_grayscale
from celnn.templates.image_processing import EDGE_DETECTION


def synthetic_image(shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Create a small synthetic binary image in ``[-1, 1]``."""
    image = -np.ones(shape, dtype=float)
    image[20:108, 24:40] = 1.0
    image[24:40, 20:108] = 1.0
    for offset in range(20, 108):
        image[offset, offset] = 1.0
    return image


def main(argv: list[str]) -> int:
    input_path = Path(argv[1]) if len(argv) > 1 else None
    output_path = Path(argv[2]) if len(argv) > 2 else Path("edge_output.png")

    if input_path is not None and input_path.exists():
        u = load_grayscale(input_path)
    else:
        u = synthetic_image()
        if input_path is not None:
            print(
                f"Input path '{input_path}' was not found. "
                "Using a synthetic image."
            )

    net = CellularNetwork(
        input=u,
        feedback=EDGE_DETECTION.feedback,
        control=EDGE_DETECTION.control,
        bias=EDGE_DETECTION.bias,
        activation=piecewise_linear,
        boundary="reflect",
    )

    result = net.run(SimulationConfig(t_start=0.0, t_end=10.0, dt=0.1))
    save_grayscale(result.output, output_path)
    print(f"Saved edge map to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
