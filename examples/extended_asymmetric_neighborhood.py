"""Run an arbitrary 11x15 template over a 1080x1080 matrix.

The neighborhood reaches five cells up and down, seven cells left and
right, and only one cell along each diagonal.
"""

from __future__ import annotations

import numpy as np

from celnn import CellularNetwork, SimulationConfig
from celnn.templates import Template

GRID_SHAPE = (1080, 1080)
VERTICAL_RADIUS = 5
HORIZONTAL_RADIUS = 7
TEMPLATE_SHAPE = (
    2 * VERTICAL_RADIUS + 1,
    2 * HORIZONTAL_RADIUS + 1,
)


def build_arbitrary_template() -> Template:
    """Create an extended cross stencil with immediate diagonals."""
    feedback = np.zeros(TEMPLATE_SHAPE, dtype=np.float32)
    control = np.zeros(TEMPLATE_SHAPE, dtype=np.float32)
    center_row = VERTICAL_RADIUS
    center_col = HORIZONTAL_RADIUS

    feedback[center_row, center_col] = 0.40
    control[center_row, center_col] = 0.50

    for distance in range(1, VERTICAL_RADIUS + 1):
        feedback_weight = 0.02 / distance
        control_weight = 0.03 / distance
        feedback[center_row - distance, center_col] = feedback_weight
        feedback[center_row + distance, center_col] = feedback_weight
        control[center_row - distance, center_col] = control_weight
        control[center_row + distance, center_col] = control_weight

    for distance in range(1, HORIZONTAL_RADIUS + 1):
        feedback_weight = 0.015 / distance
        control_weight = 0.02 / distance
        feedback[center_row, center_col - distance] = feedback_weight
        feedback[center_row, center_col + distance] = feedback_weight
        control[center_row, center_col - distance] = control_weight
        control[center_row, center_col + distance] = control_weight

    for row_offset in (-1, 1):
        for col_offset in (-1, 1):
            row = center_row + row_offset
            col = center_col + col_offset
            feedback[row, col] = 0.01
            control[row, col] = 0.015

    return Template(
        name="extended_asymmetric_neighborhood",
        feedback=feedback,
        control=control,
        bias=0.0,
        description=(
            "Arbitrary 2D template with radius 5 vertically, radius 7 "
            "horizontally, and radius 1 on the diagonals."
        ),
        tags=["demo", "extended-neighborhood", "asymmetric-radius"],
    )


def build_input_matrix() -> np.ndarray:
    """Create a deterministic 1080x1080 matrix in the interval [-1, 1]."""
    rows = np.linspace(-1.0, 1.0, GRID_SHAPE[0], dtype=np.float32)[:, None]
    cols = np.linspace(-1.0, 1.0, GRID_SHAPE[1], dtype=np.float32)[None, :]
    return np.sin(np.pi * rows) * np.cos(np.pi * cols)


def main() -> int:
    template = build_arbitrary_template()
    input_matrix = build_input_matrix()

    network = CellularNetwork.from_template(
        template=template,
        input=input_matrix,
        activation="tanh_activation",
        boundary="reflect",
        dtype=np.float32,
        device="auto",
        metadata={"example": "extended_asymmetric_neighborhood"},
    )
    result = network.run(
        SimulationConfig(
            t_end=0.1,
            dt=0.1,
            solver="semi_implicit_euler",
        )
    )

    print("Input shape:", input_matrix.shape)
    print("Template shape:", template.feedback.shape)
    print("Vertical reach:", VERTICAL_RADIUS)
    print("Horizontal reach:", HORIZONTAL_RADIUS)
    print("Diagonal reach: 1")
    print("Backend:", result.metadata["backend"])
    print("Output shape:", result.output.shape)
    print(
        "Output range:",
        float(result.output.min()),
        float(result.output.max()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
