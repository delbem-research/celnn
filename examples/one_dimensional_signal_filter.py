"""Filter a noisy one-dimensional signal with a Cellular Neural Network."""

from __future__ import annotations

import numpy as np

from celnn import CellularNetwork, SimulationConfig
from celnn.activations import tanh_activation
from celnn.domains.signal import generate_noisy_sine, plot_signal_comparison


def main() -> int:
    signal = generate_noisy_sine(
        samples=512, cycles=4.0, noise_scale=0.15, seed=7
    )

    net = CellularNetwork(
        input=signal,
        feedback=np.array([0.2, 1.0, 0.2]),
        control=np.array([0.1, 0.8, 0.1]),
        bias=0.0,
        activation=tanh_activation,
        boundary="reflect",
    )

    result = net.run(
        SimulationConfig(
            t_start=0.0,
            t_end=5.0,
            dt=0.05,
            return_trajectory=True,
        )
    )

    filtered = (
        result.trajectory_output[-1]
        if result.trajectory_output is not None
        else result.output
    )
    print("Input shape:", signal.shape)
    print("Filtered shape:", filtered.shape)

    try:
        plot_signal_comparison(
            signal,
            filtered,
            title="Noisy vs filtered signal",
            path="signal_filter.png",
        )
        print("Saved optional plot to signal_filter.png")
    except Exception as exc:
        print(f"Plot skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
