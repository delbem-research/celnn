"""Numerical solvers for cellular dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .exceptions import OptionalDependencyError, SolverError
from .result import SimulationResult

if TYPE_CHECKING:
    from .network import CellularNetwork
    from .simulation import SimulationConfig


def solve(
    network: "CellularNetwork", config: "SimulationConfig"
) -> SimulationResult:
    """Solve a cellular network trajectory using the configured method."""
    if config.solver == "euler":
        return _solve_euler(network, config)
    if config.solver in {
        "semi_implicit_euler",
        "semi-implicit-euler",
        "semi_implicit",
    }:
        return _solve_semi_implicit_euler(network, config)
    if config.solver == "solve_ivp":
        return _solve_ivp(network, config)
    raise SolverError(
        f"Unknown solver '{config.solver}'. Expected "
        "'euler', 'semi_implicit_euler', or 'solve_ivp'."
    )


def _maybe_print_progress(
    config: "SimulationConfig", index: int, total: int
) -> None:
    if config.progress:
        print(f"[celnn] step {index}/{total}")


def _convergence_info(
    final_state: np.ndarray, previous_state: np.ndarray
) -> dict[str, float]:
    delta = np.max(np.abs(final_state - previous_state))
    return {
        "max_abs_state_delta": float(delta),
        "approx_converged": bool(delta < 1e-6),
    }


def _store_result(
    network: "CellularNetwork",
    state: np.ndarray,
    times: list[float],
    trajectory_state: list[np.ndarray] | None,
    trajectory_output: list[np.ndarray] | None,
    previous_state: np.ndarray,
) -> SimulationResult:
    final_output = network.output(state)
    metadata = {
        "solver": network._last_solver,
        "boundary": network.boundary,
        "backend": network.backend.name,
        "device": network.device,
        "shape": state.shape,
    }
    if trajectory_state is not None and trajectory_output is not None:
        time_array = np.asarray(times, dtype=float)
        return SimulationResult(
            state=state.copy(),
            output=final_output.copy(),
            time=time_array,
            trajectory_state=np.stack(trajectory_state, axis=0),
            trajectory_output=np.stack(trajectory_output, axis=0),
            metadata=metadata,
            convergence=_convergence_info(state, previous_state),
        )
    return SimulationResult(
        state=state.copy(),
        output=final_output.copy(),
        time=np.asarray([times[-1]], dtype=float),
        metadata=metadata,
        convergence=_convergence_info(state, previous_state),
    )


def _solve_euler(
    network: "CellularNetwork", config: "SimulationConfig"
) -> SimulationResult:
    times = config.time_points()
    state = network.state.copy()
    previous_state = state.copy()
    trajectory_state: list[np.ndarray] | None = (
        [] if config.return_trajectory else None
    )
    trajectory_output: list[np.ndarray] | None = (
        [] if config.return_trajectory else None
    )
    stored_times: list[float] = []

    if config.return_trajectory:
        trajectory_state.append(state.copy())
        trajectory_output.append(network.output(state))
        stored_times.append(float(times[0]))

    total_steps = max(len(times) - 1, 1)
    for index in range(1, len(times)):
        dt = float(times[index] - times[index - 1])
        previous_state = state.copy()
        state = state + dt * network.derivative(state)
        _maybe_print_progress(config, index, total_steps)
        if config.return_trajectory and (
            index % config.store_every == 0 or index == len(times) - 1
        ):
            trajectory_state.append(state.copy())
            trajectory_output.append(network.output(state))
            stored_times.append(float(times[index]))

    network.state = state.copy()
    network._last_solver = "euler"
    result_times = (
        stored_times if config.return_trajectory else [float(times[-1])]
    )
    return _store_result(
        network,
        state,
        result_times,
        trajectory_state,
        trajectory_output,
        previous_state,
    )


def _solve_semi_implicit_euler(
    network: "CellularNetwork", config: "SimulationConfig"
) -> SimulationResult:
    times = config.time_points()
    state = network.state.copy()
    previous_state = state.copy()
    trajectory_state: list[np.ndarray] | None = (
        [] if config.return_trajectory else None
    )
    trajectory_output: list[np.ndarray] | None = (
        [] if config.return_trajectory else None
    )
    stored_times: list[float] = []

    if config.return_trajectory:
        trajectory_state.append(state.copy())
        trajectory_output.append(network.output(state))
        stored_times.append(float(times[0]))

    total_steps = max(len(times) - 1, 1)
    for index in range(1, len(times)):
        dt = float(times[index] - times[index - 1])
        previous_state = state.copy()
        drive = network.drive(state)
        state = (state + dt * drive) / (1.0 + dt)
        _maybe_print_progress(config, index, total_steps)
        if config.return_trajectory and (
            index % config.store_every == 0 or index == len(times) - 1
        ):
            trajectory_state.append(state.copy())
            trajectory_output.append(network.output(state))
            stored_times.append(float(times[index]))

    network.state = state.copy()
    network._last_solver = "semi_implicit_euler"
    result_times = (
        stored_times if config.return_trajectory else [float(times[-1])]
    )
    return _store_result(
        network,
        state,
        result_times,
        trajectory_state,
        trajectory_output,
        previous_state,
    )


def _solve_ivp(
    network: "CellularNetwork", config: "SimulationConfig"
) -> SimulationResult:
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OptionalDependencyError(
            "The 'solve_ivp' solver requires SciPy. "
            "Install it with `pip install celnn[scipy]`."
        ) from exc

    times = config.time_points()
    initial_state = network.state.astype(float, copy=True)
    flat_initial = initial_state.ravel()

    def rhs(_t: float, flat_state: np.ndarray) -> np.ndarray:
        reshaped = flat_state.reshape(initial_state.shape)
        return network.derivative(reshaped).ravel()

    sol = solve_ivp(
        rhs,
        t_span=(float(times[0]), float(times[-1])),
        y0=flat_initial,
        t_eval=times,
        vectorized=False,
    )
    if not sol.success:
        raise SolverError(f"SciPy solve_ivp failed: {sol.message}")

    trajectory = sol.y.T.reshape((-1,) + initial_state.shape)
    final_state = trajectory[-1].copy()
    previous_state = (
        trajectory[-2].copy() if len(trajectory) > 1 else initial_state.copy()
    )
    network.state = final_state.copy()
    network._last_solver = "solve_ivp"

    if config.return_trajectory:
        stride = config.store_every
        stored_states = list(trajectory[::stride].copy())
        stored_times = list(sol.t[::stride].astype(float))
        if stored_times[-1] != float(sol.t[-1]):
            stored_states.append(trajectory[-1].copy())
            stored_times.append(float(sol.t[-1]))
        stored_outputs = [network.output(state) for state in stored_states]
        return SimulationResult(
            state=final_state.copy(),
            output=network.output(final_state),
            time=np.asarray(stored_times, dtype=float),
            trajectory_state=np.stack(stored_states, axis=0),
            trajectory_output=np.stack(stored_outputs, axis=0),
            metadata={
                "solver": "solve_ivp",
                "boundary": network.boundary,
                "shape": final_state.shape,
            },
            convergence=_convergence_info(final_state, previous_state),
        )

    return SimulationResult(
        state=final_state.copy(),
        output=network.output(final_state),
        time=np.asarray([float(sol.t[-1])], dtype=float),
        metadata={
            "solver": "solve_ivp",
            "boundary": network.boundary,
            "shape": final_state.shape,
        },
        convergence=_convergence_info(final_state, previous_state),
    )
