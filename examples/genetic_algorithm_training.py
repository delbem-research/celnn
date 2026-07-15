from __future__ import annotations

import numpy as np

from celnn import CellularNetwork, SimulationConfig
from celnn.templates import Template
from celnn.training import GAConfig, GATrainer, TrainingDataset

SHOW_GENERATIONS = True


def _format_genome(template: Template) -> str:
    a = np.asarray(template.feedback, dtype=float).ravel()
    b = np.asarray(template.control, dtype=float).ravel()
    z = float(np.asarray(template.bias, dtype=float).reshape(-1)[0])

    def fmt(values: np.ndarray) -> str:
        return "[" + " ".join(f"{value:+.3f}" for value in values) + "]"

    return f"A={fmt(a)}  B={fmt(b)}  z={z:+.3f}"


def print_generations(result) -> None:
    """Imprime o melhor indivíduo de cada geração, passo a passo."""
    print("Evolução geração a geração (melhor indivíduo):")
    print(f"{'ger':>4}  {'melhor':>10}  {'média':>10}  genoma")
    for gen, (best_fit, mean_fit) in enumerate(
        zip(
            result.fitness_history,
            result.mean_fitness_history,
            strict=True,
        ),
        start=1,
    ):
        genome = _format_genome(result.best_per_generation[gen - 1])
        print(f"{gen:>4}  {best_fit:>10.6f}  {mean_fit:>10.6f}  {genome}")
    print()


def build_dataset() -> TrainingDataset:
    inputs = [
        np.linspace(-1.0, 1.0, 32),
        np.sin(np.linspace(0.0, 2.0 * np.pi, 32)),
    ]
    targets = [np.convolve(u, np.ones(5) / 5.0, mode="same") for u in inputs]
    return TrainingDataset.from_pairs(inputs, targets)


def main() -> int:
    seed_template = Template(
        name="smoothing_seed",
        feedback=[0.05, 0.1, 0.05],
        control=[0.0, 1.0, 0.0],
        bias=0.0,
        description="Initial smoothing-like 1D template.",
    )

    trainer = GATrainer(
        template=seed_template,
        dataset=build_dataset(),
        config=SimulationConfig(t_end=5.0, dt=0.1, solver="euler"),
        ga_config=GAConfig(
            pop_size=20,
            n_generations=10,
            bounds=(-1.0, 1.0),
            bias_bounds=(-0.5, 0.5),
            cx_prob=0.5,          # prob. de crossover
            mut_prob=0.2,         # prob. de um indivíduo sofrer mutação
            mut_sigma=0.1,        # sigma do ruído gaussiano da mutação
            tournament_size=3,    # k do torneio de seleção
            elite_size=1,         # melhor
            seed_template=seed_template,
        ),
        activation="identity",
        boundary="reflect",
        seed=42,
    )

    result = trainer.run()

    if SHOW_GENERATIONS:
        print_generations(result)

    print(f"Best fitness: {result.best_fitness:.6f}")
    print(f"Fitness history: {result.fitness_history}")
    print(f"Best template feedback: {result.best_template.feedback}")
    print(f"Best template control:  {result.best_template.control}")
    print(f"Best template bias:     {result.best_template.bias}")

    best_template = result.best_template
    print("\nEvaluating the best template on the first sample:")
    dataset = build_dataset()
    sample = dataset.samples[0]
    net = CellularNetwork(
        input=sample.input,
        feedback=best_template.feedback,
        control=best_template.control,
        bias=best_template.bias,
        activation="identity",
        boundary="reflect",
    )
    evaluation = net.run(SimulationConfig(t_end=5.0, dt=0.1, solver="euler"))
    mse = float(np.mean((evaluation.output - sample.target) ** 2))
    print(f"Final MSE: {mse:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
