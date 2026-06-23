"""Genetic-algorithm-based template trainer.

This module implements a simple genetic algorithm, powered by the
`DEAP <https://deap.readthedocs.io/en/master/>`_ library, that searches
for the ``(A, B, z)`` coefficients of a CelNN template that best fit a
:class:`celnn.training.dataset.TrainingDataset`.

Each candidate template is encoded as a flat vector of floats:

* the feedback template ``A`` (shape inherited from the seed template),
* the control template ``B`` (same shape as ``A``),
* a single scalar bias ``z``.

The fitness function is the mean squared error between the network
output produced by the candidate and the target output, plus an
optional L2 regularizer.

The trainer exposes a small, opinionated configuration surface: the
search bounds, the population size, the number of generations, the
mutation probability/sigma, the crossover probability, the tournament
size, and the elite size. Sensible defaults are provided, but tweaking
those values is the most common way to trade off between exploration
and convergence speed.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.exceptions import CelNNError
from ..core.network import CellularNetwork
from ..core.simulation import SimulationConfig
from ..core.templates import Template
from ..utils.doc import optional_dependency_message
from .dataset import TrainingDataset, TrainingSample
from .fitness import LossFn, NetworkFactory, evaluate_template, mse_loss


def _try_import_deap() -> tuple[Any, Any, Any]:
    """Import the DEAP submodules used by the trainer.

    Returns
    -------
    tuple
        ``(base, creator, tools)`` from DEAP.
    """
    try:
        from deap import base, creator, tools  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise CelNNError(
            optional_dependency_message(
                "deap", "ga", "Genetic-algorithm training"
            )
        ) from exc
    return base, creator, tools


@dataclass(slots=True)
class GATrainingResult:
    """Container returned by :meth:`GATrainer.run`."""

    best_template: Template
    best_fitness: float
    fitness_history: list[float] = field(default_factory=list)
    mean_fitness_history: list[float] = field(default_factory=list)
    best_per_generation: list[Template] = field(default_factory=list)
    n_generations: int = 0
    n_evaluations: int = 0
    population_size: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to a JSON-friendly dictionary."""
        return {
            "best_template": self.best_template.to_dict(),
            "best_fitness": float(self.best_fitness),
            "fitness_history": [
                float(value) for value in self.fitness_history
            ],
            "mean_fitness_history": [
                float(value) for value in self.mean_fitness_history
            ],
            "n_generations": int(self.n_generations),
            "n_evaluations": int(self.n_evaluations),
            "population_size": int(self.population_size),
            "config": deepcopy(self.config),
            "seed": self.seed,
        }


@dataclass(slots=True)
class GAConfig:
    """Hyperparameters controlling the genetic algorithm.

    Attributes
    ----------
    pop_size
        Number of individuals in the population.
    n_generations
        Number of generations to evolve.
    bounds
        Inclusive ``(low, high)`` bounds for the feedback and control
        coefficients.
    bias_bounds
        Optional explicit bounds for the bias term. When ``None`` the
        ``bounds`` range is used.
    cx_prob
        Probability that two individuals undergo crossover.
    mut_prob
        Probability that an individual undergoes mutation.
    mut_sigma
        Standard deviation of the Gaussian mutation. Either a single
        float (shared by all genes) or a sequence of two floats
        ``(sigma_feedback, sigma_control_and_bias)``.
    tournament_size
        Number of individuals competing in tournament selection.
    elite_size
        Number of best individuals copied verbatim to the next
        generation.
    seed_template
        Optional initial template whose coefficients are used to seed
        the first individual of the population.
    """

    pop_size: int = 30
    n_generations: int = 20
    bounds: tuple[float, float] = (-2.0, 2.0)
    bias_bounds: tuple[float, float] | None = None
    cx_prob: float = 0.5
    mut_prob: float = 0.2
    mut_sigma: float | tuple[float, float] = 0.1
    tournament_size: int = 3
    elite_size: int = 1
    seed_template: Template | None = None


class GATrainer:
    """Train a CelNN template by minimizing a fitness function with a GA.

    Parameters
    ----------
    template
        Reference :class:`Template` whose shape (feedback/control
        dimensions, dtype) defines the search space. The numeric values
        of the reference are used only when ``config.seed_template`` is
        left unset.
    dataset
        Training dataset providing the (input, target) pairs used to
        compute the fitness.
    config
        :class:`SimulationConfig` used to run the network during
        evaluation. Use a short, coarse configuration to keep the
        genetic search fast.
    ga_config
        Optional :class:`GAConfig` with the algorithm hyperparameters.
    activation
        Activation function name or callable forwarded to
        :class:`CellularNetwork`.
    boundary
        Boundary mode for the network (e.g. ``"reflect"``,
        ``"constant"``).
    boundary_value
        Value used to pad constant boundaries.
    dtype
        Numeric dtype used for the network.
    device
        Compute device (``"cpu"``, ``"gpu"``, or ``"auto"``).
    loss_fn
        Per-sample loss function. Defaults to mean squared error.
    regularization
        Multiplier for the L2 penalty applied to the candidate
        coefficients. ``0.0`` disables regularization.
    network_factory
        Optional override for the network factory. When ``None`` a
        default factory is built from the trainer configuration.
    seed
        Optional integer seed forwarded to DEAP's random sources and to
        :class:`numpy.random.default_rng`. Setting the seed makes the
        training deterministic.
    """

    def __init__(
        self,
        template: Template,
        dataset: TrainingDataset,
        config: SimulationConfig,
        *,
        ga_config: GAConfig | None = None,
        activation: str | Callable[..., np.ndarray] = "piecewise_linear",
        boundary: str = "reflect",
        boundary_value: float = 0.0,
        dtype: Any = float,
        device: str = "cpu",
        loss_fn: LossFn = mse_loss,
        regularization: float = 0.0,
        network_factory: NetworkFactory | None = None,
        seed: int | None = None,
    ) -> None:
        template.validate()
        feedback, control, _ = template.as_arrays()
        if feedback.shape != control.shape:
            raise CelNNError(
                "Seed template feedback and control must share the same "
                f"shape. Got feedback={feedback.shape} and "
                f"control={control.shape}."
            )
        self.template = template
        self.dataset = dataset
        self.config = config
        self.ga_config = ga_config or GAConfig()
        self.activation = activation
        self.boundary = boundary
        self.boundary_value = float(boundary_value)
        self.dtype = dtype
        self.device = device
        self.loss_fn = loss_fn
        self.regularization = float(regularization)
        self.seed = seed

        self._feedback_shape: tuple[int, ...] = tuple(feedback.shape)
        self._control_shape: tuple[int, ...] = tuple(control.shape)
        self._feedback_size = int(feedback.size)
        self._control_size = int(control.size)
        self._genome_size = (
            self._feedback_size + self._control_size + 1
        )

        bias_low, bias_high = self._resolve_bias_bounds()
        self._bounds: tuple[float, float] = self._normalize_bounds(
            self.ga_config.bounds
        )
        if not (bias_low <= bias_high):
            raise CelNNError(
                "bias_bounds must satisfy low <= high. "
                f"Got {bias_low} > {bias_high}."
            )
        self._bias_bounds = (float(bias_low), float(bias_high))

        self._deap_random = random.Random(seed)
        self._numpy_random = np.random.default_rng(seed)
        factory = self._build_default_factory()
        self._network_factory = (
            network_factory if network_factory is not None else factory
        )
        self._initial_genome: list[float] = self._build_initial_genome(
            template
        )

    def _resolve_bias_bounds(self) -> tuple[float, float]:
        if self.ga_config.bias_bounds is not None:
            return self._normalize_bounds(self.ga_config.bias_bounds)
        return self._normalize_bounds(self.ga_config.bounds)

    @staticmethod
    def _normalize_bounds(
        bounds: tuple[float, float],
    ) -> tuple[float, float]:
        low, high = float(bounds[0]), float(bounds[1])
        if low > high:
            raise CelNNError(
                f"Bounds must satisfy low <= high. Got ({low}, {high})."
            )
        return low, high

    def _build_initial_genome(self, template: Template) -> list[float]:
        feedback, control, _ = template.as_arrays()
        bias_value = float(
            np.asarray(template.bias, dtype=float).reshape(-1)[0]
        )
        return (
            np.asarray(feedback, dtype=float).ravel().tolist()
            + np.asarray(control, dtype=float).ravel().tolist()
            + [bias_value]
        )

    def _build_default_factory(self) -> NetworkFactory:
        def factory(sample: TrainingSample) -> CellularNetwork:
            return CellularNetwork(
                input=sample.input,
                initial_state=sample.initial_state,
                feedback=np.zeros(self._feedback_shape, dtype=float),
                control=np.zeros(self._control_shape, dtype=float),
                bias=0.0,
                activation=self.activation,
                boundary=self.boundary,
                boundary_value=self.boundary_value,
                dtype=self.dtype,
                device=self.device,
            )

        return factory

    def _decode_genome(
        self, genome: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if len(genome) != self._genome_size:
            raise CelNNError(
                f"Genome size mismatch. Expected {self._genome_size} genes, "
                f"got {len(genome)}."
            )
        array = np.asarray(genome, dtype=float)
        feedback = array[: self._feedback_size].reshape(self._feedback_shape)
        control = array[
            self._feedback_size : self._feedback_size + self._control_size
        ].reshape(self._control_shape)
        bias = float(array[-1])
        return feedback, control, bias

    def _build_template(
        self, genome: Sequence[float], name_suffix: str = ""
    ) -> Template:
        feedback, control, bias = self._decode_genome(genome)
        suffix = f"_{name_suffix}" if name_suffix else ""
        return Template(
            name=f"{self.template.name}{suffix}",
            feedback=feedback,
            control=control,
            bias=bias,
            description=(
                f"Template produced by GA training "
                f"(seed: {self.template.name})."
            ),
            tags=["ga-trained", "deap"],
            metadata={
                "ga_trainer": {
                    "pop_size": self.ga_config.pop_size,
                    "n_generations": self.ga_config.n_generations,
                    "bounds": list(self._bounds),
                    "bias_bounds": list(self._bias_bounds),
                    "mut_sigma": self._normalize_mut_sigma(),
                    "regularization": self.regularization,
                },
            },
        )

    def _normalize_mut_sigma(self) -> list[float]:
        sigma = self.ga_config.mut_sigma
        if isinstance(sigma, (int, float)):
            return [float(sigma), float(sigma)]
        if len(sigma) == 2:
            return [float(sigma[0]), float(sigma[1])]
        raise CelNNError(
            "mut_sigma must be a float or a 2-tuple "
            "(sigma_feedback, sigma_control_bias)."
        )

    def _make_network_factory(
        self, genome: Sequence[float]
    ) -> NetworkFactory:
        feedback, control, bias = self._decode_genome(genome)
        feedback_value = feedback
        control_value = control
        bias_value = bias

        def factory(sample: TrainingSample) -> CellularNetwork:
            return CellularNetwork(
                input=sample.input,
                initial_state=sample.initial_state,
                feedback=feedback_value,
                control=control_value,
                bias=bias_value,
                activation=self.activation,
                boundary=self.boundary,
                boundary_value=self.boundary_value,
                dtype=self.dtype,
                device=self.device,
            )

        return factory

    def _evaluate_individual(self, individual: list[float]) -> tuple[float]:
        factory = self._make_network_factory(individual)
        fitness_value = evaluate_template(
            network_factory=factory,
            config=self.config,
            dataset=self.dataset,
            loss_fn=self.loss_fn,
            regularization=self.regularization,
        )
        return (float(fitness_value),)

    def _build_toolbox(self) -> tuple[Any, Any, Any]:
        base, creator, tools = _try_import_deap()
        # Always create a fresh FitnessMin class so re-using the same
        # trainer with different datasets does not leak state.
        if not hasattr(creator, "celnn_fitness_min"):
            creator.create("celnn_fitness_min", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "celnn_individual"):
            creator.create(
                "celnn_individual",
                list,
                fitness=creator.celnn_fitness_min,
            )

        toolbox = base.Toolbox()
        low, high = self._bounds
        bias_low, bias_high = self._bias_bounds
        sigmas = self._normalize_mut_sigma()
        feedback_size = self._feedback_size
        control_size = self._control_size
        total = self._genome_size

        def init_individual() -> Any:
            individual = [
                self._deap_random.uniform(low, high)
                for _ in range(total)
            ]
            # Pin the last gene to the bias range.
            individual[-1] = self._deap_random.uniform(bias_low, bias_high)
            return creator.celnn_individual(individual)

        def init_seeded() -> Any:
            seed_genome = list(self._initial_genome)
            feedback_end = feedback_size + control_size
            for index in range(total - 1):
                if index < feedback_end:
                    mutation = self._deap_random.gauss(0.0, sigmas[0])
                    seed_genome[index] = min(
                        max(seed_genome[index] + mutation, low), high
                    )
            mutation = self._deap_random.gauss(0.0, sigmas[1])
            seed_genome[-1] = min(
                max(seed_genome[-1] + mutation, bias_low), bias_high
            )
            return creator.celnn_individual(seed_genome)

        def init_population() -> list[Any]:
            population = [
                init_individual() for _ in range(self.ga_config.pop_size)
            ]
            if self.ga_config.seed_template is not None:
                population[0] = init_seeded()
            return population

        toolbox.register("population", init_population)
        toolbox.register("evaluate", self._evaluate_individual)
        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.ga_config.tournament_size,
        )
        return toolbox, base, tools

    def _clip_individual(self, individual: list[float]) -> list[float]:
        low, high = self._bounds
        bias_low, bias_high = self._bias_bounds
        total = self._genome_size
        feedback_size = self._feedback_size
        control_size = self._control_size
        for index in range(total - 1):
            individual[index] = min(max(individual[index], low), high)
        if feedback_size > 0 and control_size > 0:
            for index in range(feedback_size, feedback_size + control_size):
                individual[index] = min(max(individual[index], low), high)
        individual[-1] = min(max(individual[-1], bias_low), bias_high)
        return individual

    def _apply_mutation(
        self, individual: list[float], tools: Any
    ) -> tuple[list[float]]:
        mut_sigma = self.ga_config.mut_sigma
        feedback_size = self._feedback_size

        if isinstance(mut_sigma, (int, float)):
            sigma_feedback = float(mut_sigma)
            sigma_control_bias = float(mut_sigma)
        else:
            sigma_feedback = float(mut_sigma[0])
            sigma_control_bias = float(mut_sigma[1])

        mutated = list(individual)
        indpb = 1.0 / max(self._genome_size, 1)
        for index in range(self._genome_size):
            if self._deap_random.random() < indpb:
                if index < feedback_size:
                    sigma = sigma_feedback
                else:
                    sigma = sigma_control_bias
                mutated[index] += self._numpy_random.normal(0.0, sigma)
        self._clip_individual(mutated)
        return (mutated,)

    def _apply_crossover(
        self, ind1: list[float], ind2: list[float], tools: Any
    ) -> tuple[list[float], list[float]]:
        child1, child2 = tools.cxTwoPoint(ind1, ind2)
        self._clip_individual(child1)
        self._clip_individual(child2)
        return child1, child2

    def run(self) -> GATrainingResult:
        """Run the genetic algorithm and return the best template found."""
        # DEAP operators (selTournament, cxTwoPoint) draw from the global
        # random module, so it must be seeded as well for determinism.
        if self.seed is not None:
            random.seed(self.seed)
        toolbox, base, tools = self._build_toolbox()

        population = toolbox.population()
        n_evaluations = 0

        invalid_individuals = [
            individual
            for individual in population
            if not individual.fitness.valid
        ]
        fitnesses = list(map(toolbox.evaluate, invalid_individuals))
        for individual, fitness in zip(
            invalid_individuals, fitnesses, strict=True
        ):
            individual.fitness.values = fitness
        n_evaluations += len(invalid_individuals)

        best_per_generation: list[Template] = []
        fitness_history: list[float] = []
        mean_fitness_history: list[float] = []

        for _ in range(self.ga_config.n_generations):
            elite = tools.selBest(population, k=self.ga_config.elite_size)
            elite = [deepcopy(individual) for individual in elite]

            offspring = tools.selTournament(
                population,
                k=len(population) - self.ga_config.elite_size,
                tournsize=self.ga_config.tournament_size,
            )
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(
                offspring[::2],
                offspring[1::2],
                strict=False,
            ):
                if self._deap_random.random() < self.ga_config.cx_prob:
                    new1, new2 = self._apply_crossover(child1, child2, tools)
                    child1[:] = new1
                    child2[:] = new2
                    del child1.fitness.values, child2.fitness.values

            for mutant in offspring:
                if self._deap_random.random() < self.ga_config.mut_prob:
                    mutated = self._apply_mutation(mutant, tools)
                    mutant[:] = mutated[0]
                    del mutant.fitness.values

            invalid = [
                individual
                for individual in offspring
                if not individual.fitness.valid
            ]
            fitnesses = list(map(toolbox.evaluate, invalid))
            for individual, fitness in zip(invalid, fitnesses, strict=True):
                individual.fitness.values = fitness
            n_evaluations += len(invalid)

            population[:] = elite + offspring

            best = tools.selBest(population, k=1)[0]
            best_per_generation.append(
                self._build_template(
                    best,
                    name_suffix=f"gen{len(best_per_generation)}",
                )
            )
            best_fitness = float(best.fitness.values[0])
            fitness_history.append(best_fitness)
            mean_fitness = float(
                sum(ind.fitness.values[0] for ind in population)
                / len(population)
            )
            mean_fitness_history.append(mean_fitness)

        best = tools.selBest(population, k=1)[0]
        best_template = self._build_template(best, name_suffix="best")
        best_fitness = float(best.fitness.values[0])
        best_per_generation.append(best_template)

        return GATrainingResult(
            best_template=best_template,
            best_fitness=best_fitness,
            fitness_history=fitness_history,
            mean_fitness_history=mean_fitness_history,
            best_per_generation=best_per_generation,
            n_generations=self.ga_config.n_generations,
            n_evaluations=n_evaluations,
            population_size=self.ga_config.pop_size,
            config=self._config_summary(),
            seed=self.seed,
        )

    def _config_summary(self) -> dict[str, Any]:
        return {
            "ga": {
                "pop_size": self.ga_config.pop_size,
                "n_generations": self.ga_config.n_generations,
                "bounds": list(self._bounds),
                "bias_bounds": list(self._bias_bounds),
                "cx_prob": self.ga_config.cx_prob,
                "mut_prob": self.ga_config.mut_prob,
                "mut_sigma": self._normalize_mut_sigma(),
                "tournament_size": self.ga_config.tournament_size,
                "elite_size": self.ga_config.elite_size,
                "regularization": self.regularization,
                "loss_fn": self.loss_fn.__name__,
                "seed": self.seed,
            },
            "simulation": self.config.to_dict(),
            "feedback_shape": list(self._feedback_shape),
            "control_shape": list(self._control_shape),
        }
