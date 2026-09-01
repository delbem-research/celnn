"""Tests for the genetic-algorithm-based template trainer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("deap", reason="DEAP is optional")

from celnn import SimulationConfig
from celnn.core.exceptions import CelNNError, ShapeMismatchError
from celnn.core.templates import Template
from celnn.training import (
    GAConfig,
    GATrainer,
    GATrainingResult,
    TrainingDataset,
    TrainingSample,
    evaluate_template,
    l2_penalty,
    mse_loss,
)
from celnn.training.dataset import TrainingDataset as _Dataset


def test_training_sample_validates_shape():
    u = np.zeros((4, 4))
    target = np.zeros((3, 3))
    with pytest.raises(ShapeMismatchError):
        TrainingSample(input=u, target=target)


def test_training_sample_validates_initial_state_shape():
    with pytest.raises(ShapeMismatchError, match="initial_state"):
        TrainingSample(
            input=np.zeros(4),
            target=np.zeros(4),
            initial_state=np.zeros(3),
        )


def test_training_dataset_from_pairs():
    u1 = np.linspace(-1, 1, 8)
    u2 = np.linspace(0, 1, 8)
    dataset = TrainingDataset.from_pairs([u1, u2], [u1.copy(), u2.copy()])
    assert len(dataset) == 2
    assert dataset.samples[0].input.shape == u1.shape


def test_training_dataset_rejects_mismatched_shapes():
    u1 = np.zeros((4, 4))
    u2 = np.zeros((3, 3))
    with pytest.raises(ShapeMismatchError):
        TrainingDataset.from_pairs([u1, u2], [u1.copy(), u2.copy()])


def test_training_dataset_rejects_empty():
    with pytest.raises(ValueError):
        TrainingDataset(samples=[])


def test_training_dataset_rejects_inconsistent_inputs():
    u1 = np.zeros((4, 4))
    u2 = np.zeros((3, 3))
    with pytest.raises(ShapeMismatchError):
        _Dataset(
            samples=[
                TrainingSample(input=u1, target=u1.copy()),
                TrainingSample(input=u2, target=u2.copy()),
            ]
        )


def test_mse_loss_zero_for_perfect_match():
    array = np.linspace(-1, 1, 8)
    assert mse_loss(array, array) == 0.0


def test_mse_loss_returns_expected_value():
    target = np.array([1.0, 0.0, -1.0])
    output = np.array([0.0, 0.0, 0.0])
    assert mse_loss(target, output) == pytest.approx(2.0 / 3.0)


def test_mse_loss_rejects_shape_mismatch():
    with pytest.raises(CelNNError):
        mse_loss(np.zeros(4), np.zeros(5))


def test_l2_penalty_is_sum_of_squares():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.5, 0.5]])
    z = np.array([2.0])
    assert l2_penalty(a, b, z) == pytest.approx(2.0 + 0.5 + 4.0)


def _make_1d_identity_template() -> Template:
    return Template(
        name="identity_1d_seed",
        feedback=[0.0, 1.0, 0.0],
        control=[0.0, 1.0, 0.0],
        bias=0.0,
    )


def _make_dataset() -> TrainingDataset:
    u1 = np.linspace(-1, 1, 8)
    u2 = np.linspace(-0.5, 0.5, 8)
    return TrainingDataset.from_pairs([u1, u2], [u1.copy(), u2.copy()])


def _make_ga_config(pop_size: int = 8, n_generations: int = 3) -> GAConfig:
    return GAConfig(
        pop_size=pop_size,
        n_generations=n_generations,
        bounds=(-1.0, 1.0),
        bias_bounds=(-0.5, 0.5),
        seed_template=_make_1d_identity_template(),
    )


def test_ga_trainer_runs_and_returns_result():
    trainer = GATrainer(
        template=_make_1d_identity_template(),
        dataset=_make_dataset(),
        config=SimulationConfig(t_end=1.0, dt=0.1, solver="euler"),
        ga_config=_make_ga_config(),
        activation="identity",
        boundary="reflect",
        seed=1234,
    )
    result = trainer.run()
    assert isinstance(result, GATrainingResult)
    assert result.best_fitness >= 0.0
    assert result.n_generations == 3
    assert result.population_size == 8
    assert len(result.fitness_history) == 3
    assert len(result.mean_fitness_history) == 3
    assert len(result.best_per_generation) == 4
    assert result.best_template.feedback.shape == (3,)


def test_ga_trainer_mean_fitness_improves_with_generations():
    template = _make_1d_identity_template()
    dataset = _make_dataset()
    trainer = GATrainer(
        template=template,
        dataset=dataset,
        config=SimulationConfig(t_end=5.0, dt=0.1, solver="euler"),
        ga_config=GAConfig(
            pop_size=10,
            n_generations=5,
            bounds=(-1.0, 1.0),
            bias_bounds=(-0.5, 0.5),
            seed_template=template,
        ),
        activation="identity",
        boundary="reflect",
        seed=11,
    )
    result = trainer.run()
    # The mean fitness of the final generation should be no worse than
    # the mean fitness of the first one. This is a more reliable proxy
    # for "the GA is making progress" than comparing best vs best
    # across short runs.
    assert (
        result.mean_fitness_history[-1]
        <= result.mean_fitness_history[0] + 1e-3
    )


def test_ga_trainer_is_deterministic_regardless_of_global_random_state():
    import random

    def _run() -> GATrainingResult:
        trainer = GATrainer(
            template=_make_1d_identity_template(),
            dataset=_make_dataset(),
            config=SimulationConfig(t_end=1.0, dt=0.1, solver="euler"),
            ga_config=_make_ga_config(),
            activation="identity",
            boundary="reflect",
            seed=1234,
        )
        return trainer.run()

    # DEAP operators (selTournament, cxTwoPoint) draw from the global
    # random module; run() must shield results from its outside state.
    random.seed(123)
    first = _run()
    random.seed(999)
    second = _run()
    assert first.best_fitness == second.best_fitness
    assert first.fitness_history == second.fitness_history
    assert first.mean_fitness_history == second.mean_fitness_history


def test_ga_trainer_rejects_mismatched_seed_shapes():
    bad_template = Template(
        name="bad",
        feedback=[0.0, 1.0, 0.0],
        control=[0.0, 1.0],
        bias=0.0,
    )
    with pytest.raises(CelNNError):
        GATrainer(
            template=bad_template,
            dataset=_make_dataset(),
            config=SimulationConfig(t_end=0.1, dt=0.05),
            ga_config=_make_ga_config(),
        )


def test_ga_trainer_rejects_invalid_bounds():
    template = _make_1d_identity_template()
    with pytest.raises(CelNNError):
        GATrainer(
            template=template,
            dataset=_make_dataset(),
            config=SimulationConfig(t_end=0.1, dt=0.05),
            ga_config=GAConfig(
                pop_size=4,
                n_generations=1,
                bounds=(1.0, -1.0),
            ),
        )


def test_ga_trainer_serializes_to_dict():
    trainer = GATrainer(
        template=_make_1d_identity_template(),
        dataset=_make_dataset(),
        config=SimulationConfig(t_end=0.5, dt=0.1, solver="euler"),
        ga_config=_make_ga_config(pop_size=4, n_generations=1),
        activation="identity",
        boundary="reflect",
        seed=0,
    )
    result = trainer.run()
    payload = result.to_dict()
    assert "best_template" in payload
    assert "fitness_history" in payload
    assert "config" in payload
    assert payload["n_generations"] == 1


def test_evaluate_template_returns_max_loss_on_failure():
    dataset = TrainingDataset.from_pairs(
        [np.ones(4)], [np.zeros(4)]
    )

    def bad_factory(sample):
        raise CelNNError("forced failure")

    loss = evaluate_template(
        network_factory=bad_factory,
        config=SimulationConfig(t_end=0.1, dt=0.05),
        dataset=dataset,
    )
    assert loss == pytest.approx(1e12)


def test_evaluate_template_with_regularization_changes_value():
    template = _make_1d_identity_template()
    dataset = _make_dataset()

    def factory(sample):
        from celnn import CellularNetwork

        return CellularNetwork(
            input=sample.input,
            initial_state=sample.initial_state,
            feedback=template.feedback,
            control=template.control,
            bias=template.bias,
            activation="identity",
            boundary="reflect",
        )

    no_reg = evaluate_template(
        network_factory=factory,
        config=SimulationConfig(t_end=1.0, dt=0.1, solver="euler"),
        dataset=dataset,
    )
    with_reg = evaluate_template(
        network_factory=factory,
        config=SimulationConfig(t_end=1.0, dt=0.1, solver="euler"),
        dataset=dataset,
        regularization=1.0,
    )
    assert with_reg > no_reg


def test_ga_trainer_2d_template_runs():
    template = Template(
        name="edge_seed",
        feedback=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        control=[[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
        bias=-1.0,
    )
    image = np.zeros((6, 6))
    image[2:4, 2:4] = 1.0
    target = image.copy()
    dataset = TrainingDataset.from_pairs([image], [target])
    trainer = GATrainer(
        template=template,
        dataset=dataset,
        config=SimulationConfig(t_end=1.0, dt=0.1, solver="euler"),
        ga_config=GAConfig(
            pop_size=6,
            n_generations=2,
            bounds=(-2.0, 2.0),
            bias_bounds=(-3.0, 1.0),
            seed_template=template,
        ),
        activation="piecewise_linear",
        boundary="reflect",
        seed=11,
    )
    result = trainer.run()
    assert result.best_template.feedback.shape == (3, 3)
    assert result.best_template.control.shape == (3, 3)
