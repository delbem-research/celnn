import celnn

EXPECTED_PUBLIC_API = [
    "CellularNetwork",
    "AssociativeFieldState",
    "AssociativeMemoryState",
    "DeltaHebbianMemory",
    "DeltaHebbianRule",
    "DifferentiableCellularNetwork",
    "HebbianRule",
    "NormalizedDeltaHebbianField",
    "OjaRule",
    "PlasticLinear",
    "Plasticity",
    "PlasticityRule",
    "PlasticityState",
    "SimulationConfig",
    "SimulationResult",
    "Template",
    "TemplateRegistry",
    "identity",
    "piecewise_linear",
    "relu_activation",
    "saturated_linear",
    "sign_activation",
    "sigmoid_activation",
    "tanh_activation",
]


def test_top_level_public_api_is_deliberate():
    assert celnn.__all__ == EXPECTED_PUBLIC_API


def test_runtime_version_comes_from_installed_metadata():
    assert celnn.__version__
