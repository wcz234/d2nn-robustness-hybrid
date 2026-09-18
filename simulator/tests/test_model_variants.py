import pytest
import torch
import torch.nn.functional as F

from d2nn import D2NN, D2NNImager, RayleighSommerfeldPropagation
from model_variants import ElectronicMLPClassifier, HybridD2NNClassifier
from perturbations import LayerPerturbation, PerturbationConfig, PerturbationDraw, PerturbationSampler


def optical_kwargs(*, size=12, num_layers=2):
    return {
        "num_layers": num_layers,
        "size": size,
        "num_classes": 10,
        "wavelength": 852e-9,
        "layer_distance": 30e-3,
        "pixel_size": 1e-6,
        "input_distance": 491.302e-3,
        "output_distance": 575.304e-3,
        "propagation_backend": "fft",
    }


def test_clean_draw_is_exactly_compatible_with_unperturbed_d2nn_forward():
    torch.manual_seed(3)
    model = D2NN(**optical_kwargs()).eval()
    inputs = torch.rand(2, 1, 28, 28)
    targets = torch.tensor([1, 4])
    clean_draw = PerturbationSampler(PerturbationConfig(), seed=9).sample(num_layers=2, size=12)

    baseline = model.forward_with_metrics(inputs, target=targets)
    clean = model.forward_with_metrics(inputs, target=targets, perturbation_draw=clean_draw)

    for key in ("scores", "logits", "intensity", "contrast"):
        assert torch.equal(baseline[key], clean[key])
    assert not any("perturb" in key for key in model.state_dict())


@pytest.mark.parametrize("backend", ["direct", "fft"])
def test_dynamic_axial_distance_matches_fresh_propagator(backend):
    kwargs = {"size": 6, "wavelength": 0.75e-3, "distance": 0.8e-3, "pixel_size": 0.4e-3}
    base = RayleighSommerfeldPropagation(**kwargs, chunk_size=12, backend=backend)
    expected = RayleighSommerfeldPropagation(
        **{**kwargs, "distance": 0.9e-3},
        chunk_size=12,
        backend=backend,
    )
    field = torch.randn(2, 6, 6, dtype=torch.cfloat)

    dynamic_output = base(field, distance_override=0.9e-3)
    expected_output = expected(field)

    assert torch.allclose(dynamic_output, expected_output, atol=1e-5, rtol=1e-5)


def test_dynamic_axial_distance_rejects_non_positive_values():
    propagation = RayleighSommerfeldPropagation(
        size=4,
        wavelength=0.75e-3,
        distance=0.8e-3,
        pixel_size=0.4e-3,
        backend="fft",
    )
    with pytest.raises(ValueError, match="positive"):
        propagation(torch.ones(1, 4, 4, dtype=torch.cfloat), distance_override=0)


def test_perturbed_d2nn_has_finite_phase_gradients_and_expected_shapes():
    torch.manual_seed(5)
    model = D2NN(**optical_kwargs()).train()
    inputs = torch.rand(2, 1, 28, 28)
    targets = torch.tensor([0, 1])
    config = PerturbationConfig(
        lateral_shift_max_px=0.4,
        axial_shift_max_m=1e-4,
        phase_noise_std_rad=0.05,
        phase_quantization_levels=8,
        detector_noise_std_relative=0.01,
    )
    draw = PerturbationSampler(config, seed=77).sample(
        num_layers=2,
        size=12,
        batch_size=2,
        detector_features=model.detector_feature_count,
    )

    result = model.forward_with_metrics(inputs, target=targets, perturbation_draw=draw)
    F.cross_entropy(result["logits"], targets).backward()

    assert result["scores"].shape == (2, 10)
    assert result["intensity"].shape == (2, 12, 12)
    for layer in model.layers:
        assert layer.phase.grad is not None
        assert torch.isfinite(layer.phase.grad).all()


def test_hybrid_classifier_returns_detector_features_and_backpropagates_through_both_stages():
    torch.manual_seed(11)
    model = HybridD2NNClassifier(**optical_kwargs(size=16), pool_size=4, hidden_dim=8).train()
    inputs = torch.rand(2, 1, 28, 28)
    targets = torch.tensor([2, 3])
    draw = PerturbationSampler(
        PerturbationConfig(lateral_shift_max_px=0.25, detector_noise_std_relative=0.02),
        seed=12,
    ).sample(num_layers=2, size=16, batch_size=2, detector_features=model.detector_feature_count)

    result = model.forward_with_metrics(inputs, target=targets, perturbation_draw=draw)
    F.cross_entropy(result["logits"], targets).backward()

    assert result["scores"].shape == (2, 10)
    assert result["logits"].shape == (2, 10)
    assert result["intensity"].shape == (2, 16, 16)
    assert result["detector_features"].shape == (2, 16)
    assert torch.allclose(result["scores"].sum(dim=1), torch.ones(2), atol=1e-6)
    assert all(layer.phase.grad is not None and torch.isfinite(layer.phase.grad).all() for layer in model.layers)
    assert all(parameter.grad is not None for parameter in model.electronic_head.parameters())


def test_electronic_baseline_returns_shared_output_contract_and_gradients():
    torch.manual_seed(13)
    model = ElectronicMLPClassifier(input_shape=(1, 28, 28), hidden_dim=7)
    inputs = torch.rand(3, 1, 28, 28)
    targets = torch.tensor([1, 2, 3])

    result = model.forward_with_metrics(inputs, target=targets)
    F.cross_entropy(result["logits"], targets).backward()

    assert result["scores"].shape == (3, 10)
    assert result["logits"].shape == (3, 10)
    assert result["contrast"].shape == (3,)
    assert torch.allclose(result["scores"].sum(dim=1), torch.ones(3), atol=1e-6)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_electronic_baseline_accepts_clean_draw_but_rejects_optical_perturbations():
    model = ElectronicMLPClassifier()
    inputs = torch.rand(1, 1, 28, 28)
    clean_draw = PerturbationDraw(layers=(LayerPerturbation(),))
    shifted_draw = PerturbationDraw(layers=(LayerPerturbation(lateral_shift_px=(1.0, 0.0)),))

    assert model(inputs, perturbation_draw=clean_draw).shape == (1, 10)
    with pytest.raises(ValueError, match="not applicable"):
        model(inputs, perturbation_draw=shifted_draw)


def test_imager_rejects_classification_detector_noise_in_all_propagation_paths():
    model = D2NNImager(
        num_layers=1,
        size=8,
        wavelength=0.75e-3,
        layer_distance=30e-3,
        pixel_size=0.4e-3,
        input_distance=30e-3,
        output_distance=30e-3,
        propagation_backend="fft",
    )
    inputs = torch.rand(1, 1, 8, 8)
    draw = PerturbationDraw(
        layers=(LayerPerturbation(),),
        detector_noise_relative=torch.tensor([[0.1]]),
    )

    embedded = torch.ones(1, 8, 8, dtype=torch.cfloat)
    calls = (
        lambda: model(inputs, perturbation_draw=draw),
        lambda: model.propagate(inputs, perturbation_draw=draw),
        lambda: model.propagate_field(inputs, perturbation_draw=draw),
        lambda: model.propagate_embedded_field(embedded, perturbation_draw=draw),
    )

    for call in calls:
        with pytest.raises(ValueError, match="detector noise"):
            call()


def test_model_variants_validate_shapes_and_head_dimensions():
    with pytest.raises(ValueError, match="input_shape"):
        ElectronicMLPClassifier(input_shape=(28, 28))
    with pytest.raises(ValueError, match="hidden_dim"):
        ElectronicMLPClassifier(hidden_dim=0)
    with pytest.raises(ValueError, match="pool_size"):
        HybridD2NNClassifier(**optical_kwargs(), pool_size=0)
