import math

import pytest
import torch

from perturbations import (
    LayerPerturbation,
    PerturbationConfig,
    PerturbationDraw,
    PerturbationSampler,
    apply_detector_noise,
    build_phase_modulation,
    quantize_phase_uniform,
    shift_complex_modulation,
)


def test_perturbation_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="lateral_shift_max_px"):
        PerturbationConfig(lateral_shift_max_px=-1)
    with pytest.raises(ValueError, match="phase_quantization_levels"):
        PerturbationConfig(phase_quantization_levels=1)
    with pytest.raises(ValueError, match="detector_noise_std_relative"):
        PerturbationConfig(detector_noise_std_relative=float("nan"))


def test_sampler_is_deterministic_and_respects_declared_bounds():
    config = PerturbationConfig(
        lateral_shift_max_px=1.5,
        axial_shift_max_m=2e-4,
        phase_noise_std_rad=0.2,
        phase_quantization_levels=8,
        detector_noise_std_relative=0.05,
    )
    kwargs = {"num_layers": 3, "size": 6, "batch_size": 4, "detector_features": 10}
    first = PerturbationSampler(config, seed=123).sample(**kwargs)
    repeated = PerturbationSampler(config, seed=123).sample(**kwargs)

    assert first.phase_quantization_levels == 8
    assert torch.equal(first.detector_noise_relative, repeated.detector_noise_relative)
    for index, (left, right) in enumerate(zip(first.layers, repeated.layers)):
        assert left.lateral_shift_px == right.lateral_shift_px
        assert left.axial_shift_m == right.axial_shift_m
        assert torch.equal(left.phase_noise_rad, right.phase_noise_rad)
        assert all(abs(value) <= config.lateral_shift_max_px for value in left.lateral_shift_px)
        if index < len(first.layers) - 1:
            assert abs(left.axial_shift_m) <= config.axial_shift_max_m
        else:
            assert left.axial_shift_m == 0.0

    different = PerturbationSampler(config, seed=124).sample(**kwargs)
    assert first.layers[0].lateral_shift_px != different.layers[0].lateral_shift_px


def test_sampler_requires_detector_dimensions_when_noise_is_enabled():
    sampler = PerturbationSampler(PerturbationConfig(detector_noise_std_relative=0.1), seed=1)
    with pytest.raises(ValueError, match="batch_size and detector_features"):
        sampler.sample(num_layers=2, size=4)


def test_detector_shape_does_not_change_later_optical_draws_for_direct_sampler_use():
    config = PerturbationConfig(
        lateral_shift_max_px=0.5,
        axial_shift_max_m=1e-4,
        phase_noise_std_rad=0.1,
        detector_noise_std_relative=0.2,
    )
    narrow = PerturbationSampler(config, seed=19)
    wide = PerturbationSampler(config, seed=19)

    narrow_draws = [
        narrow.sample(num_layers=2, size=5, batch_size=2, detector_features=10)
        for _ in range(2)
    ]
    wide_draws = [
        wide.sample(num_layers=2, size=5, batch_size=2, detector_features=16)
        for _ in range(2)
    ]

    for narrow_draw, wide_draw in zip(narrow_draws, wide_draws):
        for narrow_layer, wide_layer in zip(narrow_draw.layers, wide_draw.layers):
            assert narrow_layer.lateral_shift_px == wide_layer.lateral_shift_px
            assert narrow_layer.axial_shift_m == wide_layer.axial_shift_m
            assert torch.equal(narrow_layer.phase_noise_rad, wide_layer.phase_noise_rad)


def test_zero_config_produces_clean_draw():
    draw = PerturbationSampler(PerturbationConfig(), seed=7).sample(num_layers=2, size=4)
    assert draw.is_clean()


def test_uniform_phase_quantization_has_exact_levels_and_straight_through_gradient():
    phase = torch.linspace(-0.3, 2 * math.pi + 0.3, 17, requires_grad=True)
    quantized = quantize_phase_uniform(phase, 4, straight_through=True)
    step = 2 * math.pi / 4

    assert torch.all(quantized >= 0)
    assert torch.all(quantized < 2 * math.pi)
    assert torch.allclose(torch.remainder(quantized / step, 1), torch.zeros_like(quantized), atol=1e-6)

    quantized.sum().backward()
    assert torch.equal(phase.grad, torch.ones_like(phase))


def test_complex_shift_moves_right_without_wraparound_and_uses_transparent_padding():
    modulation = torch.ones(5, 5, dtype=torch.cfloat)
    modulation[2, 2] = -1 + 0j

    shifted = shift_complex_modulation(modulation, (1.0, 0.0))

    assert torch.allclose(shifted[2, 3], torch.tensor(-1 + 0j))
    assert torch.allclose(shifted[2, 2], torch.tensor(1 + 0j))
    assert torch.allclose(shifted[:, 0], torch.ones(5, dtype=torch.cfloat))


def test_subpixel_complex_shift_preserves_finite_phase_gradients():
    phase = torch.randn(6, 6, requires_grad=True)
    shifted = shift_complex_modulation(torch.exp(1j * phase), (0.35, -0.4))

    assert torch.all(shifted.abs() <= 1 + 1e-6)
    shifted.real.square().mean().backward()

    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()


@pytest.mark.parametrize("levels", [2, 4])
def test_quantized_half_pixel_shift_has_bounded_phase_gradients(levels):
    step = 2 * math.pi / levels
    columns = torch.arange(8).remainder(levels) * step
    phase = columns.repeat(8, 1).requires_grad_()
    quantized = quantize_phase_uniform(phase, levels, straight_through=True)

    shifted = shift_complex_modulation(torch.exp(1j * quantized), (0.5, 0.0))
    shifted.real.sum().backward()

    assert torch.isfinite(shifted.real).all()
    assert torch.all(shifted.abs() <= 1 + 1e-6)
    assert phase.grad is not None
    assert torch.isfinite(phase.grad).all()
    assert phase.grad.abs().max() < 10
    if levels == 2:
        assert shifted.abs().min() < 1e-5


def test_phase_modulation_validates_noise_shape():
    phase = torch.zeros(4, 4)
    perturbation = LayerPerturbation(phase_noise_rad=torch.zeros(3, 3))
    with pytest.raises(ValueError, match="phase noise shape"):
        build_phase_modulation(phase, perturbation, None, straight_through=False)


def test_detector_noise_is_relative_and_non_negative():
    measurements = torch.tensor([[1.0, 3.0]])
    relative_noise = torch.tensor([[-10.0, 1.0]])

    noisy = apply_detector_noise(measurements, relative_noise)

    assert torch.equal(noisy, torch.tensor([[0.0, 5.0]]))


def test_detector_noise_scale_remains_in_the_autograd_graph():
    measurements = torch.tensor([[2.0, 4.0]], requires_grad=True)
    relative_noise = torch.tensor([[0.2, -0.4]])

    apply_detector_noise(measurements, relative_noise).sum().backward()

    assert torch.allclose(measurements.grad, torch.full_like(measurements, 0.9))


def test_draw_rejects_wrong_layer_count():
    draw = PerturbationDraw(layers=(LayerPerturbation(),))
    with pytest.raises(ValueError, match="model has 2"):
        draw.validate_num_layers(2)


def test_draw_rejects_axial_shift_after_final_layer():
    draw = PerturbationDraw(layers=(LayerPerturbation(axial_shift_m=1e-4),))
    with pytest.raises(ValueError, match="no following inter-layer gap"):
        draw.validate_num_layers(1)
