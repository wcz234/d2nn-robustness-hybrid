"""Deterministic perturbation primitives for simulated D2NN robustness studies."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


TWO_PI = 2 * math.pi
TORCH_SEED_MODULUS = 2**63
DETECTOR_STREAM_OFFSET = 0x5DEECE66D


def _validate_non_negative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def derive_detector_seed(seed: int) -> int:
    """Derive a stable detector RNG seed distinct from the optical stream."""
    return (int(seed) + DETECTOR_STREAM_OFFSET) % TORCH_SEED_MODULUS


@dataclass(frozen=True)
class PerturbationConfig:
    """Distribution parameters for one simulated optical-device draw.

    Lateral and axial offsets use independent uniform distributions over their
    configured symmetric ranges. Phase and detector noise use zero-mean normal
    distributions with the configured standard deviations.

    `train_phase_quantization_levels` is a training-only quantization-aware knob:
    it quantizes the phase during training so the model optimizes against the
    discrete deployment mask, while validation and test keep clean continuous
    phases. It is deliberately separate from `phase_quantization_levels`, which
    is applied during training *and* validation/test as a deployment condition.

    `train_phase_filter_std_px` is the analogous training-only spatial smoothing
    knob: it low-pass filters the phase during training, which is the
    phase-filtering regularization used by misalignment-resilient diffractive
    networks. Validation and test keep clean continuous phases.
    """

    lateral_shift_max_px: float = 0.0
    axial_shift_max_m: float = 0.0
    phase_noise_std_rad: float = 0.0
    phase_quantization_levels: int | None = None
    detector_noise_std_relative: float = 0.0
    train_phase_quantization_levels: int | None = None
    train_phase_filter_std_px: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "lateral_shift_max_px",
            "axial_shift_max_m",
            "phase_noise_std_rad",
            "detector_noise_std_relative",
            "train_phase_filter_std_px",
        ):
            _validate_non_negative_finite(name, float(getattr(self, name)))
        for name in ("phase_quantization_levels", "train_phase_quantization_levels"):
            levels = getattr(self, name)
            if levels is not None and (isinstance(levels, bool) or int(levels) != levels or levels < 2):
                raise ValueError(f"{name} must be an integer >= 2 or None")
        if (
            self.train_phase_quantization_levels is not None
            and self.phase_quantization_levels is not None
            and self.train_phase_quantization_levels != self.phase_quantization_levels
        ):
            raise ValueError(
                "train_phase_quantization_levels and phase_quantization_levels must match "
                "when both are set, so that training and deployment quantization agree"
            )


@dataclass(frozen=True)
class LayerPerturbation:
    """Realized perturbations for one diffractive layer and its following gap."""

    lateral_shift_px: tuple[float, float] = (0.0, 0.0)
    axial_shift_m: float = 0.0
    phase_noise_rad: torch.Tensor | None = None


@dataclass(frozen=True)
class PerturbationDraw:
    """A fully realized perturbation draw passed through one model forward."""

    layers: tuple[LayerPerturbation, ...]
    phase_quantization_levels: int | None = None
    detector_noise_relative: torch.Tensor | None = None
    phase_filter_std_px: float = 0.0

    def layer(self, index: int) -> LayerPerturbation:
        if index < 0 or index >= len(self.layers):
            raise ValueError(f"perturbation draw has {len(self.layers)} layers; requested index {index}")
        return self.layers[index]

    def validate_num_layers(self, num_layers: int) -> None:
        if len(self.layers) != num_layers:
            raise ValueError(f"perturbation draw has {len(self.layers)} layers; model has {num_layers}")
        if self.layers and self.layers[-1].axial_shift_m != 0.0:
            raise ValueError("the final diffractive layer has no following inter-layer gap")

    def is_clean(self) -> bool:
        if self.phase_quantization_levels is not None:
            return False
        if self.phase_filter_std_px:
            return False
        if self.detector_noise_relative is not None and torch.count_nonzero(self.detector_noise_relative):
            return False
        for layer in self.layers:
            if layer.lateral_shift_px != (0.0, 0.0) or layer.axial_shift_m != 0.0:
                return False
            if layer.phase_noise_rad is not None and torch.count_nonzero(layer.phase_noise_rad):
                return False
        return True


class PerturbationSampler:
    """Stateful sampler with independently advancing optical and detector RNG streams."""

    def __init__(self, config: PerturbationConfig, seed: int, *, detector_seed: int | None = None):
        self.config = config
        self.seed = int(seed)
        self.detector_seed = derive_detector_seed(self.seed) if detector_seed is None else int(detector_seed)
        self._optical_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self._detector_generator = torch.Generator(device="cpu").manual_seed(self.detector_seed)

    def _uniform_symmetric(self, maximum: float) -> float:
        if maximum == 0:
            return 0.0
        unit_value = torch.rand((), generator=self._optical_generator, dtype=torch.float64).item()
        return (2 * unit_value - 1) * maximum

    @staticmethod
    def _normal(
        shape: tuple[int, ...],
        std: float,
        dtype: torch.dtype,
        device,
        generator: torch.Generator,
    ) -> torch.Tensor | None:
        if std == 0:
            return None
        values = torch.randn(shape, generator=generator, dtype=dtype, device="cpu") * std
        return values.to(device=device)

    def sample(
        self,
        *,
        num_layers: int,
        size: int,
        batch_size: int | None = None,
        detector_features: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        phase_quantization_levels: int | None = None,
        phase_filter_std_px: float | None = None,
    ) -> PerturbationDraw:
        if num_layers < 1 or size < 1:
            raise ValueError("num_layers and size must be positive")

        layers = []
        for layer_index in range(num_layers):
            dx = self._uniform_symmetric(self.config.lateral_shift_max_px)
            dy = self._uniform_symmetric(self.config.lateral_shift_max_px)
            axial_shift = 0.0
            if layer_index < num_layers - 1:
                axial_shift = self._uniform_symmetric(self.config.axial_shift_max_m)
            layers.append(
                LayerPerturbation(
                    lateral_shift_px=(dx, dy),
                    axial_shift_m=axial_shift,
                    phase_noise_rad=self._normal(
                        (size, size),
                        self.config.phase_noise_std_rad,
                        dtype,
                        device,
                        self._optical_generator,
                    ),
                )
            )

        detector_noise = None
        if self.config.detector_noise_std_relative > 0:
            if batch_size is None or detector_features is None:
                raise ValueError("batch_size and detector_features are required for detector noise")
            if batch_size < 1 or detector_features < 1:
                raise ValueError("batch_size and detector_features must be positive")
            detector_noise = self._normal(
                (batch_size, detector_features),
                self.config.detector_noise_std_relative,
                dtype,
                device,
                self._detector_generator,
            )

        return PerturbationDraw(
            layers=tuple(layers),
            phase_quantization_levels=(
                self.config.phase_quantization_levels
                if phase_quantization_levels is None
                else phase_quantization_levels
            ),
            detector_noise_relative=detector_noise,
            phase_filter_std_px=(
                self.config.train_phase_filter_std_px
                if phase_filter_std_px is None
                else phase_filter_std_px
            ),
        )


def quantize_phase_uniform(phase: torch.Tensor, levels: int | None, *, straight_through: bool) -> torch.Tensor:
    """Uniformly quantize wrapped phase, optionally using a straight-through gradient."""
    if levels is None:
        return phase
    if isinstance(levels, bool) or int(levels) != levels or levels < 2:
        raise ValueError("levels must be an integer >= 2 or None")

    step = TWO_PI / int(levels)
    wrapped = torch.remainder(phase, TWO_PI)
    quantized = torch.remainder(torch.round(wrapped / step), int(levels)) * step
    if straight_through:
        return phase + (quantized - phase).detach()
    return quantized


def shift_complex_modulation(
    modulation: torch.Tensor,
    lateral_shift_px: tuple[float, float],
) -> torch.Tensor:
    """Translate a sampled complex modulation without wraparound.

    Positive x moves the pattern toward increasing columns; positive y moves it
    toward increasing rows. Values outside the shifted plate are transparent
    (complex transmission 1 + 0j), not zero-amplitude padding. Subpixel shifts
    bilinearly interpolate the complex transmission, so the effective amplitude
    can be below one at phase discontinuities.
    """
    if modulation.ndim != 2 or not torch.is_complex(modulation):
        raise ValueError("modulation must be a 2D complex tensor")
    shift_x, shift_y = (float(value) for value in lateral_shift_px)
    if not math.isfinite(shift_x) or not math.isfinite(shift_y):
        raise ValueError("lateral shifts must be finite")
    if shift_x == 0.0 and shift_y == 0.0:
        return modulation

    height, width = modulation.shape
    theta = modulation.real.new_tensor(
        [[[1.0, 0.0, -2.0 * shift_x / width], [0.0, 1.0, -2.0 * shift_y / height]]]
    )
    delta = modulation - 1
    channels = torch.stack((delta.real, delta.imag), dim=0).unsqueeze(0)
    grid = F.affine_grid(theta, channels.shape, align_corners=False)
    shifted = F.grid_sample(channels, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return torch.complex(shifted[0, 0], shifted[0, 1]) + 1


def smooth_phase_gaussian(phase: torch.Tensor, std_px: float) -> torch.Tensor:
    """Low-pass filter a phase map with a separable Gaussian kernel.

    This is the phase-filtering operator used by misalignment-resilient diffractive
    networks: smoothing the phase reduces high spatial-frequency content, which is the
    content most sensitive to layer misalignment. Padding replicates edges so the filter
    introduces no artificial phase jump at the boundary. The kernel is differentiable,
    so the operator can be applied inside the training loop.
    """
    if not math.isfinite(std_px) or std_px <= 0:
        raise ValueError("phase filter std must be finite and positive")
    if phase.ndim != 2:
        raise ValueError("phase filter expects a two-dimensional phase map")

    radius = max(1, int(math.ceil(3.0 * std_px)))
    offsets = torch.arange(-radius, radius + 1, dtype=phase.dtype, device=phase.device)
    kernel = torch.exp(-(offsets**2) / (2.0 * std_px**2))
    kernel = kernel / kernel.sum()

    # Separable convolution: rows then columns.
    padded = F.pad(phase.unsqueeze(0).unsqueeze(0), (radius, radius, 0, 0), mode="replicate")
    filtered = F.conv2d(padded, kernel.view(1, 1, 1, -1))
    padded = F.pad(filtered, (0, 0, radius, radius), mode="replicate")
    filtered = F.conv2d(padded, kernel.view(1, 1, -1, 1))
    return filtered.squeeze(0).squeeze(0)


def build_phase_modulation(
    phase: torch.Tensor,
    layer_perturbation: LayerPerturbation | None,
    phase_quantization_levels: int | None,
    *,
    straight_through: bool,
    phase_filter_std_px: float = 0.0,
) -> torch.Tensor:
    """Build the effective phase modulation in the declared simulation order."""
    effective_phase = quantize_phase_uniform(
        phase,
        phase_quantization_levels,
        straight_through=straight_through,
    )
    if phase_filter_std_px:
        effective_phase = smooth_phase_gaussian(effective_phase, phase_filter_std_px)
    if layer_perturbation is None:
        return torch.exp(1j * effective_phase)

    phase_noise = layer_perturbation.phase_noise_rad
    if phase_noise is not None:
        if phase_noise.shape != phase.shape:
            raise ValueError(f"phase noise shape {phase_noise.shape} does not match phase shape {phase.shape}")
        effective_phase = effective_phase + phase_noise.to(device=phase.device, dtype=phase.dtype)
    modulation = torch.exp(1j * effective_phase)
    return shift_complex_modulation(modulation, layer_perturbation.lateral_shift_px)


def apply_detector_noise(measurements: torch.Tensor, relative_noise: torch.Tensor | None) -> torch.Tensor:
    """Apply relative additive readout noise and enforce non-negative energies."""
    if relative_noise is None:
        return measurements
    if relative_noise.shape != measurements.shape:
        raise ValueError(
            f"detector noise shape {relative_noise.shape} does not match measurements {measurements.shape}"
        )
    scale = measurements.mean(dim=1, keepdim=True)
    noisy = measurements + scale * relative_noise.to(device=measurements.device, dtype=measurements.dtype)
    return noisy.clamp_min(0)
