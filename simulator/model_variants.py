"""Classification model variants used by the robust optical simulation study."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from d2nn import D2NN, detector_contrast
from perturbations import PerturbationDraw, apply_detector_noise


def _validate_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class HybridD2NNClassifier(D2NN):
    """D2NN optical front end followed by a small electronic MLP head.

    With `linear_head=True` the electronic head is reduced to a single linear map from the
    pooled features to class scores. That ablation separates the contribution of the
    learned readout representation from the contribution of nonlinear electronic
    processing; the optical front end is identical in both configurations.
    """

    def __init__(self, *args, pool_size=8, hidden_dim=32, linear_head=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_size = _validate_positive_int("pool_size", pool_size)
        self.hidden_dim = _validate_positive_int("hidden_dim", hidden_dim)
        self.linear_head = bool(linear_head)
        self.feature_pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        if self.linear_head:
            self.electronic_head = nn.Linear(self.detector_feature_count, self.num_classes)
        else:
            self.electronic_head = nn.Sequential(
                nn.Linear(self.detector_feature_count, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.num_classes),
            )

    @property
    def detector_feature_count(self) -> int:
        return self.pool_size**2

    def detector_features_from_intensity(self, intensity):
        return self.feature_pool(intensity.unsqueeze(1)).flatten(1)

    def forward(self, x, perturbation_draw: PerturbationDraw | None = None):
        return self.forward_with_metrics(x, perturbation_draw=perturbation_draw)["scores"]

    def forward_with_metrics(self, x, target=None, perturbation_draw: PerturbationDraw | None = None):
        intensity = self.output_intensity(x, perturbation_draw)
        detector_features = self.detector_features_from_intensity(intensity)
        detector_noise = None if perturbation_draw is None else perturbation_draw.detector_noise_relative
        detector_features = apply_detector_noise(detector_features, detector_noise)
        feature_scale = detector_features.mean(dim=1, keepdim=True).clamp_min(1e-8)
        logits = self.electronic_head(detector_features / feature_scale)
        scores = F.softmax(logits, dim=1)
        result = {
            "scores": scores,
            "logits": logits,
            "intensity": intensity,
            "detector_features": detector_features,
        }
        if target is not None:
            result["contrast"] = detector_contrast(scores, target)
        return result


class ElectronicMLPClassifier(nn.Module):
    """Small pure-electronic image classifier for a transparent simulation baseline."""

    def __init__(self, input_shape=(1, 28, 28), num_classes=10, hidden_dim=18):
        super().__init__()
        if len(input_shape) != 3 or any(int(value) < 1 for value in input_shape):
            raise ValueError("input_shape must contain three positive dimensions")
        self.input_shape = tuple(int(value) for value in input_shape)
        self.num_classes = _validate_positive_int("num_classes", num_classes)
        self.hidden_dim = _validate_positive_int("hidden_dim", hidden_dim)
        input_features = math.prod(self.input_shape)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def _validate_input(self, x):
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"expected input shape (*, {self.input_shape}), got {tuple(x.shape)}")

    @staticmethod
    def _validate_perturbation(perturbation_draw):
        if perturbation_draw is not None and not perturbation_draw.is_clean():
            raise ValueError("optical and detector perturbations are not applicable to the electronic baseline")

    def forward(self, x, perturbation_draw: PerturbationDraw | None = None):
        self._validate_input(x)
        self._validate_perturbation(perturbation_draw)
        return F.softmax(self.classifier(x), dim=1)

    def forward_with_metrics(self, x, target=None, perturbation_draw: PerturbationDraw | None = None):
        self._validate_input(x)
        self._validate_perturbation(perturbation_draw)
        logits = self.classifier(x)
        scores = F.softmax(logits, dim=1)
        result = {"scores": scores, "logits": logits}
        if target is not None:
            result["contrast"] = detector_contrast(scores, target)
        return result


class LeNet5Classifier(nn.Module):
    """LeNet-5-style convolutional reference baseline with no optical propagation.

    This exists to answer the objection that a 784-18-10 MLP is too weak an
    electronic reference on MNIST. It is a pure-electronic control, so optical and
    detector perturbations are rejected exactly as for `ElectronicMLPClassifier`.
    """

    def __init__(self, input_shape=(1, 28, 28), num_classes=10, conv_channels=(6, 16), hidden_dim=120):
        super().__init__()
        if len(input_shape) != 3 or any(int(value) < 1 for value in input_shape):
            raise ValueError("input_shape must contain three positive dimensions")
        channels, height, width = (int(value) for value in input_shape)
        if channels != 1:
            raise ValueError("LeNet5Classifier expects a single-channel input")
        if height != 28 or width != 28:
            raise ValueError("LeNet5Classifier expects a 28x28 input")
        if len(conv_channels) != 2:
            raise ValueError("conv_channels must contain exactly two stage widths")

        self.input_shape = (channels, height, width)
        self.num_classes = _validate_positive_int("num_classes", num_classes)
        self.conv_channels = tuple(_validate_positive_int("conv_channels", value) for value in conv_channels)
        self.hidden_dim = _validate_positive_int("hidden_dim", hidden_dim)

        first, second = self.conv_channels
        self.features = nn.Sequential(
            nn.Conv2d(channels, first, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(first, second, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        # 28x28 -> conv5 -> 24x24 -> pool2 -> 12x12 -> conv5 -> 8x8 -> pool2 -> 4x4
        flattened = second * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    @property
    def detector_feature_count(self) -> int:
        return self.hidden_dim

    def _validate_input(self, x):
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"expected input shape (*, {self.input_shape}), got {tuple(x.shape)}")

    @staticmethod
    def _validate_perturbation(perturbation_draw):
        if perturbation_draw is not None and not perturbation_draw.is_clean():
            raise ValueError("optical and detector perturbations are not applicable to the electronic baseline")

    def forward(self, x, perturbation_draw: PerturbationDraw | None = None):
        self._validate_input(x)
        self._validate_perturbation(perturbation_draw)
        return F.softmax(self.classifier(self.features(x)), dim=1)

    def forward_with_metrics(self, x, target=None, perturbation_draw: PerturbationDraw | None = None):
        self._validate_input(x)
        self._validate_perturbation(perturbation_draw)
        logits = self.classifier(self.features(x))
        scores = F.softmax(logits, dim=1)
        result = {"scores": scores, "logits": logits}
        if target is not None:
            result["contrast"] = detector_contrast(scores, target)
        return result
