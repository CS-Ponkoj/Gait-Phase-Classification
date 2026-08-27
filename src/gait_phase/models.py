"""Baseline and temporal model definitions with a locked four-class output."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .constants import PHASES


def hog_descriptor(image: Image.Image, size: tuple[int, int] = (64, 64), cells: int = 8, bins: int = 9) -> np.ndarray:
    """Compute a compact unsigned-gradient HOG descriptor without extra dependencies."""
    pixels = np.asarray(image.convert("L").resize(size), dtype=np.float32) / 255.0
    gradient_y, gradient_x = np.gradient(pixels)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = (np.arctan2(gradient_y, gradient_x) % math.pi) * bins / math.pi
    cell_height, cell_width = size[1] // cells, size[0] // cells
    features: list[float] = []
    for row in range(cells):
        for column in range(cells):
            y_slice = slice(row * cell_height, (row + 1) * cell_height)
            x_slice = slice(column * cell_width, (column + 1) * cell_width)
            histogram = np.zeros(bins, dtype=np.float32)
            cell_bins = np.floor(orientation[y_slice, x_slice]).astype(int) % bins
            cell_magnitude = magnitude[y_slice, x_slice]
            for bin_index in range(bins):
                histogram[bin_index] = cell_magnitude[cell_bins == bin_index].sum()
            histogram /= np.linalg.norm(histogram) + 1e-8
            features.extend(histogram.tolist())
    return np.asarray(features, dtype=np.float32)


def build_frame_cnn(pretrained: bool = False):
    import torch.nn as nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(PHASES))
    if model.classifier[1].out_features != 4:
        raise RuntimeError("Frame CNN output contract is not four classes.")
    return model


def build_temporal_tcn(pretrained: bool = False, hidden_channels: int = 128):
    import torch
    import torch.nn as nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    class TemporalTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            self.encoder = backbone.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            encoder_channels = backbone.classifier[1].in_features
            self.temporal = nn.Sequential(
                nn.Conv1d(encoder_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(hidden_channels, len(PHASES))

        def forward(self, inputs):
            if inputs.ndim != 5:
                raise ValueError("Temporal model expects [batch, time, channels, height, width].")
            batch, time, channels, height, width = inputs.shape
            features = self.encoder(inputs.reshape(batch * time, channels, height, width))
            features = self.pool(features).flatten(1).reshape(batch, time, -1).transpose(1, 2)
            temporal = self.temporal(features)
            center = temporal[:, :, time // 2]
            return self.classifier(center)

    model = TemporalTCN()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 3, 64, 64))
    if output.shape != (1, 4):
        raise RuntimeError("Temporal TCN output contract is not four classes.")
    return model
