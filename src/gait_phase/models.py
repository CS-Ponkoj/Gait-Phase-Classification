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


def build_thermal_gait_phasenet(
    pretrained: bool = False,
    feature_dim: int = 256,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
    attention_heads: int = 4,
    max_window: int = 99,
):
    """Build the dense, boundary-aware thermal silhouette model.

    The model combines whole-body and lower-body silhouette features, a small
    adjacent-frame motion branch, residual dilated temporal convolutions, and
    one self-attention layer. It emits one four-class prediction and one phase
    boundary logit for every input frame.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    if feature_dim < 32 or feature_dim % 8:
        raise ValueError("Thermal GaitPhaseNet feature_dim must be divisible by 8 and at least 32.")
    if attention_heads < 1 or feature_dim % attention_heads:
        raise ValueError("attention_heads must divide feature_dim.")
    if not dilations or any(int(value) < 1 for value in dilations):
        raise ValueError("At least one positive temporal dilation is required.")

    class ResidualDilatedTemporalBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            self.depthwise = nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            )
            self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
            self.normalization = nn.GroupNorm(8, channels)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(0.2)

        def forward(self, inputs):
            transformed = self.depthwise(inputs)
            transformed = self.pointwise(transformed)
            transformed = self.normalization(transformed)
            transformed = self.activation(transformed)
            return inputs + self.dropout(transformed)

    class ThermalGaitPhaseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            first = backbone.features[0][0]
            single_channel = nn.Conv2d(
                1,
                first.out_channels,
                kernel_size=first.kernel_size,
                stride=first.stride,
                padding=first.padding,
                dilation=first.dilation,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    single_channel.weight.copy_(first.weight.mean(dim=1, keepdim=True))
            backbone.features[0][0] = single_channel
            self.encoder = backbone.features
            encoder_channels = backbone.classifier[1].in_features
            self.motion_encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
                nn.GroupNorm(4, 16),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.spatial_projection = nn.Sequential(
                nn.Linear(encoder_channels * 2 + 64, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Dropout(0.2),
            )
            self.temporal_blocks = nn.Sequential(
                *(ResidualDilatedTemporalBlock(feature_dim, int(dilation)) for dilation in dilations)
            )
            self.position = nn.Parameter(torch.zeros(1, max_window, feature_dim))
            nn.init.trunc_normal_(self.position, std=0.02)
            self.attention = nn.MultiheadAttention(
                feature_dim,
                attention_heads,
                dropout=0.1,
                batch_first=True,
            )
            self.attention_normalization = nn.LayerNorm(feature_dim)
            self.phase_head = nn.Linear(feature_dim, len(PHASES))
            self.boundary_head = nn.Linear(feature_dim, 1)

        def forward(self, inputs):
            if inputs.ndim != 5 or inputs.shape[2] != 1:
                raise ValueError("Thermal GaitPhaseNet expects [batch, time, 1, height, width].")
            batch, time, channels, height, width = inputs.shape
            if time > self.position.shape[1]:
                raise ValueError(f"Input window {time} exceeds configured maximum {self.position.shape[1]}.")
            flattened = inputs.reshape(batch * time, channels, height, width)
            feature_maps = self.encoder(flattened)
            global_features = functional.adaptive_avg_pool2d(feature_maps, 1).flatten(1)
            lower_start = max(0, feature_maps.shape[-2] // 2)
            lower_features = functional.adaptive_avg_pool2d(feature_maps[:, :, lower_start:, :], 1).flatten(1)

            differences = torch.zeros_like(inputs)
            differences[:, 1:] = torch.abs(inputs[:, 1:] - inputs[:, :-1])
            motion_features = self.motion_encoder(
                differences.reshape(batch * time, channels, height, width)
            ).flatten(1)
            combined = torch.cat((global_features, lower_features, motion_features), dim=1)
            features = self.spatial_projection(combined).reshape(batch, time, -1)

            temporal = self.temporal_blocks(features.transpose(1, 2)).transpose(1, 2)
            temporal = temporal + self.position[:, :time]
            attended, _ = self.attention(temporal, temporal, temporal, need_weights=False)
            temporal = self.attention_normalization(temporal + attended)
            phase_logits = self.phase_head(temporal)
            boundary_logits = self.boundary_head(temporal).squeeze(-1)
            return {"phase_logits": phase_logits, "boundary_logits": boundary_logits}

    model = ThermalGaitPhaseNet()
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 9, 1, 64, 44))
    if output["phase_logits"].shape != (1, 9, 4) or output["boundary_logits"].shape != (1, 9):
        raise RuntimeError("Thermal GaitPhaseNet output contract is invalid.")
    model.train()
    return model
