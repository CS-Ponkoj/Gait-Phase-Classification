"""Deterministic training and run artifact management."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

from .constants import PHASES, PHASE_TO_INDEX
from .metrics import compute_metrics


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:d}m {seconds:02d}s"


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def create_run_directory(root: str | Path, model_name: str, config: dict[str, Any]) -> Path:
    serialized = json.dumps(config, sort_keys=True).encode("utf-8")
    suffix = hashlib.sha256(serialized).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / f"{timestamp}-{model_name}-{suffix}"
    counter = 1
    while path.exists():
        path = Path(root) / f"{timestamp}-{model_name}-{suffix}-{counter}"
        counter += 1
    path.mkdir(parents=True)
    return path


def environment_record() -> dict[str, object]:
    record: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    for package in ("numpy", "pandas", "sklearn", "PIL", "torch", "torchvision"):
        try:
            module = __import__(package)
            record[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            record[package] = "not-installed"
    return record


def eligible_labeled(
    frame: pd.DataFrame,
    label_source: str = "adjudicated",
    allow_provisional: bool = False,
) -> pd.DataFrame:
    if label_source == "adjudicated":
        labels = frame["adjudicated_label"].fillna("").astype(str)
        annotation_source = "human_adjudicated"
        annotation_version = ""
        annotation_confidence = frame.get("annotation_confidence", "")
    elif label_source == "provisional":
        if not allow_provisional:
            raise ValueError("Provisional training requires the explicit --allow-provisional flag.")
        required = {
            "provisional_label",
            "provisional_source",
            "provisional_version",
            "provisional_confidence",
        }
        if missing := required - set(frame.columns):
            raise ValueError(f"Prepared provisional manifest is missing columns: {sorted(missing)}")
        labels = frame["provisional_label"].fillna("").astype(str)
        if not frame["provisional_confidence"].eq("medium").all():
            raise ValueError("Primary provisional training accepts medium-confidence labels only.")
        annotation_source = frame["provisional_source"].astype(str)
        annotation_version = frame["provisional_version"].astype(str)
        annotation_confidence = frame["provisional_confidence"].astype(str)
    else:
        raise ValueError(f"Unknown label source: {label_source}")
    exclusion = frame["exclusion_reason"].fillna("").astype(str)
    result = frame[labels.isin(PHASES) & exclusion.eq("")].copy()
    if result.empty:
        raise ValueError(f"No eligible {label_source} four-phase samples are available.")
    result["target_label"] = labels.loc[result.index]
    result["target_label_source"] = annotation_source if isinstance(annotation_source, str) else annotation_source.loc[result.index]
    result["target_annotation_version"] = (
        annotation_version if isinstance(annotation_version, str) else annotation_version.loc[result.index]
    )
    result["target_annotation_confidence"] = (
        annotation_confidence
        if isinstance(annotation_confidence, str)
        else annotation_confidence.loc[result.index]
    )
    return result


def prediction_table(source: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    if len(source) != len(predictions):
        raise ValueError("Prediction count does not match sample count.")
    columns = ["sample_id", "dataset", "subject_id", "sequence_id", "condition", "frame_index"]
    result = source[columns].copy()
    result["y_true"] = source["target_label"].astype(str).to_numpy()
    result["y_pred"] = predictions
    result["label_source"] = source["target_label_source"].astype(str).to_numpy()
    result["annotation_version"] = source["target_annotation_version"].astype(str).to_numpy()
    result["annotation_confidence"] = source["target_annotation_confidence"].astype(str).to_numpy()
    return result


class FrameDataset:
    def __init__(self, frame: pd.DataFrame, workspace: str | Path, image_size: tuple[int, int], augment: bool = False) -> None:
        self.frame = frame.reset_index(drop=True)
        self.workspace = Path(workspace)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        import torch

        row = self.frame.iloc[index]
        with Image.open(self.workspace / str(row["relative_path"])) as image:
            prepared = image.convert("RGB").resize(self.image_size)
            if self.augment and bool(torch.rand(()) < 0.5):
                prepared = ImageOps.mirror(prepared)
            array = np.asarray(prepared, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1))
        return tensor, PHASE_TO_INDEX[str(row["target_label"])], index


def prepare_aligned_silhouette(
    image: Image.Image,
    size: tuple[int, int] = (88, 128),
    margin: int = 4,
) -> Image.Image:
    """Crop, aspect-preserve, bottom-align, and center a silhouette."""
    width, height = (int(value) for value in size)
    if width < 16 or height < 16:
        raise ValueError("Aligned silhouette dimensions must both be at least 16 pixels.")
    if margin < 0 or margin * 2 >= min(width, height):
        raise ValueError("Aligned silhouette margin is invalid for the requested canvas.")
    grayscale = image.convert("L")
    binary = grayscale.point(lambda value: 255 if value > 0 else 0)
    bounding_box = binary.getbbox()
    canvas = Image.new("L", (width, height), 0)
    if bounding_box is None:
        return canvas
    cropped = binary.crop(bounding_box)
    scale = min((width - 2 * margin) / cropped.width, (height - 2 * margin) / cropped.height)
    resized = cropped.resize(
        (max(1, int(round(cropped.width * scale))), max(1, int(round(cropped.height * scale)))),
        Image.Resampling.NEAREST,
    )
    left = (width - resized.width) // 2
    top = height - margin - resized.height
    canvas.paste(resized, (left, top))
    return canvas


def _translate_without_wrap(image: Image.Image, horizontal: int, vertical: int) -> Image.Image:
    translated = Image.new(image.mode, image.size, 0)
    translated.paste(image, (horizontal, vertical))
    return translated


def _scale_on_canvas(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    resized = image.resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new(image.mode, image.size, 0)
    canvas.paste(resized, ((width - resized.width) // 2, height - resized.height))
    return canvas


def _clip_augmentation_parameters() -> dict[str, object]:
    import torch

    morphology = int(torch.randint(-1, 2, ()).item()) if bool(torch.rand(()) < 0.20) else 0
    return {
        "flip": bool(torch.rand(()) < 0.5),
        "horizontal": int(torch.randint(-4, 5, ()).item()),
        "vertical": int(torch.randint(-2, 3, ()).item()),
        "scale": float(0.94 + 0.12 * torch.rand(()).item()),
        "morphology": morphology,
    }


def _apply_clip_augmentation(image: Image.Image, parameters: dict[str, object]) -> Image.Image:
    prepared = ImageOps.mirror(image) if bool(parameters["flip"]) else image
    prepared = _scale_on_canvas(prepared, float(parameters["scale"]))
    prepared = _translate_without_wrap(
        prepared,
        int(parameters["horizontal"]),
        int(parameters["vertical"]),
    )
    morphology = int(parameters["morphology"])
    if morphology > 0:
        prepared = prepared.filter(ImageFilter.MaxFilter(3))
    elif morphology < 0:
        prepared = prepared.filter(ImageFilter.MinFilter(3))
    return prepared


class TemporalDataset(FrameDataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        workspace: str | Path,
        image_size: tuple[int, int],
        window: int,
        augment: bool = False,
        temporal_control: str = "ordered",
        seed: int = 0,
    ) -> None:
        if window < 3 or window % 2 == 0:
            raise ValueError("Temporal window must be an odd integer of at least three.")
        ordered = frame.copy()
        ordered["numeric_frame_index"] = pd.to_numeric(ordered["frame_index"], errors="raise")
        ordered = ordered.sort_values(["sequence_id", "numeric_frame_index", "relative_path"]).reset_index(drop=True)
        super().__init__(ordered, workspace, image_size, augment)
        self.window = window
        if temporal_control not in {"ordered", "repeated", "shuffled"}:
            raise ValueError(f"Unknown temporal control: {temporal_control}")
        self.temporal_control = temporal_control
        self.seed = int(seed)
        previous_sequence = self.frame["sequence_id"].shift(1)
        previous_frame = self.frame["numeric_frame_index"].shift(1)
        breaks = self.frame["sequence_id"].ne(previous_sequence) | self.frame["numeric_frame_index"].ne(previous_frame + 1)
        self.frame["temporal_segment"] = breaks.cumsum().astype(int)
        self.segment_indices = {
            int(segment): indices.to_list() for segment, indices in self.frame.groupby("temporal_segment").groups.items()
        }
        self.position = {
            global_index: position
            for indices in self.segment_indices.values()
            for position, global_index in enumerate(indices)
        }

    def __getitem__(self, index: int):
        import torch

        row = self.frame.iloc[index]
        indices = self.segment_indices[int(row["temporal_segment"])]
        position = self.position[index]
        radius = self.window // 2
        selected = [indices[min(max(position + offset, 0), len(indices) - 1)] for offset in range(-radius, radius + 1)]
        if self.temporal_control == "repeated":
            selected = [index] * self.window
        elif self.temporal_control == "shuffled":
            selected = np.random.default_rng(self.seed + index).permutation(selected).tolist()
        apply_flip = self.augment and bool(torch.rand(()) < 0.5)
        images = []
        for selected_index in selected:
            selected_row = self.frame.iloc[selected_index]
            with Image.open(self.workspace / str(selected_row["relative_path"])) as image:
                prepared = image.convert("RGB").resize(self.image_size)
                if apply_flip:
                    prepared = ImageOps.mirror(prepared)
                array = np.asarray(prepared, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(array.transpose(2, 0, 1)))
        return torch.stack(images), PHASE_TO_INDEX[str(row["target_label"])], index


class DenseTemporalDataset(FrameDataset):
    """Create overlapping, gap-safe clips with a label for every frame."""

    def __init__(
        self,
        frame: pd.DataFrame,
        workspace: str | Path,
        image_size: tuple[int, int],
        window: int,
        stride: int,
        augment: bool = False,
    ) -> None:
        if window < 3 or window % 2 == 0:
            raise ValueError("Dense temporal window must be an odd integer of at least three.")
        if stride < 1 or stride > window:
            raise ValueError("Dense temporal stride must be between one and the window length.")
        ordered = frame.copy()
        ordered["numeric_frame_index"] = pd.to_numeric(ordered["frame_index"], errors="raise")
        ordered = ordered.sort_values(["sequence_id", "numeric_frame_index", "relative_path"]).reset_index(drop=True)
        super().__init__(ordered, workspace, image_size, augment)
        self.window = int(window)
        self.stride = int(stride)
        previous_sequence = self.frame["sequence_id"].shift(1)
        previous_frame = self.frame["numeric_frame_index"].shift(1)
        breaks = self.frame["sequence_id"].ne(previous_sequence) | self.frame["numeric_frame_index"].ne(previous_frame + 1)
        self.frame["temporal_segment"] = breaks.cumsum().astype(int)
        self.segment_indices = {
            int(segment): indices.to_list() for segment, indices in self.frame.groupby("temporal_segment").groups.items()
        }
        self.clips: list[tuple[int, int]] = []
        boundary_count = 0
        for segment, indices in self.segment_indices.items():
            labels = self.frame.loc[indices, "target_label"].astype(str).to_numpy()
            boundary_count += int(np.sum(labels[1:] != labels[:-1]))
            maximum_start = max(0, len(indices) - self.window)
            starts = list(range(0, maximum_start + 1, self.stride))
            if not starts or starts[-1] != maximum_start:
                starts.append(maximum_start)
            self.clips.extend((segment, start) for start in starts)
        self.boundary_positive_count = boundary_count

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int):
        import torch

        segment, start = self.clips[index]
        segment_indices = self.segment_indices[segment]
        selected = segment_indices[start : start + self.window]
        valid_count = len(selected)
        selected = selected + [selected[-1]] * (self.window - valid_count)
        parameters = _clip_augmentation_parameters() if self.augment else None
        images = []
        targets = []
        boundaries = []
        global_indices = []
        for position, selected_index in enumerate(selected):
            row = self.frame.iloc[selected_index]
            with Image.open(self.workspace / str(row["relative_path"])) as image:
                prepared = prepare_aligned_silhouette(image, self.image_size)
            if parameters is not None:
                prepared = _apply_clip_augmentation(prepared, parameters)
            array = np.asarray(prepared, dtype=np.float32) / 255.0
            images.append(torch.from_numpy(array[None, :, :]))
            targets.append(PHASE_TO_INDEX[str(row["target_label"])])
            segment_position = start + min(position, valid_count - 1)
            is_boundary = False
            if segment_position > 0:
                previous_index = segment_indices[segment_position - 1]
                is_boundary = str(self.frame.iloc[previous_index]["target_label"]) != str(row["target_label"])
            boundaries.append(float(is_boundary))
            global_indices.append(selected_index if position < valid_count else -1)
        mask = [position < valid_count for position in range(self.window)]
        return (
            torch.stack(images),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(boundaries, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(global_indices, dtype=torch.long),
        )


def train_torch_model(
    model,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    workspace: str | Path,
    run_dir: Path,
    config: dict[str, Any],
    temporal: bool = False,
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    seed = int(config["study"]["seed"])
    settings = config["training"]
    image_size = tuple(int(value) for value in config["data"]["image_size"])
    dataset_class = TemporalDataset if temporal else FrameDataset
    dataset_kwargs = (
        {
            "window": int(settings["temporal_window"]),
            "temporal_control": str(settings.get("temporal_control", "ordered")),
            "seed": seed,
        }
        if temporal
        else {}
    )
    train_dataset = dataset_class(train_frame, workspace, image_size, augment=bool(settings.get("augmentation", True)), **dataset_kwargs)
    validation_dataset = dataset_class(validation_frame, workspace, image_size, augment=False, **dataset_kwargs)
    generator = torch.Generator().manual_seed(seed)
    workers = int(settings.get("num_workers", 0))
    requested_device = str(settings.get("device", "auto"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        required_architecture = f"sm_{capability[0]}{capability[1]}"
        supported_architectures = set(torch.cuda.get_arch_list())
        if required_architecture not in supported_architectures:
            raise RuntimeError(
                f"The installed PyTorch build does not support this GPU ({required_architecture}). "
                "Run scripts\\setup-gpu.ps1, then retry training."
            )
    resolved_device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if resolved_device == "auto":
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    loader_options = {
        "batch_size": int(settings["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    config["training_run"]["resolved_device"] = str(device)
    config["training_run"]["cuda_device_name"] = (
        torch.cuda.get_device_name(0) if device.type == "cuda" else None
    )
    progress_every = max(1, int(settings.get("progress_every_batches", 100)))
    model.to(device)
    counts = train_frame["target_label"].value_counts()
    weights = torch.tensor([len(train_frame) / (len(PHASES) * counts.get(label, 1)) for label in PHASES], dtype=torch.float32, device=device)
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    best_loss, patience = float("inf"), 0
    history: list[dict[str, float]] = []
    checkpoint = run_dir / "model.pt"
    print(
        f"device={device} train_samples={len(train_dataset)} "
        f"validation_samples={len(validation_dataset)} train_batches={len(train_loader)} "
        f"validation_batches={len(validation_loader)}",
        flush=True,
    )
    for epoch in range(int(settings["epochs"])):
        model.train()
        train_losses = []
        epoch_started = time.monotonic()
        for batch_number, (inputs, targets, _) in enumerate(train_loader, start=1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            if batch_number == 1 or batch_number % progress_every == 0 or batch_number == len(train_loader):
                elapsed = time.monotonic() - epoch_started
                batches_per_second = batch_number / max(elapsed, 1e-9)
                remaining = (len(train_loader) - batch_number) / max(batches_per_second, 1e-9)
                gpu_memory = ""
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / (1024**3)
                    reserved = torch.cuda.memory_reserved(device) / (1024**3)
                    gpu_memory = f" gpu={allocated:.2f}/{reserved:.2f}GB"
                print(
                    f"[train] epoch={epoch + 1}/{int(settings['epochs'])} "
                    f"batch={batch_number}/{len(train_loader)} "
                    f"({100 * batch_number / len(train_loader):.1f}%) "
                    f"loss={np.mean(train_losses):.6f} elapsed={_format_duration(elapsed)} "
                    f"eta={_format_duration(remaining)}{gpu_memory}",
                    flush=True,
                )
        model.eval()
        validation_losses = []
        validation_started = time.monotonic()
        with torch.no_grad():
            for batch_number, (inputs, targets, _) in enumerate(validation_loader, start=1):
                inputs, targets = inputs.to(device), targets.to(device)
                validation_losses.append(float(loss_function(model(inputs), targets).cpu()))
                if (
                    batch_number == 1
                    or batch_number % progress_every == 0
                    or batch_number == len(validation_loader)
                ):
                    elapsed = time.monotonic() - validation_started
                    batches_per_second = batch_number / max(elapsed, 1e-9)
                    remaining = (len(validation_loader) - batch_number) / max(batches_per_second, 1e-9)
                    print(
                        f"[validation] epoch={epoch + 1}/{int(settings['epochs'])} "
                        f"batch={batch_number}/{len(validation_loader)} "
                        f"({100 * batch_number / len(validation_loader):.1f}%) "
                        f"loss={np.mean(validation_losses):.6f} elapsed={_format_duration(elapsed)} "
                        f"eta={_format_duration(remaining)}",
                        flush=True,
                    )
        validation_loss = float(np.mean(validation_losses))
        train_loss = float(np.mean(train_losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        print(
            f"epoch={epoch + 1}/{int(settings['epochs'])} "
            f"train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-6:
            best_loss, patience = validation_loss, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience += 1
            if patience >= int(settings["early_stopping_patience"]):
                break
    pd.DataFrame(history).to_csv(run_dir / "learning_curves.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    predictions = np.empty(len(validation_dataset), dtype=object)
    with torch.no_grad():
        for inputs, _, indices in validation_loader:
            output = model(inputs.to(device)).argmax(1).cpu().numpy()
            predictions[indices.numpy()] = [PHASES[index] for index in output]
    return prediction_table(validation_dataset.frame, predictions)


def dense_temporal_loss(
    outputs: dict[str, object],
    targets,
    boundaries,
    mask,
    class_weights,
    boundary_positive_weight,
    boundary_weight: float,
    smoothness_weight: float,
    order_weight: float,
):
    """Return the phase, boundary, smoothing, and cyclic-order training loss."""
    import torch
    import torch.nn.functional as functional

    phase_logits = outputs["phase_logits"]
    boundary_logits = outputs["boundary_logits"]
    if phase_logits.shape[:2] != targets.shape or boundary_logits.shape != targets.shape:
        raise ValueError("Dense model outputs do not match the target sequence shape.")
    valid_logits = phase_logits[mask]
    valid_targets = targets[mask]
    phase_loss = functional.cross_entropy(valid_logits, valid_targets, weight=class_weights)
    boundary_loss = functional.binary_cross_entropy_with_logits(
        boundary_logits[mask],
        boundaries[mask],
        pos_weight=boundary_positive_weight,
    )

    pair_mask = mask[:, 1:] & mask[:, :-1]
    non_boundary_mask = pair_mask & boundaries[:, 1:].eq(0)
    log_probabilities = functional.log_softmax(phase_logits, dim=-1)
    changes = torch.clamp(
        (log_probabilities[:, 1:] - log_probabilities[:, :-1]).pow(2),
        max=4.0,
    ).mean(dim=-1)
    smoothness_loss = changes[non_boundary_mask].mean() if bool(non_boundary_mask.any()) else changes.sum() * 0.0

    bad_transitions = torch.ones(
        len(PHASES),
        len(PHASES),
        dtype=phase_logits.dtype,
        device=phase_logits.device,
    )
    for current, allowed in enumerate(({0, 1}, {1, 2}, {2, 3}, {3, 0})):
        bad_transitions[current, list(allowed)] = 0.0
    probabilities = functional.softmax(phase_logits, dim=-1)
    transition_cost = torch.einsum(
        "bti,ij,btj->bt",
        probabilities[:, :-1],
        bad_transitions,
        probabilities[:, 1:],
    )
    order_loss = transition_cost[pair_mask].mean() if bool(pair_mask.any()) else transition_cost.sum() * 0.0
    total = (
        phase_loss
        + float(boundary_weight) * boundary_loss
        + float(smoothness_weight) * smoothness_loss
        + float(order_weight) * order_loss
    )
    return {
        "total": total,
        "phase": phase_loss,
        "boundary": boundary_loss,
        "smoothness": smoothness_loss,
        "order": order_loss,
    }


def train_thermal_gait_phasenet(
    model,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    workspace: str | Path,
    run_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Train Thermal GaitPhaseNet and aggregate dense overlapping predictions."""
    import torch
    from torch.utils.data import DataLoader

    seed = int(config["study"]["seed"])
    settings = config["training"]
    model_settings = settings["thermal_gait_phasenet"]
    image_size = tuple(int(value) for value in model_settings["image_size"])
    window = int(settings["temporal_window"])
    stride = int(model_settings["clip_stride"])
    train_dataset = DenseTemporalDataset(
        train_frame,
        workspace,
        image_size,
        window,
        stride,
        augment=bool(settings.get("augmentation", True)),
    )
    validation_dataset = DenseTemporalDataset(
        validation_frame,
        workspace,
        image_size,
        window,
        stride,
        augment=False,
    )
    generator = torch.Generator().manual_seed(seed)
    workers = int(settings.get("num_workers", 0))
    requested_device = str(settings.get("device", "auto"))
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")
    resolved_device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
    if resolved_device == "auto":
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(0)
        required_architecture = f"sm_{capability[0]}{capability[1]}"
        if required_architecture not in set(torch.cuda.get_arch_list()):
            raise RuntimeError(
                f"The installed PyTorch build does not support this GPU ({required_architecture}). "
                "Run scripts\\setup-gpu.ps1, then retry training."
            )
    config["training_run"]["resolved_device"] = str(device)
    config["training_run"]["cuda_device_name"] = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    config["training_run"]["dense_sequence_output"] = True
    config["training_run"]["aligned_image_size"] = list(image_size)
    config["training_run"]["clip_stride"] = stride

    loader_options = {
        "batch_size": int(settings["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    model.to(device)

    counts = train_frame["target_label"].value_counts()
    class_weights = torch.tensor(
        [np.sqrt(len(train_frame) / (len(PHASES) * counts.get(label, 1))) for label in PHASES],
        dtype=torch.float32,
        device=device,
    )
    positives = max(1, train_dataset.boundary_positive_count)
    boundary_positive_weight = torch.tensor(
        min(20.0, max(1.0, (len(train_frame) - positives) / positives)),
        dtype=torch.float32,
        device=device,
    )
    encoder_parameters = list(model.encoder.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_parameter_ids]
    learning_rate = float(settings["learning_rate"])
    backbone_learning_rate = float(model_settings.get("backbone_learning_rate", learning_rate / 20.0))
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": backbone_learning_rate},
            {"params": head_parameters, "lr": learning_rate},
        ],
        weight_decay=float(model_settings.get("weight_decay", 1e-4)),
    )
    epochs = int(settings["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = device.type == "cuda" and bool(model_settings.get("mixed_precision", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    freeze_epochs = int(model_settings.get("freeze_encoder_epochs", 2))
    progress_every = max(1, int(settings.get("progress_every_batches", 100)))
    loss_weights = model_settings["loss_weights"]
    checkpoint = run_dir / "model.pt"
    best_loss = float("inf")
    patience = 0
    history: list[dict[str, float]] = []
    print(
        f"device={device} train_frames={len(train_frame)} validation_frames={len(validation_frame)} "
        f"train_clips={len(train_dataset)} validation_clips={len(validation_dataset)} "
        f"window={window} stride={stride} amp={use_amp}",
        flush=True,
    )

    def calculate_losses(outputs, targets, boundaries, mask):
        return dense_temporal_loss(
            outputs,
            targets,
            boundaries,
            mask,
            class_weights,
            boundary_positive_weight,
            float(loss_weights["boundary"]),
            float(loss_weights["smoothness"]),
            float(loss_weights["order"]),
        )

    for epoch in range(epochs):
        encoder_frozen = epoch < freeze_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad = not encoder_frozen
        model.train()
        if encoder_frozen:
            model.encoder.eval()
        train_losses: list[float] = []
        epoch_started = time.monotonic()
        for batch_number, (inputs, targets, boundaries, mask, _) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            boundaries = boundaries.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                losses = calculate_losses(model(inputs), targets, boundaries, mask)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(losses["total"].detach().cpu()))
            if batch_number == 1 or batch_number % progress_every == 0 or batch_number == len(train_loader):
                elapsed = time.monotonic() - epoch_started
                rate = batch_number / max(elapsed, 1e-9)
                remaining = (len(train_loader) - batch_number) / max(rate, 1e-9)
                gpu_memory = ""
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / (1024**3)
                    reserved = torch.cuda.memory_reserved(device) / (1024**3)
                    gpu_memory = f" gpu={allocated:.2f}/{reserved:.2f}GB"
                print(
                    f"[train] epoch={epoch + 1}/{epochs} batch={batch_number}/{len(train_loader)} "
                    f"({100 * batch_number / len(train_loader):.1f}%) loss={np.mean(train_losses):.6f} "
                    f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)} "
                    f"encoder={'frozen' if encoder_frozen else 'trainable'}{gpu_memory}",
                    flush=True,
                )

        model.eval()
        validation_losses: list[float] = []
        validation_started = time.monotonic()
        with torch.no_grad():
            for batch_number, (inputs, targets, boundaries, mask, _) in enumerate(validation_loader, start=1):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                boundaries = boundaries.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    losses = calculate_losses(model(inputs), targets, boundaries, mask)
                validation_losses.append(float(losses["total"].cpu()))
                if batch_number == 1 or batch_number % progress_every == 0 or batch_number == len(validation_loader):
                    elapsed = time.monotonic() - validation_started
                    rate = batch_number / max(elapsed, 1e-9)
                    remaining = (len(validation_loader) - batch_number) / max(rate, 1e-9)
                    print(
                        f"[validation] epoch={epoch + 1}/{epochs} batch={batch_number}/{len(validation_loader)} "
                        f"({100 * batch_number / len(validation_loader):.1f}%) "
                        f"loss={np.mean(validation_losses):.6f} elapsed={_format_duration(elapsed)} "
                        f"eta={_format_duration(remaining)}",
                        flush=True,
                    )
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "encoder_frozen": float(encoder_frozen),
                "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
            }
        )
        print(
            f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            patience = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience += 1
            if patience >= int(settings["early_stopping_patience"]):
                break
        scheduler.step()

    pd.DataFrame(history).to_csv(run_dir / "learning_curves.csv", index=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    probability_sums = np.zeros((len(validation_dataset.frame), len(PHASES)), dtype=np.float64)
    boundary_sums = np.zeros(len(validation_dataset.frame), dtype=np.float64)
    prediction_counts = np.zeros(len(validation_dataset.frame), dtype=np.int64)
    with torch.no_grad():
        for inputs, _, _, mask, indices in validation_loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
            probabilities = torch.softmax(outputs["phase_logits"], dim=-1).cpu().numpy()
            boundary_probabilities = torch.sigmoid(outputs["boundary_logits"]).cpu().numpy()
            indices_array = indices.numpy()
            mask_array = mask.numpy()
            for clip in range(len(indices_array)):
                valid = mask_array[clip] & (indices_array[clip] >= 0)
                selected = indices_array[clip][valid]
                np.add.at(probability_sums, selected, probabilities[clip][valid])
                np.add.at(boundary_sums, selected, boundary_probabilities[clip][valid])
                np.add.at(prediction_counts, selected, 1)
    if np.any(prediction_counts == 0):
        raise RuntimeError("Dense validation aggregation left one or more frames without a prediction.")
    averaged = probability_sums / prediction_counts[:, None]
    predictions = np.asarray([PHASES[index] for index in averaged.argmax(axis=1)], dtype=object)
    result = prediction_table(validation_dataset.frame, predictions)
    result["boundary_probability"] = boundary_sums / prediction_counts
    labels = validation_dataset.frame["target_label"].astype(str).to_numpy()
    sequence_ids = validation_dataset.frame["sequence_id"].astype(str).to_numpy()
    frame_indices = validation_dataset.frame["numeric_frame_index"].to_numpy(dtype=int)
    true_boundaries = np.zeros(len(result), dtype=int)
    true_boundaries[1:] = (
        (labels[1:] != labels[:-1])
        & (sequence_ids[1:] == sequence_ids[:-1])
        & (frame_indices[1:] == frame_indices[:-1] + 1)
    ).astype(int)
    result["y_boundary"] = true_boundaries
    return result


def save_run_records(run_dir: Path, config: dict[str, Any], predictions: pd.DataFrame) -> None:
    import yaml

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(compute_metrics(predictions), stream, indent=2)
    with (run_dir / "environment.json").open("w", encoding="utf-8") as stream:
        json.dump(environment_record(), stream, indent=2)
