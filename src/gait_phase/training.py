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
from PIL import Image, ImageOps

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


class TemporalDataset(FrameDataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        workspace: str | Path,
        image_size: tuple[int, int],
        window: int,
        augment: bool = False,
    ) -> None:
        if window < 3 or window % 2 == 0:
            raise ValueError("Temporal window must be an odd integer of at least three.")
        ordered = frame.copy()
        ordered["numeric_frame_index"] = pd.to_numeric(ordered["frame_index"], errors="raise")
        ordered = ordered.sort_values(["sequence_id", "numeric_frame_index", "relative_path"]).reset_index(drop=True)
        super().__init__(ordered, workspace, image_size, augment)
        self.window = window
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
    dataset_kwargs = {"window": int(settings["temporal_window"])} if temporal else {}
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


def save_run_records(run_dir: Path, config: dict[str, Any], predictions: pd.DataFrame) -> None:
    import yaml

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(compute_metrics(predictions), stream, indent=2)
    with (run_dir / "environment.json").open("w", encoding="utf-8") as stream:
        json.dump(environment_record(), stream, indent=2)
