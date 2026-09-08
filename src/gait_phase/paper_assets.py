"""Generate traceable conference-paper tables and figures from predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import PHASES
from .metrics import compute_metrics, subject_macro_f1_interval


def make_paper_assets(predictions: pd.DataFrame, output_dir: str | Path, bootstrap_iterations: int = 2000) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(predictions)
    interval = subject_macro_f1_interval(predictions, iterations=bootstrap_iterations)
    metrics["subject_macro_f1_interval"] = interval
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    per_class = pd.DataFrame(metrics["per_class"]).T.rename_axis("phase").reset_index()
    per_class.to_csv(output / "per_class_metrics.csv", index=False)
    grouped_records = []
    for keys, group in predictions.groupby(["dataset", "condition"], dropna=False):
        group_metrics = compute_metrics(group)
        grouped_records.append(
            {
                "dataset": keys[0],
                "condition": keys[1],
                "samples": group_metrics["samples"],
                "subjects": group_metrics["subjects"],
                "subject_macro_f1": group_metrics["subject_macro_f1"],
                "balanced_accuracy": group_metrics["balanced_accuracy"],
            }
        )
    pd.DataFrame(grouped_records).to_csv(output / "dataset_condition_metrics.csv", index=False)
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(4), [label.replace("_", "\n") for label in PHASES], rotation=20, ha="right")
    axis.set_yticks(range(4), [label.replace("_", "\n") for label in PHASES])
    axis.set_xlabel("Predicted phase")
    axis.set_ylabel("Reference phase")
    axis.set_title("Four-phase confusion matrix")
    for row in range(4):
        for column in range(4):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=200)
    plt.close(figure)
    summary = (
        "# Generated Results Summary\n\n"
        f"- Samples: {metrics['samples']}\n"
        f"- Subjects: {metrics['subjects']}\n"
        f"- Subject-macro F1: {metrics['subject_macro_f1']:.4f} "
        f"(95% clustered bootstrap CI {interval['lower']:.4f}-{interval['upper']:.4f})\n"
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}\n\n"
        "This file was generated from saved predictions. Do not edit reported values manually.\n"
    )
    (output / "results_summary.md").write_text(summary, encoding="utf-8")
    return metrics
