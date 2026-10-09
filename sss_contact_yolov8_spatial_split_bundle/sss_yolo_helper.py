"""Small helpers used by the single-channel SSS YOLOv8 Colab notebook.

The bundle contains derived 150x150 grayscale patches and one-class YOLO
labels.  This module deliberately keeps data preparation explicit: group 02
and group 03 are training, group 04 is validation, and group 01 is the blind
test set.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw


GROUPS = ("group_01", "group_02", "group_03", "group_04")


def safe_extract_zip(zip_path: str | Path, destination: str | Path) -> Path:
    """Extract a user-uploaded ZIP while rejecting path traversal entries."""

    zip_path = Path(zip_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with ZipFile(zip_path) as archive:
        for info in archive.infolist():
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP member: {info.filename!r}")
            target = (root / member).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"ZIP member escapes extraction root: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    return destination


def find_bundle_root(extraction_root: str | Path) -> Path:
    """Find the directory containing data/images and data/labels."""

    root = Path(extraction_root)
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir())
    for candidate in candidates:
        if (candidate / "data" / "images").is_dir() and (candidate / "data" / "labels").is_dir():
            return candidate
    raise FileNotFoundError("Could not find a bundle root containing data/images and data/labels")


def read_yolo_labels(path: str | Path) -> np.ndarray:
    """Read one-class YOLO rows as float32 [class, cx, cy, w, h]."""

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO values, got {len(values)}")
        row = [float(value) for value in values]
        if not np.all(np.isfinite(row)):
            raise ValueError(f"{path}:{line_number}: non-finite YOLO value")
        if int(row[0]) != 0 or not np.all((0.0 <= np.asarray(row[1:])) & (np.asarray(row[1:]) <= 1.0)):
            raise ValueError(f"{path}:{line_number}: expected class 0 and normalized coordinates")
        rows.append(row)
    return np.asarray(rows, dtype=np.float32).reshape((-1, 5))


def group_statistics(bundle_root: str | Path, groups: Iterable[str] = GROUPS) -> list[dict[str, Any]]:
    """Count images, positive patches, and boxes from the actual bundle."""

    bundle_root = Path(bundle_root)
    rows: list[dict[str, Any]] = []
    for group in groups:
        image_dir = bundle_root / "data" / "images" / group
        label_dir = bundle_root / "data" / "labels" / group
        images = sorted(image_dir.glob("*.png"))
        positive_patches = 0
        boxes = 0
        missing_labels: list[str] = []
        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(image_path.name)
                continue
            count = len(read_yolo_labels(label_path))
            boxes += count
            positive_patches += int(count > 0)
        rows.append(
            {
                "group": group,
                "images": len(images),
                "positive_patches": positive_patches,
                "boxes": boxes,
                "negative_patches": len(images) - positive_patches,
                "missing_labels": len(missing_labels),
            }
        )
        if missing_labels:
            raise FileNotFoundError(f"{group}: missing labels for {missing_labels[:3]}")
    return rows


def write_dataset_yaml(
    bundle_root: str | Path,
    output_path: str | Path,
    val_group: str = "group_04",
    test_group: str = "group_01",
) -> Path:
    """Write the fixed group-level train/val/test split for Ultralytics."""

    bundle_root = Path(bundle_root).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_groups = ("group_02", "group_03")
    requested_groups = (*train_groups, val_group, test_group)
    if val_group == test_group or any(group not in GROUPS for group in requested_groups):
        raise ValueError(f"Invalid or overlapping split: train={train_groups}, val={val_group}, test={test_group}")
    # JSON is valid YAML and avoids an extra runtime dependency in this helper.
    payload = {
        "path": str(bundle_root),
        "train": [f"data/images/{group}" for group in train_groups],
        "val": f"data/images/{val_group}",
        "test": f"data/images/{test_group}",
        "nc": 1,
        "names": ["contact"],
        "channels": 1,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def select_examples(
    bundle_root: str | Path,
    group: str = "group_04",
    positive_count: int = 6,
    negative_count: int = 6,
    seed: int = 42,
) -> list[Path]:
    """Choose deterministic positive and negative images for qualitative review."""

    bundle_root = Path(bundle_root)
    image_dir = bundle_root / "data" / "images" / group
    label_dir = bundle_root / "data" / "labels" / group
    positives: list[Path] = []
    negatives: list[Path] = []
    for image_path in sorted(image_dir.glob("*.png")):
        labels = read_yolo_labels(label_dir / f"{image_path.stem}.txt")
        (positives if len(labels) else negatives).append(image_path)
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    return positives[:positive_count] + negatives[:negative_count]


def plot_training_history(results_csv: str | Path, output_path: str | Path) -> Path:
    """Create an explicit loss/metric figure from Ultralytics results.csv."""

    import matplotlib.pyplot as plt
    import pandas as pd

    results_csv = Path(results_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(results_csv)
    frame.columns = [str(column).strip() for column in frame.columns]

    def first_matching_column(prefix: str) -> str | None:
        """Pick one column when an old/reused CSV contains duplicate variants."""

        candidates = [
            column
            for column in frame.columns
            if column == prefix or column.startswith(f"{prefix}.")
        ]
        return candidates[0] if candidates else None

    epoch_column = first_matching_column("epoch")
    raw_rows = len(frame)
    if epoch_column is not None:
        frame[epoch_column] = pd.to_numeric(frame[epoch_column], errors="coerce")
        frame = frame.dropna(subset=[epoch_column]).copy()
        epoch_values = frame[epoch_column].to_numpy()
        reset_positions = np.flatnonzero(np.diff(epoch_values) <= 0) + 1
        latest_start = int(reset_positions[-1]) if len(reset_positions) else 0
        frame = frame.iloc[latest_start:].copy()
        frame = frame.drop_duplicates(subset=[epoch_column], keep="last").sort_values(epoch_column)
    print(f"Using latest training segment: {len(frame)} rows from {raw_rows} CSV rows")

    panels = [
        (("train/box_loss", "val/box_loss"), "Box loss"),
        (("train/cls_loss", "val/cls_loss"), "Class loss"),
        (("train/dfl_loss", "val/dfl_loss"), "DFL loss"),
        (("metrics/mAP50(B)", "metrics/mAP50-95(B)"), "Validation mAP"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    epochs = (
        frame[epoch_column]
        if epoch_column is not None
        else pd.Series(np.arange(1, len(frame) + 1), index=frame.index)
    )
    for axis, (columns, title) in zip(axes, panels):
        plotted = False
        for column in columns:
            selected = first_matching_column(column)
            if selected is None:
                continue
            values = pd.to_numeric(frame[selected], errors="coerce")
            valid = epochs.notna() & values.notna()
            axis.plot(epochs[valid], values[valid], label=column, linewidth=2)
            plotted = True
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "column not present", ha="center", va="center")
    fig.suptitle("Single-channel SSS YOLOv8 training history")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def metrics_to_dict(metrics: Any) -> dict[str, float]:
    """Extract common scalar metrics without depending on one Ultralytics release."""

    output: dict[str, float] = {}
    for key, value in getattr(metrics, "results_dict", {}).items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    box = getattr(metrics, "box", None)
    for name in ("mp", "mr", "map50", "map75", "map"):
        if box is not None and hasattr(box, name):
            try:
                output[name] = float(getattr(box, name))
            except (TypeError, ValueError):
                pass
    return output


def save_prediction_previews(
    model: Any,
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    imgsz: int = 256,
    conf: float = 0.25,
    iou: float = 0.50,
    device: str | int = 0,
) -> list[Path]:
    """Run grayscale path inference and save annotated prediction images."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prediction from file paths otherwise defaults to RGB in some releases.
    # The trained DetectionModel is explicitly marked as one-channel here.
    if hasattr(model, "model"):
        model.model.ch = 1
        if hasattr(model.model, "yaml"):
            model.model.yaml["channels"] = 1
    results = model.predict(
        source=[str(Path(path)) for path in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=False,
        verbose=False,
    )
    saved: list[Path] = []
    for image_path, result in zip(image_paths, results):
        output_path = output_dir / f"{Path(image_path).stem}_prediction.png"
        result.save(filename=str(output_path))
        # Add the manually supplied ground-truth boxes in yellow so the
        # qualitative panel shows predictions and labels together.  The
        # prediction image is saved at the source image dimensions.
        label_path = Path(image_path).parents[2] / "labels" / Path(image_path).parent.name / f"{Path(image_path).stem}.txt"
        with Image.open(output_path).convert("RGB") as preview:
            draw = ImageDraw.Draw(preview)
            width, height = preview.size
            for _, cx, cy, box_width, box_height in read_yolo_labels(label_path):
                x1 = (float(cx) - float(box_width) / 2.0) * width
                y1 = (float(cy) - float(box_height) / 2.0) * height
                x2 = (float(cx) + float(box_width) / 2.0) * width
                y2 = (float(cy) + float(box_height) / 2.0) * height
                draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=max(1, width // 75))
            preview.save(output_path)
        saved.append(output_path)
    return saved


def image_mode_and_size(path: str | Path) -> dict[str, Any]:
    """Return a tiny audit record for one source image."""

    with Image.open(path) as image:
        return {"mode": image.mode, "size": list(image.size)}
