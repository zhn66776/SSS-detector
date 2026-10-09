"""Small, dependency-light helpers for the SSS contact smoke test notebook.

The helper deliberately keeps the source PNGs read-only. It loads native
150x150 single-band patches, converts them in memory to the three-channel
tensor expected by the selected ONNX checkpoint, maps predictions back to the
native patch coordinates, and evaluates only the single exported contact-like
class.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


def read_yolo_labels(path: str | Path) -> np.ndarray:
    """Read class/x/y/w/h labels; return an empty (0, 5) array when empty."""

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO values, got {len(values)}")
        rows.append([float(value) for value in values])
    return np.asarray(rows, dtype=np.float32).reshape((-1, 5))


def yolo_to_xyxy(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized YOLO labels to [x1, y1, x2, y2] native pixels."""

    if labels.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    xywh = labels[:, 1:5].astype(np.float32).copy()
    xywh[:, 0] *= width
    xywh[:, 1] *= height
    xywh[:, 2] *= width
    xywh[:, 3] *= height
    result = np.empty((len(xywh), 4), dtype=np.float32)
    result[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    result[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    result[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    result[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return result


def _resize_and_pad(
    rgb: np.ndarray,
    target_hw: tuple[int, int],
    mode: str,
    pad_value: int = 114,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Return RGB input, scale and (pad_x, pad_y) in model-input pixels."""

    target_h, target_w = target_hw
    source_h, source_w = rgb.shape[:2]
    if mode not in {"resize", "native_pad"}:
        raise ValueError("mode must be 'resize' or 'native_pad'")

    scale = min(target_w / source_w, target_h / source_h)
    if mode == "native_pad":
        scale = min(1.0, scale)

    new_w = max(1, int(round(source_w * scale)))
    new_h = max(1, int(round(source_h * scale)))
    if (new_w, new_h) == (source_w, source_h):
        resized = rgb
    else:
        resized = np.asarray(Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR))

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0
    left = int(round(pad_x))
    top = int(round(pad_y))
    right = min(target_w, left + new_w)
    bottom = min(target_h, top + new_h)
    canvas[top:bottom, left:right] = resized[: bottom - top, : right - left]
    return canvas, float(scale), (float(left), float(top))


def preprocess_patch(
    image_path: str | Path,
    target_hw: tuple[int, int] = (256, 256),
    mode: str = "resize",
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Load a native grayscale patch and prepare a 1x3xHxW ONNX tensor."""

    image_path = Path(image_path)
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.ndim != 2:
        raise ValueError(f"Expected a single-band image, got shape {gray.shape} for {image_path}")
    rgb = np.repeat(gray[..., None], 3, axis=2)
    canvas, scale, pad = _resize_and_pad(rgb, target_hw, mode=mode)
    tensor = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    metadata = {
        "source_width": int(gray.shape[1]),
        "source_height": int(gray.shape[0]),
        "input_width": int(target_hw[1]),
        "input_height": int(target_hw[0]),
        "scale": scale,
        "pad_x": pad[0],
        "pad_y": pad[1],
        "mode": mode,
    }
    return tensor, metadata, canvas


def _xyxy_input_to_native(
    boxes: np.ndarray,
    metadata: dict[str, Any],
) -> np.ndarray:
    """Undo letterbox scaling and padding."""

    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    result = boxes.astype(np.float32).copy()
    scale = float(metadata["scale"])
    result[:, [0, 2]] = (result[:, [0, 2]] - float(metadata["pad_x"])) / scale
    result[:, [1, 3]] = (result[:, [1, 3]] - float(metadata["pad_y"])) / scale
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, float(metadata["source_width"]))
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, float(metadata["source_height"]))
    return result


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_boxes = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection
    return intersection / np.maximum(union, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Class-agnostic NMS for the already class-filtered contact predictions."""

    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        overlaps = _iou_one_to_many(boxes[current], boxes[order[1:]])
        order = order[1:][overlaps <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_yolo_output(
    outputs: Sequence[np.ndarray],
    metadata: dict[str, Any],
    class_filter: int = 0,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_count: int = 4,
) -> list[dict[str, Any]]:
    """Decode the common Ultralytics YOLOv8 ONNX detection output.

    The helper accepts either [1, C, N] or [1, N, C] output layouts and both
    C=4+nc and C=5+nc variants. The selected class is returned as ``contact``.
    """

    if not outputs:
        return []
    output = max(outputs, key=lambda value: int(np.prod(value.shape)))
    array = np.asarray(output)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Unsupported ONNX detection output shape: {output.shape}")

    expected_c_without_objectness = 4 + class_count
    expected_c_with_objectness = 5 + class_count
    if array.shape[0] in {expected_c_without_objectness, expected_c_with_objectness}:
        array = array.T
    elif array.shape[1] not in {expected_c_without_objectness, expected_c_with_objectness}:
        # Some exports have a class count in metadata that differs from the
        # advertised card. Keep the standard [x,y,w,h, class scores] layout.
        if array.shape[0] < array.shape[1]:
            array = array.T

    channels = array.shape[1]
    if channels == expected_c_with_objectness:
        boxes_cxcywh = array[:, :4]
        objectness = array[:, 4:5]
        class_scores = array[:, 5:]
    elif channels >= expected_c_without_objectness:
        boxes_cxcywh = array[:, :4]
        objectness = None
        class_scores = array[:, 4:]
    else:
        raise ValueError(f"Cannot decode output shape {output.shape}; interpreted channels={channels}")

    if class_filter >= class_scores.shape[1]:
        return []
    if objectness is None:
        scores = class_scores[:, class_filter]
    else:
        scores = objectness[:, 0] * class_scores[:, class_filter]
    keep = scores >= float(confidence_threshold)
    if not np.any(keep):
        return []

    boxes_cxcywh = boxes_cxcywh[keep].astype(np.float32)
    scores = scores[keep].astype(np.float32)
    boxes = np.empty_like(boxes_cxcywh)
    boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    keep_indices = nms(boxes, scores, iou_threshold)
    boxes = _xyxy_input_to_native(boxes[keep_indices], metadata)
    scores = scores[keep_indices]
    return [
        {
            "class_id": int(class_filter),
            "class_name": "contact",
            "confidence": float(score),
            "bbox_xyxy": [float(value) for value in box],
        }
        for box, score in zip(boxes, scores)
    ]


def run_onnx_inference(
    session: Any,
    input_name: str,
    image_path: str | Path,
    target_hw: tuple[int, int] = (256, 256),
    preprocess_mode: str = "resize",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    class_filter: int = 0,
    class_count: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    tensor, metadata, display_image = preprocess_patch(image_path, target_hw, preprocess_mode)
    outputs = session.run(None, {input_name: tensor})
    predictions = decode_yolo_output(
        outputs,
        metadata,
        class_filter=class_filter,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        class_count=class_count,
    )
    return predictions, metadata, display_image


def match_predictions(
    predictions: Sequence[dict[str, Any]],
    ground_truth_xyxy: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedy one-to-one matching for one patch."""

    ordered = sorted(predictions, key=lambda item: float(item["confidence"]), reverse=True)
    matched_gt: set[int] = set()
    rows: list[dict[str, Any]] = []
    tp = 0
    for prediction in ordered:
        box = np.asarray(prediction["bbox_xyxy"], dtype=np.float32)
        overlaps = _iou_one_to_many(box, ground_truth_xyxy)
        best_index = int(np.argmax(overlaps)) if len(overlaps) else -1
        best_iou = float(overlaps[best_index]) if best_index >= 0 else 0.0
        is_tp = best_index >= 0 and best_iou >= iou_threshold and best_index not in matched_gt
        if is_tp:
            matched_gt.add(best_index)
            tp += 1
        rows.append({**prediction, "is_true_positive": bool(is_tp), "best_iou": best_iou})
    return {
        "predictions": rows,
        "tp": tp,
        "fp": len(ordered) - tp,
        "fn": int(len(ground_truth_xyxy) - tp),
    }


def aggregate_metrics(records: Sequence[dict[str, Any]], iou_threshold: float = 0.5) -> dict[str, Any]:
    """Aggregate fixed-threshold TP/FP/FN metrics by group."""

    total = {"tp": 0, "fp": 0, "fn": 0, "images": len(records)}
    by_group: dict[str, dict[str, Any]] = {}
    for record in records:
        match = match_predictions(record["predictions"], record["ground_truth_xyxy"], iou_threshold)
        group = str(record["group"])
        group_total = by_group.setdefault(group, {"tp": 0, "fp": 0, "fn": 0, "images": 0})
        for key in ("tp", "fp", "fn"):
            total[key] += int(match[key])
            group_total[key] += int(match[key])
        group_total["images"] += 1
    for item in [total, *by_group.values()]:
        denominator_precision = item["tp"] + item["fp"]
        denominator_recall = item["tp"] + item["fn"]
        item["precision"] = item["tp"] / denominator_precision if denominator_precision else 0.0
        item["recall"] = item["tp"] / denominator_recall if denominator_recall else 0.0
        item["f1"] = (
            2 * item["precision"] * item["recall"] / (item["precision"] + item["recall"])
            if item["precision"] + item["recall"]
            else 0.0
        )
    return {"overall": total, "by_group": by_group, "iou_threshold": iou_threshold}


def average_precision(records: Sequence[dict[str, Any]], iou_threshold: float = 0.5) -> float:
    """Compute integral AP for the single filtered class at a fixed IoU."""

    total_ground_truth = sum(len(record["ground_truth_xyxy"]) for record in records)
    if total_ground_truth == 0:
        return float("nan")
    candidates: list[tuple[float, int, np.ndarray]] = []
    for record_index, record in enumerate(records):
        for prediction in record["predictions"]:
            candidates.append(
                (
                    float(prediction["confidence"]),
                    record_index,
                    np.asarray(prediction["bbox_xyxy"], dtype=np.float32),
                )
            )
    candidates.sort(key=lambda item: item[0], reverse=True)
    matched_by_record: dict[int, set[int]] = {}
    true_positives: list[float] = []
    false_positives: list[float] = []
    for _, record_index, box in candidates:
        ground_truth = records[record_index]["ground_truth_xyxy"]
        overlaps = _iou_one_to_many(box, ground_truth)
        best_index = int(np.argmax(overlaps)) if len(overlaps) else -1
        matched = matched_by_record.setdefault(record_index, set())
        if best_index >= 0 and float(overlaps[best_index]) >= iou_threshold and best_index not in matched:
            matched.add(best_index)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)
    if not true_positives:
        return 0.0
    cumulative_tp = np.cumsum(np.asarray(true_positives, dtype=np.float64))
    cumulative_fp = np.cumsum(np.asarray(false_positives, dtype=np.float64))
    recalls = cumulative_tp / total_ground_truth
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    recall_points = np.concatenate(([0.0], recalls, [1.0]))
    precision_points = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(precision_points) - 2, -1, -1):
        precision_points[index] = max(precision_points[index], precision_points[index + 1])
    changed = np.where(recall_points[1:] != recall_points[:-1])[0]
    return float(np.sum((recall_points[changed + 1] - recall_points[changed]) * precision_points[changed + 1]))


def draw_overlay(
    image_path: str | Path,
    ground_truth_xyxy: np.ndarray,
    predictions: Sequence[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Write a separate visualization; never writes beside the input image."""

    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box in ground_truth_xyxy:
        draw.rectangle(tuple(float(value) for value in box), outline=(255, 220, 0), width=2)
    matched = match_predictions(predictions, ground_truth_xyxy)
    for prediction in matched["predictions"]:
        color = (0, 220, 80) if prediction["is_true_positive"] else (255, 60, 60)
        box = prediction["bbox_xyxy"]
        draw.rectangle(tuple(float(value) for value in box), outline=color, width=2)
        draw.text((float(box[0]) + 1, float(box[1]) + 1), f"{prediction['confidence']:.2f}", fill=color)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def json_dump(value: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
