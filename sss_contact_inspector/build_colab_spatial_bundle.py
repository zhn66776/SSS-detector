#!/usr/bin/env python3
"""Build a spatially held-out, Drive-ready YOLOv8 SSS bundle.

The annotation group numbers are randomized.  The ``spatial_chunk`` field in
group_summary.csv is the actual north-to-south spatial order.  This builder
uses chunks 1-7 for train, 8-9 for validation, and 10-11 for test, then moves
patches within 15 m of a cross-split footprint into a buffer directory.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "sss_contact_inspector" / "yolo_manual_annotation_grid_150px_stride100_v1_cleaned" / "validation_annotation_batches_1000"
OUTPUT = PROJECT / "sss_contact_yolov8_spatial_split_bundle"
ZIP_PATH = PROJECT / "sss_contact_yolov8_spatial_split_bundle.zip"
MIN_CROSS_SPLIT_GAP_M = 15.0


def read_label_rows(image_json: Path) -> list[str]:
    """Convert AnyLabeling rectangles to normalized one-class YOLO rows."""

    if not image_json.exists():
        return []
    payload = json.loads(image_json.read_text(encoding="utf-8"))
    rows: list[str] = []
    for index, shape in enumerate(payload.get("shapes", [])):
        if shape.get("label") != "contact":
            raise ValueError(f"{image_json}:{index}: unexpected label {shape.get('label')!r}")
        points = shape.get("points", [])
        if shape.get("shape_type") != "rectangle" or len(points) != 2:
            raise ValueError(f"{image_json}:{index}: expected a two-point rectangle")
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if not (0 <= x0 <= x1 <= 150 and 0 <= y0 <= y1 <= 150):
            raise ValueError(f"{image_json}:{index}: box outside 150x150 image")
        values = (0, (x0 + x1) / 300, (y0 + y1) / 300, (x1 - x0) / 150, (y1 - y0) / 150)
        rows.append(" ".join(str(values[0]) if j == 0 else f"{value:.8f}" for j, value in enumerate(values)))
    return rows


def load_rows() -> tuple[list[dict[str, object]], dict[str, int]]:
    summary_rows = list(csv.DictReader((SOURCE / "group_summary.csv").open(newline="", encoding="utf-8")))
    group_to_chunk = {row["annotation_group"]: int(row["spatial_chunk"]) for row in summary_rows}
    patch_rows: list[dict[str, object]] = []
    for row in csv.DictReader((SOURCE / "validation_batches.csv").open(newline="", encoding="utf-8")):
        group = row["annotation_group"]
        patch_id = row["patch_id"]
        json_path = SOURCE / group / "images" / f"{patch_id}.json"
        label_rows = read_label_rows(json_path)
        patch_rows.append(
            {
                "patch_id": patch_id,
                "group": group,
                "spatial_chunk": group_to_chunk[group],
                "source_image": SOURCE / group / "images" / f"{patch_id}.png",
                "source_json": json_path if json_path.exists() else None,
                "label_rows": label_rows,
                "x0": float(row["footprint_left_easting"]),
                "x1": float(row["footprint_right_easting"]),
                "y0": float(row["footprint_bottom_northing"]),
                "y1": float(row["footprint_top_northing"]),
            }
        )
    return patch_rows, group_to_chunk


def coarse_split(chunk: int) -> str:
    if 1 <= chunk <= 7:
        return "train"
    if 8 <= chunk <= 9:
        return "val"
    if 10 <= chunk <= 11:
        return "test"
    raise ValueError(f"Unexpected spatial chunk {chunk}")


def rectangle_gap(left: dict[str, object], right: dict[str, object]) -> float:
    dx = max(0.0, float(left["x0"]) - float(right["x1"]), float(right["x0"]) - float(left["x1"]))
    dy = max(0.0, float(left["y0"]) - float(right["y1"]), float(right["y0"]) - float(left["y1"]))
    return math.hypot(dx, dy)


def mark_buffer(patch_rows: list[dict[str, object]]) -> set[str]:
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in patch_rows:
        by_split[coarse_split(int(row["spatial_chunk"]))].append(row)

    buffered: set[str] = set()
    # The non-adjacent regions are already far apart.  Checking all pairs is
    # still small enough for this 10,756-patch manifest and keeps the rule
    # explicit and reproducible.
    for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
        for left in by_split[left_name]:
            for right in by_split[right_name]:
                if abs(float(left["x0"]) - float(right["x0"])) > 220:
                    continue
                if rectangle_gap(left, right) < MIN_CROSS_SPLIT_GAP_M:
                    buffered.add(str(left["patch_id"]))
                    buffered.add(str(right["patch_id"]))
    return buffered


def split_counts(patch_rows: list[dict[str, object]], buffered: set[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in patch_rows:
        split = "buffer" if str(row["patch_id"]) in buffered else coarse_split(int(row["spatial_chunk"]))
        counts[split]["patches"] += 1
        counts[split]["boxes"] += len(row["label_rows"])
        counts[split]["positive_patches"] += int(bool(row["label_rows"]))
    return {key: dict(value) for key, value in counts.items()}


def write_metadata(patch_rows: list[dict[str, object]], buffered: set[str], group_to_chunk: dict[str, int], counts: dict[str, dict[str, int]]) -> None:
    metadata = OUTPUT / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / "group_summary.csv", metadata / "group_summary.csv")
    shutil.copy2(SOURCE / "validation_batches.csv", metadata / "validation_batches.csv")

    fieldnames = ["patch_id", "original_group", "spatial_chunk", "coarse_split", "final_split", "x0", "x1", "y0", "y1", "positive", "box_count"]
    with (metadata / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(patch_rows, key=lambda item: str(item["patch_id"])):
            patch_id = str(row["patch_id"])
            writer.writerow(
                {
                    "patch_id": patch_id,
                    "original_group": row["group"],
                    "spatial_chunk": row["spatial_chunk"],
                    "coarse_split": coarse_split(int(row["spatial_chunk"])),
                    "final_split": "buffer" if patch_id in buffered else coarse_split(int(row["spatial_chunk"])),
                    "x0": row["x0"],
                    "x1": row["x1"],
                    "y0": row["y0"],
                    "y1": row["y1"],
                    "positive": int(bool(row["label_rows"])),
                    "box_count": len(row["label_rows"]),
                }
            )

    manifest = {
        "bundle_version": "sss-yolov8-spatial-split-v1",
        "source": "Costa Head cleaned validation annotation batches",
        "source_patch_count": len(patch_rows),
        "source_box_count": sum(len(row["label_rows"]) for row in patch_rows),
        "spatial_order_field": "spatial_chunk",
        "group_numbers_are_annotation_order": True,
        "coarse_split": {
            "train_spatial_chunks": list(range(1, 8)),
            "val_spatial_chunks": [8, 9],
            "test_spatial_chunks": [10, 11],
        },
        "group_to_spatial_chunk": group_to_chunk,
        "cross_split_buffer": {
            "minimum_gap_m": MIN_CROSS_SPLIT_GAP_M,
            "buffered_patch_count": len(buffered),
            "buffer_not_in_train_val_test": True,
        },
        "counts": counts,
        "class": {"nc": 1, "name": "contact", "image_size_px": [150, 150], "channels": 1},
    }
    (metadata / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (OUTPUT / "dataset.yaml").write_text(
        json.dumps(
            {
                "train": "data/images/train",
                "val": "data/images/val",
                "test": "data/images/test",
                "nc": 1,
                "names": ["contact"],
                "channels": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    group_text = ", ".join(f"{group} -> chunk {chunk}" for group, chunk in sorted(group_to_chunk.items()))
    (OUTPUT / "README.md").write_text(
        "# SSS YOLOv8 spatial split bundle\n\n"
        "This bundle contains 150x150 single-channel PNG patches and one-class YOLO labels.\n\n"
        "## Split\n\n"
        "- train: spatial chunks 1-7\n"
        "- validation: spatial chunks 8-9\n"
        "- test: spatial chunks 10-11\n"
        f"- buffer: patches within {MIN_CROSS_SPLIT_GAP_M:g} m of a cross-split footprint; not used by train/val/test\n\n"
        "The group numbers are annotation order and were randomized. Use `spatial_chunk`, not the numeric group name, to reason about geography.\n\n"
        f"Group mapping: {group_text}\n\n"
        "`dataset.yaml` is ready for Ultralytics. The original AnyLabeling JSON sidecars are retained under `metadata/original_anylabeling_json/` for provenance.\n",
        encoding="utf-8",
    )


def copy_data(patch_rows: list[dict[str, object]], buffered: set[str]) -> None:
    for row in patch_rows:
        patch_id = str(row["patch_id"])
        split = "buffer" if patch_id in buffered else coarse_split(int(row["spatial_chunk"]))
        image_dir = OUTPUT / "data" / "images" / split
        label_dir = OUTPUT / "data" / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source_image"], image_dir / f"{patch_id}.png")
        (label_dir / f"{patch_id}.txt").write_text("\n".join(row["label_rows"]) + ("\n" if row["label_rows"] else ""), encoding="utf-8")

        if row["source_json"] is not None:
            destination = OUTPUT / "metadata" / "original_anylabeling_json" / str(row["group"])
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["source_json"], destination / f"{patch_id}.json")


def main() -> None:
    if OUTPUT.exists() or ZIP_PATH.exists():
        raise SystemExit(f"Refusing to overwrite {OUTPUT} or {ZIP_PATH}")
    patch_rows, group_to_chunk = load_rows()
    buffered = mark_buffer(patch_rows)
    counts = split_counts(patch_rows, buffered)
    OUTPUT.mkdir(parents=True)
    copy_data(patch_rows, buffered)
    write_metadata(patch_rows, buffered, group_to_chunk, counts)

    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT.parent))

    print(json.dumps({"output": str(OUTPUT), "zip": str(ZIP_PATH), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
