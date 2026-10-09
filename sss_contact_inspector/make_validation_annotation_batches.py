#!/usr/bin/env python3
"""Partition validation images into reproducible manual-annotation batches."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "yolo_manual_annotation_grid_150px_stride100_v1_cleaned"
SOURCE_INDEX = SOURCE / "patch_index.csv"
OUTPUT = SOURCE / "validation_annotation_batches_1000"
BATCH_SIZE = 1000
GROUP_COUNT = 11
RANDOM_SEED = 20260921


class BatchError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_validation_rows() -> tuple[list[str], list[dict[str, str]]]:
    with SOURCE_INDEX.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row["split"] == "val"]
    if len(rows) != 10756:
        raise BatchError(f"expected 10,756 validation patches, found {len(rows)}")
    if any(row["split"] != "val" for row in rows):
        raise BatchError("non-validation row entered validation batches")
    return fields, rows


def serpentine_order(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_y: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_y[int(row["y0"])].append(row)
    ordered: list[dict[str, str]] = []
    for row_index, y0 in enumerate(sorted(by_y)):
        line = sorted(by_y[y0], key=lambda row: int(row["x0"]))
        if row_index % 2:
            line.reverse()
        ordered.extend(line)
    if len({row["patch_id"] for row in ordered}) != len(rows):
        raise BatchError("serpentine order duplicated or dropped a patch")
    return ordered


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def group_summary(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(group_rows),
        "first_patch_id": group_rows[0]["patch_id"],
        "last_patch_id": group_rows[-1]["patch_id"],
        "min_x0": min(int(row["x0"]) for row in group_rows),
        "max_x0": max(int(row["x0"]) for row in group_rows),
        "min_y0": min(int(row["y0"]) for row in group_rows),
        "max_y0": max(int(row["y0"]) for row in group_rows),
        "min_centre_easting": min(float(row["centre_easting"]) for row in group_rows),
        "max_centre_easting": max(float(row["centre_easting"]) for row in group_rows),
        "min_centre_northing": min(float(row["centre_northing"]) for row in group_rows),
        "max_centre_northing": max(float(row["centre_northing"]) for row in group_rows),
    }


def main() -> int:
    if not SOURCE.is_dir() or not SOURCE_INDEX.is_file():
        raise BatchError("cleaned validation source is missing")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise BatchError(f"batch output already exists and is non-empty: {OUTPUT}")
    fields, rows = read_validation_rows()
    source_index_hash_before = sha256_file(SOURCE_INDEX)
    ordered = serpentine_order(rows)
    spatial_chunks = [ordered[start : start + BATCH_SIZE] for start in range(0, len(ordered), BATCH_SIZE)]
    if len(spatial_chunks) != GROUP_COUNT or [len(chunk) for chunk in spatial_chunks] != [1000] * 10 + [756]:
        raise BatchError("unexpected batch sizes")
    # Keep the 756-patch remainder as the final group; randomize the ten
    # full-size spatial chunks so every early group still has 1,000 images.
    full_group_order = list(range(GROUP_COUNT - 1))
    random.Random(RANDOM_SEED).shuffle(full_group_order)
    order = full_group_order + [GROUP_COUNT - 1]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    batch_fields = fields + [
        "annotation_group",
        "annotation_order",
        "spatial_chunk",
        "within_group_order",
        "source_dataset_file_relative_path",
        "batch_image_relative_path",
        "label_relative_path",
        "label_status",
    ]
    master_rows: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    link_modes: dict[str, int] = defaultdict(int)

    for annotation_index, spatial_index in enumerate(order, start=1):
        group_name = f"group_{annotation_index:02d}"
        group_dir = OUTPUT / group_name
        image_dir = group_dir / "images"
        label_dir = group_dir / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        group_rows: list[dict[str, Any]] = []
        for within_index, source_row in enumerate(spatial_chunks[spatial_index], start=1):
            source_path = SOURCE / source_row["file_relative_path"]
            if not source_path.is_file():
                raise BatchError(f"source validation image missing: {source_path}")
            image_relative = Path("images") / source_path.name
            label_relative = Path("labels") / (source_path.stem + ".txt")
            destination = group_dir / image_relative
            mode = link_or_copy(source_path, destination)
            link_modes[mode] += 1
            row = {
                **source_row,
                "annotation_group": group_name,
                "annotation_order": annotation_index,
                "spatial_chunk": spatial_index + 1,
                "within_group_order": within_index,
                "source_dataset_file_relative_path": source_row["file_relative_path"],
                "batch_image_relative_path": (Path(group_name) / image_relative).as_posix(),
                "label_relative_path": (Path(group_name) / label_relative).as_posix(),
                "label_status": "pending_manual_annotation",
                "file_relative_path": image_relative.as_posix(),
            }
            group_rows.append(row)
            master_rows.append(row)
        write_csv(group_dir / "patch_index.csv", batch_fields, group_rows)
        group_summaries.append(
            {
                "annotation_order": annotation_index,
                "annotation_group": group_name,
                "spatial_chunk": spatial_index + 1,
                **group_summary(group_rows),
            }
        )
        (group_dir / "README.md").write_text(
            f"# Validation annotation {group_name}\n\n"
            f"This group contains {len(group_rows)} validation patches in the recommended annotation order. "
            "Images are native 150x150 grayscale PNGs. The labels directory is intentionally empty until manual YOLO annotation is completed.\n\n"
            "Keep the within-group order when practical; adjacent overlapping patches were kept near each other to reduce inconsistent repeated marking.\n",
            encoding="utf-8",
        )

    master_rows.sort(key=lambda row: int(row["annotation_order"]) * 100000 + int(row["within_group_order"]))
    write_csv(OUTPUT / "validation_batches.csv", batch_fields, master_rows)
    group_fields = [
        "annotation_order",
        "annotation_group",
        "spatial_chunk",
        "count",
        "first_patch_id",
        "last_patch_id",
        "min_x0",
        "max_x0",
        "min_y0",
        "max_y0",
        "min_centre_easting",
        "max_centre_easting",
        "min_centre_northing",
        "max_centre_northing",
    ]
    write_csv(OUTPUT / "group_summary.csv", group_fields, group_summaries)

    source_index_hash_after = sha256_file(SOURCE_INDEX)
    if source_index_hash_before != source_index_hash_after:
        raise BatchError("source patch_index.csv changed while batching")
    group_sizes = [int(summary["count"]) for summary in sorted(group_summaries, key=lambda row: row["annotation_order"])]
    if len(master_rows) != len(rows) or len({row["patch_id"] for row in master_rows}) != len(rows):
        raise BatchError("batch manifest does not cover each validation patch exactly once")
    if group_sizes != [1000] * 10 + [756]:
        raise BatchError(f"unexpected annotation group sizes: {group_sizes}")

    manifest = {
        "manifest_version": "validation-annotation-batches-1000-v1",
        "purpose": "Manual YOLO annotation batching for the cleaned validation image set.",
        "source_dataset": str(SOURCE),
        "source_patch_index_sha256": source_index_hash_before,
        "total_validation_patches": len(rows),
        "group_count": GROUP_COUNT,
        "batch_size": BATCH_SIZE,
        "group_sizes_in_annotation_order": group_sizes,
        "ordering_recommendation": "Keep adjacent overlapping patches together in spatial serpentine chunks, randomize the ten full-size chunks with a fixed seed, and keep the 756-patch remainder as group 11.",
        "ordering": {
            "within_spatial_chunk": "raster y0 rows from north to south; x0 direction alternates each row",
            "group_order_randomized": True,
            "remainder_group_fixed_last": True,
            "random_seed": RANDOM_SEED,
            "spatial_chunk_to_annotation_order": {str(index + 1): order.index(index) + 1 for index in range(GROUP_COUNT)},
        },
        "groups": sorted(group_summaries, key=lambda row: row["annotation_order"]),
        "storage": {
            "link_or_copy_counts": dict(link_modes),
            "source_images_modified": False,
            "label_files_present": False,
        },
        "manual_annotation": {
            "label_status": "pending_manual_annotation",
            "use_group_images_directories": True,
            "write_yolo_txt_to_each_group_labels_directory": True,
        },
    }
    (OUTPUT / "batch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "README.md").write_text(
        "# Validation annotation batches\n\n"
        "Recommended workflow: annotate `group_01` through `group_11` in order. "
        "The groups are spatially coherent internally, which keeps the 50 px-overlap neighbours close enough to reduce inconsistent repeated labels. "
        "The order of the 11 spatial chunks was then randomized with the fixed seed recorded in `batch_manifest.json`, so early groups are not tied to one geographic corner.\n\n"
        "Each group contains an `images/` directory and an empty `labels/` directory. "
        "Write one YOLO `.txt` file per image into that same group’s `labels/` directory. "
        "`validation_batches.csv` is the master mapping from patch coordinates to group/order.\n\n"
        "The images are hard links to the cleaned validation dataset where supported; no source patch pixels were changed or duplicated.\n",
        encoding="utf-8",
    )

    complete = {
        "status": "COMPLETE_VALIDATION_BATCHES",
        "batch_manifest": "batch_manifest.json",
        "validation_batches": "validation_batches.csv",
        "groups": GROUP_COUNT,
        "group_sizes": group_sizes,
        "all_validation_patches_assigned_once": True,
        "label_files_present": False,
        "manual_annotation_pending": True,
    }
    (OUTPUT / "COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": complete["status"], "output": str(OUTPUT), "group_order": order, "group_sizes": group_sizes, "link_or_copy_counts": dict(link_modes)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
