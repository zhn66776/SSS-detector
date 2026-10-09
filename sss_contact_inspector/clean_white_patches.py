#!/usr/bin/env python3
"""Create a reversible near-white-cleaned derivative of the manual grid.

The original complete grid is never modified.  Suspect near-white patches are
hard-linked (or copied if hard-links are unavailable) into a quarantine area,
while only the remaining patches are exposed under images/train and
images/val.  The coordinate/component report makes the spatial decision
auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "yolo_manual_annotation_grid_150px_stride100_v1"
OUTPUT = ROOT / "yolo_manual_annotation_grid_150px_stride100_v1_cleaned"

# Conservative empirical signature for the ECW's near-white/no-coverage fill.
# It matches the repeated right-upper blank-area pattern, not merely a high
# mean: a valid textured SSS patch normally has either lower pixels, higher
# variance, or many more distinct levels.
WHITE_MEAN_MIN = 252.5
WHITE_VARIANCE_MAX = 15.0
WHITE_MIN_PIXEL = 244
WHITE_UNIQUE_MAX = 20
PATCH_SIZE = 150
GRID_STRIDE = 100


class CleaningError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_feature(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (PATCH_SIZE, PATCH_SIZE):
            raise CleaningError(f"unexpected source image format: {path}")
        image.load()
        histogram = image.histogram()
    total = PATCH_SIZE * PATCH_SIZE
    first = next(index for index, count in enumerate(histogram) if count)
    last = max(index for index, count in enumerate(histogram) if count)
    unique = sum(1 for count in histogram if count)
    total_value = sum(index * count for index, count in enumerate(histogram))
    total_square = sum(index * index * count for index, count in enumerate(histogram))
    mean = total_value / total
    variance = max(0.0, total_square / total - mean * mean)
    return {
        "mean": mean,
        "variance": variance,
        "min_pixel": first,
        "max_pixel": last,
        "unique_pixel_values": unique,
        "fraction_ge_245": sum(histogram[245:]) / total,
        "fraction_ge_250": sum(histogram[250:]) / total,
    }


def is_near_white(feature: dict[str, Any]) -> bool:
    return (
        feature["mean"] >= WHITE_MEAN_MIN
        and feature["variance"] <= WHITE_VARIANCE_MAX
        and feature["min_pixel"] >= WHITE_MIN_PIXEL
        and feature["max_pixel"] == 255
        and feature["unique_pixel_values"] <= WHITE_UNIQUE_MAX
    )


def read_source_rows() -> tuple[list[str], list[dict[str, str]]]:
    path = SOURCE / "patch_index.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise CleaningError("source patch_index.csv is empty")
    if not (SOURCE / "COMPLETE.json").is_file():
        raise CleaningError("source grid is not marked COMPLETE")
    return fields, rows


def component_analysis(rows: list[dict[str, str]], features: dict[str, dict[str, Any]]):
    row_by_id = {row["patch_id"]: row for row in rows}
    cells: dict[tuple[str, int, int], str] = {}
    for row in rows:
        feature = features[row["patch_id"]]
        if is_near_white(feature):
            cell = (row["split"], int(row["x0"]) // GRID_STRIDE, int(row["y0"]) // GRID_STRIDE)
            cells[cell] = row["patch_id"]

    components: list[list[tuple[str, int, int]]] = []
    seen: set[tuple[str, int, int]] = set()
    for start in sorted(cells):
        if start in seen:
            continue
        seen.add(start)
        queue = [start]
        component: list[tuple[str, int, int]] = []
        for cell in queue:
            component.append(cell)
            split, grid_x, grid_y = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    neighbour = (split, grid_x + dx, grid_y + dy)
                    if neighbour in cells and neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
        components.append(component)

    components.sort(key=lambda component: (-len(component), component[0]))
    component_by_patch: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for index, component in enumerate(components, start=1):
        split = component[0][0]
        component_id = f"{split}_near_white_component_{index:04d}"
        patch_ids = [cells[cell] for cell in component]
        rows_for_component = [row_by_id[patch_id] for patch_id in patch_ids]
        x_values = [int(row["x0"]) for row in rows_for_component]
        y_values = [int(row["y0"]) for row in rows_for_component]
        e_values = [float(row["centre_easting"]) for row in rows_for_component]
        n_values = [float(row["centre_northing"]) for row in rows_for_component]
        size = len(component)
        component_class = "isolated" if size == 1 else "clustered"
        summary = {
            "component_id": component_id,
            "split": split,
            "component_size": size,
            "component_class": component_class,
            "min_x0": min(x_values),
            "max_x0": max(x_values),
            "min_y0": min(y_values),
            "max_y0": max(y_values),
            "min_centre_easting": min(e_values),
            "max_centre_easting": max(e_values),
            "min_centre_northing": min(n_values),
            "max_centre_northing": max(n_values),
            "centroid_easting": sum(e_values) / size,
            "centroid_northing": sum(n_values) / size,
        }
        summaries.append(summary)
        for patch_id in patch_ids:
            component_by_patch[patch_id] = {
                "component_id": component_id,
                "component_size": size,
                "component_class": component_class,
            }
    return component_by_patch, summaries


def format_float(value: float) -> str:
    return f"{value:.9f}"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (PATCH_SIZE, PATCH_SIZE):
                return False
            image.load()
        return True
    except Exception:
        return False


def dry_run(rows: list[dict[str, str]], features: dict[str, dict[str, Any]], component_by_patch: dict[str, dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    white_rows = [row for row in rows if row["patch_id"] in component_by_patch]
    retained = [row for row in rows if row["patch_id"] not in component_by_patch]
    by_split = defaultdict(int)
    for row in white_rows:
        by_split[row["split"]] += 1
    component_sizes = sorted((summary["component_size"] for summary in summaries), reverse=True)
    result = {
        "mode": "dry-run",
        "source": str(SOURCE),
        "output_if_run": str(OUTPUT),
        "threshold": {
            "mean_min": WHITE_MEAN_MIN,
            "variance_max": WHITE_VARIANCE_MAX,
            "min_pixel_min": WHITE_MIN_PIXEL,
            "unique_pixel_values_max": WHITE_UNIQUE_MAX,
        },
        "counts": {
            "source_total": len(rows),
            "near_white_total": len(white_rows),
            "near_white_by_split": dict(by_split),
            "retained_total": len(retained),
            "retained_train": sum(row["split"] == "train" for row in retained),
            "retained_val": sum(row["split"] == "val" for row in retained),
            "near_white_components_8_connected": len(summaries),
            "components_ge_2": sum(size >= 2 for size in component_sizes),
            "isolated_components": sum(size == 1 for size in component_sizes),
        },
        "component_sizes_largest_20": component_sizes[:20],
        "no_files_written": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def run(rows: list[dict[str, str]], fields: list[str], features: dict[str, dict[str, Any]], component_by_patch: dict[str, dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise CleaningError(f"cleaned output already exists and is non-empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in (
        OUTPUT / "images" / "train",
        OUTPUT / "images" / "val",
        OUTPUT / "quarantine_near_white" / "images" / "train",
        OUTPUT / "quarantine_near_white" / "images" / "val",
    ):
        path.mkdir(parents=True, exist_ok=True)

    patch_fields = fields + ["cleaning_status"]
    quarantine_fields = [
        "component_id",
        "component_size",
        "component_class",
        "split",
        "patch_id",
        "x0",
        "y0",
        "x1",
        "y1",
        "centre_pixel_x",
        "centre_pixel_y",
        "centre_easting",
        "centre_northing",
        "original_file_relative_path",
        "quarantine_file_relative_path",
        "mean",
        "variance",
        "min_pixel",
        "max_pixel",
        "unique_pixel_values",
        "fraction_ge_245",
        "fraction_ge_250",
        "cleaning_action",
    ]
    component_fields = [
        "component_id",
        "split",
        "component_size",
        "component_class",
        "min_x0",
        "max_x0",
        "min_y0",
        "max_y0",
        "min_centre_easting",
        "max_centre_easting",
        "min_centre_northing",
        "max_centre_northing",
        "centroid_easting",
        "centroid_northing",
    ]
    retained_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    link_modes: defaultdict[str, int] = defaultdict(int)
    started = time.monotonic()
    for index, row in enumerate(rows, start=1):
        source_file = SOURCE / row["file_relative_path"]
        if not source_file.is_file() or not valid_image(source_file):
            raise CleaningError(f"invalid source image during cleaning: {source_file}")
        suspect = row["patch_id"] in component_by_patch
        if suspect:
            destination_relative = Path("quarantine_near_white") / row["file_relative_path"]
            destination = OUTPUT / destination_relative
            status = "quarantined_near_white"
        else:
            destination_relative = Path(row["file_relative_path"])
            destination = OUTPUT / destination_relative
            status = "retained_non_near_white"
        link_modes[link_or_copy(source_file, destination)] += 1
        if not valid_image(destination):
            raise CleaningError(f"invalid cleaned image: {destination}")
        if suspect:
            component = component_by_patch[row["patch_id"]]
            feature = features[row["patch_id"]]
            quarantine_rows.append(
                {
                    **component,
                    **{key: row.get(key, "") for key in ("split", "patch_id", "x0", "y0", "x1", "y1", "centre_pixel_x", "centre_pixel_y", "centre_easting", "centre_northing")},
                    "original_file_relative_path": row["file_relative_path"],
                    "quarantine_file_relative_path": destination_relative.as_posix(),
                    **feature,
                    "cleaning_action": "excluded_from_images_but_preserved_for_review",
                }
            )
        else:
            retained_rows.append({**row, "cleaning_status": status, "file_relative_path": destination_relative.as_posix()})
        if index % 10000 == 0:
            print(f"cleaning {index}/{len(rows)} elapsed_s={time.monotonic()-started:.1f}", flush=True)

    retained_rows.sort(key=lambda row: (row["split"], int(row["y0"]), int(row["x0"])))
    quarantine_rows.sort(key=lambda row: (row["split"], int(row["y0"]), int(row["x0"])))
    summaries.sort(key=lambda row: (-int(row["component_size"]), row["split"], int(row["min_y0"]), int(row["min_x0"])))
    write_csv(OUTPUT / "patch_index.csv", patch_fields, retained_rows)
    write_csv(OUTPUT / "quarantine_near_white.csv", quarantine_fields, quarantine_rows)
    write_csv(OUTPUT / "white_component_summary.csv", component_fields, summaries)

    skipped_source = SOURCE / "skipped_windows.csv"
    skipped_destination = OUTPUT / "skipped_windows.csv"
    shutil.copy2(skipped_source, skipped_destination)

    # Re-run the exact cross-split rectangle validator on the retained rows.
    sys.path.insert(0, str(ROOT))
    import export_yolo_manual_grid as base  # noqa: E402

    train = [base.Candidate("train", int(row["x0"]), int(row["y0"])) for row in retained_rows if row["split"] == "train"]
    validation = [base.Candidate("val", int(row["x0"]), int(row["y0"])) for row in retained_rows if row["split"] == "val"]
    split_validation = base.split_footprint_validation(train, validation)
    if not split_validation["validation_passed"]:
        raise CleaningError("cleaned train/validation footprint validation failed")
    (OUTPUT / "split_validation.json").write_text(json.dumps(split_validation, indent=2) + "\n", encoding="utf-8")

    # No near-white signature may remain under the actual training image roots.
    retained_near_white = []
    total_cleaned_bytes = 0
    total_quarantine_bytes = 0
    for row in retained_rows:
        path = OUTPUT / row["file_relative_path"]
        feature = image_feature(path)
        if is_near_white(feature):
            retained_near_white.append(row["patch_id"])
        total_cleaned_bytes += path.stat().st_size
    for row in quarantine_rows:
        total_quarantine_bytes += (OUTPUT / row["quarantine_file_relative_path"]).stat().st_size
    if retained_near_white:
        raise CleaningError(f"near-white signature remains in cleaned images: {retained_near_white[:5]}")

    parent_manifest_path = SOURCE / "dataset_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_complete = json.loads((SOURCE / "COMPLETE.json").read_text(encoding="utf-8"))
    component_sizes = sorted((int(summary["component_size"]) for summary in summaries), reverse=True)
    isolated = sum(size == 1 for size in component_sizes)
    cleaned_manifest = {
        "manifest_version": "costa-yolo-manual-annotation-grid-150px-stride100-v1-cleaned",
        "purpose": "Reversible near-white-cleaned derivative for manual YOLO annotation; quarantine is retained for review.",
        "labels_present": False,
        "yolo_bbox_or_class_labels_present": False,
        "all_annotations_still_pending_manual_review": True,
        "parent_dataset": {
            "path": str(SOURCE),
            "dataset_manifest_sha256": sha256_file(parent_manifest_path),
            "complete_status": parent_complete.get("status"),
        },
        "source_inputs": parent_manifest.get("inputs"),
        "source_raster": parent_manifest.get("source_raster"),
        "grid": parent_manifest.get("grid"),
        "split_rule": parent_manifest.get("split_rule"),
        "white_cleaning": {
            "method": "image-level near-white signature plus 8-connected global-grid component audit",
            "mean_min": WHITE_MEAN_MIN,
            "variance_max": WHITE_VARIANCE_MAX,
            "min_pixel_min": WHITE_MIN_PIXEL,
            "max_pixel_required": 255,
            "unique_pixel_values_max": WHITE_UNIQUE_MAX,
            "eight_connected_neighbour_definition": "global grid cells with delta x and delta y each in {-1,0,1}, excluding (0,0); diagonal patch windows overlap because stride is 100 px",
            "suspect_patches_removed_from_training_roots": True,
            "suspect_patches_deleted": False,
            "original_v1_preserved": True,
        },
        "counts": {
            "source_total": len(rows),
            "retained_total": len(retained_rows),
            "retained_train": sum(row["split"] == "train" for row in retained_rows),
            "retained_validation": sum(row["split"] == "val" for row in retained_rows),
            "quarantined_near_white_total": len(quarantine_rows),
            "quarantined_near_white_train": sum(row["split"] == "train" for row in quarantine_rows),
            "quarantined_near_white_validation": sum(row["split"] == "val" for row in quarantine_rows),
            "near_white_components_8_connected": len(summaries),
            "near_white_components_ge_2": sum(size >= 2 for size in component_sizes),
            "isolated_near_white_components": isolated,
            "largest_near_white_component": component_sizes[0] if component_sizes else 0,
            "parent_buffer_windows_not_exported": parent_manifest.get("counts", {}).get("buffer_windows_discarded_not_exported"),
        },
        "storage": {
            "cleaned_png_bytes": total_cleaned_bytes,
            "quarantine_png_bytes": total_quarantine_bytes,
            "link_or_copy_counts": dict(link_modes),
        },
        "split_validation": split_validation,
        "artifacts": {
            "training_images": "images/train/",
            "validation_images": "images/val/",
            "retained_patch_index": "patch_index.csv",
            "quarantine_index": "quarantine_near_white.csv",
            "component_summary": "white_component_summary.csv",
            "skipped_windows": "skipped_windows.csv",
            "split_validation": "split_validation.json",
            "readme": "README.md",
        },
    }
    (OUTPUT / "dataset_manifest.json").write_text(json.dumps(cleaned_manifest, indent=2) + "\n", encoding="utf-8")
    readme = """# Costa Head SSS manual annotation grid — near-white-cleaned derivative

This is a reversible derivative of `yolo_manual_annotation_grid_150px_stride100_v1`.
It is still an unlabelled manual-annotation candidate set: no YOLO bounding
boxes or class labels are present.

Use only `images/train/` and `images/val/` for model-training preparation. The
`quarantine_near_white/` tree is deliberately excluded from those roots. It
contains the original near-white candidate PNGs for manual review, and
`quarantine_near_white.csv` gives their projected coordinates and global-grid
8-connected component IDs. The original v1 directory is also preserved.

The filter uses a conservative near-white signature measured on the raw
150×150 single-channel PNG: mean >= 252.5, variance <= 15, minimum pixel >=
244, maximum pixel = 255, and no more than 20 distinct grey values. It does
not inspect contacts, labels, CSV/DBF/shapefile records, or YOLO annotations.

The signature candidates were projected back to the same global 100 px grid.
`white_component_summary.csv` records their connected components. Diagonal
neighbours count as connected because 150 px patches with 100 px stride
overlap diagonally. Most suspect patches form a large continuous region; the
small/isolated candidates remain in quarantine rather than being deleted, so
an unusually bright valid SSS patch can be restored after human review.

The cleaned PNGs remain native 0.1 m/px, 150×150 px, lossless grayscale PNGs.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    required = [
        OUTPUT / "patch_index.csv",
        OUTPUT / "quarantine_near_white.csv",
        OUTPUT / "white_component_summary.csv",
        OUTPUT / "skipped_windows.csv",
        OUTPUT / "split_validation.json",
        OUTPUT / "dataset_manifest.json",
        OUTPUT / "README.md",
    ]
    if any(not path.is_file() for path in required):
        raise CleaningError("cleaned required artifact missing")
    if not split_validation["validation_passed"] or retained_near_white:
        raise CleaningError("cleaned COMPLETE gate failed")
    complete = {
        "status": "COMPLETE_CLEANED",
        "parent_dataset": str(SOURCE),
        "training_images_are_near_white_filtered": True,
        "near_white_quarantine_preserved": True,
        "retained_train": sum(row["split"] == "train" for row in retained_rows),
        "retained_validation": sum(row["split"] == "val" for row in retained_rows),
        "quarantined_near_white": len(quarantine_rows),
        "isolated_near_white_components": isolated,
        "minimum_train_validation_footprint_gap_m": split_validation["minimum_gap_m"],
        "labels_present": False,
        "all_yolo_annotations_still_pending_manual_completion": True,
    }
    (OUTPUT / "COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": complete["status"], "output": str(OUTPUT), "counts": cleaned_manifest["counts"], "storage": cleaned_manifest["storage"], "split_validation": split_validation}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="scan and report; do not create the cleaned output")
    args = parser.parse_args()
    try:
        fields, rows = read_source_rows()
        features: dict[str, dict[str, Any]] = {}
        started = time.monotonic()
        for index, row in enumerate(rows, start=1):
            features[row["patch_id"]] = image_feature(SOURCE / row["file_relative_path"])
            if index % 50000 == 0:
                print(f"scanned {index}/{len(rows)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        component_by_patch, summaries = component_analysis(rows, features)
        if args.dry_run:
            dry_run(rows, features, component_by_patch, summaries)
        else:
            run(rows, fields, features, component_by_patch, summaries)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
