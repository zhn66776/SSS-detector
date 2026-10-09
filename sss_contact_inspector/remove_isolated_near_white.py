#!/usr/bin/env python3
"""Remove the explicitly reviewed 21 isolated candidates from the cleaned derivative."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "yolo_manual_annotation_grid_150px_stride100_v1_cleaned"
QUARANTINE_CSV = DATASET / "quarantine_near_white.csv"
COMPONENT_CSV = DATASET / "white_component_summary.csv"
MANIFEST_PATH = DATASET / "dataset_manifest.json"
COMPLETE_PATH = DATASET / "COMPLETE.json"
REVIEW_SHEET = DATASET / "review" / "isolated_near_white_review_sheet.png"


class UpdateError(RuntimeError):
    pass


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, document: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if not DATASET.is_dir() or not MANIFEST_PATH.is_file() or not COMPLETE_PATH.is_file():
        raise UpdateError("cleaned dataset is missing its completion artifacts")
    with QUARANTINE_CSV.open(encoding="utf-8", newline="") as handle:
        quarantine_reader = csv.DictReader(handle)
        quarantine_fields = list(quarantine_reader.fieldnames or [])
        quarantine_rows = list(quarantine_reader)
    isolated = [row for row in quarantine_rows if int(row["component_size"]) == 1]
    if len(isolated) != 21:
        raise UpdateError(f"expected exactly 21 isolated rows, found {len(isolated)}")
    if any(row["split"] != "train" for row in isolated):
        raise UpdateError("the isolated rows are not all train rows")
    if any(not (DATASET / row["quarantine_file_relative_path"]).is_file() for row in isolated):
        raise UpdateError("one or more isolated quarantine files are missing")

    deleted_bytes = 0
    for row in isolated:
        target = DATASET / row["quarantine_file_relative_path"]
        if not target.is_file() or DATASET not in target.parents:
            raise UpdateError(f"refusing to remove unexpected target: {target}")
        deleted_bytes += target.stat().st_size
        target.unlink()

    remaining_quarantine = [row for row in quarantine_rows if int(row["component_size"]) != 1]
    write_csv(QUARANTINE_CSV, quarantine_fields, remaining_quarantine)

    with COMPONENT_CSV.open(encoding="utf-8", newline="") as handle:
        component_reader = csv.DictReader(handle)
        component_fields = list(component_reader.fieldnames or [])
        component_rows = list(component_reader)
    remaining_components = [row for row in component_rows if int(row["component_size"]) != 1]
    write_csv(COMPONENT_CSV, component_fields, remaining_components)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    counts = manifest.setdefault("counts", {})
    counts["quarantined_near_white_total"] = len(remaining_quarantine)
    counts["quarantined_near_white_train"] = sum(row["split"] == "train" for row in remaining_quarantine)
    counts["quarantined_near_white_validation"] = sum(row["split"] == "val" for row in remaining_quarantine)
    counts["near_white_components_8_connected"] = len(remaining_components)
    counts["near_white_components_ge_2"] = len(remaining_components)
    counts["isolated_near_white_components"] = 0
    counts["largest_near_white_component"] = max((int(row["component_size"]) for row in remaining_components), default=0)
    counts["isolated_candidates_deleted_from_this_derivative"] = len(isolated)
    cleaning = manifest.setdefault("white_cleaning", {})
    cleaning["isolated_candidates_deleted_from_this_derivative"] = True
    cleaning["isolated_candidates_deleted_count"] = len(isolated)
    cleaning["isolated_candidates_retained_in_quarantine"] = False
    cleaning["parent_v1_still_preserved_as_archival_source"] = True
    storage = manifest.setdefault("storage", {})
    storage["quarantine_png_bytes"] = sum(
        (DATASET / row["quarantine_file_relative_path"]).stat().st_size for row in remaining_quarantine
    )
    storage["deleted_isolated_quarantine_png_bytes"] = deleted_bytes
    storage["link_or_copy_counts"] = {"hardlink": len(remaining_quarantine) + int(counts.get("retained_total", 0))}
    manifest["artifacts"]["isolated_review_sheet"] = "removed after explicit user deletion request"
    write_json(MANIFEST_PATH, manifest)

    complete = json.loads(COMPLETE_PATH.read_text(encoding="utf-8"))
    complete["quarantined_near_white"] = len(remaining_quarantine)
    complete["isolated_near_white_components"] = 0
    complete["isolated_candidates_deleted_from_this_derivative"] = len(isolated)
    complete["isolated_review_sheet_removed"] = True
    write_json(COMPLETE_PATH, complete)

    if REVIEW_SHEET.exists():
        REVIEW_SHEET.unlink()
    review_dir = REVIEW_SHEET.parent
    if review_dir.is_dir() and not any(review_dir.iterdir()):
        review_dir.rmdir()

    with QUARANTINE_CSV.open(encoding="utf-8", newline="") as handle:
        final_rows = list(csv.DictReader(handle))
    if any(int(row["component_size"]) == 1 for row in final_rows):
        raise UpdateError("isolated rows still appear in quarantine CSV")
    print(json.dumps({
        "status": "UPDATED",
        "dataset": str(DATASET),
        "deleted_isolated_patches": len(isolated),
        "remaining_quarantined_clustered_patches": len(final_rows),
        "deleted_png_bytes": deleted_bytes,
        "review_sheet_removed": True,
        "training_images_unchanged": True,
        "parent_v1_preserved": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
