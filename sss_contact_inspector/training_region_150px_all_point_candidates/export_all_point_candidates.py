#!/usr/bin/env python3
"""Export every decodable Costa point contact inside the supplied polygon.

This is intentionally a separate derived dataset from the thresholded
717-sample export.  Mean and variance are recorded for review, but neither is
used to exclude a patch in this version.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageStat


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
THRESHOLDED_EXPORT = ROOT / "training_region_150px_dataset"
sys.path.insert(0, str(THRESHOLDED_EXPORT))
import export_dataset as base  # noqa: E402


RAW_DIR = OUT / "patches_raw"
PREVIEW_DIR = OUT / "patches_preview_marked"
POLYGON_COPY = OUT / "polygon_used.geojson"
ACCEPTED_CSV = OUT / "accepted_contacts.csv"
SKIPPED_CSV = OUT / "skipped_contacts.csv"
MANIFEST = OUT / "dataset_manifest.json"
QUALITY_SUMMARY = OUT / "quality_summary.md"
CONTACT_SHEET = OUT / "preview_contact_sheet.png"

PATCH_SIZE = base.PATCH_SIZE


def csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_contact_sheet(rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = 24
    cell_size = 90
    thumbnail_size = 80
    title_height = 50
    row_count = max(1, math.ceil(len(rows) / columns))
    sheet = Image.new("RGB", (columns * cell_size, title_height + row_count * cell_size), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        f"Costa Head all polygon-inside point previews: {len(rows)} x 150 px; red ring = listing coordinate",
        fill="black",
        font=ImageFont.load_default(),
    )
    for index, row in enumerate(rows, start=1):
        with Image.open(OUT / row["preview_file"]) as image:
            thumbnail = image.convert("RGB").resize((thumbnail_size, thumbnail_size), Image.Resampling.NEAREST)
        column = (index - 1) % columns
        row_number = (index - 1) // columns
        x = column * cell_size + 5
        y = title_height + row_number * cell_size + 5
        sheet.paste(thumbnail, (x, y))
        draw.rectangle((x, y, x + thumbnail_size - 1, y + thumbnail_size - 1), outline=(80, 80, 80), width=1)
        draw.text((x + 2, y + 2), str(index), fill="black", font=ImageFont.load_default())
    sheet.save(CONTACT_SHEET, format="PNG", optimize=True)
    return {
        "path": "preview_contact_sheet.png",
        "columns": columns,
        "cell_size_px": cell_size,
        "thumbnail_size_px": thumbnail_size,
        "source": "patches_preview_marked only",
    }


def validate(
    accepted: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    inside_items: list[dict[str, Any]],
    rings: list[list[list[float]]],
    raster: dict[str, Any],
    source_unchanged: dict[str, bool],
) -> dict[str, Any]:
    if len(accepted) + len(skipped) != len(inside_items):
        raise RuntimeError("accepted + skipped does not equal polygon-inside candidates")
    checked_raw = 0
    checked_preview = 0
    max_distance = 0.0
    candidate_keys = {(item["row"].get("source_file", ""), item["row"].get("source_row", "")) for item in inside_items}
    output_keys = {(row.get("source_file", ""), row.get("source_row", "")) for row in accepted + skipped}
    if candidate_keys != output_keys:
        raise RuntimeError("accepted/skipped rows do not cover exactly the inside candidates")
    for row in accepted:
        e = float(row["e"])
        n = float(row["n"])
        if not base.point_in_polygon_strict(e, n, rings):
            raise RuntimeError(f"accepted coordinate is outside polygon: {row['uid']}")
        x0, y0, x1, y1 = [int(row[f"source_window_{axis}"]) for axis in ("x0", "y0", "x1", "y1")]
        if x0 < 0 or y0 < 0 or x1 >= raster["width"] or y1 >= raster["height"] or (x1 - x0 + 1, y1 - y0 + 1) != (PATCH_SIZE, PATCH_SIZE):
            raise RuntimeError(f"source window out of bounds: {row['uid']}")
        distance = float(row["patch_contact_distance_from_75_75_px"])
        max_distance = max(max_distance, distance)
        if distance > base.MAX_CENTER_DISTANCE_PX + 1e-9:
            raise RuntimeError(f"contact is too far from (75,75): {row['uid']}")
        with Image.open(OUT / row["raw_patch_file"]) as raw:
            if raw.size != (PATCH_SIZE, PATCH_SIZE) or raw.mode != "L":
                raise RuntimeError(f"raw patch is not 150x150 L: {row['raw_patch_file']}")
            checked_raw += 1
        with Image.open(OUT / row["preview_file"]) as preview:
            if preview.size != (PATCH_SIZE, PATCH_SIZE) or preview.mode != "RGB":
                raise RuntimeError(f"preview is not 150x150 RGB: {row['preview_file']}")
            checked_preview += 1
    if not all(source_unchanged.values()):
        raise RuntimeError("one or more source files changed")
    with Image.open(CONTACT_SHEET) as sheet:
        if sheet.mode != "RGB" or sheet.width <= 0 or sheet.height <= 0:
            raise RuntimeError("contact sheet is invalid")
    return {
        "all_polygon_inside_candidates_accounted": True,
        "quality_thresholds_applied": False,
        "accepted_coordinates_strictly_inside_polygon": True,
        "source_windows_within_ecw_bounds": True,
        "raw_patches_150x150_single_channel": True,
        "raw_patches_without_red_marking": True,
        "raw_patch_count_checked": checked_raw,
        "preview_patches_150x150_rgb": True,
        "preview_patch_count_checked": checked_preview,
        "max_center_deviation_px": max_distance,
        "contact_sheet_preview_only": True,
        "all_checked_sources_unchanged": True,
    }


def summary(
    counts: dict[str, Any],
    means: list[float],
    variances: list[float],
    cache_info: dict[str, Any],
) -> str:
    mean_stats = base.metric_summary(means)
    variance_stats = base.metric_summary(variances)
    return "\n".join(
        [
            "# Costa Head 150 px all point candidates",
            "",
            "This is an unfiltered review set. Every decodable point contact strictly inside the supplied polygon is exported. "
            "The official listing remains provenance and is not asserted to be verified ground truth.",
            "",
            "## Counts",
            "",
            f"- Costa point contacts: **{counts['costa_point_records']}**.",
            f"- Strictly inside polygon: **{counts['polygon_inside_point_candidates']}**.",
            f"- Exported raw/preview pairs: **{counts['accepted']}**.",
            f"- Structural/decode skips: **{counts['skipped']}**.",
            f"- Polygon outside point contacts: **{counts['outside_polygon_point_records']}**.",
            "",
            "## Quality metrics recorded but not used as filters",
            "",
            "No grey-mean or grey-variance threshold was applied in this directory. Metrics are included so the user can make the later visual decision.",
            f"- Grey mean: n={mean_stats.get('count', 0)}, min={mean_stats.get('min', 0):.3f}, median={mean_stats.get('median', 0):.3f}, max={mean_stats.get('max', 0):.3f}.",
            f"- Grey variance: n={variance_stats.get('count', 0)}, min={variance_stats.get('min', 0):.3f}, median={variance_stats.get('median', 0):.3f}, max={variance_stats.get('max', 0):.3f}.",
            "",
            "## Reproducibility",
            "",
            "- No random sampling or train/validation/test split was performed.",
            f"- Only bounded 150x150 ECW windows were read; the persistent decoder used at most {base.MAX_WORKERS} serialized worker and a {base.DECODE_CACHE_SIZE}-window cache.",
            f"- Decoder cache statistics: hits={cache_info['hits']}, misses={cache_info['misses']}, maxsize={cache_info['maxsize']}.",
            "- Later splitting should use spatial blocks or survey/campaign groups, not random adjacent contacts.",
            "",
        ]
    )


def main() -> None:
    required = [base.POLYGON_PATH, base.CONTACTS_PATH, base.METADATA_PATH, base.ECW_PATH, base.ERS_PATH, base.DECODER_PATH]
    if any(not path.exists() for path in required):
        raise RuntimeError("one or more required inputs are missing")
    source_before = {str(path): base.snapshot(path) for path in required if path.is_file()}
    polygon_document, rings = base.load_polygon()
    metadata, raster = base.load_raster_metadata()
    contacts = json.loads(base.CONTACTS_PATH.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    base.shutil.copyfile(base.POLYGON_PATH, POLYGON_COPY)

    costa_points = [row for row in contacts if isinstance(row, dict) and row.get("dataset") == "costa" and row.get("geometry") == "point"]
    costa_lines = [row for row in contacts if isinstance(row, dict) and row.get("dataset") == "costa" and row.get("geometry") == "line"]
    uid_counts = Counter(base.clean_text(row.get("uid")) for row in costa_points if base.clean_text(row.get("uid")))
    inside_items: list[dict[str, Any]] = []
    outside_count = 0
    invalid_count = 0
    for contact_list_index, row in enumerate(costa_points, start=1):
        e = base.finite_number(row.get("e"))
        n = base.finite_number(row.get("n"))
        if e is None or n is None:
            invalid_count += 1
            continue
        if not base.point_in_polygon_strict(e, n, rings):
            outside_count += 1
            continue
        inside_items.append({
            "ordinal": len(inside_items) + 1,
            "contact_list_index": contact_list_index,
            "row": row,
            "e": e,
            "n": n,
        })

    used_stems: dict[str, str] = {}
    stem_by_uid: dict[str, str] = {}
    for item in inside_items:
        uid = base.clean_text(item["row"].get("uid"))
        if uid and uid not in stem_by_uid:
            stem_by_uid[uid] = base.safe_stem(uid, used_stems)

    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    means: list[float] = []
    variances: list[float] = []

    for item in inside_items:
        row = item["row"]
        uid = base.clean_text(row.get("uid"))
        if not uid:
            skipped.append(base.skipped_row(item, "incomplete_fields", "uid is missing; stable filename cannot be created"))
            skip_counts["incomplete_fields"] += 1
            continue
        if uid_counts[uid] > 1:
            skipped.append(base.skipped_row(item, "duplicate_uid", f"uid occurs {uid_counts[uid]} times"))
            skip_counts["duplicate_uid"] += 1
            continue
        window = base.window_for(item["e"], item["n"], raster)
        if not base.window_in_bounds(window, raster):
            skipped.append(base.skipped_row(item, "window_out_of_raster_bounds", "150x150 source window is outside ECW bounds", window=window))
            skip_counts["window_out_of_raster_bounds"] += 1
            continue
        try:
            pixels = base.decode_window_pixels(window["x0"], window["y0"], window["x1"], window["y1"])
        except Exception as exc:
            skipped.append(base.skipped_row(item, "ecw_decode_failed", str(exc)[:500], window=window))
            skip_counts["ecw_decode_failed"] += 1
            continue
        image = Image.frombytes("L", (PATCH_SIZE, PATCH_SIZE), pixels)
        stats = ImageStat.Stat(image)
        mean = float(stats.mean[0])
        variance = float(stats.var[0])
        means.append(mean)
        variances.append(variance)
        raw_image, preview_image = base.make_preview(pixels, window)
        filename = f"{stem_by_uid[uid]}.png"
        raw_relative = f"patches_raw/{filename}"
        preview_relative = f"patches_preview_marked/{filename}"
        raw_image.save(OUT / raw_relative, format="PNG", optimize=True)
        preview_image.save(OUT / preview_relative, format="PNG", optimize=True)
        accepted_row = base.accepted_row(item, window, mean, variance, raw_relative, preview_relative)
        accepted_row["quality_filter_status"] = "not_applied"
        accepted.append(accepted_row)

    accepted.sort(key=lambda row: int(row["contact_list_index"]))
    skipped.sort(key=lambda row: int(row["contact_list_index"]))
    accepted_columns = base.ACCEPTED_COLUMNS + ["quality_filter_status"]
    base.write_csv(ACCEPTED_CSV, accepted_columns, accepted)
    base.write_csv(SKIPPED_CSV, base.SKIPPED_COLUMNS, skipped)
    sheet_info = build_contact_sheet(accepted)

    source_after = {path: base.snapshot(Path(path)) for path in source_before}
    source_unchanged = {path: source_before[path] == source_after[path] for path in source_before}
    validation = validate(accepted, skipped, inside_items, rings, raster, source_unchanged)
    cache = base.decode_window_pixels.cache_info()
    cache_info = {"hits": cache.hits, "misses": cache.misses, "maxsize": cache.maxsize, "currsize": cache.currsize}

    counts = {
        "contacts_total": len(contacts),
        "costa_point_records": len(costa_points),
        "costa_line_records": len(costa_lines),
        "polygon_inside_point_candidates": len(inside_items),
        "accepted": len(accepted),
        "skipped": len(skipped),
        "accepted_plus_skipped": len(accepted) + len(skipped),
        "outside_polygon_point_records": outside_count,
        "invalid_coordinate_point_records": invalid_count,
        "skip_reason_counts": dict(skip_counts),
    }
    QUALITY_SUMMARY.write_text(summary(counts, means, variances, cache_info), encoding="utf-8")

    manifest = {
        "manifest_version": "costa-training-region-150px-all-point-candidates-v1",
        "dataset": "costa",
        "purpose": "Unfiltered visual review set; user decides later which contacts to retain.",
        "label_provenance_note": "Official contact-list listings are retained as provenance and are not asserted to be verified ground truth.",
        "inputs": {
            "polygon": {
                "source_path": str(base.POLYGON_PATH),
                "derived_copy": "polygon_used.geojson",
                "crs": "EPSG:32630",
                "sha256": base.sha256_file(base.POLYGON_PATH),
                "geometry": polygon_document["features"][0]["geometry"],
                "copy_byte_identical_to_source": POLYGON_COPY.read_bytes() == base.POLYGON_PATH.read_bytes(),
            },
            "contacts": {"source_path": str(base.CONTACTS_PATH), "sha256": base.sha256_file(base.CONTACTS_PATH), "record_count": len(contacts)},
            "ecw": {"source_path": str(base.ECW_PATH), "size_bytes": source_before[str(base.ECW_PATH)]["size_bytes"], "decoder": str(base.DECODER_PATH), "band": base.DECODER_BAND},
            "ers": {"source_path": str(base.ERS_PATH), "size_bytes": source_before[str(base.ERS_PATH)]["size_bytes"]},
            "metadata": {"source_path": str(base.METADATA_PATH), "raster_id": metadata["datasets"]["costa"]["rasters"][0]["id"]},
        },
        "source_raster": {
            "crs": "EPSG:32630",
            "width_px": raster["width"],
            "height_px": raster["height"],
            "bounds_world_m": {"left": raster["left"], "right": raster["right"], "bottom": raster["bottom"], "top": raster["top"]},
            "native_resolution_m_per_px": base.NATIVE_RESOLUTION_M,
            "decoder_info": raster["decoder_info"],
            "read_policy": "Only bounded 150x150 source windows were requested; the full ECW was never decoded into RAM or converted to TIFF.",
        },
        "polygon": {"crs": "EPSG:32630", "strict_membership": True, "coordinates_world_m": rings},
        "selection": {
            "dataset": "costa",
            "geometry": "point",
            "line_contacts_patch_policy": "excluded; no single centre patch generated",
            "quality_filter_applied": False,
            "quality_filter_decision": "No mean or variance threshold was applied; grey_mean and grey_variance are review metadata only.",
            "quality_metrics_source": "unmarked raw grayscale patch",
        },
        "patch": {
            "width_px": PATCH_SIZE,
            "height_px": PATCH_SIZE,
            "physical_size_m": 15.0,
            "raw_mode": "L",
            "preview_mode": "RGB",
            "source_window_inclusive_pixel_coordinates": True,
            "pixel_mapping": {"x": "(e - raster.left) / resolution_x", "y": "(raster.top - n) / resolution_y", "upper_left": "round(pixel_center) - 75"},
            "patch_contact_reference_px": [75, 75],
            "maximum_allowed_distance_from_reference_px": base.MAX_CENTER_DISTANCE_PX,
            "preview_ring": {"shape": "hollow_circle", "radius_px": 7, "line_width_px": 2, "meaning": "contact-list coordinate only; not a target footprint or size label"},
        },
        "randomness": {"used": False, "seed": None, "ordering": "contacts.json order; output CSVs sorted by contact_list_index"},
        "concurrency_and_cache": {"max_workers": base.MAX_WORKERS, "max_in_flight_tasks": base.MAX_IN_FLIGHT, "decode_cache": cache_info},
        "counts": counts,
        "files": {"raw_patches": "patches_raw/", "marked_previews": "patches_preview_marked/", "accepted_csv": "accepted_contacts.csv", "skipped_csv": "skipped_contacts.csv", "quality_summary": "quality_summary.md", "polygon_copy": "polygon_used.geojson", "preview_contact_sheet": sheet_info},
        "split_policy": {"random_split_performed": False, "future_recommendation": "Use spatial blocks or survey/campaign groups for train/validation/test, not random adjacent contacts."},
        "source_integrity": {"checked_paths": source_unchanged, "all_checked_sources_unchanged": all(source_unchanged.values())},
        "quality_distribution": {"grey_mean": base.metric_summary(means), "grey_variance": base.metric_summary(variances), "thresholds_applied": False},
        "validation": validation,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if base._ECW_WORKER:
        base._ECW_WORKER.close()
    print(json.dumps({"output": str(OUT), "counts": counts, "cache": cache_info, "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
