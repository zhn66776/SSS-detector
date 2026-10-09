#!/usr/bin/env python3
"""Build a low-memory, file://-friendly SSS inspection viewer.

The source TIFFs are decoded one scanline at a time and converted into a
Deep-Zoom-like JPEG pyramid.  The browser then keeps only a bounded cache of
visible 256 px tiles instead of decoding the source GeoTIFFs in full.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import CRS
from tifffile import TiffFile


TILE_SIZE = 256
JPEG_QUALITY = 84


def log(message: str) -> None:
    print(message, flush=True)


def normalize_category(value: str) -> str:
    value = " ".join((value or "").strip().split())
    fixes = {
        "": "Unspecified",
        "Lone anglular contact": "Lone angular contact",
        "Seabed Mound (subcrop/outcrop": "Seabed Mound (subcrop/outcrop)",
    }
    return fixes.get(value, value)


def number(value: str | None) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def read_contacts(contact_dir: Path) -> tuple[dict[str, list[dict]], Counter]:
    by_block: dict[str, list[dict]] = defaultdict(list)
    categories: Counter = Counter()
    for csv_path in sorted(contact_dir.glob("*.csv")):
        match = re.search(r"blk(\d+)", csv_path.name, re.IGNORECASE)
        if not match:
            continue
        block = match.group(1).zfill(2)
        with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                easting = number(row.get("SSS_Easting") or row.get("SSS_Eastin"))
                northing = number(row.get("SSS_Northing") or row.get("SSS_Northi"))
                if easting is None or northing is None:
                    continue
                raw_category = " ".join((row.get("Description") or "").strip().split())
                category = normalize_category(raw_category)
                all_values: list[str] = []
                for value in row.values():
                    if isinstance(value, list):
                        all_values.extend(str(v).strip() for v in value)
                    elif value is not None:
                        all_values.append(str(value).strip())
                contact = {
                    "id": (row.get("SSS_ID") or f"row-{row_number}").strip(),
                    "e": easting,
                    "n": northing,
                    "l": number(row.get("Length_m")),
                    "w": number(row.get("Width_m")),
                    "h": number(row.get("Height_m")),
                    "c": category,
                    "m": "*" in all_values,
                }
                if raw_category and raw_category != category:
                    contact["r"] = raw_category
                by_block[block].append(contact)
                categories[category] += 1
    return dict(by_block), categories


def normalize_mag_risk(value: str) -> str:
    value = " ".join((value or "").strip().split())
    fixes = {"Negligibe": "Negligible", "high": "High", "": "Unspecified"}
    return fixes.get(value, value)


def read_mag_contacts(contact_dir: Path) -> tuple[dict[str, list[dict]], Counter]:
    by_block: dict[str, list[dict]] = defaultdict(list)
    risks: Counter = Counter()

    def pick(row: dict, *keys: str) -> str:
        for key in keys:
            value = row.get(key)
            if value is not None and str(value).strip():
                return " ".join(str(value).strip().split())
        return ""

    for csv_path in sorted(contact_dir.glob("*.csv")):
        match = re.search(r"blk(\d+)", csv_path.name, re.IGNORECASE)
        if not match:
            continue
        block = match.group(1).zfill(2)
        with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                easting = number(pick(row, "Mag_Easting", "Mag_Eastin", "Easting"))
                northing = number(pick(row, "Mag_Northing", "Mag_Northi", "Northing"))
                if easting is None or northing is None:
                    continue
                risk = normalize_mag_risk(pick(row, "UXO_Risk", "UXO Risk", "UXO Risk Level"))
                contact = {
                    "id": pick(row, "Mag_ID") or f"row-{row_number}",
                    "e": easting,
                    "n": northing,
                    "r": risk,
                    "d": number(pick(row, "Burial_Depth_m", "Burial_Dep")),
                    "g": number(pick(row, "Grid_value_nT_per_m", "Grid_Value")),
                    "w": number(pick(row, "Target_weight_kg", "Target_wei")),
                }
                optional = {
                    "sid": pick(row, "SSS_ID"),
                    "c": pick(row, "Classification", "Classifica"),
                    "cf": pick(row, "SSS_Confidence", "Confidence"),
                    "a": pick(row, "Additional Comments", "Additional_Comments"),
                }
                for key, value in optional.items():
                    if value and value != "0":
                        contact[key] = value
                by_block[block].append(contact)
                risks[risk] += 1
    return dict(by_block), risks


def read_world_file(path: Path) -> tuple[float, float, float, float, float, float]:
    values = [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != 6:
        raise ValueError(f"Expected six values in {path}")
    a, d, b, e, c, f = values
    if abs(d) > 1e-12 or abs(b) > 1e-12:
        raise ValueError(f"Rotated world files are not supported: {path}")
    return a, d, b, e, c, f


def validate_crs(source: Path, tif_paths: list[Path]) -> dict:
    extracted_root = source.parent
    metadata_paths = {
        "contact_metadata": source / "contact_outer" / "_metadata" / "medin_metadata.xml",
        "imagery_metadata": source / "imagery_outer" / "_metadata" / "medin_metadata.xml",
        "mag_metadata": extracted_root / "MAG" / "contact_outer" / "_metadata" / "medin_metadata.xml",
        "mbes_metadata": extracted_root / "MBES" / "sample_inner" / "_metadata" / "medin_metadata.xml",
    }
    metadata_epsg: dict[str, int] = {}
    for name, path in metadata_paths.items():
        if not path.is_file():
            raise ValueError(f"CRS metadata not found: {path}")
        match = re.search(r"urn:ogc:def:crs:EPSG::(\d+)", path.read_text(errors="replace"))
        if not match:
            raise ValueError(f"EPSG code not found in {path}")
        metadata_epsg[name] = int(match.group(1))
    prj_epsg: dict[str, int | None] = {}
    for tif_path in tif_paths:
        prj_path = tif_path.with_suffix(".prj")
        if not prj_path.is_file():
            raise ValueError(f"Projection file not found: {prj_path}")
        prj_epsg[tif_path.name] = CRS.from_wkt(prj_path.read_text()).to_epsg()
    observed = set(metadata_epsg.values()) | set(prj_epsg.values())
    if observed != {32631}:
        raise ValueError(f"CRS mismatch: {sorted(str(value) for value in observed)}")
    return {
        "epsg": 32631,
        "name": "WGS 84 / UTM zone 31N",
        "contact_metadata": metadata_epsg["contact_metadata"],
        "imagery_metadata": metadata_epsg["imagery_metadata"],
        "mag_metadata": metadata_epsg["mag_metadata"],
        "mbes_metadata": metadata_epsg["mbes_metadata"],
        "projection_files_checked": len(prj_epsg),
    }


def safe_id(path: Path) -> str:
    block = re.search(r"blk(\d+)", path.name, re.IGNORECASE)
    date = re.search(r"_(20\d{6})_blk", path.name, re.IGNORECASE)
    if not block:
        raise ValueError(f"Cannot determine block from {path.name}")
    suffix = date.group(1) if date else "unknown"
    short = "plain" if "Seabed_and_Sub_Seabed" not in path.name else "mapping"
    return f"blk{block.group(1).zfill(2)}-{suffix}-{short}"


def source_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def save_jpeg(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    image.save(temp, format="JPEG", quality=JPEG_QUALITY, optimize=False, progressive=False)
    os.replace(temp, path)


def build_base_tiles(tif_path: Path, target: Path) -> tuple[int, int, int, str]:
    with TiffFile(tif_path) as tif:
        page = tif.pages[0]
        width = int(page.imagewidth)
        height = int(page.imagelength)
        samples = int(page.samplesperpixel)
        max_level = math.ceil(math.log2(max(width, height)))
        level_dir = target / str(max_level)
        level_dir.mkdir(parents=True, exist_ok=True)

        palette = page.colormap
        if palette is None:
            palette8 = np.vstack([np.arange(256, dtype=np.uint8)] * 3)
        else:
            palette8 = (palette // 257).astype(np.uint8)
        color_mode = "RGB" if not (
            np.array_equal(palette8[0], palette8[1])
            and np.array_equal(palette8[1], palette8[2])
        ) else "L"
        has_alpha = samples > 1

        current_group = -1
        index_rows: np.ndarray | None = None
        alpha_rows: np.ndarray | None = None
        valid_rows = 0

        def flush_group(group: int, rows: int) -> None:
            if group < 0 or index_rows is None or rows <= 0:
                return
            y_tile = group
            for x_tile, x0 in enumerate(range(0, width, TILE_SIZE)):
                x1 = min(width, x0 + TILE_SIZE)
                indices = index_rows[:rows, x0:x1]
                alpha = alpha_rows[:rows, x0:x1] if alpha_rows is not None else None
                if alpha is not None and not np.any(alpha):
                    continue
                if color_mode == "L":
                    pixels = palette8[0][indices]
                    if alpha is not None:
                        a16 = alpha.astype(np.uint16)
                        pixels = ((pixels.astype(np.uint16) * a16 + 255 * (255 - a16) + 127) // 255).astype(np.uint8)
                else:
                    pixels = np.moveaxis(palette8[:, indices], 0, -1)
                    if alpha is not None:
                        a16 = alpha.astype(np.uint16)[..., None]
                        pixels = ((pixels.astype(np.uint16) * a16 + 255 * (255 - a16) + 127) // 255).astype(np.uint8)
                save_jpeg(Image.fromarray(pixels, mode=color_mode), level_dir / f"{x_tile}_{y_tile}.jpg")

        for segment, index, _shape in page.segments(sort=True, maxworkers=1):
            if segment is None:
                continue
            y0 = int(index[2])
            x0 = int(index[3])
            rows = segment[0]
            if rows.ndim == 2:
                rows = rows[..., None]
            group = y0 // TILE_SIZE
            if group != current_group:
                flush_group(current_group, valid_rows)
                current_group = group
                group_height = min(TILE_SIZE, height - group * TILE_SIZE)
                index_rows = np.zeros((group_height, width), dtype=np.uint8)
                alpha_rows = np.zeros((group_height, width), dtype=np.uint8) if has_alpha else None
                valid_rows = group_height
            local_y = y0 - group * TILE_SIZE
            row_count = min(rows.shape[0], valid_rows - local_y)
            col_count = min(rows.shape[1], width - x0)
            index_rows[local_y : local_y + row_count, x0 : x0 + col_count] = rows[:row_count, :col_count, 0]
            if alpha_rows is not None:
                alpha_rows[local_y : local_y + row_count, x0 : x0 + col_count] = rows[:row_count, :col_count, 1]
        flush_group(current_group, valid_rows)
    return width, height, max_level, color_mode


def build_lower_levels(target: Path, width: int, height: int, max_level: int, mode: str) -> None:
    previous_width, previous_height = width, height
    fill = 255 if mode == "L" else (255, 255, 255)
    for level in range(max_level - 1, -1, -1):
        previous_dir = target / str(level + 1)
        level_dir = target / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)
        level_width = math.ceil(previous_width / 2)
        level_height = math.ceil(previous_height / 2)
        x_tiles = math.ceil(level_width / TILE_SIZE)
        y_tiles = math.ceil(level_height / TILE_SIZE)
        for y_tile in range(y_tiles):
            for x_tile in range(x_tiles):
                child_paths = [
                    previous_dir / f"{x_tile * 2 + dx}_{y_tile * 2 + dy}.jpg"
                    for dy in range(2)
                    for dx in range(2)
                ]
                if not any(path.exists() for path in child_paths):
                    continue
                canvas = Image.new(mode, (TILE_SIZE * 2, TILE_SIZE * 2), fill)
                for dy in range(2):
                    for dx in range(2):
                        child = previous_dir / f"{x_tile * 2 + dx}_{y_tile * 2 + dy}.jpg"
                        if child.exists():
                            with Image.open(child) as image:
                                canvas.paste(image.convert(mode), (dx * TILE_SIZE, dy * TILE_SIZE))
                valid_width = min(TILE_SIZE * 2, previous_width - x_tile * TILE_SIZE * 2)
                valid_height = min(TILE_SIZE * 2, previous_height - y_tile * TILE_SIZE * 2)
                if valid_width <= 0 or valid_height <= 0:
                    continue
                parent_width = math.ceil(valid_width / 2)
                parent_height = math.ceil(valid_height / 2)
                parent = canvas.crop((0, 0, valid_width, valid_height)).resize(
                    (parent_width, parent_height), Image.Resampling.LANCZOS
                )
                save_jpeg(parent, level_dir / f"{x_tile}_{y_tile}.jpg")
        previous_width, previous_height = level_width, level_height


def build_tiles(tif_path: Path, tile_root: Path) -> tuple[int, int, int, str]:
    signature = source_signature(tif_path)
    complete = tile_root / "complete.json"
    if complete.exists():
        try:
            state = json.loads(complete.read_text())
            if state.get("source") == signature and state.get("tile_size") == TILE_SIZE:
                log(f"  reuse completed tiles: {tile_root.name}")
                return state["width"], state["height"], state["max_level"], state["mode"]
        except (OSError, ValueError, KeyError):
            pass
    if tile_root.exists():
        shutil.rmtree(tile_root)
    tile_root.mkdir(parents=True, exist_ok=True)
    log("  decode base level (single worker, scanline streaming)")
    width, height, max_level, mode = build_base_tiles(tif_path, tile_root)
    log(f"  build overview levels 0..{max_level - 1}")
    build_lower_levels(tile_root, width, height, max_level, mode)
    state = {
        "source": signature,
        "tile_size": TILE_SIZE,
        "width": width,
        "height": height,
        "max_level": max_level,
        "mode": mode,
    }
    complete.write_text(json.dumps(state, indent=2) + "\n")
    return width, height, max_level, mode


def dataset_label(block: str, location: str, duplicate_rank: int | None) -> str:
    if duplicate_rank is not None:
        return f"Block {block} — {'north' if duplicate_rank == 0 else 'south'} sheet"
    if location == "Export Cable Route":
        return f"Block {block} — export cable route"
    return f"Block {block} — wind farm site"


def write_data_scripts(output: Path, contacts: dict[str, list[dict]]) -> None:
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for block, records in contacts.items():
        payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
        content = f'window.SSS_CONTACTS=window.SSS_CONTACTS||{{}};window.SSS_CONTACTS["{block}"]={payload};\n'
        (data_dir / f"blk{block}.js").write_text(content, encoding="utf-8")


def write_mag_data_scripts(output: Path, contacts: dict[str, list[dict]]) -> None:
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for block, records in contacts.items():
        payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
        content = f'window.MAG_CONTACTS=window.MAG_CONTACTS||{{}};window.MAG_CONTACTS["{block}"]={payload};\n'
        (data_dir / f"mag-blk{block}.js").write_text(content, encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    tif_paths = sorted((source / "imagery").rglob("*.tif"))
    contact_dir = source / "contacts" / "SSS Contact list"
    if not tif_paths:
        raise SystemExit(f"No TIFFs found under {source / 'imagery'}")
    if not contact_dir.is_dir():
        raise SystemExit(f"Contact directory not found: {contact_dir}")
    output.mkdir(parents=True, exist_ok=True)

    crs_evidence = validate_crs(source, tif_paths)

    contacts, category_counts = read_contacts(contact_dir)
    write_data_scripts(output, contacts)
    mag_contact_dir = source.parent / "MAG" / "contacts" / "MAG and SSS Contact list"
    if not mag_contact_dir.is_dir():
        raise SystemExit(f"MAG contact directory not found: {mag_contact_dir}")
    mag_contacts, mag_risk_counts = read_mag_contacts(mag_contact_dir)
    write_mag_data_scripts(output, mag_contacts)
    geophysics_path = output / "geophysics" / "layers.json"
    if not geophysics_path.is_file():
        raise SystemExit(f"Geophysical layers not found: {geophysics_path}")
    geophysics = json.loads(geophysics_path.read_text())
    native_path = output / "native" / "assets.json"
    if not native_path.is_file():
        raise SystemExit(f"Native-resolution assets not found: {native_path}")
    native_assets = json.loads(native_path.read_text())
    lossless_path = output / "lossless" / "manifest.json"
    if not lossless_path.is_file():
        raise SystemExit(f"Lossless SSS tiles not found: {lossless_path}")
    lossless_sss = json.loads(lossless_path.read_text())

    prelim: list[dict] = []
    for tif_path in tif_paths:
        match = re.search(r"blk(\d+)", tif_path.name, re.IGNORECASE)
        block = match.group(1).zfill(2) if match else "??"
        a, _d, _b, e, c, f = read_world_file(tif_path.with_suffix(".tfw"))
        prelim.append({"path": tif_path, "block": block, "a": a, "e": e, "c": c, "f": f})

    duplicates: dict[str, list[dict]] = defaultdict(list)
    for item in prelim:
        duplicates[item["block"]].append(item)
    for values in duplicates.values():
        values.sort(key=lambda item: item["f"], reverse=True)

    datasets: list[dict] = []
    for index, item in enumerate(sorted(prelim, key=lambda row: (int(row["block"]), -row["f"])) , start=1):
        tif_path = item["path"]
        dataset_id = safe_id(tif_path)
        log(f"[{index}/{len(prelim)}] {tif_path.name}")
        width, height, max_level, _mode = build_tiles(tif_path, output / "tiles" / dataset_id)
        a, _d, _b, e, c, f = item["a"], 0.0, 0.0, item["e"], item["c"], item["f"]
        left = c - a / 2
        top = f - e / 2
        right = left + a * width
        bottom = top + e * height
        block = item["block"]
        location = tif_path.parent.name
        duplicate_rank = duplicates[block].index(item) if len(duplicates[block]) > 1 else None
        count_inside = sum(
            min(left, right) <= point["e"] <= max(left, right)
            and min(bottom, top) <= point["n"] <= max(bottom, top)
            for point in contacts.get(block, [])
        )
        datasets.append(
            {
                "id": dataset_id,
                "label": dataset_label(block, location, duplicate_rank),
                "block": block,
                "width": width,
                "height": height,
                "maxLevel": max_level,
                "tileSize": TILE_SIZE,
                "left": round(left, 6),
                "top": round(top, 6),
                "right": round(right, 6),
                "bottom": round(bottom, 6),
                "pixelX": a,
                "pixelY": e,
                "contacts": count_inside,
            }
        )

    colors = [
        "#ff595e", "#ffca3a", "#8ac926", "#ff70a6", "#1982c4", "#6a4c93",
        "#ff924c", "#43aa8b", "#f4d35e", "#00b4d8", "#f72585", "#f1faee",
        "#577590", "#adb5bd", "#b8f2e6", "#c77dff",
    ]
    categories = [
        {"name": name, "count": count, "color": colors[i % len(colors)]}
        for i, (name, count) in enumerate(category_counts.most_common())
    ]
    mag_colors = {"High": "#ff375f", "Moderate": "#ff9f0a", "Low": "#ffd60a", "Negligible": "#64d2ff", "Unspecified": "#adb5bd"}
    mag_categories = [
        {"name": name, "count": count, "color": mag_colors.get(name, "#bf5af2")}
        for name, count in sorted(mag_risk_counts.items(), key=lambda item: (-(item[0] == "High"), -(item[0] == "Moderate"), -(item[0] == "Low"), -(item[0] == "Negligible"), item[0]))
    ]

    def contains(item: dict, point: dict) -> bool:
        return (
            min(item["left"], item["right"]) <= point["e"] <= max(item["left"], item["right"])
            and min(item["bottom"], item["top"]) <= point["n"] <= max(item["bottom"], item["top"])
        )

    same_block_coverage = 0
    fallback_coverage = 0
    uncovered = 0
    for block, records in contacts.items():
        same_block = [item for item in datasets if item["block"] == block]
        for point in records:
            if any(contains(item, point) for item in same_block):
                same_block_coverage += 1
            elif any(contains(item, point) for item in datasets):
                fallback_coverage += 1
            else:
                uncovered += 1
    coverage = {
        "same_block": same_block_coverage,
        "fallback_sheet": fallback_coverage,
        "uncovered": uncovered,
        "total": sum(category_counts.values()),
    }
    if uncovered:
        raise ValueError(f"{uncovered} contacts are not covered by any SSS image")

    mag_same_block = 0
    mag_fallback = 0
    mag_uncovered = 0
    for block, records in mag_contacts.items():
        same_block = [item for item in datasets if item["block"] == block]
        for point in records:
            if any(contains(item, point) for item in same_block):
                mag_same_block += 1
            elif any(contains(item, point) for item in datasets):
                mag_fallback += 1
            else:
                mag_uncovered += 1
    mag_coverage = {
        "same_block": mag_same_block,
        "fallback_sheet": mag_fallback,
        "uncovered": mag_uncovered,
        "total": sum(mag_risk_counts.values()),
    }
    if mag_uncovered:
        raise ValueError(f"{mag_uncovered} MAG targets are not covered by any SSS image")

    template = args.template.read_text(encoding="utf-8")
    html = (
        template.replace("__DATASETS_JSON__", json.dumps(datasets, ensure_ascii=False, separators=(",", ":")))
        .replace("__CATEGORIES_JSON__", json.dumps(categories, ensure_ascii=False, separators=(",", ":")))
        .replace("__TOTAL_CONTACTS__", str(sum(category_counts.values())))
        .replace("__CRS_JSON__", json.dumps(crs_evidence, ensure_ascii=False, separators=(",", ":")))
        .replace("__COVERAGE_JSON__", json.dumps(coverage, ensure_ascii=False, separators=(",", ":")))
        .replace("__GEOPHYSICS_JSON__", json.dumps(geophysics, ensure_ascii=False, separators=(",", ":")))
        .replace("__MAG_CATEGORIES_JSON__", json.dumps(mag_categories, ensure_ascii=False, separators=(",", ":")))
        .replace("__TOTAL_MAG_CONTACTS__", str(sum(mag_risk_counts.values())))
        .replace("__NATIVE_JSON__", json.dumps(native_assets, ensure_ascii=False, separators=(",", ":")))
    )
    (output / "index.html").write_text(html, encoding="utf-8")
    summary = {
        "datasets": len(datasets),
        "contacts": sum(category_counts.values()),
        "categories": dict(category_counts),
        "coordinate_system": crs_evidence["name"],
        "epsg": crs_evidence["epsg"],
        "crs_evidence": crs_evidence,
        "contact_image_coverage": coverage,
        "mag_contacts": sum(mag_risk_counts.values()),
        "mag_risk_categories": dict(mag_risk_counts),
        "mag_target_sss_coverage": mag_coverage,
        "geophysical_layers": geophysics,
        "native_assets": native_assets,
        "lossless_sss": {
            "datasets": len(lossless_sss),
            "tile_count": sum(item["tile_count"] for item in lossless_sss),
            "format": "PNG",
            "native_only": True,
        },
    }
    (output / "build-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"Complete: {output / 'index.html'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
